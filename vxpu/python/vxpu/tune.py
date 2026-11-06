# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The sweep-and-write tool for the GEMV templates' tiles (hero run 16).

Given the shape classes a step launches (triton_ops.gemv_class keys: gemv_rows, dual_gemv_swiglu,
gemv_pe), it races every tile set the class can run on the card (measure.time_job: the kernel
over weights cycled past the L2, captured and replayed), prints a table per shape (the current
tile and its time, the best tile and its time, the floor at the card's bandwidth) and writes the
winners into a kernel table in the form the engine reads (plan.load_table: "blocks" entries per
class, a rows_band where the shapes file gives one), merged over an existing table's entries, so
one rebuild picks them up for those shapes. The candidates are the Go planner's (pkg/kernelplan
Candidates: BN 16/32/64, BK 64 to 512, the K split where the caller takes partial sums, warps 4
and 8, 2 as well under 16 MB; the engine's own blocks first), with the pipeline depth (stages 2
and 4) tried on each class's winner.

    python -m vxpu.tune SHAPES [--table T.json] [--out T.json] [--dry-run] [--device cuda:0]
    VXPU_TUNE_SWEEP=SHAPES python -m vxpu.tune ...     the shapes file from the environment

SHAPES is a kernel record (its stats.gemv_classes: the classes the step ran, with the blocks it
ran them under), or JSON {"classes": [class, ...], "bands": {class: [lo, hi]}, "ran": {class:
blocks}}, or a plain list of classes. --out defaults to --table (the merged table written in
place); --dry-run prints the report and the entries and writes nothing.
"""

import argparse
import json
import os
import sys
import time

from . import plan, triton_ops

STAGES_PASS = (2, 4)
BLOCK_ORDER = ("BK", "BN", "KSPLIT", "stages", "warps")


def block_key(blocks):
    return " ".join(f"{k}={int(blocks[k])}" for k in BLOCK_ORDER)


def _field(c, name, default=0):
    v = c.get(name, default)
    return int(v) if isinstance(v, int) else default


def candidates(c, ran=None, sms=0):
    """The block sets a class (measure.parse_class) can run, the engine's own (ran) first: the
    Go planner's rules (pkg/kernelplan/measure.go Candidates)."""
    template = c["template"]
    if template == "dual_gemv_swiglu":
        bns = [16, 32, 64]
    elif template == "gemv_pe":
        bns = [16, 32]
    else:
        bns = [32, 64] if c.get("epi") == "quant" else [16, 32, 64]
    bks = [64, 128, 256, 512]
    splits = [1, 2, 4, 8, 16] if template == "dual_gemv_swiglu" or _field(c, "split") == 1 else [1]
    warps = [2, 4, 8] if c["n"] * c["k"] < 16 << 20 else [4, 8]
    rows_per_group = c["n"] // _field(c, "groups") if _field(c, "groups") > 1 else c["n"]
    hd, rd = _field(c, "hd"), _field(c, "rd")

    def fits(bn, bk, ks):
        if c["n"] % bn or rows_per_group % bn or c["k"] % ks or (c["k"] // ks) % bk or bk > c["k"]:
            return False
        if template == "gemv_pe" and (bn % 2 or (hd - rd) % bn):
            return False
        if template == "dual_gemv_swiglu" and bn % 32 and ks == 1:
            return False
        if ks > 1 and sms > 0 and (c["n"] // bn) * (ks // 2) >= 2 * sms:
            return False
        return True
    out, seen = [], set()

    def add(b):
        k = block_key(b)
        if k not in seen:
            seen.add(k)
            out.append(b)
    if ran and fits(int(ran["BN"]), int(ran["BK"]), max(1, int(ran.get("KSPLIT", 1)))):
        add({"BN": int(ran["BN"]), "BK": int(ran["BK"]), "KSPLIT": max(1, int(ran.get("KSPLIT", 1))),
             "warps": int(ran.get("warps") or 4), "stages": int(ran.get("stages") or 3)})
    for bn in bns:
        for bk in bks:
            for ks in splits:
                if fits(bn, bk, ks):
                    for w in warps:
                        add({"BN": bn, "BK": bk, "KSPLIT": ks, "warps": w, "stages": 3})
    return out


def default_blocks(c):
    """The dispatcher's own tiles for a class the record did not name (gemv_rows' rules, without the
    card: native 16 weight rows a program, the quantizing epilogue at least 32)."""
    bm, bn, bk = triton_ops._rows_blocks(c["rows"], c["k"])
    if _field(c, "native") == 1:
        bn = 16
    if c.get("epi") == "quant":
        bn = max(bn, 32)
    return {"BN": bn, "BK": bk, "KSPLIT": 1, "warps": 4, "stages": 3}


def race(classes, timer, ran=None, sms=0, floors=None, log=None, stages_pass=STAGES_PASS):
    """Time every candidate of every class with timer(class_key, blocks) -> microseconds (raising on
    a set the kernel cannot run), then the pipeline depths on each winner. Returns {class: {"best":
    blocks, "us": float, "ran": blocks or None, "ran_us": float or None, "results": [(blocks, us or
    None, error or None)], "failed": int}}."""
    ran = ran or {}
    out = {}
    for key in classes:
        from .measure import parse_class
        c = parse_class(key)
        own = ran.get(key) or default_blocks(c)
        results = []
        for b in candidates(c, own, sms):
            results.append(_time(timer, key, b, log))
        best = _fastest(results)
        if best is not None and stages_pass:
            for st in stages_pass:
                b = dict(best[0], stages=st)
                if block_key(b) not in {block_key(r[0]) for r in results}:
                    results.append(_time(timer, key, b, log))
            best = _fastest(results)
        own_key = block_key(own)
        ran_us = next((us for b, us, _ in results if us is not None and block_key(b) == own_key), None)
        failed = sum(1 for _, us, _ in results if us is None)
        out[key] = {"best": best[0] if best else None, "us": best[1] if best else None, "ran": own, "ran_us": ran_us,
                    "results": results, "failed": failed, "floor_us": (floors or {}).get(key)}
    return out


def _time(timer, key, blocks, log):
    try:
        us = float(timer(key, blocks))
        err = None
    except Exception as e:  # a block set the kernel cannot compile or run is a result too
        us, err = None, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    if log is not None:
        log(f"[tune] {key} {block_key(blocks)}: {f'{us:.2f} us' if us is not None else err}")
    return blocks, us, err


def _fastest(results):
    timed = [(b, us) for b, us, _ in results if us is not None]
    return min(timed, key=lambda r: r[1]) if timed else None


def entries(raced, bytes_of, bands=None):
    """The table's "blocks" entries from a race: the winner per class with its time, bytes, floor
    and efficiency, the rows_band the shapes file gave, and why it was chosen (the Go form)."""
    out = []
    for key, r in raced.items():
        if r["best"] is None:
            continue
        nbytes = bytes_of(key)
        floor = r.get("floor_us")
        why = f"{len(r['results'])} candidates, {r['failed']} failed"
        if r["ran_us"] is not None:
            why += f"; the engine's own blocks {r['ran_us']:.1f} us"
        e = {"blocks": {k: int(r["best"][k]) for k in ("BK", "BN", "KSPLIT", "stages", "warps")}, "bytes": int(nbytes),
             "shapes": key, "us": round(r["us"], 2), "why": why}
        if floor is not None:
            e["floor_us"] = round(floor, 2)
            e["efficiency"] = round(floor / r["us"], 3)
        if bands and key in bands:
            e["rows_band"] = [int(bands[key][0]), int(bands[key][1])]
        out.append(e)
    return out


def report(raced, a_step=None):
    """The table: shape, the current tile and its time, the best tile and its time, the floor."""
    lines = []
    for key, r in raced.items():
        n = f" ({a_step[key]} a step)" if a_step and key in a_step else ""
        floor = r.get("floor_us")
        if r["best"] is None:
            lines.append(f"{key}{n}: no candidate ran ({r['failed']} failed); the engine's blocks stay")
            continue
        cur = (f"{block_key(r['ran'])} {r['ran_us']:.1f} us" if r["ran_us"] is not None
               else f"{block_key(r['ran'])} (not timed: it does not fit the class)")
        best = f"{block_key(r['best'])} {r['us']:.1f} us"
        gain = f", {100 * r['us'] / r['ran_us']:.0f}% of the current" if r["ran_us"] else ""
        at = f", floor {floor:.1f} us ({100 * floor / r['us']:.0f}% of bandwidth)" if floor is not None else ""
        lines.append(f"{key}{n}: current {cur}; best {best}{gain}{at}; {len(r['results']) - r['failed']} candidates ran")
    return lines


def merge_table(table, new_entries, gpu=None, explain=()):
    """An existing table (a dict as the file holds it, or None) with new_entries replacing its
    entries of the same classes, in the form plan.load_table reads and the Go planner writes:
    blocks sorted by class, format set, the other sections kept."""
    out = dict(table or {})
    out["format"] = plan.TABLE_FORMAT
    if gpu:
        out["gpu"] = gpu
    replaced = {e["shapes"] for e in new_entries}
    kept = [b for b in (out.get("blocks") or []) if b.get("shapes") not in replaced]
    out["blocks"] = sorted(kept + list(new_entries), key=lambda b: b["shapes"])
    if explain:
        out["explain"] = list(explain)
    return out


def write_table(table, path):
    text = json.dumps(table, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    plan.load_table(text)  # what we wrote is what the engine reads
    with open(path, "w") as f:
        f.write(text)


def read_shapes(path):
    """(classes, ran {class: blocks}, bands {class: [lo, hi]}, a_step {class: launches a step}) from a
    kernel record, a {"classes", "ran", "bands"} object, or a list of classes."""
    with open(os.path.expanduser(path)) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(k) for k in data], {}, {}, {}
    if data.get("format") == plan.RECORD_FORMAT:
        classes = (data.get("stats") or {}).get("gemv_classes") or {}
        if not classes:
            raise ValueError(f"{path}: a kernel record with no stats.gemv_classes (the GEMV classes the step launched)")
        return (list(classes), {k: v["blocks"] for k, v in classes.items() if v.get("blocks")}, {},
                {k: (v.get("sites") or v.get("count") or 0) for k, v in classes.items()})
    classes = [str(k) for k in data.get("classes") or []]
    if not classes:
        raise ValueError(f"{path}: no classes (a list, {{\"classes\": [...]}}, or a kernel record)")
    return classes, dict(data.get("ran") or {}), dict(data.get("bands") or {}), {}


def cuda_timer(device):
    """measure.time_job on the card, with the L2 size from the device."""
    import torch
    from . import measure
    props = torch.cuda.get_device_properties(torch.device(device))
    l2 = getattr(props, "L2_cache_size", measure.L2_BYTES_DEFAULT) or measure.L2_BYTES_DEFAULT
    return lambda key, blocks: measure.time_job(key, blocks, device, l2)[0], props


def main(argv=None, timer=None, gpu=None, sms=0, gbps=None):
    ap = argparse.ArgumentParser(prog="python -m vxpu.tune", description=__doc__.split("\n\n")[0])
    ap.add_argument("shapes", nargs="?", default=os.environ.get("VXPU_TUNE_SWEEP"),
                    help="the shapes file (default: VXPU_TUNE_SWEEP)")
    ap.add_argument("--table", default=None, help="the kernel table to merge into (default: a new table)")
    ap.add_argument("--out", default=None, help="where to write the merged table (default: --table)")
    ap.add_argument("--dry-run", action="store_true", help="print the report and the entries, write nothing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-stages", action="store_true", help="skip the pipeline-depth pass on the winners")
    ap.add_argument("--gbps", type=float, default=gbps, help="the floors' bandwidth (default: measured now)")
    a = ap.parse_args(argv)
    if not a.shapes:
        ap.error("a shapes file: the argument or VXPU_TUNE_SWEEP")
    out_path = a.out or a.table
    if not a.dry_run and not out_path:
        ap.error("--out (or --table) names where the winners go; --dry-run prints without writing")
    from . import measure
    classes, ran, bands, a_step = read_shapes(a.shapes)
    table = plan.load_table(a.table) if a.table else None
    if table is not None:  # the table's own entry is the current tile where the record did not say what ran
        for key, blocks in (table.get("blocks_index") or {}).items():
            ran.setdefault(key, blocks)
        table = {k: v for k, v in table.items() if k not in ("index", "blocks_index", "blocks_bands")}
    if timer is None:
        timer, props = cuda_timer(a.device)
        gpu, sms = props.name, props.multi_processor_count
        bw = a.gbps or measure.bandwidth_gbps(a.device)
    else:
        bw = a.gbps or 0.0
    floors = {k: measure.class_bytes(k) / bw / 1e3 for k in classes} if bw else {}
    t0 = time.perf_counter()
    raced = race(classes, timer, ran=ran, sms=sms, floors=floors, log=lambda s: print(s, file=sys.stderr, flush=True),
                 stages_pass=() if a.no_stages else STAGES_PASS)
    lines = report(raced, a_step)
    print(f"[tune] {len(classes)} classes on {gpu or a.device}, {sum(len(r['results']) for r in raced.values())} "
          f"candidates timed in {time.perf_counter() - t0:.0f} s" + (f", bandwidth {bw:.0f} GB/s" if bw else ""))
    for line in lines:
        print(line)
    new = entries(raced, measure.class_bytes, bands)
    merged = merge_table(table, new, gpu=gpu, explain=lines)
    if a.dry_run:
        print(f"[tune] dry run: {len(new)} entries not written" + (f" to {out_path}" if out_path else ""))
        print(json.dumps(new, indent=1, sort_keys=True))
        return 0
    write_table(merged, out_path)
    print(f"[tune] {len(new)} entries written to {out_path} ({len(merged['blocks'])} blocks entries in all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
