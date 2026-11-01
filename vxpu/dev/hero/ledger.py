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

"""The step's own kernel times joined to the kernel record's sites, and the
kernel families against their floors (hero run 11, the roofline ledger).

    python ledger.py RECORD.json [--kprof KPROF.json] [--bandwidth 1460] [--step-times OUT.json]
                     [--expert-mb 18.8 --moe-layers 40 --topk 6 --worst 15]

The record (VXPU_PLAN_RECORD) carries per site the bytes it must move and
the lowering's microbenchmark of the chosen body; `vxpu kernels check`
prints that ledger. This script adds what the record cannot know on its
own: the kernel profile of one decode step (hero-kprof.py's
kprof-last-step.json: every kernel of the step in launch order with its
microseconds), so each kernel family's time stands against the bytes of
the sites that launch it, and each site gets the step's time rather than
the microbenchmark's. The join is by launch order: the record lists the
sites in graph order and the step launches them in that order, so the
profile's kernels are handed out to the sites in sequence by the family
each site's body launches (FAMILIES below; from triton_ops.py), and the
kernels no site claims (the experts, which are host nodes outside the
record; the activation recode, which belongs to the GEMV after it; the
compiler's small kernels) are reported on their own lines. The experts'
floor is top-k times the layers times the bytes of one expert, an
estimate the caller sets (the record's distinct_experts statistic when
it has one). --step-times writes {node: us} for `vxpu kernels check
--step-times`.
"""

import argparse
import collections
import json
import re
import sys

# the Triton kernels each lowered body launches (triton_ops.py), by the function names the record's
# "kernels" lists carry, with how many launches of each one call makes at one row (the attention's
# q and out projections are two GEMVs each: the low-rank pair), and the share of the body's bytes each
# family reads (the shared expert's w1 and w3 go to the dual GEMV, w2 to gemv_pe)
FAMILIES = {
    # read off one layer of the one-row step's launch order (kprof) against the record's kernel lists, and
    # checked against the step's totals: 211 gemv_rows = 40 x (attn_q 1 + window_kv 1 + attn_out 2 + shared w2 1)
    # + 10 indexer linears + the head. attn_q is wq_a (5120x1280, gemv_rows) then wq_b with the rotary epilogue
    # (1280x32768, gemv_pe: 87% of its bytes); the shared expert is the dual GEMV (w1, w3) and w2 as gemv_rows.
    "vxpu_triton.attn_q": {"_gemv_rows_kernel": (1, 0.13), "_row_post_kernel": (1, 0.0), "_gemv_pe_kernel": (1, 0.87)},
    "vxpu_triton.window_kv": {"_gemv_rows_kernel": (1, 1.0), "_row_post_kernel": (1, 0.0)},
    "vxpu_triton.attn_select": {"_sparse_attn_kernel": (1, 1.0), "_sparse_attn_combine_kernel": (1, 0.0)},
    "vxpu_triton.attn_out": {"_gemv_rows_kernel": (2, 1.0)},
    "vxpu_triton.fp8_gemv": {"_gemv_rows_kernel": (1, 1.0)},
    "vxpu_triton.act_quant_dequant": {"_act_quant_dequant_kernel": (1, 0.0)},
    "vxpu_triton.linear_f32": {"_gemv_rows_kernel": (1, 1.0)},
    "vxpu_triton.shared_ffn": {"_dual_gemv_swiglu_kernel": (1, 2 / 3), "_swiglu_rows_kernel": (1, 0.0),
                               "_gemv_rows_kernel": (1, 1 / 3)},
    "vxpu_triton.ffn_front_split": {"_front_stats_kernel": (1, 0.0), "_front_dots_kernel": (1, 0.5),
                                    "_ffn_front_b_kernel": (1, 0.5)},
    "vxpu_triton.hc_front_split": {"_front_stats_kernel": (1, 0.0), "_front_dots_kernel": (1, 0.5),
                                   "_ffn_front_b_kernel": (1, 0.5)},
    "vxpu_triton.ffn_front": {"_ffn_front_kernel": (1, 1.0)},
    "vxpu_triton.rotary": {"_gemv_rows_kernel": (1, 1.0)},
    "vxpu_triton.swiglu_rows": {"_swiglu_rows_kernel": (1, 1.0)},
}
# kernels that belong to the body after them (the activation recode a GEMV consumes)
PROLOGUE = {"_x_mxfp8_kernel"}
# host nodes outside the record: the experts and the crossings
HOST = {"_fp4_direct_gate_up_kernel": "experts", "_fp4_direct_down_kernel": "experts",
        "_fp4_grouped_gate_up_kernel": "experts", "_fp4_grouped_down_kernel": "experts",
        "_swiglu_quant_kernel": "experts", "_grouped_schedule_kernel": "experts", "_combine_kernel": "experts",
        "_ring_write_kernel": "crossings", "Memcpy PtoP": "crossings", "Memcpy DtoD": "crossings"}


def family(name):
    """A kernel's family: the Triton kernel name, or the compiler's kernel kind."""
    name = name.split("(")[0].strip()
    if name.startswith("triton_"):
        return "compiler:" + name.split("_")[1]  # poi, red, per
    if name.startswith("void ") or "::" in name:
        return "aten:" + re.sub(r"<.*", "", name.split("::")[-1])[:40]
    return name


def expected(site):
    """The launches a site's body makes, by family."""
    out = collections.Counter()
    for fn in site.get("kernels") or []:
        for fam, (count, _share) in FAMILIES.get(fn, {}).items():
            out[fam] += count
    return out


TAIL_OPS = ("vxpu::shared_ffn_front", "vxpu::shared_ffn", "vxpu::shared_ffn_combine")  # after a layer's experts


def join(record, kprof):
    """Hand the step's kernels to the record's sites, one layer at a time.
    The experts' first kernel marks each MoE layer (one launch a layer), so
    the profile splits into segments: after layer L's experts come L's tail
    (the shared expert and its combine) and then layer L+1's head (the
    fronts, attention, the router). Within a segment a kernel goes to the
    tail's sites while they still declare launches of its family, else to
    the head's sites in record order; nothing a mistake in one layer can
    do reaches the next. Returns ({node: us}, {family: [count, us]} for
    the unclaimed kernels, {family: [count, us]} for every kernel, and the
    (op, family) pairs whose joined launches differ from the declared)."""
    sites = [s for s in record["sites"] if s.get("kernels")]
    by_layer = collections.defaultdict(list)
    for s in sites:
        by_layer[s.get("layer")].append(s)
    layers = sorted(l for l in by_layer if l is not None)
    marker = "_fp4_direct_gate_up_kernel"
    names = [family(e["name"]) for e in kprof]
    marks = [i for i, n in enumerate(names) if n == marker] or [i for i, n in enumerate(names) if n == "_fp4_grouped_gate_up_kernel"]
    bounds = [0] + marks + [len(kprof)]
    by_family = collections.defaultdict(lambda: [0, 0.0])
    unclaimed = collections.defaultdict(lambda: [0, 0.0])
    step_times = collections.defaultdict(float)
    joined = collections.Counter()

    def pools(seg):
        # segment 0: layer 0's head; segment k>0: layer k-1's tail then layer k's head; the last: the head of the model
        tail = [s for s in by_layer.get(layers[seg - 1], [])] if 0 < seg <= len(layers) else []
        tail = [s for s in tail if s["op"] in TAIL_OPS]
        head = [s for s in by_layer.get(layers[seg], [])] if seg < len(layers) else by_layer.get(None, [])
        head = [s for s in head if s["op"] not in TAIL_OPS]
        return [(s, expected(s)) for s in tail] + [(s, expected(s)) for s in head]

    for seg in range(len(bounds) - 1):
        pool = pools(seg)
        prologue = 0.0
        for event in kprof[bounds[seg]:bounds[seg + 1]]:
            fam = family(event["name"])
            us = float(event["us"])
            by_family[fam][0] += 1
            by_family[fam][1] += us
            if fam in PROLOGUE:
                prologue += us
                continue
            if fam in HOST:
                unclaimed[HOST[fam]][0] += 1
                unclaimed[HOST[fam]][1] += us
                continue
            for site, left in pool:
                if left.get(fam, 0) > 0:
                    left[fam] -= 1
                    joined[(site["op"], fam)] += 1
                    step_times[site["node"]] += us + prologue
                    prologue = 0.0
                    break
            else:
                unclaimed[fam][0] += 1
                unclaimed[fam][1] += us
    declared = collections.Counter()
    for s in sites:
        for fam, n in expected(s).items():
            declared[(s["op"], fam)] += n
    mismatches = {k: (declared[k], joined.get(k, 0)) for k in declared if declared[k] != joined.get(k, 0)}
    return dict(step_times), dict(unclaimed), dict(by_family), mismatches


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("record")
    ap.add_argument("--kprof", help="hero-kprof.py's kprof-last-step.json")
    ap.add_argument("--bandwidth", type=float, default=0, help="GB/s (default: the record's measurement, else 1460)")
    ap.add_argument("--step-times", help="write {node: us} here for vxpu kernels check --step-times")
    ap.add_argument("--expert-mb", type=float, default=18.8, help="bytes of one expert's three projections, MB")
    ap.add_argument("--moe-layers", type=int, default=40)
    ap.add_argument("--topk", type=float, default=6, help="distinct experts a layer (the record's statistic when present)")
    ap.add_argument("--worst", type=int, default=15)
    args = ap.parse_args(argv)
    record = json.load(open(args.record))
    bw = args.bandwidth or record.get("hardware", {}).get("gpu_bandwidth_gb_s") or 1460.0
    sites = record["sites"]
    with_bytes = [s for s in sites if s.get("bytes")]
    print(f"record: {len(sites)} sites, {len(with_bytes)} with bytes; bandwidth {bw:.0f} GB/s")

    def floor_us(nbytes):
        return nbytes / (bw * 1e3)

    # the lowering's microbenchmark against the floor, per site and per op
    per_op = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])  # sites, measured, floor us, measured us
    lines = []
    for s in with_bytes:
        b = s["bytes"]["weights"] + s["bytes"]["activations"]
        us = (s.get("measured_us") or {}).get(s.get("chosen")) or 0.0
        per_op[s["op"]][0] += 1
        per_op[s["op"]][2] += floor_us(b)
        if us:
            per_op[s["op"]][1] += 1
            per_op[s["op"]][3] += us
            lines.append((us - floor_us(b), s, b, us))
    print("\nthe lowering's microbenchmarks: per op (sites, measured, floor us, measured us, efficiency)")
    for op, (n, m, f, u) in sorted(per_op.items(), key=lambda kv: -(kv[1][3] - kv[1][2])):
        eff = f"{100 * f / u:.0f}%" if u else "-"
        print(f"  {op:22s} {n:4d} {m:4d} {f:9.1f} {u:9.1f} {eff:>5s}")
    print(f"\nworst {args.worst} sites by lost us (microbenchmark): op, site, shapes, chosen, MB, floor us, us, efficiency")
    for lost, s, b, us in sorted(lines, key=lambda x: -x[0])[: args.worst]:
        print(f"  {s['op']:18s} {s['site']:30s} {s.get('shapes', ''):32s} {s['chosen']:16s} {b / 1e6:7.1f} "
              f"{floor_us(b):7.1f} {us:7.1f} {100 * floor_us(b) / us:4.0f}%")

    if not args.kprof:
        return 0
    kprof = json.load(open(args.kprof))
    step_times, unclaimed, by_family, mismatches = join(record, kprof)
    if mismatches:
        print("JOIN MISMATCH: launches handed out differ from what the sites declare (op, family): declared, joined")
        for k, (d, j) in sorted(mismatches.items()):
            print(f"  {k[0]:22s} {k[1]:30s} {d:5d} {j:5d}")
    else:
        print("join: every (op, family) got exactly the launches the sites declare")
    total_us = sum(v[1] for v in by_family.values())
    print(f"\nthe step's profile: {len(kprof)} kernels, {total_us / 1000:.2f} ms GPU time")
    # per family: time from the profile, bytes from the sites that claimed kernels of it (shared bodies split evenly)
    fam_bytes = collections.defaultdict(float)
    node_bytes = {s["node"]: s["bytes"]["weights"] + s["bytes"]["activations"] for s in with_bytes}
    for s in with_bytes:
        shares = collections.Counter()
        for fn in s.get("kernels") or []:
            for fam, (_count, share) in FAMILIES.get(fn, {}).items():
                shares[fam] += share
        scale = 1.0 / max(1.0, sum(shares.values()))  # a body listed twice (two GEMVs) reads its bytes once
        for fam, share in shares.items():
            fam_bytes[fam] += node_bytes[s["node"]] * share * scale
    experts_bytes = args.topk * args.moe_layers * args.expert_mb * 1e6
    stat = record.get("stats", {}).get("distinct_experts")
    if isinstance(stat, (int, float)):
        experts_bytes = stat * args.moe_layers * args.expert_mb * 1e6
    print("per kernel family (launches, us, mean us, bytes attributed, floor us, efficiency); experts' bytes "
          f"{'from the record' if isinstance(stat, (int, float)) else 'estimated at top-k'}")
    rows = []
    for fam, (n, us) in by_family.items():
        b = fam_bytes.get(fam, 0.0)
        if HOST.get(fam) == "experts":
            b = None  # reported as one line below
        rows.append((us, fam, n, b))
    experts_us = sum(us for us, fam, n, b in rows if HOST.get(fam) == "experts")
    experts_n = sum(n for us, fam, n, b in rows if HOST.get(fam) == "experts")
    for us, fam, n, b in sorted(rows, key=lambda r: -r[0]):
        if HOST.get(fam) == "experts":
            continue
        if b:
            print(f"  {fam:34s} {n:5d} {us:8.1f} {us / n:6.1f} {b / 1e6:9.1f} MB {floor_us(b):8.1f} {100 * floor_us(b) / us:4.0f}%")
        else:
            print(f"  {fam:34s} {n:5d} {us:8.1f} {us / n:6.1f}        -           -    -")
    if experts_n:
        print(f"  {'experts (host nodes)':34s} {experts_n:5d} {experts_us:8.1f} {experts_us / experts_n:6.1f} "
              f"{experts_bytes / 1e6:9.1f} MB {floor_us(experts_bytes):8.1f} {100 * floor_us(experts_bytes) / experts_us:4.0f}%")
    claimed_us = sum(step_times.values())
    print(f"\njoined to sites: {len(step_times)} sites claimed {claimed_us / 1000:.2f} ms; unclaimed: "
          + ", ".join(f"{k} {v[0]}x {v[1] / 1000:.2f} ms" for k, v in sorted(unclaimed.items(), key=lambda kv: -kv[1][1])))
    per_op_step = collections.defaultdict(lambda: [0, 0.0, 0.0])  # sites claimed, step us, floor us
    for s in with_bytes:
        us = step_times.get(s["node"])
        if us:
            per_op_step[s["op"]][0] += 1
            per_op_step[s["op"]][1] += us
            per_op_step[s["op"]][2] += floor_us(node_bytes[s["node"]])
    print("\nper op from the step's own time (sites claimed, step us, floor us, efficiency, lost us)")
    for op, (n, us, f) in sorted(per_op_step.items(), key=lambda kv: -(kv[1][1] - kv[1][2])):
        print(f"  {op:22s} {n:4d} {us:9.1f} {f:9.1f} {100 * f / us:4.0f}% {us - f:8.1f}")
    print(f"\nworst {args.worst} sites by lost us (the step's own time): op, site, shapes, MB, floor us, step us, efficiency")
    step_lines = []
    for s in with_bytes:
        us = step_times.get(s["node"])
        if not us:
            continue
        b = node_bytes[s["node"]]
        step_lines.append((us - floor_us(b), s, b, us))
    for lost, s, b, us in sorted(step_lines, key=lambda x: -x[0])[: args.worst]:
        print(f"  {s['op']:18s} {s['site']:30s} {s.get('shapes', ''):32s} {b / 1e6:7.1f} {floor_us(b):7.1f} {us:7.1f} "
              f"{100 * floor_us(b) / us:4.0f}%")
    step_floor = sum(floor_us(node_bytes[n]) for n in step_times) + floor_us(experts_bytes)
    print(f"\nstep floor (sites joined + experts): {step_floor / 1000:.2f} ms against {total_us / 1000:.2f} ms of GPU time "
          f"({100 * step_floor / total_us:.0f}%)")
    if args.step_times:
        json.dump(step_times, open(args.step_times, "w"), indent=1, sort_keys=True)
        print(f"wrote {args.step_times}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
