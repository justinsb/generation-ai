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

"""Hero 15, the expert GEMM form's fusions (FP4_FUSE_EPI): one layer's routed-expert node at the
128-row verify shape, timed per kernel and as a whole, with and without each fusion, and every
arm's output checked bit for bit against the unfused GEMM form's and the direct kernels'.

The shape is the Nov 6 measurement's: 128 rows, top-6 (768 pairs) over the slots of one card,
each slot a tensor-parallel slice of an expert (K 5120, I 576 of 2304: 4.7 MB a slice, about
120 distinct a call), every pair held (the TP4 placement). --slice 2304 is the whole expert,
--ep 4 the expert-parallel map (one expert in four held here).

  python bench_fuse.py [--rows 128] [--arms 0,1,3,7,15] [--iters 12] [--graph] [--no-direct]

Per arm (a value of FP4_FUSE_EPI, or "fold" for FP4_GEMM_FOLD=1), one line: the kernels' time by
family under the profiler (gate_up, down, swiglu, schedule, aten: the fills, casts, gathers and
reduces around the launches, other), launches a call, the call's wall time by events (and the
replay of a captured graph of it with --graph, the step's form), and the identity of its output
with arm 0's and with the direct kernels'. RESULT lines carry the same as JSON.
"""
import argparse
import json
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from vxpu import triton_ops as T

K, I_FULL, TOPK = 5120, 2304, 6
READ_PEAK = 1535.0  # GB/s, the card's single-tensor read (hero run 11)


def banks(dev, slots, I, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)

    def bank(rows, inner):
        wq = torch.randint(0, 256, (slots, rows, inner // 2), device=dev, dtype=torch.uint8, generator=g)
        sc = torch.randint(120, 128, (slots, rows, inner // 32), device=dev, dtype=torch.uint8, generator=g)
        return wq, sc

    w1, s1 = bank(I, K)
    w3, s3 = bank(I, K)
    w2, s2 = bank(K, I)
    return w1, s1, w3, s3, w2, s2


def slot_map(dev, slots, ep):
    """The pool's map over the layer's experts: every expert held (ep 1, the tensor-parallel slices) or one in ep."""
    m = torch.full((slots * ep,), -1, dtype=torch.int64)
    m[::ep] = torch.arange(slots)
    return m.to(dev)


def draw(rows, experts, g):
    """Expert ids [rows, TOPK]: each row's top-6, distinct, uniform over the layer's experts."""
    return torch.stack([torch.randperm(experts, generator=g)[:TOPK] for _ in range(rows)])


def set_arm(arm):
    """The GEMM form forced, the fusions per the arm; "fold" is the Nov 5 fold, "direct" the direct kernels."""
    T.FP4_DIRECT, T.FP4_GEMM_MIN_PAIRS, T.FP4_GEMM_FOLD, T.FP4_FUSE_EPI, T.FP4_PDL = "0", 1, False, 0, None
    if arm == "direct":
        T.FP4_DIRECT = "1"
    elif arm == "fold":
        T.FP4_GEMM_FOLD = True
    else:
        T.FP4_FUSE_EPI = int(arm)


def call(x, wt, smap, idx, b):
    return T.fp4_experts_decode(x, wt, smap, *b, 7.0, per_pair=False, indices=idx, start=0)


def family(name):
    for key in ("gate_up", "down", "swiglu", "schedule"):
        if key in name:
            return key
    if any(k in name for k in ("elementwise", "reduce", "fill", "index", "Memset", "Memcpy", "copy")):
        return "aten"
    return "other"


def timed(x, wt, smap, idxs, b, iters, graph):
    for i in range(2):
        call(x, wt, smap, idxs[i], b)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        call(x, wt, smap, idxs[2 + i], b)
        ends[i].record()
    torch.cuda.synchronize()
    wall = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))[iters // 2]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            call(x, wt, smap, idxs[2 + i], b)
        torch.cuda.synchronize()
    per, launches = {}, 0
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        launches += 1
        per.setdefault(family(e.name), []).append(e.device_time if hasattr(e, "device_time") else e.cuda_time)
    per = {k: sum(v) / iters for k, v in per.items()}
    replay = None
    if graph:  # the step's form: the node's launches captured once, replayed; the idx buffer is rewritten between replays
        idx_buf = idxs[2].clone()
        keep = []
        T.CAPTURE_KEEP = keep
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                call(x, wt, smap, idx_buf, b)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call(x, wt, smap, idx_buf, b)
            torch.cuda.synchronize()
            times = []
            for i in range(iters):
                idx_buf.copy_(idxs[2 + i])
                torch.cuda.synchronize()
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record()
                g.replay()
                en.record()
                torch.cuda.synchronize()
                times.append(st.elapsed_time(en) * 1e3)
            replay = sorted(times)[iters // 2]
        finally:
            T.CAPTURE_KEEP = None
    return per, launches / iters, wall, replay


def identity(got, ref):
    if torch.equal(got, ref):
        return "identical"
    d = (got - ref).abs()
    where = int(d.view(-1).argmax().item())
    return f"differs: max |diff| {d.max().item():.3e} at {divmod(where, got.shape[1])}, {int((d > 0).sum().item())} elements"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--arms", default="0,1,3,7,15", help="values of FP4_FUSE_EPI, and 'fold' for the Nov 5 fold")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--slots", type=int, default=128, help="slots on this card (a layer's experts under TP4)")
    ap.add_argument("--slice", type=int, default=576, help="I of a slot: 576 the TP4 slice, 2304 the whole expert")
    ap.add_argument("--ep", type=int, default=1, help="experts per held one in the slot map (1: all held)")
    ap.add_argument("--graph", action="store_true", help="also time the call as a captured graph's replay")
    ap.add_argument("--no-direct", action="store_true", help="skip the direct kernels' reference output")
    ap.add_argument("--knobs", default="", help="NAME=VALUE,... set on triton_ops before the runs")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    for kv in filter(None, a.knobs.split(",")):
        k, v = kv.split("=")
        cur = getattr(T, k)
        setattr(T, k, v if cur is None or isinstance(cur, str) else type(cur)(v))
    I = a.slice
    b = banks(dev, a.slots, I)
    smap = slot_map(dev, a.slots, a.ep)
    g = torch.Generator().manual_seed(7)
    idxs = [draw(a.rows, a.slots * a.ep, g).to(dev) for _ in range(a.iters + 2)]
    x = (torch.randn(a.rows, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    wt = torch.rand(a.rows, TOPK, generator=g).to(dev)
    held = (smap[idxs[2].view(-1)] >= 0).sum().item()
    distinct = len({int(s) for s in smap[idxs[2].view(-1)].tolist() if s >= 0})
    per_slot = 3 * (I * K // 2 + I * K // 32)
    floor = distinct * per_slot / READ_PEAK / 1e3
    print(f"rows {a.rows} pairs {a.rows * TOPK} held {held} distinct slots {distinct} of {a.slots} (I {I}, K {K}); "
          f"HBM floor of the call {floor:.1f} us; arms {a.arms}; knobs {a.knobs}; "
          f"gdc {'yes' if T._gdc_wait is not None else 'no'}", flush=True)
    refs = {}
    if not a.no_direct:
        set_arm("direct")
        refs["direct"] = call(x, wt, smap, idxs[2], b).clone()
    for arm in a.arms.split(","):
        set_arm(arm)
        try:
            per, launches, wall, replay = timed(x, wt, smap, idxs, b, a.iters, a.graph)
            got = call(x, wt, smap, idxs[2], b).clone()
        except Exception as e:  # noqa: BLE001
            print(f"arm {arm}: failed: {str(e).splitlines()[0][:200]}", flush=True)
            continue
        if arm == "0":
            refs["unfused"] = got
        checks = {name: identity(got, ref) for name, ref in refs.items()}
        ksum = sum(per.values())
        parts = " ".join(f"{k} {per.get(k, 0.0):6.1f}" for k in ("gate_up", "down", "swiglu", "schedule", "aten", "other"))
        rep = f" graph {replay:6.1f}" if replay is not None else ""
        print(f"arm {arm:>6s} | {parts} | kernels {ksum:6.1f} launches {launches:4.1f} wall {wall:6.1f}{rep} us | "
              f"{100 * floor / max(ksum, 1e-9):3.0f}% of HBM | " + "; ".join(f"vs {k}: {v}" for k, v in checks.items()),
              flush=True)
        print("RESULT " + json.dumps({"arm": arm, "rows": a.rows, "pairs": a.rows * TOPK, "held": held, "distinct": distinct,
                                      "slots": a.slots, "slice": I, "ep": a.ep, "kernels_us": per, "kernels_sum_us": ksum,
                                      "launches": launches, "wall_us": wall, "graph_us": replay, "hbm_floor_us": floor,
                                      "identity": checks, "knobs": a.knobs}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
