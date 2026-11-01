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

# The fixed cost of a graph node on this GPU, inside one CUDA graph: what a kernel costs whatever it does, so a
# step's launch-bound tail has a floor (hero run 11). N dependent repetitions of each node kind are captured in
# one graph and replayed; the time per repetition is printed.
#   python launch_floor.py [N]
# Measured on an RTX PRO 6000 (Nov 1 2026, N = 500): one-element kernel 0.75 us, memset node 0.69, fill kernel
# (torch.zeros) 0.69, one-program kernel over a 32 KB row 1.47, three of those in a chain 4.82 (1.6 each), a 32 KB
# device-to-device copy 0.97.
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _tiny(x_ptr, n: tl.constexpr):
    v = tl.load(x_ptr)
    tl.store(x_ptr, v + 1.0)


@triton.jit
def _row(x_ptr, off, BLOCK: tl.constexpr):
    idx = off + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + idx)
    tl.store(x_ptr + idx, v * 1.0001)


def measure(label, body, n):
    """The time per repetition of body(i) for i in range(n), captured in one CUDA graph."""
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        body(0)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for i in range(n):
                body(i)
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    print(f"LAUNCHFLOOR {label}: {dt * 1e6 / n:.2f} us per repetition", flush=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    x = torch.zeros(1, device="cuda")
    cnt = torch.zeros(64, dtype=torch.int32, device="cuda")
    y = torch.zeros(64 << 20, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        _tiny[(1,)](x, 1)
        _row[(1,)](y, 0, 16384)
    torch.cuda.synchronize()
    measure("one-element kernel", lambda i: _tiny[(1,)](x, 1), n)
    measure("memset node (counter.zero_())", lambda i: cnt.zero_(), n)
    measure("memset node + one-element kernel", lambda i: (cnt.zero_(), _tiny[(1,)](x, 1)), n)
    measure("fill kernel (torch.zeros(64) each call)", lambda i: torch.zeros(64, dtype=torch.int32, device="cuda"), n)
    measure("one-program 32 KB row kernel", lambda i: _row[(1,)](y, (i * 16384) % (60 << 20), 16384), n)
    measure("three dependent one-program 32 KB row kernels",
            lambda i: [_row[(1,)](y, ((3 * i + k) * 16384) % (60 << 20), 16384) for k in range(3)], n)
    measure("32 KB device-to-device copy", lambda i: y[16384 * (i % 100):16384 * (i % 100 + 1)].copy_(y[16384 * 1000:16384 * 1001]), n)


if __name__ == "__main__":
    main()
