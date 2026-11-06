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

"""The world of a rank that lives in a process of its own (#539, slice 2).

parallel.World answers the step's collectives for N ranks on N threads of one process: a
barrier and N slots. ProcessWorld is that world for one rank, with the other ranks in other
processes: the eager collective publishes this rank's partial in its exchange mailbox and
reads every other rank's from theirs, in rank order, and sums as the thread world does, so
the totals are the same bits on every rank and the same bits as the one-process engine's.
The step runs on the calling thread (there is one rank here), inside the engine's scope.

This is the eager CPU path only. The captured step's exchange (World.buffers: inboxes and
flags on every rank's GPU, written by peer stores from one process) needs CUDA IPC handles
across processes, which is slice 4's work with the per-rank NCCL communicators; buffers()
says so.
"""

import struct
import types

import torch

from .. import parallel
from . import mailbox

DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.int32, torch.uint8, torch.bool]
_HEAD = struct.Struct("<BB")  # dtype code, ndim; then ndim int64 sizes


def exchange_boxes(size, capacity, slots=2):
    """The exchange mailboxes of a world of `size` ranks: rank r publishes its partials in
    box x{r}, read by every other rank. `capacity` bounds one partial's bytes."""
    return {f"x{r}": mailbox.BoxSpec(r, [j for j in range(size) if j != r], capacity, slots) for r in range(size)}


def pack_tensor(x):
    x = x.detach().contiguous().cpu()
    code = DTYPES.index(x.dtype)
    head = _HEAD.pack(code, x.dim()) + struct.pack(f"<{x.dim()}q", *x.shape)
    return head + x.reshape(-1).view(torch.uint8).numpy().tobytes() if x.numel() else head


def unpack_tensor(payload):
    code, ndim = _HEAD.unpack_from(payload, 0)
    shape = struct.unpack_from(f"<{ndim}q", payload, _HEAD.size)
    body = payload[_HEAD.size + 8 * ndim:]
    dtype = DTYPES[code]
    if not body:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(body), dtype=dtype).reshape(shape).clone()


class ProcessWorld(parallel.World):
    """parallel.World for rank `rank` of `size` ranks, one per process, over `region` (a
    mailbox.Region whose boxes include exchange_boxes(size) and whose barrier's members include
    the ranks). `devices` is every rank's device, as in the thread world; `local` is the one
    rank this process builds (Engine._build_tp, parallel.Decode)."""

    def __init__(self, region, rank, devices, scope=None):
        super().__init__(devices, scope=scope)
        self.region = region
        self.rank_id = int(rank)
        self.local = [self.rank_id]
        self._local = types.SimpleNamespace(rank=None)  # set while a step runs (run); ops.py reads it
        self._boxes = [region.mailbox[f"x{r}"] for r in range(self.size)]
        self.exchanges = 0

    def rank(self):
        if self._local.rank is None:
            raise RuntimeError("ppr: a collective ran outside the rank's step")
        return self._local.rank

    def run(self, fns, timeout=None):
        """fns has one entry, this rank's; it runs here, on the calling thread, in the engine's scope."""
        if len(fns) != 1:
            raise ValueError(f"ppr: {len(fns)} functions for the one rank of this process")
        self._local.rank = self.rank_id
        try:
            with self.scope() if self.scope is not None else parallel._nullcontext():
                return [fns[0]()]
        finally:
            self._local.rank = None

    def barrier(self):
        if getattr(self, "solo", False):
            return
        self.region.barrier.wait(self.rank_id)

    def _exchange(self, x):
        """Every rank's x in rank order: mine published, the others' received (a clone of mine,
        as the thread world keeps: the caller overwrites x with the result)."""
        r = self.rank()
        self.exchanges += 1
        self._boxes[r].publish(pack_tensor(x))
        parts = []
        for j in range(self.size):
            if j == r:
                parts.append(x.detach().clone())
            else:
                _seq, payload = self._boxes[j].receive(r)
                parts.append(unpack_tensor(payload).to(x.device))
        return parts

    def buffers(self, k, like):
        raise NotImplementedError("ppr: the captured step's exchange buffers live on the ranks' GPUs and are written by "
                                  "peer stores from one process; across processes they need CUDA IPC handles or the "
                                  "per-rank NCCL communicators (#539, slice 4)")
