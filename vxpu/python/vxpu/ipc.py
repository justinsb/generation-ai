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

"""Device buffers shared between the processes of one step over CUDA IPC (hero run 16, #539).

One process per rank (multiproc.py) has no view of the peers' cards, so the push and scatter sum
forms, whose kernels store into the peers' inboxes and raise flags in their memory through peer
pointers, could not run across processes: parallel.World.buffers and allreduce.Comm allocated
every card's buffers from the one process. Here each process allocates its own card's buffers,
exports them once at start-up as CUDA IPC handles (torch.multiprocessing's reducer: the handle
of the allocation block, the offset and size, the reference counter the consumers hold), writes
them to a file in the directory the multiproc launcher already uses for NCCL's unique id, reads
the peers' files and opens their handles; the peers' buffers are then device tensors in this
process, on the peers' devices, and the kernels store into them as they stored into the one
process's allocations. Peer access between the cards is enabled as before (allreduce._enable_peers).

Every exchange is a rendezvous: share() returns only when every rank's file is there, so it
must be called by every rank, in the same order, before any capture (opening a handle inside a
capture is not allowed, as allocating there was not). Opened handles stay open for the process's
life; torch keeps an exporter's allocation alive until every consumer has released its reference,
so the owner never frees under a peer. The files carry the run's tag (the unique id file's
modification time), so a stale file from an earlier run in the same directory is never read.
"""

import hashlib
import os
import pickle
import time

TIMEOUT = float(os.environ.get("VXPU_IPC_TIMEOUT", "7200"))  # a peer's load may be long (its packages build)


def _default_reduce(tensor):
    from torch.multiprocessing.reductions import reduce_tensor
    fn, args = reduce_tensor(tensor)
    return pickle.dumps((fn, args))


def _default_rebuild(payload):
    fn, args = pickle.loads(payload)
    return fn(*args)


class Exchange:
    """The files of one run under `directory`: rank `rank` of `size` publishes a payload under a
    name and gathers every rank's. `reduce`/`rebuild` turn a tensor into bytes and back (torch's
    CUDA IPC reducer; a test's fake)."""

    def __init__(self, directory, rank, size, tag, reduce=None, rebuild=None):
        self.directory, self.rank, self.size, self.tag = directory, int(rank), int(size), str(tag)
        self.reduce = reduce or _default_reduce
        self.rebuild = rebuild or _default_rebuild
        self.opened = []  # what rebuild() returned, kept alive for the process's life (close() drops them)
        os.makedirs(directory, exist_ok=True)

    def _path(self, name, rank):
        short = hashlib.sha1(name.encode()).hexdigest()[:12]
        return os.path.join(self.directory, f"ipc-{self.tag}-{short}.r{rank}")

    def publish(self, name, payload):
        path = self._path(name, self.rank)
        with open(path + ".tmp", "wb") as f:
            f.write(payload)
        os.replace(path + ".tmp", path)

    def gather(self, name, timeout=None):
        """Every rank's payload for `name`, in rank order (waits for the peers' files)."""
        timeout = TIMEOUT if timeout is None else timeout
        out = [None] * self.size
        deadline = time.time() + timeout
        pending = set(range(self.size))
        while pending:
            for r in sorted(pending):
                path = self._path(name, r)
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        out[r] = f.read()
                    pending.discard(r)
            if pending:
                if time.time() > deadline:
                    raise TimeoutError(f"ipc: rank {self.rank} waited {timeout:.0f} s for ranks {sorted(pending)} at {name!r}")
                time.sleep(0.01)
        return out

    def share(self, name, tensor):
        """My `tensor` exported and every rank's opened: a list by rank, mine the tensor itself."""
        self.publish(name, self.reduce(tensor))
        out = []
        for r, payload in enumerate(self.gather(name)):
            if r == self.rank:
                out.append(tensor)
            else:
                t = self.rebuild(payload)
                self.opened.append(t)
                out.append(t)
        return out

    def barrier(self, name, timeout=None):
        """Every rank arrives at `name` before any leaves."""
        self.publish(name, b"")
        self.gather(name, timeout)

    def close(self):
        """Drop the peers' buffers this process opened (before the owners free theirs)."""
        self.opened.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass


_EXCHANGE = None


def exchange():
    """The process's Exchange under the multiproc launcher's directory (VXPU_MP_UID's), tagged by the
    unique id file's modification time (rank 0 writes it at every run's start)."""
    global _EXCHANGE
    if _EXCHANGE is None:
        from . import multiproc
        uid = os.environ["VXPU_MP_UID"]
        multiproc.write_uid()
        deadline = time.time() + TIMEOUT
        while not os.path.exists(uid):
            if time.time() > deadline:
                raise TimeoutError(f"ipc: no unique id file at {uid}")
            time.sleep(0.05)
        tag = str(os.stat(uid).st_mtime_ns)
        _EXCHANGE = Exchange(os.path.dirname(os.path.abspath(uid)), multiproc.rank(), multiproc.world_size(), tag)
    return _EXCHANGE


def share(name, tensor):
    return exchange().share(name, tensor)


def barrier(name):
    return exchange().barrier(name)


def key_name(prefix, key):
    """A file name for a buffer key (a tuple the step graph builds): stable across the ranks."""
    return f"{prefix}:{key!r}"
