# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One process per rank (hero run 15, the first slice of the process-per-rank execution model).

With VXPU_MP_WORLD=N and VXPU_MP_RANK=r set, this process is rank r of a tensor-parallel world of
N processes, one card each (cuda:r), instead of one process driving N cards from threads. The
engine's rank loops visit only this rank (parallel.World.ranks), every collective of the step is an
NCCL call on this rank's communicator (an all-reduce of the fp32 partial, then the reference
epilogue; an all-gather for the head), issued on the current stream and so captured into the
rank's CUDA graph, and the host loop that drafts, accepts and launches runs in every process over
the same data: the ranks hold the gathered logits after the head, so each process makes the same
choices without a broadcast (a timing-free policy is required: fixed, slots by survival).

The binding is ctypes over the libnccl torch ships, one communicator per process, made from the
unique id rank 0 writes to VXPU_MP_UID (a file the launcher names). NCCL connects its channels at a
communicator's first collective with a handshake over the bootstrap socket, which would deadlock
if that first collective were issued inside a capture with the peers elsewhere, so init() runs a
set of eager warm-up collectives across all ranks before any capture.
"""
import ctypes
import ctypes.util
import glob
import os
import time

import torch

_LIB = None
_COMM = None
_RANK = None
_WORLD = None

NCCL_FLOAT = 7
NCCL_BF16 = 9
NCCL_SUM = 0
_DTYPES = {torch.float32: NCCL_FLOAT, torch.bfloat16: NCCL_BF16}


def active():
    return "VXPU_MP_WORLD" in os.environ


def rank():
    return int(os.environ["VXPU_MP_RANK"])


def world_size():
    return int(os.environ["VXPU_MP_WORLD"])


def device():
    return torch.device("cuda", rank())


def _lib():
    global _LIB
    if _LIB is None:
        here = os.path.dirname(torch.__file__)
        cands = (glob.glob(os.path.join(here, "..", "nvidia", "nccl", "lib", "libnccl.so*"))
                 + glob.glob(os.path.join(here, "lib", "libnccl*")))
        found = ctypes.util.find_library("nccl")
        if found:
            cands.append(found)
        if not cands:
            raise RuntimeError("multiproc: no libnccl next to torch or on the library path")
        _LIB = ctypes.CDLL(cands[0])
        _LIB.ncclGetErrorString.restype = ctypes.c_char_p
    return _LIB


class _UniqueId(ctypes.Structure):
    _fields_ = [("b", ctypes.c_char * 128)]


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"multiproc: {what} failed: {_lib().ncclGetErrorString(rc).decode()} ({rc})")


def init(timeout=600.0):
    """This process's communicator: rank 0 writes the unique id to VXPU_MP_UID, the others wait for
    it, every rank joins; then the warm-up collectives. Idempotent."""
    global _COMM, _RANK, _WORLD
    if _COMM is not None:
        return
    _RANK, _WORLD = rank(), world_size()
    torch.cuda.set_device(_RANK)
    lib = _lib()
    path = os.environ["VXPU_MP_UID"]
    uid = _UniqueId()
    if _RANK == 0:
        _check(lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        with open(path + ".tmp", "wb") as f:
            f.write(bytes(uid.b))
        os.replace(path + ".tmp", path)
    else:
        deadline = time.time() + timeout
        while not (os.path.exists(path) and os.path.getsize(path) == 128):
            if time.time() > deadline:
                raise RuntimeError(f"multiproc: rank {_RANK} waited {timeout:.0f} s for the unique id at {path}")
            time.sleep(0.05)
        with open(path, "rb") as f:
            uid.b = f.read(128)
    comm = ctypes.c_void_p()
    started = time.perf_counter()
    _check(lib.ncclCommInitRank(ctypes.byref(comm), _WORLD, uid, _RANK), f"ncclCommInitRank rank {_RANK}")
    _COMM = comm
    ver = ctypes.c_int()
    lib.ncclGetVersion(ctypes.byref(ver))
    # the warm-up: every peer connection NCCL makes lazily (ring and tree, small and large messages, both
    # collectives, both dtypes), eagerly and with the stream drained, before any capture records a collective
    with torch.cuda.device(_RANK):
        for dtype in (torch.float32, torch.bfloat16):
            for n in (1, 4096, 1 << 18, 1 << 22, 1 << 24):
                x = torch.full((n,), float(_RANK + 1), dtype=dtype, device=_RANK)
                y = allreduce(x)
                torch.cuda.synchronize(_RANK)
                want = _WORLD * (_WORLD + 1) / 2
                if abs(float(y[0]) - want) > 1e-3 * want:
                    raise RuntimeError(f"multiproc: the warm-up all-reduce read {float(y[0])}, not {want}")
            for n in (16, 1 << 16, 1 << 20):
                x = torch.full((2, n), float(_RANK + 1), dtype=dtype, device=_RANK)
                g = allgather(x)
                torch.cuda.synchronize(_RANK)
                if tuple(g.shape) != (2, _WORLD * n) or float(g[0, -1]) != float(_WORLD):
                    raise RuntimeError(f"multiproc: the warm-up all-gather read shape {tuple(g.shape)}, last {float(g[0, -1])}")
    print(f"[vxpu] multiproc: rank {_RANK} of {_WORLD} on cuda:{_RANK}, NCCL {ver.value}, communicator and warm-up in "
          f"{time.perf_counter() - started:.1f} s", flush=True)


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream(_RANK).cuda_stream)


def allreduce(x):
    """The sum over the ranks of x (fp32 or bf16, any shape), a new tensor on the current stream."""
    if _COMM is None:
        init()
    xc = x.contiguous()
    out = torch.empty_like(xc)
    _check(_lib().ncclAllReduce(ctypes.c_void_p(xc.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                                ctypes.c_size_t(xc.numel()), _DTYPES[xc.dtype], NCCL_SUM, _COMM, _stream()),
           "ncclAllReduce")
    return out


def allgather(x):
    """Every rank's x concatenated along the last dimension in rank order (the fork's all_gather)."""
    if _COMM is None:
        init()
    xc = x.contiguous()
    buf = torch.empty((_WORLD,) + tuple(xc.shape), dtype=xc.dtype, device=xc.device)
    _check(_lib().ncclAllGather(ctypes.c_void_p(xc.data_ptr()), ctypes.c_void_p(buf.data_ptr()),
                                ctypes.c_size_t(xc.numel()), _DTYPES[xc.dtype], _COMM, _stream()), "ncclAllGather")
    return buf.movedim(0, xc.dim() - 1).reshape(tuple(xc.shape[:-1]) + (_WORLD * xc.shape[-1],))


def allreduce_exact(x):
    """x summed as fp32 across the ranks and rounded back to x's dtype: the reference's class."""
    if x.dtype == torch.float32:
        return allreduce(x)
    return allreduce(x.float()).to(x.dtype)
