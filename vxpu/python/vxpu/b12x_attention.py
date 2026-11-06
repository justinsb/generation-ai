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
"""Their sparse-MLA decode kernel (the vLLM fork's b12x attention.sparse_mla) as the body of
vxpu::attn_select and vxpu::attn_select_rows (hero run 15, VXPU_ATTENTION_KERNEL=b12x).

Their kernel takes q [rows, heads, d] bf16, a paged KV cache [pages, 64, d] bf16, physical slot
indices [rows, width] int32 with a valid count per row, a float32 sink per head and the scale. Our
attention at a layer is one softmax over two caches, the window ring and the compressed cache, so
each call concatenates the two into one padded buffer per cache row (a copy a layer a step), builds
the selection as the reference does (the fork's get_window_topk_idxs for the ring, the compressed
indices offset past the ring), maps it to physical slots (cache row x padded length + index, the
valid entries compacted first), and runs their kernel; the epilogue (the rope's inverse on the last
rope_dim dims and the fp8 grid) is the reference's aten ops. Our index kernels and caches are
unchanged. A plan per (rows, heads, width, scale) is prepared at its first call, before any capture.
"""
import math
import os
import sys

import torch

ATTENTION_KERNEL = os.environ.get("VXPU_ATTENTION_KERNEL", "vxpu")
B12X_PATH = os.environ.get("VXPU_B12X_PATH", "")
PAGE = 64
_mods = None
_plans = {}


def on():
    """Whether the plan routes the attention ops to their kernel."""
    return ATTENTION_KERNEL == "b12x"


def _modules():
    global _mods
    if _mods is None:
        for p in reversed([p for p in B12X_PATH.split(":") if p]):
            if p not in sys.path:
                sys.path.insert(0, p)
        from b12x.attention import sparse_mla
        from b12x.attention.sparse_mla import _preparation
        _mods = (sparse_mla, _preparation)
    return _mods


class _Plan:
    """One prepared plan of theirs: the caps, the plan, its state and its scratch buffer."""

    def __init__(self, rows, heads, width, d, scale, device):
        sparse_mla, prep = _modules()
        self.caps = sparse_mla.Caps(device=device, num_q_heads=heads, max_q_rows=rows, max_width=width,
                                    softmax_scale=float(scale), dtype=torch.bfloat16, kv_dtype=torch.bfloat16,
                                    head_dim=d, v_head_dim=d, mode="decode", max_batch=rows,
                                    max_chunks_per_row=max(1, math.ceil(width / PAGE)), page_size=PAGE,
                                    has_attention_sink=True)
        self.plan = sparse_mla.plan(self.caps)
        self.state = prep.state(self.plan, device=device)  # prepared with its default configuration, scratch included

    def run(self, q2, kvc, idx, lengths, cache_lengths, sink):
        binding = self.state.bind(q=q2, selected_indices=idx, cache_seqlens_int32=cache_lengths,
                                  nsa_cache_seqlens_int32=lengths, kv_cache=kvc)
        return self.state.run(binding, kv_cache=kvc, attention_sink=sink)


def _plan(rows, heads, width, d, scale, device):
    key = (rows, heads, width, d, float(scale), str(device))
    p = _plans.get(key)
    if p is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"b12x attention: the plan for {key} is not prepared and the stream is capturing; the "
                               "prime's eager step prepares every shape before the capture")
        p = _plans[key] = _Plan(rows, heads, width, d, scale, device)
    return p


def _paged(kv_window, kv_compress):
    """The two caches of every cache row side by side in one buffer padded to whole pages: [rows,
    Lp, d] viewed as [rows * Lp / 64, 64, d]; returns the view, the ring's slots and Lp."""
    B, slots, d = kv_window.shape
    n2 = kv_compress.shape[1]
    L = slots + n2
    Lp = -(-L // PAGE) * PAGE
    buf = torch.empty(B, Lp, d, dtype=kv_window.dtype, device=kv_window.device)
    buf[:, :slots].copy_(kv_window)
    if n2:
        buf[:, slots:L].copy_(kv_compress)
    if Lp != L:
        buf[:, L:].zero_()
    return buf.view(B * Lp // PAGE, PAGE, d), slots, L, Lp


def _run(q2, kv_window, kv_compress, idx, cache_row, attn_sink, scale):
    """q2 [rows, heads, d]; idx [rows, width] indices into the concatenation (-1 none); cache_row
    [rows] the cache row each query reads. Returns o [rows, heads, d] bf16."""
    rows, heads, d = q2.shape
    kvc, slots, L, Lp = _paged(kv_window, kv_compress)
    valid = idx >= 0
    phys = torch.where(valid, idx + (cache_row.to(idx.dtype) * Lp).unsqueeze(1), idx)
    order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)  # the valid entries first, in order
    phys = torch.gather(phys, 1, order).to(torch.int32).contiguous()
    lengths = valid.sum(1).to(torch.int32)
    cache_lengths = torch.full((rows,), L, dtype=torch.int32, device=q2.device)
    plan = _plan(rows, heads, idx.shape[1], d, scale, q2.device)
    out = plan.run(q2.contiguous(), kvc, phys, lengths, cache_lengths, attn_sink.float().contiguous())
    if isinstance(out, tuple):
        out = out[0]
    return out.reshape(rows, heads, d).to(q2.dtype)


def _finish(o, freqs_table, start, seqlen, rope_dim):
    """The reference's epilogue: the rope's inverse on the last rope_dim dims, then the fp8 grid."""
    from . import ops
    ds, ka = ops._fork_kernels()
    hd = o.shape[-1]
    tail = ops._rotary_reference(o[..., hd - rope_dim:], ops._freqs_at(freqs_table, start, seqlen), inverse=True).to(o.dtype)
    o = torch.cat([o[..., :hd - rope_dim], tail], dim=-1).contiguous()
    return ka.act_quant(o, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype, True)


def attn_select(q, kv_window, kv_compress, attn_sink, compress_idxs, softmax_scale, freqs_table, start, seqlen,
                rope_dim, window):
    """vxpu::attn_select (a decode step: q [b, s, h, d], every row its own caches)."""
    from . import ops
    ds, _ = ops._fork_kernels()
    b, s, h, d = q.shape
    slots = kv_window.shape[1]
    widx = ds.get_window_topk_idxs(window, slots, b, s, start, q.device)
    idx = torch.cat([widx, torch.where(compress_idxs >= 0, compress_idxs + slots, compress_idxs)], dim=-1)
    cache_row = torch.arange(b, device=q.device).repeat_interleave(s)
    o = _run(q.reshape(b * s, h, d), kv_window, kv_compress, idx.reshape(b * s, -1), cache_row, attn_sink, softmax_scale)
    return _finish(o.view(b, s, h, d), freqs_table, start, seqlen, rope_dim)


def attn_select_rows(q, kv_window, kv_compress, attn_sink, compress_idxs, softmax_scale, freqs_table, pos, rows, newest,
                     rope_dim, window):
    """vxpu::attn_select_rows (the flat token-row form: q [t, 1, h, d], whole caches, row r reads cache row rows[r])."""
    from . import ops
    ds, _ = ops._fork_kernels()
    t, _, h, d = q.shape
    slots = kv_window.shape[1]
    widx = ds.get_window_topk_idxs(window, slots, t, 1, pos, q.device, newest=newest)
    idx = torch.cat([widx, torch.where(compress_idxs >= 0, compress_idxs + slots, compress_idxs)], dim=-1)
    o = _run(q.reshape(t, h, d), kv_window, kv_compress, idx.reshape(t, -1), rows, attn_sink, softmax_scale)
    return _finish(o.view(t, 1, h, d), freqs_table, pos, 1, rope_dim)
