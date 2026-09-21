# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paged attention over a head-major KV cache for a query wider than one token.

``page_attn_head_major`` buys LX page residency with an unrolled matmul per query group and
a ``stack`` epilogue that cannot fuse. Past one query token the page transfer that buys is
amortised over every query row, so this kernel spends it instead: batched GQA over
``[kv_head, group, query, D]``, one accumulator, store fuses.
"""

import torch

from torch_spyre._inductor import spyre_hint

# T/8, Hnum/4 (32 cores total) is the best matmul division found for this
# kernel's shape (num_heads=32, num_kv_heads=8, head_size=128, padded_query_len
# up to 512): 930.7us -> 792.9us over the framework's own unhinted default,
# via spyre_hint(work_div=...) alone -- see torch-spyre's
# EXPERIMENTS_SUMMARY.md (gather-to-lx paged-attention Q/K/V investigation)
# for the full sweep and ground truth. Assumes num_kv_heads is divisible by
# 4; a model with a different num_kv_heads may need a different split (not
# yet swept).
_MATMUL_SPLIT = {"T": 8, "Hnum": 4}


def page_attn_head_major_prefill_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_tables,
    mask_tiles,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Online softmax attention over ``num_blocks`` pages of the unfolded cache.

    Shapes are ``page_attn_head_major``'s, except ``page_index_tables``: one [1] int32 device
    tensor per active block, indexing ``[num_blocks, num_kv_heads, block_size, head_size]``.
    """
    num_queries_per_kv = num_heads // num_kv_heads

    # Split the flat head axis into [num_kv_heads, num_queries_per_kv] BEFORE the
    # gather, not after: the gather's own output then already carries the two
    # head sub-axes as real dims, so torch-spyre can commit an Hnum x Hgrp
    # matmul division directly onto it without a clone.
    query_split = query.reshape(query.shape[0], num_kv_heads, num_queries_per_kv, head_size)
    # Gathered, not sliced: a compiled region reads a view from offset 0 and ignores its
    # strides (torch-spyre#3770).
    #
    # Name q_rows (the gather's own output, still [T, Hnum, Hgrp, D] -- real,
    # separate dims) here, not q (post-permute): spyre_hint's named_dims=
    # only attaches to FX nodes actually created within its scope, and a
    # bare permute is a pure view Inductor elides/folds into the matmul's
    # own read index rather than materializing its own buffer -- a hint
    # wrapping only the permute never survives to reach any buffer's
    # metadata. Naming the gather's real output here, and hinting its
    # division directly (only "T" is a legal split axis for a gather; a
    # gather can only split the axis it indexes over), lets it reach LX at
    # full core utilization instead of falling back to whatever division
    # `_distribute_work` would otherwise pick unprompted. Letting
    # propagation carry the names through the subsequent permute (the same
    # way it already handles k_page.transpose(-2,-1) with no hint of its
    # own) is what makes the matmul's own work_div hint below resolve.
    with spyre_hint(named_dims=["T", "Hnum", "Hgrp", "D"], work_div=_MATMUL_SPLIT):
        q_rows = query_split.index_select(0, query_row_index[:padded_query_len])
    q = q_rows.permute(1, 2, 0, 3)

    def _hinted_matmul(a, b):
        with spyre_hint(work_div=_MATMUL_SPLIT):
            return torch.matmul(a, b)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        # One row of the unfolded cache: the folded per-kv-head gather exists to split for LX
        # residency. index_select, not subscripting, which lowers to aten.index and fails eager.
        page_idx = page_index_tables[i]
        with spyre_hint(named_dims=["Hnum", "Hgrp", "St", "D"]):
            k_page = k_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
        with spyre_hint(named_dims=["Hnum", "Hgrp", "St", "D"]):
            v_page = v_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
        mask_tile = mask_tiles[i]

        scores = _hinted_matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so capping after it
            # would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = _hinted_matmul(tile_probs, v_page)
            tile_sum = tile_probs.sum(dim=-1, keepdim=True)
        else:
            assert tile_max is not None
            assert tile_sum is not None
            assert tile_output is not None
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale
            tile_sum = tile_sum * rescale
            tile_probs = torch.exp(scores - new_max)
            tile_output = tile_output + _hinted_matmul(tile_probs, v_page)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = tile_output / tile_sum
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if out is not None:
        # Storing the full padded extent keeps this sequence's real query_len out of the
        # arguments, so it is not specialized on.
        out.index_copy_(0, query_row_index[:padded_query_len], attn[:padded_query_len])
        return out
    return attn
