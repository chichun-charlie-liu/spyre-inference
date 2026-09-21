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
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor import spyre_hint

# The joint core-division/placement co-optimizer doesn't return in practical
# time for this kernel's graph, and doesn't help this kernel even when it
# does converge. Equivalent to CO_OPTIMIZING_LX_PLANNING=0.
_spyre_config.co_optimizing_lx_planning = False


def _matmul_split(num_kv_heads: int) -> dict[str, int]:
    """Best-known attention-matmul work division for this kernel's shapes.

    ~1.7x faster than the framework's unhinted default, via
    ``spyre_hint(work_div=...)`` alone -- see torch-spyre's
    EXPERIMENTS_SUMMARY.md (gather-to-lx paged-attention investigation) for
    the full sweep. Splitting ``Hnum`` requires it to divide evenly; models
    with a ``num_kv_heads`` not divisible by 4 fall back to splitting only
    the query-token axis (not yet swept for a better split of their own).
    """
    if num_kv_heads % 4 == 0:
        return {"T": 8, "Hnum": 4}
    return {"T": 8}


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

    # Gathered, not sliced: a compiled region reads a view from offset 0 and ignores its
    # strides (torch-spyre#3770).
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )

    matmul_split = _matmul_split(num_kv_heads)

    def _hinted_matmul(a, b):
        with spyre_hint(work_div=matmul_split):
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
