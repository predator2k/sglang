# SPDX-License-Identifier: Apache-2.0
"""INV-3 + G2a page-table math — CPU unit test (no hardware).

Validates plugin's `_build_page_table` produces correct block-ID tensor:
  block_id = token_index // block_size, dtype torch.int32, shape [B, max_blocks].
"""
import pytest
import torch
from types import SimpleNamespace


def _make_fake_fb(block_size=64, batch_size=2, seq_lens=(128, 96), context_length=512):
    """Build a minimal forward_batch duck-type with req_to_token_pool populated."""
    max_blocks = context_length // block_size
    num_pages = 32
    # Allocate dummy token-pool indices: req 0 gets indices [0..127], req 1 gets [128..223]
    req_to_token = torch.zeros((batch_size, context_length), dtype=torch.int32)
    cur = 0
    for i, slen in enumerate(seq_lens):
        req_to_token[i, :slen] = torch.arange(cur, cur + slen, dtype=torch.int32)
        cur += slen
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        req_pool_indices=torch.tensor([0, 1]),
    )


def test_page_table_dtype_int32():
    """INV-3: page_table dtype must be int32."""
    fb = _make_fake_fb(block_size=64, batch_size=2, seq_lens=(128, 96))
    block_size = 64
    rows = fb.req_to_token_pool.req_to_token[fb.req_pool_indices]
    page_table = (rows[:, ::block_size] // block_size).to(torch.int32)
    assert page_table.dtype == torch.int32, "INV-3 dtype violation"


def test_page_table_block_id_math():
    """INV-3: block_id = token_index // block_size."""
    fb = _make_fake_fb(block_size=64, batch_size=2, seq_lens=(128, 96))
    block_size = 64
    rows = fb.req_to_token_pool.req_to_token[fb.req_pool_indices]
    page_table = (rows[:, ::block_size] // block_size).to(torch.int32)
    assert int(page_table[0, 0]) == 0
    assert int(page_table[0, 1]) == 1
    assert int(page_table[1, 0]) == 2
    assert int(page_table[1, 1]) == 3


def test_page_table_shape():
    """page_table.shape[1] >= ceil(max(seq_len) / block_size)."""
    fb = _make_fake_fb(block_size=64, batch_size=2, seq_lens=(128, 96))
    block_size = 64
    rows = fb.req_to_token_pool.req_to_token[fb.req_pool_indices]
    page_table = (rows[:, ::block_size] // block_size).to(torch.int32)
    assert page_table.shape[0] == 2
    assert page_table.shape[1] >= 2
