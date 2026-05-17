# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""tt_xla_prewarm — server-side JIT pre-warm covering every prefill bucket + decode.

Why: probe_c3_per_token.log shows the first prefill (~10s) and first decode (~9s)
are both JIT compiles. Without pre-warm, every fresh process pays both cliffs on
its first real request. Worse, real requests vary in input length, so they hit
different prefill buckets — each bucket needs its own compile. The reviewer of
spec v5.3 flagged this: pre-warming only the max bucket leaves p99 cliff-prone
for any non-max-bucket request.

This function enumerates every bucket value _get_pad_bucket() can return and runs
one dummy prefill at each, plus one dummy decode. Total cost: O(num_buckets * 9s)
at startup, once per process lifetime.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.tenstorrent.models.tt_xla_model import (
        TenstorrentXLAGenericCausalLM,
    )

logger = logging.getLogger(__name__)


def _enumerate_buckets(max_cache_len: int) -> list[int]:
    """Yield all bucket sizes _get_pad_bucket() can return, ascending."""
    step = int(os.environ.get("SGLANG_TT_PREFILL_PAD_STEP", "0"))
    if step > 0:
        return [
            min(b, max_cache_len)
            for b in range(step, max_cache_len + step, step)
        ]
    # Power-of-2 buckets starting at 32, capped at max_cache_len.
    buckets: list[int] = []
    bucket = 32
    while bucket <= max_cache_len:
        buckets.append(bucket)
        bucket *= 2
    if buckets and buckets[-1] < max_cache_len:
        buckets.append(max_cache_len)
    return buckets


def tt_xla_prewarm(model: "TenstorrentXLAGenericCausalLM") -> None:
    """Compile every prefill bucket + the single decode shape. One-time at startup."""
    import torch_xla

    device = model.device
    max_cache_len = model.max_cache_len

    buckets = _enumerate_buckets(max_cache_len)
    logger.info(f"[TT-XLA] pre-warm: {len(buckets)} buckets {buckets} + 1 decode")

    for bucket in buckets:
        t0 = time.perf_counter()

        # Dummy prefill — use token id 1 (valid for all sentencepiece-based tokenizers).
        input_ids = torch.ones((1, bucket), dtype=torch.int32, device=device)
        cache_pos = torch.arange(0, bucket, device=device)
        position_ids = torch.arange(0, bucket, dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.ones((1, max_cache_len), dtype=torch.int32, device=device)

        # Reset so this dummy prefill doesn't corrupt any subsequent real request.
        model._static_cache.reset()
        model._full_attn_mask.fill_(0)
        model._cache_pos = 0
        model._needs_reset = True

        with torch.no_grad():
            _ = model.compiled_model(
                input_ids=input_ids,
                past_key_values=model._static_cache,
                cache_position=cache_pos,
                use_cache=True,
                attention_mask=attn_mask,
                position_ids=position_ids,
            )
        torch_xla.sync()
        logger.info(
            f"[TT-XLA] pre-warm prefill bucket={bucket} compiled in {time.perf_counter()-t0:.1f}s"
        )

    # One dummy decode at cache_pos=0.
    t0 = time.perf_counter()
    input_ids = torch.ones((1, 1), dtype=torch.int32, device=device)
    cache_pos = torch.tensor([0], device=device)
    position_ids = torch.tensor([[0]], dtype=torch.long, device=device)
    attn_mask = torch.ones((1, max_cache_len), dtype=torch.int32, device=device)
    with torch.no_grad():
        _ = model.compiled_model(
            input_ids=input_ids,
            past_key_values=model._static_cache,
            cache_position=cache_pos,
            use_cache=True,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
    torch_xla.sync()
    logger.info(f"[TT-XLA] pre-warm decode compiled in {time.perf_counter()-t0:.1f}s")

    # Final reset so the next real request starts clean.
    model._static_cache.reset()
    model._full_attn_mask.fill_(0)
    model._cache_pos = 0
    model._needs_reset = False
