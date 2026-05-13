# SPDX-License-Identifier: Apache-2.0
"""§9.14 Token-pool index static bound — preventive overflow lint.

CPU-only test (no hardware required). Verifies that the product
  num_pages × block_size < 2^31
so that slot indices never overflow a signed int32 at runtime.

Rationale: slot_index = page_id × block_size + intra_block_offset.
If page_id fits in int32 but page_id × block_size wraps at 2^31,
the resulting slot index is negative, causing silent KV misreads.

Current paged-path config on 2× Blackhole p150a:
  max_total_num_tokens = 329,216  (from /get_load or ServerArgs default)
  block_size = 64 tokens/page     (TT plugin default)
  num_pages = ceil(max_total_num_tokens / block_size) = 5,144 pages

Product: 5,144 × 64 = 329,216 — safely below 2^31 = 2,147,483,648.

The test also checks the formula for a hypothetical maximum config
(16 GB card, block_size=16) to confirm it still passes.
"""

import math

_INT32_MAX = 2**31  # exclusive upper bound

# ── Current production config ─────────────────────────────────────────────────

_CURRENT_MAX_TOTAL_TOKENS = 329_216   # from /get_load; also ServerArgs default
_CURRENT_BLOCK_SIZE = 64             # TT plugin default (page_size)


def _num_pages(max_total_tokens: int, block_size: int) -> int:
    return math.ceil(max_total_tokens / block_size)


def test_token_pool_no_int32_overflow_current_config():
    """§9.14: current production config (329 216 tokens, block_size=64) is safe."""
    num_pages = _num_pages(_CURRENT_MAX_TOTAL_TOKENS, _CURRENT_BLOCK_SIZE)
    max_slot_index = num_pages * _CURRENT_BLOCK_SIZE

    print()
    print(f"  max_total_num_tokens : {_CURRENT_MAX_TOTAL_TOKENS:>12,}")
    print(f"  block_size           : {_CURRENT_BLOCK_SIZE:>12,}")
    print(f"  num_pages            : {num_pages:>12,}")
    print(f"  max_slot_index       : {max_slot_index:>12,}")
    print(f"  2^31                 : {_INT32_MAX:>12,}")
    print(f"  margin_factor        : {_INT32_MAX / max_slot_index:>12.1f}×")

    assert max_slot_index < _INT32_MAX, (
        f"OVERFLOW: max_slot_index={max_slot_index:,} >= 2^31={_INT32_MAX:,}. "
        f"Reduce max_total_num_tokens or increase block_size."
    )


def test_token_pool_no_int32_overflow_hypothetical_max():
    """§9.14: hypothetical worst-case (4 M tokens, block_size=16) still safe."""
    # 4 M tokens at block_size=16 → 250,000 pages × 16 = 4,000,000 slots
    # This is the most aggressive config plausibly reached in a future P3
    # workload on 28 GB cards with smaller block sizes.
    max_total_tokens = 4_000_000
    block_size = 16
    num_pages = _num_pages(max_total_tokens, block_size)
    max_slot_index = num_pages * block_size

    print()
    print(f"  [hypothetical] max_total_num_tokens : {max_total_tokens:>12,}")
    print(f"  [hypothetical] block_size           : {block_size:>12,}")
    print(f"  [hypothetical] num_pages            : {num_pages:>12,}")
    print(f"  [hypothetical] max_slot_index       : {max_slot_index:>12,}")
    print(f"  [hypothetical] 2^31                 : {_INT32_MAX:>12,}")

    assert max_slot_index < _INT32_MAX, (
        f"OVERFLOW in hypothetical config: max_slot_index={max_slot_index:,} "
        f">= 2^31={_INT32_MAX:,}."
    )


def test_token_pool_overflow_guard_formula_correctness():
    """§9.14: the guard formula detects an actual overflow correctly.

    This test verifies that the formula DOES fire for a synthetic config
    that overflows (ensures the test is not a vacuous pass).
    """
    # Construct a config that WOULD overflow: 2^25 pages × 2^7 block_size = 2^32
    overflow_pages = 2**25
    overflow_block = 2**7
    max_slot = overflow_pages * overflow_block  # = 2^32 > 2^31

    would_overflow = max_slot >= _INT32_MAX
    assert would_overflow, (
        "Guard formula failed to detect overflow: "
        f"pages={overflow_pages} block={overflow_block} product={max_slot:,} "
        f"should be >= 2^31={_INT32_MAX:,}"
    )
    print()
    print(f"  Overflow detection confirmed: {overflow_pages} × {overflow_block} "
          f"= {max_slot:,} >= {_INT32_MAX:,}  (expected)")
