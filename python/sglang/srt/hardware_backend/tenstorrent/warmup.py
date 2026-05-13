"""Decode-shape JIT warmup driver.

Runs a dummy short prefill + one decode step through `TTTransformersExecutionBackend`
so tt-metal's program cache compiles before the first real request. The first
real prompt then skips the multi-second JIT cost.

Logs `warmup_start` / `warmup_done` per spec §6.2.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_backend import (
        TTTransformersExecutionBackend,
    )

logger = logging.getLogger("sglang.srt.hardware_backend.tenstorrent")

_WARMUP_REQ_ID = "__warmup__"


def warm_decode_shape(
    wrapper: "TTTransformersExecutionBackend",
    *,
    dummy_token: int = 0,
    prompt_len: int = 32,
) -> None:
    """Prime tt-metal's program cache with the decode-step shape.

    Runs prefill on a `prompt_len`-long sequence of `dummy_token`, then one
    decode step, then frees the request. The wrapper's TT-side state ends
    up identical to "no warmup ran" but the program cache now contains the
    compiled decode kernel.
    """
    logger.info(
        "warmup_start",
        extra={"shape": (1, 1), "prompt_len": prompt_len},
    )
    t0 = time.time()
    wrapper._do_new_request(_WARMUP_REQ_ID, prompt_tokens=[dummy_token] * prompt_len)
    try:
        wrapper._do_extend(_WARMUP_REQ_ID)
        wrapper._do_decode_step(_WARMUP_REQ_ID, dummy_token)
    finally:
        wrapper._do_free(_WARMUP_REQ_ID)
    logger.info("warmup_done", extra={"elapsed_s": time.time() - t0})
