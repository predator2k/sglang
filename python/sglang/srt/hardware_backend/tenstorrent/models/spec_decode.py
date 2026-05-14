# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""SpecDecodeAdapter — verify-batch routing for plugin path.

INV-8: P3a uses SGLang's native speculative_algorithm runtime. This adapter
is a THIN wrapper — it routes forward_batch.spec_info presence to
_verify_forward (draft-tree shape), or to _standard_forward (existing
prefill/decode path).

Does NOT implement a verification loop. SGLang's spec scheduler owns that.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class SpecDecodeAdapter:
    """Wraps a TenstorrentLlamaForCausalLM with verify-batch routing.

    Usage (called from TenstorrentLlamaForCausalLM.forward):
        if getattr(forward_batch, "spec_info", None) is not None:
            return self._spec_adapter.forward(forward_batch)
        # else: existing standard forward path
    """

    def __init__(self, model):
        self._model = model

    def forward(self, forward_batch: "ForwardBatch") -> "LogitsProcessorOutput":
        spec_info = getattr(forward_batch, "spec_info", None)
        # Only route to verify path when the forward_mode actually is
        # TARGET_VERIFY. EAGLE additionally uses DRAFT_EXTEND / DRAFT_EXTEND_V2
        # forward modes whose spec_info is an EagleDraftInput (no
        # draft_token_num field) — those should go through the standard
        # path so the draft model's regular extend/decode runs.
        fm = forward_batch.forward_mode
        if spec_info is not None and fm.is_target_verify():
            return self._verify_forward(forward_batch)
        return self._standard_forward(forward_batch)

    def _verify_forward(self, forward_batch: "ForwardBatch") -> "LogitsProcessorOutput":
        """Verify draft-proposed tokens.

        forward_batch.spec_info carries draft tree; forward_batch.input_ids is
        the flattened draft-token sequence; forward_batch.positions is per-
        draft-token position. We call plugin's prefill_forward in "small
        prefill" mode (it returns logits per draft position).
        """
        return self._model._call_prefill_for_verify(forward_batch)

    def _standard_forward(self, forward_batch: "ForwardBatch") -> "LogitsProcessorOutput":
        """Delegate to model's existing forward path."""
        return self._model._call_standard_forward(forward_batch)
