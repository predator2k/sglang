# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""TT-native stub for EAGLE's MultiStepDraftBackend (P3a.2 T2.2 boot).

SGLang's `draft_utils.DraftBackendFactory.create_decode_backend` whitelists
a fixed set of CUDA-centric backends (triton, flashinfer, fa3, etc.). None
of them work on TT — our paged path uses tt_transformers directly via
model.forward, bypassing SGLang's attention backend layer.

This stub satisfies the EAGLE worker's "backend has these methods" contract
just enough to let boot complete. The draft model's actual attention happens
inside its own tt_transformers forward pass; the metadata bookkeeping
SGLang's backends normally do is a no-op here because our paged path
manages its own kv layout.

Limitations (expected, documented for follow-up):
  - No real cuda-graph state — irrelevant since TT doesn't use cuda graphs.
  - kv_indptr / kv_indices arrays are returned empty; the draft's forward
    doesn't read them.
  - Spec-decoding will still run, but draft acceptance may be miscomputed
    if SGLang's accept logic dereferences our empty index tensors.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class TTMultiStepDraftBackend:
    """Pass-through backend for EAGLE draft on tt_transformers paged kv."""

    def __init__(
        self,
        model_runner: Any,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.device = getattr(model_runner, "device", "cpu")
        self.max_context_len = getattr(
            getattr(model_runner, "server_args", None), "context_length", 16384
        )
        self.num_head = (
            model_runner.model_config.num_attention_heads
            if hasattr(model_runner, "model_config")
            else 32
        )
        self.pool_len = (
            model_runner.req_to_token_pool.req_to_token.shape[1]
            if hasattr(model_runner, "req_to_token_pool")
            else 16384
        )
        self.page_size = getattr(
            getattr(model_runner, "server_args", None), "page_size", 64
        )
        # SGLang's MultiStepDraftBackend exposes `.attn_backends` as a list of
        # per-step inner backends. Provide an empty list so any iteration is
        # a no-op (the actual attention is done by tt_transformers internally).
        self.attn_backends: list = []
        # Pre-allocated buffers some downstream code expects to read.
        max_bs = getattr(
            getattr(model_runner, "req_to_token_pool", None), "size", 64
        ) * self.topk
        self.kv_indptr = torch.zeros(
            (self.speculative_num_steps, max_bs + 1),
            dtype=torch.int32,
            device=self.device,
        )

    def common_template(
        self,
        forward_batch,
        kv_indices_buffer,
        call_fn,
    ):
        """Iterate speculative steps invoking the model's draft forward.

        SGLang's reference backends populate per-step kv_indptr/kv_indices on
        forward_batch.spec_info before calling `call_fn`. We skip the kv
        bookkeeping (our model's forward doesn't read it) and just dispatch.
        """
        if call_fn is None:
            return
        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch):
        # No metadata setup needed — tt_transformers manages its own kv state.
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # No cuda graphs on TT.
        pass

    def init_forward_metadata_capture_cuda_graph(self, forward_batch):
        pass

    def init_forward_metadata_replay_cuda_graph(self, forward_batch, **kwargs):
        pass

    def forward_extend(self, *args, **kwargs):
        # Routed through model.forward → tt_transformers.
        raise NotImplementedError(
            "TTMultiStepDraftBackend.forward_extend should not be called; "
            "the draft model's tt_transformers forward handles attention."
        )

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "TTMultiStepDraftBackend.forward_decode should not be called; "
            "the draft model's tt_transformers forward handles attention."
        )
