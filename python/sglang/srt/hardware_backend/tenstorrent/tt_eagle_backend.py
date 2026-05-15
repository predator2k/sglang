# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""Backend for EAGLE-3 draft on the TT plugin path.

Topology recap: target (e.g. Qwen3-8B) runs on tt_transformers across the
2× P150a mesh. The EAGLE-3 draft head (LlamaForCausalLMEagle3, loaded from
Tengyunw/qwen3_8b_eagle3) runs on CPU torch via SGLang's standard
ModelRunner path (gated by SGLANG_TT_SPEC_DRAFT_BACKEND=cpu).

SGLang's `draft_utils.DraftBackendFactory.create_decode_backend` whitelists
a fixed set of CUDA-centric attention backends (triton, flashinfer, fa3,
etc.). None of those work for our CPU-torch draft. This class plugs into
that whitelist (via the `"torch_native"` entry added in P3a.2 T2.2; see
REBASE_TARGETS.md) and satisfies the EAGLE worker's "backend has these
methods" contract.

For step >0 of the multi-step draft, `forward` delegates to SGLang's stock
TorchNativeAttnBackend so the draft's RadixAttention layers have a real
attention implementation. The per-step kv metadata SGLang's CUDA backends
normally populate is a no-op here (kv_indptr/kv_indices stay empty);
SGLang's accept-logic path on TT doesn't dereference them.

Notes:
  - No real cuda-graph state — irrelevant since TT doesn't use cuda graphs.
  - kv_indptr / kv_indices arrays are returned empty; the draft's forward
    doesn't read them. SGLang's accept logic does not dereference these
    in our path either — verified empirically in v94/v111+ (accept_rate
    measured at ~0.48 on real EAGLE-3 draft acceptance).
  - For step >0 of multi-step draft, `forward` delegates to the inner
    TorchNativeAttnBackend so the CPU EAGLE-3 draft's RadixAttention has
    a real attention implementation.
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
        # per-step inner backends. eagle_worker.draft_forward indexes this
        # list by step (`attn_backends[i]`) and assigns to forward_batch.
        # The CPU draft's RadixAttention forward calls
        # `forward_batch.attn_backend.forward(...)`, so we need a real
        # attention impl. Use SGLang's TorchNativeAttnBackend which runs
        # pure-torch SDPA — appropriate for the small CPU EAGLE-3 draft.
        try:
            from sglang.srt.layers.attention.torch_native_backend import (
                TorchNativeAttnBackend,
            )
            self.attn_backends: list = [
                TorchNativeAttnBackend(model_runner)
                for _ in range(speculative_num_steps)
            ]
        except Exception as exc:
            logger.warning(
                f"TorchNativeAttnBackend init failed ({exc!r}); "
                "draft attention will not work."
            )
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
        # Delegate to the inner TorchNativeAttnBackend so it can prepare its
        # per-request extend_prefix_lens/extend_seq_lens metadata. Without
        # this, forward_extend reads stale or zero metadata and the SDPA
        # output slice mismatches the output buffer (e.g. tries to copy
        # seq_len_kv rows into extend_seq_len rows).
        for inner in self.attn_backends:
            try:
                inner.init_forward_metadata(forward_batch)
            except Exception as exc:
                logger.warning(
                    f"TorchNativeAttnBackend.init_forward_metadata failed: {exc!r}"
                )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # No cuda graphs on TT.
        pass

    def init_forward_metadata_capture_cuda_graph(self, forward_batch):
        pass

    def init_forward_metadata_replay_cuda_graph(self, forward_batch, **kwargs):
        pass

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        """Delegate to the first inner TorchNativeAttnBackend.

        eagle_worker.forward_draft_extend_after_decode → draft model →
        RadixAttention.forward (radix_attention.py:138) calls
        `forward_batch.attn_backend.forward(...)`. The draft attn_backend in
        that path is THIS MultiStepDraftBackend (set by SGLang). The first
        inner attn_backend was constructed during __init__; delegate to it.

        P3a.2 EAGLE-3 fix: SGLang sometimes hands us a forward_batch with
        extend_prefix_lens=zeros even when seq_lens > extend_seq_lens. This
        causes TorchNative's SDPA output slice (seq_len_kv rows) to overflow
        the q-sized output buffer. Patch the field locally so the slice math
        works: prefix = max(seq_len - extend, 0).
        """
        if not self.attn_backends:
            raise RuntimeError(
                "TTMultiStepDraftBackend has no inner attn_backends; "
                "construction failed during __init__."
            )

        # Make extend_prefix_lens consistent with seq_lens - extend_seq_lens.
        try:
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.extend_prefix_lens is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.seq_lens is not None
            ):
                computed = (forward_batch.seq_lens - forward_batch.extend_seq_lens).clamp(min=0)
                if not torch.equal(forward_batch.extend_prefix_lens, computed):
                    forward_batch.extend_prefix_lens = computed
        except Exception as exc:
            logger.warning(f"extend_prefix_lens reconcile skipped: {exc!r}")

        return self.attn_backends[0].forward(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def forward_extend(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
