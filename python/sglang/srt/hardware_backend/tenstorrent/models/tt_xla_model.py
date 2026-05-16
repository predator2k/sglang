# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

"""TenstorrentXLAGenericCausalLM -- HF model wrapper compiled via tt-xla.

Follows Pattern A (ModelRegistry): loaded by SGLang's ModelRunner, same
interface as TTModels in tt_llm.py.

tt-xla key constraints discovered during P3b Task 2.1:
  - torch.compile(backend="tt") is REQUIRED -- naive model.to(xla_device)
    produces inf logits.
  - SPMD mode (xr.use_spmd()) enables multi-device tensor parallelism.
  - First invocation per unique input shape triggers JIT compilation (~9s for
    TinyLlama 1.1B on 2x P150a Blackhole).
  - ttir.paged_update_cache is NOT supported by TT-MLIR, so StaticCache on
    the XLA device causes scatter lowering failures. Workaround: run without
    KV cache (use_cache=False), reprocessing the full sequence each step.
    This is O(n^2) in sequence length but avoids the compiler limitation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class TenstorrentXLAGenericCausalLM(nn.Module):
    """Generic HF CausalLM compiled via torch_xla for TT devices.

    SGLang's ModelRunner calls forward(input_ids, positions, forward_batch)
    and expects a LogitsProcessorOutput.

    KV cache is managed externally by tracking the full token history on CPU.
    Each forward pass reprocesses the entire sequence (no device-side cache)
    to avoid ttir.paged_update_cache lowering failure in TT-MLIR.
    """

    def __init__(
        self,
        config,
        quant_config=None,
        **kwargs,
    ):
        super().__init__()

        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        from transformers import AutoModelForCausalLM

        self.config = config
        model_path = getattr(config, "_name_or_path", None)
        logger.info(f"[TT-XLA] Loading {model_path}")

        # Enable SPMD for multi-device tensor parallelism.
        os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
        xr.set_device_type("TT")
        xr.use_spmd()

        self.device = torch_xla.device()
        self.num_devices = xr.global_runtime_device_count()
        logger.info(
            f"[TT-XLA] Device: {self.device}, num_devices: {self.num_devices}"
        )

        # Load HF model in bfloat16, then compile with tt backend.
        # use_cache=False: avoid StaticCache scatter ops that TT-MLIR cannot
        # lower (ttir.paged_update_cache not implemented).
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
        )
        self.hf_model.eval()
        self.hf_model = self.hf_model.to(self.device)
        self.compiled_model = torch.compile(self.hf_model, backend="tt")
        logger.info(
            f"[TT-XLA] Model compiled: {type(self.hf_model).__name__}"
        )

        # Read server args for cache sizing.
        try:
            from sglang.srt.server_args import get_global_server_args

            server_args = get_global_server_args()
            self.max_cache_len = server_args.context_length or 2048
        except Exception:
            self.max_cache_len = 2048

        # CPU-side token history for full-sequence recompute.
        # Each forward pass sends the entire accumulated token sequence
        # to the compiled model.
        self._token_history: list[int] = []

        logger.info(
            f"[TT-XLA] No-cache mode (full recompute per step), "
            f"max_cache_len={self.max_cache_len}"
        )

    def load_weights(self, weights):
        """No-op: weights already loaded via AutoModelForCausalLM.from_pretrained.

        SGLang's ModelRunner.load_model calls model.load_weights(weight_iter)
        after __init__. For TT-XLA the HF model is fully loaded and compiled
        during __init__, so we just drain the iterator without using it.
        """
        # Drain the generator to avoid resource warnings.
        for _ in weights:
            pass
        logger.info("[TT-XLA] load_weights: skipped (model loaded in __init__)")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor = None,
    ):
        """Run compiled HF model forward pass on TT device.

        Args:
            input_ids: [total_tokens] flat token IDs
            positions: [total_tokens] position indices
            forward_batch: ForwardBatch with mode, seq_lens, etc.

        Returns:
            LogitsProcessorOutput with next_token_logits on CPU
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        # Determine sequence lengths from forward_batch.
        if forward_batch.forward_mode.is_extend():
            return self._forward_prefill(input_ids, positions, forward_batch)
        else:
            return self._forward_decode(input_ids, positions, forward_batch)

    def _forward_prefill(self, input_ids, positions, forward_batch):
        """Prefill: process full prompt through the model."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        # For B=1 prefill: input_ids is [seq_len], reshape to [1, seq_len].
        seq_len = input_ids.shape[0]

        # Store token history for future decode recomputes.
        self._token_history = input_ids.cpu().tolist()

        # Cast to int32: TT-metal Blackhole kernels only support
        # Int32/UInt32/UInt16 for integer ops; torch.long (int64) fails.
        input_2d = input_ids.unsqueeze(0).to(torch.int32).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self.compiled_model(
                input_ids=input_2d,
                use_cache=False,
            )
        dt = time.perf_counter() - t0
        logger.info(f"[TT-XLA] Prefill {seq_len} tokens in {dt:.2f}s")

        # Extract last-position logits and move to CPU.
        logits = output.logits[:, -1:, :].to("cpu").float()

        return LogitsProcessorOutput(
            next_token_logits=logits.squeeze(0),
        )

    def _forward_decode(self, input_ids, positions, forward_batch):
        """Decode: append new token and recompute full sequence."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        # Append the new token to history.
        new_tokens = input_ids.cpu().tolist()
        self._token_history.extend(new_tokens)

        # Build full sequence tensor (int32: TT-metal Blackhole kernels
        # only support Int32/UInt32/UInt16).
        full_seq = torch.tensor(
            [self._token_history], dtype=torch.int32
        ).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self.compiled_model(
                input_ids=full_seq,
                use_cache=False,
            )
        dt = time.perf_counter() - t0

        seq_len = len(self._token_history)
        if seq_len % 10 == 0:
            logger.info(
                f"[TT-XLA] Decode recompute at pos {seq_len} in {dt:.3f}s"
            )

        logits = output.logits[:, -1:, :].to("cpu").float()

        return LogitsProcessorOutput(
            next_token_logits=logits.squeeze(0),
        )
