# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

"""TenstorrentXLAGenericCausalLM -- HF model wrapper compiled via tt-xla.

Follows Pattern A (ModelRegistry): loaded by SGLang's ModelRunner, same
interface as TTModels in tt_llm.py.

tt-xla key constraints discovered during P3b Task 2.1:
  - torch.compile(backend="tt") is REQUIRED -- naive model.to(xla_device)
    produces inf logits.
  - StaticCache is required for generation; DynamicCache triggers recompilation.
  - SPMD mode (xr.use_spmd()) enables multi-device tensor parallelism.
  - First invocation per unique input shape triggers JIT compilation (~9s for
    TinyLlama 1.1B on 2x P150a Blackhole).
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
        from transformers.cache_utils import StaticCache

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
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=True,
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

        # Allocate a StaticCache on device. tt-xla requires StaticCache
        # to avoid recompilation on every forward.
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads

        self.static_cache = StaticCache(
            config=config,
            max_batch_size=1,
            max_cache_len=self.max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )
        self.static_cache.early_initialization(
            batch_size=1,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )
        # Move cache to XLA device.
        for layer in self.static_cache.layers:
            layer.keys = layer.keys.to(self.device)
            layer.values = layer.values.to(self.device)

        # Track current cache position for decode steps.
        self._cache_position = 0

        # Pre-allocate attention mask for max_cache_len.
        self._full_attn_mask = torch.zeros(
            (1, self.max_cache_len), dtype=torch.long
        ).to(self.device)

        logger.info(
            f"[TT-XLA] StaticCache allocated: max_cache_len={self.max_cache_len}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )

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
        input_2d = input_ids.unsqueeze(0).to(self.device)

        # Set up cache position for this prefill.
        cache_pos = torch.arange(0, seq_len).to(self.device)

        # Set attention mask: mark prompt positions as 1.
        # Reset mask first, then set prompt tokens.
        self._full_attn_mask = torch.zeros(
            (1, self.max_cache_len), dtype=torch.long
        ).to(self.device)
        self._full_attn_mask[0, :seq_len] = 1

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self.compiled_model(
                input_ids=input_2d,
                past_key_values=self.static_cache,
                cache_position=cache_pos,
                use_cache=True,
                attention_mask=self._full_attn_mask,
            )
        dt = time.perf_counter() - t0
        logger.debug(f"[TT-XLA] Prefill {seq_len} tokens in {dt:.2f}s")

        # Extract last-position logits and move to CPU.
        logits = output.logits[:, -1:, :].to("cpu").float()

        self._cache_position = seq_len

        return LogitsProcessorOutput(
            next_token_logits=logits.squeeze(0),
        )

    def _forward_decode(self, input_ids, positions, forward_batch):
        """Decode: process one token at a time."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        # For B=1 decode: input_ids is [1], reshape to [1, 1].
        input_2d = input_ids.unsqueeze(0).to(self.device)

        cache_pos = torch.tensor([self._cache_position]).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self.compiled_model(
                input_ids=input_2d,
                past_key_values=self.static_cache,
                cache_position=cache_pos,
                use_cache=True,
                attention_mask=self._full_attn_mask,
            )
        dt = time.perf_counter() - t0
        logger.debug(f"[TT-XLA] Decode step at pos {self._cache_position} in {dt:.3f}s")

        logits = output.logits[:, -1:, :].to("cpu").float()

        self._cache_position += 1

        return LogitsProcessorOutput(
            next_token_logits=logits.squeeze(0),
        )
