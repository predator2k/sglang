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

PJRT plugin compiler workarounds (P3b), HW-validated 2026-05-16:
  - Blocker 1 (ttir.paged_update_cache): When use_cache=False (the naive
    workaround for cache issues), HF calls aten.scatter_.src for internal
    operations, which pjrt-plugin-tt 1.1.0 lowers to ttir.paged_update_cache
    (not implemented). Workaround: use StaticCache with explicit
    cache_position. StaticCache.update() uses index_copy_ (not scatter_),
    which TT-MLIR lowers correctly. Proven by test_tt_xla_smoke.py and
    test_tt_xla_e2e.py on 2x P150a Blackhole.
  - Blocker 2 (add_int UInt8): When use_cache=False, HF's internal
    _update_causal_mask creates boolean/UInt8 tensors that Blackhole's add_int
    kernel rejects (only Int32/UInt32/UInt16 supported). Workaround: pass
    an explicit Int32 attention_mask so HF skips its internal mask creation.

  - Blocker 3 (position_ids device mismatch): transformers >=5.6 computes
    position_ids internally via past_seen_tokens (CPU) + arange(device=XLA),
    which torch.compile rejects as a cross-device add. Workaround: pass
    explicit position_ids on XLA device so HF skips internal computation.

Known limitations discovered during HW validation:
  - torch.compile graph retracing after cache mutation (zero_) fails with
    error 13; multi-sequence serving needs a single compile trace.
  - JIT compilation takes ~9s per unique input shape (prefill vs decode);
    second invocation with cached graph runs in ~0.001s per token.
  - StaticCache.layers[*].cumulative_length stays on CPU; tt_torch backend
    auto-moves it, but logs warnings. Harmless.
"""

from __future__ import annotations

import logging
import os
import time
import torch
from torch import nn

# transformers 5.6.0 raises KeyError('flash_attn') when flash-attn is
# not installed, which breaks tt_torch imports.  Populate the mapping
# before any transformers submodule touches it.
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash-attn"])
except Exception:
    pass

logger = logging.getLogger(__name__)


class TenstorrentXLAGenericCausalLM(nn.Module):
    """Generic HF CausalLM compiled via torch_xla for TT devices.

    SGLang's ModelRunner calls forward(input_ids, positions, forward_batch)
    and expects a LogitsProcessorOutput.

    KV cache strategy: device-resident StaticCache with explicit cache_position
    and Int32 attention_mask. This avoids both PJRT compiler blockers:
      - StaticCache uses index_copy_ (not scatter_), so Blocker 1 is avoided
      - Explicit Int32 mask bypasses HF's internal UInt8 mask (Blocker 2)
    The pattern is proven by test_tt_xla_smoke.py (TinyLlama generation).
    """

    def __init__(
        self,
        config,
        quant_config=None,
        **kwargs,
    ):
        super().__init__()

        import torch_xla
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

        # Load HF model: use_cache=True (StaticCache needs it),
        # attn_implementation="eager" (SDPA corrupts device state on tt-xla).
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=True,
            attn_implementation="eager",
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

        # Extract model geometry for StaticCache initialization.
        self._num_kv_heads = config.num_key_value_heads
        self._head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self._num_layers = config.num_hidden_layers

        # StaticCache with early initialization (Blocker 1 workaround).
        # Created with device="cpu" for early_initialization, then cache
        # tensors are moved to XLA device. StaticCache.update() uses
        # index_copy_ (not scatter_), which TT-MLIR lowers correctly.
        from sglang.srt.hardware_backend.tenstorrent.models.tt_functional_cache import (
            TTFunctionalCache,
        )
        self._static_cache = TTFunctionalCache(
            config=config,
            max_batch_size=1,  # bs > 1 wired in Phase 3b
            max_cache_len=self.max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )
        # Force early allocation of all layer cache tensors.
        self._static_cache.early_initialization(
            batch_size=1,
            num_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )

        # Move cache KV tensors to XLA device for the compiled model.
        # cumulative_length and layer.device stay on CPU — tt_torch
        # backend auto-moves them during torch.compile tracing.
        for layer in self._static_cache.layers:
            layer.keys = layer.keys.to(self.device)
            layer.values = layer.values.to(self.device)

        # Pre-allocate the full attention mask (Int32 -- Blocker 2 workaround).
        # Blackhole's add_int kernel only supports Int32/UInt32/UInt16.
        # By supplying an explicit Int32 mask, HF skips its internal
        # _update_causal_mask which would produce UInt8 tensors.
        self._full_attn_mask = torch.zeros(
            (1, self.max_cache_len), dtype=torch.int32,
        )

        # Tracking for the current cache fill position.
        self._cache_pos = 0
        self._needs_reset = False

        logger.info(
            f"[TT-XLA] StaticCache mode, "
            f"max_cache_len={self.max_cache_len}, "
            f"num_kv_heads={self._num_kv_heads}, "
            f"head_dim={self._head_dim}"
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

    def _get_pad_bucket(self, seq_len: int) -> int:
        """Round seq_len up to a fixed bucket for graph reuse.

        Uses SGLANG_TT_PREFILL_PAD_STEP (default: power-of-2 buckets).
        Set to a fixed value like 2048 to compile only one prefill graph.
        """
        step = int(os.environ.get("SGLANG_TT_PREFILL_PAD_STEP", "0"))
        if step > 0:
            return min(((seq_len + step - 1) // step) * step, self.max_cache_len)
        bucket = 32
        while bucket < seq_len:
            bucket *= 2
        return min(bucket, self.max_cache_len)

    def _set_cumulative_length(self, value: int):
        """Set cumulative_length on all cache layers BEFORE a compiled call.

        tt_torch auto-moves CPU tensors to XLA on every compiled call
        (logs: "Force moving the argument to XLA"). So changing the CPU
        tensor here ensures the XLA copy has the correct value when the
        compiled graph runs. We DON'T clear KV tensors — the attention
        mask blocks stale positions.
        """
        for layer in self._static_cache.layers:
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length.fill_(value)

    def _reset_cache(self):
        """Reset for a new sequence.

        T2.1 (v5.3 spec): the prior workaround called torch._dynamo.reset() here
        to force re-export of the model. That invalidated JIT-compiled graphs on
        every request and caused ~9s prefill + ~9s first-decode cliffs per request
        (probe_c3_warmup_vs_cache_*.log). TTFunctionalCache replaces the in-place
        index_copy_ that required dynamo.reset; we no longer reset the JIT cache.
        """
        self._static_cache.reset()
        self._full_attn_mask.fill_(0)
        self._cache_pos = 0

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
        """Prefill: process full prompt through the model with StaticCache.

        Uses the same pattern proven in test_tt_xla_smoke.py:
        1. Reset cache for new sequence
        2. Create cache_position covering [0, seq_len)
        3. Set attention mask bits for valid positions
        4. Pass explicit Int32 mask + cache_position to avoid both blockers
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        seq_len = input_ids.shape[0]

        # Reset cache for new sequence (skip first — tensors are already zero).
        if self._needs_reset:
            self._reset_cache()
        else:
            self._needs_reset = True
            self._full_attn_mask.fill_(0)
            self._cache_pos = 0

        # Right-pad input to a fixed bucket for JIT graph reuse.
        pad_len = self._get_pad_bucket(seq_len)

        input_padded = torch.zeros(pad_len, dtype=torch.int32)
        input_padded[:seq_len] = input_ids.to(torch.int32)
        input_2d = input_padded.unsqueeze(0).to(self.device)

        cache_pos = torch.arange(0, pad_len)
        cache_pos_dev = cache_pos.to(self.device)

        position_ids = torch.zeros(pad_len, dtype=torch.long)
        position_ids[:seq_len] = torch.arange(0, seq_len)
        position_ids = position_ids.unsqueeze(0).to(self.device)

        self._full_attn_mask[:, :seq_len] = 1
        attn_mask_dev = self._full_attn_mask.to(self.device)

        # Decode starts at seq_len (right after real tokens, skipping pad KV).
        self._cache_pos = seq_len

        # Set cumulative_length=0 so StaticCacheLayer writes KV at [0, pad_len).
        # After forward, cumulative_length will be pad_len on the XLA copy,
        # but we reset it to seq_len before the first decode call.
        self._set_cumulative_length(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self.compiled_model(
                input_ids=input_2d,
                past_key_values=self._static_cache,
                cache_position=cache_pos_dev,
                use_cache=True,
                attention_mask=attn_mask_dev,
                position_ids=position_ids,
            )
        dt = time.perf_counter() - t0
        logger.info(f"[TT-XLA] Prefill {seq_len} tokens (padded {pad_len}) in {dt:.2f}s")

        # Extract logits at the last REAL token (not pad position).
        logits = output.logits[:, seq_len - 1:seq_len, :].to("cpu").float()

        return LogitsProcessorOutput(
            next_token_logits=logits.squeeze(0),
        )

    def _forward_decode(self, input_ids, positions, forward_batch):
        """Decode: process single new token using StaticCache.

        Incremental decode: only the new token is sent through the model;
        KV values for prior tokens are already in the StaticCache.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        t_start = time.perf_counter()

        input_2d = input_ids.unsqueeze(0).to(torch.int32).to(self.device)
        cache_pos_dev = torch.tensor([self._cache_pos]).to(self.device)
        position_ids = torch.tensor(
            [[self._cache_pos]], dtype=torch.long
        ).to(self.device)
        self._full_attn_mask[:, self._cache_pos] = 1
        attn_mask_dev = self._full_attn_mask.to(self.device)
        self._set_cumulative_length(self._cache_pos)
        self._cache_pos += 1

        t_prep = time.perf_counter()

        with torch.no_grad():
            output = self.compiled_model(
                input_ids=input_2d,
                past_key_values=self._static_cache,
                cache_position=cache_pos_dev,
                use_cache=True,
                attention_mask=attn_mask_dev,
                position_ids=position_ids,
            )

        t_fwd = time.perf_counter()

        logits = output.logits[:, -1:, :].to("cpu").float()

        t_end = time.perf_counter()

        if self._cache_pos % 10 == 0:
            logger.info(
                f"[TT-XLA] Decode pos {self._cache_pos}: "
                f"prep={1000*(t_prep-t_start):.1f}ms "
                f"fwd={1000*(t_fwd-t_prep):.1f}ms "
                f"logits={1000*(t_end-t_fwd):.1f}ms "
                f"total={1000*(t_end-t_start):.1f}ms"
            )

        return LogitsProcessorOutput(
            next_token_logits=logits.squeeze(0),
        )
