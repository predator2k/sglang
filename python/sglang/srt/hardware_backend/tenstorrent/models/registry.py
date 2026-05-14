# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""
This module handles the runtime patching of SGLang to use TT-Metal models.
"""

import logging

from sglang.srt.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def _build_tt_model_registry():
    """Mapping from HuggingFace architecture names to TT model classes."""
    from .tt_llm import (
        TenstorrentGptOssForCausalLM,
        TenstorrentLlamaForCausalLM,
        TenstorrentMistralForCausalLM,
        TenstorrentQwenForCausalLM,
    )
    return {
        "LlamaForCausalLM": TenstorrentLlamaForCausalLM,
        "Qwen2ForCausalLM": TenstorrentQwenForCausalLM,
        "Qwen3ForCausalLM": TenstorrentQwenForCausalLM,
        "MistralForCausalLM": TenstorrentMistralForCausalLM,
        "GptOssForCausalLM": TenstorrentGptOssForCausalLM,
    }


def get_tt_model_class_for_arch(arch: str):
    """Look up the Tenstorrent model class for a given HF architecture string.

    Used by P3a.2 eagle_draft.py to resolve the draft model class without
    relying on SGLang's loader pipeline. Returns None if the arch is unknown.
    """
    return _build_tt_model_registry().get(arch)


def register_tt_models():
    """Register TT-Metal models with SGLang's model registry."""
    logger.info("[TT-Plugin] register_tt_models() called")
    try:
        TT_MODEL_REGISTRY = _build_tt_model_registry()
        logger.info("[TT-Plugin] Imported TT model classes successfully")

        # CRITICAL: Directly patch SGLang's ModelRegistry
        ModelRegistry.models.update(TT_MODEL_REGISTRY)
        logger.info(
            f"[TT-Plugin] ✓ Registered {len(TT_MODEL_REGISTRY)} TT models: {list(TT_MODEL_REGISTRY.keys())}"
        )

    except Exception as e:
        logger.error(
            f"[TT-Plugin] Error registering TT models: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
