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
        # WS-B (qwen3_5): placeholder registration — reuse the Qwen3 dense bridge
        # so SGLang's loader and tt_transformers' Qwen generator are at least reachable.
        # The actual decode path will hit Gated-DeltaNet / MRoPE / partial-rotary /
        # attn_output_gate gaps that WS-A is expected to close. We register both
        # the standalone causal LM arch and the ConditionalGeneration arch that HF
        # auto-selects for the multimodal Qwen3.5 release. Vision is intentionally
        # routed to the same text-only bridge; multimodal is out of scope for first port.
        "Qwen3_5ForCausalLM": TenstorrentQwenForCausalLM,
        "Qwen3_5ForConditionalGeneration": TenstorrentQwenForCausalLM,
        # TODO(WS-A): MoE (256 experts top-8) and MTP variants are out of scope for
        # first port — registered here as the same placeholder so that an explicit
        # arch mismatch isn't the first failure. They will fail later inside the
        # bridge until the dedicated kernels land.
        "Qwen3_5MoeForCausalLM": TenstorrentQwenForCausalLM,
        "Qwen3_5MoeForConditionalGeneration": TenstorrentQwenForCausalLM,
        "Qwen3_5ForCausalLMMTP": TenstorrentQwenForCausalLM,
        "MistralForCausalLM": TenstorrentMistralForCausalLM,
        "GptOssForCausalLM": TenstorrentGptOssForCausalLM,
    }


def get_tt_model_class_for_arch(arch: str):
    """Look up the Tenstorrent model class for a given HF architecture string.

    Used by P3a.2 eagle_draft.py to resolve the draft model class without
    relying on SGLang's loader pipeline. Returns None if the arch is unknown.
    """
    return _build_tt_model_registry().get(arch)


_LAZY_EAGLE_DRAFTS = {
    # arch name → "package.module:ClassName"
    "LlamaForCausalLMEagle3": "sglang.srt.models.llama_eagle3:LlamaForCausalLMEagle3",
}


def _patch_model_registry_lazy_eagle3():
    """Hook ModelRegistry.resolve_model_cls to lazy-import EAGLE drafts.

    We can't import sglang.srt.models.llama_eagle3 at TT plugin load time
    because the plugin loads while sglang.srt.layers.utils.multi_platform
    is mid-execution (current_platform → discover_platforms → load TT
    plugin → here). By the time resolve_model_cls is first called for an
    EAGLE draft (during draft worker init in the scheduler subprocess),
    the layers chain has finished loading and the import succeeds.
    """
    if getattr(ModelRegistry, "_tt_lazy_eagle_patched", False):
        return

    original_resolve = ModelRegistry.resolve_model_cls

    def patched_resolve(self, architectures):
        for arch in architectures:
            if arch in _LAZY_EAGLE_DRAFTS and arch not in self.models:
                mod_path, cls_name = _LAZY_EAGLE_DRAFTS[arch].split(":")
                try:
                    import importlib
                    mod = importlib.import_module(mod_path)
                    self.models[arch] = getattr(mod, cls_name)
                    logger.info(
                        f"[TT-Plugin] lazy-registered {arch} from {mod_path}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[TT-Plugin] lazy import of {arch} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        return original_resolve(architectures)

    import types
    ModelRegistry.resolve_model_cls = types.MethodType(patched_resolve, ModelRegistry)
    ModelRegistry._tt_lazy_eagle_patched = True
    logger.info("[TT-Plugin] ✓ Patched ModelRegistry.resolve_model_cls for lazy EAGLE drafts")


def _build_tt_xla_registry():
    """Mapping from HF architectures to the generic tt-xla wrapper class."""
    from .tt_xla_model import TenstorrentXLAGenericCausalLM

    TT_XLA_ARCHITECTURES = [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "MistralForCausalLM",
        "PhiForCausalLM",
        "Phi3ForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
    ]
    return {arch: TenstorrentXLAGenericCausalLM for arch in TT_XLA_ARCHITECTURES}


def register_tt_models():
    """Register TT models with SGLang's model registry.

    Dispatches between tt_transformers (model-specific classes) and tt_xla
    (generic wrapper compiled via torch.compile(backend='tt')) based on
    SGLANG_TT_EXECUTION_BACKEND.
    """
    logger.info("[TT-Plugin] register_tt_models() called")
    try:
        from sglang.srt.hardware_backend.tenstorrent.execution import (
            resolve_execution_backend_name,
        )

        backend = resolve_execution_backend_name()

        if backend == "tt_xla":
            # tt-xla: register generic wrapper for supported architectures.
            TT_MODEL_REGISTRY = _build_tt_xla_registry()
            logger.info(
                f"[TT-Plugin] Registered tt-xla for {len(TT_MODEL_REGISTRY)} architectures: "
                f"{list(TT_MODEL_REGISTRY.keys())}"
            )
        else:
            # tt_transformers: register model-specific classes.
            TT_MODEL_REGISTRY = _build_tt_model_registry()
            logger.info(
                f"[TT-Plugin] Imported TT model classes successfully"
            )

        # CRITICAL: Directly patch SGLang's ModelRegistry
        ModelRegistry.models.update(TT_MODEL_REGISTRY)
        logger.info(
            f"[TT-Plugin] Registered {len(TT_MODEL_REGISTRY)} TT models: "
            f"{list(TT_MODEL_REGISTRY.keys())}"
        )

        # P3a.2 EAGLE-3 draft registration (only relevant for tt_transformers).
        if backend != "tt_xla":
            _patch_model_registry_lazy_eagle3()

    except Exception as e:
        logger.error(
            f"[TT-Plugin] Error registering TT models: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
