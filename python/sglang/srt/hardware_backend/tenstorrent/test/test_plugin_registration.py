# SPDX-License-Identifier: Apache-2.0
"""P2a.1 §9.b6 model-registry namespace test (INV-6 + plugin port verification).

CPU-only — no hardware needed. Imports plugin classes and confirms they're
registered under the Tenstorrent* namespace in SGLang's ModelRegistry.
"""
import pytest


def test_plugin_classes_importable():
    from sglang.srt.hardware_backend.tenstorrent.models import (
        TenstorrentGptOssForCausalLM,
        TenstorrentLlamaForCausalLM,
        TenstorrentMistralForCausalLM,
        TenstorrentQwenForCausalLM,
    )
    for cls in [
        TenstorrentLlamaForCausalLM,
        TenstorrentQwenForCausalLM,
        TenstorrentMistralForCausalLM,
        TenstorrentGptOssForCausalLM,
    ]:
        assert cls.__name__.startswith("Tenstorrent"), f"INV-6 violation: {cls.__name__}"


def test_model_registry_patched():
    # Trigger side-effect registration
    from sglang.srt.hardware_backend.tenstorrent import models  # noqa
    from sglang.srt.models.registry import ModelRegistry

    # Plugin registers under HuggingFace arch names (LlamaForCausalLM, etc.)
    for arch in ["LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM", "GptOssForCausalLM"]:
        assert arch in ModelRegistry.models, f"{arch} not in registry"
        cls = ModelRegistry.models[arch]
        assert cls.__name__.startswith("Tenstorrent"), \
            f"INV-6 violation: {arch} → {cls.__name__}"
