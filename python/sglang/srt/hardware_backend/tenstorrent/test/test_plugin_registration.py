# SPDX-License-Identifier: Apache-2.0
"""P2a.1 §9.b6 model-registry namespace test (INV-6 + plugin port verification).

CPU-only — no hardware needed. Imports plugin classes and confirms they're
registered under the Tenstorrent* namespace in SGLang's ModelRegistry.
"""
import sys

import pytest

# Keys that test_chunked_failure_recovery.py inserts as lightweight stubs at
# collection time. When both test files are collected in the same pytest process,
# those stubs shadow the real package and cause ImportError here.  Evicting them
# before each test in this file lets the real importer re-resolve the packages
# from disk.
_CHUNKED_RECOVERY_STUBS = [
    "sglang.srt.hardware_backend.tenstorrent.models",
    "sglang.srt.hardware_backend.tenstorrent.models.tt_llm",
    "sglang.srt.hardware_backend.tenstorrent.models.tt_utils",
    "sglang.srt.hardware_backend.tenstorrent.models.worker_setup",
    "sglang.srt.hardware_backend.tenstorrent",
    "sglang.srt.hardware_backend",
    "sglang.srt",
    "sglang",
    "sglang.srt.server_args",
    "sglang.srt.layers",
    "sglang.srt.layers.logits_processor",
]


@pytest.fixture(autouse=True)
def _evict_stubs():
    """Remove lightweight stub modules that test_chunked_failure_recovery installs.

    Those stubs are inserted at collection time (module-level code runs during
    pytest collection) and have ``__spec__ = None`` (the fingerprint of a bare
    ``types.ModuleType`` stub).  Evicting them before each test lets Python
    re-import the real packages from disk.

    Real sglang modules always have a proper ``ModuleSpec``, so this eviction
    is safe — it only removes fakes.
    """
    for key in _CHUNKED_RECOVERY_STUBS:
        mod = sys.modules.get(key)
        if mod is not None and getattr(mod, "__spec__", "sentinel") is None:
            sys.modules.pop(key, None)
    yield


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
