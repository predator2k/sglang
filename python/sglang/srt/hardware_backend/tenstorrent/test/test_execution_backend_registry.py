"""Unit tests for the TT execution backend registry + factory.

Mirrors the SGLang attention_registry test pattern: confirm the two P1
entries are registered, that the auto/env-var resolution path picks the
right class, and that the tt_xla placeholder raises with a pointer to the
post-P1 work.
"""

import pytest
from unittest.mock import patch

from sglang.srt.hardware_backend.tenstorrent.execution import (
    TT_EXECUTION_BACKENDS,
    get_tt_execution_backend,
    resolve_execution_backend_name,
)
from sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_backend import (
    TTTransformersExecutionBackend,
)
from sglang.srt.hardware_backend.tenstorrent.execution.tt_xla_backend import (
    TTXLAExecutionBackend,
)


def test_registry_has_both_entries():
    assert TT_EXECUTION_BACKENDS["tt_transformers_single"] is TTTransformersExecutionBackend
    assert TT_EXECUTION_BACKENDS["tt_xla"] is TTXLAExecutionBackend


def test_auto_resolves_to_tt_transformers_single_in_p2a():
    assert resolve_execution_backend_name("auto") == "tt_transformers_single"
    assert resolve_execution_backend_name("") == "tt_transformers_single"
    assert resolve_execution_backend_name(None) == "tt_transformers_single"


def test_explicit_tt_xla_resolves_to_tt_xla():
    assert resolve_execution_backend_name("tt_xla") == "tt_xla"


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown TT execution backend"):
        get_tt_execution_backend("frobnicate")


def test_tt_xla_construction_raises_with_pointer():
    cls = get_tt_execution_backend("tt_xla")
    with pytest.raises(NotImplementedError, match="P2-coverage"):
        cls(model_path="/tmp", mesh_device=None, max_seq_len=128, max_batch_size=1, token_to_kv_pool=None)


def test_env_var_selects_backend(monkeypatch):
    """The worker's resolve path is `resolve_execution_backend_name()` with
    no args — the env var is the only input. Test that explicitly so a
    regression in `envs.SGLANG_TT_EXECUTION_BACKEND.get()` doesn't slip past.

    EnvField.get() reads os.environ on every call (no lru_cache) — verified
    against environ.py:54. So monkeypatch.setenv is sufficient; no cache
    busting needed.
    """
    monkeypatch.setenv("SGLANG_TT_EXECUTION_BACKEND", "tt_xla")
    assert resolve_execution_backend_name() == "tt_xla"


def test_env_var_unset_defaults_to_tt_transformers_single(monkeypatch):
    monkeypatch.delenv("SGLANG_TT_EXECUTION_BACKEND", raising=False)
    assert resolve_execution_backend_name() == "tt_transformers_single"
