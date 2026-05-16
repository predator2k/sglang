"""Unit tests for the TT execution backend registry + factory.

Mirrors the SGLang attention_registry test pattern: confirm the
tt_transformers entries are registered, that the auto/env-var resolution
path picks the right name, and that tt_xla resolves correctly as a
name (it uses Pattern A / ModelRegistry, not the execution backend registry).
"""

import pytest

from sglang.srt.hardware_backend.tenstorrent.execution import (
    TT_EXECUTION_BACKENDS,
    get_tt_execution_backend,
    resolve_execution_backend_name,
)
from sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_backend import (
    TTTransformersExecutionBackend,
)


def test_registry_has_tt_transformers():
    assert TT_EXECUTION_BACKENDS["tt_transformers_single"] is TTTransformersExecutionBackend


def test_auto_resolves_to_tt_transformers_paged():
    assert resolve_execution_backend_name("auto") == "tt_transformers_paged"
    assert resolve_execution_backend_name("") == "tt_transformers_paged"
    assert resolve_execution_backend_name(None) == "tt_transformers_paged"


def test_explicit_tt_xla_resolves_to_tt_xla():
    assert resolve_execution_backend_name("tt_xla") == "tt_xla"


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown TT execution backend"):
        get_tt_execution_backend("frobnicate")


def test_tt_xla_not_in_execution_registry():
    """tt_xla uses Pattern A (ModelRegistry), not the execution backend registry."""
    assert "tt_xla" not in TT_EXECUTION_BACKENDS


def test_env_var_selects_backend(monkeypatch):
    """The worker's resolve path is `resolve_execution_backend_name()` with
    no args -- the env var is the only input.
    """
    monkeypatch.setenv("SGLANG_TT_EXECUTION_BACKEND", "tt_xla")
    assert resolve_execution_backend_name() == "tt_xla"


def test_env_var_unset_defaults_to_tt_transformers_paged(monkeypatch):
    monkeypatch.delenv("SGLANG_TT_EXECUTION_BACKEND", raising=False)
    assert resolve_execution_backend_name() == "tt_transformers_paged"
