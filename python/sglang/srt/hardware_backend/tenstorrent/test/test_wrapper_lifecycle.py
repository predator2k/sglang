"""Unit tests for TTTransformersExecutionBackend lifecycle methods.

CPU-only — ttnn, create_tt_model, and Generator are all mocked at module
scope so this can run anywhere. Hardware coverage lands in Phase F.4.
"""

import pytest
from unittest.mock import MagicMock, patch

from sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_backend import (
    TTTransformersExecutionBackend,
)

_MODULE = "sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_backend"


def _create_tt_model_4tuple():
    """create_tt_model returns (tt_model_args, model, tt_kv_cache, state_dict).

    __init__ does ``self._model_args, self._model, self._tt_kv_cache, _state_dict
    = create_tt_model(...)`` — a bare MagicMock can't unpack to 4, so every
    test that exercises __init__ must set this return_value explicitly.
    """
    return (MagicMock(), MagicMock(), MagicMock(), MagicMock())


@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
@patch(f"{_MODULE}.ttnn")
def test_lifecycle_roundtrip(_mock_ttnn, _mock_gen, mock_create, monkeypatch):
    # monkeypatch.setenv("LLAMA_DIR", "") forces pytest to snapshot the
    # env-var and restore it after the test — production __init__ writes
    # os.environ["LLAMA_DIR"] = model_path unconditionally and would
    # otherwise leak across tests.
    monkeypatch.setenv("LLAMA_DIR", "")
    mock_create.return_value = _create_tt_model_4tuple()
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        be.new_request("r1", [1, 2, 3])
        assert "r1" in be._req_state
        _ = be.extend("r1")
        assert be._req_state["r1"]["current_offset"] == 3
        for _ in range(3):
            _ = be.decode_step("r1", last_token=42)
        assert be._req_state["r1"]["current_offset"] == 6
        be.free("r1")
        assert "r1" not in be._req_state


@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
@patch(f"{_MODULE}.ttnn")
def test_extend_then_free_no_decode(_mock_ttnn, _mock_gen, mock_create, monkeypatch):
    """Prefill-only path: new_request → extend → free (no decode_step)."""
    monkeypatch.setenv("LLAMA_DIR", "")
    mock_create.return_value = _create_tt_model_4tuple()
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        be.new_request("r1", [1, 2, 3, 4, 5])
        _ = be.extend("r1")
        # After extend, current_offset == prompt_len.
        assert be._req_state["r1"]["current_offset"] == 5
        be.free("r1")
        assert "r1" not in be._req_state


@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
@patch(f"{_MODULE}.ttnn")
def test_reset_all_clears_state(_mock_ttnn, _mock_gen, mock_create, monkeypatch):
    monkeypatch.setenv("LLAMA_DIR", "")
    mock_create.return_value = _create_tt_model_4tuple()
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        be.new_request("r1", [1])
        be.new_request("r2", [2])
        assert set(be._req_state) == {"r1", "r2"}
        be.reset_all()
        assert be._req_state == {}


@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
@patch(f"{_MODULE}.ttnn")
def test_duplicate_req_id_raises(_mock_ttnn, _mock_gen, mock_create, monkeypatch):
    monkeypatch.setenv("LLAMA_DIR", "")
    mock_create.return_value = _create_tt_model_4tuple()
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        be.new_request("r1", [1, 2, 3])
        with pytest.raises(ValueError, match="already active"):
            be.new_request("r1", [4, 5, 6])


@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
@patch(f"{_MODULE}.ttnn")
def test_unknown_req_id_raises_keyerror(_mock_ttnn, _mock_gen, mock_create, monkeypatch):
    monkeypatch.setenv("LLAMA_DIR", "")
    mock_create.return_value = _create_tt_model_4tuple()
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        with pytest.raises(KeyError, match="unknown req_id"):
            be.extend("nope")
        with pytest.raises(KeyError, match="unknown req_id"):
            be.decode_step("nope", last_token=0)


@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
@patch(f"{_MODULE}.ttnn")
def test_free_unknown_is_idempotent(_mock_ttnn, _mock_gen, mock_create, monkeypatch):
    monkeypatch.setenv("LLAMA_DIR", "")
    mock_create.return_value = _create_tt_model_4tuple()
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        # No raise, no crash, no state change.
        be.free("never_existed")
        assert be._req_state == {}


def test_missing_model_path_raises():
    """Path validation must fire before the ttnn-None guard so this fires
    cleanly on a CPU-only host (Round-8 fix verifies this ordering)."""
    with pytest.raises(FileNotFoundError, match="mount"):
        TTTransformersExecutionBackend(
            model_path="/does/not/exist", mesh_device=MagicMock(), max_seq_len=256
        )


@patch(f"{_MODULE}._TT_ACTIVE_REQUESTS")
@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
@patch(f"{_MODULE}.ttnn")
def test_active_requests_gauge_inc_and_dec(
    _mock_ttnn, _mock_gen, mock_create, mock_gauge, monkeypatch
):
    """new_request increments the gauge; free decrements it."""
    monkeypatch.setenv("LLAMA_DIR", "")
    mock_create.return_value = _create_tt_model_4tuple()
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        be.new_request("r1", [1])
        mock_gauge.inc.assert_called_once()
        be.free("r1")
        mock_gauge.dec.assert_called_once()


@patch(f"{_MODULE}._TT_ACTIVE_REQUESTS")
@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
@patch(f"{_MODULE}.ttnn")
def test_reset_all_zeros_gauge(_mock_ttnn, _mock_gen, mock_create, mock_gauge, monkeypatch):
    monkeypatch.setenv("LLAMA_DIR", "")
    mock_create.return_value = _create_tt_model_4tuple()
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        be.new_request("r1", [1])
        be.new_request("r2", [2])
        be.reset_all()
        mock_gauge.set.assert_called_with(0)
