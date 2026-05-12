"""Forward-mode dispatch guard tests for TTTpModelWorker.

Verifies spec §3.2 invariant #6: only EXTEND/DECODE/IDLE are supported.
MIXED / SPLIT_PREFILL / DLLM_EXTEND must raise NotImplementedError
(NOT silently fall into the EXTEND branch via is_extend()).
"""
import pytest
import torch
from unittest.mock import MagicMock

from sglang.srt.hardware_backend.tenstorrent.tp_worker import TTTpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _mock_mwb(mode: ForwardMode):
    mwb = MagicMock()
    mwb.forward_mode = mode
    mwb.reqs = [MagicMock(rid="r1")]
    # input_ids must have shape and tolist() / [-1].item() semantics
    mwb.input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    return mwb


def _bare_worker():
    """Construct a TTTpModelWorker without running __init__ (which would
    open the mesh). Backs the minimal attrs _forward_batch_generation_tt
    touches: execution_backend (MagicMock for all calls) and _tt_active_rids
    (so _cleanup_stale_rids' lazy-init path doesn't matter for the guard test).
    """
    worker = TTTpModelWorker.__new__(TTTpModelWorker)
    worker.execution_backend = MagicMock()
    # Mocked extend / decode_step return torch tensors (host-side logits)
    # so argmax + .item() work cleanly.
    worker.execution_backend.extend.return_value = torch.zeros(128256)
    worker.execution_backend.decode_step.return_value = torch.zeros(128256)
    return worker


@pytest.mark.parametrize(
    "bad_mode",
    [ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL, ForwardMode.DLLM_EXTEND],
)
def test_unsupported_modes_raise(bad_mode):
    """MIXED / SPLIT_PREFILL / DLLM_EXTEND must NOT silently enter EXTEND
    branch via is_extend() — they must raise per invariant #6."""
    worker = _bare_worker()
    with pytest.raises(NotImplementedError, match="P1 supports EXTEND/DECODE/IDLE only"):
        worker._forward_batch_generation_tt(_mock_mwb(bad_mode))


def test_idle_returns_empty_result():
    """IDLE must return GenerationBatchResult with next_token_logits=None
    and can_run_cuda_graph=False (mirrors MLX tp_worker.py:136-140)."""
    worker = _bare_worker()
    result = worker._forward_batch_generation_tt(_mock_mwb(ForwardMode.IDLE))
    assert result.logits_output.next_token_logits is None
    assert result.can_run_cuda_graph is False
    # MLX-parity: next_token_ids stays at the dataclass default (None)
    assert result.next_token_ids is None


def test_extend_path_calls_backend():
    """EXTEND mode should drive new_request + extend on the backend and
    return greedy-sampled next_token_ids."""
    worker = _bare_worker()
    # extend returns a fixed-argmax tensor so we can assert the sampled id
    fake_logits = torch.zeros(128256)
    fake_logits[42] = 1.0
    worker.execution_backend.extend.return_value = fake_logits

    mwb = _mock_mwb(ForwardMode.EXTEND)
    result = worker._forward_batch_generation_tt(mwb)

    worker.execution_backend.new_request.assert_called_once()
    worker.execution_backend.extend.assert_called_once_with("r1")
    assert result.next_token_ids is not None
    assert int(result.next_token_ids.flatten()[0].item()) == 42
    assert result.can_run_cuda_graph is False


def test_decode_path_calls_backend():
    """DECODE mode should drive decode_step with the last input token id."""
    worker = _bare_worker()
    fake_logits = torch.zeros(128256)
    fake_logits[7] = 1.0
    worker.execution_backend.decode_step.return_value = fake_logits

    mwb = _mock_mwb(ForwardMode.DECODE)
    # input_ids[-1].item() should be 3 → last_tok=3
    result = worker._forward_batch_generation_tt(mwb)

    worker.execution_backend.decode_step.assert_called_once_with("r1", 3)
    assert int(result.next_token_ids.flatten()[0].item()) == 7
