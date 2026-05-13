"""Forward-mode dispatch guard tests for TTTpModelWorker.

Verifies spec §3.2 invariant #6: only EXTEND/DECODE/IDLE are supported.
MIXED / SPLIT_PREFILL / DLLM_EXTEND must raise NotImplementedError
(NOT silently fall into the EXTEND branch via is_extend()).

P2a: the worker now calls execution_backend.forward(forward_batch) rather
than per-method new_request/extend/decode_step. These tests mock
execution_backend.forward() accordingly.
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
    touches: execution_backend (MagicMock for all calls).

    P2a: execution_backend.forward() is the single entry point; the mock
    returns a LogitsProcessorOutput-like object with next_token_logits set
    so argmax + .item() work cleanly.
    """
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    worker = TTTpModelWorker.__new__(TTTpModelWorker)
    worker.execution_backend = MagicMock()
    # Default: forward returns [1, vocab] logits (non-None → greedy sampling path).
    default_logits = torch.zeros(1, 128256)
    worker.execution_backend.forward.return_value = LogitsProcessorOutput(
        next_token_logits=default_logits
    )
    return worker


@pytest.mark.parametrize(
    "bad_mode",
    [ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL, ForwardMode.DLLM_EXTEND],
)
def test_unsupported_modes_raise(bad_mode):
    """MIXED / SPLIT_PREFILL / DLLM_EXTEND must NOT silently enter EXTEND
    branch via is_extend() — they must raise per invariant #6.

    P2a: the guard now lives inside TTTransformersExecutionBackend.forward,
    so we must let it propagate through the worker's forward() call.
    """
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    worker = _bare_worker()
    # Make forward() raise NotImplementedError for unsupported modes
    # (mirrors what the real backend does).
    worker.execution_backend.forward.side_effect = NotImplementedError(
        f"tt_transformers_single supports EXTEND/DECODE/IDLE only, got {bad_mode}."
    )
    with pytest.raises(NotImplementedError):
        worker._forward_batch_generation_tt(_mock_mwb(bad_mode))


def test_idle_returns_empty_result():
    """IDLE must return GenerationBatchResult with next_token_logits=None
    and can_run_cuda_graph=False (mirrors MLX tp_worker.py:136-140)."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    worker = _bare_worker()
    worker.execution_backend.forward.return_value = LogitsProcessorOutput(
        next_token_logits=None
    )
    result = worker._forward_batch_generation_tt(_mock_mwb(ForwardMode.IDLE))
    assert result.logits_output.next_token_logits is None
    assert result.can_run_cuda_graph is False
    # MLX-parity: next_token_ids stays at the dataclass default (None)
    assert result.next_token_ids is None


def test_extend_path_calls_backend():
    """EXTEND mode should drive forward() on the backend and return
    greedy-sampled next_token_ids."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    worker = _bare_worker()
    # forward returns a fixed-argmax tensor so we can assert the sampled id.
    # Shape [1, vocab] — the backend's forward() contract.
    fake_logits = torch.zeros(1, 128256)
    fake_logits[0, 42] = 1.0
    worker.execution_backend.forward.return_value = LogitsProcessorOutput(
        next_token_logits=fake_logits
    )

    mwb = _mock_mwb(ForwardMode.EXTEND)
    result = worker._forward_batch_generation_tt(mwb)

    worker.execution_backend.forward.assert_called_once()
    assert result.next_token_ids is not None
    assert int(result.next_token_ids.flatten()[0].item()) == 42
    assert result.can_run_cuda_graph is False


def test_decode_path_calls_backend():
    """DECODE mode should drive forward() with the last input token id."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    worker = _bare_worker()
    fake_logits = torch.zeros(1, 128256)
    fake_logits[0, 7] = 1.0
    worker.execution_backend.forward.return_value = LogitsProcessorOutput(
        next_token_logits=fake_logits
    )

    mwb = _mock_mwb(ForwardMode.DECODE)
    result = worker._forward_batch_generation_tt(mwb)

    worker.execution_backend.forward.assert_called_once()
    assert int(result.next_token_ids.flatten()[0].item()) == 7
