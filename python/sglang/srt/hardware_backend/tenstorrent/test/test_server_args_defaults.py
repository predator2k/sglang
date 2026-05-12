"""Unit tests for TTSRTPlatform.apply_server_args_defaults (spec §6.1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sglang.srt.hardware_backend.tenstorrent.platform import TTSRTPlatform


def test_defaults_set_p1_constraints():
    plat = TTSRTPlatform()
    sa = MagicMock(tp_size=None)
    plat.apply_server_args_defaults(sa)
    assert sa.max_running_requests == 1
    assert sa.chunked_prefill_size == -1
    assert sa.disable_radix_cache is True
    assert sa.disable_overlap_schedule is True  # spec §10 risk #12 guard
    assert sa.sampling_backend == "pytorch"
    assert sa.pre_warm_nccl is False
    assert sa.cpu_offload_gb == 0
    assert sa.enable_torch_compile is False
    # G.5 fix: server_args.device must be "cpu" (not "tenstorrent") —
    # torch.get_device_module() in ModelRunner.init_torch_distributed
    # doesn't know about the "tenstorrent" device type. The TT platform
    # identity flows from SGLANG_PLATFORM=tenstorrent independently.
    assert sa.device == "cpu"
    # G.5: grammar must be disabled — bundled xgrammar lacks StructuralTag.
    assert sa.grammar_backend == "none"
    assert sa.tp_size == 1
    assert sa.enable_dp_attention is False


def test_rejects_explicit_tp_2():
    plat = TTSRTPlatform()
    sa = MagicMock(tp_size=2)
    with pytest.raises(ValueError, match="TT backend requires --tp 1"):
        plat.apply_server_args_defaults(sa)


def test_accepts_tp_1():
    plat = TTSRTPlatform()
    sa = MagicMock(tp_size=1)
    plat.apply_server_args_defaults(sa)
    assert sa.tp_size == 1
