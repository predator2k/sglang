# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>

"""§9.12.a-e chunked-prefill failure recovery unit tests.

CPU-only — no hardware or ttnn required.  All TT-backend heavy imports are
mocked so this suite runs on any host that has a standard SGLang checkout.

NOTE: §9.12.e in the plan also specifies that the scheduler clears
``self.chunked_req`` after a failure.  That wiring is exercised by end-to-end
testing (T2.1).  Here we only verify the **model-side** sentinel
(``_last_chunked_failure``); the scheduler-side path is tested separately.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import pytest


# ---------------------------------------------------------------------------
# Module-level stubs for heavy optional dependencies.
#
# We inject thin fake modules into sys.modules BEFORE importing tt_llm so
# that the module-level ``from .tt_utils import BaseMetalDeviceRunner`` and
# other optional imports resolve without errors on CPU-only hosts.
# ---------------------------------------------------------------------------


def _install_stubs():
    """Insert minimal stubs for modules that tt_llm imports at module level."""

    def _make_pkg(*parts):
        """Create nested fake package hierarchy a.b.c … and return the leaf."""
        parent = None
        for i, name in enumerate(parts):
            full = ".".join(parts[: i + 1])
            if full not in sys.modules:
                mod = types.ModuleType(full)
                mod.__path__ = []  # mark as package
                sys.modules[full] = mod
                if parent is not None:
                    setattr(parent, name, mod)
            parent = sys.modules[full]
        return parent

    # tt_utils stub (provides BaseMetalDeviceRunner)
    tt_utils_path = (
        "sglang.srt.hardware_backend.tenstorrent.models.tt_utils"
    )
    if tt_utils_path not in sys.modules:
        stub = types.ModuleType(tt_utils_path)
        stub.BaseMetalDeviceRunner = MagicMock()
        sys.modules[tt_utils_path] = stub

    # worker_setup stub
    ws_path = (
        "sglang.srt.hardware_backend.tenstorrent.models.worker_setup"
    )
    if ws_path not in sys.modules:
        stub = types.ModuleType(ws_path)
        stub.setup_worker_from_process_title = MagicMock()
        sys.modules[ws_path] = stub

    # sglang.srt.server_args stub (get_global_server_args)
    _make_pkg("sglang", "srt", "server_args")
    sa_mod = sys.modules["sglang.srt.server_args"]
    if not hasattr(sa_mod, "get_global_server_args"):
        mock_args = MagicMock()
        mock_args.page_size = 64
        mock_args.context_length = 4096
        mock_args.max_running_requests = 32
        mock_args.model_path = ""
        sa_mod.get_global_server_args = MagicMock(return_value=mock_args)

    # sglang.srt.layers.logits_processor stub
    _make_pkg("sglang", "srt", "layers", "logits_processor")
    lp_mod = sys.modules["sglang.srt.layers.logits_processor"]
    if not hasattr(lp_mod, "LogitsProcessorOutput"):
        class _LogitsProcessorOutput:
            def __init__(self, next_token_logits=None, **kw):
                self.next_token_logits = next_token_logits
        lp_mod.LogitsProcessorOutput = _LogitsProcessorOutput

    # sglang.srt.hardware_backend.tenstorrent.models (package)
    _make_pkg(
        "sglang",
        "srt",
        "hardware_backend",
        "tenstorrent",
        "models",
    )


_install_stubs()

# Now we can safely import TTModels from tt_llm via importlib.
# Because tt_llm.py uses relative imports (from .tt_utils import …) we must
# register it as part of the tenstorrent.models package before exec_module.
import importlib.util
import pathlib

_TT_LLM_PATH = (
    pathlib.Path(__file__).parent.parent / "models" / "tt_llm.py"
)

_PKG_NAME = "sglang.srt.hardware_backend.tenstorrent.models"
_MOD_NAME = f"{_PKG_NAME}.tt_llm"


def _load_tt_llm():
    spec = importlib.util.spec_from_file_location(
        _MOD_NAME,
        str(_TT_LLM_PATH),
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG_NAME
    # Register before exec so relative imports inside tt_llm resolve to our stubs
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_tt_llm_mod = _load_tt_llm()
TTModels = _tt_llm_mod.TTModels


# ---------------------------------------------------------------------------
# Helpers to build a minimal TTModels instance without running __init__
# ---------------------------------------------------------------------------


def _make_instance():
    """Return a bare TTModels instance with __init__ bypassed."""
    obj = object.__new__(TTModels)
    # Attributes that on_chunked_prefill_failure and forward need:
    obj._last_chunked_failure = False
    return obj


def _make_req(*, prefix_len: int = 4, fill_len: int = 6, pool_idx: int = 2):
    """Build a minimal mock Req."""
    req = MagicMock()
    req.req_pool_idx = pool_idx
    req.prefix_indices = list(range(prefix_len))
    req.fill_ids = list(range(fill_len))
    req.skip_radix_cache_insert = False
    return req


def _make_forward_batch(*, req, chunk_len: int = 3, is_extend: bool = True):
    """Build a minimal mock ForwardBatch for a chunked-prefill scenario."""
    fb = MagicMock()

    # forward_mode
    fb.forward_mode.is_extend.return_value = is_extend
    fb.forward_mode.is_decode.return_value = not is_extend

    # chunked_req — non-None signals a live chunked-prefill
    fb.chunked_req = req

    # reqs[0] is the chunked request (B=1 assumption)
    fb.reqs = [req]

    # out_cache_loc — the slots written for this chunk
    fb.out_cache_loc = torch.tensor(list(range(chunk_len)), dtype=torch.int64)

    # req_to_token_pool with a real tensor for the slice-write assertion
    pool = MagicMock()
    pool.req_to_token = torch.zeros(
        (8, req.req_pool_idx + 16), dtype=torch.int64
    )
    # Pre-fill the chunk range with non-zero values so we can assert zeroing
    start = len(req.prefix_indices) + len(req.fill_ids) - chunk_len
    end = len(req.prefix_indices) + len(req.fill_ids)
    pool.req_to_token[req.req_pool_idx, start:end] = 999
    fb.req_to_token_pool = pool

    # token_to_kv_pool_allocator with a capturable free()
    fb.token_to_kv_pool_allocator = MagicMock()

    return fb


# ---------------------------------------------------------------------------
# §9.12.a – allocator.free() called with exactly the failed-chunk slot list
# ---------------------------------------------------------------------------


def test_9_12_a_allocator_free_called_with_exact_slot_list():
    """§9.12.a: on_chunked_prefill_failure must call allocator.free(out_cache_loc_this_chunk)."""
    chunk_len = 3
    req = _make_req()
    fb = _make_forward_batch(req=req, chunk_len=chunk_len)

    inst = _make_instance()
    inst.on_chunked_prefill_failure(
        req=req,
        out_cache_loc_this_chunk=fb.out_cache_loc,
        allocator=fb.token_to_kv_pool_allocator,
        req_to_token_pool=fb.req_to_token_pool,
    )

    fb.token_to_kv_pool_allocator.free.assert_called_once()
    call_arg = fb.token_to_kv_pool_allocator.free.call_args[0][0]
    assert torch.equal(call_arg, fb.out_cache_loc), (
        f"allocator.free() received {call_arg}, expected {fb.out_cache_loc}"
    )


# ---------------------------------------------------------------------------
# §9.12.b – req_to_token entries in the failed-chunk range are zeroed
# ---------------------------------------------------------------------------


def test_9_12_b_req_to_token_zeroed_in_failed_chunk_range():
    """§9.12.b: req_to_token[pool_idx, start:end] must be zeroed after failure."""
    chunk_len = 3
    req = _make_req(prefix_len=4, fill_len=6, pool_idx=2)
    fb = _make_forward_batch(req=req, chunk_len=chunk_len)

    # Pre-assert: range should be non-zero (set to 999 in helper)
    pool_idx = req.req_pool_idx
    start = len(req.prefix_indices) + len(req.fill_ids) - chunk_len  # 4+6-3 = 7
    end = len(req.prefix_indices) + len(req.fill_ids)                # 4+6   = 10
    assert fb.req_to_token_pool.req_to_token[pool_idx, start:end].sum() != 0, (
        "Pre-condition failed: expected non-zero values in the chunk range"
    )

    inst = _make_instance()
    inst.on_chunked_prefill_failure(
        req=req,
        out_cache_loc_this_chunk=fb.out_cache_loc,
        allocator=fb.token_to_kv_pool_allocator,
        req_to_token_pool=fb.req_to_token_pool,
    )

    zeroed_slice = fb.req_to_token_pool.req_to_token[pool_idx, start:end]
    assert zeroed_slice.sum().item() == 0, (
        f"Expected req_to_token[{pool_idx}, {start}:{end}] all-zero, got {zeroed_slice}"
    )


# ---------------------------------------------------------------------------
# §9.12.c – skip_radix_cache_insert set to True
# ---------------------------------------------------------------------------


def test_9_12_c_skip_radix_cache_insert_set_true():
    """§9.12.c: req.skip_radix_cache_insert must be True after failure."""
    req = _make_req()
    fb = _make_forward_batch(req=req)

    assert not req.skip_radix_cache_insert, "Pre-condition: must start False"

    inst = _make_instance()
    inst.on_chunked_prefill_failure(
        req=req,
        out_cache_loc_this_chunk=fb.out_cache_loc,
        allocator=fb.token_to_kv_pool_allocator,
        req_to_token_pool=fb.req_to_token_pool,
    )

    assert req.skip_radix_cache_insert is True, (
        "req.skip_radix_cache_insert must be True after chunked-prefill failure"
    )


# ---------------------------------------------------------------------------
# §9.12.d – set_finish_with_abort called with the expected reason string
# ---------------------------------------------------------------------------


def test_9_12_d_finished_reason_is_finish_abort():
    """§9.12.d: req.set_finish_with_abort must be called with 'tt_backend_chunked_prefill_failure'."""
    req = _make_req()
    fb = _make_forward_batch(req=req)

    inst = _make_instance()
    inst.on_chunked_prefill_failure(
        req=req,
        out_cache_loc_this_chunk=fb.out_cache_loc,
        allocator=fb.token_to_kv_pool_allocator,
        req_to_token_pool=fb.req_to_token_pool,
    )

    req.set_finish_with_abort.assert_called_once_with("tt_backend_chunked_prefill_failure")


# ---------------------------------------------------------------------------
# §9.12.e – _last_chunked_failure sentinel set on the model instance
# ---------------------------------------------------------------------------


def test_9_12_e_last_chunked_failure_sentinel_set():
    """§9.12.e: after forward() catches a chunked-prefill exception,
    self._last_chunked_failure must be True.

    NOTE: scheduler-side clearing of self.chunked_req is tested through
    end-to-end wiring (T2.1).  This test verifies only the model sentinel.
    """
    inst = _make_instance()
    assert not inst._last_chunked_failure, "Pre-condition: sentinel starts False"

    chunk_len = 3
    req = _make_req()
    fb = _make_forward_batch(req=req, chunk_len=chunk_len, is_extend=True)

    # Patch out the internal helpers that forward() calls so they raise immediately,
    # simulating a TT-backend failure mid-extend.
    with patch.object(
        TTModels, "_build_page_table", side_effect=RuntimeError("tt backend boom")
    ):
        result = inst.forward(
            input_ids=torch.zeros(chunk_len, dtype=torch.long),
            positions=torch.zeros(chunk_len, dtype=torch.long),
            forward_batch=fb,
        )

    assert inst._last_chunked_failure is True, (
        "_last_chunked_failure must be True after a chunked-prefill forward() failure"
    )
    # Verify the handler also ran: allocator.free must have been called
    fb.token_to_kv_pool_allocator.free.assert_called_once()
    # Return value should be a LogitsProcessorOutput with next_token_logits=None
    assert result.next_token_logits is None, (
        "forward() must return LogitsProcessorOutput(next_token_logits=None) on failure"
    )
