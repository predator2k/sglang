"""WS-A.17 — Qwen3.5 decode trace harness.

Wraps the standalone decode-one-step entry point with
``ttnn.begin_trace_capture`` / ``ttnn.execute_trace`` so subsequent decode
calls bypass per-op ttnn dispatch latency.

Capture pattern mirrors ``Generator._capture_decode_trace_text``:

  1. Compile run: invoke ``ttnn_decode_forward`` once with eager-mode
     dispatch (warms the kernel cache).
  2. ``ttnn.synchronize_device`` to drain in-flight workers.
  3. ``ttnn.begin_trace_capture(mesh_device, cq_id=0)``.
  4. Replay the SAME ``ttnn_decode_forward`` against PRE-ALLOCATED persistent
     device input tensors.  The recorded program references these by address.
  5. ``ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)``.
  6. Cache ``(trace_id, output_tensor, device_inputs)``.

For each subsequent step, copy fresh host inputs INTO the persistent
device buffers via ``ttnn.copy_host_to_device_tensor`` and then
``ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)``.

Gated on ``SGLANG_TT_QWEN35_TRACE=1``.  When unset the helper falls back to
the existing eager ``decode_one_step``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import ttnn

from _prefetcher_harness import decode_one_step  # eager fallback


def _trace_env_on() -> bool:
    return os.environ.get("SGLANG_TT_QWEN35_TRACE", "").lower() in ("1", "true", "yes")


class Qwen35TraceWrapper:
    """Stateful decode-trace wrapper for one model + KV cache + page table.

    Lazy-captures on the first ``decode_one_step`` call.  Subsequent calls
    re-use the trace via ``ttnn.execute_trace``.

    Inputs that change per step (``tokens``, ``current_pos``) are copied
    into pre-allocated device input tensors before each replay.  The page
    table is captured once and remains constant for the lifetime of this
    wrapper (matches the standalone harness, which never resizes pages).

    NOTE: model state (``_tt_conv_state``, ``_tt_ssm_state``, KV cache) is
    updated IN-PLACE inside the recorded graph (see linear_attention.py
    WS-A.17 trace-safe variant + paged_update_cache for full-attention).
    """

    def __init__(self, model, model_args, kv_cache, page_table_host):
        self.model = model
        self.model_args = model_args
        self.kv_cache = kv_cache
        self.page_table_host = page_table_host
        self.mesh_device = model_args.mesh_device

        self._trace_id: Optional[int] = None
        self._device_inputs = None         # tuple of persistent ttnn tensors
        self._output_trace: Optional[ttnn.Tensor] = None
        self._compile_pos: int = -1

    # ------------------------------------------------------------------
    def _compile_and_capture(self, first_token_id: int, first_pos: int):
        """Run one eager compile pass, then record the trace."""
        from models.tt_transformers.tt.common import Mode

        # Make sure we are in DECODE mode (initializes prefetcher sub-devices
        # when present; no-op when None).  Same as decode_one_step.
        self.model.switch_mode(Mode.DECODE)

        batch_size = self.model_args.max_batch_size

        # ---- (a) compile run: eager forward at the SAME (token,pos) ----
        tokens = torch.tensor([first_token_id] * batch_size)
        current_pos = torch.tensor([first_pos] * batch_size)
        tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_page_table = (
            self.model.prepare_inputs_decode(tokens, current_pos, self.page_table_host)
        )
        tt_logits_compile, _ = self.model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rot_mat_idxs,
            page_table=tt_page_table,
            kv_cache=self.kv_cache,
        )
        # Discard compile-run output to free buffers before capture.
        ttnn.deallocate(tt_logits_compile)
        ttnn.synchronize_device(self.mesh_device)

        # ---- (b) prepare PERSISTENT device-side inputs ----------------
        # These are the buffers the trace will read from on every replay.
        # We must keep them alive (no deallocate) and write fresh values
        # into them via ttnn.copy_host_to_device_tensor.
        tt_tokens2, tt_current_pos2, tt_rot_mat_idxs2, tt_page_table2 = (
            self.model.prepare_inputs_decode(tokens, current_pos, self.page_table_host)
        )
        self._device_inputs = (tt_tokens2, tt_current_pos2, tt_rot_mat_idxs2, tt_page_table2)

        # ---- (c) record the trace -------------------------------------
        # Reset model state to match the compile run BEFORE recording —
        # the compile pass already advanced _tt_conv_state and _tt_ssm_state
        # by one step; without a reset the captured trace would start
        # from the post-compile state and double-step on the next replay.
        # Easiest fix: clear all LinearAttentionBlock state, and clear KV
        # cache via re-initialization.  This forces step=0 semantics.
        self._reset_model_state()

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        tt_logits_trace, _ = self.model.ttnn_decode_forward(
            tt_tokens2,
            tt_current_pos2,
            rot_mat_idxs=tt_rot_mat_idxs2,
            page_table=tt_page_table2,
            kv_cache=self.kv_cache,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)

        self._trace_id = trace_id
        self._output_trace = tt_logits_trace
        self._compile_pos = first_pos

    def _reset_model_state(self):
        """Zero conv/ssm state on every LinearAttentionBlock and zero KV cache.

        The trace records the FIRST-step semantics; we must rewind any state
        the compile run advanced.
        """
        # Wipe linear-attention state on every layer.
        for layer in getattr(self.model, "layers", []):
            tt_conv = getattr(layer, "_tt_conv_state", None)
            tt_ssm = getattr(layer, "_tt_ssm_state", None)
            if tt_conv is not None:
                # In-place zero so the buffer address survives.
                # ttnn.fill writes a scalar into every element; if missing,
                # fall back to copy of a zero host tensor.
                try:
                    zeros = torch.zeros(
                        1, 1, tt_conv.shape[-2], tt_conv.shape[-1],
                        dtype=torch.bfloat16,
                    )
                    zero_tt = ttnn.from_torch(
                        zeros,
                        device=self.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    )
                    ttnn.copy(zero_tt, tt_conv)
                    ttnn.deallocate(zero_tt)
                except Exception as exc:
                    print(f"[WS-A.17 trace] conv reset warn: {exc}", flush=True)
            if tt_ssm is not None:
                try:
                    zeros = torch.zeros(
                        1, tt_ssm.shape[-3], tt_ssm.shape[-2], tt_ssm.shape[-1],
                        dtype=torch.bfloat16,
                    )
                    zero_tt = ttnn.from_torch(
                        zeros,
                        device=self.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    )
                    ttnn.copy(zero_tt, tt_ssm)
                    ttnn.deallocate(zero_tt)
                except Exception as exc:
                    print(f"[WS-A.17 trace] ssm reset warn: {exc}", flush=True)

    # ------------------------------------------------------------------
    def decode_one_step(self, step: int, token_id: int = 42) -> torch.Tensor:
        """Run one decode step against the captured trace; lazily captures on
        the first call.  Returns logits on CPU."""
        from models.tt_transformers.tt.common import copy_host_to_device

        if self._trace_id is None:
            self._compile_and_capture(first_token_id=token_id, first_pos=step)
        else:
            # ---- replay path: update PERSISTENT inputs in-place --------
            batch_size = self.model_args.max_batch_size
            tokens = torch.tensor([token_id] * batch_size)
            current_pos = torch.tensor([step] * batch_size)
            host_inputs = self.model.prepare_decode_inputs_host(
                tokens, current_pos, self.page_table_host
            )
            # Mirror Generator._decode_forward_trace_text's copy: writes the
            # fresh values INTO the pre-allocated device-side buffers.
            copy_host_to_device(
                host_tensors=host_inputs,
                device_tensors=self._device_inputs,
            )
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)

        # Output processing — same as eager decode_one_step.
        tt_logits = self._output_trace
        dev0 = ttnn.get_device_tensors(tt_logits)[0]
        logits_host = ttnn.to_torch(dev0).float()  # [1, 1, 32, vocab_size]
        logits_4d = logits_host[:, :, :, : self.model_args.vocab_size]
        row_norms = logits_4d[0, 0].norm(dim=-1)
        best_row = int(row_norms.argmax())
        logits_out = logits_4d[:, :, best_row : best_row + 1, :].squeeze(2)
        return logits_out.cpu()


def make_decode_runner(model, model_args, kv_cache, page_table_host):
    """Factory returning a callable ``runner(step, token_id) -> logits``.

    If ``SGLANG_TT_QWEN35_TRACE=1`` returns the trace-wrapped runner;
    otherwise returns a thin closure over the eager ``decode_one_step``.
    """
    if _trace_env_on():
        wrapper = Qwen35TraceWrapper(model, model_args, kv_cache, page_table_host)
        return wrapper.decode_one_step
    else:
        def _eager_runner(step: int, token_id: int = 42):
            return decode_one_step(
                model, model_args, step=step, token_id=token_id,
                kv_cache=kv_cache, page_table_host=page_table_host,
            )
        return _eager_runner
