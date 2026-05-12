"""TTTransformersExecutionBackend — thin adapter between SGLang's worker and tt_transformers.

Captured tt_transformers API surface (Phase 0.2 evidence) the wrapper drives:

    create_tt_model(mesh_device, instruct, max_batch_size, optimizations,
                    max_seq_len, paged_attention_config=None,
                    dtype=ttnn.bfloat8_b, state_dict=None, num_layers=None)
        -> (tt_model_args, model, tt_kv_cache, state_dict)

    Generator(model_list, model_args_list, mesh_device, processor=None,
              tokenizer=None)
        .prefill_forward_text(tokens, ...)             # full-batch prefill
        .prefill_forward_single_user_text(tokens, page_table, user_id,
                                          last_token_idx, kv_cache=None,
                                          model_id=-1, **kwargs)
                                                       # P1's batch=1 path
        .decode_forward_text(tokens, start_pos, page_table=None,
                             kv_cache=None, enable_trace=True,
                             read_from_device=True, sampling_params=None)
        .read_decode_output(tt_out, async_read=False)
        .process_decode_output_host(tt_out, is_tokens=False)

    LlamaForCausalLM (Generator subclass):
        .allocate_kv_cache(*args, **kwargs)            # vllm-only path

The model picks weights from the LLAMA_DIR env var inside ModelArgs.__init__
— our wrapper sets LLAMA_DIR from the constructor's model_path argument
before calling create_tt_model so the tt_transformers code finds the right
checkpoint without leaking docker-mount semantics into upper layers.

F.2 (this revision) fills in the 5 lifecycle methods. Integration smoke
on real hardware lands in F.4.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tenstorrent.execution import (
    register_tt_execution_backend,
)
from sglang.srt.hardware_backend.tenstorrent.execution.base import (
    TTExecutionBackend,
)

# In llama_adapter.py these lived inside __init__ (deferred for
# non-TT-host importability). For F.3 mocking we need them on the
# module namespace, so hoist them but keep the non-TT-host
# importability via try/except + None sentinels.
try:
    import ttnn
    from models.tt_transformers.tt.common import create_tt_model
    from models.tt_transformers.tt.generator import Generator
    from models.tt_transformers.tt.model_config import DecodersPrecision
except ImportError:
    ttnn = None
    create_tt_model = None
    Generator = None
    DecodersPrecision = None

# Prometheus instruments for spec §6.2 performance counters. Module-scope so
# F.3 tests can mock them flat (e.g. `@patch(f"{_MODULE}._TT_EXTEND_LATENCY_MS")`).
# Fall back to a no-op shim if prometheus_client is unavailable (e.g. CI smoke
# environments) so import never fails.
try:
    from prometheus_client import Gauge, Histogram
except ImportError:
    class _NoOpMetric:
        def labels(self, *a, **kw):
            return self

        def observe(self, *a, **kw):
            pass

        def inc(self, *a, **kw):
            pass

        def dec(self, *a, **kw):
            pass

        def set(self, *a, **kw):
            pass

    def Histogram(*a, **kw):  # type: ignore[no-redef]
        return _NoOpMetric()

    def Gauge(*a, **kw):  # type: ignore[no-redef]
        return _NoOpMetric()


_TT_EXTEND_LATENCY_MS = Histogram(
    "tt_extend_latency_ms",
    "Per-request extend (prefill) latency on the TT backend, in milliseconds.",
)
_TT_DECODE_LATENCY_MS = Histogram(
    "tt_decode_latency_ms",
    "Per-step decode latency on the TT backend, in milliseconds.",
)
_TT_D2H_LATENCY_MS = Histogram(
    "tt_d2h_latency_ms",
    "Per-call device-to-host transfer latency on the TT backend, in milliseconds.",
)
_TT_ACTIVE_REQUESTS = Gauge(
    "tt_active_requests",
    "Currently-active request count on the TT backend (0 or 1 in P1).",
)


logger = logging.getLogger("sglang.srt.hardware_backend.tenstorrent")


def _suggestion_for_not_implemented(detail: str) -> str:
    """Re-raise NotImplementedError from tt_transformers with a hint.

    The hint is the docker-image-commit verification message from spec §7.1
    row 4 ("Generator surface drift" recovery). Surfaces a clear next step
    rather than re-raising opaque NotImplementedError.
    """
    return (
        f"tt_transformers raised NotImplementedError: {detail}. "
        "Verify the tt-metal docker commit matches the one captured in "
        "phase0-evidence.txt; this surface may have drifted in a newer image."
    )


@register_tt_execution_backend("tt_transformers")
class TTTransformersExecutionBackend(TTExecutionBackend):
    """Black-box wrapper around a tt_transformers Generator for SGLang.

    P1 contract: caller (TTTpModelWorker) drives the wrapper through
    `new_request → extend → decode_step → free` while `MeshDeviceCtx` owns
    the underlying mesh device lifecycle. The wrapper holds no mesh state
    itself — it borrows the mesh_device from the worker and only manages
    per-request KV handles in `self._req_state`.
    """

    def __init__(
        self,
        model_path: str,
        mesh_device,
        *,
        max_seq_len: int,
        dtype: str = "bf16",
        max_batch_size: int = 1,
        instruct: bool = True,
    ):
        if not os.path.isdir(model_path):
            # F.3 asserts match="mount" on this message — keep the keyword.
            raise FileNotFoundError(
                f"model_path {model_path!r} not found; mount with "
                f"-v $HOME/tt-models:/models when running docker"
            )

        # Option A guard: gate every downstream tt_transformers call behind
        # a single check so non-TT-host imports stay safe but construction
        # fails fast with a precise remediation pointer.
        if create_tt_model is None or ttnn is None:
            raise RuntimeError(
                "tt_transformers / ttnn not importable; install tt-metal or "
                "activate its venv (source /home/container_app_user/tt-metal/"
                "python_env/bin/activate)"
            )

        # tt_transformers' ModelArgs reads LLAMA_DIR / HF_MODEL from env
        # and asserts exactly one is set (model_config.py:499). The
        # tt-inference-server docker image pre-sets BOTH, so clear HF_MODEL
        # before writing LLAMA_DIR or the assert fires.
        os.environ.pop("HF_MODEL", None)
        os.environ["LLAMA_DIR"] = model_path

        _DTYPE_MAP = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"dtype must be one of {sorted(_DTYPE_MAP)}; got {dtype!r}"
            )
        tt_dtype = _DTYPE_MAP[dtype]

        # `optimizations` is a callable(model_args) → DecodersPrecision —
        # the demo's "performance" preset, identical to phase 0.1's
        # passing run on this hardware.
        def _opt_factory(model_args):
            return DecodersPrecision.performance(
                model_args.n_layers, model_args.model_name
            )

        try:
            self._model_args, self._model, self._tt_kv_cache, _state_dict = (
                create_tt_model(
                    mesh_device=mesh_device,
                    instruct=instruct,
                    max_batch_size=max_batch_size,
                    optimizations=_opt_factory,
                    max_seq_len=max_seq_len,
                    paged_attention_config=None,  # P1: tt_transformers default KV
                    dtype=tt_dtype,
                )
            )
        except NotImplementedError as e:
            raise NotImplementedError(_suggestion_for_not_implemented(str(e))) from e

        # Generator wraps the model + tokenizer for the prefill/decode API.
        # data_parallel=1 → singleton lists for model and model_args.
        try:
            self._generator = Generator(
                [self._model],
                [self._model_args],
                mesh_device,
                processor=None,
                tokenizer=self._model_args.tokenizer,
            )
        except NotImplementedError as e:
            raise NotImplementedError(_suggestion_for_not_implemented(str(e))) from e

        self._mesh_device = mesh_device
        self._max_seq_len = max_seq_len
        self._dtype = dtype
        self._req_state: dict[str, Any] = {}

        # SGLANG_TT_PREFILL_PAD_STEP — empty string = "use default 128".
        # The tt_transformers prefill kernel hard-asserts seq_len % 128 == 0,
        # so the step MUST be a positive multiple of 128. Higher values
        # (e.g. 256, 512) trade memory for fewer recompiles when prompt-
        # length distributions cluster.
        pad_step_raw = envs.SGLANG_TT_PREFILL_PAD_STEP.get() or "128"
        try:
            self._prefill_pad_step = int(pad_step_raw)
        except ValueError:
            raise ValueError(
                f"SGLANG_TT_PREFILL_PAD_STEP must be an integer, got {pad_step_raw!r}"
            )
        if self._prefill_pad_step <= 0 or self._prefill_pad_step % 128 != 0:
            raise ValueError(
                f"SGLANG_TT_PREFILL_PAD_STEP must be a positive multiple of 128 "
                f"(tt_transformers' hard kernel constraint); got {self._prefill_pad_step}"
            )

        logger.info(
            "tt_execution_backend_init",
            extra={
                "backend": "tt_transformers",
                "model_path": model_path,
                "max_seq_len": max_seq_len,
                "dtype": dtype,
                "max_batch_size": max_batch_size,
                "n_layers": getattr(self._model_args, "n_layers", None),
                "model_name": getattr(self._model_args, "model_name", None),
            },
        )

    # ----- Per-request lifecycle (Phase F.2) -----

    def new_request(self, req_id: str, prompt_tokens: Any) -> None:
        """Allocate per-request bookkeeping.

        tt_transformers (with ``paged_attention_config=None``) manages KV
        state internally and binds it to a user slot. We don't touch ttnn
        here — actual KV writes happen on the first ``extend`` call. This
        method just records the prompt + position counter the worker uses
        to drive subsequent ``decode_step`` calls.
        """
        if req_id in self._req_state:
            raise ValueError(f"req_id {req_id!r} already active")

        # Accept torch tensors as well as plain Python sequences.
        if hasattr(prompt_tokens, "tolist"):
            tokens = prompt_tokens.tolist()
        else:
            tokens = list(prompt_tokens)
        tokens = [int(t) for t in tokens]

        self._req_state[req_id] = {
            "prompt_tokens": tokens,
            "current_offset": 0,
            "user_id": 0,  # P1 batch=1: every request lives in slot 0.
        }

        _TT_ACTIVE_REQUESTS.inc()

        logger.debug(
            "req.new",
            extra={"req_id": req_id, "prompt_len": len(tokens)},
        )

    def extend(self, req_id: str):
        """Run prefill on the request's prompt; return last-token logits.

        Drives ``Generator.prefill_forward_single_user_text`` (the
        batch=1 path from the captured Phase 0.2 surface) and then pulls
        logits to host via ``process_decode_output_host``.
        """
        if req_id not in self._req_state:
            raise KeyError(f"unknown req_id {req_id!r}")
        state = self._req_state[req_id]
        tokens = state["prompt_tokens"]

        real_prompt_len = len(tokens)
        # tt_transformers' prefill kernel asserts seq_len % 128 == 0. Right-pad
        # the input with the LAST real token so we satisfy the constraint
        # without changing the semantic last-token logits we'll read out.
        # We read logits at last_token_idx=(real_prompt_len-1)%32 within the
        # 32-token block tt_transformers returns; the padded tail's KV state
        # sits in the cache but should not affect decode_step output as long
        # as decode's start_pos=real_prompt_len caps attention at the real
        # position (verified on hardware in Phase G.5).
        step = self._prefill_pad_step
        padded_len = ((real_prompt_len + step - 1) // step) * step
        if padded_len > real_prompt_len:
            filler = tokens[-1]
            padded_tokens = list(tokens) + [filler] * (padded_len - real_prompt_len)
        else:
            padded_tokens = list(tokens)
        # tt_transformers' Model.prepare_inputs_prefill asserts tokens.dim() == 2.
        # We're batch=1 in P1, so unsqueeze to [1, padded_len].
        token_tensor = torch.as_tensor(padded_tokens, dtype=torch.long).unsqueeze(0)

        start = time.perf_counter()
        try:
            tt_out = self._generator.prefill_forward_single_user_text(
                token_tensor,
                page_table=None,           # non-paged in P1
                user_id=state["user_id"],  # P1 batch=1 always 0
                last_token_idx=real_prompt_len - 1,
                kv_cache=None,             # tt_transformers manages KV internally
            )
            # Prefill output uses Model.process_output_prefill (NOT
            # Generator.process_decode_output_host — that's decode-only).
            # The device's get_last_token=(idx//32)*32 returns only the last
            # 32-token block of logits, so the index passed here is
            # (real_prompt_len-1) % 32 — the local position within that block.
            # The model returns [vocab] directly.
            # TODO(F.4): split D2H timing from the surrounding device call so
            # we can populate _TT_D2H_LATENCY_MS independently. Real-HW only.
            last_logits = self._model.process_output_prefill(
                tt_out, last_token_idx=(real_prompt_len - 1) % 32
            )
        except Exception as exc:
            logger.error(
                "req.error",
                extra={"req_id": req_id, "stage": "prefill", "exception": repr(exc)},
                exc_info=True,
            )
            raise
        finally:
            _TT_EXTEND_LATENCY_MS.observe((time.perf_counter() - start) * 1000.0)

        state["current_offset"] = real_prompt_len

        logger.debug(
            "req.extend",
            extra={
                "req_id": req_id,
                "prompt_len": real_prompt_len,
                "padded_prompt_len": padded_len,
                "current_offset": state["current_offset"],
            },
        )

        return last_logits

    def decode_step(self, req_id: str, last_token: int):
        """Advance by one token; return next-token logits.

        Drives ``Generator.decode_forward_text`` and pulls logits to host
        via ``process_decode_output_host``.
        """
        if req_id not in self._req_state:
            raise KeyError(f"unknown req_id {req_id!r}")
        state = self._req_state[req_id]
        offset = state["current_offset"]

        # tt_transformers decode expects 2D token tensor [B=1, 1] and
        # start_pos as a LongTensor (Generator.decode_forward_text:262 calls
        # torch.chunk(start_pos, data_parallel, 0); ints don't chunk).
        token_tensor = torch.as_tensor([[int(last_token)]], dtype=torch.long)
        start_pos_tensor = torch.as_tensor([offset], dtype=torch.long)

        start = time.perf_counter()
        try:
            # read_from_device=True makes decode_forward_text ALREADY call
            # process_decode_output_host internally — returns host tensor
            # of shape [B, S, vocab]. Do not re-process.
            # TODO(F.4): split D2H timing from the surrounding device call so
            # we can populate _TT_D2H_LATENCY_MS independently. Real-HW only.
            logits = self._generator.decode_forward_text(
                token_tensor,
                start_pos=start_pos_tensor,
                page_table=None,
                kv_cache=None,
                enable_trace=True,
                read_from_device=True,
                sampling_params=None,
            )
        except Exception as exc:
            logger.error(
                "req.error",
                extra={"req_id": req_id, "stage": "decode", "exception": repr(exc)},
                exc_info=True,
            )
            raise
        finally:
            _TT_DECODE_LATENCY_MS.observe((time.perf_counter() - start) * 1000.0)

        # decode_forward_text(read_from_device=True) returns [B=1, S=1, vocab];
        # reduce to flat [vocab].
        next_logits = logits[0, 0, :]

        state["current_offset"] = offset + 1

        logger.debug(
            "req.decode",
            extra={
                "req_id": req_id,
                "current_offset": state["current_offset"],
            },
        )

        return next_logits

    def free(self, req_id: str) -> None:
        """Release per-request bookkeeping.

        The non-paged tt_transformers KV cache is bound to a fixed user
        slot; with batch=1 and ``user_id=0`` the slot is implicitly
        overwritten by the next ``extend``. There is no explicit per-
        request KV free call in the captured Phase 0.2 surface, so this
        method only drops host-side state and the active-requests gauge.
        Idempotent: an unknown req_id logs a WARN and returns.
        """
        if req_id not in self._req_state:
            logger.warning("req.free_unknown", extra={"req_id": req_id})
            return

        self._req_state.pop(req_id)
        _TT_ACTIVE_REQUESTS.dec()

        logger.debug("req.free", extra={"req_id": req_id})

    def reset_all(self) -> None:
        """Drop all per-request state (e.g. on scheduler shutdown / restart)."""
        cleared = len(self._req_state)
        self._req_state.clear()
        _TT_ACTIVE_REQUESTS.set(0)

        logger.info("req.reset_all", extra={"cleared": cleared})
