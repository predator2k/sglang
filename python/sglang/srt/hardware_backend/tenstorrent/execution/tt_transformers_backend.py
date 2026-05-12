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

Real forward (F.2) and integration smoke (F.4) land next.
"""

from __future__ import annotations

import logging
import os
from typing import Any

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

        # tt_transformers' ModelArgs reads LLAMA_DIR / HF_MODEL from env.
        # Set LLAMA_DIR so create_tt_model picks up our model path without
        # leaking docker mount semantics into the rest of the wrapper.
        os.environ["LLAMA_DIR"] = model_path

        tt_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[dtype]

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

        logger.info(
            "tt_wrapper_init",
            extra={
                "model_path": model_path,
                "max_seq_len": max_seq_len,
                "dtype": dtype,
                "max_batch_size": max_batch_size,
                "n_layers": getattr(self._model_args, "n_layers", None),
                "model_name": getattr(self._model_args, "model_name", None),
            },
        )

    # ----- Per-request lifecycle (Phase F.2 fills in the real bodies) -----

    def new_request(self, req_id: str, prompt_tokens) -> None:
        """Allocate per-request KV / position state. Phase F.2 body."""
        raise NotImplementedError(
            "TTTransformersExecutionBackend.new_request lands in Phase F.2"
        )

    def extend(self, req_id: str):
        """Run prefill on prompt_tokens. Returns last-token logits. Phase F.2."""
        raise NotImplementedError(
            "TTTransformersExecutionBackend.extend lands in Phase F.2"
        )

    def decode_step(self, req_id: str, last_token: int):
        """Run one decode step. Returns next-token logits. Phase F.2."""
        raise NotImplementedError(
            "TTTransformersExecutionBackend.decode_step lands in Phase F.2"
        )

    def free(self, req_id: str) -> None:
        """Release per-request state. Phase F.2 body."""
        raise NotImplementedError(
            "TTTransformersExecutionBackend.free lands in Phase F.2"
        )

    def reset_all(self) -> None:
        """Drop all per-request state (e.g. on scheduler restart). Phase F.2."""
        raise NotImplementedError(
            "TTTransformersExecutionBackend.reset_all lands in Phase F.2"
        )
