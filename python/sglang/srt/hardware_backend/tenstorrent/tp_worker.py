"""Tenstorrent-specific TpModelWorker subclass.

Inherits from TpModelWorker for scheduler integration. Overrides
_init_model_runner to wire up MeshDeviceCtx + execution-backend registry +
warmup + TTModelRunner stub. forward_batch_generation lands in Phase G.2.

The worker MUST NOT import ttnn directly (spec §3.2 invariant #7) — all
device interaction goes through the resolved TTExecutionBackend instance.
"""

from __future__ import annotations

import logging

from sglang.srt.hardware_backend.tenstorrent.execution import (
    get_tt_execution_backend,
)
from sglang.srt.hardware_backend.tenstorrent.model_runner import TTModelRunner
from sglang.srt.hardware_backend.tenstorrent.platform import MeshDeviceCtx
from sglang.srt.managers.tp_worker import TpModelWorker

logger = logging.getLogger("sglang.srt.hardware_backend.tenstorrent")


class TTTpModelWorker(TpModelWorker):
    """TpModelWorker subclass that routes inference through a TT execution backend.

    The worker holds three TT-specific attributes:
      - self._mesh_ctx: MeshDeviceCtx (owns the 1x2 mesh device lifecycle)
      - self.execution_backend: a TTExecutionBackend impl (e.g.
        TTTransformersExecutionBackend), resolved via the registry factory
        so SGLANG_TT_EXECUTION_BACKEND can swap in tt_xla post-P1
      - self._model_runner: TTModelRunner stub (bookkeeping only; no real
        forward — that lives in execution_backend)

    Greedy sampling and tensor bridging happen on host inside G.2's
    forward_batch_generation; nothing here calls into ttnn.
    """

    def _init_model_runner(self):
        # MeshDeviceCtx must be constructed from inside the Scheduler
        # subprocess (spec §7.3) — that's exactly this function's
        # call-site, because TpModelWorker.__init__ runs _init_model_runner
        # from the scheduler's subprocess.
        logger.info("tt_worker_init_start")
        self._mesh_ctx = MeshDeviceCtx()

        # Resolve the execution backend via the registry. Reads
        # SGLANG_TT_EXECUTION_BACKEND env var; "auto"/"" → "tt_transformers".
        # NEVER import TTTransformersExecutionBackend directly here —
        # spec §3.2 invariant #7 forbids it (worker depends only on
        # the ABC + registry).
        BackendCls = get_tt_execution_backend()
        max_seq_len = self.server_args.context_length or 4096

        self.execution_backend = BackendCls(
            self.server_args.model_path,
            self._mesh_ctx.mesh,
            max_seq_len=max_seq_len,
        )

        # Warmup. Phase E.3 deliverable; call whatever entry point
        # warmup.py exposes. If warmup is a no-op stub at this stage,
        # this call is harmless. Phase E.3 (or a follow-up) populates
        # the on-disk kernel cache for all bucketed shapes.
        try:
            from sglang.srt.hardware_backend.tenstorrent import warmup

            warm_fn = getattr(warmup, "warm_decode_shape", None)
            if warm_fn is not None:
                warm_fn(self.execution_backend)
        except Exception as exc:
            logger.warning(
                "tt_warmup_failed",
                extra={"reason": repr(exc)},
            )

        # Instantiate the stub ModelRunner. Field name MUST be
        # self._model_runner (with underscore) — TpModelWorker's
        # parent code writes/reads this exact attribute; self.model_runner
        # is a property over the same field.
        # Match MLX's kwarg-passing pattern exactly (mlx/tp_worker.py:61-79).
        self._model_runner = TTModelRunner(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
        )

        logger.info(
            "tt_worker_init_done",
            extra={
                "execution_backend": type(self.execution_backend).__name__,
                "max_seq_len": max_seq_len,
            },
        )

    def get_pad_input_ids_func(self):
        """Override since the stub ModelRunner has no real tokenizer-pad.

        P1 doesn't batch (max_running_requests=1), so padding is trivial.
        MLX returns None here (mlx/tp_worker.py:85-86); we do the same.
        """
        return None
