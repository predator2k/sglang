"""Tenstorrent-specific TpModelWorker subclass.

Inherits from TpModelWorker for scheduler integration. Overrides
_init_model_runner to wire up MeshDeviceCtx + execution-backend registry +
warmup + TTModelRunner stub. forward_batch_generation lands in Phase G.2.

The worker MUST NOT import ttnn directly (spec §3.2 invariant #7) — all
device interaction goes through the resolved TTExecutionBackend instance.
"""

from __future__ import annotations

import logging

from sglang.srt.hardware_backend.tenstorrent import warmup
from sglang.srt.hardware_backend.tenstorrent.execution import (
    get_tt_execution_backend,
)
from sglang.srt.hardware_backend.tenstorrent.model_runner import TTModelRunner
from sglang.srt.hardware_backend.tenstorrent.platform import MeshDeviceCtx
from sglang.srt.managers.tp_worker import TpModelWorker

logger = logging.getLogger("sglang.srt.hardware_backend.tenstorrent")

# DEFAULT_TT_MAX_SEQ_LEN: fallback when --context-length is not passed.
# Matches the P1 budget in spec §6.1 (4 GB KV per card / 64 KB per token →
# ~64k tokens fits; we cap at 4096 to keep first-time compile costs low
# and stay well under the bucketed warmup set).
DEFAULT_TT_MAX_SEQ_LEN = 4096


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

        try:
            # Resolve the execution backend via the registry. Reads
            # SGLANG_TT_EXECUTION_BACKEND env var; "auto"/"" → "tt_transformers".
            # NEVER import TTTransformersExecutionBackend directly here —
            # spec §3.2 invariant #7 forbids it (worker depends only on
            # the ABC + registry).
            BackendCls = get_tt_execution_backend()
            max_seq_len = self.server_args.context_length or DEFAULT_TT_MAX_SEQ_LEN

            self.execution_backend = BackendCls(
                self.server_args.model_path,
                self._mesh_ctx.mesh,
                max_seq_len=max_seq_len,
                max_batch_size=1,
                token_to_kv_pool=None,  # simple/B=1 path — no paged allocator
            )

            # Warmup. Phase E.3 deliverable; call whatever entry point
            # warmup.py exposes. If warmup is a no-op stub at this stage,
            # this call is harmless. Phase E.3 (or a follow-up) populates
            # the on-disk kernel cache for all bucketed shapes.
            try:
                warm_fn = getattr(warmup, "warm_decode_shape", None)
                if warm_fn is not None:
                    warm_fn(self.execution_backend)
            except Exception as exc:
                logger.warning(
                    "tt_warmup_failed",
                    extra={"reason": repr(exc)},
                    exc_info=True,
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
        except BaseException:
            # Ensure the mesh is closed if anything below the mesh-open line
            # fails — otherwise the next init attempt collides on hardware that
            # only atexit would have eventually freed.
            self._mesh_ctx.close()
            raise

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

    def forward_batch_generation(
        self,
        model_worker_batch,
        forward_batch=None,
        pp_proxy_tensors=None,
        is_verify=False,
        skip_attn_backend_init=False,
    ) -> "GenerationBatchResult":
        """Polarity mirrors MLX (mlx/tp_worker.py:103-114): if mwb is not
        None, take our path; else fall back to parent (None → speculative
        decoding scratch path; never reached in P1).
        """
        if model_worker_batch is not None:
            return self._forward_batch_generation_tt(model_worker_batch)
        return super().forward_batch_generation(
            model_worker_batch,
            forward_batch,
            pp_proxy_tensors,
            is_verify,
            skip_attn_backend_init,
        )

    def _forward_batch_generation_tt(self, mwb) -> "GenerationBatchResult":
        # In-method imports avoid module-load issues during plugin discovery —
        # sglang.srt.managers chain is heavy. Spec §3.2 invariant #7 still
        # respected: no ttnn / concrete-backend imports.
        import torch

        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult

        forward_batch = self._build_forward_batch(mwb)
        logits_output = self.execution_backend.forward(forward_batch)

        if logits_output.next_token_logits is None:
            # IDLE / abort path — mirrors MLX (mlx/tp_worker.py:136-140).
            return GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=False,
            )

        # Greedy sampling on host. We never call model_runner.sample because
        # model_runner.sampler is None per spec §3.2 invariant #2. Backend
        # already returns host torch tensors — no ttnn import here.
        next_token_ids = torch.argmax(
            logits_output.next_token_logits.float(), dim=-1, keepdim=True
        ).long()

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    def _build_forward_batch(self, mwb):
        """Build a duck-typed forward_batch namespace from a ModelWorkerBatch.

        For the single-user backend, ForwardBatch (with KV pool wiring) is
        overkill — expose only the fields the simple backend reads. Phase 3
        will build a real ForwardBatch for the paged backend.

        The forward_mode guard (EXTEND/DECODE/IDLE only) and 0-req / multi-req
        guards have moved into TTTransformersExecutionBackend.forward, keeping
        the worker thin.
        """
        from types import SimpleNamespace
        return SimpleNamespace(
            forward_mode=mwb.forward_mode,
            input_ids=mwb.input_ids,
            batch_size=len(mwb.reqs) if mwb.reqs else 0,
            reqs=mwb.reqs or [],
        )
