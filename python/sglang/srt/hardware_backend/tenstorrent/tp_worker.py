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

    def _cleanup_stale_rids(self, forward_mode, current_rids: set):
        """Release per-request backend state for reqs that dropped out of
        the decode batch. Called from _forward_batch_generation_tt before
        invoking new_request / decode_step.
        """
        if not hasattr(self, "_tt_active_rids"):
            self._tt_active_rids = set()
        if forward_mode.is_decode():
            stale = self._tt_active_rids - current_rids
            for rid in stale:
                try:
                    self.execution_backend.free(rid)
                except Exception as exc:
                    logger.warning(
                        "tt_free_failed",
                        extra={"req_id": rid, "reason": repr(exc)},
                        exc_info=True,
                    )
            self._tt_active_rids = current_rids
        else:
            self._tt_active_rids |= current_rids

    def _forward_batch_generation_tt(self, mwb) -> "GenerationBatchResult":
        # In-method imports avoid module-load issues during plugin discovery —
        # sglang.srt.managers chain is heavy. Spec §3.2 invariant #7 still
        # respected: no ttnn / concrete-backend imports.
        import torch

        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        # IDLE: mirror MLX (mlx/tp_worker.py:136-140) exactly.
        if mwb.forward_mode.is_idle():
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        # P1 supports EXTEND/DECODE only besides IDLE. Use explicit equality
        # because ForwardMode.is_extend() ALSO returns True for MIXED,
        # DRAFT_EXTEND, TARGET_VERIFY, SPLIT_PREFILL, DLLM_EXTEND — those must
        # raise per spec §3.2 invariant #6.
        if mwb.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise NotImplementedError(
                f"P1 supports EXTEND/DECODE/IDLE only, got {mwb.forward_mode}. "
                f"Chunked prefill / mixed / DLLM must stay disabled via "
                f"apply_server_args_defaults."
            )

        # max_running_requests=1 ⇒ at most one req in the batch normally.
        # SGLang's scheduler can briefly dispatch a 0-req batch between
        # active requests (e.g. when a finished req is being torn down
        # and the next hasn't started yet); treat that like IDLE rather
        # than crashing the scheduler subprocess with an AssertionError.
        # A multi-req batch in P1 is a real bug — raise so we surface it.
        if mwb.reqs is None or len(mwb.reqs) == 0:
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )
        if len(mwb.reqs) > 1:
            raise NotImplementedError(
                f"P1 supports max_running_requests=1; scheduler dispatched "
                f"{len(mwb.reqs)} reqs in mode {mwb.forward_mode}"
            )
        req_id = mwb.reqs[0].rid

        # Reconcile per-request backend state before dispatch (mirrors MLX).
        self._cleanup_stale_rids(
            mwb.forward_mode, {req.rid for req in mwb.reqs}
        )

        if mwb.forward_mode == ForwardMode.EXTEND:
            # mwb.input_ids is a 1-D torch.LongTensor of token IDs on CPU.
            # Coerce to list[int]; the backend handles padding to step.
            prompt_tokens = mwb.input_ids.tolist()
            self.execution_backend.new_request(req_id, prompt_tokens)
            logits = self.execution_backend.extend(req_id)  # host torch.Tensor[vocab]
        else:  # DECODE
            last_tok = int(mwb.input_ids[-1].item())
            logits = self.execution_backend.decode_step(req_id, last_tok)

        # Greedy sampling on host. We never call model_runner.sample because
        # model_runner.sampler is None per spec §3.2 invariant #2. Backend
        # already returns host torch tensors — no ttnn import here.
        next_token_ids = torch.argmax(logits.float(), dim=-1, keepdim=True).long()

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
