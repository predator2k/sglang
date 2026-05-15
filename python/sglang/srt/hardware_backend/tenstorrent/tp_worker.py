"""Tenstorrent-specific TpModelWorker subclass.

Inherits from TpModelWorker for scheduler integration. In *simple* mode
(``tt_transformers_single``) overrides ``_init_model_runner`` to wire up
MeshDeviceCtx + execution-backend registry + warmup + TTModelRunner stub.
In *paged* mode (``tt_transformers_paged``) delegates entirely to the
standard ``TpModelWorker`` init so SGLang's ModelRunner loads
``TenstorrentLlamaForCausalLM`` via ModelRegistry; ``forward_batch_generation``
likewise delegates to ``super()`` for paged mode.

The worker MUST NOT import ttnn directly (spec §3.2 invariant #7) — all
device interaction goes through the resolved TTExecutionBackend instance
(simple path) or through TenstorrentLlamaForCausalLM / BaseMetalDeviceRunner
(paged path).

Mesh-device ownership in paged mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``MeshDeviceCtx`` opens the mesh with fabric + ROW dispatch (the simple-path
approach). In paged mode the plugin's ``BaseMetalDeviceRunner.set_device()``
(called from ``TTModels.__init__``) handles mesh opening instead.  Having both
open the same physical devices would conflict, so in paged mode we skip
``MeshDeviceCtx`` entirely and let the plugin own device lifecycle.
"""

from __future__ import annotations

import logging

from sglang.srt.hardware_backend.tenstorrent.execution import (
    resolve_execution_backend_name,
)
from sglang.srt.managers.tp_worker import TpModelWorker

logger = logging.getLogger("sglang.srt.hardware_backend.tenstorrent")

# DEFAULT_TT_MAX_SEQ_LEN: fallback when --context-length is not passed.
# Matches the P1 budget in spec §6.1 (4 GB KV per card / 64 KB per token →
# ~64k tokens fits; we cap at 4096 to keep first-time compile costs low
# and stay well under the bucketed warmup set).
DEFAULT_TT_MAX_SEQ_LEN = 4096


class TTTpModelWorker(TpModelWorker):
    """TpModelWorker subclass that dispatches between simple and paged TT paths.

    Simple mode (SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single or auto
    resolves to single):
      - self._mesh_ctx: MeshDeviceCtx (owns the 1x2 mesh device lifecycle)
      - self.execution_backend: TTTransformersExecutionBackend
      - self._model_runner: TTModelRunner stub (bookkeeping only)
      - forward_batch_generation: custom greedy-sampling path

    Paged mode (SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged):
      - No MeshDeviceCtx (BaseMetalDeviceRunner inside TTModels owns the mesh)
      - self._model_runner: standard SGLang ModelRunner loading
        TenstorrentLlamaForCausalLM from ModelRegistry
      - forward_batch_generation: delegates to super() (standard SGLang path)
    """

    def __init__(self, **kwargs):
        # Resolve mode BEFORE super().__init__ so that our _init_model_runner
        # override knows which path to take.  super().__init__ calls
        # self._init_model_runner() via Python's MRO, so the flag must be set
        # before the super().__init__ call.
        self._paged_mode = resolve_execution_backend_name() == "tt_transformers_paged"
        logger.info(
            "tt_worker_dispatch",
            extra={"paged_mode": self._paged_mode},
        )

        # P3a.2 EAGLE-3: register draft model classes that SGLang's
        # auto-registration missed due to the layers.utils circular import
        # at TT plugin load time. By now all SGLang modules are settled.
        try:
            from sglang.srt.models.registry import ModelRegistry
            if "LlamaForCausalLMEagle3" not in ModelRegistry.models:
                from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
                ModelRegistry.models["LlamaForCausalLMEagle3"] = LlamaForCausalLMEagle3
                logger.info("Late-registered LlamaForCausalLMEagle3 for EAGLE-3 draft")
        except Exception as exc:
            logger.warning(f"EAGLE-3 late registration failed: {exc!r}")

        # P3a.2 EAGLE-3: SGLang's RotaryEmbedding picks a fallback kernel
        # path when none of (cuda, cpu, xpu, npu, musa, mps) is detected.
        # On TT, all those flags are False, so RotaryEmbedding tries to
        # import `vllm._custom_ops.rotary_embedding` which is absent.
        # Forcing `_is_cpu = True` on the base module steers Python init to
        # the native-torch forward path; runtime forward dispatches based
        # on tensor device (CPU torch handles the draft model fine).
        try:
            import sglang.srt.layers.rotary_embedding.base as _rope_base
            _rope_base._is_cpu = True
            logger.info("Forced rotary_embedding._is_cpu=True (TT: route to native torch)")
        except Exception as exc:
            logger.warning(f"RotaryEmbedding _is_cpu patch failed: {exc!r}")

        # P3a.2 EAGLE-3: SGLang's llama.set_embed (used to share embedding
        # weights between target and EAGLE-3 draft) hard-codes
        # torch.cuda.synchronize() / empty_cache(). On a TT-only build
        # (torch without CUDA) those raise "Torch not compiled with CUDA".
        # Replace them with no-ops globally — CUDA isn't available so
        # there's nothing to sync.
        try:
            import torch
            if not torch.cuda.is_available():
                torch.cuda.synchronize = lambda *a, **k: None
                torch.cuda.empty_cache = lambda *a, **k: None
                logger.info("Stubbed torch.cuda.synchronize/empty_cache (no CUDA available)")
        except Exception as exc:
            logger.warning(f"torch.cuda stub failed: {exc!r}")

        # P3a.2 EAGLE-3: SGLang's CPU draft path triggers torch.compile()
        # (e.g. on apply_rotary_emb), and inductor calls
        # triton.compiler.compiler.triton_key which is missing in our
        # triton-cpu install. Disable dynamo/inductor entirely so models
        # run as plain eager PyTorch.
        try:
            import torch._dynamo
            torch._dynamo.config.disable = True
            import os
            os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
            logger.info("Disabled torch._dynamo/inductor (TT: triton-cpu lacks triton_key)")
        except Exception as exc:
            logger.warning(f"torch._dynamo disable failed: {exc!r}")

        # P3a.2 EAGLE-3: SGLang's tree-build + verify CUDA C++ ops live in
        # sgl_kernel (not installed on a TT-only image). Inject torch
        # fallbacks into eagle_utils' module namespace so
        # `sgl_build_tree_kernel_efficient` resolves at line ~138 of
        # eagle_utils.py and verify_tree_greedy_func dispatches to a real
        # impl on TT instead of silently no-op'ing.
        try:
            from sglang.srt.hardware_backend.tenstorrent.tt_eagle_kernels import (
                install_tt_eagle_kernels,
            )
            install_tt_eagle_kernels()
        except Exception as exc:
            logger.warning(f"TT EAGLE kernel fallback install failed: {exc!r}")

        super().__init__(**kwargs)

    def _init_model_runner(self):
        if self._paged_mode:
            self._init_model_runner_paged()
        else:
            self._init_model_runner_simple()

    def _init_model_runner_paged(self):
        """Paged path: standard SGLang ModelRunner loads TenstorrentLlamaForCausalLM.

        MeshDeviceCtx is NOT opened here — device lifecycle belongs to
        BaseMetalDeviceRunner inside TTModels.__init__ (the plugin-absorbed
        architecture). Opening a second mesh on the same physical devices would
        cause a ttnn double-open conflict.
        """
        logger.info("tt_worker_init_paged_start")
        super()._init_model_runner()
        logger.info(
            "tt_worker_init_paged_done",
            extra={"model_runner": type(self._model_runner).__name__},
        )

    def _init_model_runner_simple(self):
        """Simple/P1 path: MeshDeviceCtx + TTExecutionBackend + TTModelRunner stub.

        MeshDeviceCtx must be constructed from inside the Scheduler subprocess
        (spec §7.3) — that's exactly this function's call-site, because
        TpModelWorker.__init__ runs _init_model_runner from the scheduler's
        subprocess.
        """
        from sglang.srt.hardware_backend.tenstorrent import warmup
        from sglang.srt.hardware_backend.tenstorrent.execution import (
            get_tt_execution_backend,
        )
        from sglang.srt.hardware_backend.tenstorrent.model_runner import TTModelRunner
        from sglang.srt.hardware_backend.tenstorrent.platform import MeshDeviceCtx

        logger.info("tt_worker_init_start")
        self._mesh_ctx = MeshDeviceCtx()

        try:
            # Resolve the execution backend via the registry. Reads
            # SGLANG_TT_EXECUTION_BACKEND env var; "auto"/"" → "tt_transformers_single".
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
        """Override for simple path: stub ModelRunner has no real tokenizer-pad.

        P1 doesn't batch (max_running_requests=1), so padding is trivial.
        MLX returns None here (mlx/tp_worker.py:85-86); we do the same for
        simple mode.  Paged mode delegates to the standard ModelRunner which
        has the real model's pad_input_ids method if it exists.
        """
        if self._paged_mode:
            return super().get_pad_input_ids_func()
        return None

    def forward_batch_generation(
        self,
        model_worker_batch,
        forward_batch=None,
        pp_proxy_tensors=None,
        is_verify=False,
        skip_attn_backend_init=False,
    ) -> "GenerationBatchResult":
        """Dispatch between paged (standard SGLang) and simple (custom TT) paths.

        Paged mode: delegates entirely to super() — SGLang's ModelRunner calls
        TenstorrentLlamaForCausalLM.forward(input_ids, positions, forward_batch).

        Simple mode: mirrors MLX (mlx/tp_worker.py:103-114) — if mwb is not
        None, take our custom greedy path; else fall back to parent (None →
        speculative decoding scratch path; never reached in P1).
        """
        if self._paged_mode:
            return super().forward_batch_generation(
                model_worker_batch,
                forward_batch,
                pp_proxy_tensors,
                is_verify,
                skip_attn_backend_init,
            )
        # Simple path
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
            logits_output.next_token_logits.float(), dim=-1
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
