# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>

import logging
import math
import os
from contextlib import suppress

import torch
from torch import nn

from .tt_utils import BaseMetalDeviceRunner

logger = logging.getLogger(__name__)


class TTModels(nn.Module):
    # Class-level cache of the first-instantiated TTModels' MeshDevice. EAGLE
    # spec-decode mode loads the draft model after the main, and SGLang's
    # standard ModelRunner doesn't go through eagle_draft.load_eagle_draft —
    # it calls our constructor directly with no mesh_device kwarg. P3a.0 T0.3
    # proved a 2nd MeshDevice in the same process is INFEASIBLE, so when we
    # detect this scenario (env var SGLANG_TT_SPEC_DRAFT_PATH set AND
    # _shared_main_mesh is already populated), we auto-cohost on the existing
    # mesh instead of opening a new one.
    _shared_main_mesh = None
    # Class-level constants for default/fallback values
    DEFAULT_BLOCK_SIZE = 64
    DEFAULT_MAX_BATCH_SIZE = 32
    DEFAULT_NUM_LAYERS = 32
    DEFAULT_NUM_KV_HEADS = 8
    DEFAULT_MAX_TOKENS = 131072
    MAX_TOKENS_GPT_OSS = 1024
    MAX_TOKENS_DEEPSEEK_WORMHOLE = 32768
    MAX_TOKENS_WORMHOLE_CONSTRAINED = 65536

    def __init__(self, config, quant_config=None, tt_model=None, mesh_device=None, **kwargs):
        super().__init__()

        # Lazy import: avoids pulling in sglang.srt.layers chain at class-definition time
        # (needed for CPU-only unit tests where logits_processor → dp_attention → configs
        # would fail due to optional deps like Lfm2Config / triton not present).
        from sglang.srt.server_args import get_global_server_args

        # Setup worker environment FIRST, before any tt-metal imports or model init
        # This sets TT_METAL_CACHE and TT_CACHE_HOME per worker for isolation
        from sglang.srt.hardware_backend.tenstorrent.models.worker_setup import (
            setup_worker_from_process_title,
        )

        setup_worker_from_process_title()

        self.config = config  # hf model config
        self.kv_caches = None  # will be allocated on device in allocate_on_device()
        self.block_size = (
            get_global_server_args().page_size or self.DEFAULT_BLOCK_SIZE
        )  # Block size comes from server's --page-size arg

        if tt_model is not None:  # model initialized only once
            self.tt_model = tt_model
        else:
            server_args = (
                get_global_server_args()
            )  # Initialize TT model - get params from server args
            self.max_batch_size = (
                server_args.max_running_requests or self.DEFAULT_MAX_BATCH_SIZE
            )
            self.max_seq_len = server_args.context_length
            # For multi-worker data parallelism, each SGLang worker handles its own DP slice
            # so tt_data_parallel should be 1 per worker (SGLang's dp_size workers provide the parallelism)
            self.tt_data_parallel = (
                1  # Each worker uses all devices on its assigned slot
            )
            self.optimizations = os.environ.get(
                "TT_METAL_OPTIMIZATIONS", "performance"
            )  # From --optimizations CLI arg
            self.override_tt_config = None

            logger.info(
                f"[TT-SGLANG] Model init: max_batch_size={self.max_batch_size}, max_seq_len={self.max_seq_len}, page_size={self.block_size}"
            )
            # Initialize TT device using BaseMetalDeviceRunner
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            # HF_MODEL drives tt_transformers.ModelArgs. For the main model,
            # this is the server's --model-path. For the EAGLE draft (loaded
            # by SGLang's eagle_worker via a second TTModels instance), it
            # must point at the draft's weights, not the main's. We resolve
            # from config._name_or_path when available (HuggingFace AutoConfig
            # sets this to the model path it loaded from).
            cfg_path = getattr(config, "_name_or_path", None) or ""
            os.environ["HF_MODEL"] = cfg_path or get_global_server_args().model_path

            self.device_runner = BaseMetalDeviceRunner(device_id=str(rank))
            _layout = os.environ.get("SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT", "shared")
            if mesh_device is not None:
                # P3a.2 EAGLE: cohost on a pre-opened mesh (typically the main
                # model's mesh). The runner.set_device() override pattern from
                # P3a.0 T0.3 proved cohost feasibility; here we expose it as a
                # constructor parameter so eagle_draft.py can pass main.mesh_device.
                self.mesh_device = mesh_device
                self.device_runner.ttnn_device = mesh_device  # match runner state
            elif (
                TTModels._shared_main_mesh is not None
                and _layout == "shared"
            ):
                # Auto-cohost: a previous TTModels instance is already on a
                # mesh. SGLang's eagle_worker loads the draft via standard
                # ModelRunner with no mesh_device kwarg, so we infer cohost
                # intent from the presence of a prior mesh.
                logger.info(
                    f"[TT-SGLANG] auto-cohost draft on shared mesh id="
                    f"{id(TTModels._shared_main_mesh)} (P3a.2 T2.1 Layout-A)"
                )
                self.mesh_device = TTModels._shared_main_mesh
                self.device_runner.ttnn_device = self.mesh_device
            elif (
                TTModels._shared_main_mesh is not None
                and _layout == "split"
            ):
                # P3a.2 T2.2.H — Layout-B split-mesh cohost: target on chip 0,
                # draft on chip 1. The original Layout-B finding (T0.3) tested
                # only `mesh_shape=(1,2)` on the second model — both tried to
                # claim all chips and ETH-timed-out. Proper split uses
                # `mesh_shape=(1,1)` + `physical_device_ids=[N]` so each model
                # owns a disjoint chip.
                #
                # Diagnosed need (T2.2.G): Layout-A causes wo (and likely
                # other DRAM-sharded weights) to alias across models on the
                # shared mesh — second model's `ttnn.as_tensor` produces the
                # first model's per-device shape. Splitting the mesh breaks
                # the aliasing because each model has its own DRAM region.
                import ttnn
                next_chip = 1  # target took chip 0 by convention
                logger.info(
                    f"[TT-SGLANG] split-mesh cohost: opening draft on chip {next_chip} "
                    f"with mesh_shape=(1,1) (P3a.2 T2.2.H Layout-B)"
                )
                self.mesh_device = ttnn.open_mesh_device(
                    mesh_shape=ttnn.MeshShape(1, 1),
                    physical_device_ids=[next_chip],
                    # P3a.2 T2.2.H: dropped from 50 MB to 10 MB — solo
                    # Blackhole p150a DRAM allocator may silently lock
                    # when asked for 50 MB trace region.
                    trace_region_size=10000000,
                    num_command_queues=1,
                )
                self.device_runner.ttnn_device = self.mesh_device
            elif _layout == "split":
                # P3a.2 T2.2.H Layout-B: first model (target) opens a (1,1)
                # mesh on chip 0 only, leaving chip 1 for the draft.
                import ttnn
                logger.info(
                    "[TT-SGLANG] split-mesh cohost: opening target on chip 0 "
                    "with mesh_shape=(1,1) (P3a.2 T2.2.H Layout-B)"
                )
                self.mesh_device = ttnn.open_mesh_device(
                    mesh_shape=ttnn.MeshShape(1, 1),
                    physical_device_ids=[0],
                    trace_region_size=50000000,
                    num_command_queues=1,
                )
                self.device_runner.ttnn_device = self.mesh_device
                if TTModels._shared_main_mesh is None:
                    TTModels._shared_main_mesh = self.mesh_device
            else:
                self.mesh_device = self.device_runner.set_device()
                # First TTModels in the process becomes the "main"; cache its
                # mesh for any subsequently-instantiated drafts.
                if TTModels._shared_main_mesh is None:
                    TTModels._shared_main_mesh = self.mesh_device

        # P3a: spec-decode routing adapter (INV-8) — initialised after all setup
        from sglang.srt.hardware_backend.tenstorrent.models.spec_decode import SpecDecodeAdapter
        self._spec_adapter = SpecDecodeAdapter(self)
        # Per-request prev_emit token cache for spec verify (Issue 2 fix).
        # Keyed by req_pool_idx so concurrent requests in a shared verify batch
        # each see their own previously-committed token id, avoiding the
        # "Batch size mismatch" crash when prefill bs=1 prev_emit was applied
        # to verify bs>1.
        self._prev_emit_per_req: dict[int, int] = {}

    def on_chunked_prefill_failure(
        self, req, out_cache_loc_this_chunk, allocator, req_to_token_pool
    ):
        """5-step chunked-prefill failure recovery (spec §3.7).

        Called when the TT backend raises during a chunked-prefill extend.
        Returns True so the caller wraps GenerationBatchResult(bypass_chunked_req=True).

        Step a: return slots from failed chunk back to allocator.
        Step b: zero the req_to_token_pool entries written for this chunk.
        Step c: prevent partial-prefix insertion into RadixCache.
        Step d: mark the request as aborted so the scheduler tears it down cleanly.
        Step e: signal the caller via return value (sentinel set by forward wrapper).
        """
        # Step a: return slots from failed chunk
        allocator.free(out_cache_loc_this_chunk)

        # Step b: revert req_to_token writes for this chunk
        start = len(req.prefix_indices) + len(req.fill_ids) - len(out_cache_loc_this_chunk)
        end = len(req.prefix_indices) + len(req.fill_ids)
        req_to_token_pool.req_to_token[req.req_pool_idx, start:end] = 0

        # Step c: prevent partial-prefix insertion into RadixCache
        req.skip_radix_cache_insert = True

        # Step d: mark abort
        req.set_finish_with_abort("tt_backend_chunked_prefill_failure")

        # Step e: signal scheduler (sentinel is set by the forward() wrapper)
        return True

    def forward(  # function running on either prefil or decode mode
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,  # ForwardBatch — lazy import to avoid logits_processor chain
        input_embeds: torch.Tensor = None,
    ):  # -> LogitsProcessorOutput — lazy import
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        self._last_chunked_failure = False  # reset per-call

        # P3a: route speculative verify-batch to SpecDecodeAdapter (INV-8)
        if getattr(forward_batch, "spec_info", None) is not None:
            return self._spec_adapter.forward(forward_batch)

        try:
            return self._call_standard_forward(forward_batch)
        except Exception as exc:
            if (
                forward_batch.forward_mode.is_extend()
                and getattr(forward_batch, "chunked_req", None) is not None
            ):
                # B=1 chunked-prefill assumption: the chunked request is the only req in batch
                failed_req = forward_batch.reqs[0]
                self.on_chunked_prefill_failure(
                    req=failed_req,
                    out_cache_loc_this_chunk=forward_batch.out_cache_loc,
                    allocator=forward_batch.token_to_kv_pool_allocator,
                    req_to_token_pool=forward_batch.req_to_token_pool,
                )
                self._last_chunked_failure = True
                return LogitsProcessorOutput(next_token_logits=None)
            raise

    def _call_standard_forward(self, forward_batch):
        """Execute the existing EXTEND / DECODE branch logic.

        Extracted from forward() by P3a.1 T1.2 to allow SpecDecodeAdapter to
        reach it without going through the spec_info early-exit check.
        """
        import torch as _torch
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        page_table = self._build_page_table(
            forward_batch
        )  # returns block IDs for every user in current batch

        if forward_batch.forward_mode.is_extend():  # prefill mode
            padded_tokens = self._flatten_to_padded(forward_batch.input_ids, forward_batch)

            # Use extend_seq_lens (NEW token lengths) for prompt_lens, not seq_lens (TOTAL length)
            # seq_lens includes cached prefix tokens, but padded_tokens only contains NEW tokens
            prompt_lens_tensor = (
                forward_batch.extend_seq_lens
                if forward_batch.extend_seq_lens is not None
                else forward_batch.seq_lens
            )

            prompt_lens = prompt_lens_tensor.tolist()

            logits = self.tt_model.prefill_forward(
                tokens=padded_tokens.to(_torch.int32),
                page_table=page_table,
                kv_cache=self.kv_caches,
                prompt_lens=prompt_lens,
            )
            logger.debug("tt_model.prefill_forward executed")
            # returns scores for every possible next word and sglang picks the most likely one ( it will become the next token )
            squeezed = logits.squeeze(1)
            # Cache argmax token per request for the FIRST spec-verify cycle.
            try:
                req_indices = forward_batch.req_pool_indices.tolist()
                argmax_tokens = squeezed.argmax(dim=-1).tolist()
                if not isinstance(argmax_tokens, list):
                    argmax_tokens = [argmax_tokens]
                for req_idx, tok in zip(req_indices, argmax_tokens):
                    self._prev_emit_per_req[int(req_idx)] = int(tok)
            except Exception as exc:
                logger.warning(f"[TT-SGLANG] prev_emit cache update skipped: {exc}")
            # P3a.2 EAGLE-3: SGLang's eagle_worker reads
            # logits_output.hidden_states and stuffs it into
            # EagleDraftInput.hidden_states for the draft model. The TT
            # generator surfaces only final logits, not mid-layer hidden
            # states; we provide a zero placeholder shaped [N, hidden_size]
            # so the draft forward doesn't crash on a None .shape access.
            # The draft will then produce garbage proposals (every token
            # rejected by verify), but the EAGLE harness runs end-to-end.
            extra = {}
            if getattr(self, "capture_aux_hidden_states", False):
                # EAGLE-3 draft concatenates `embeds` (shape [N_input_tokens,
                # hidden_size]) with hidden_states along dim=-1. The draft
                # expects the same N row count, not the squeezed batch size.
                # Draft weights are bfloat16, so match that dtype.
                n_tokens = int(forward_batch.input_ids.shape[0])
                hidden = _torch.zeros(
                    (n_tokens, self.config.hidden_size),
                    dtype=_torch.bfloat16,
                )
                extra["hidden_states"] = hidden
            return LogitsProcessorOutput(next_token_logits=squeezed, **extra)

        elif forward_batch.forward_mode.is_decode():  # decode mode
            tokens = forward_batch.input_ids.unsqueeze(
                1
            ).to(
                _torch.int32
            )  # make it batch_size x seq_len dimensions (in decode mode seq_len = 1 ), cast to int32
            start_pos = forward_batch.positions.to(
                _torch.int32
            )  # at which position is each request starting, cast to int32
            actual_bsz = tokens.shape[
                0
            ]  # number of requests in current batch (needed later to slice output)
            tokens, start_pos, page_table = self._pad_decode_batch(
                tokens, start_pos, page_table
            )  # pad batch to required size for TT-Metal

            decode_output = (
                self.tt_model.decode_forward(  # call TT-Metal decode forward
                    tokens=tokens,
                    start_pos=start_pos,
                    page_table=page_table,
                    kv_cache=self.kv_caches,
                    enable_trace=True,
                    read_from_device=True,
                )
            )
            logger.debug("tt_model.decode_forward executed")
            # returns scores for every possible next word and sglang picks the most likely one ( it will become the next token )
            logits = decode_output[0]
            logits = logits[:actual_bsz]  # ignore output of padded requests
            return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))

        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

    def _install_lm_head_hidden_capture_once(self):
        """Wrap the target's final-norm to save its INPUT (pre-norm last-layer
        hidden state) before the norm op consumes it. EAGLE-3's midlayer
        re-norms the hidden state via its own hidden_norm (llama_eagle3.py:86),
        so it expects an UN-normed input. Capturing post-norm (lm_head input)
        gives a re-normed value that drifts the draft out of distribution
        and yields accept_rate=0; capturing pre-norm matches training.

        Reads the ttnn tensor to host inside the wrapper's __call__ (multi-
        device meshes require per-device read via get_device_tensors).
        """
        if getattr(self, "_lm_head_hidden_capture_installed", False):
            return
        try:
            inner = self.tt_model.model[0]
            original_norm = inner.norm

            class _PreNormHiddenCaptureWrapper:
                def __init__(_w, parent, orig):
                    _w._parent = parent
                    _w._orig = orig
                def __call__(_w, x, *args, **kwargs):
                    # Only capture in DECODE mode (verify only uses decode);
                    # prefill calls model.norm too but we don't need it.
                    mode = kwargs.get("mode", None)
                    if mode is None and args:
                        # mode may be passed positionally as 2nd arg
                        mode = args[1] if len(args) >= 2 else None
                    try:
                        import ttnn as _ttnn_local
                        from models.tt_transformers.tt.common import Mode as _Mode
                        is_decode = (mode == _Mode.DECODE) or (
                            isinstance(mode, str) and mode.lower() == "decode"
                        )
                        if is_decode:
                            try:
                                shards = _ttnn_local.get_device_tensors(x)
                                host = _ttnn_local.to_torch(shards[0])
                            except Exception:
                                host = _ttnn_local.to_torch(x)
                            _w._parent._captured_hidden_host = host
                    except Exception as _exc:
                        if not getattr(_w._parent, "_norm_capture_warned", False):
                            logger.warning(f"[TT-SGLANG] pre-norm capture failed: {_exc!r}")
                            _w._parent._norm_capture_warned = True
                    return _w._orig(x, *args, **kwargs)
                def __getattr__(_w, name):
                    return getattr(_w._orig, name)

            inner.norm = _PreNormHiddenCaptureWrapper(self, original_norm)
            self._lm_head_hidden_capture_installed = True
            logger.info(
                "[TT-SGLANG] Installed pre-norm hidden-state capture wrapper "
                "(EAGLE-3 acceptance recovery)"
            )
        except Exception as exc:
            logger.warning(
                f"[TT-SGLANG] pre-norm hidden capture install failed: {exc!r}; "
                "verify will fall back to bonus-embed stub"
            )
            self._lm_head_hidden_capture_installed = False

    def _read_captured_hidden_host(self, bs):
        """Return the captured pre-lm-head hidden state as a host torch
        tensor of shape [bs, hidden_size], or None if unavailable.

        The wrapper reads the ttnn tensor to host inside its __call__ so the
        captured value is already a torch tensor by the time we read it
        here. Shape variants seen so far: [bs, 1, hidden], [1, 1, bs, hidden],
        [num_devices, 1, 1, bs, hidden] (concatenated across mesh).
        """
        import torch as _torch
        host = getattr(self, "_captured_hidden_host", None)
        if host is None:
            return None
        try:
            # Squeeze all-singleton leading dims.
            while host.dim() > 2 and host.shape[0] == 1:
                host = host.squeeze(0)
            if host.dim() == 3 and host.shape[1] == 1:
                host = host.squeeze(1)
            if host.dim() >= 2 and host.shape[-1] == self.config.hidden_size:
                return host[:bs].to(_torch.bfloat16).contiguous()
            return None
        except Exception as exc:
            logger.warning(f"[TT-SGLANG] reshape captured hidden failed: {exc!r}")
            return None

    def _call_prefill_for_verify(self, forward_batch):
        """Verify-batch forward (P3a.1 T1.2 v4 + Issue 2 fix).

        NGRAM STATUS: TODO — operationally usable but not bit-exact.
        ============================================================
        This implementation produces correct-shape verify logits and
        survives concurrent chat-completion traffic, but inherits an
        architectural gap: tt-metal's `decode_forward` has no
        tree-attention self-mask, so the input token at position
        `seq_lens` leaks into output logits via standard causal
        self-attention. CUDA NGRAM uses flashinfer's
        `prefill_wrapper_verify` with tree-mask root self-mask to
        suppress that leak; we cannot replicate this without kernel
        work in `paged_scaled_dot_product_attention_decode`.

        Measured impact (Llama-3.1-8B-Instruct BFP8, P3a.1 T1.4 suite):
          - 89/100 byte-exact NGRAM-vs-baseline on diverse prompts
          - Visible token-doubling artifacts ("$22", "ege gegs sold sold")
            in some outputs; usually don't affect final-answer
            extraction on simple benchmarks.

        Next steps for full correctness:
          - Implement tree-mask in tt-metal decode SDPA (see
            `scripts/tt_metal_patches/04-attention-self-mask-flag.patch`
            for the patch shape — currently blocked on trace-capture
            pre-allocation plumbing).
          - OR pivot to EAGLE (P3a.2), which has much higher draft
            acceptance and so the self-attention leak matters less.

        Per SGLang team guidance 2026-05-13: invest in EAGLE instead.

        Implementation: single decode at `position=seq_lens` with
        `input=prev_emit` (cached per-request to handle concurrent
        verify batches). Output logits tiled to [bs*dn, vocab] —
        siblings always fail acceptance because they're compared
        against the same root argmax, preserving bonus-only semantics.
        """
        import torch as _torch
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        bs = forward_batch.batch_size
        spec_info = forward_batch.spec_info
        draft_token_num = int(spec_info.draft_token_num)
        flat_input = forward_batch.input_ids
        assert flat_input.shape[0] == bs * draft_token_num, (
            f"verify input_ids shape mismatch: got {flat_input.shape[0]}, "
            f"expected bs*dn = {bs}*{draft_token_num} = {bs * draft_token_num}"
        )
        tokens_per_user = flat_input.view(bs, draft_token_num).to(_torch.int32)
        positions_per_user = forward_batch.positions.view(bs, draft_token_num).to(
            _torch.int32
        )

        page_table = self._build_page_table(forward_batch)

        # SGLang allocates slots only at positions [seq_lens, seq_lens+dn-1]
        # for the draft tokens; position seq_lens-1 is NOT freshly allocated
        # in verify mode (its slot belongs to the previously-committed token
        # if any, or is unset for the very first verify cycle). Decoding at
        # seq_lens-1 risks writing to an unmanaged slot and corrupting state.
        # Decoding at seq_lens uses NGRAM's allocated slot but with input =
        # prev_emit, which threads baseline-equivalent kv through the
        # protocol: SGLang's _free_cache will move kv from this slot to the
        # final bonus position, so the bonus's kv = encoding(prev_emit) which
        # matches baseline's "decode of prev_emit at the bonus position".
        # Per-request prev_emit token lookup (Issue 2 fix: avoids the
        # shape mismatch that crashed when one cached logits tensor was
        # applied to a concurrent verify batch).
        req_indices = forward_batch.req_pool_indices.tolist()
        prev_emit_list = []
        miss_count = 0
        for i, req_idx in enumerate(req_indices):
            tok = self._prev_emit_per_req.get(int(req_idx), None)
            if tok is None:
                miss_count += 1
                tok = int(tokens_per_user[i, 0].item())
            prev_emit_list.append(int(tok))
        if miss_count > 0:
            logger.warning(
                f"[TT-SGLANG] prev_emit cache miss for {miss_count}/{bs} reqs"
            )
        prev_emit = _torch.tensor(prev_emit_list, dtype=_torch.int32).unsqueeze(1)  # [bs,1]

        positions_step = positions_per_user[:, 0]  # [bs] = seq_lens
        padded_tokens, padded_positions, padded_pt = self._pad_decode_batch(
            prev_emit, positions_step, page_table
        )
        # P3a.2 T2.2.Q: install a lazy lm_head wrapper that captures the
        # pre-lm-head (= post-norm last-layer hidden state) ttnn tensor on
        # the first decode call. The wrapper is Python-only (no ttnn op)
        # so it doesn't change the trace; on subsequent trace replays the
        # captured tensor's memory address is reused by tt-metal's trace
        # engine, so reading it via ttnn.to_torch() yields the latest
        # hidden state. This is the input that EAGLE-3's draft midlayer
        # was trained on — substituting it should bring accept_rate > 0.
        self._install_lm_head_hidden_capture_once()

        # P3a.2 T2.2.Q: trace replay deallocates intermediate ttnn tensors,
        # so the lm_head-input reference captured during trace compile
        # becomes invalid after replay (TT_THROW "Tensor is not allocated").
        # Disable trace for verify so each call freshly allocates the
        # intermediate, making it readable via ttnn.to_torch(). Cost:
        # per-call Python overhead through ttnn ops (~5–10× slower than
        # trace replay). Worth it for getting accept_rate > 0.
        decode_out = self.tt_model.decode_forward(
            tokens=padded_tokens,
            start_pos=padded_positions,
            page_table=padded_pt,
            kv_cache=self.kv_caches,
            enable_trace=False,
            read_from_device=True,
        )
        logits = decode_out[0][:bs].squeeze(1)  # [bs, vocab]

        # Read back the captured post-norm hidden state (may be None on first
        # call before trace replayed, or if wrapper failed to install).
        captured_hidden_host = self._read_captured_hidden_host(bs)
        # Update per-request prev_emit token cache for the next verify cycle.
        try:
            argmax_tokens = logits.argmax(dim=-1).tolist()
            if not isinstance(argmax_tokens, list):
                argmax_tokens = [argmax_tokens]
            for req_idx, tok in zip(req_indices, argmax_tokens):
                self._prev_emit_per_req[int(req_idx)] = int(tok)
        except Exception as exc:
            logger.warning(f"[TT-SGLANG] prev_emit cache update skipped: {exc}")
        vocab = logits.shape[-1]
        tiled = logits.unsqueeze(1).expand(-1, draft_token_num, -1).reshape(-1, vocab)
        # P3a.2 EAGLE-3: eagle_worker.verify (eagle_worker.py:958) sets
        # spec_info.hidden_states = logits_output.hidden_states so the draft
        # can re-feed accepted-token hidden states next round. The TT device
        # doesn't expose intermediate hidden states for decode, so we
        # approximate: use the target's embedding of each draft token as a
        # signal-carrying hidden_states. This is NOT the true post-norm
        # hidden_state EAGLE-3 was trained on, but it's a better-than-zero
        # input that at least encodes token identity. Likely keeps
        # acceptance rate low (the EAGLE draft was trained on real
        # last-layer hidden states), but degrades gracefully — vs zeros,
        # which leaves the draft no signal at all.
        n_verify_tokens = bs * draft_token_num
        if captured_hidden_host is not None:
            # Real pre-norm hidden state per batch [bs, hidden]. Tile across
            # all draft_token_num positions.
            hidden_stub = (
                captured_hidden_host[:bs]
                .to(_torch.bfloat16)
                .unsqueeze(1)
                .expand(-1, draft_token_num, -1)
                .reshape(n_verify_tokens, -1)
                .contiguous()
            )
            # Debug log on first use to confirm capture path is live.
            if not getattr(self, "_logged_captured_stats", False):
                try:
                    norm = float(captured_hidden_host[:bs].float().norm())
                    logger.info(
                        f"[TT-SGLANG] EAGLE verify using captured hidden: "
                        f"shape={tuple(captured_hidden_host[:bs].shape)} "
                        f"dtype={captured_hidden_host.dtype} "
                        f"l2_norm={norm:.2f}"
                    )
                except Exception:
                    pass
                self._logged_captured_stats = True
        else:
            try:
                embed_w, _ = self.get_embed_and_head()  # cached after first call
                if embed_w is not None and embed_w.numel() > 0:
                    vocab_size = int(embed_w.shape[0])
                    bonus_per_batch = logits.argmax(dim=-1).to(_torch.long).clamp(min=0, max=vocab_size - 1)
                    bonus_hidden = embed_w[bonus_per_batch].to(_torch.bfloat16)
                    hidden_stub = bonus_hidden.unsqueeze(1).expand(-1, draft_token_num, -1).reshape(
                        n_verify_tokens, -1
                    ).contiguous()
                else:
                    hidden_stub = _torch.zeros(
                        (n_verify_tokens, self.config.hidden_size), dtype=_torch.bfloat16
                    )
            except Exception as exc:
                logger.warning(f"verify hidden_states fallback failed ({exc!r}); zeros stub")
                hidden_stub = _torch.zeros(
                    (n_verify_tokens, self.config.hidden_size), dtype=_torch.bfloat16
                )
        return LogitsProcessorOutput(next_token_logits=tiled, hidden_states=hidden_stub)

    # P3a.2 T2.2: EAGLE worker expects target/draft model to expose
    # embed/lm_head tensors for weight tying. tt_transformers keeps these on
    # the TT device, not as host nn.Parameters, so we return empty host-side
    # placeholders. The draft model loaded via tt_transformers already has
    # its own embed/lm_head on device — set_embed_and_head() is a no-op
    # (we cannot actually retie device-resident weights with host tensors,
    # but EAGLE's standard tie-weights path isn't strictly required for the
    # draft to function; it'll use its own loaded weights independently).
    def get_embed_and_head(self):
        """Return host-side embed and lm_head weights for EAGLE weight tying.

        SGLang's eagle_worker calls this on the target so it can pass the
        embed (and optionally lm_head) tensor into the EAGLE draft via
        set_embed/set_embed_and_head. tt_transformers keeps these weights
        on the TT device, so we lazily load the host copy from the HF
        checkpoint at /models/<target>/model.safetensors* the first time
        this is called. Cached on the instance afterwards.
        """
        import torch as _torch
        if getattr(self, "_cached_embed_head", None) is not None:
            return self._cached_embed_head

        from sglang.srt.server_args import get_global_server_args
        model_path = get_global_server_args().model_path
        embed_w = None
        head_w = None
        try:
            import os
            from safetensors import safe_open
            # HuggingFace stores embed at "model.embed_tokens.weight" and
            # lm_head at "lm_head.weight" (or shares it with embed when
            # tie_word_embeddings=True).
            for fname in sorted(os.listdir(model_path)):
                if not fname.endswith(".safetensors"):
                    continue
                with safe_open(os.path.join(model_path, fname), framework="pt", device="cpu") as f:
                    keys = set(f.keys())
                    if embed_w is None and "model.embed_tokens.weight" in keys:
                        embed_w = f.get_tensor("model.embed_tokens.weight")
                    if head_w is None and "lm_head.weight" in keys:
                        head_w = f.get_tensor("lm_head.weight")
                if embed_w is not None and head_w is not None:
                    break
            if head_w is None and embed_w is not None:
                # tie_word_embeddings=True case
                head_w = embed_w
            if embed_w is None:
                logger.warning(
                    f"get_embed_and_head: no embed weight found in {model_path}; "
                    "falling back to empty tensor (EAGLE draft may break)"
                )
                embed_w = _torch.empty(0)
                head_w = _torch.empty(0)
            else:
                logger.info(
                    f"get_embed_and_head: loaded embed={tuple(embed_w.shape)} "
                    f"head={tuple(head_w.shape)} from {model_path}"
                )
        except Exception as exc:
            logger.warning(f"get_embed_and_head: HF load failed ({exc!r}); using empty placeholder")
            embed_w = _torch.empty(0)
            head_w = _torch.empty(0)

        self._cached_embed_head = (embed_w, head_w)
        return embed_w, head_w

    def set_embed_and_head(self, embed, head):
        # No-op: draft model already has its own weights from tt_transformers.
        # EAGLE's intent is to share weights between target/draft to save memory
        # and ensure consistent tokenization, but on TT the weights live on
        # device and re-tying them at this layer isn't tractable.
        pass

    def set_embed(self, embed):
        pass

    @property
    def hot_token_id(self):
        return None  # EAGLE3 hot-token filter not applicable here

    def set_eagle3_layers_to_capture(self, layer_ids=None):
        """EAGLE-3 aux hidden-state capture stub for the TT backend.

        SGLang's ModelRunner.init_aux_hidden_state_capture() calls this on the
        target model so it knows which layer-level hidden states to expose to
        the EAGLE-3 draft. The TT generator runs the full forward inside
        tt-metal and only surfaces final logits at the SGLang boundary, so we
        cannot currently capture mid-layer hidden states. Store the requested
        layer IDs for diagnostics and accept the call so boot proceeds; the
        draft will receive None for aux hidden states and SGLang's eagle_worker
        falls back to using the final hidden state.
        """
        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = getattr(self.config, "num_hidden_layers", None)
            if num_layers:
                layer_ids = [2, num_layers // 2, num_layers - 3]
        self.eagle3_layers_to_capture = layer_ids
        logger.info(
            f"[TT-SGLANG] set_eagle3_layers_to_capture: layer_ids={layer_ids} "
            "(stub — TT generator surfaces only final logits)"
        )

    def allocate_on_device(self):  # function aloocating kv cache
        """
        Allocate the actual KV cache on the TT-Metal device.
        This method should be called from the SGlang's ModelRunner after the pool is initialized.
        """
        import ttnn
        from sglang.srt.server_args import get_global_server_args

        # Get hardware info from mesh_device (already opened by device_runner in tt_utils)
        num_devices_per_model = self.mesh_device.get_num_devices()
        is_wormhole = "wormhole_b0" in ttnn.get_arch_name()
        model_path = get_global_server_args().model_path or ""
        # Get max tokens based on hardware + model configuration
        max_tokens_all_users = self._get_max_tokens_for_hardware(
            model_path, num_devices_per_model, is_wormhole
        )

        # Build KV Cache Shape
        # 1. Account for worst-case batch allocation, Each user in batch might touch a new block
        page_size = (
            get_global_server_args().page_size
        )  # get page size from server args ( how many tokens fit in one block )
        max_batch = (
            get_global_server_args().max_running_requests or self.DEFAULT_MAX_BATCH_SIZE
        )  # max concurrent requests
        max_tokens_all_users += (
            page_size * max_batch
        )  # if all users need new block at the same time
        num_tt_blocks = math.ceil(
            max_tokens_all_users / page_size
        )  # total number of blocks after worst case scenario
        # 2. Calculate num_kv_heads with tensor parallelism adjustment
        num_devices = (
            self.mesh_device.get_num_devices() // self.tt_data_parallel
        )  # Calculate num_kv_heads with tensor parallelism adjustment
        total_kv_heads = getattr(
            self.config,
            "num_key_value_heads",
            getattr(self.config, "num_attention_heads", self.DEFAULT_NUM_KV_HEADS),
        )
        num_kv_heads = total_kv_heads // min(
            num_devices, total_kv_heads
        )  # kv heads per device
        head_size = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )

        kv_cache_shape = (
            num_tt_blocks,  # num_blocks
            num_kv_heads,  # num_kv_heads (TP-adjusted)
            page_size,  # block_size
            head_size,  # head_size
        )
        # Get num_layers from config (like sglang's model_config.get_num_layers_by_block_type())
        num_layers = getattr(
            self.config,
            "num_hidden_layers",
            getattr(
                self.config,
                "n_layers",
                getattr(self.config, "n_layer", self.DEFAULT_NUM_LAYERS),
            ),
        )  # Llama/Mistral, GPT-Neo, GPT-2
        dtype = torch.bfloat16
        # Allocate KV cache directly on TT-Metal
        self.kv_caches = self.tt_model.allocate_kv_cache(
            kv_cache_shape=kv_cache_shape, dtype=dtype, num_layers=num_layers
        )
        logger.debug("tt_model.allocate_kv_cache executed")

    def load_weights(self, weights):  # TT model loads weights during initialization
        pass

    # ======================================== helper functions ============================================

    def _build_page_table(self, forward_batch):
        """Converts SGLang's token indices (memory positions) per user to block IDs per user.
        helper function for forward function"""
        from sglang.srt.server_args import get_global_server_args

        req_to_token_pool = forward_batch.req_to_token_pool  # get token pool
        req_pool_indices = (
            forward_batch.req_pool_indices
        )  # which rows from token pool are used in current batch
        batch_req_tokens = req_to_token_pool.req_to_token[
            req_pool_indices
        ]  # get tokens used in current batch
        server_args = get_global_server_args()
        block_size = server_args.page_size  # get block size
        page_table = (
            batch_req_tokens[:, ::block_size] // block_size
        )  # convert token indices to block IDs
        # Truncate to exact number of blocks the model expects (context_length // block_size)
        max_blocks = server_args.context_length // block_size
        page_table = page_table[:, :max_blocks]
        return page_table.to(torch.int32)

    def _flatten_to_padded(
        self, input_ids: torch.Tensor, forward_batch  # ForwardBatch — lazy import
    ) -> torch.Tensor:
        """Converts SGLang's flattened input_ids to padded batch structure (batch_size, max_len).
        helper function for prefill mode (in forward function)

        IMPORTANT: In extend mode, input_ids only contains NEW tokens being extended,
        not the full sequence. Use extend_seq_lens (new token counts) NOT seq_lens (total length).
        """
        batch_size = forward_batch.batch_size  # number of requests

        # Use extend_seq_lens (new tokens only) if available, otherwise fall back to seq_lens
        # extend_seq_lens = length of NEW tokens in input_ids
        # seq_lens = TOTAL sequence length (including already-cached prefix)
        if forward_batch.extend_seq_lens is not None:
            extend_lens = forward_batch.extend_seq_lens  # new tokens per request
        else:
            extend_lens = forward_batch.seq_lens  # fallback for non-chunked prefill

        max_len = int(
            torch.max(extend_lens).item()
        )  # length of the longest NEW token chunk

        padded_tokens = torch.zeros(
            (batch_size, max_len), dtype=torch.int32, device=input_ids.device
        )
        # sglang gives us flattened input_ids (new tokens only) and their lengths
        # we need to reconstruct batch structure to (batch, max_len)
        start = 0
        for i in range(batch_size):
            length = int(extend_lens[i].item())  # length of NEW tokens for this request
            padded_tokens[i, :length] = input_ids[start : start + length]
            start += length

        return padded_tokens

    def _pad_decode_batch(
        self, tokens: torch.Tensor, start_pos: torch.Tensor, page_table: torch.Tensor
    ):
        """Pads decode batch to required TT-Metal fixed batch size. We pad smaller batches with dummy requests.
        Helper function for decode mode (in forward function)"""
        # calculate required batch size
        dp = getattr(
            self.tt_model, "data_parallel", len(self.tt_model.model)
        )  # number of model replicas
        max_bsz = self.tt_model.model_args[
            0
        ].max_batch_size  # max batch size for this model
        required_bsz = dp * max_bsz  # total slots across all devices
        actual_bsz = tokens.shape[0]  # number of requests in current batch

        # check if we have too many requests
        if actual_bsz > required_bsz:
            raise ValueError(
                f"Decode batch {actual_bsz} exceeds TT capacity {required_bsz}"
            )

        if actual_bsz < required_bsz:  # pad if we have less requests than required
            pad_n = required_bsz - actual_bsz  # number of requests to add
            pad_tokens = torch.zeros(
                (pad_n, 1), dtype=tokens.dtype, device=tokens.device
            )
            tokens = torch.cat(
                [tokens, pad_tokens], dim=0
            )  # add padding tokens to the end of the batch
            # CRITICAL: pad positions with -1 (tells TT-Metal these are invalid slots to skip)
            pad_pos = torch.full(
                (pad_n,), -1, dtype=start_pos.dtype, device=start_pos.device
            )
            start_pos = torch.cat([start_pos, pad_pos], dim=0)
            # pad page table with 0s (dummy block IDs)
            if page_table is not None:
                pt_width = page_table.shape[1]  # max number of blocks per request
                pad_pt = torch.zeros(
                    (pad_n, pt_width), dtype=page_table.dtype, device=page_table.device
                )
                page_table = torch.cat([page_table, pad_pt], dim=0)

        return tokens, start_pos, page_table

    def _get_max_tokens_for_hardware(
        self, model_path: str, num_devices: int, is_wormhole: bool
    ) -> int:
        """Calculates max token limit based on model and hardware configuration.
        Helper function for allocate_on_device"""
        # Default token limit (generous for Blackhole, multi-device, etc.)
        max_tokens = self.DEFAULT_MAX_TOKENS

        # Override for memory-constrained cases (same as sglang plugin)
        if "gpt-oss" in model_path.lower():
            max_tokens = self.MAX_TOKENS_GPT_OSS
        elif "DeepSeek-R1-0528" in model_path and is_wormhole:
            max_tokens = self.MAX_TOKENS_DEEPSEEK_WORMHOLE
        elif is_wormhole:
            # Wormhole-specific memory constraints based on model + device count
            wormhole_limits = [
                (
                    ["Llama-3.1-8B", "Mistral-7B", "gemma-3-4b"],
                    1,
                    self.MAX_TOKENS_WORMHOLE_CONSTRAINED,
                ),  # N150
                (
                    ["DeepSeek-R1-Distill-Qwen-14B", "Qwen2.5-14B"],
                    2,
                    self.MAX_TOKENS_WORMHOLE_CONSTRAINED,
                ),  # N300
                (
                    ["Llama-3.2-90B", "Qwen2.5-VL-72B"],
                    8,
                    self.MAX_TOKENS_WORMHOLE_CONSTRAINED,
                ),  # T3K
            ]
            for models, devices, limit in wormhole_limits:
                if num_devices == devices and any(m in model_path for m in models):
                    max_tokens = limit
                    break

        return max_tokens

    # ======================================== destructor ============================================
    def __del__(self):  # Destructor to clean up TT resources
        with suppress(AttributeError):
            if hasattr(
                self, "tt_model"
            ):  # Delete TT model first in case there are model artifacts
                del self.tt_model
            # Close mesh device using device runner
            if hasattr(self, "device_runner") and self.device_runner is not None:
                self.device_runner.close_device()
                logger.info("Mesh device closed in destructor")


def _create_tt_model_class(
    backend_module_path: str, backend_class_name: str, class_name: str
):
    """Factory to create SGLang wrapper classes for TT-Metal models.

    Args:
        backend_module_path: Import path for the backend module
        backend_class_name: Name of the class to import from the backend module
        class_name: Name for the created wrapper class
    """

    class TTModelForCausalLM(TTModels):
        def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
            super().__init__(config, quant_config, tt_model, **kwargs)

            # Lazy import of TT backend class
            from models.tt_transformers.tt.generator_sglang import (
                GptOssForCausalLM as TT_GptOss,
            )
            from models.tt_transformers.tt.generator_sglang import (
                LlamaForCausalLM as TT_Llama,
            )
            from models.tt_transformers.tt.generator_sglang import (
                MistralForCausalLM as TT_Mistral,
            )
            from models.tt_transformers.tt.generator_sglang import (
                QwenForCausalLM as TT_Qwen,
            )

            backend_classes = {
                "LlamaForCausalLM": TT_Llama,
                "QwenForCausalLM": TT_Qwen,
                "MistralForCausalLM": TT_Mistral,
                "GptOssForCausalLM": TT_GptOss,
            }
            tt_backend_class = backend_classes[backend_class_name]

            self.tt_model = tt_backend_class.initialize_sglang_model(
                config,
                self.mesh_device,
                self.max_batch_size,
                self.max_seq_len,
                tt_data_parallel=self.tt_data_parallel,
                optimizations=self.optimizations,
            )
            logger.info(f"{backend_class_name}.initialize_sglang_model executed")
            self.allocate_on_device()

    TTModelForCausalLM.__name__ = class_name
    TTModelForCausalLM.__qualname__ = class_name
    return TTModelForCausalLM


# Create all TT model wrapper classes using factory pattern
TenstorrentLlamaForCausalLM = _create_tt_model_class(
    "models.tt_transformers.tt.generator_sglang",
    "LlamaForCausalLM",
    "TenstorrentLlamaForCausalLM",
)
TenstorrentQwenForCausalLM = _create_tt_model_class(
    "models.tt_transformers.tt.generator_sglang", "QwenForCausalLM", "TenstorrentQwenForCausalLM"
)
TenstorrentMistralForCausalLM = _create_tt_model_class(
    "models.tt_transformers.tt.generator_sglang",
    "MistralForCausalLM",
    "TenstorrentMistralForCausalLM",
)
TenstorrentGptOssForCausalLM = _create_tt_model_class(
    "models.tt_transformers.tt.generator_sglang",
    "GptOssForCausalLM",
    "TenstorrentGptOssForCausalLM",
)

# All available TT model classes for SGLang
EntryClass = [
    TenstorrentLlamaForCausalLM,
    TenstorrentQwenForCausalLM,
    TenstorrentMistralForCausalLM,
    TenstorrentGptOssForCausalLM,
]
