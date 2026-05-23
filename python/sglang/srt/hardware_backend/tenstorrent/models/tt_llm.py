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
        self._first_emit_per_req: dict[int, int] = {}

        # --- Decode hot-path optimization state (v119 perf gap closure) ---
        # Pre-allocate padding tensors once to avoid per-step torch.zeros +
        # torch.cat allocations in _pad_decode_batch. Initialised lazily on
        # first decode call when tt_model.model_args is available.
        self._decode_pad_ready = False
        self._decode_required_bsz = None
        self._decode_pad_tokens = None
        self._decode_pad_pos = None
        self._decode_pad_pt = None
        # Pre-allocated output buffer for padded tokens/positions
        self._decode_tokens_buf = None
        self._decode_pos_buf = None
        self._decode_pt_buf = None
        # Decode-step microsecond timing (opt-in via SGLANG_TT_DECODE_TIMING=1)
        self._decode_timing_enabled = os.environ.get("SGLANG_TT_DECODE_TIMING", "0") == "1"
        self._decode_timing_count = 0
        self._decode_timing_accum = {}  # phase_name -> cumulative_ms

    def _init_decode_pad_buffers(self, page_table_width: int):
        """One-time lazy init of pre-allocated decode padding buffers.

        Called on the first decode step, after tt_model is fully initialised
        and we know the page_table width. Pre-allocates all padding tensors
        and reusable output buffers so the hot path does only in-place copies
        instead of per-step allocation + concatenation.
        """
        dp = getattr(self.tt_model, "data_parallel", len(self.tt_model.model))
        max_bsz = self.tt_model.model_args[0].max_batch_size
        required_bsz = dp * max_bsz
        self._decode_required_bsz = required_bsz

        # Pre-allocate padding slices (the part beyond actual_bsz)
        # For B=1 workloads, pad_n = required_bsz - 1
        pad_n = required_bsz - 1  # worst case (actual_bsz=1)
        if pad_n > 0:
            self._decode_pad_tokens = torch.zeros((pad_n, 1), dtype=torch.int32)
            self._decode_pad_pos = torch.full((pad_n,), -1, dtype=torch.int32)
            self._decode_pad_pt = torch.zeros(
                (pad_n, page_table_width), dtype=torch.int32
            )

        # Pre-allocate full-sized output buffers
        self._decode_tokens_buf = torch.zeros(
            (required_bsz, 1), dtype=torch.int32
        )
        self._decode_pos_buf = torch.full(
            (required_bsz,), -1, dtype=torch.int32
        )
        self._decode_pt_buf = torch.zeros(
            (required_bsz, page_table_width), dtype=torch.int32
        )

        self._decode_pad_ready = True
        logger.info(
            f"[TT-SGLANG] decode pad buffers initialised: "
            f"required_bsz={required_bsz}, pt_width={page_table_width}"
        )

    def _pad_decode_batch_fast(
        self, tokens: torch.Tensor, start_pos: torch.Tensor,
        page_table: torch.Tensor
    ):
        """Zero-allocation decode batch padding using pre-allocated buffers.

        Replaces _pad_decode_batch in the hot path. Instead of creating new
        tensors and concatenating every step, copies the actual batch data
        into the pre-allocated buffers via narrow+copy_ and returns views.
        """
        actual_bsz = tokens.shape[0]
        required_bsz = self._decode_required_bsz

        if actual_bsz == required_bsz:
            return tokens, start_pos, page_table

        if actual_bsz > required_bsz:
            raise ValueError(
                f"Decode batch {actual_bsz} exceeds TT capacity {required_bsz}"
            )

        # Copy actual data into the pre-allocated buffers
        buf_tok = self._decode_tokens_buf
        buf_pos = self._decode_pos_buf
        buf_pt = self._decode_pt_buf

        # Tokens: [actual_bsz, 1] -> buf[0:actual_bsz]
        buf_tok[:actual_bsz].copy_(tokens)
        buf_tok[actual_bsz:].zero_()

        # Positions: [actual_bsz] -> buf[0:actual_bsz], rest stays -1
        buf_pos[:actual_bsz].copy_(start_pos)
        buf_pos[actual_bsz:].fill_(-1)

        # Page table: [actual_bsz, width] -> buf[0:actual_bsz]
        if page_table is not None:
            pt_w = page_table.shape[1]
            buf_pt[:actual_bsz, :pt_w].copy_(page_table)
            buf_pt[actual_bsz:].zero_()

        return buf_tok, buf_pos, buf_pt

    def _log_decode_timing_step(self, phases: dict):
        """Accumulate per-phase decode timing for periodic logging.

        Called once per decode step with all phase timings for that step.
        Logs averaged timings every 50 steps.
        """
        for phase, elapsed_ms in phases.items():
            if phase not in self._decode_timing_accum:
                self._decode_timing_accum[phase] = 0.0
            self._decode_timing_accum[phase] += elapsed_ms
        self._decode_timing_count += 1

        # Log every 50 steps
        if self._decode_timing_count % 50 == 0:
            n = 50
            avg = {k: v / n for k, v in self._decode_timing_accum.items()}
            total = sum(avg.values())
            logger.info(
                f"[TT-SGLANG] decode timing (avg over {n} steps): "
                f"total={total:.2f}ms "
                + " ".join(f"{k}={v:.2f}ms" for k, v in sorted(avg.items()))
            )
            self._decode_timing_accum = {}

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

        _timing = (
            self._decode_timing_enabled
            and forward_batch.forward_mode.is_decode()
        )
        if _timing:
            import time as _time
            _tpt0 = _time.perf_counter()

        page_table = self._build_page_table(
            forward_batch
        )  # returns block IDs for every user in current batch

        if _timing:
            _tpt1 = _time.perf_counter()

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

            # P3a.2 T2.2.Z+9: detect chat-template prompts here in prefill
            # (the only path that sees raw prompt token IDs). If the input_ids
            # contain Qwen3 chat-template markers (<|im_start|>, <|im_end|>,
            # <think>), tag the request so the EAGLE verify path can engage
            # tree-mask automatically.
            try:
                if not hasattr(self, "_chat_template_per_req"):
                    self._chat_template_per_req = {}
                CHAT_TOKS = {151644, 151645, 151648}
                # input_ids is [total_new_tokens] flat. Use extend_seq_lens
                # to slice per-request. For batch=1 the whole tensor is one req.
                all_ids = forward_batch.input_ids.tolist()
                ext_lens = (
                    forward_batch.extend_seq_lens.tolist()
                    if forward_batch.extend_seq_lens is not None
                    else [len(all_ids)]
                )
                req_indices_pf = forward_batch.req_pool_indices.tolist()
                off = 0
                for ri, l in zip(req_indices_pf, ext_lens):
                    chunk = all_ids[off : off + l]
                    is_chat = any(t in CHAT_TOKS for t in chunk)
                    self._chat_template_per_req[int(ri)] = is_chat
                    if not getattr(self, "_logged_prefill_chat_detect", False):
                        logger.info(
                            f"[TT-SGLANG] prefill chat-detect req={ri} "
                            f"is_chat={is_chat} prompt_len={l} "
                            f"first_5_ids={chunk[:5]} last_5_ids={chunk[-5:]}"
                        )
                        self._logged_prefill_chat_detect = True
                    off += l
            except Exception as exc:
                logger.warning(f"[TT-SGLANG] prefill chat-detect failed: {exc!r}")
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
            # _timing was set before _build_page_table above
            if _timing:
                _t0 = _time.perf_counter()

            tokens = forward_batch.input_ids.unsqueeze(1).to(_torch.int32)
            start_pos = forward_batch.positions.to(_torch.int32)
            actual_bsz = tokens.shape[0]

            if _timing:
                _t1 = _time.perf_counter()

            # Lazy-init pre-allocated padding buffers on first decode call
            if not self._decode_pad_ready:
                self._init_decode_pad_buffers(page_table.shape[1])

            # Use zero-allocation padding path
            tokens, start_pos, page_table = self._pad_decode_batch_fast(
                tokens, start_pos, page_table
            )

            if _timing:
                _t2 = _time.perf_counter()

            # WS-A.19: device-0-only fast readback. After all_gather inside the
            # decode trace, every device in the mesh holds the FULL [1,1,32,vocab]
            # logits tensor.  The canonical ``decode_forward(read_from_device=True)``
            # path calls ``read_decode_output`` which issues ``tt_out.cpu()`` on the
            # mesh handle — that pulls EVERY device's shard to host, doubling the
            # PCIe traffic for a TP=N mesh (~17.6 ms host-copy on 2× P150a for
            # Qwen3.5-0.8B's 152K vocab × 32-row tile × bf16).
            #
            # The fast path here mirrors the standalone trace harness
            # (``_qwen35_trace_harness.decode_one_step``): dispatch + read only
            # device 0's view.  Logits are mathematically identical because the
            # all_gather inside the trace replicates the full vocab to every
            # device, so device 0 already carries the complete answer.
            #
            # Default ON (SGLANG_TT_WSA19_FAST_READ=1).  Set to "0" to revert
            # to the canonical full-mesh readback (regression-bisect aid).
            _fast_read = os.environ.get("SGLANG_TT_WSA19_FAST_READ", "1") == "1"
            _wsa19_timing = os.environ.get("SGLANG_TT_WSA19_TIMING", "0") == "1"

            if _fast_read:
                import ttnn as _ttnn_wsa19
                _t_disp0 = _time.perf_counter() if _wsa19_timing else 0.0
                tt_out = self.tt_model.decode_forward(
                    tokens=tokens,
                    start_pos=start_pos,
                    page_table=page_table,
                    kv_cache=self.kv_caches,
                    enable_trace=True,
                    read_from_device=False,
                )
                _t_disp1 = _time.perf_counter() if _wsa19_timing else 0.0
                # tt_out is a list of (tt_logits, tt_log_probs) per data_parallel rank.
                # data_parallel is 1 for the SGLang server path so we always index [0].
                _tt_logits_dev = tt_out[0][0] if isinstance(tt_out[0], tuple) else tt_out[0]
                # Device-0-only readback: get_device_tensors returns the per-device
                # storage list (host- or device-resident depending on storage state).
                # Since _tt_logits_dev is device-resident, .cpu() on dev0 blocks for
                # GPU finish then copies ONLY device 0's shard.
                _dev0 = _ttnn_wsa19.get_device_tensors(_tt_logits_dev)[0]
                _logits_host = _ttnn_wsa19.to_torch(_dev0)
                _t_read1 = _time.perf_counter() if _wsa19_timing else 0.0
                # Mirror ``process_output_decode``'s shape contract:
                #   [1, 1, padded_batch, vocab_padded] -> [B, S=1, vocab_actual]
                _B = self.tt_model.model_args[0].max_batch_size
                _vocab = self.tt_model.model[0].vocab_size
                _logits_t = _logits_host[:, :, :_B, :_vocab].view(_B, 1, -1).float()
                decode_output = [_logits_t]
                _t_proc1 = _time.perf_counter() if _wsa19_timing else 0.0
                if _wsa19_timing and _timing:
                    self._wsa19_split_accum = getattr(self, "_wsa19_split_accum", {
                        "fwd_dispatch": 0.0, "fwd_read": 0.0, "fwd_process": 0.0,
                        "_n": 0,
                    })
                    self._wsa19_split_accum["fwd_dispatch"] += (_t_disp1 - _t_disp0) * 1000
                    self._wsa19_split_accum["fwd_read"] += (_t_read1 - _t_disp1) * 1000
                    self._wsa19_split_accum["fwd_process"] += (_t_proc1 - _t_read1) * 1000
                    self._wsa19_split_accum["_n"] += 1
                    if self._wsa19_split_accum["_n"] % 50 == 0:
                        n = self._wsa19_split_accum["_n"]
                        logger.info(
                            f"[WSA19] fwd-split FAST (avg over last 50): "
                            f"dispatch={self._wsa19_split_accum['fwd_dispatch']/50:.2f}ms "
                            f"read={self._wsa19_split_accum['fwd_read']/50:.2f}ms "
                            f"process={self._wsa19_split_accum['fwd_process']/50:.2f}ms "
                            f"cum_steps={n}"
                        )
                        for _k in ("fwd_dispatch", "fwd_read", "fwd_process"):
                            self._wsa19_split_accum[_k] = 0.0
            else:
                decode_output = self.tt_model.decode_forward(
                    tokens=tokens,
                    start_pos=start_pos,
                    page_table=page_table,
                    kv_cache=self.kv_caches,
                    enable_trace=True,
                    read_from_device=True,
                )

            if _timing:
                _t3 = _time.perf_counter()

            logits = decode_output[0]
            logits = logits[:actual_bsz]

            if _timing:
                _t4 = _time.perf_counter()
                self._log_decode_timing_step({
                    "pt_build": (_tpt1 - _tpt0) * 1000,
                    "prep": (_t1 - _t0) * 1000,
                    "pad": (_t2 - _t1) * 1000,
                    "fwd": (_t3 - _t2) * 1000,
                    "post": (_t4 - _t3) * 1000,
                })

            return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))

        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

    def _install_aux_layer_capture_once(self):
        """Wrap target decoder layers [2, mid, N-3] to capture their OUTPUT
        ttnn tensors (read to host inside each wrapper's __call__) so we can
        concatenate the three aux hidden states as [N, 3*hidden] — what
        EAGLE-3 was actually trained on. set_eagle3_layers_to_capture log
        already confirms these are the layer_ids the EAGLE-3 draft expects.
        """
        if getattr(self, "_aux_layer_capture_installed", False):
            return
        try:
            inner = self.tt_model.model[0]
            num_layers = len(inner.layers)
            # Match SGLang's default for EAGLE-3:
            #   layer_ids = [2, num_layers // 2, num_layers - 3]
            self._aux_layer_ids = [2, num_layers // 2, num_layers - 3]
            self._captured_aux_layer_host = {i: None for i in self._aux_layer_ids}

            # Deferred-aux-read optimization (path B in v114_tpot_analysis):
            # store a device-side `ttnn.clone(out)` and skip the inline
            # `to_torch`. The host read happens later in
            # `_finalize_aux_capture()` after decode_forward returns, so the
            # inline host-sync no longer blocks subsequent layers.
            # Default on; opt out with SGLANG_TT_EAGLE_DEFERRED_AUX=0.
            # Validated v114b: accept_rate preserved exactly vs baseline
            # across 3 probes; TPOT improved 2-5 ms/tok (2-3%).
            self._captured_aux_layer_device = {i: None for i in self._aux_layer_ids}

            def _make_wrapper(parent, orig, layer_id):
                class _LayerCaptureWrapper:
                    def __call__(_w, *args, **kwargs):
                        out = orig(*args, **kwargs)
                        try:
                            import ttnn as _ttnn_local
                            import torch as _torch_local
                            if os.environ.get("SGLANG_TT_EAGLE_DEFERRED_AUX", "1") != "0":
                                # Defer host read: clone device tensor so it
                                # survives downstream layers; read in batch
                                # after the full forward returns.
                                parent._captured_aux_layer_device[layer_id] = (
                                    _ttnn_local.clone(out)
                                )
                            else:
                                shards = list(_ttnn_local.get_device_tensors(out))
                                per_dev = [_ttnn_local.to_torch(s) for s in shards]
                                host = _torch_local.cat(per_dev, dim=-1) if len(per_dev) > 1 else per_dev[0]
                                parent._captured_aux_layer_host[layer_id] = host
                            if not getattr(parent, f"_logged_layer_{layer_id}", False):
                                _shape = (
                                    "deferred"
                                    if os.environ.get("SGLANG_TT_EAGLE_DEFERRED_AUX", "1") != "0"
                                    else f"shape={tuple(parent._captured_aux_layer_host[layer_id].shape)} dtype={parent._captured_aux_layer_host[layer_id].dtype}"
                                )
                                logger.info(
                                    f"[TT-SGLANG] aux layer {layer_id} captured: {_shape}"
                                )
                                setattr(parent, f"_logged_layer_{layer_id}", True)
                        except Exception as _exc:
                            if not getattr(parent, f"_warned_layer_{layer_id}", False):
                                logger.warning(
                                    f"[TT-SGLANG] aux layer {layer_id} capture failed: {_exc!r}"
                                )
                                setattr(parent, f"_warned_layer_{layer_id}", True)
                        return out
                    def __getattr__(_w, name):
                        return getattr(orig, name)
                return _LayerCaptureWrapper()

            inner.layers = [
                _make_wrapper(self, layer, i) if i in self._aux_layer_ids else layer
                for i, layer in enumerate(inner.layers)
            ]
            self._aux_layer_capture_installed = True
            # Invalidate any existing decode trace so the next decode_forward
            # recaptures WITH the clone ops from the aux wrapper. Without this,
            # trace replay would skip the clones (captured before wrapper install)
            # and aux values would be stale.
            try:
                for m in self.tt_model.model:
                    for attr in ("trace_ids", "_trace_state_text"):
                        if hasattr(m, attr):
                            setattr(m, attr, None)
                logger.info("[TT-SGLANG] Invalidated decode trace for aux-capture recapture")
            except Exception as _te:
                logger.warning(f"[TT-SGLANG] Trace invalidation failed: {_te!r}")
            logger.info(
                f"[TT-SGLANG] Installed aux-layer capture on layers={self._aux_layer_ids} "
                f"(of {num_layers} total)"
            )
        except Exception as exc:
            logger.warning(f"[TT-SGLANG] aux-layer capture install failed: {exc!r}")
            self._aux_layer_capture_installed = False

    def _finalize_aux_capture(self):
        """Drain deferred device-side aux captures into the host dict.

        Called after `decode_forward` returns when the deferred-aux-read
        optimization is on (default; opt-out with
        SGLANG_TT_EAGLE_DEFERRED_AUX=0). Each layer's cloned device
        tensor is read to host now (out of the layer-chain critical path)
        and then released so subsequent decodes can re-clone.
        """
        if os.environ.get("SGLANG_TT_EAGLE_DEFERRED_AUX", "1") == "0":
            return
        import ttnn as _ttnn
        import torch as _torch
        dev_map = getattr(self, "_captured_aux_layer_device", None)
        if not dev_map:
            return
        for lid in list(dev_map.keys()):
            dev_t = dev_map.get(lid)
            if dev_t is None:
                continue
            try:
                shards = list(_ttnn.get_device_tensors(dev_t))
                per_dev = [_ttnn.to_torch(s) for s in shards]
                host = _torch.cat(per_dev, dim=-1) if len(per_dev) > 1 else per_dev[0]
                self._captured_aux_layer_host[lid] = host
            except Exception as exc:
                logger.warning(
                    f"[TT-SGLANG] deferred aux read for layer {lid} failed: {exc!r}"
                )
            finally:
                dev_map[lid] = None  # release ref

    def _read_captured_aux_concat_host(self, bs):
        """Concatenate the 3 captured aux-layer hidden states as [bs, 3*hidden]."""
        import torch as _torch
        layer_ids = getattr(self, "_aux_layer_ids", None)
        if not layer_ids:
            return None
        captured = getattr(self, "_captured_aux_layer_host", {})
        per_layer = []
        for lid in layer_ids:
            host = captured.get(lid)
            if host is None:
                return None
            while host.dim() > 2 and host.shape[0] == 1:
                host = host.squeeze(0)
            if host.dim() == 3 and host.shape[1] == 1:
                host = host.squeeze(1)
            if host.dim() < 2 or host.shape[-1] != self.config.hidden_size:
                return None
            per_layer.append(host[:bs].to(_torch.bfloat16))
        out = _torch.cat(per_layer, dim=-1).contiguous()  # [bs, 3*hidden]
        return out

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
                    try:
                        import ttnn as _ttnn_local
                        import torch as _torch_local
                        # The hidden state is TP-sharded across num_devices
                        # devices along the hidden dim. Read each device's
                        # shard, concat to recover full [..., hidden].
                        try:
                            shards = list(_ttnn_local.get_device_tensors(x))
                            per_dev = [_ttnn_local.to_torch(s) for s in shards]
                            if len(per_dev) > 1:
                                host = _torch_local.cat(per_dev, dim=-1)
                            else:
                                host = per_dev[0]
                        except Exception:
                            host = _ttnn_local.to_torch(x)
                        _w._parent._captured_hidden_host = host
                        if not getattr(_w._parent, "_first_norm_call_logged", False):
                            logger.info(
                                f"[TT-SGLANG] pre-norm wrapper FIRST call: "
                                f"shape={tuple(host.shape)} dtype={host.dtype} "
                                f"mode={kwargs.get('mode', '<missing>')}"
                            )
                            _w._parent._first_norm_call_logged = True
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
        """Return the captured pre-norm hidden state as a host torch tensor
        of shape [bs, hidden_size], or None if unavailable."""
        import torch as _torch
        host = getattr(self, "_captured_hidden_host", None)
        if host is None:
            return None
        try:
            orig_shape = tuple(host.shape)
            # Squeeze all-singleton leading dims.
            while host.dim() > 2 and host.shape[0] == 1:
                host = host.squeeze(0)
            if host.dim() == 3 and host.shape[1] == 1:
                host = host.squeeze(1)
            if host.dim() >= 2 and host.shape[-1] == self.config.hidden_size:
                if not getattr(self, "_logged_read_path", False):
                    logger.info(
                        f"[TT-SGLANG] read captured hidden: orig={orig_shape} → "
                        f"final={tuple(host.shape)} → slice[:bs={bs}]"
                    )
                    self._logged_read_path = True
                return host[:bs].to(_torch.bfloat16).contiguous()
            if not getattr(self, "_logged_read_path", False):
                logger.warning(
                    f"[TT-SGLANG] captured hidden shape mismatch: "
                    f"orig={orig_shape} reduced={tuple(host.shape)} "
                    f"expected last dim={self.config.hidden_size}"
                )
                self._logged_read_path = True
            return None
        except Exception as exc:
            logger.warning(f"[TT-SGLANG] reshape captured hidden failed: {exc!r}")
            return None

    def _call_prefill_for_verify(self, forward_batch):
        """Verify-batch forward. Entered from SpecDecodeAdapter._verify_forward
        on `forward_mode.is_target_verify()` — i.e. both NGRAM and EAGLE-3.

        Current implementation (P3a.2 T2.2.W, v92+): per-position
        multi-decode. Runs `draft_token_num` sequential decodes, one per
        spec-batch position, capturing real 3-aux-layer hidden states from
        target layers [2, 18, 33] after each. The captured `[bs*dn, 3*hidden]`
        tensor feeds EAGLE-3's spec_info.hidden_states. Each decode runs
        with `enable_trace=False` because trace mode deallocates the
        aux-layer intermediates we need to capture. v111 added prefill-time
        chat-template detection so /v1/chat/completions runs in the same
        launch as /generate.

        NGRAM verify also flows through this path. The original NGRAM
        implementation (single decode at position=seq_lens with
        input=prev_emit, output tiled to [bs*dn, vocab]) was replaced by
        the multi-decode rewrite; the rewrite is still correct for NGRAM
        but pays dn× more decode calls per cycle than the bonus-tiled
        form. The NGRAM 89/100 byte-exact-vs-baseline ceiling remains
        because tt-metal lacks a content-aware tree-mask SDPA (see memory
        `tenstorrent-p3a-ngram-architectural-limit`).

        Tree-mask emulation: the `_skip_self_attention` attention-layer
        flag (tt-metal patch 04, fork commit `8cc0c7acc5`) is engaged
        per-batch via runtime loop / chat-flag detection further below.
        It bounds SDPA at cur_pos-1, emulating a root self-mask. The
        original trace-capture blocker doesn't fire because this verify
        path runs with trace disabled.

        Full content-aware tree-mask SDPA in
        `paged_scaled_dot_product_attention_decode` would close the
        remaining gaps (100/100 byte-exact NGRAM, no token doublings
        under EAGLE-3) but is multi-file tt-metal kernel work, out of
        scope here.
        """
        import time as _time
        import torch as _torch
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        _verify_t0 = _time.perf_counter()

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

        self._install_aux_layer_capture_once()
        self._install_lm_head_hidden_capture_once()

        # P3a.2 T2.2.W: per-position multi-decode.
        # Run draft_token_num sequential decodes, one per spec-batch position.
        # Position 0 uses prev_emit; positions 1..dtn-1 use draft proposed
        # tokens. After each decode, capture the 3-aux-layer hidden state
        # for that position. This gives us REAL per-position hidden states
        # instead of tiling the bonus position across all dtn positions
        # (which caused v87-v91 double-token artifacts even with various
        # perturbation schemes — the dominant bonus_aux signal washed out
        # any perturbation we tried).
        #
        # Cost: draft_token_num× more decode_forward calls per verify
        # cycle. With dtn=6 and prior verify=~150ms/decode, each verify
        # cycle takes ~900ms. Net throughput depends on whether the
        # quality boost in accept_rate compensates.
        per_pos_aux = []  # list of [bs, 3*hidden] tensors
        per_pos_logits = []  # list of [bs, vocab] tensors

        # P3a.2 T2.2.Z+5: tree-mask emulation via _skip_self_attention,
        # gated by RUNTIME LOOP DETECTION per request. The flag breaks
        # the chat-template "OkayOkay..." loop but degrades raw-generate.
        # Heuristic: track consecutive-same-token count per request via
        # self._consec_same_count_per_req. When count >= threshold, the
        # request is loop-stuck — enable tree-mask for that batch.
        # Otherwise leave it off (preserves raw-generate quality).
        # Also honor SGLANG_TT_EAGLE_TREE_MASK env override (1=force on,
        # 0=force off, "auto"=use detector, default).
        if not hasattr(self, "_consec_same_count_per_req"):
            self._consec_same_count_per_req = {}
        _env = os.environ.get("SGLANG_TT_EAGLE_TREE_MASK", "auto")
        if _env == "1":
            _tree_mask_on = True
        elif _env == "0":
            _tree_mask_on = False
        else:  # "auto" — prefill-set chat flag + runtime loop safety net
            # T2.2.Z+9: chat detection happens in PREFILL (the only path
            # that sees raw prompt token IDs). Prefill scans input_ids for
            # Qwen3 chat-template tokens and sets _chat_template_per_req[ri].
            # Here we just read the flag.
            chat_per_req = getattr(self, "_chat_template_per_req", {})
            _tree_mask_on = any(
                chat_per_req.get(int(r), False)
                or self._consec_same_count_per_req.get(int(r), 0) >= 1
                for r in req_indices
            )
        _attn_layers = []
        if _tree_mask_on:
            try:
                for layer in self.tt_model.model[0].layers:
                    inner = getattr(layer, "_orig", layer)
                    a = getattr(inner, "attention", None)
                    if a is not None:
                        _attn_layers.append(a)
                        a._skip_self_attention = True
                if not getattr(self, "_logged_skip_self_attn", False):
                    logger.info(
                        f"[TT-SGLANG] Tree-mask ENGAGED (env={_env}) "
                        f"on {len(_attn_layers)} attention layers"
                    )
                    self._logged_skip_self_attn = True
            except Exception as exc:
                logger.warning(f"[TT-SGLANG] tree-mask wire-up failed: {exc!r}")

        # ---------- fast verify: single decode + tile ----------
        # SGLANG_TT_VERIFY_FAST (default "1"): run ONE decode_forward at
        # position 0 (prev_emit) and tile its logits / aux across all
        # draft_token_num positions. This reduces verify latency from
        # dn * decode_time (~72ms for dn=2) to 1 * decode_time (~20-36ms).
        #
        # Trade-off: logits at positions 1..dn-1 are approximations (same
        # as position 0), so draft tokens at those positions will likely
        # be rejected. accept_length drops to ~1.0 (bonus only). But the
        # cycle time savings dominate: 1 decode per cycle vs dn decodes.
        # Net: cycle produces ~1 token in ~20-36ms vs baseline's 1 token
        # in ~37ms, a wash or small win. With any draft acceptance > 0,
        # we get 2+ tokens per cycle.
        #
        # Gated by SGLANG_TT_VERIFY_FAST=1 (default on). Set to "0" to
        # fall back to the original per-position multi-decode path.
        _fast_verify = os.environ.get("SGLANG_TT_VERIFY_FAST", "1") != "0"

        try:
            if _fast_verify:
                # --- fast path: single decode at position 0 ---
                token_0 = prev_emit  # [bs, 1]
                pos_0 = positions_step.to(_torch.int32)  # [bs]
                padded_tokens_0, padded_positions_0, padded_pt_0 = self._pad_decode_batch(
                    token_0, pos_0, page_table
                )
                decode_out_0 = self.tt_model.decode_forward(
                    tokens=padded_tokens_0,
                    start_pos=padded_positions_0,
                    page_table=padded_pt_0,
                    kv_cache=self.kv_caches,
                    enable_trace=True,  # trace always on for fast path
                    read_from_device=True,
                )
                self._finalize_aux_capture()
                logits_0 = decode_out_0[0][:bs].squeeze(1)  # [bs, vocab]
                aux_0 = self._read_captured_aux_concat_host(bs)  # [bs, 3*hidden]

                # Now fill KV for remaining positions 1..dn-1 using
                # sequential decodes — but WITHOUT reading logits back
                # from device (skip the expensive host sync). We only need
                # the KV cache filled for accepted positions.
                # Actually, we also need logits for positions 1..dn-1 for
                # verification. Two sub-strategies:
                #
                # Sub-A (SGLANG_TT_VERIFY_FAST_TILE=1, default): tile
                #   position 0's logits to all positions. KV for positions
                #   1..dn-1 is NOT filled (accepted tokens' KV gets
                #   re-encoded on the next cycle's position 0 decode
                #   anyway — SGLang's protocol re-fills the bonus slot).
                #   Fastest: 1 decode per cycle.
                #
                # Sub-B (SGLANG_TT_VERIFY_FAST_TILE=0): still do decodes
                #   for positions 1..dn-1 but use trace (fast). This gets
                #   real per-position logits while keeping trace on.
                #   Cost: dn decodes but each is fast (traced, no aux).
                _tile_mode = os.environ.get("SGLANG_TT_VERIFY_FAST_TILE", "1") != "0"

                if _tile_mode:
                    # Tile position 0's logits and aux to all positions
                    for _i in range(draft_token_num):
                        per_pos_logits.append(logits_0)
                        per_pos_aux.append(aux_0)
                    if not getattr(self, "_logged_fast_verify_tile", False):
                        logger.info(
                            f"[TT-SGLANG] FAST VERIFY (tile mode): 1 decode "
                            f"for dn={draft_token_num}, trace=True"
                        )
                        self._logged_fast_verify_tile = True
                else:
                    # Position 0 already done
                    per_pos_logits.append(logits_0)
                    per_pos_aux.append(aux_0)
                    # Positions 1..dn-1: decode with trace, skip aux capture
                    for i in range(1, draft_token_num):
                        token_i = tokens_per_user[:, i : i + 1]  # [bs, 1]
                        pos_i = (positions_step + i).to(_torch.int32)
                        padded_tokens_i, padded_positions_i, padded_pt_i = self._pad_decode_batch(
                            token_i, pos_i, page_table
                        )
                        decode_out_i = self.tt_model.decode_forward(
                            tokens=padded_tokens_i,
                            start_pos=padded_positions_i,
                            page_table=padded_pt_i,
                            kv_cache=self.kv_caches,
                            enable_trace=True,
                            read_from_device=True,
                        )
                        logits_i = decode_out_i[0][:bs].squeeze(1)
                        per_pos_logits.append(logits_i)
                        # Tile position 0's aux for remaining positions
                        # (skip per-position capture — saves ~3ms/pos)
                        per_pos_aux.append(aux_0)
                    if not getattr(self, "_logged_fast_verify_notile", False):
                        logger.info(
                            f"[TT-SGLANG] FAST VERIFY (no-tile mode): "
                            f"{draft_token_num} decodes with trace=True, "
                            f"aux tiled from pos 0"
                        )
                        self._logged_fast_verify_notile = True
            else:
                # --- original path: per-position multi-decode ---
                for i in range(draft_token_num):
                    if i == 0:
                        token_i = prev_emit  # [bs, 1] — the bonus
                    else:
                        token_i = tokens_per_user[:, i : i + 1]  # [bs, 1] — draft @ pos i
                    pos_i = (positions_step + i).to(_torch.int32)  # [bs]
                    padded_tokens_i, padded_positions_i, padded_pt_i = self._pad_decode_batch(
                        token_i, pos_i, page_table
                    )
                    _verify_trace = os.environ.get("SGLANG_TT_VERIFY_TRACE", "1") != "0"
                    decode_out_i = self.tt_model.decode_forward(
                        tokens=padded_tokens_i,
                        start_pos=padded_positions_i,
                        page_table=padded_pt_i,
                        kv_cache=self.kv_caches,
                        enable_trace=_verify_trace,
                        read_from_device=True,
                    )
                    # Path B (deferred-aux-read, default on): drain device-side
                    # clones into the host dict now that the forward is done.
                    # No-op when SGLANG_TT_EAGLE_DEFERRED_AUX=0.
                    self._finalize_aux_capture()
                    logits_i = decode_out_i[0][:bs].squeeze(1)  # [bs, vocab]
                    aux_i = self._read_captured_aux_concat_host(bs)  # [bs, 3*hidden]
                    per_pos_logits.append(logits_i)
                    per_pos_aux.append(aux_i)
        finally:
            # Always clear the flag — other code paths (regular decode for
            # actual generation) should NOT use tree-mask.
            for a in _attn_layers:
                a._skip_self_attention = False

        # logits-per-position: stack to [bs, dtn, vocab]
        logits_per_pos = _torch.stack(per_pos_logits, dim=1)  # [bs, dtn, vocab]

        # T2.2.Z+2 reverted: the no-repeat-bigram mask broke /generate too
        # (outputs degenerated to "Tokyo!... " after first token). Trade-off
        # not worth it. Chat-template /v1/chat/completions limitation
        # remains documented as a known issue requiring tree-mask SDPA in
        # tt-metal C++ kernel work.

        # logits[:, 0] is what we use to update prev_emit cache (bonus position).
        logits = per_pos_logits[0]
        # T2.2.W: aux concat is the LAST position's aux (used as fallback if
        # per-position assembly fails); per_pos_aux carries all dtn entries.
        captured_aux_concat = per_pos_aux[0]
        captured_hidden_host = self._read_captured_hidden_host(bs)
        # Update per-request prev_emit token cache for the next verify cycle.
        # T2.2.Z+5: also track consecutive-same count so the tree-mask
        # loop-detector knows when a request is stuck.
        try:
            argmax_tokens = logits.argmax(dim=-1).tolist()
            if not isinstance(argmax_tokens, list):
                argmax_tokens = [argmax_tokens]
            for req_idx, tok in zip(req_indices, argmax_tokens):
                ri = int(req_idx)
                prev = self._prev_emit_per_req.get(ri, None)
                if prev is not None and prev == int(tok):
                    self._consec_same_count_per_req[ri] = (
                        self._consec_same_count_per_req.get(ri, 0) + 1
                    )
                else:
                    self._consec_same_count_per_req[ri] = 0
                self._prev_emit_per_req[ri] = int(tok)
                # Record the FIRST emit per request for chat-mode detection.
                # Subsequent emits don't update it (only the first verify
                # cycle's bonus matters for chat-vs-raw classification).
                if ri not in self._first_emit_per_req:
                    self._first_emit_per_req[ri] = int(tok)
        except Exception as exc:
            logger.warning(f"[TT-SGLANG] prev_emit cache update skipped: {exc}")
        vocab = logits.shape[-1]
        # T2.2.W: real per-position logits laid out as [bs*dtn, vocab].
        # Position 0 = bonus (logits[0]), positions 1..dtn-1 = drafts.
        tiled = logits_per_pos.reshape(bs * draft_token_num, vocab)
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
        # T2.2.W: assemble per-position aux concat. Each position has its
        # own real captured 3-aux-layer hidden state from running the
        # target decode at that position. No perturbation needed — these
        # are the real signals the EAGLE-3 draft was trained on.
        all_per_pos_valid = all(a is not None for a in per_pos_aux)
        if all_per_pos_valid:
            hidden_stub = _torch.zeros(
                (n_verify_tokens, 3 * self.config.hidden_size),
                dtype=_torch.bfloat16,
            )
            for b in range(bs):
                for i in range(draft_token_num):
                    hidden_stub[b * draft_token_num + i] = per_pos_aux[i][b]
            if not getattr(self, "_logged_per_pos_aux", False):
                logger.info(
                    f"[TT-SGLANG] EAGLE verify PER-POSITION AUX: "
                    f"{draft_token_num} positions × shape "
                    f"{tuple(per_pos_aux[0].shape)} each (true per-pos hidden)"
                )
                self._logged_per_pos_aux = True
        elif captured_aux_concat is not None:
            # Fallback: tile last-position aux + cumulative embed perturb.
            try:
                embed_w, _ = self.get_embed_and_head()
                has_embed = embed_w is not None and embed_w.numel() > 0
            except Exception:
                has_embed = False
                embed_w = None
            hidden_stub = _torch.zeros(
                (n_verify_tokens, 3 * self.config.hidden_size),
                dtype=_torch.bfloat16,
            )
            for b in range(bs):
                hidden_stub[b * draft_token_num] = captured_aux_concat[b]
                if has_embed:
                    vocab_size = int(embed_w.shape[0])
                    cum_embed = _torch.zeros(self.config.hidden_size, dtype=_torch.bfloat16)
                    for i in range(1, draft_token_num):
                        tok_id = int(flat_input[b * draft_token_num + i].item())
                        tok_id = max(0, min(tok_id, vocab_size - 1))
                        embed_vec = embed_w[tok_id].to(_torch.bfloat16)
                        cum_embed = cum_embed + embed_vec
                        embed_triplet = _torch.cat([cum_embed] * 3, dim=-1)
                        hidden_stub[b * draft_token_num + i] = (
                            captured_aux_concat[b] + 0.1 * embed_triplet
                        )
        elif captured_hidden_host is not None:
            # Fallback: single-layer pre-norm hidden_state. Empirically
            # bound at accept_rate=0 (v77-v86) — kept only so the pipeline
            # remains coherent if aux wrappers fail to install.
            hidden_single = captured_hidden_host[:bs].to(_torch.bfloat16)
            hidden_stub = (
                hidden_single
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
        # --- verify cycle timing ---
        _verify_elapsed_ms = (_time.perf_counter() - _verify_t0) * 1000
        if not hasattr(self, "_verify_timing_count"):
            self._verify_timing_count = 0
            self._verify_timing_sum = 0.0
        self._verify_timing_count += 1
        self._verify_timing_sum += _verify_elapsed_ms
        # Log every 10 cycles to avoid flooding
        if self._verify_timing_count % 10 == 0:
            avg_ms = self._verify_timing_sum / self._verify_timing_count
            logger.info(
                f"[TT-SGLANG] verify cycle #{self._verify_timing_count}: "
                f"this={_verify_elapsed_ms:.1f}ms avg={avg_ms:.1f}ms "
                f"fast={_fast_verify} dn={draft_token_num} bs={bs}"
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

        # WS-A.18: multimodal HF configs (e.g. Qwen3_5Config) nest the text
        # decoder params under ``config.text_config``; non-multimodal configs
        # carry them at the top level. Prefer the nested values when present
        # so head/dim lookups don't AttributeError on Qwen3.5. Identical to
        # the top-level path for every flat-config model.
        _txt_cfg = getattr(self.config, "text_config", None) or self.config

        def _txt_attr(name, default=None):
            return getattr(_txt_cfg, name, getattr(self.config, name, default))

        total_kv_heads = _txt_attr(
            "num_key_value_heads",
            _txt_attr("num_attention_heads", self.DEFAULT_NUM_KV_HEADS),
        )
        num_kv_heads = total_kv_heads // min(
            num_devices, total_kv_heads
        )  # kv heads per device
        # WS-A.18: Qwen3.5 needs the WS-A.10 KV-head replicate factor honored
        # at KV-cache allocation time too — the per-device cache must have
        # exactly the ``n_local_kv_heads`` slots that paged_update_cache will
        # write, and ModelArgs.n_kv_heads is the post-replicate count. Read
        # from the in-process ModelArgs (set during initialize_sglang_model)
        # if available; falls back to the HF-config-derived value for every
        # other model.
        try:
            _ma = self.tt_model.model_args[0]
            num_kv_heads = int(_ma.n_kv_heads) // min(int(_ma.num_devices), int(_ma.n_kv_heads))
        except (AttributeError, IndexError, TypeError):
            pass
        head_size = _txt_attr(
            "head_dim",
            _txt_attr("hidden_size", 0) // max(_txt_attr("num_attention_heads", 1), 1),
        )

        kv_cache_shape = (
            num_tt_blocks,  # num_blocks
            num_kv_heads,  # num_kv_heads (TP-adjusted)
            page_size,  # block_size
            head_size,  # head_size
        )
        # Get num_layers from config (like sglang's model_config.get_num_layers_by_block_type())
        # WS-A.18: same multimodal-aware lookup as the head/dim attrs.
        num_layers = _txt_attr(
            "num_hidden_layers",
            _txt_attr(
                "n_layers",
                _txt_attr("n_layer", self.DEFAULT_NUM_LAYERS),
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
        helper function for forward function.

        v119 optimisation: cache the page_table for decode steps since
        it only changes when a new block boundary is crossed (every
        page_size tokens). The cache is keyed by the req_pool_indices
        fingerprint and the first slot value; invalidated on any prefill.
        """
        from sglang.srt.server_args import get_global_server_args

        req_to_token_pool = forward_batch.req_to_token_pool
        req_pool_indices = forward_batch.req_pool_indices
        batch_req_tokens = req_to_token_pool.req_to_token[req_pool_indices]
        server_args = get_global_server_args()
        block_size = server_args.page_size
        page_table = (
            batch_req_tokens[:, ::block_size] // block_size
        )
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


_QWEN35_LOADER_SHIMS_INSTALLED = False


def _is_qwen35_config(config) -> bool:
    """True iff this HF config describes a Qwen3.5-* model.

    Detection precedence: ``model_type`` first (most robust — set by the
    HF Qwen3_5Config subclass), then architecture name, then a path-name
    fallback for offline checkpoints whose ``model_type`` may be missing.
    """
    mt = getattr(config, "model_type", "") or ""
    if mt.startswith("qwen3_5"):
        return True
    archs = list(getattr(config, "architectures", None) or [])
    if any(a.startswith("Qwen3_5") for a in archs):
        return True
    name = (getattr(config, "_name_or_path", "") or "").lower()
    return "qwen3.5" in name or "qwen3_5" in name


def _install_qwen35_loader_shims_once():
    """Install the Qwen3.5 safetensors-direct loader + text-only shim.

    Mirrors the standalone ``_qwen35_harness._install_qwen35_loader_shims``
    so the server path runs the same loader as the standalone PCC/perf
    harnesses (WS-A.8 / WS-A.10 / WS-A.15).

    Why this is needed on the server path:

      * Qwen3.5-0.8B ships ``architectures=["Qwen3_5ForConditionalGeneration"]``
        with a ``vision_config`` block, so ``ModelArgs._set_hf_params`` flips
        ``is_multimodal=True`` and ``ModelArgs.load_state_dict`` routes
        through ``convert_vision_hf_to_meta_no_qkv_permute`` which does NOT
        apply the WS-A.10 ``kv_head_replicate_factor=2`` transform.
        Result: the first full-attention layer (layer 3) raises
        ``RuntimeError: shape '[2, 256, 1024]' is invalid for input of
        size 262144`` because the wk weight is still un-replicated.

      * Forcing ``is_multimodal=False`` is consistent with WS-B's "vision
        out of scope for first port" rule — the TT plugin only wires the
        text decoder, the multimodal/vision tower is intentionally not
        ported.

      * Direct safetensors loading is faster (skips full HF model
        construction) and matches the standalone PCC baseline byte-for-byte.

    Idempotent: subsequent calls are no-ops via the module-level guard.
    """
    global _QWEN35_LOADER_SHIMS_INSTALLED
    if _QWEN35_LOADER_SHIMS_INSTALLED:
        return

    import glob

    import safetensors.torch as st

    from models.tt_transformers.tt import model_config as _mc
    from models.tt_transformers.tt.load_checkpoints import (
        convert_hf_to_meta_no_qkv_permute,
        standardize_hf_keys,
    )

    def _direct_safetensors_loader(self):
        # Only intercept Qwen3.5; every other model falls through to the
        # original canonical loader. The wrapper keeps Llama/Qwen3-8B/
        # Mistral/GptOss byte-equivalent.
        if not _is_qwen35_config(getattr(self, "hf_config", None)):
            return _orig_load_state_dict(self)
        sd = {}
        for p in sorted(glob.glob(os.path.join(self.CKPT_DIR, "model.safetensors*.safetensors"))):
            sd.update(st.load_file(p))
        # Qwen3.5 nests text params under "model.language_model.*"
        # (multimodal vision-prefix layout); standardize_hf_keys only knows
        # the flat "model.*" layout, so we re-key up front.
        sd = {k.replace("model.language_model.", "model.", 1): v for k, v in sd.items()}
        # Drop the visual + MTP heads — text-only port doesn't need them and
        # they confuse the downstream HF→Meta rename pass.
        sd = {
            k: v
            for k, v in sd.items()
            if not k.startswith("model.visual.") and not k.startswith("mtp.")
        }
        # Qwen3.5 ties word embeddings, so the safetensors only ship
        # ``model.embed_tokens.weight`` (no ``lm_head.weight``). Synthesize
        # ``lm_head.weight`` from the embedding before standardize_hf_keys
        # collapses both to a single key.
        if "lm_head.weight" not in sd and "model.embed_tokens.weight" in sd:
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
        sd = standardize_hf_keys(sd)
        _orig_kv = int(getattr(self, "_n_kv_heads_orig", self.n_kv_heads))
        _kv_replicate = int(getattr(self, "kv_head_replicate_factor", 1))
        sd = convert_hf_to_meta_no_qkv_permute(
            sd,
            self.head_dim,
            self.n_heads,
            _orig_kv,
            kv_head_replicate_factor=_kv_replicate,
        )
        return sd

    _orig_load_state_dict = _mc.ModelArgs.load_state_dict
    _orig_set_hf_params = _mc.ModelArgs._set_hf_params

    def _set_hf_params_text_only(self, ckpt_dir):
        _orig_set_hf_params(self, ckpt_dir)
        # Only flip is_multimodal for Qwen3.5; leaves every other model
        # (Llama-3.2-Vision, Mistral-Pixtral, etc.) untouched.
        if _is_qwen35_config(getattr(self, "hf_config", None)) and getattr(self, "is_multimodal", False):
            logger.info(
                "[TT-Plugin] forcing is_multimodal=False for Qwen3.5 (text-only port)"
            )
            self.is_multimodal = False

    _mc.ModelArgs.load_state_dict = _direct_safetensors_loader  # type: ignore[assignment]
    _mc.ModelArgs._set_hf_params = _set_hf_params_text_only  # type: ignore[assignment]
    _QWEN35_LOADER_SHIMS_INSTALLED = True
    logger.info("[TT-Plugin] Qwen3.5 loader shims installed (idempotent)")


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

            # WS-A.18: Qwen3.5 needs the safetensors-direct loader + the
            # is_multimodal=False shim so the canonical multimodal converter
            # (which lacks WS-A.10 KV-replicate) is bypassed. Installed
            # lazily and only fires for Qwen3.5 configs; every other model
            # is byte-equivalent to the pre-WS-A.18 path.
            if _is_qwen35_config(config):
                _install_qwen35_loader_shims_once()

            # Tenstorrent-p1: DRAM prefetcher (decode-stage weight prefetch)
            # opt-in via SGLANG_TT_USE_PREFETCHER. Forwarded to tt-metal's
            # initialize_sglang_text_transformer; only takes effect when the
            # model is in `VERIFIED_MODEL_CONFIGS` and dims fit the
            # is_prefetcher_supported() L1/CB checks. Default off.
            _use_prefetcher = os.environ.get("SGLANG_TT_USE_PREFETCHER", "0") == "1"
            self.tt_model = tt_backend_class.initialize_sglang_model(
                config,
                self.mesh_device,
                self.max_batch_size,
                self.max_seq_len,
                tt_data_parallel=self.tt_data_parallel,
                optimizations=self.optimizations,
                use_prefetcher=_use_prefetcher,
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
