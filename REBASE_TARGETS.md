# Rebase Targets

This file tracks in-tree patch sites that diverge from upstream SGLang main.
Review and re-apply on each monthly rebase against `sgl-project/sglang:main`.

---

## Patch 1 — `GenerationBatchResult.bypass_chunked_req`

**File:** `python/sglang/srt/model_executor/forward_batch_info.py`
**Branch task:** P1 simple backend
**Summary:** Added `bypass_chunked_req` field to `GenerationBatchResult` to allow
the Tenstorrent backend to signal that chunked prefill should be skipped for a
given request.
**Rebase note:** If upstream modifies `GenerationBatchResult`, re-apply this field
and ensure the Tenstorrent model runner still reads it.

---

## Patch 2 — Lazy flashinfer.comm import for non-CUDA hosts

**Files:**
- `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`

**Branch task:** P2a.1 T1.4 — soft CUDA imports
**Summary:** `flashinfer.comm.cuda_ipc.CudaRTLibrary()` raises `AssertionError`
(not `ImportError`) when `libcudart` is not loaded. Changed the existing
`except ImportError` to `except (ImportError, AssertionError)` so the module
stays importable on non-CUDA hosts (e.g. Tenstorrent Blackhole targets).
MoE models will fail at instantiation time with a clean error rather than at
import time.
**Rebase note:** If upstream changes the flashinfer import guard in
`token_dispatcher/flashinfer.py`, ensure `AssertionError` is still caught.
Upstream PR reference: none (fork-only, per N9 constraint).

---

_Monthly rebase reminder: run `git rebase upstream/main` and verify each patch
site above still applies cleanly. Update this file if patch files or line
numbers change._

## SGLang upstream-class patches in our fork (active monthly rebase target)

| File | Line range (approx) | Purpose | First landed |
|---|---|---|---|
| `python/sglang/srt/managers/utils.py` | `GenerationBatchResult` dataclass | Add `bypass_chunked_req: bool = False` for tenstorrent chunked-prefill failure recovery | 2026-05-13 (P2a.2 T2.1) |
| `python/sglang/srt/managers/scheduler.py` | post-forward result handler (`process_batch_result`, extend branch) | Read `bypass_chunked_req`, clear `self.chunked_req` if set | 2026-05-13 (P2a.2 T2.1) |

---

## tt-metal upstream-bug patches (P2a.1 hardware-smoke discoveries, 2026-05-13)

| Path | Issue | Submit upstream? |
|---|---|---|
| `tt-metal/models/tt_transformers/tt/generator_sglang.py` (lines 155, 198, 241, 305) | `super().decode_forward_text()` doesn't exist; correct method is `decode_forward`. Affects every SGLang user of the plugin path. | **YES** — file Tenstorrent issue + PR |
| `tt-metal/models/common/llama_models.py` (line 12) | Hard import of `AutoModelForVision2Seq` (removed in transformers 5.x). | Yes — soft-import |
| `tt-metal/models/tt_transformers/tt/model_config.py` (lines 2693, 2967) | Same hard-import issue; also `rope_theta` lost from `LlamaConfig.to_dict()` in transformers 5.x. | Yes — `rope_theta` resolution via getattr |

Local patch files at `python/sglang/srt/hardware_backend/tenstorrent/scripts/tt_metal_patches/`.

---

## SGLang speculative-worker patches (P3a.0 T0.2, R-P3-3 HIGH)

| File | Lines | Purpose | First landed |
|---|---|---|---|
| `python/sglang/srt/speculative/ngram_worker.py` | 48, 229, 231 | device-agnostic NGRAM worker | 2026-05-13 (P3a.0 T0.2) |
| `python/sglang/srt/speculative/eagle_info.py` | 102, 103, 104, 106, 109, 112 + 685 comment | device dispatch from worker | 2026-05-13 (P3a.0 T0.2) |
| `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` | 325 | multi-layer EAGLE device dispatch | 2026-05-13 (P3a.0 T0.2) |
| `python/sglang/srt/speculative/spec_utils.py` | 587 | sim_accept_index device dispatch | 2026-05-13 (P3a.0 T0.2) |
| `python/sglang/srt/server_args.py` | 3753-3762 | NGRAM CUDA-gate widened to allow `current_platform.device_name == "tenstorrent"` | 2026-05-13 (P3a.1 T1.3) |
| `python/sglang/srt/speculative/ngram_worker.py` | 6 | `sgl_kernel.speculative.reconstruct_indices_from_tree_mask` import made soft with pure-Python CPU fallback (port of `csrc/speculative/ngram_utils.cu`) | 2026-05-13 (P3a.1 T1.3) |
| `python/sglang/srt/speculative/ngram_info.py` | 39-47 | `sgl_kernel.verify_tree_greedy` import added an `else` branch with a pure-Python CPU fallback (port of `csrc/speculative/eagle_utils.cu::VerifyTreeGreedy`) for non-CUDA/MUSA/HIP platforms | 2026-05-13 (P3a.1 T1.3) |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | 409-426 | Out-of-tree `current_platform.get_mha_kv_pool_cls()` branch now also passes `enable_kv_cache_copy=(speculative_algorithm is not None)`, matching the in-tree MHATokenToKVPool branch; otherwise NGRAM `move_kv_cache` asserts | 2026-05-13 (P3a.1 T1.3) |
| `python/sglang/srt/speculative/draft_utils.py` | 42-65, ~95 | Added `"torch_native"` to EAGLE decode + extend backend maps, routing to `TTMultiStepDraftBackend` (see `hardware_backend/tenstorrent/tt_eagle_backend.py`) | 2026-05-13 (P3a.2 T2.2) |
