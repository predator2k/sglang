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

## tt-metal upstream-bug patches (P2a.1 hardware-smoke discoveries, 2026-05-13)

| Path | Issue | Submit upstream? |
|---|---|---|
| `tt-metal/models/tt_transformers/tt/generator_sglang.py` (lines 155, 198, 241, 305) | `super().decode_forward_text()` doesn't exist; correct method is `decode_forward`. Affects every SGLang user of the plugin path. | **YES** — file Tenstorrent issue + PR |
| `tt-metal/models/common/llama_models.py` (line 12) | Hard import of `AutoModelForVision2Seq` (removed in transformers 5.x). | Yes — soft-import |
| `tt-metal/models/tt_transformers/tt/model_config.py` (lines 2693, 2967) | Same hard-import issue; also `rope_theta` lost from `LlamaConfig.to_dict()` in transformers 5.x. | Yes — `rope_theta` resolution via getattr |

Local patch files at `python/sglang/srt/hardware_backend/tenstorrent/scripts/tt_metal_patches/`.
