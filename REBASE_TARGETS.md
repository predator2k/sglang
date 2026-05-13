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
