# SGLang-on-Tenstorrent — Phase 3b Design Spec

**Date:** 2026-05-16
**Duration:** 8+ weeks (10 weeks budgeted to absorb slippage)
**Branch:** `tenstorrent-p1` on `origin` (`predator2k/sglang`)
**Hardware:** 2× Tenstorrent Blackhole P150a
**Predecessor:** P3a (speculative decoding + correctness eval, finishing)

---

## 1. Goals

P3b is a dual-track phase:

1. **Close the performance gap** — Rebase the tt-metal fork to a modern base (target: the same commit TT vLLM uses, currently `e867533`; or newer if stable) so BFP4 precision works correctly for Qwen3-8B. Target: match or beat TT vLLM's 32.6 tok/s. Pin the exact commit at rebase time after verifying firmware and KMD compatibility.
2. **Add tt-xla as a second execution backend** — Enable broader model coverage without per-model hand-ports. Any PyTorch model compilable via tt-xla becomes servable through SGLang.

Both backends coexist. User selects explicitly via `SGLANG_TT_EXECUTION_BACKEND`.

---

## 2. Current State (P3a exit)

| Metric | Value |
|--------|-------|
| SGLang Qwen3-8B (BFP4+HIFI2) | 26.9 tok/s, correct output |
| SGLang Qwen3-8B (full HIFI) | 24.9 tok/s, correct output |
| SGLang Qwen3-8B (BFP4+LOFI) | 28.0 tok/s, chat/thinking degrades |
| TT vLLM Qwen3-8B (BFP4+LOFI) | 32.6 tok/s, correct output |
| SGLang Qwen3-1.7B | 46.9 tok/s |
| EAGLE-3 spec decoding | 9.8 tok/s (slower than no-spec) |
| GSM8K accuracy (73 samples) | 78.1% (model card ~80%) |
| SGLang overhead | 1.9% (0.4ms per token) |
| Supported models (tt_transformers) | Llama, Qwen3, Mistral, GPT-OSS |
| tt-xla status | Not attempted on this hardware |

**Root cause of perf gap:** Our tt-metal fork (`89686ee78d`) has a Blackhole distributed_norm workaround (force_unsharded -> DRAM round-trip) that degrades BFP4 output quality. TT's newer tt-metal (`e867533`) handles this natively. The fix requires rebasing the fork and rebuilding the container image with matched firmware.

---

## 3. Architecture

### 3.1 Execution Backend Selection

```
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged  (default, current)
SGLANG_TT_EXECUTION_BACKEND=tt_xla                 (new in P3b)
```

User chooses explicitly. No auto-detection or fallback.

### 3.2 Two Integration Patterns

The codebase has two distinct integration patterns. Understanding both is critical for tt-xla design.

**Pattern A: ModelRegistry path (tt_transformers_paged, the production default)**
- `tp_worker.py` detects `tt_transformers_paged` and calls `super()._init_model_runner()` 
- SGLang's standard `ModelRunner` loads a class from `ModelRegistry` (e.g., `TenstorrentQwenForCausalLM`)
- That class (`TTModels` in `tt_llm.py`) owns the tt_transformers Generator, mesh device, and forward logic
- The `TTExecutionBackend` ABC is NOT used in this path

**Pattern B: ABC path (tt_transformers_single, legacy B=1)**
- `tp_worker.py` detects non-paged mode and calls `_init_model_runner_simple()`
- Opens a `MeshDeviceCtx`, passes `mesh_device` to a `TTExecutionBackend` subclass
- The backend owns model loading and forward dispatch
- Currently only used for the legacy single-request path

**tt-xla follows Pattern A (ModelRegistry).** Rationale: Pattern A is the production path with full SGLang integration (batching, scheduling, sampling). Pattern B is legacy and limited to greedy sampling. The tt-xla wrapper class registers in `ModelRegistry` just like the tt_transformers classes.

```python
class TenstorrentXLAGenericCausalLM(nn.Module):
    """Generic wrapper: loads any HF CausalLM, compiles via torch_xla.
    Registered in ModelRegistry. Follows the same interface as TTModels:
    __init__(config, ...) and forward(forward_batch) -> LogitsProcessorOutput.
    """
```

### 3.3 Model Registry for tt-xla

The current `registry.py` maps HF architecture names to TT-specific model classes (`TenstorrentLlamaForCausalLM`, etc.). For tt-xla, a single generic wrapper class handles any HF model.

When `SGLANG_TT_EXECUTION_BACKEND=tt_xla`, `register_tt_models()` registers `TenstorrentXLAGenericCausalLM` for a list of supported HF architectures (e.g., `LlamaForCausalLM`, `Qwen2ForCausalLM`, `PhiForCausalLM`, `GemmaForCausalLM`, etc.). This is an explicit list, not a wildcard — adding new architectures requires adding them to the list. This avoids modifying SGLang core's `ModelRegistry` (per standing instruction N9). The list is maintained in `registry.py` alongside the existing tt_transformers registrations.

### 3.4 tt-xla Device Management

tt-xla uses the PJRT interface, which manages its own device initialization separate from ttnn's `MeshDevice`. Key implications:
- tt-xla and tt_transformers CANNOT share the same `mesh_device` — they use different device APIs
- Only one backend active per server process (enforced by existing `SGLANG_TT_EXECUTION_BACKEND` switch)
- `tp_worker.py` dispatch: both `tt_transformers_paged` and `tt_xla` use Pattern A (`super()._init_model_runner()`). The difference is purely which model class `ModelRegistry` resolves — `TenstorrentQwenForCausalLM` (tt_transformers) vs `TenstorrentXLAGenericCausalLM` (tt-xla). No new dispatch branch needed in `tp_worker.py` for tt-xla; the existing `_paged_mode = True` path works for both. The `_paged_mode` flag is renamed to `_uses_model_registry` for clarity.
  - `tt_transformers_paged` / `tt_xla` → Pattern A (`super()._init_model_runner()`, mesh/device init inside model class)
  - `tt_transformers_single` → Pattern B (legacy ABC path, `MeshDeviceCtx` + `TTExecutionBackend`)
- Shared utilities in `tt_utils.py` are limited to: HF token setup, thread limits, device visibility env vars

### 3.5 File Map

**Modified files:**
| Path | Change |
|------|--------|
| `models/registry.py` | Add tt-xla architecture list registration |
| `models/tt_utils.py` | Extract shared device utilities (HF token, thread limits) from `BaseMetalDeviceRunner` |
| `tp_worker.py` | Rename `_paged_mode` to `_uses_model_registry`; include `tt_xla` in the Pattern A branch |
| `scripts/bootstrap_container.sh` | Add `TT_METAL_IMAGE_TAG` env var support (currently hardcoded to `dev`) |

**New files:**
| Path | Responsibility |
|------|---------------|
| `models/tt_xla_model.py` | `TenstorrentXLAGenericCausalLM` — HF model wrapper compiled via torch_xla |
| `scripts/benchmark_dual_backend.py` | Head-to-head comparison script |
| `test/test_tt_xla_smoke.py` | tt-xla basic inference test |
| `test/test_tt_xla_serving.py` | tt-xla end-to-end serving test |

**Removed files:**
| Path | Reason |
|------|--------|
| `execution/tt_xla_backend.py` | Dead code — tt-xla uses Pattern A (ModelRegistry), not Pattern B (ABC) |

---

## 4. Track 1: tt-metal Rebase (Weeks 1-4)

### 4.1 Pre-rebase Validation

Before rebasing, verify the target commit's compatibility:
1. Check target commit's KMD version requirement against host KMD 2.8.0
2. Check firmware bundle version compatibility with host FW 19.6.0
3. If either requires an update, budget the update as a sub-task (1-2 days)

### 4.2 Patch Audit

Our fork has ~20 commits on `89686ee78d` (12 base commits from P1/P2 + 8 P3a-era additions including prefetcher scaffolding, model_config entries, and EAGLE tree-mask flag). Additionally, 4 legacy patch files in `scripts/tt_metal_patches/` are applied at container startup (these overlap with the base commits in the fork image). For each commit/patch, determine:
- **Drop** — upstream fixed the same issue
- **Cherry-pick** — still needed (SGLang-specific integration code)
- **Adapt** — needed but conflicts with upstream changes

Known patches likely still needed:
- `generator_sglang.py` modifications (use_prefetcher param)
- `model_config.py` Qwen3 MAX_PREFILL_CHUNK_SIZE entries
- `attention.py` `_skip_self_attention` flag for EAGLE tree-mask

Known patches likely droppable:
- `distributed_norm.py` force_unsharded Blackhole workaround
- Grid-y clamp workarounds (upstream Blackhole support improved)
- DRAM mem-config fallbacks (upstream handles natively)

### 4.3 Container Image Build

Build a new container image from the rebased source:
1. Full tt-metal C++ compile (1013 targets, ~10 min on this machine)
2. Firmware compiled from same source (no pre-compiled mismatch)
3. SGLang + dependencies pre-installed
4. Tag as `localhost/local-tt-metal:p3b`
5. **Preserve old image** `localhost/local-tt-metal:dev` as rollback target

### 4.4 Validation

- BFP4+LOFI correct output: completions, chat, thinking mode
- Throughput >= 32 tok/s (matching TT vLLM)
- GSM8K accuracy >= 78% (matching P3a baseline)
- All existing P2/P3a tests pass
- **EAGLE cohost validation**: verify EAGLE-3 still boots and produces output on the rebased base (mesh sharing, aux-layer capture, verify loop)
- DRAM prefetcher re-attempt (L1 CB clash may be fixed in newer base)

### 4.5 Rollback Plan

If the rebase breaks fundamental functionality (paged attention, EAGLE cohost, model loading):
1. Revert to `localhost/local-tt-metal:dev` image (P3a container)
2. `bootstrap_container.sh` keeps both image tags; env var `TT_METAL_IMAGE_TAG` selects which one
3. Debugging continues on the rebased branch without blocking production use on the old image

### 4.6 Exit Criteria

SGLang Qwen3-8B BFP4+LOFI >= 32 tok/s with correct output across all endpoints. EAGLE cohost functional.

---

## 5. Track 2: tt-xla Bring-up (Weeks 5-7)

### 5.1 Discovery (Week 5)

**Go/no-go gate.** Before any integration work:

1. Install tt-xla in a container (may need its own image or a layer on top of the rebased one)
2. Verify tt-xla + ttnn can coexist in the same Python environment (version conflicts between PyTorch, torch_xla, and ttnn are common). If not, tt-xla gets a separate venv or container.
3. Run tt-xla test suite / examples on P150a — does it initialize?
4. Run a basic model forward pass (GPT-2 or Llama-3.2-1B) through tt-xla
5. Measure: correct output? Throughput? First-compilation latency?
6. Document findings in evidence file
7. **Produce a time estimate** for integration (weeks 6-7) based on discovery findings

**If tt-xla fails basic bring-up on Blackhole:** Pivot remaining weeks to alternative work — more tt_transformers model coverage, batched throughput optimization, or EAGLE spec decoding improvements.

### 5.2 Integration (Week 6)

Implement `models/tt_xla_model.py` (new file):
- `TenstorrentXLAGenericCausalLM(nn.Module)` wraps a standard HuggingFace model
- Follows Pattern A: registered in `ModelRegistry`, loaded by SGLang's `ModelRunner`
- Same interface as `TTModels`: `__init__(config, ...)` and `forward(forward_batch) -> LogitsProcessorOutput`
- `forward()` returns logits (not sampled tokens) — SGLang's sampler handles sampling
- Compiles via `torch_xla` PJRT interface to TT device
- No paged attention — simple contiguous KV cache with fixed max sequence length
- PJRT device init happens inside `__init__`, not in `tp_worker.py`
- **First-request compilation latency**: document expected wait time; consider AOT compilation or graph caching if torch_xla supports it
- The existing `execution/tt_xla_backend.py` stub (Pattern B / ABC) becomes dead code and is removed

**Known limitations:**
- Without paged attention, tt-xla serving has no cache reuse, no prefix sharing, and fixed max sequence length allocation. Multi-turn chat and batch sizes >1 will be memory-inefficient. Paged attention via tt-xla is a P3c concern.
- tt-xla follows Pattern A (ModelRegistry path) which supports full SGLang sampling (temperature, top-k/p, etc.). The legacy Pattern B (ABC simple path) is greedy-only and is NOT used for tt-xla.

### 5.3 First Model Serving (Week 7)

Pick a model NOT in tt_transformers as the proof case:
- Primary candidate: **Phi-4** (14B) — verify it fits in 2×P150a DRAM first (weights: 14B × 2 bytes BF16 = ~28 GB; contiguous KV cache at 2048 seq_len adds several GB; activations ~2 GB; total ~35 GB against ~56 GB available — tight but plausible). If memory is insufficient, reduce max_seq_len or switch to fallback.
- Fallback: **Gemma-3-4B** (smaller, fits easily)

Serve end-to-end via SGLang with tt-xla backend. Verify correct output, basic throughput, no crashes over 10 minutes.

### 5.4 Exit Criteria

One non-tt_transformers model serving correctly via SGLang + tt-xla on P150a. Throughput >= 5 tok/s (minimum viable, not competitive — just proves the path works).

---

## 6. Track 3: Model Coverage + Tuning (Weeks 8-10)

### 6.1 Required Deliverables (Weeks 8-9)

These are required for P3b exit:

1. **Cross-backend comparison**: run Qwen3-8B on both tt_transformers and tt-xla, document perf/quality tradeoff
2. **Model compatibility matrix**: table of model × backend × device -> throughput, correctness, status
3. **6-hour soak test** on tt_transformers backend (schedule explicitly: 6h wall-clock + 2h for debugging)

### 6.2 Stretch Goals (Week 10+)

These improve the product but are NOT required for P3b exit:

- Second tt-xla model (beyond the one required in Track 2)
- 6-hour soak test on tt-xla backend
- DRAM prefetcher on rebased tt-metal
- EAGLE spec decoding with BFP4
- Batched throughput (batch=8/32) on both backends
- tt-xla graph caching / AOT compilation for faster cold start
- Third+ tt-xla model

### 6.3 tt-xla Model Candidates

Priority order:
1. **Phi-4** (14B) — no tt_transformers support
2. **Gemma-3-4B** — not validated on our Blackhole fork
3. **DeepSeek-R1-Distill variants** — if memory allows
4. **Qwen3-8B via tt-xla** — cross-backend comparison (required deliverable)

For each: verify correctness, measure throughput, document.

### 6.4 Deliverables

| Deliverable | Description |
|-------------|-------------|
| Model compatibility matrix | model × backend × device -> throughput, correctness, status |
| `execution/tt_xla_backend.py` | Full implementation |
| `models/registry.py` | tt-xla fallback registration |
| `scripts/benchmark_dual_backend.py` | Head-to-head comparison |
| Evidence files | Per-model correctness + throughput JSON |
| Updated `bootstrap_container.sh` | New image reference + old image preserved |

---

## 7. P3b Exit Criteria

| # | Criterion | Measurable? |
|---|-----------|-------------|
| 1 | tt_transformers Qwen3-8B BFP4+LOFI >= 32 tok/s with correct output (matching vLLM) | Yes — benchmark script |
| 2 | tt-xla serving at least 1 model not in tt_transformers, >= 5 tok/s | Yes — serving test |
| 3 | Both backends selectable via `SGLANG_TT_EXECUTION_BACKEND` env var | Yes — smoke test |
| 4 | Model compatibility matrix published | Yes — file exists |
| 5 | 6-hour soak test on tt_transformers backend without degradation | Yes — throughput drift < 5% |
| 6 | EAGLE cohost functional on rebased tt-metal | Yes — EAGLE smoke test |

**Stretch (not required):** 2nd tt-xla model, tt-xla soak test, EAGLE faster than no-spec, prefetcher working.

---

## 8. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| tt-metal rebase conflicts (20 patches) | HIGH | Audit each patch first; budget 1 week for resolution. Rollback to old image if blocked. |
| Container firmware mismatch | HIGH | Build firmware from source in same image. Verify KMD/FW compat before starting. |
| KMD/firmware version incompatibility | HIGH | Check target commit's requirements against host KMD 2.8.0 / FW 19.6.0 before rebase. Budget 1-2 days for host-side updates if needed. |
| tt-xla doesn't work on Blackhole | HIGH | Go/no-go gate at Week 5; pivot to alternative work if fails. |
| tt-xla + ttnn environment conflict | MEDIUM | Test coexistence in Week 5 discovery. Use separate venv/container if needed. |
| tt-xla first-compilation latency (minutes) | MEDIUM | Document cold-start time. Investigate AOT compilation / graph caching. |
| tt-xla op coverage gaps for target models | MEDIUM | Phi-4 fallback to Gemma-3-4B (simpler architecture). |
| EAGLE regression on rebase | MEDIUM | Explicit EAGLE cohost validation in Track 1. Rollback plan preserves P3a state. |
| tt-xla contiguous KV cache limits batch serving | LOW | Known limitation, documented. Paged attention deferred to P3c. |
| Track 3 absorbs all slippage | MEDIUM | Required deliverables separated from stretch goals. Only soak test + matrix + 1 comparison are hard requirements. |

---

## 9. Non-Goals (P3b)

- Upstream PR to sgl-project/sglang (per standing instruction)
- Multi-node / multi-card beyond 2×P150a
- Production SLA / 24h stability certification (deferred to P3c)
- EAGLE spec decoding faster than no-spec (stretch goal, not required)
- EAGLE on tt-xla backend (EAGLE uses tt_transformers-specific tree-mask and aux-layer capture; out of scope for tt-xla)
- tt-xla paged attention (use contiguous KV cache initially)
- REBASE_TARGETS.md update (will be done as part of rebase, not a separate deliverable — existing SGLang fork patches are re-verified during Track 1 patch audit)

---

## 10. Container Image Management

| Tag | Source | Purpose |
|-----|--------|---------|
| `localhost/local-tt-metal:dev` | tt-metal `89686ee78d` + 20 patches | P3a production image. Preserved as rollback. |
| `localhost/local-tt-metal:p3b` | tt-metal `e867533`+ rebased | P3b target image. Becomes default after Track 1 validation passes. |

`bootstrap_container.sh` accepts `TT_METAL_IMAGE_TAG` env var (default: `p3b` after validation, `dev` before). Both images kept until P3c.
