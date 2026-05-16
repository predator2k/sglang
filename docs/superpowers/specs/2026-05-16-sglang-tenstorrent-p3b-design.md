# SGLang-on-Tenstorrent — Phase 3b Design Spec

**Date:** 2026-05-16
**Duration:** 8+ weeks
**Branch:** `tenstorrent-p1` on `origin` (`predator2k/sglang`)
**Hardware:** 2× Tenstorrent Blackhole P150a
**Predecessor:** P3a (speculative decoding + correctness eval, finishing)

---

## 1. Goals

P3b is a dual-track phase:

1. **Close the performance gap** — Rebase the tt-metal fork to a modern base (target: the same commit TT vLLM uses, currently `e867533`; or newer if stable) so BFP4 precision works correctly for Qwen3-8B. Target: match or beat TT vLLM's 32.6 tok/s. Pin the exact commit at rebase time after verifying firmware compatibility.
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

**Root cause of perf gap:** Our tt-metal fork (`89686ee78d`) has a Blackhole distributed_norm workaround (force_unsharded → DRAM round-trip) that degrades BFP4 output quality. TT's newer tt-metal (`e867533`) handles this natively. The fix requires rebasing the fork and rebuilding the container image with matched firmware.

---

## 3. Architecture

### 3.1 Execution Backend Selection

```
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged  (default, current)
SGLANG_TT_EXECUTION_BACKEND=tt_xla                 (new in P3b)
```

User chooses explicitly. No auto-detection or fallback.

### 3.2 Backend Interface

Both backends implement `ExecutionBackend` (defined in `execution/base.py`):

```python
class ExecutionBackend:
    def load_model(config, mesh_device, **kwargs) -> model
    def prefill_forward(tokens, page_table, kv_cache, **kwargs) -> logits
    def decode_forward(tokens, start_pos, page_table, kv_cache, **kwargs) -> logits
    def allocate_kv_cache(shape, dtype, num_layers, model) -> kv_cache
```

### 3.3 File Map

**Modified files:**
| Path | Change |
|------|--------|
| `execution/tt_xla_backend.py` | Full implementation (currently stub) |
| `models/registry.py` | Add tt-xla model registrations |
| `models/tt_utils.py` | Shared device utilities for both backends |
| `scripts/bootstrap_container.sh` | Reference new container image |

**New files:**
| Path | Responsibility |
|------|---------------|
| `scripts/benchmark_dual_backend.py` | Head-to-head comparison script |
| `test/test_tt_xla_smoke.py` | tt-xla basic inference test |
| `test/test_tt_xla_serving.py` | tt-xla end-to-end serving test |

---

## 4. Track 1: tt-metal Rebase (Weeks 1-3)

### 4.1 Patch Audit

Our fork has 20 patches on `89686ee78d`. For each, determine:
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

### 4.2 Container Image Build

Build a new container image from the rebased source:
1. Full tt-metal C++ compile (1013 targets, ~10 min on this machine)
2. Firmware compiled from same source (no pre-compiled mismatch)
3. SGLang + dependencies pre-installed
4. Tag as `localhost/local-tt-metal:p3b`

### 4.3 Validation

- BFP4+LOFI correct output: completions, chat, thinking mode
- Throughput ≥ 32 tok/s (matching TT vLLM)
- GSM8K accuracy ≥ 78% (matching P3a baseline)
- All existing P2/P3a tests pass
- DRAM prefetcher re-attempt (L1 CB clash may be fixed in newer base)

### 4.4 Exit Criteria

SGLang Qwen3-8B BFP4+LOFI ≥ 32 tok/s with correct output across all endpoints.

---

## 5. Track 2: tt-xla Bring-up (Weeks 4-6)

### 5.1 Discovery (Week 4)

**Go/no-go gate.** Before any integration work:

1. Install tt-xla in a container (may need its own image or a layer on top of the rebased one)
2. Run tt-xla test suite / examples on P150a — does it initialize?
3. Run a basic model forward pass (GPT-2 or Llama-3.2-1B) through tt-xla
4. Measure: correct output? Throughput?
5. Document findings in evidence file

**If tt-xla fails basic bring-up on Blackhole:** Pivot remaining weeks to alternative work — more tt_transformers model coverage, batched throughput optimization, or EAGLE spec decoding improvements.

### 5.2 Integration (Week 5)

Implement `execution/tt_xla_backend.py`:
- Wraps a standard HuggingFace model
- Compiles via `torch_xla` PJRT interface to TT device
- No paged attention initially — simple contiguous KV cache
- Key methods: `load_model()`, `prefill_forward()`, `decode_forward()`, `allocate_kv_cache()`

### 5.3 First Model Serving (Week 6)

Pick a model NOT in tt_transformers as the proof case:
- Primary candidate: **Phi-4** (14B)
- Fallback: **Gemma-3-4B**

Serve end-to-end via SGLang with tt-xla backend. Verify correct output, basic throughput, no crashes over 10 minutes.

### 5.4 Exit Criteria

One non-tt_transformers model serving correctly via SGLang + tt-xla on P150a. Throughput documented (no specific target — new path).

---

## 6. Track 3: Model Coverage + Tuning (Weeks 7-8+)

### 6.1 tt-xla Model Expansion

Priority order:
1. **Phi-4** (14B) — no tt_transformers support
2. **Gemma-3-4B** — not validated on our Blackhole fork
3. **DeepSeek-R1-Distill variants** — if memory allows
4. **Qwen3-8B via tt-xla** — cross-backend comparison

For each: verify correctness, measure throughput, document.

### 6.2 Performance Tuning

- **tt_transformers:** Re-attempt DRAM prefetcher on rebased tt-metal. Try EAGLE spec decoding with BFP4 (may work now that BFP4 output is correct).
- **tt-xla:** Profile compilation time, check graph caching, identify hot ops.
- **Batched throughput:** Test both backends at batch=8/32.

### 6.3 Deliverables

| Deliverable | Description |
|-------------|-------------|
| Model compatibility matrix | model × backend × device → throughput, correctness, status |
| `execution/tt_xla_backend.py` | Full implementation |
| `models/registry.py` | tt-xla models registered |
| `scripts/benchmark_dual_backend.py` | Head-to-head comparison |
| Evidence files | Per-model correctness + throughput JSON |
| Updated `bootstrap_container.sh` | New image reference |

---

## 7. P3b Exit Criteria

1. tt_transformers Qwen3-8B BFP4+LOFI ≥ 32 tok/s with correct output (matching vLLM)
2. tt-xla serving at least 2 models not in tt_transformers
3. Both backends selectable via `SGLANG_TT_EXECUTION_BACKEND` env var
4. Model compatibility matrix published
5. 6-hour soak test on each backend without degradation

---

## 8. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| tt-metal rebase conflicts (20 patches) | HIGH | Audit each patch first; budget 1 week for resolution |
| Container firmware mismatch | HIGH | Build firmware from source in same image (proven: C++ build succeeded) |
| tt-xla doesn't work on Blackhole | HIGH | Go/no-go gate at Week 4; pivot to alternative work if fails |
| tt-xla too slow for production | MEDIUM | Acceptable for P3b — perf tuning is ongoing |
| EAGLE spec still slower than no-spec | MEDIUM | Re-attempt after BFP4 fix; may need single-prefill-verify path in tt-metal |

---

## 9. Non-Goals (P3b)

- Upstream PR to sgl-project/sglang (per standing instruction)
- Multi-node / multi-card beyond 2×P150a
- Production SLA / 24h stability certification (deferred to P3c)
- EAGLE spec decoding faster than no-spec (stretch goal, not required)
- tt-xla paged attention (use contiguous KV cache initially)
