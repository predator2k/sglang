# SGLang on Tenstorrent — Phase 3 Implementation Spec (P3a)

> **Date**: 2026-05-13
> **Status**: Draft (brainstorming output, pre-implementation)
> **Scope**: P3a only — first sub-phase of P3 (innovation phase). P3b/P3c/P3d outlined as deferred work.
> **Predecessor**: [`2026-05-12-sglang-tenstorrent-p2-design.md`](2026-05-12-sglang-tenstorrent-p2-design.md). P2 has shipped (34 commits since amendment on `tenstorrent-p1` branch in `predator2k/sglang`; §9.1-§9.14 + §9.b1/§9.b2 all PASS or documented partial; RadixAttention 83% hit rate validated on Blackhole).
> **Target hardware**: 2× Tenstorrent Blackhole p150a (validated). Galaxy mesh + 4× p150a deferred to P3d.

---

## 0. Executive Summary

Phase 3a is the **first sub-phase of P3 — the innovation phase**. P2 shipped a working plugin-absorbed paged path with RadixAttention. P3a turns the corner from "plumbing-verified" to **production-ready + first speculative-decoding deployment**, holding feature breadth (P3b) and exploratory work (P3c) for later sub-phases.

**P3a delivers** (3 acceptance goals):

1. **24h+ stability** on 2× Blackhole with synthetic bimodal workload (decode tok/s drift < 5% across windows)
2. **Production observability** — Prometheus metrics + 1 Grafana dashboard (no alerting in P3a)
3. **Speculative decoding — all three SGLang algorithms** (NGRAM + EAGLE + Adaptive) integrated through plugin path

P3a estimate: **6-8 weeks total** (W1-W8). All-in scope; longer than original "lean" framing because user prioritized full speculative bring-up.

P3b/P3c/P3d are scoped in §A1 — gated on P3a §10 acceptance + EAGLE result evidence.

---

## 1. Goals & Non-goals

### 1.1 P3a Goals (must ship)

| # | Goal | Verification |
|---|---|---|
| **G1** | 24h+ stability on 2× Blackhole p150a with synthetic batched workload (B=4 concurrent; prompt length distribution = 80% uniform random in [1024, 8192], 20% uniform random in [10240, 12288]; max_tokens uniform random in [200, 1000]). ~30-50万 requests across 24h. | **decode tok/s drift < 5%** across 4×6h windows (baseline = W1, tail = W4). No crash / SIGQUIT throughout. |
| **G2** | Production observability: SGLang Prometheus metrics enabled (`--enable-metrics`) + custom Grafana dashboard JSON delivered. Panels: queue depth, batched decode tok/s, RadixCache hit rate, ITL p50/p99, KV pool utilization, per-arch (Llama / Qwen3) breakdown, speculative accept_rate, accept_length. | Dashboard JSON at `python/sglang/srt/hardware_backend/tenstorrent/scripts/grafana_p3a_dashboard.json`; manual screenshot during 24h test recorded in `_fixtures/p3a_grafana_screenshot.png`. |
| **G3** | Speculative decoding — **NGRAM** integrated through plugin path. Config via `--speculative-algorithm NGRAM --speculative-num-steps N`. | ≥ 1.3× decode tok/s improvement vs no-spec baseline on Llama-3.1-8B; cache hit + RadixAttention still operational. |
| **G4** | Speculative decoding — **EAGLE** (draft model cohosted on mesh). Draft model selection deferred to Phase 0 (Q1). | ≥ 2× decode tok/s on appropriate prompts vs no-spec. Acceptance length τ (`accept_length`) ≥ 2.0. |
| **G5** | Speculative decoding — **Adaptive** auto-pick between NGRAM/EAGLE per prompt/context. | Adaptive switch logged; mixed-workload throughput ≥ better of NGRAM-only or EAGLE-only single-mode runs. |

### 1.2 Non-goals (explicitly NOT in P3a — pushed to P3b/P3c/P3d)

| # | Item | Defer to |
|---|---|---|
| N1 | 128K chunked-prefill on Llama | P3b |
| N2 | P300 chunk-size empirical table | P3b |
| N3 | LoRA / Multi-LoRA serving | P3b |
| N4 | Multimodal (vision-text) | P3c |
| N5 | HiRadixCache / disaggregation / hisparse | P3c |
| N6 | tt-xla backend | P3c |
| N7 | BFP8 → BF16 precision study | P3c |
| N8 | 4× p150a / Galaxy mesh (8,4) validation | P3d (hardware-gated) |
| N9 | Performance kernel-level tuning (ttnn kernel rewrites) | P3c (specific bottleneck pursuit) |
| N10 | New model arches (Mistral / GptOss / Gemma) | P3b W0 |
| N11 | Alerting (PagerDuty, SLO breaches) | P3b ops bundle |
| N12 | Full OpenTelemetry tracing | P3+ (out of scope this calendar quarter) |

---

## 2. Architecture

### 2.1 New / modified modules (~600 LoC + 1 dashboard JSON)

All paths relative to `python/sglang/srt/hardware_backend/tenstorrent/`.

| Path | Source | Responsibility | Size |
|---|---|---|---|
| `models/tt_llm.py` extension | modify (P2 shipped) | `_resolve_speculative_path`: read server_args `--speculative-algorithm`, select NGRAM / EAGLE / Adaptive routing. Hook into existing plugin-absorbed forward. | +80 |
| `models/spec_decode.py` | new | `SpecDecodeAdapter` — thin wrapper over plugin's `prefill_forward` + `decode_forward` for speculative verification. NGRAM lookup invoked host-side; EAGLE draft propose+verify; accept/reject bookkeeping. **Does NOT reimplement SGLang's speculative runtime** — only wires Tenstorrent inference into SGLang's existing spec scheduler. | ~400 |
| `models/eagle_draft.py` | new | EAGLE draft model loading + co-mesh placement. Calls `tt_transformers.allocate_kv_cache` for a SECOND smaller model; mesh-shape decision Q1 (see §5.2). | ~150 |
| `scripts/grafana_p3a_dashboard.json` | new | Grafana dashboard JSON. 8 panels: queue depth / batched decode tok/s / RadixCache hit rate / ITL p50+p99 / KV pool util / per-model breakdown / spec accept_rate / spec accept_length. | (data) |
| `scripts/stability_24h_bench.py` | new | 24h workload driver: bimodal prompt distribution, 30-50万 requests, 6h-window throughput collector, drift gate. Logs JSON per-window to `_fixtures/stability_24h_<date>.json`. | ~250 |
| `platform.py` extension | modify | Expose new spec-related env vars (`SGLANG_TT_SPEC_DRAFT_PATH`, `SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT`). | +30 |
| **P3a total** | | | **~910 LoC + 1 JSON** |

### 2.2 Preserved from P2 (no changes in P3a)

- P1 simple backend (T0.7 shipped) — still available via `SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single`
- `models/tt_llm.py` plugin port + Tenstorrent* namespace
- `models/tt_utils.py`, `models/worker_setup.py`, `models/registry.py`
- 3 tt-metal patches in `scripts/tt_metal_patches/`
- All P2 §9 acceptance tests — must continue to PASS in P3a's regression run

### 2.3 SGLang upstream-class patches (in our fork only, per N9)

Existing patches preserved. P3a adds **zero new** upstream-class patches — SGLang's native speculative_algorithm infrastructure is already complete; P3a only needs to wire Tenstorrent inference into it (see INV-8).

### 2.4 Architecture invariants (extend P2 INV-1..INV-7 with INV-8/INV-9)

| ID | Invariant |
|---|---|
| **INV-1..INV-7** | Inherited from P2 spec §2.4 (plugin path uses standard SGLang model registration; P1 simple is dual-track fallback) |
| **INV-8** | Speculative decoding goes through **SGLang's native `--speculative-algorithm` config** (NGRAM / EAGLE / Adaptive). P3a does **NOT** implement a separate verification loop. Tenstorrent layer only ensures (a) SGLang's spec runner can find `TenstorrentLlamaForCausalLM.forward` for the main model AND (b) EAGLE draft model's `forward()` is callable through the same plugin pattern. |
| **INV-9** | EAGLE draft model must **co-host on the same 2× Blackhole mesh** as the main model. Not allowed to require a second hardware mesh. Mesh-shape decision (shared (1,2) full mesh vs split (1,1)+(1,1)) is **deferred to Phase 0 implementation discovery (Q1)** — both options must be evaluated and the chosen one documented in `_fixtures/p3a_eagle_mesh_evidence.txt`. |

### 2.5 Data flow

**Standard paths (unchanged from P2)**:

```
HTTP → SGLang scheduler → ModelRunner.forward
  → TenstorrentLlamaForCausalLM.forward (plugin)
  → tt_model.prefill_forward / decode_forward
  → tt_transformers Generator on 2× Blackhole
```

**Speculative-decode hot path (NEW in P3a)** — driven entirely by SGLang's spec_v2 scheduler:

```
HTTP → SGLang scheduler (speculative runtime)
  for each token to generate:
    A. NGRAM mode:
       SGLang spec runner queries host-side n-gram lookup table
       → propose draft_tokens
       → main TenstorrentLlamaForCausalLM.forward in "verify" mode
       → accept first N matching draft tokens
    B. EAGLE mode:
       SGLang spec runner calls eagle_draft.forward (TT mesh, smaller model)
       → propose draft_tokens via tree
       → main TenstorrentLlamaForCausalLM.forward in "verify" mode
       → tree-aware accept/reject
    C. Adaptive mode:
       Runtime heuristic picks A vs B per request
  → SGLang sampler → next iter
```

Important: SGLang's `verify` mode on a model is a SPECIAL forward call that does batched-attention over the draft-proposed token sequence. The plugin's existing `forward()` (which routes to `tt_model.prefill_forward` or `decode_forward`) **needs to accept verify-mode batches** — extend forward branch logic. This is the main integration work in `spec_decode.py`.

### 2.6 SGLang integration points

1. **`--speculative-algorithm`** server_arg: SGLang reads it and instantiates its spec runner. We do nothing here — already supported upstream.
2. **`--speculative-draft-model-path`** server_arg: points to EAGLE draft. Our `eagle_draft.py` is loaded when this arg is present.
3. **Plugin model's `forward()`** must accept SGLang spec's verify batches — extend `models/tt_llm.py` `forward()` to route based on `forward_batch.spec_info` presence.
4. **Prometheus**: SGLang already exports spec metrics (`spec_accept_length`, etc.) when `--enable-metrics` is set. We confirm the metrics flow + add them to our Grafana panels.

---

## 3. Data Flow

### 3.1 24h stability run pipeline

```
stability_24h_bench.py (host-side)
  for 24 hours:
    generate next request: prompt_len from bimodal {80%[1024,8192], 20%[10240,12288]},
                          max_tokens from uniform [200, 1000]
    POST /v1/completions with B=4 (or queue if 4 in flight)
    record per-request:
      - submission time
      - completion time
      - prompt_tokens, completion_tokens (from response usage)
      - per-token decode time (from server timing if available; else derive)
  every 6h window:
    aggregate decode tok/s
    write JSON: _fixtures/stability_24h_<date>.json
  after 24h:
    compute drift: (window4.tok_s - window1.tok_s) / window1.tok_s
    assert |drift| < 0.05
```

### 3.2 Speculative decoding verify-mode forward

Plugin's existing `forward(input_ids, positions, forward_batch)`:

```python
def forward(self, input_ids, positions, forward_batch, ...):
    if getattr(forward_batch, "spec_info", None) is not None:
        # SGLang spec verify-batch: input_ids has [num_draft_tokens] format,
        # positions has per-draft-token positions, forward_batch.spec_info
        # carries the draft tree structure.
        return self._verify_forward(input_ids, positions, forward_batch)
    elif forward_batch.forward_mode.is_extend():
        # ... existing prefill path ...
    elif forward_batch.forward_mode.is_decode():
        # ... existing decode path ...
```

`_verify_forward` calls plugin's `tt_model.prefill_forward` (with start_pos = current decode position, max_seq_len bumped) — this is functionally a "small prefill" of the draft sequence, returning logits per draft position. SGLang's spec runner consumes these logits to accept/reject.

### 3.3 EAGLE draft model load + cohost

```
Server launch with `--speculative-algorithm EAGLE --speculative-draft-model-path /models/<draft>`:

1. SGLang reads draft_model_path → invokes our models/registry resolution
2. SGLang's ModelRunner builds a SECOND `TenstorrentLlamaForCausalLM` instance for the draft
3. eagle_draft.py decides mesh layout:
   - Option A: same (1,2) mesh shared between main + draft (interleaved kernel scheduling)
   - Option B: split mesh — main on device 0, draft on device 1
4. Both models call BaseMetalDeviceRunner.set_device() — collision detection needed
5. tt_transformers.allocate_kv_cache called twice (one per model)
6. SGLang spec scheduler now has both models loaded + forward()-able
```

The mesh-layout decision is INV-9-bound and answered in Phase 0 Q1.

### 3.4 Per-request lifecycle with speculative

```
Request arrives → scheduler queues
  → batch admission (max_running_requests=4, unchanged from P2)
  → spec runner enters verify loop
    main + draft alternating forwards
    accept tokens, emit
  → cache_finished_req (RadixCache, unchanged from P2)
```

### 3.5 Error recovery (extend P2 §3.7)

P2's `on_chunked_prefill_failure` handler still applies. New error path for speculative:
- If EAGLE draft model crashes mid-run (e.g., OOM) → fall back to NGRAM for that request, log warning
- If verify_forward raises → set req's `finish_reason = abort` per spec §3.7, emit zero-logits sentinel

---

## 4. Tests + Acceptance Gates

### 4.1 Test pyramid

```
Hardware-gated (live mesh + SGLang server, SGLANG_PLATFORM=tenstorrent)
─────────────────────────────────────────────────────────────────────
24h stability / NGRAM smoke / EAGLE smoke / Adaptive switch /
NGRAM tok/s gain / EAGLE accept_length / per-arch metrics
                            ~8 tests

Integration (CPU, mocked mesh + mocked backend)
───────────────────────────────────────────────
forward_batch.spec_info routing / SpecDecodeAdapter contract /
Grafana JSON schema validation
                            ~4 tests

Unit (CPU, no mesh, no SGLang server)
─────────────────────────────────────
NGRAM token proposal math / EAGLE accept-tree math /
Adaptive switch heuristic / dashboard JSON sanity
                            ~6 tests
```

### 4.2 §10 Acceptance Gates (new for P3a)

| # | Gate | Test file |
|---|---|---|
| §10.1 | 24h stability — synthetic bimodal workload, decode tok/s drift < 5% across 4×6h windows | `test_stability_24h_paged.py` |
| §10.2 | Grafana dashboard JSON schema valid (parses + has all 8 panels) | `test_grafana_dashboard.py` |
| §10.3 | NGRAM smoke: `--speculative-algorithm NGRAM` boots, decode succeeds | `test_speculative_ngram_smoke.py` |
| §10.4 | NGRAM perf: ≥ 1.3× decode tok/s vs no-spec on Llama-3.1-8B same workload | `test_speculative_ngram_perf.py` |
| §10.5 | EAGLE smoke: `--speculative-algorithm EAGLE --speculative-draft-model-path` boots, decode succeeds | `test_speculative_eagle_smoke.py` |
| §10.6 | EAGLE perf: `accept_length` (τ) ≥ 2.0 measured over 100 requests | `test_speculative_eagle_perf.py` |
| §10.7 | Adaptive: auto-switch logged; mixed-workload throughput ≥ better of NGRAM-only or EAGLE-only | `test_speculative_adaptive.py` |
| §10.8 | Prometheus metrics exported when `--enable-metrics`: `sglang:spec_accept_length`, `sglang:spec_accept_rate`, `sglang:cache_hit_rate`, `sglang:queue_size` all queryable | `test_prometheus_metrics_p3a.py` |
| §10.9 | P2 §9 regression: rerun the P2a/P2b suite, all gates still PASS | (regression — no new file) |
| §10.10 | EAGLE mesh cohosting: documented mesh layout decision in `_fixtures/p3a_eagle_mesh_evidence.txt` (INV-9) | (doc gate) |

### 4.3 Speculative verify-forward unit tests

`test_spec_decode_routing.py` — mock-based:
- `forward_batch.spec_info = None` → routes to existing extend/decode path (no regression)
- `forward_batch.spec_info = <verify-batch>` → routes to `_verify_forward`
- `_verify_forward` returns logits with shape `[num_draft_tokens, vocab]`

### 4.4 Test workload examples

For §10.1 stability:
```python
# stability_24h_bench.py snippet
import random
def gen_request():
    if random.random() < 0.20:
        prompt_len = random.randint(10240, 12288)  # 20% long
    else:
        prompt_len = random.randint(1024, 8192)    # 80% short
    max_tokens = random.randint(200, 1000)
    return {"prompt": _random_prompt(prompt_len), "max_tokens": max_tokens, "temperature": 0.7}
```

---

## 5. Risks / Open Questions / Migration

### 5.1 Risks (P3a)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R-P3-1 | EAGLE draft cohost on 2× Blackhole has unknown perf/memory profile | **HIGH** | Phase 0 Q1 explicit investigation (~1 week); fallback to NGRAM-only if EAGLE cohost is infeasible (downgrades G4 to "deferred to P3b/P3d") |
| R-P3-2 | 24h stability run discovers memory leak or perf drift > 5% | MEDIUM | Compressed 6h pre-run gate in Phase 1 before committing to full 24h; if drift > 5% emerges, profile + iterate before final 24h |
| R-P3-3 | SGLang spec scheduler assumes CUDA model behavior in some code path → fails on TT plugin path | MEDIUM | Phase 0 Q2 explicit grep for CUDA-specific code in `sglang/srt/speculative/*`; patch any incompatible assumption in-fork (R6 monthly rebase target) |
| R-P3-4 | Speculative `accept_length` measurements vary widely with prompt type → G4 fails on adversarial workload | MEDIUM | Define test fixture with 100 fixed prompts; calibrate threshold (2.0) on 10-prompt subset first |
| R-P3-5 | tt-metal patches (P2 era) need rebasing if we bump image | LOW | `REBASE_TARGETS.md` covers, monthly cadence |
| R-P3-6 | Grafana dashboard panel JSON drifts between Grafana versions | LOW | Pin Grafana version in dashboard JSON metadata; mark version as documentation note |

### 5.2 Open Questions (verified during implementation)

| # | Question | Best guess | When verified |
|---|---|---|---|
| Q1 | EAGLE draft model: which one, and what mesh layout? | Llama-3.2-1B + Llama-3.1-8B official pair, mesh shared (1,2) interleaved | Phase 0 W1 (~1 week explicit investigation; document in `_fixtures/p3a_eagle_mesh_evidence.txt`) |
| Q2 | Does SGLang's `sglang/srt/speculative/spec_v2_*.py` runner have any CUDA-specific code paths that break on TT plugin path? | No (spec runner should be device-agnostic — it dispatches via `model.forward(...)`) | Phase 0 W1 |
| Q3 | Does plugin's `forward()` need a new code path for SGLang's verify-batch shape, or can it route to existing `prefill_forward`? | New path needed (verify-batch has draft-tree structure, not flat tokens) | Phase 0 W1 |
| Q4 | What's the actual `accept_length` baseline on TT — closer to 2.0 (papers) or lower (BFP8 drift may degrade draft alignment with main)? | Lower bound 1.5, upper bound 2.5 | Phase 2 measurement |
| Q5 | Memory budget for cohosting Llama-3.1-8B + Llama-3.2-1B + 4 KV pools? | Should fit (8B + 1B BFP8 = ~10 GB, KV pools another 40 GB on TT, ample) | Phase 0 W1 alongside Q1 |

#### 5.2b Implementation-discovery escape hatch (from P2 spec §5.2b)

If any Q1-Q5 answer **violates** an INV-1..INV-9 invariant from §2.4, STOP implementation and re-enter brainstorming-skill replay on the affected §2 sub-section.

### 5.3 Migration timeline

```
Week  : 1   2   3   4   5   6   7   8
P3a.0 : ████                            (Phase 0: spec runner audit + EAGLE draft + mesh layout Q1-Q5)
P3a.1 :     ████████                    (Phase 1: NGRAM + spec_decode.py + verify path + tests)
P3a.2 :             ████████            (Phase 2: EAGLE draft load + cohost + verify path + EAGLE perf)
P3a.3 :                     ████        (Phase 3: Adaptive + observability + Grafana JSON)
P3a.4 :                         ████    (Phase 4: full 24h stability run + §10 acceptance gate)
```

Each phase has its own verification gate. Total **6-8 weeks** depending on hardware availability and Q1 (EAGLE cohost) complexity.

### 5.3a User-visible transitions

| Event | User impact |
|---|---|
| P3a.1 NGRAM merged | New CLI flag `--speculative-algorithm NGRAM` works on TT; default off |
| P3a.2 EAGLE merged | New CLI flag `--speculative-draft-model-path` works; default off |
| P3a.3 Observability merged | `--enable-metrics` works; Grafana dashboard JSON delivered for ops to install |
| P3a.4 24h stability merged | Documentation note in spec: "production deployment of TT plugin path validated for 24h synthetic workload" |

### 5.3b P3a → P3b decision gate

Proceed to P3b (features wave) iff:
1. §10.1-§10.10 all PASS (or G4 downgraded with documented fallback)
2. R-P3-1 (EAGLE cohost) resolved either as PASS or as documented infeasibility
3. P2 §9 regression PASSes
4. Decision evidence file `_fixtures/p3a_acceptance_report.md` committed

### 5.3c Rollback

If P3a's spec_decode wiring breaks decode of standard (non-speculative) requests:
- `--speculative-algorithm none` (or omit the flag) → SGLang skips spec runtime, uses standard ModelRunner.forward path → P2's behavior fully restored

No code rollback needed — speculative is opt-in via CLI flag.

---

## A1. P3b / P3c / P3d scope (gated on P3a)

### P3b — Features wave (W9-W14, ~5 weeks)

| Item | Notes |
|---|---|
| 128K chunked-prefill on Llama | (was P2 G4b) |
| P300 chunk-size empirical table | (was P2 G5b) |
| LoRA / Multi-LoRA serving | SGLang native; our wiring + per-arch test |
| New model arches (Mistral / GptOss / Gemma if weights available) | (was P2 N11 placeholders) |
| Alerting (PagerDuty hooks, SLO breach) | Bundled with ops layer |

### P3c — Exploration wave (W15-W22, ~7 weeks)

| Item | Notes |
|---|---|
| Multimodal (vision-text) | tt_transformers Llama vision variant; vllm-style adapter pattern |
| HiRadixCache | SGLang native; needs evaluation on TT KV layout |
| Disaggregation | Big architectural shift |
| tt-xla backend | Alternative model coverage path |
| BFP8 → BF16 throughout precision study | Compute cost vs quality tradeoff |
| Performance kernel-level tuning | Specific bottleneck pursuit |

### P3d — Scale wave (gated on hardware availability)

| Item | Notes |
|---|---|
| 4× p150a validation | Need 4 cards |
| Galaxy mesh (8,4) | Need Galaxy hardware |
| 24h+ stability at scale (4× cards) | Builds on P3a §10.1 |

---

## A2. Cross-references

**Predecessors**:
- P1 spec: [`2026-05-11-sglang-tenstorrent-p1-design.md`](2026-05-11-sglang-tenstorrent-p1-design.md)
- P2 spec (amended): [`2026-05-12-sglang-tenstorrent-p2-design.md`](2026-05-12-sglang-tenstorrent-p2-design.md)
- P2a plan v2: [`../plans/2026-05-12-sglang-tenstorrent-p2a-plan.md`](../plans/2026-05-12-sglang-tenstorrent-p2a-plan.md)

**Implementation tooling**:
- `superpowers:writing-plans` skill — produces the P3a implementation plan from this spec
- `superpowers:subagent-driven-development` skill — executes the plan
- Git remote: `predator2k/sglang` (`origin`). NEVER push to `sgl-project/sglang` per spec N9.

**Image pins** (inherited from P2):
- Current build: `localhost/local-tt-metal:dev` (sha256:`973e972bddf5`, tt-metal commit `89686ee7`)
- See `python/sglang/srt/hardware_backend/tenstorrent/scripts/reset_devices.sh` for full pin history

**Upstream SGLang spec code** (read-only for P3a):
- `python/sglang/srt/speculative/spec_v2_*.py` — SGLang's native speculative runtime
- `python/sglang/srt/managers/scheduler.py` — already integrates spec scheduler when `--speculative-algorithm` is set

**P2 carry-forward**:
- 3 tt-metal patches in `scripts/tt_metal_patches/`
- `REBASE_TARGETS.md` at repo root tracks SGLang upstream-class patches
- P1 simple backend (`tt_transformers_single`) preserved as dual-track fallback

---

**End of spec.** P3a delivers stability + observability + speculative-decoding-all-3 in 6-8 weeks. P3b/P3c/P3d gated on §5.3b acceptance.
