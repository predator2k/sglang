# tt-xla TPOT — Full-Stack Optimization Design (v5.1)

Date: 2026-05-17
Status: spec, awaiting approval
Supersedes: prior `docs/platforms/tt_xla_tpot_optimization.md` (still valid as evidence base, this doc extends scope)
Revision history:
- v5.0 (this doc, first cut)
- v5.1: incorporated red-team review + Phase 0.5 source-inspection findings
- v5.2: incorporated final review amendments — pre-warm enumerates all buckets, B.2 guards both S=1 and S>1, v145 regression comparator script, watermark gate must use different prompts
- v5.3: incorporated final-final review fixes — A.5 watermark uses first-shape-seen set (not noisy per-decode log); A.3 adds lockstep assertion; Phase 2c has explicit "if 2b failed" fallback; correctness gate swaps StableLM for Phi-4-mini to catch fused-QKV early; Phase 2b validates `--print-after-all` on ALL correctness archs

## Problem

Per-token latency on tt-xla is dominated by two distinct cost centers:

1. **Per-PJRT-call fixed overhead (~38 ms)**, primarily logits D→H transfer (~47 ms measured)
2. **JIT recompile cliffs (~9 s per shape)** that fire on every request because production `_reset_cache()` calls `torch._dynamo.reset()` between requests

Standalone steady-state TPOT for TinyLlama 2K = **70 ms/tok**. Through SGLang server, median ITL = 70 ms (parity) but mean TPOT = 212 ms because **every request pays both JIT cliffs** (prefill + first-decode = ~18 s wasted per request).

## What probing established

Probe scripts and logs: `python/sglang/srt/hardware_backend/tenstorrent/test/probe_*.py`, `_fixtures/probe_*.log`.

| Lever | Probe | Result | Implication |
|---|---|---|---|
| T2.1 (`TTFunctionalCache`, q_len=1 decode, `torch.where`) | `probe_t2_1.log` | PASS — bit-exact K=1/2/4, 0 graph breaks, 0 recompiles | Removes dynamo.reset requirement |
| A1 K=4 via `torch.where` | `probe_t2_1_perf.log` | 1.07× (vs 3.1× projected) | `torch.where + expand_as` costs ~25 ms / sub-step, offsets PJRT savings |
| `clone + index_copy_` cache | `probe_t2_1_v3.log` | FAIL — `ttir.paged_update_cache` legalization | tt-mlir fix B.2 |
| `slice_scatter` cache | `probe_t2_1_v3.log` | FAIL — recompile-limit, correctness | dead end |
| C2 device-resident inputs | `probe_3way.log` | −40% (slower) | on-device updates are *new* PJRT calls |
| Async `non_blocking=True` | `probe_3way.log` | 0% | tt-xla doesn't pipeline transfers |
| A4-aligned bs=N | `probe_a4_sweep.log` | bs=2 → 1.19×, bs=4 → 1.30×, **bs=8 → 1.40×** | Throughput gain, real |
| Steady-state ITL | `probe_c3_per_token.log` | 70 ms (matches standalone) | Scheduler is free; cliffs are JIT-only |
| Same-prompt R2 ≠ KV-cache hit | `probe_c3_warmup_vs_cache_*` | 9624 ms TTFT on R2 = R1 | radix cache disjoint from HF StaticCache; dynamo.reset wipes JIT |
| SGLang built-in warmup | `server_with_warmup.log` | CRASH — `aten::index_put` dim mismatch | tt-mlir fix B.1 |
| tt-mlir pin freshness (Phase 0.5a) | upstream `eb9005fa..main` | 3 days old, 24 commits, **0** mention paged_update_cache / index_put / index_copy / scatter | Fork is required (no upstream rescue) |
| tt-mlir B.2 root cause (Phase 0.5b) | source-trace | `CacheFillUpdatePattern` fires → canonicalizes to `paged_update_cache` → legalization requires cache to have exactly 1 user → fails for autoregressive (cache also read by attention) | Fix is local + small |

## Scope — two parallel workstreams

Each can ship independently; combined, they target 3× TPOT under realistic load.

### Workstream A — SGLang / tt-xla Python

A.1. **T2.1: replace `StaticCache` with `TTFunctionalCache` in `tt_xla_model.py`**, delete `torch._dynamo.reset()` from `_reset_cache()`. Decode q_len=1 uses `torch.where`; prefill q_len>1 falls through to `StaticCache.update` parent. **Removes the per-request JIT cliff** (~18 s per request at scale).

A.2. **tt-xla-specific server pre-warm hook** at model init: enumerate **every** `_get_pad_bucket` value (per `tt_xla_model.py:192-204`) and compile a dummy prefill at each, plus one dummy decode. Pre-compiles all shapes a real request can hit so cliffs don't fire mid-stream when a 1500-tok or 800-tok request arrives at a non-max bucket. Must use shapes/ops that tt-mlir supports (sidestep B.1 below).

A.3. **A4-aligned bs=8 batch support**: lift `max_batch_size=1` at `tt_xla_model.py:142`, scale `_full_attn_mask` shape, document the *lockstep* invariant (all sequences share `cache_position`). Gives 1.40× throughput.
    *Note: this is benchmark-shape only — real continuous batching with mixed cache_positions requires paged KV layout and is out of scope.*
    **Lockstep assertion (required)**: add `assert forward_batch.seq_lens.eq(forward_batch.seq_lens[0]).all()` (or equivalent) in `_forward_decode` when `bs > 1`. Without this guard, SGLang's continuous batcher will silently mix sequences at different decode steps in one batch, producing wrong logits (the shared `cache_position` writes the wrong KV slot for N-1 sequences).

A.4. **Soft-import patches** for missing-dep robustness: `from torchvision.io import decode_jpeg` → `try/except ImportError` in `sglang/srt/utils/common.py`. Behavior-preserving per memory `feedback_no_sglang_core_changes`.

A.5. **Observability — two watermark log lines, gated correctly**:

- **`[TT-XLA] WATERMARK dynamo_reset from <site>`** — a *stale-path detector*. Since A.1 *removes* the `torch._dynamo.reset()` call from `_reset_cache()`, this watermark should never fire. If it ever appears, a regression has re-introduced the call. Acceptance: **0 occurrences across all 10 requests**.

- **`[TT-XLA] WATERMARK first_shape_seen kind=<prefill|decode> shape=(...)`** — fires only when a `(kind, shape)` is encountered for the first time in this process. Implementation: maintain a `set[tuple]` of seen `(kind, padded_input_len)` keys; on miss, log + add. This is the "new shape = potential JIT compile" gate.
  Acceptance: **after server startup pre-warm (A.2) completes, ≤ 0 new shapes across 10 sequential requests with different prompts**. Different prompts is critical — same-prompt exercises only the seen shape and hides recompile bugs.

(Both gates replace the prior in-process `torch._dynamo.utils.counters` gate, which fails when SGLang spawns the model in a subprocess.)

### Workstream B — tt-mlir local fork (`tt-mlir-sglang`)

Maintenance: **local fork now, upstream later**, matching existing `tt-metal-sglang` pattern. Branch `tenstorrent-p1`. Base commit `eb9005fa360a80e44607e2dfd4404137b510092e` (per `tt-xla/third_party/CMakeLists.txt:8`, dated 2026-05-13).

Setup:
- Clone tt-mlir into `/home/mhnie/tt-mlir-sglang/`
- Build env: needs cmake + clang + LLVM toolchain — slim container lacks these; use a separate dev container (`ghcr.io/tenstorrent/tt-xla:latest` non-slim, or homebrew dev container)
- Configure tt-xla build via `TTMLIR_SOURCE_DIR_OVERRIDE` (see `tt-xla/third_party/CMakeLists.txt:32-37`)

Blockers, in priority order:

#### B.2 — `clone + index_copy_` lowering: **the main lever** *(3-7 days)*

**Failure mechanism (now precisely traced, Phase 0.5b)**:

```
stablehlo.scatter                                       ← from clone + index_copy_
  ↓ CacheFillUpdatePattern (StableHLOToTTIRPatterns.cpp:6081, priority=2)
ttir.update_cache
  ↓ UpdateCacheOp::getCanonicalizationPatterns (TTIROps.cpp:5707-5751, auto)
ttir.paged_update_cache
  ↓ PagedUpdateCacheOpConversionPattern (TTIRToTTNN.cpp:735-758)
ttnn.paged_update_cache     ✗ "cache argument must have exactly one user"
                              fails because autoregressive: cache is also read
                              by attention → 2 users
```

A parallel path exists and works: `stablehlo.scatter` → `StableHLOToTTIRScatterOpConversionPattern` (line 6423) → `ttir.scatter` → `ttnn.scatter` (real, working). The problem is `CacheFillUpdatePattern` claims our scatter first (higher benefit) before the generic pattern can.

**Fix (Option A, recommended)**: Add a guard in `CacheFillUpdatePattern::matchAndRewrite` at approximately line 6116 in `StableHLOToTTIRPatterns.cpp` (after the existing `getCacheUpdatePositions` early-out) — if `scatterOp.getInputs()[0]` has >1 user, return `failure()`. The generic scatter pattern at line 6422 then handles it via `ttnn.scatter`. ~5 lines of code.

**Important: guard BOTH S=1 (decode) AND S>1 (prefill) branches** in this pattern. The pattern dispatches to `FillCacheOp` (prefill) or `UpdateCacheOp` (decode), and `FillCacheOpConversionPattern` at `TTIRToTTNN.cpp:828` has the **same "exactly one user" check** as `UpdateCacheOpConversionPattern`. If we only fix decode and leave prefill going through `FillCacheOp`, prefill in `clone()`-style flows would still fail. The single user-count guard placed before the S=1 vs S>1 split covers both.

Why this is safe:
- Preserves the cache-op optimization for in-place use (HF `StaticCache` write-only buffer = exactly 1 user)
- Routes autoregressive scatter (cache written *and* read) through the generic op that works
- `ttnn.scatter` has a 256-element scatter-axis hardware limit; our scatter index has shape `[1]` (decode) or `[seq_len]` (prefill, up to 2048). The TTNN workaround pass (`ScatterOpRewritePattern`) chunks above 256, so prefill is also covered.

**Validation**: re-run `probe_t2_1_v3_writeops.py` V2 (clone+index_copy_); must produce bit-exact tokens and not crash. Then **`--print-after-all` on the lowered IR**: confirm `ttnn.scatter` appears for the cache update and no `paged_update_cache` op remains anywhere. (Watch for `StableHLOToTTIREmbeddingBackwardOpConversionPattern` at line 6315, also benefit=2, potentially claiming the scatter on some shape.)

#### B.1 — `aten::index_put` dim mismatch *(3-7 days)*

**Symptom**: `RuntimeError: Error while lowering: aten::index_put, xla_shape=bf16[1,4,2624,64]; Input dimension should be either 1 or equal to the output dimension; the 2th operand dimension is 2, the 2th output dimension is 1` (crashes SGLang built-in warmup).

**Where**: likely in a `stablehlo.scatter` (or `dynamic_update_slice`) lowering pattern that doesn't handle the index broadcast shape SGLang's warmup uses.

**Risk noted by reviewer**: B.1 fix may expose a B.2-class failure deeper in the same code path. Order: do B.2 first, *then* B.1 (so when B.1 succeeds, downstream lowering already works).

**Win**: unblocks SGLang's built-in warmup path — small but real (alternate path to attack JIT cliffs without writing our own A.2 hook).

#### B.3 — `torch.where + expand_as` kernel speed *(probably moot)*

If B.2 lands and our functional cache uses `clone + index_copy_` instead of `torch.where`, the ~25 ms / sub-step overhead disappears. B.3 only matters if B.2 stalls or doesn't fully solve A1. Defer; revisit only if A1 K=4 perf is still >50 ms / tok after B.2.

#### B.4 — Anything else found during implementation

Reserve scope. Likely candidates: prefill-bucket-related lowering, attention-mask edge cases, sampler kernel, Phi-3+ fused-QKV scatter shape (per reviewer risk #3 — scatter shape may differ from Llama because of `[1, 3*n_heads, 1, d]` vs `[1, n_heads, 1, d]`; B.2 pattern may need to match more shapes).

## Phasing

Parallel tracks, weekly checkpoints.

| Phase | WS | Work | Days | Gate |
|---|---|---|---|---|
| **0 ✓** | A + B | Probes + spec | done | this doc approved |
| **0.5 ✓** | B | Stale-pin + source inspection | done | confirmed B.2 path |
| 1a | A | T2.1: `TTFunctionalCache` swap + delete `dynamo.reset()` + soft-imports + watermark logging | 2-3 | bit-exact 5 archs; 10 sequential requests log 0 `dynamo_reset` watermarks |
| 1b | B | tt-mlir-sglang fork + dev build env + `TTMLIR_SOURCE_DIR_OVERRIDE` | 1-2 | `make tt-xla` builds against local tt-mlir |
| 2a | A | Server pre-warm hook at startup | 1 | `bench_serving --num-prompts 10`: TPOT mean ≈ steady, no per-request cliff watermarks |
| 2b | B | B.2 fix (Option A guard in `CacheFillUpdatePattern`) | 3-7 | `probe_t2_1_v3_writeops.py` V2 PASS, bit-exact, **`--print-after-all` shows `ttnn.scatter` on ALL 5 correctness archs (not just TinyLlama), zero residual `paged_update_cache` or `embedding_backward` ops** |
| 2c | A | If 2b PASSED: re-run A1 K=4 with clone+index_copy_ cache. If 2b FAILED: skip 2c, target ~75 ms / tok (A.1+A.2 only). | 1 | TPOT < 50 ms/tok ⇒ A1 ships; ≥ 50 ms ⇒ defer A1, accept A.1+A.2+A.3 alone |
| 3a | B | B.1 fix (`aten::index_put` lowering) | 3-7 | SGLang built-in warmup runs without crash |
| 3b | A | A4-aligned bs=8 wiring | 1 | bs=8 throughput ≥ 1.35× of bs=1 |
| 4 | B | B.3 only if A1 still slow; B.4 as discovered | open | continued improvements |
| 5 | A | 7-arch sweep + write `regression_check_v145.py` (≤ 30 LOC, loads fixture, re-runs `bench_ttxla_tpot_all_models.py`, asserts < 10% per-model TPOT delta) + run it | 2 | all 16 models PASS; comparator script passes |

Total realistic: **2-3 weeks** if B.2 is straightforward; **3-4 weeks** with buffer.

## Acceptance criteria

### Performance gates
| Metric | Status quo | Target | How measured |
|---|---|---|---|
| TPOT mean (TinyLlama 2K, 64 out) | 212 ms | **≤ 75 ms** (no A1) / **≤ 45 ms** (with A1) | `probe_c3_warmup_vs_cache.sh` Pass 2 |
| TPOT p95 | unmeasured | **≤ 100 ms** (no A1) / **≤ 60 ms** (with A1) | bench_serving with `--num-prompts 10` |
| TPOT p99 | 8990 ms (JIT cliff) | **≤ 100 ms** (no A1) / **≤ 60 ms** (with A1) | same |
| bs=8 throughput | 16.9 tok/s | ≥ 23 tok/s (1.4×) | `probe_a4_bs_sweep` |
| Mid-decode recompile rate | 1+ per request | 0 across 10 requests | grep server.log for `dynamo_reset` watermark |
| HBM watermark | unmeasured | stable across 100 sequential tokens | torch_xla mem trace; ≤ 5 % growth |

### Correctness gates
- Bit-exact greedy match against `StaticCache` reference for first 20 tokens on each of 5 reference archs (TinyLlama, Llama-3.2-1B, Qwen3-1.7B, Mistral-7B, **Phi-4-mini** — replaces StableLM-2-1.6B to surface fused-QKV scatter-shape divergence early, before Phase 5)
- `bench_ttxla_tpot_all_models.py` PASS for all 16 models
- **`v145_ttxla_tpot_all_models.json` regression check** — re-run, compare to fixture, all PASS, no model regressing >10 % TPOT
- Multi-request: 10 sequential requests, server.log shows 0 `dynamo_reset` watermarks after R1

### Build / fork gates
- tt-mlir-sglang at `/home/mhnie/tt-mlir-sglang/` on branch `tenstorrent-p1`
- ≥ 1 patch landed (B.2 minimum, B.1 optional)
- tt-xla builds against local override
- Each patch upstream'd as PR opportunistically (non-blocking)

## Outcome table (revised after Phase 0.5)

| Scenario | TPOT mean (est.) | Reasoning |
|---|---|---|
| Status quo | 212 ms | every request pays both JIT cliffs |
| T2.1 only (A.1) | ~90-130 ms | dynamo.reset gone; R1 still pays cliff; mid-decode shape changes still might |
| T2.1 + pre-warm (A.1 + A.2) | **~75 ms** | both cliffs gone at startup → matches standalone steady-state |
| + B.2 + A1 K=4 (clone+index_copy_) | **~35-45 ms** | PJRT amortized 4×; cheap KV update |
| + B.3 (torch.where kernel) | ~30-35 ms | only if B.2 doesn't already solve A1 |
| + A4 bs=8 on top of above | throughput 1.4× of the above | latency-throughput tradeoff |

## Risks (with mitigations)

| # | Risk | Mitigation |
|---|---|---|
| 1 | B.2 fix may have edge cases (Phi-3+ fused QKV produces different scatter shape) | B.2 pattern must match by scatter signature, not strict shape; B.4 reserve absorbs |
| 2 | tt-mlir rebase debt | Periodic (~monthly) rebase against upstream; pin tracked in spec |
| 3 | B.1 may expose deeper B.2-class failure | Order: do B.2 first; if B.1 then exposes a new pattern, that's B.4 |
| 4 | Phi-3+ uses fused QKV → feature-detect via `hasattr(layer.self_attn, 'qkv_proj')` | Already in A.1 plan; runtime guard fall-back to StaticCache |
| 5 | A4 lockstep doesn't match real serving | Document as benchmark-only; continuous batching out of scope |
| 6 | tt-mlir fork rebuilds pjrt plugin → v145 regression possible | `v145` regression gate before each rebase; rollback via `git revert` on tt-mlir-sglang branch |
| 7 | dev container setup is more work than estimated | Phase 1b budgeted 1-2 days (not half-day) |
| 8 | TTFunctionalCache list-rebind: long-run HBM growth | Acceptance gate: HBM stable across 100 sequential tokens |
| 9 | Phasing 3b ambiguity (what if A1 lands at 50 ms?) | Explicit gate: A1 ships if < 50 ms/tok; defer otherwise |

## Out of scope

- **Continuous batching with mixed cache_position** — paged KV layout; separate effort
- **KV cache hit via radix prefix matching** — would require bridging HF Cache into SGLang KV pool
- **Speculative decoding (A6, EAGLE) for tt-xla** — EAGLE-3 already shipping per memory `tenstorrent-eagle3-working`
- **C2 device-resident inputs, async pipelining** — confirmed not levers
- **BFP8 weights, int8 KV** — Tenstorrent-stack work, not in this scope

## Maintenance / upstream plan

- Local fork `tt-mlir-sglang` matches `tt-metal-sglang` pattern (branch `tenstorrent-p1` per memory `tenstorrent-tt-metal-fork`).
- Each fix lands as a clear single-commit patch.
- Upstream PRs filed after integration-level validation; not blocking on review.
- tt-xla build switches to local override via `TTMLIR_SOURCE_DIR_OVERRIDE=/home/mhnie/tt-mlir-sglang/`.
- tt-metal compatibility: tt-mlir builds against tt-metal-sglang (already at `/home/mhnie/tt-metal-sglang/`). If tt-mlir-sglang needs newer tt-metal symbols, rebase tt-metal-sglang first.
- Document each rebase cycle in this spec's revision history.

## Evidence (fixtures + scripts)

All in `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/`:
- `probe_t2_1.log` — TTFunctionalCache K=1/2/4 bit-exact
- `probe_t2_1_perf.log` — A1 K=4 = 55 ms/tok (no speedup)
- `probe_t2_1_v3.log` — clone+index_copy_ and slice_scatter fail
- `probe_3way.log` — C2 hurts (-40 %), async no-op
- `probe_a4_sweep.log` — bs=8 = 1.40× throughput
- `probe_c3_per_token.log` — 9 s × 2 JIT cliffs per request, 70 ms steady
- `probe_c3_warmup_vs_cache_20260517_*` — R2-same-prompt cliffs reappear → dynamo.reset wipes JIT every request
- `probe_c3_scheduler_20260517_*/server_with_warmup.log` — SGLang warmup `aten::index_put` crash
- `v145_ttxla_tpot_all_models.json` — golden TPOT baseline; regression gate

Probe scripts (git-tracked under same dir as fixtures): `probe_t2_1_functional_cache.py`, `probe_t2_1_perf.py`, `probe_t2_1_v3_writeops.py`, `probe_c2_device_inputs.py`, `probe_a4_aligned_bs2.py`, `probe_a4_bs_sweep.py`, `probe_async_transfer.py`, `probe_c3_scheduler_overhead.sh`, `probe_c3_per_token.py`, `probe_c3_warmup_vs_cache.sh`.

To regenerate all evidence: run each script via `docker exec tt-xla-eval bash …` against a running `tt-xla-eval` container set up via `setup_ttxla_container.sh`. Each output is recorded back into `_fixtures/`.
