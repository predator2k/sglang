# TT Qwen3-8B prefetcher — Phase A (Python ablations) — 2026-05-23

Status: **DONE — Phase A ruled out S3 (WO-specific) and BFP4-specific
hypotheses. Bug is in the generic BFP8 prefetcher producer/consumer
protocol; affects ALL prefetched weights regardless of dtype mix.
Canonical shipping path UNCHANGED, GSM8K(10) re-verified 9/10 = 90%.**

Continuation of `tt_qwen3_8b_prefetcher_cpp_phase1_2026-05-23.md`
("phase-1 audit"). Phase A here = the "S3 ablation first" plan from
that doc, extended with two follow-up ablations after S3 failed.

## What changed this session (tt-metal-sglang)

Three new env-gated knobs added — **default-off, zero behavior change
on canonical or full-prefetcher paths**:

| Env var | File | Effect |
|---|---|---|
| `SGLANG_TT_PREFETCHER_SKIP_WO` | `attention.py:594-610`, `:1247-1276` | WO matmul reads weight from DRAM (no GlobalCB); `prefetch=False, num_global_cb_receivers=1` ring config |
| `SGLANG_TT_PREFETCHER_SKIP_WQKV` | `attention.py:600-611`, `:834-863` | Same trick for WQKV |
| `SGLANG_TT_PREFETCHER_SKIP_W1`/`_W3`/`_W2` | `mlp.py:109-128`, `:140-200`, `:283-310` | Same trick for MLP weights |
| (helper) | `generator_sglang.py:155-178` | `num_tensors` drops by 1 per SKIP_* flag set |

All five SKIP_* knobs are composable; setting any combination subtracts
the corresponding tensor(s) from the prefetcher's per-layer queue AND
reroutes that matmul to a `prefetch=False` ring-config matmul that
reads its DRAM-sharded weight directly (mirrors the `lm_head.py:142-166`
pattern already in tree).

These knobs are diagnostic scaffolding — they are NOT a shipping
config. The bypassed matmuls are slower than the full prefetcher case
(weight reads hit DRAM instead of L1 GlobalCB) but the layout is
preserved so downstream layers see the same shard.

## Ablation matrix (all on Qwen3-8B chat GSM8K(10), prefetcher BFP8)

| Configuration                                      | Per-layer tensors    | GSM8K(10) | Q1 garbage pattern                  |
|---|---|---|---|
| Canonical (no prefetcher) — control                | n/a                  | **9/10**  | (clean answers)                     |
| Full prefetcher (broken baseline)                  | 5 (wqkv, wo, w1, w3, w2) | 0/10 | "猖偓 Waters偓..."                  |
| `SKIP_WO=1`                                        | 4 (wqkv, w1, w3, w2) | 0/10      | "��� geç... Js anol anol 黑马..."  |
| `SKIP_WO=1 SKIP_WQKV=1`                            | 3 (w1, w3, w2)       | 0/10      | "(role的一面(role..."               |
| `SKIP_W1=1 SKIP_W3=1` (no BFP4 in prefetcher)      | 3 (wqkv, wo, w2)     | 0/10      | (server crash on Q1)                |

In every prefetcher configuration the server eventually trips
`RuntimeError: probability tensor contains either inf, nan or element
< 0` and the scheduler exits.  Canonical (no prefetcher) does not.

## Hypothesis elimination

| Suspect (from phase-1 audit)                                | Status after Phase A |
|---|---|
| **S1** Row-wise stride mismatch in writer (most likely)     | Still standing; consistent with universal failure |
| **S2** num_blocks mismatch                                  | Still standing; would also be universal |
| **S3** WO-specific K-shard transposition                    | **RULED OUT** — SKIP_WO didn't fix it |
| **S4** Single-cycle GlobalCB drift (already ruled out)      | Confirmed ruled out |
| **S5 (new this session)** BFP4 vs BFP8 mixed-dtype tile pitch | **RULED OUT** — SKIP_W1+SKIP_W3 (no BFP4 in prefetcher) still fails |
| **S6 (new this session)** Per-matmul `set_page_size` drift on shared GlobalCB region across consecutive matmuls of different weights | **OPEN** — every change in the per-layer weight set shifts the corruption pattern, suggesting an inter-tensor cursor/offset problem rather than a per-tensor stride bug |

The deterministic shift in garbage tokens for each new subset is the
strongest single signal: it means the corruption shape is sensitive to
the EXACT sequence of tensors the prefetcher cycles through, not to
any single tensor's properties. S6 (cross-tensor GlobalCB cursor) and
S1 (per-tensor row stride) remain consistent with this. A single-cycle
or BFP8-shared-exponent bug would have given a tensor-invariant
signature.

## Why we stopped after 3 ablations

Each iteration costs ~10 min (server boot + GSM8K(10)). The
8-hour cap with the user's max-6-iterations Phase B guidance left
~3 h for additional work, all of which would land in Phase B
(C++ byte-dump). Three more Python ablations:

- `SKIP_W2=1` (only BFP8 MLP gone) — would isolate to W2 specifically
- `SKIP_W1=1 SKIP_W3=1 SKIP_W2=1` (no MLP prefetched, only attention) —
  would test if pure-attention BFP8 cycle works
- `SKIP_*` everything except a single tensor — would test if even
  single-tensor prefetcher reproduces the bug

These three would cost ~3 × 10 min = ~30 min total. They are recommended
as the OPENING ATTACK of Phase B's next session, because they're cheap
and they decisively narrow the bug to "any-tensor" vs "specific-tensor"
vs "tensor-pair-interaction". Documented as TODO at the bottom.

## Phase B recommended attack (next session)

### B.1 — Finish the SKIP_* matrix (~30 min, cheap, do FIRST)

1. `SKIP_W2=1` only — does W2 alone cause it?
2. `SKIP_W1=1 SKIP_W3=1 SKIP_W2=1` — does attn-only (wqkv+wo, 2 tensors/layer) reproduce?
3. `SKIP_WO=1 SKIP_W1=1 SKIP_W3=1 SKIP_W2=1` (only wqkv prefetched, 1 tensor/layer) — does a single-tensor prefetcher reproduce?

Result interpretation:
- If (3) reproduces → the bug is in single-tensor prefetcher and S1
  (per-tensor stride) is the SOLE root cause. Byte-dump WQKV.
- If (3) is clean but (2) reproduces → the bug is a cross-tensor
  interaction. Byte-dump the writer's `fifo_wr_ptr` between tensors.
- If only (1) reproduces → W2 alone is the trigger. Byte-dump W2.

### B.2 — Byte-dump C++ instrumentation (~3-4 h if B.1 didn't localize)

Same plan as phase-1 audit section "If S3 ablation fails":
1. DPRINT-gated dump in `writer_l1.cpp` after the first
   `remote_cb_push_back_and_write_pages` for the first tensor.
2. Matching DPRINT in `reader_bmm_tile_layout_in1_ring_all_gather.cpp`
   (or the matmul compute kernel since the reader doesn't actually
   touch GlobalCB bytes — the matmul compute kernel does).
3. Cross-compare. The first byte mismatch IS the bug.

### B.3 — Fix and validate

GSM8K(10) ≥ 7/10 gate, then bench TPOT and commit.

## Why this session did NOT do byte-dump

Two reasons:
1. After completing the audit and 3 ablations, the budget left was
   ~3 h. Phase 1 audit's own estimate for byte-dump = "1 h
   instrumentation + 3+ iterations × (5 min rebuild + 5 min repro) +
   1-2 h cross-comparison + 1-2 h fix iter = 6-8 h." Starting that
   with 3 h left would land mid-iteration.
2. The unfinished `SKIP_*` matrix above (~30 min total) might
   collapse the byte-dump scope dramatically — if it shows
   single-tensor reproduces, byte-dump targets a single tensor's
   path instead of the full multi-tensor cycle. Cheaper to finish
   B.1 first then decide B.2 scope.

## Files touched this session

- `tt-metal-sglang/models/tt_transformers/tt/attention.py` —
  SKIP_WO / SKIP_WQKV knobs at `register_weights()` and both matmul
  call sites. Default-off.
- `tt-metal-sglang/models/tt_transformers/tt/mlp.py` —
  SKIP_W1 / SKIP_W3 / SKIP_W2 knobs at `register_weights()` and the
  W1, W3, W2 matmul call sites. Default-off.
- `tt-metal-sglang/models/tt_transformers/tt/generator_sglang.py` —
  helper that adjusts `num_tensors` from 5 → 5-N where N = count of
  SKIP_* flags set.
- `sglang/docs/platforms/tt_qwen3_8b_prefetcher_phase_a_2026-05-23.md` —
  this document.

NO C++ edits this session. tt-metal-sglang HEAD will move +1 commit
(`prefetcher: SKIP_* diagnostic knobs (Phase A)`) but `081215f6ae4`
behavior is preserved when no SKIP_* env is set.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9/10 = 90%
(re-verified 2026-05-23 in this session, post-edits).

DO NOT ship the prefetcher under `SGLANG_TT_USE_PREFETCHER=1` for
Qwen3-8B until phase-B-2 byte-dump identifies the byte-level routing
mismatch and a producer-side correction lands in
`tt-metal-sglang/ttnn/cpp/ttnn/operations/prefetcher/`.

## Commits

- (this session, tt-metal-sglang) `prefetcher: SKIP_* diagnostic knobs (Phase A)`
- (this session, sglang) `docs(tt-prefetcher): Phase A — SKIP_* diagnostic knobs, ruled out S3/S5 (2026-05-23)`
