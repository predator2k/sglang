# TT Qwen3-8B prefetcher — Phase B.9 (SKIP_* reroute completion attempt — EVIDENCE_ADVANCE: no commit; canonical baseline noisier than docs suggest) — 2026-05-23

Status: **EVIDENCE_ADVANCE — 5 iterations of SKIP_* reroute changes (mesh_mapper 1-D, untilize_out, attention.py WO/WQKV variants) all CHANGED the garbage signature (proving each ingredient is load-bearing) but NONE produced a stable correctness gate. Iter-2 (mlp.py ShardTensorToMesh + W2 untilize_out=True) produced ONE coherent-English chat response and a partial 1/7-correct GSM8K (Q2=260 bit-exact) on its first launch, but the result was NOT reproducible across server restarts. SKIP_ALL with cumulative changes also failed (different garbage signatures each iteration). Working tree fully REVERTED to `c1e3437b78e` (B.8) at session end. Canonical baseline re-verified at 9/10 = 90% in a fresh-server run, confirming no regression from this session.**

Continuation of `tt_qwen3_8b_prefetcher_phase_b8_2026-05-23.md`. Executes the dispatch's "Attack A — reroute completion" plan (Step 0-8).

## Bottom line

| Iter | Config | Edit | GSM8K(10) chat | Q1 signature | Decision |
|---|---|---|---|---|---|
| (baseline B.8) | SKIP_W2=1 | (none) | 0/10 | `房东benh火力ifty主权悠悠 bambS齊 Canon 新规...` | broken |
| 1 | SKIP_W2=1 | mlp `_make_skip_ring_weight`: ShardTensor2dMesh → ShardTensorToMesh(dim=dims[1]) | 0/10 (server died) | `<think> Grammar Grammar Grammar...` | engaged — KEEP |
| 2 | SKIP_W2=1 | + `_pc_w2_skip` untilize_out=True | 1/7 visible (Q2=260 CORRECT) + chat probe `Okay, the user is asking "What is 2+2?"...` | **COHERENT ENGLISH** (but ONE launch only) | apparent WIN — KEEP and validate |
| 2-retry | SKIP_W2=1 | (same code) | 0/10 server died | `<think>� hacking一朵的带领灰尘�做的事...` | **NON-REPRODUCIBLE**; signature different from Iter-2 success |
| 3 | SKIP_ALL | + WQKV mesh_mapper 1-D + WO untilize_out | 0/10 | `<think>看見邮寄卖给掐handhand...` | REVERTED both attn changes |
| 5 | SKIP_ALL | only mlp Iter-1+2 | 0/10 | `<think>冏 terminal terminal terminal terminal...` | broken |
| 6 | SKIP_ALL | + WO untilize_out only (attn) | 0/10 | `<think>_heap_heap_heap_heap...` | broken |
| canonical re-verify (run 1, contaminated) | (no envs) | (none) | 6/10 (Q1 INVALID 131s timeout from chat-probe contamination) | clean math | session-state noise |
| canonical re-verify (run 2, fresh) | (no envs) | (none) | **9/10 = 90%** (Q3 only WRONG; long-thinking error) | clean math | **CONFIRMS documented baseline; no regression** |

## Iter-2 — the elusive coherent-English flicker

After applying mlp.py Iter-1 (1-D mesh_mapper) + Iter-2 (W2 untilize_out=True),
the FIRST chat probe of the FIRST server launch returned:

```
<think>
Okay, the user is asking "What is 2+2?" and wants the answer to be just
the number. Let me make sure I understand the question correctly. They
want the result
```

This was the first coherent prefetcher-enabled output in the entire
debugging chain. The GSM8K(10) immediately after got Q2=260 correctly
(bit-exact match). Q1 garbage parse failure but `last_line='$$'` (model
trying LaTeX format). Q3-Q7 generated plausible but wrong arithmetic.

**On a SECOND server launch with identical envs and code**, the chat
probe returned `<think>� hacking一朵的带领灰尘...` — pure garbage. Eval
also collapsed.

This is consistent with the underlying prefetcher being non-deterministic
(broken GlobalCB BFP8 cycle): when SKIP_W2=1, only W2 takes the reroute,
but WQKV/WO/W1/W3 still go through the broken prefetcher. Coherent
output requires ALL the prefetcher's outputs to be correct, which the
broken cycle only achieves probabilistically.

**Decisive interpretation**: my Iter-2 W2 reroute may or may not be
correct — without a test that can be reproduced across launches, I
cannot claim a fix. The 1-D mesh_mapper change is conceptually sound
(it matches lm_head exactly) and the W2 untilize_out follows the same
prescription. But the underlying prefetcher non-determinism prevents
clean verification.

## SKIP_ALL — direct test of reroute correctness

When SKIP_ALL=1 is set (all 5 weights bypassed), the prefetcher is
short-circuited per the B.7 zero-tensor guard. ALL 5 matmuls take the
reroute path. This is the cleanest test of whether the reroute
itself is correct.

Three attempted SKIP_ALL configs (Iter-3, Iter-5, Iter-6 in the table
above) all produced garbage with DIFFERENT signatures (handhand → terminal
→ _heap), confirming each ingredient is load-bearing on the kernel input
but NONE produces correct output. The 5-iteration cap was reached
without finding the missing ingredient.

## Canonical re-verify: 9/10 = 90% (CONFIRMED in clean state)

Two canonical GSM8K(10) chat runs were performed:

1. **Run 1 (contaminated, 6/10)**: after multiple chat probes and the
   SKIP_* experiments, the first canonical eval hit a 131s timeout on
   Q1 (INVALID) and degraded other answers. Result: 6/10.
2. **Run 2 (fresh server, 9/10)**: after a clean server restart with
   no chat probes, the canonical eval hit the documented
   baseline. Q1+Q2+Q4+Q5+Q6+Q7+Q8+Q9+Q10 all CORRECT; only Q3 wrong
   (160→120, long-thinking-chain reasoning error). **9/10 = 90.0%
   confirms the documented baseline.**

This confirms my session changes did NOT regress canonical (they're in
SKIP-only branches anyway; reverted at session end regardless). The
initial 6/10 was session-state contamination, not a code regression.

## Suspect ranking after Phase B.9

| ID | Suspect | Status |
|---|---|---|
| S1 | Row-wise stride mismatch in writer | RULED OUT (Phase B.3 byte-dump) |
| S2 | num_blocks mismatch | OPEN, unlikely |
| S3 | WO-specific K-shard transposition | RULED OUT (Phase A) |
| S5 | BFP4/BFP8 mixed tile pitch | RULED OUT (Phase A) |
| S6 | Per-layer cross-block address drift in producer | RULED OUT (Path A B.4-extended) |
| S7 | Compute-kernel `update_rd_ptr_to_ring_index` wrap | RULED OUT (Phase B.6) |
| S8 | Per-tensor block_size in reader_dram.cpp | RULED OUT (Path A re-eval) |
| S9 | Producer NOC posted-writes flush insufficient | OPEN (Attack B testing concurrently) |
| S10 | Sub-device CB-address misalignment | OPEN |
| SKIP_W2 reroute (mesh_mapper + untilize) | (mlp.py only) | **PARTIALLY engaged — flicker of coherent output; non-reproducible across launches** |
| SKIP_W1/W3 reroute (mesh_mapper only) | (mlp.py only) | OPEN — untested in isolation |
| SKIP_WO reroute (untilize_out) | (attn.py) | OPEN — failed in SKIP_ALL config but not tested with SKIP_WO=1 alone |
| SKIP_WQKV reroute (1-D mesh_mapper) | (attn.py) | OPEN — made things worse in SKIP_ALL; not tested alone |

## Path forward — recommended next session

### Step 1: Re-run SKIP_W2=1 with Iter-2 changes across multiple launches

The non-reproducibility of Iter-2's coherent-English flicker is the
core unsolved question. Run 5-10 fresh launches of SKIP_W2=1 +
mlp.py Iter-1+2 patch and tabulate signatures + GSM8K(10) results.
If even ONE launch gives ≥4/10, the reroute fix IS partially
working. If all give 0/10 with varying garbage, the fix isn't
helping.

### Step 2: Triangulate the SKIP_ALL ingredient

Per the dispatch's intended workflow but on a clean reproducible
baseline:

1. SKIP_W2 only (reference signature for ANY-coherent vs ALL-garbage).
2. SKIP_W2 + SKIP_W1: did W1 break? → W1 needs untilize_out + re-tile
3. SKIP_W2 + SKIP_W3: did W3 break? → W3 needs same
4. SKIP_W2 + SKIP_WO: did WO break? → WO needs additional ingredient
   beyond untilize_out
5. SKIP_W2 + SKIP_WQKV: did WQKV break? → WQKV's 2D mesh_mapper is OK
   on (1,2), but maybe not in combination with WO

Each pairwise test narrows the failing weight.

### Step 3: Investigate the SKIP_ALL failures from this session

The Iter-3/5/6 SKIP_ALL signatures (handhand → terminal → _heap) are
all "Chinese/English token noise" patterns, not random-byte garbage.
This suggests the kernel IS producing OUTPUT, just with corrupted
weights. The character-distribution shifts confirm the kernel reads
the bytes — they're just wrong bytes.

A byte-level dump on the SKIP_ALL path (taking advantage of the
no-prefetcher state for clean experiment) would localize WHERE the
weight bytes are first incorrect: at the producer? at the matmul reader?

## What this session did NOT do

- Did NOT commit the Iter-2 patch (mesh_mapper + W2 untilize_out)
  because the chat coherence was not reproducible across launches.
- Did NOT find the missing ingredient(s) for SKIP_ALL to reach ≥7/10.
- Did NOT modify any C++ kernel code (Attack B's domain).
- Did NOT push to remote — working tree cleaned, no new tt-metal-sglang
  commits.

## Working state at end of session

- tt-metal-sglang HEAD: **`c1e3437b78e`** (Phase B.8 reroute ring-grid
  variants) — UNCHANGED from session start.
- Working tree: clean (all SKIP-path Python edits reverted).
- C++ kernel tree: pristine.
- Container `/tt-metal/models/tt_transformers/tt/{mlp.py, attention.py}`
  re-synced to match HEAD.
- Canonical baseline (today, fresh server): **9/10 = 90%** — matches
  documented baseline. No regression from this session's changes (all
  reverted; were behavior-preserving anyway).
- All cards healthy.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024. The 9/10 baseline is the
documented target; today's 6/10 suggests the baseline needs a clean
re-verification in the next session.

DO NOT ship the prefetcher under `SGLANG_TT_USE_PREFETCHER=1` for
Qwen3-8B. The prefetcher path remains broken (Path A's BFP8 GlobalCB
cycle), and the SKIP_* reroute fallback also remains broken (this
session's Iter-1..6 all changed the garbage signature but none
reached a reproducible correctness gate).

## Commits this session

- (sglang) `<this doc>` — pending.
- (tt-metal-sglang) — NO commits; working tree clean at session end.

## File-level summary

- `tt-metal-sglang/models/tt_transformers/tt/mlp.py` — reverted, no
  net change.
- `tt-metal-sglang/models/tt_transformers/tt/attention.py` — reverted,
  no net change.
- `sglang/docs/platforms/tt_qwen3_8b_prefetcher_phase_b9_2026-05-23.md`
  — this doc, NEW.
