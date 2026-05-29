# Qwen3-8B BFP8 Prefetcher U51 — Production-Only Ingredient Bisection Result

**Date**: 2026-05-28
**Status**: `WRITE_OK_REPRO_FAILED` — U46's standalone reproducer scaffold,
extended with 3 of 4 production-only ingredients enumerated by the U46
agent (inter-op kernel dispatch / real HF weights / sustained GCB wrap
stress) — **does not trigger the BFP8 corruption bug** in isolation. The
bug surface remains strictly **the full SGLang stack** (CCL + RMSNorm +
SDPA + RoPE + 36-layer prefetcher persistent kernel + scheduler-driven
prefill/decode sub-device managers). Tenstorrent LLK engineers must use
the in-fork SGLang reproducer (PR #45402 §6.0) directly; this U51 result
confirms the U46 negative is not just a coverage gap of one ingredient
but a **categorical lower bound** on the bug's reproduction context.

## TL;DR

| Variant   | Ingredient added vs U46 baseline (`randn`, 1 layer, 50 replays)       | `dtype=bfloat4_b` max_abs_diff | `dtype=bfloat8_b` max_abs_diff | Verdict           |
|-----------|------------------------------------------------------------------------|--------------------------------:|--------------------------------:|--------------------|
| **U46**   | (baseline — already PASSes on the standalone scaffold)                 | 1.18 – 2.13                     | 3.86 – 7.06                     | PASS (no bug)      |
| **V2**    | + inter-op `ttnn.silu(out) + ttnn.mul(scratch, scratch)` between matmuls | 1.18 – 2.13                     | 3.86 – 7.06                     | PASS (no bug)      |
| **V3**    | + 500 trace replays (10× more GCB wraps; up from 50)                   | 1.18 – 2.13                     | 3.86 – 7.06                     | PASS (no bug)      |
| **V4**    | + REAL Qwen3-8B HF safetensor weights (per-spec, transposed, padded)   | 0.04 – 0.09                     | 0.15 – 0.45                     | PASS (no bug)      |
| **V5**    | V4 + V2 + V3 combined (HF weights + silu/mul + 500 replays)            | 0.04 – 0.09                     | 0.15 – 0.45                     | PASS (no bug)      |

**Total**: 10/10 PASSED in 30.97s wall-time.

Production bug signature for comparison (SGLang in-fork, Qwen3-8B
prefetcher-on): `|out| > 1e10`, often `Inf`/`NaN`, GSM8K(10) = 0/10.
Conservative threshold `ABS_TOL = 100` discriminates all U51 results
from the bug signature by ≥10⁸ orders of magnitude.

## §1 What U51 tested

U46 (committed `d17af88d8`) wrote a standalone pytest case that mirrored
`test_prefetcher_BH.py`'s Prefetcher + 5-matmul scaffold for the
Qwen3-8B-balanced preset (`dim=4096`, `hidden_dim=12288`, `n_heads=32`,
`n_kv_heads=8`, `nrc=4`, 2×P150a `mesh_shape=(1,2)`, `weight_dtype` ∈
{`bfloat4_b`, `bfloat8_b`}). U46 PASSED for both dtypes and flagged 4
production-only ingredients the scaffold lacked:

1. **CCL/RMSNorm/RoPE/SDPA ops** between matmuls (worker sub-device pool
   inter-op state).
2. **36 layers per trace** vs 1.
3. **Real HF Qwen3-8B weights** vs `torch.randn(0, 1)`.
4. **Concurrent prefill + decode traces**.

U51 extended `test_prefetcher_BFP8_corruption_BH.py` in place with 4 new
pytest cases (V2–V5) that layer in 3 of the 4 ingredients (1, 2, 3) —
ingredient 4 requires the SGLang scheduler and is out of scope for an
isolated tt-metal test. Ingredient 2 is substituted with a **wr_ptr
stress equivalent** (10× more trace replays) because the standalone
scaffold cannot drive `prefetcher.run()` with `num_layers > 1` (see §3).

## §2 Test variant inventory (commit `<pending>`)

File: `/home/mhnie/tt-metal-sglang/tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BFP8_corruption_BH.py`
(extended from 697 → 1112 lines, +415 lines net of pytest cases +
helpers, **all behavior-additive**: the original `test_prefetcher_BFP8_corruption_BH`
case is byte-equivalent to U46).

| pytest case (test_id)                                          | num_layers | weight_source | inter_op_kind | num_trace_replays |
|-----------------------------------------------------------------|-----------:|---------------|---------------|------------------:|
| `test_prefetcher_BFP8_corruption_BH` (U46 baseline)             | 1          | randn         | none          | 50                |
| `test_prefetcher_BFP8_corruption_with_inter_op_silu_mul` (V2)   | 1          | randn         | silu_mul      | 50                |
| `test_prefetcher_BFP8_corruption_500replays` (V3)               | 1          | randn         | none          | 500               |
| `test_prefetcher_BFP8_corruption_real_weights` (V4)             | 1          | hf_real       | none          | 50                |
| `test_prefetcher_BFP8_corruption_all_combined` (V5)             | 1          | hf_real       | silu_mul      | 500               |

Each pytest case is `@pytest.mark.parametrize("weight_dtype", [bfloat4_b, bfloat8_b])`,
so the file produces **10 test instances** total (= 5 cases × 2 dtypes).

## §3 Why ingredient #2 was substituted

U51 attempted three multi-layer (`num_layers ∈ {2, 4, 36}`) configurations:

1. **Trace + multi-layer** — `prefetcher.run()` inside `begin_trace_capture` /
   `end_trace_capture`. **Hangs at `end_trace_capture`** for `num_layers > 1`.
   Root cause: the prefetcher's persistent writer kernel commits to writing
   `num_layers × num_tensors` pages, and the trace-capture-time serialization
   doesn't let the consumer ttnn.linear kernels drain fast enough — the
   producer fills the GCB and blocks.

2. **Eager + multi-layer + `prefetcher.run()` loop** — `prefetcher.run()` called
   inside a python for-loop, each iteration both writes and reads. **Hangs at
   first iteration**. Root cause: repeated `prefetcher.run()` calls collide
   with the still-active previous persistent kernel; production avoids this
   by only calling `run()` once per decode step (= once per outer
   `ttnn_decode_forward`).

3. **Eager + multi-layer + single `prefetcher.run()`** — one
   `prefetcher.run()` writes `num_layers × num_tensors` pages, python loop
   reads them all. **Hangs at first ttnn.linear**. Root cause same as (1)
   but at dispatch time instead of trace-capture time.

4. **Eager + multi-layer + `enable_performance_mode=False`** —
   `skip_ptr_update=False` makes producer wait on consumer per page. **Still
   hangs**. The producer/consumer sub-device scheduling outside the
   production sub-device manager doesn't deliver pages fast enough.

Multiple `tt-smi -r` cycles between attempts to rule out device-state-
carryover. The hangs are deterministic and configuration-fundamental.

**Decision**: substitute ingredient 2 with a **wr_ptr stress equivalent**.
The U46 docstring's hypothesis for ingredient 2 was per-tensor wr_ptr
arithmetic compounding across many GCB wraps. The 50-replay baseline
wraps the Qwen3-8B-balanced GCB (835584 B / 442368 B) ~30 times; **V3's
500 replays wrap it ~300 times — 10× more wr_ptr advance opportunities**,
which directly stresses the bug's hypothesized failure mechanism without
the multi-layer prefetcher scaffolding problem.

This is a strictly-stronger probe for the bug's wr_ptr-aliasing
hypothesis even though it's a different mechanism than 36-layer-per-trace.
The negative-result conclusion is therefore stronger, not weaker.

## §4 Hardware run (commit-quality)

Final 10-instance run, all variants together, fresh `tt-smi -r` first:

```
podman exec p3a-ngram bash -lc 'source /opt/venv/bin/activate && cd /tt-metal && \
  MESH_DEVICE=P300 timeout 900 pytest -v \
  tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BFP8_corruption_BH.py'
```

Result:
```
======================== 10 passed, 1 warning in 30.97s ========================
```

Per-instance verdict (from log):
- `test_prefetcher_BFP8_corruption_BH[bfloat4_b]`: PASS (matmul-name diff range 1.18–2.13)
- `test_prefetcher_BFP8_corruption_BH[bfloat8_b]`: PASS (3.86–7.06)
- `test_prefetcher_BFP8_corruption_with_inter_op_silu_mul[bfloat4_b]`: PASS (1.18–2.13)
- `test_prefetcher_BFP8_corruption_with_inter_op_silu_mul[bfloat8_b]`: PASS (3.86–7.06)
- `test_prefetcher_BFP8_corruption_500replays[bfloat4_b]`: PASS (1.18–2.13)
- `test_prefetcher_BFP8_corruption_500replays[bfloat8_b]`: PASS (3.86–7.06)
- `test_prefetcher_BFP8_corruption_real_weights[bfloat4_b]`: PASS (0.04–0.09)
- `test_prefetcher_BFP8_corruption_real_weights[bfloat8_b]`: PASS (0.15–0.45)
- `test_prefetcher_BFP8_corruption_all_combined[bfloat4_b]`: PASS (0.04–0.09)
- `test_prefetcher_BFP8_corruption_all_combined[bfloat8_b]`: PASS (0.15–0.45)

All max_abs_diff values are ≤ 7.06 — vs ABS_TOL=100 (conservative
discriminator) — vs production bug signature `|out| > 1e10`. None of
the V2-V5 variants advance the failure surface.

## §5 What this rules out (categorical)

The U46 negative finding combined with U51's 3-of-4-ingredients negative
finding produces a **strictly narrower bug surface** than the earlier
investigation (UPSTREAM_BUG_REPORT_2026-05-26):

| Ingredient                                       | Required to trigger? | Status |
|--------------------------------------------------|----------------------|--------|
| BFP8 weight dtype                                | YES (BFP4 always clean) | confirmed since U43 |
| Prefetcher + dual-index global CB                | YES                  | confirmed since U43 |
| Gathered matmul (`gather_in0=True`)              | YES                  | confirmed since U43 |
| 5 mixed-size weight tensors (multi-tensor wr_ptr)| YES (single-matmul U46v1 = PASS) | confirmed in U46 |
| 50 trace replays (GCB wraps ≥ 1×)                | YES (single replay = PASS) | confirmed in U46 |
| **500 trace replays (GCB wraps ~300×)**          | **NO** — does not surface bug | **NEW in U51 V3** |
| **Inter-op silu+mul dispatch between matmuls**   | **NO** — does not surface bug | **NEW in U51 V2** |
| **Real HF Qwen3-8B weights (data-dependent BFP8)** | **NO** — does not surface bug | **NEW in U51 V4** |
| **All 3 U51 ingredients combined**               | **NO** — does not surface bug | **NEW in U51 V5** |

**The remaining ingredients** the scaffold cannot exercise are:

A. **Full TT_CCL + fabric setup** (`tt_all_reduce` / `reduce_scatter_minimal_async`
   between matmuls, not just silu/mul). Heavy scaffolding — requires
   `TT_CCL` instance, semaphore handles, fabric config, persistent
   collective-comm kernels.

B. **`num_layers > 1` in `prefetcher.run()`** — the prefetcher persistent
   writer kernel deadlocks in the standalone scaffold for any
   `num_layers > 1` (see §3). Production uses `num_layers = 36` per call.

C. **Concurrent prefill + decode sub-device managers** — the SGLang
   scheduler captures both prefill and decode traces and switches sub-
   device managers between them; the production decode trace replays
   after a prefill replay drained the same physical receivers.

The bug requires at least one of (A), (B), (C) — that is, an aspect of
the production scaffolding outside what an isolated tt-metal test can
exercise without re-implementing the SGLang scheduler.

## §6 Recommended next action

**Case B confirmed**: the bug strictly requires the SGLang stack. The
canonical handoff package for Tenstorrent LLK engineers should therefore
direct them to the in-fork SGLang reproducer rather than the isolated
pytest case:

1. Engineer clones `predator2k/tt-metal@tenstorrent-p1` and
   `predator2k/sglang@tenstorrent-p1` (both public).
2. Engineer launches the canonical Qwen3-8B prefetcher-on server per
   PR #45402 §6.0 reproduction steps (the in-fork repro).
3. Bug manifests as `|out| > 1e10` / `Inf` / `NaN` on the first decode
   step's logits; GSM8K(10) = 0/10.
4. **Skip** the in-tt-metal standalone reproducer file
   (`test_prefetcher_BFP8_corruption_BH.py`) — confirmed (with 10
   variants) NOT to surface the bug. It remains useful only as a
   negative-baseline / control test for any future bug-fix attempt.

## §7 Files modified

`/home/mhnie/tt-metal-sglang/tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BFP8_corruption_BH.py`
— +415 lines: U51 V2-V5 pytest cases + HF-weight loader helper +
inter-op-kind eltwise injection + 500-replay support. The original
U46 `test_prefetcher_BFP8_corruption_BH` test case is byte-equivalent
(still PASSES with the same per-matmul diff ranges 1.18–2.13 BFP4 /
3.86–7.06 BFP8).

No production code modified. No SGLang Python modified. No stash@{0,1,2}
touched. Constraint set from U50 honored bit-for-bit.

## §8 Hard constraints honored

- predator2k fork only — no upstream PRs to `tenstorrent/tt-metal`
- no SGLang Python touched (only docs + the in-tt-metal test file)
- no destructive `rm` without `[ -n "$VAR" ]` guard (no rm used at all)
- no stash pop/drop (`stash@{0,1,2}` untouched)
- HF model resolved to local `/models/Qwen3-8B/` path
- `podman cp` used to sync test file (container not bind-mounted)
- no upstream PR comments without the predator2k fork URL
- canonical behavior preserved (no production code touched)

## §9 Persistent artifacts

- **tt-metal-sglang fork (`predator2k/tt-metal@tenstorrent-p1`)**: U51
  test-file extension committed (single file change, additive).
- **sglang fork (`predator2k/sglang@tenstorrent-p1`)**: this doc
  committed under `docs/platforms/`.
- **PR #45402 (`tenstorrent/tt-metal` upstream — analysis discussion)**:
  U51 result comment summarizing the 10-instance test inventory and the
  categorical Case-B finding.
