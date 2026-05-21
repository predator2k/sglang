# tt_transformers DRAM-prefetcher Option A results (2026-05-21)

This document records the final outcome of the Option A prefetcher plan begun from the 2026-05-19 handoff. It follows the same structure as the original handoff and supersedes it with measured results.

## TL;DR

- **Final TPOT: 31.35 ms** (vs baseline 37.66 ms = **+6.31 ms / +17% improvement**).
- **Winning phase: A.3** — delivered via `SGLANG_TT_DISABLE_PREFILL_TRACE=1` env var, NOT the originally-planned L1 tuning route. The env var makes prefill bypass the prefetcher's GlobalCB coexistence issue by running under the DEFAULT sub-device manager and loading the DECODE manager (with GCB) only for decode.
- **A.1 attempted, BLOCKED**: upstream irregular-grid bug in `Prefetcher.dynamic_worker_core_grid(32)`; kernel surgery was mechanically correct but `cb_in0` arrived pre-tainted with bf16 -Inf from an upstream sharded op miscomputing on the 7-range non-rect layout.
- **A.2 skipped** per the plan preamble's gate: A.3 delivered ≥3 ms improvement, so A.1+A.2 are gated out.
- **All 3 bench runs PASS**: run1_cold 31.8 ms, run2_warm_new_prompt 31.3 ms, run3_warm_kv_hit 31.3 ms.

---

## Final state (commits & artifacts)

### `/home/mhnie/tt-metal-sglang/` — branch `tenstorrent-p1`

Commit stack (oldest relevant → HEAD):

| commit | description |
|---|---|
| `142c11b40c2` | A.3 win: `SGLANG_TT_DISABLE_PREFILL_TRACE` env-var lever |
| `a46a236be73` | A.3 extras: optional `core_range_set` kwarg through `layer_norm`/`rms_norm` (dead code — no callers pass the kwarg; committed for structural completeness) |
| `ef371f8f320` | Reverted Task 9 (reverts an intermediate WIP commit) |
| `5a12b4b0c0e` | Revert "prefetcher A.1 prep" — A.1 kernel surgery reverted after upstream-grid-bug confirmed |

Stash state:

| ref | contents | status |
|---|---|---|
| `stash@{0}` | Task 11 A.1 WIP (post-debug, includes `merge_ranges` fix in `enumerate_occupied_cores`) | Recoverable if upstream grid bug is fixed |
| `stash@{1}` | Task 11 A.1 WIP (pre-debug, BLOCKED on correctness + 2x latency regression) | Superseded by `stash@{0}`; kept for history |
| `stash@{2}` | Cascade WIP pre-A.1 — earlier-session experimental work | Not A.1-relevant |

### `/home/mhnie/sglang/` — branch `tenstorrent-p1`

| artifact | path | contents |
|---|---|---|
| Final bench fixture | `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_prefetcher_final.json` | 31.35 ms canonical TPOT — this session (2026-05-21) |
| Baseline fixture | `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_prefetcher_off.json` | 37.66 ms — prefetcher OFF |
| Prior-win fixture | `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_prefetcher_no_prefill_trace_v4.json` | 31.28 ms — original A.3 win measurement; final is within 0.07 ms |

---

## Per-phase outcomes

### Phase A.4 (infrastructure) — DELIVERED

Standalone smoke pytest, PCC harness, shared `_prefetcher_harness.py`, and `rebuild_tt_metal_kernels.sh` were all committed in earlier commits on `tenstorrent-p1`. The four standalone-correct portions of the original WIP patch were also committed:

- `generator_sglang.py`: iterate `ring_sizes` in the outer prefetcher gate
- `prefetcher.py`: `worker_start_core` lazy property
- `rope.py`: lazy lookup of `worker_start_core` at `get_rot_mats` call time
- `model_config.py`: same lazy lookup in SDPA decode program config and output mem config

These are behavior-preserving when `use_prefetcher=False` and upstreamable as standalone bug fixes.

### Phase A.3 (L1 budget tuning) — DELIVERED (via alternative mechanism)

The originally-planned A.3 lever was to tune `block_h`/`block_w` parameters of the unsharded RMSNorm to shrink its per-core static CB. That avenue was not pursued because a more direct lever was found first.

The **actual A.3 win**: `SGLANG_TT_DISABLE_PREFILL_TRACE=1` env var (commit `142c11b40c2`). When set, prefill bypasses the prefetcher's GlobalCB coexistence issue entirely by running under the DEFAULT sub-device manager (no GCB resident). The DECODE manager (with GCB) is loaded only for the decode phase. Net effect:

- Prefill is ~10 ms slower per TTFT (acceptable trade-off for throughput-focused workloads).
- Decode TPOT improves by **+6.31 ms** because the prefetcher can now coexist with the L1 layout without clashing circular buffers.

The `core_range_set` plumbing committed at `a46a236be73` (optional kwarg through `layer_norm`/`rms_norm`) is **dead code** — no callers pass the new kwarg. It is committed for structural consistency in case a future engineer wants to build on it, but it has no effect on any current benchmark number.

### Phase A.1 (non-rectangular sharded RMSNorm) — BLOCKED (definitive structural finding)

The kernel surgery required for A.1 was:

1. Per-core enumeration helper (`enumerate_occupied_cores` in `sharded_layernorm_factory_helpers.cpp`) with a `merge_ranges` ordering fix to align with the factory's `corerange_to_cores(all_cores, ...)` iteration order.
2. Removal of the `TT_FATAL` on `shard_spec.grid.num_cores() != bbox_num_cores` in `layernorm_device_operation.cpp:185`.
3. Sender-kernel unicast conversion across 3 multicast sites.
4. Receiver-kernel symmetric updates.

This work was mechanically completed. Tracy-decomp investigation during debug showed the standalone-eager cost of A.1 was **+5.3 ms / decode**, not the originally-feared +33 ms — a meaningful finding that changes the risk calculus for future attempts.

However, DPRINT + L1 hash inspection on sender's `cb_in0` and receivers' `cb_ex_global` revealed the inf-tainted output came from **UPSTREAM**: `cb_in0` (the input to RMSNorm) already contained bf16 -Inf BEFORE the kernel ran.

Root cause: `Prefetcher.dynamic_worker_core_grid(32)` returns a **7-range irregular layout** despite the comment claiming a 4×8 rectangle. Some upstream sharded op (MLP `linear_w2`, attention `wo`, or the all_gather) silently miscomputes on irregular grids, writing -Inf into the tensor that becomes `cb_in0` for the subsequent RMSNorm.

**A.1 is infeasible on the current code shape without fixing the upstream grid bug.** The work was reverted via `5a12b4b0c0e`.

### Phase A.2 (HEIGHT_SHARDED sharded RMSNorm) — SKIPPED

The plan preamble's gate logic authorizes this: "After A.3: if TPOT improves ≥3 ms, skip A.1+A.2." A.3 delivered +6.31 ms. Additionally, A.2 would have the same upstream-grid-bug exposure as A.1, since the irregular-grid layout is the current output of `dynamic_worker_core_grid`. It is unlikely to deliver correct output without the upstream fix.

---

## A/B TPOT table

| Configuration | TPOT (ms) | vs baseline | Notes |
|---|---|---|---|
| Baseline (prefetcher OFF) | 37.66 | — | `v146_3run_server_prefetcher_off.json` |
| A.3 only (`SGLANG_TT_DISABLE_PREFILL_TRACE`) | 31.28 | **+6.39 ms** | `v146_3run_server_prefetcher_no_prefill_trace_v4.json` (prior session) |
| A.1 attempt (per-core unicast, broken correctness) | ~71 ms (eager standalone) | — | Inf-tainted output; standalone-eager number is not comparable to server-traced numbers; shown for reference only |
| Final (A.3, post-revert, this session) | 31.35 | **+6.31 ms** | `v146_3run_server_prefetcher_final.json` (2026-05-21) |

The 0.07 ms difference between prior-session 31.28 ms and this-session 31.35 ms is within normal run-to-run variance.

---

## PCC

The committed PCC fixture (`prefetcher_pcc_baseline.pt`, norms 2591/2036/2052) was captured at an earlier state. The PCC test's in-process OFF-baseline capture is **BROKEN at HEAD** — a pre-existing regression that no commit in this session caused or fixed. Running `test_prefetcher_pcc.py` on pristine HEAD returns `norm=0` for the OFF baseline; this was verified before and after this session's commits.

For end-to-end correctness validation, the canonical smoke remains the **5K+5K translation prompt** via the SGLang server (see `tenstorrent-p1-implementation-status` memory). Server logs at run2/run3 show no -Inf or NaN in logits when `SGLANG_TT_DISABLE_PREFILL_TRACE=1` is set.

---

## Upstream PR split recommendation

Per the original handoff's "clean PR scope" framing, the absorbed work decomposes into 3 standalone PR candidates against `tenstorrent/tt-metal`:

**PR 1 — Standalone gate fix + `worker_start_core` property** (`generator_sglang.py` iterate ring_sizes; `prefetcher.py` `worker_start_core` lazy property). Smallest, low-risk, behavior-preserving. Recommended first.

**PR 2 — Lazy `worker_start_core` lookup in RoPE / SDPA** (`rope.py`, `model_config.py`). Behavior-preserving rename of a stored-stale reference to a lazy attribute lookup. Independent of PR 1.

**PR 3 — `SGLANG_TT_DISABLE_PREFILL_TRACE` env-var path** (the actual winning code, in `models/tt_transformers/tt/generator.py`). Most complex because it requires the sub-device-manager-switch helpers in `prefetcher.py`. Should ship together with the A.3 extras (the `core_range_set` plumbing in `layer_norm`/`rms_norm`) so the upstream surface is consistent.

Per user memory `feedback_no_upstream_pr`: **do not open any upstream PRs in this session.** The above is for future planning only.

---

## Critical execution gotchas

These are load-bearing for anyone running the bench or server in this configuration. Each burned real time to discover.

**`SGLANG_TT_DISABLE_PREFILL_TRACE=1` is MANDATORY when running the server with `SGLANG_TT_USE_PREFETCHER=1`.** Without it, the server hangs on run2's `Tensor.cpu()` call in the V2.5 split-logits path. After such a hang, the PCIe bus is left in `0xffffffff` state requiring `tt-smi -r 0,1` hardware reset. This is the whole reason A.3 won — the env var is the A.3 lever; without the hang, there is no incentive to discover it.

**`--context-len 2048` is mandatory for `bench_3run_server_alive.py` at 1024/256.** The bench default `context_len = input_len + output_len = 1280` triggers `model_config.py`'s "capped_warmup_seq_len must be a power of 2" assertion.

**Port-clear regex requires escaped `\.`.** Use `pkill -9 -f "sglang\.launch_server.*--port 30000"` — without the backslash, pkill matches its own enclosing argv (which contains the literal pattern string) and self-terminates the parent bash process (exit 137). This was load-bearing for every bench in the entire cascade.

**Container `p3a-ngram` and host `/home/mhnie/tt-metal-sglang/` can diverge.** After C++ changes on host, sync via `podman cp` then rebuild via `/sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh`. Do not rebuild manually inside the container — the script captures the correct include paths and flags.

---

## Outstanding follow-up work

These are independent investigations; none block the current +17% TPOT win.

**Upstream grid bug in `Prefetcher.dynamic_worker_core_grid`** (`models/tt_transformers/tt/prefetcher.py:422` in-container, `:391-396` on host has dead nested-function code). It returns a 7-range irregular layout instead of the 4×8 rectangle its comment claims. Fixing this would unblock A.1, A.2, and likely other future sharded-op work. Estimated 4-12 hours for someone familiar with tt-metal core-range algebra.

**PCC harness OFF-baseline regression**: in-process OFF capture in `test_prefetcher_pcc.py` returns `norm=0` at HEAD (pre-existing). The committed `.pt` baseline is from an earlier state. Needs root-causing before PCC validation can serve as a reliable regression gate.

**A.1 kernel WIP at `stash@{0}`**: the `merge_ranges` ordering fix in `enumerate_occupied_cores` is real and correct. It aligns iteration with the factory's `corerange_to_cores(all_cores, ...)` call order. If A.1 is ever resurrected after the upstream grid bug is fixed, this stash is the right starting point — do not start from scratch.

---

## Cross-links

- Original handoff: [`tt_transformers_prefetcher_handoff_2026-05-19.md`](tt_transformers_prefetcher_handoff_2026-05-19.md)
- Plan: [`../superpowers/plans/2026-05-19-tt-prefetcher-option-a-plan.md`](../superpowers/plans/2026-05-19-tt-prefetcher-option-a-plan.md)
- Per-op breakdown (canonical pre-prefetcher): [`tt_transformers_perop_breakdown_2026-05-18.md`](tt_transformers_perop_breakdown_2026-05-18.md)
- TPOT decomposition (memory): `tenstorrent-tt-transformers-tpot-decomposition`
- tt-metal fork pointer (memory): `tenstorrent-tt-metal-fork`
- Profiling guide: [`tt_metal_profiling_guide.md`](tt_metal_profiling_guide.md)
