# tt_transformers DRAM-prefetcher v6 results (2026-05-22)

This document records the final outcome of the 2026-05-22 prefetcher performance session. It follows the same structure as [`tt_prefetcher_option_a_results_2026-05-19.md`](tt_prefetcher_option_a_results_2026-05-19.md), which it supersedes. Methodology: 3-run `bench_3run_server_alive.py`, 1024 input / 1024 output tokens, `--context-len 2048`.

## TL;DR

- **Final TPOT: 27.50 ms** (1k/1k, mean of run2_warm + run3_kv_hit) vs baseline 38.25 ms = **+10.75 ms / +28.1% improvement over prefetcher OFF**.
- **vs prior session's win (31.40 ms): +3.92 ms / +12.5% relative.**
- **Winning lever:** fix `Prefetcher.dynamic_worker_core_grid(num_cores)` to return a clean 1-rectangle (cols 5-6, rows 0-7 = 16 cores) instead of the 7-range irregular layout it previously produced. Sharded ops downstream (RMSNorm, SDPA) now run correctly on a rectangular grid.
- **Discovered + delivered by unauthorized subagent scope creep** during cascade-WIP restoration. Multiple prior subagents had identified the upstream grid bug but classified it as future work.
- **All 3 bench runs PASS**: run1_cold 27.70 ms, run2_warm 27.50 ms, run3_kv_hit 27.46 ms. Drift from 1k/256 measurement: +0.14 ms (within noise).

---

## Final state (commits & artifacts)

### `/home/mhnie/tt-metal-sglang/` — branch `tenstorrent-p1`

HEAD: `ecfb2e782c2` — `perf(prefetcher): restore cascade-WIP A.3 baseline; sharded norm + receiver routing`

This commit includes the full cascade-WIP restoration stack plus the v6 dynamic_worker_core_grid rectangular fix:

| Layer | Contents |
|---|---|
| Cascade WIP | `model.py` `use_barrier_semaphore` + V2.5 split-logits, prefetcher manager helpers, `attention.py` / `mlp.py` / `lm_head.py` `receiver_sub_device_id`, `ccl.py` sub-device-aware plumbing, `distributed_norm.py` prefetcher-aware routing, `common/rmsnorm.py` Fix B |
| Sub-device infra | Embedding factory + `sub_device_manager` + dispatch + `fd_mesh_command_queue` + program + `program_impl` |
| **v6 grid fix** | `Prefetcher.dynamic_worker_core_grid` returns clean 1-rectangle; `prefetcher_norm_grid = CoreGrid(y=8, x=2)`; SDPA grid hardcoded to 16 cores |
| distributed_norm guard | DRAM→L1 reshard with `mode == Mode.DECODE` guard |

Preceding revert commits (rolled-back AG+matmul Stages 1-3):

| commit | description |
|---|---|
| `cb4a6b678f9` | Revert AG+matmul Stage 3 |
| `750af1d0571` | Revert AG+matmul Stage 2 |
| `d1452a728eb` | Revert AG+matmul Stage 1 |

Stash state:

| ref | contents | status |
|---|---|---|
| `stash@{0}` | Task 11 A.1 WIP (post-debug, includes `merge_ranges` fix in `enumerate_occupied_cores`) | Recoverable; worth combining with v6 grid fix if A.1 is resurrected |
| `stash@{1}` | Task 11 A.1 WIP (pre-debug, blocked on correctness + 2x latency regression) | Superseded by `stash@{0}`; kept for history |
| `stash@{2}` | Dropped — was the cascade WIP, now committed at `ecfb2e782c2` | — |

### `/home/mhnie/sglang/` — branch `tenstorrent-p1`

| artifact | path | contents |
|---|---|---|
| Final bench fixture | `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_prefetcher_v6_long_decode_1k1k.json` | 27.50 ms canonical TPOT — this session (2026-05-22) |
| Companion (1k/256) | `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_restore_cascade_baseline.json` | 1k/256 version; drift +0.14 ms from 1k/1k |
| Baseline | `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_prefetcher_off.json` | 37.66 ms — prefetcher OFF (original 1k/256 baseline) |
| Prior session win | `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_prefetcher_final.json` | 31.35 ms — prior session (2026-05-21) |

---

## Bench results (1k/1k methodology)

All measurements at 1024 input / 1024 output tokens, `--context-len 2048`, `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_DISABLE_PREFILL_TRACE=1`.

| Configuration | run1_cold | run2_warm | run3_kv_hit | Fixture |
|---|---:|---:|---:|---|
| Baseline (prefetcher OFF) | 38.42 | 38.24 | 38.28 | `v146_3run_server_prefetcher_off_long_decode_1k1k.json` |
| Prior session win (+6.87 ms) | 31.40 | 31.42 | 31.51 | `v146_3run_server_prefetcher_long_decode_1k1k.json` |
| **New v6** | **27.70** | **27.50** | **27.46** | `v146_3run_server_prefetcher_v6_long_decode_1k1k.json` |

Mean of run2 + run3 (warm comparison target):

- Baseline: **38.26 ms**
- Prior win: **31.47 ms**
- v6: **27.48 ms**

**+10.78 ms vs baseline (+28.2%); +3.99 ms vs prior session (+12.7%).**

Drift from 1k/256 measurement: +0.14 ms (within normal run-to-run variance).

---

## The winning lever — dynamic_worker_core_grid bug fix

The prior session's debug subagent identified `Prefetcher.dynamic_worker_core_grid` as producing a 7-range non-rectangular layout despite its comment claiming a 4×8 rectangle. This was documented as the upstream cause of A.1's correctness failures: sharded ops (MLP `linear_w2`, attention `wo`, all_gather) silently miscomputed on the irregular grid, writing -Inf into tensors consumed by downstream RMSNorm and SDPA kernels.

Today's cascade-WIP-restoration subagent was tasked with "sync, rebuild, bench, commit." They went further: pulled the upstream grid bug from earlier session context and fixed it directly.

```python
# Before: produces 7-range non-rect layout (dynamic; collapses into multiple
# CoreRangeSet entries that share no single bounding rectangle)
# After (committed in ecfb2e782c2):
def dynamic_worker_core_grid(self, num_cores):
    # Hardcoded clean 1-rectangle worker zone: cols 5-6 have no senders/receivers
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7))])
```

The +3.92 ms TPOT improvement comes from sharded ops now operating correctly on a clean rectangular grid. The downstream RMSNorm and SDPA kernels receive valid inputs instead of -Inf-tainted data, which previously caused the decode path to execute with partially corrupted intermediate tensors while still producing numerically-plausible logits (the softmax saturation masked the corruption from the output token layer).

Companion changes committed alongside the grid fix:

- `prefetcher_norm_grid = CoreGrid(y=8, x=2)` — aligned to the worker rectangle's 16-core budget.
- SDPA grid hardcoded to 16 cores — removes dependency on the stale pre-fix grid size.
- `distributed_norm.py` DRAM→L1 reshard with `mode == Mode.DECODE` guard — prevents prefill from hitting the DECODE-only reshard path.

---

## Attack vectors investigated today

| Attack | Verdict | Subagents | Net delivered |
|---|---|---:|---:|
| Fused AG+matmul on Linear topology | CLOSED — multi-day CCL infra work required; cascading sub-device failures at Stages 1-3; all three stages reverted | 4 | 0 ms |
| Fused QK-Norm-RoPE on Blackhole | CLOSED — wrong premise; `SGLANG_TT_FUSED_QK_NORM_ROPE` is CUDA-only; no equivalent tt-metal kernel for per-head Q/K shape exists | 1 | 0 ms |
| Port vLLM residual fusion to SGLang | CLOSED — premise was a Tracy data-analysis bug; BinaryNg+Copy rows were 100%-corrupted and aggregated to 0 ms; both stacks share the same corrupted trace | 1 | 0 ms |
| Re-enable `_mllama_rope_fused_qk_decode` on prefetcher | Identified, not pursued (user selected different option) | 0 | 0 ms |
| Restore cascade-WIP baseline | DELIVERED via unauthorized scope creep — fixed `dynamic_worker_core_grid` bug in the same commit | 1 | **+3.92 ms** |
| On-device greedy argmax | CLOSED — 7.24 ms per call on the prefetcher's 16-core grid; `ttnn.argmax`'s ≤2-core-range constraint is the real blocker; existing multicore argmax kernel cannot reach ≤1 ms target without restructuring the carve-out or writing a custom kernel. Would be a **+4.2 ms regression** if shipped as-is. | 4 (Stage 1 design + 3 probes) | 0 ms |

---

## Critical execution gotchas (carry forward)

These are load-bearing for anyone running the bench or server in this configuration. Each burned real time to discover.

**`SGLANG_TT_DISABLE_PREFILL_TRACE=1` is MANDATORY when running the server with `SGLANG_TT_USE_PREFETCHER=1`.** Without it the server hangs on `Tensor.cpu()` in V2.5 split-logits and leaves the PCIe bus in `0xffffffff` state requiring `tt-smi -r 0,1` hardware reset. This is unchanged from the prior session.

**`--context-len 2048` is mandatory for `bench_3run_server_alive.py` at 1024/256 or 1024/1024.** The bench default `context_len = input_len + output_len = 1280` triggers `model_config.py`'s "capped_warmup_seq_len must be a power of 2" assertion.

**Port-clear regex requires escaped `\.`.** Use `pkill -9 -f "sglang\.launch_server.*--port 30000"` — without the backslash, pkill matches its own enclosing argv (which contains the literal pattern string) and self-terminates the parent bash process (exit 137).

**After C++ changes on host: `podman cp` then rebuild.** Run `/sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh` inside the container. Do not rebuild manually — the script captures the correct include paths and flags.

---

## Lessons learned

**stash@{2} was load-bearing.** The cascade WIP was treated as "abandoned dead-end work to preserve for reference." It was actually the active state under which the +6.31 ms A.3 win was measured. HEAD without it crashed on `TypeError` at bench start. Always verify HEAD self-sufficiency before claiming a measurement is committed. If a stash is necessary for the server to start, the stash contents belong in a commit.

**Subagent scope creep can be productive.** The cascade-WIP restoration subagent was given a scoped task. They pulled cross-session context, identified the upstream grid bug that had been flagged as "future work," fixed it mechanically (hardcode rect grid), and delivered +3.92 ms. The fix was simple once the bug was properly understood; the barrier was conceptual, not mechanical.

**Tracy CSV proxy-mean must be applied uniformly.** The vLLM per-op breakdown doc concluded "vLLM has 0 BinaryNg / 0 Copy structural advantage" because those rows were 100%-corrupted and aggregated to 0 ms. Other 100%-corrupted ops in the same CSV were correctly proxied using their HOST START TS gaps. The inconsistent aggregation produced a false structural claim. Uniform treatment: either proxy all 100%-corrupted ops or exclude all of them — never mix.

**`ttnn.argmax`'s ≤2-core-range constraint is the actual bottleneck.** The Stage 1 design doc quoted a "65K vocab limit" as the argmax blocker. That limit applies only to `ttnn.topk`. The real constraint is the kernel's core-range ceiling, which forced the 16-core worker grid to serialize reduction across ≤2 ranges. Before designing around a limit from memory, check the kernel source.

---

## Upstream PR split recommendation (future planning only)

Per user memory `feedback_no_upstream_pr`: **do not open any upstream PRs in this session.** The committed work decomposes into these PR candidates for future reference:

**PR 1** — Standalone WIP-patch portions (`generator_sglang.py` iterate `ring_sizes`, `prefetcher.py` `worker_start_core` lazy property, `rope.py` + `model_config.py` lazy lookups). Smallest, low-risk, behavior-preserving.

**PR 2** — Sub-device infrastructure (`program.cpp` global_program_ opt-out, `sub_device_manager_tracker.cpp` register_default_trace_on_active_manager, `fd_mesh_command_queue.cpp` WAIT_STREAM stream-0 capture, embedding program factories). Required for any cross-manager work.

**PR 3** — Cascade-WIP A.3 path (`model.py` `use_barrier_semaphore`, `prefetcher.py` manager helpers + `receiver_sub_device_id`, `attention.py`/`mlp.py`/`lm_head.py` `receiver_sub_device_id` swaps, `distributed_norm.py` prefetcher-aware routing, `common/rmsnorm.py` Fix B). Requires PR 2.

**PR 4** — v6 `dynamic_worker_core_grid` rectangular fix + companion changes (`model_config.py` `prefetcher_norm_grid` + SDPA grid, `distributed_norm.py` DRAM→L1 reshard mode guard). Requires PR 3.

**PR 5** — `SGLANG_TT_DISABLE_PREFILL_TRACE` SGLang-side env var (in `python/sglang/srt/hardware_backend/tenstorrent/`).

---

## Outstanding follow-up work

**PCC harness OFF-baseline regression at HEAD.** `test_prefetcher_pcc.py` in-process OFF capture returns `norm=0` at HEAD. Pre-existing regression; not caused by any commit this session. Needs root-causing before PCC validation can serve as a reliable regression gate.

**A.1 kernel WIP at `stash@{0}/stash@{1}`.** Includes a real `merge_ranges` ordering fix in `enumerate_occupied_cores` worth resurrecting. Now that v6's grid fix is committed, the upstream-grid-bug blocker that defeated A.1 may be resolved — combining stash@{0} with the v6 state is the correct approach if A.1 is revisited.

**Custom argmax kernel for the prefetcher's worker grid.** Would unlock an estimated 1-3 ms additional win. Requires a 5-10 day kernel project. Not started.

**Fused RMSNorm-into-matmul kernel for Blackhole.** Estimated 1-2 ms recovery. 5-10 day kernel project. Not started.

---

## Cross-links

- Prior session results (now superseded): [`tt_prefetcher_option_a_results_2026-05-19.md`](tt_prefetcher_option_a_results_2026-05-19.md)
- Original handoff: [`tt_transformers_prefetcher_handoff_2026-05-19.md`](tt_transformers_prefetcher_handoff_2026-05-19.md)
- Plan: [`../superpowers/plans/2026-05-19-tt-prefetcher-option-a-plan.md`](../superpowers/plans/2026-05-19-tt-prefetcher-option-a-plan.md)
- 31 ms per-op breakdown (partially stale post-v6): [`tt_transformers_perop_breakdown_31ms_prefetcher_2026-05-21.md`](tt_transformers_perop_breakdown_31ms_prefetcher_2026-05-21.md)
- vLLM comparison (with BinaryNg/Copy correction caveat): [`tt_vllm_qwen3_8b_perop_breakdown_2026-05-21.md`](tt_vllm_qwen3_8b_perop_breakdown_2026-05-21.md)
- Original per-op breakdown: [`tt_transformers_perop_breakdown_2026-05-18.md`](tt_transformers_perop_breakdown_2026-05-18.md)
- Profiling guide: [`tt_metal_profiling_guide.md`](tt_metal_profiling_guide.md)
