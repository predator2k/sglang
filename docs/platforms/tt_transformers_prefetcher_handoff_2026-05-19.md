# tt_transformers DRAM-prefetcher enablement — handoff (2026-05-19)

> **SUPERSEDED-BY:** [`tt_prefetcher_option_a_results_2026-05-19.md`](tt_prefetcher_option_a_results_2026-05-19.md) — final TPOT 31.35 ms via SGLANG_TT_DISABLE_PREFILL_TRACE (A.3 alternative lever); A.1/A.2 phases not landed.

This handoff captures the state of an attempt to enable the DRAM prefetcher for Qwen3-8B on 2× Blackhole P150a in the tt_transformers/SGLang path, where it should give an estimated **5-10 ms TPOT improvement (37 → 27-32 ms)**. The attempt stopped after hitting cascading C++ kernel constraints. Everything is reverted; baseline preserved. This document captures the path so the next session (mine or someone else's) can resume.

## TL;DR

- **Baseline TPOT preserved at 37.66 ms.** All in-flight patches reverted via `git checkout`; container files synced back via `podman cp`.
- **Real bugs in tt-metal's prefetcher integration were identified and fixable in Python**: gate inconsistency (`generator_sglang.py:84` uses default `ring_size=16` but the constructor's `legal_receiver_cores` searches wider), 3 hardcoded `CoreCoord(1, 0)` start_core sites that become stale post-receiver-carve, RoPE captures `start_core` at `__init__` instead of lazily at call time.
- **The non-Python blocker is L1 budget on Blackhole** combined with sharded-RMSNorm-on-Blackhole limitations: the prefetcher's per-core global CB (418 KB at `ring_size=64`) plus the P3a.2-forced unsharded RMSNorm's static CB region (~1.26 MB) exceeds 1.5 MB L1 per core. The natural escape — use sharded RMSNorm — runs into three separate C++ TT_FATAL assertions (HEIGHT_SHARDED unsupported, shard-height vs physical-height mismatch, non-rectangular core grid).
- **WIP diff (144 lines, 5 files in `tt-metal-sglang` fork) saved at:**
  `python/sglang/srt/hardware_backend/tenstorrent/patches/tt-metal-prefetcher-enablement-WIP-2026-05-19.patch`

---

## Current state (confirmed baseline)

| metric | value | source |
|---|---:|---|
| Qwen3-8B TPOT_warm canonical | **37.66 ms** | `_fixtures/v146_3run_server_prefetcher_off.json` (run2_warm_new_prompt) |
| `SGLANG_TT_USE_PREFETCHER=1` baseline | 37.6 ms (no delta — gate blocks) | `_fixtures/v146_3run_server_prefetcher_on.json` |
| Per-op breakdown (Tracy CSV) | Matmul 41%, BinaryNg 20%, Copy 12%, Collectives 13%, Norm 10% | `docs/platforms/tt_transformers_perop_breakdown_2026-05-18.md` |
| Theoretical floor (DRAM-bound) | ~8 ms | model size 1.5GB / 1TB/s aggregate BW × 2 chips |

Sessions where this work was done: 2026-05-18, 2026-05-19 (early hours). Container used: `p3a-ngram` (tt-metal baked-in image; modifications go in via `podman cp`).

---

## What the prefetcher actually does and why it would help

`/home/mhnie/tt-metal-sglang/models/tt_transformers/tt/prefetcher.py` (≈700 LOC) is Tenstorrent's hand-rolled DRAM-to-L1 weight prefetcher. It sets up:

- A small number of "sender" cores that read weights from DRAM (1-2 per chip, MUX-safe rows).
- A larger ring of "receiver" cores that copy from senders into a per-core "global CB" via NoC multicast.
- Per-layer matmul ops then read weights from this global CB (which sits in L1) instead of DRAM, hiding DRAM latency behind compute.

For Qwen3-8B 2× Blackhole P150a, the **CB size per worker core** depends on `ring_size`:

| ring_size | bytes_per_core | fits 850 KB cap? |
|---:|---:|:---:|
| 8 | 3,342,336 | ✗ |
| 16 (default) | 1,671,168 | ✗ |
| 24 | 1,253,376 | ✗ |
| **64** | **417,792** | **✓** |
| 80 | 334,234 | ✓ |

Only `ring_size ∈ {64, 80}` make the per-core L1 fit. The constructor's `legal_receiver_cores = [1, 2, 3, 8, 10]` lets it search these. The outer gate at `generator_sglang.py:84` hardcodes the default `ring_size=16` → rejects every receiver-count except those that already passed at 16.

Estimated upside: per-decode forward, matmul fires 144 times (4 per layer × 36 layers), each DRAM-bound by reading ~1 GB BFP8 weights total. The prefetcher overlaps DRAM read with the previous matmul's compute. The theoretical save is **all of the DRAM-load wait per-matmul**, bounded by what's actually overlap-shadowable. Memory entry `tenstorrent-tt-metal-fork.md` claims 5-10 ms TPOT was the original design target; my prior agent estimated the same.

---

## What we tried (7 iterations, in order)

| iter | what changed | result |
|---|---|---|
| 1 | `generator_sglang.py`: iterate `legal_receiver_cores` in gate | Gate now passes at ring_size=64. Constructor runs, allocates 180 prefetched tensors, MUX-safe mapping applied. RoPE then crashes: `TT_FATAL @ work_split.cpp:162 sub_core_grids.contains(start_core)` — hardcoded `CoreCoord(1, 0)` is no longer inside the worker grid (carved by receiver remap) |
| 2 | `rope.py:781`: store `prefetcher.worker_start_core` instead of `(1,0)` | Same crash — stored at `__init__` time, BEFORE `prefetcher.init(Mode.DECODE)` carves receivers. Stale by the time `get_rot_mats` runs |
| 3 | `rope.py:966`: read `prefetcher.worker_start_core` LAZILY at call time | RoPE crash gone. Next crash: `TT_THROW @ program.cpp:1403`: "Statically allocated circular buffers in program 66 clash with L1 buffers on core range [0-0]. L1 buffer allocated at 1124096 and static circular buffer region ends at 1257984." LayerNorm program at core (0,0) clashes with prefetcher's global CB allocation |
| 4 | `prefetcher.py:381`: subtract (0,0) from `all_worker_cores_range_set` | New crash: `TT_FATAL @ program.cpp:1811 num_intersections == num_cores` "Kernel group cores do not match sub device cores". Excluding (0,0) from the worker grid breaks the sub_device-to-kernel-group invariant (other ops register sub_device_id pointing to a set that now doesn't match the kernel's intended cores) |
| 5 | Revert (0,0) exclusion. `distributed_norm.py`: skip P3a.2-forced unsharded path when prefetcher is on. Reshard DRAM input to HEIGHT-sharded before sharded RMSNorm | `AttributeError: 'MemoryConfig' object has no attribute 'is_dram'` — wrong API |
| 6 | Fix API: `x.memory_config().buffer_type == ttnn.BufferType.DRAM` | `TT_FATAL @ layernorm_device_operation.cpp:164` "Height sharded inputs are not supported" — sharded RMSNorm rejects HEIGHT_SHARDED with a `// TODO: Add support for this` comment at line 161 |
| 7 | Switch to WIDTH-sharded with `shape=(TILE_SIZE, head_dim/N)` | `TT_FATAL @ tensor_spec.cpp:143` "Shard height 32 must match physical height 128 for width sharded" — WIDTH-sharded requires `shard_h = full_tensor_h` |
| (would-be 8) | Use full height + dynamic effective_cores to make worker grid divide head_dim evenly | `TT_FATAL @ layernorm_device_operation.cpp:185` "Sharded layernorm does not support non-rectangular core grids. The shard spec grid has 64 cores but its bounding box spans 88 cores" — third sharded RMSNorm restriction. Stopped here |

---

## The WIP patch (144 lines, 5 files)

Saved at `python/sglang/srt/hardware_backend/tenstorrent/patches/tt-metal-prefetcher-enablement-WIP-2026-05-19.patch`.

Apply from `/home/mhnie/tt-metal-sglang/` root:

```bash
cd /home/mhnie/tt-metal-sglang
git apply /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-metal-prefetcher-enablement-WIP-2026-05-19.patch
# then sync into the container:
for f in models/tt_transformers/tt/prefetcher.py \
         models/tt_transformers/tt/rope.py \
         models/tt_transformers/tt/model_config.py \
         models/tt_transformers/tt/generator_sglang.py \
         models/tt_transformers/tt/distributed_norm.py; do
  podman cp "/home/mhnie/tt-metal-sglang/$f" "p3a-ngram:/tt-metal/$f"
done
```

### Contents (file-by-file)

| file | what the patch does | status |
|---|---|---|
| `generator_sglang.py:82-94` | Iterate `legal_receiver_cores` ring_sizes [8,16,24,64,80] in the outer prefetcher-supported gate. Accept if ANY pass. | **Standalone-correct**; even without the rest of the patch this is an unambiguous bug fix that mirrors what the constructor does |
| `prefetcher.py:Prefetcher` | Add `worker_start_core` property returning `ranges[0].start` of `all_worker_cores_range_set` (fallback `CoreCoord(1, 0)`) | **Standalone-correct** — defensive helper for the start_core sites |
| `rope.py:966` | Read `self.prefetcher.worker_start_core` lazily at `get_rot_mats` time (was: stored at `__init__` time at line 781, where post-init carve makes it stale) | **Standalone-correct** — fixes a latent bug for any model that activates the prefetcher |
| `model_config.py:1541, 1789` | Same lazy lookup in `get_attn_sdpa_decode_program_config` and `get_attn_sdpa_output_mem_config` | **Standalone-correct** |
| `distributed_norm.py:108-148` | When prefetcher is on, skip the P3a.2 force-unsharded path; reshard DRAM input to L1 WIDTH-sharded on the worker grid, then call sharded RMSNorm | **WIP / blocked** — hits the 3 sharded-RMSNorm kernel constraints (HEIGHT-sharded unsupported, shard-height=physical-height, rectangular-grid-only) |

### Standalone usefulness

The first 4 patches above fix real bugs even without enabling the prefetcher. They are upstreamable as separate small PRs. They don't change behavior when `use_prefetcher=False`.

The 5th patch (`distributed_norm.py`) only triggers when prefetcher is on, so it is also opt-in. Currently it is **broken** by the sharded-RMSNorm kernel constraints documented above; do not enable it without the kernel work in Option A.

---

## Option A — continue the kernel work over multiple sessions

**Scope: 5-15 days of focused tt-metal kernel work plus rebuild cycles.**

The blocking C++ kernel work, in dependency order:

### A.1 Non-rectangular core grid support in sharded RMSNorm

- **File**: `/home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:178-185`
- **Block**: TT_FATAL on `shard_spec.grid.num_cores() != bbox_num_cores`
- **Approach**: remove the TT_FATAL AND modify the program factory + per-core kernel data flow to enumerate only-the-occupied-cores correctly (not the full bounding box). Files: `sharded_layernorm_factory_helpers.cpp/hpp`, `device/kernels/dataflow/writer_unary_sharded_ln.cpp`, `device/kernels/dataflow/reader_*_sharded_ln_*.cpp`.
- **Why it's hard**: the multicast NoC routing in the sharded RMSNorm assumes a rectangular bbox so the sender→receiver topology is regular. Non-rectangular needs either (a) per-core explicit NoC route tables (more setup work but works), or (b) "fill the holes" with no-op cores (wastes parallelism). Both require kernel changes.
- **Effort**: 2-4 days including PCC verification on a known-rectangular config first.

### A.2 HEIGHT_SHARDED support in sharded RMSNorm

- **File**: same `layernorm_device_operation.cpp:162`
- **Block**: TT_FATAL on `memory_layout == HEIGHT_SHARDED`. Has a `// TODO: Add support for this (should be similar to interleaved)` comment from Tenstorrent.
- **Approach**: implement the height-sharded variant. Likely a new program factory function. The TODO suggests it should mirror the interleaved variant since each core has a full row of data.
- **Effort**: 1-2 days.
- **Note**: only needed if you can't make the WIDTH-sharded path work (which currently requires shard_h=physical_h — fine, but requires the head_dim to divide evenly into TILE-aligned chunks per worker, which we can satisfy with a rectangular sub-grid).

### A.3 L1 budget tuning of the unsharded RMSNorm

- Alternative to A.1/A.2: just shrink the unsharded RMSNorm's per-core static CB so it fits alongside the prefetcher's 418 KB.
- **File**: `LayerNormDefaultProgramConfig` configuration / `layernorm_op_multi_core.cpp`
- **Approach**: smaller `block_h` / `block_w` parameters; or split the work across MORE cores (the default may underutilize). Each parameter trade-off needs PCC validation.
- **Effort**: 1-3 days; lower-risk than A.1 because it's tuning not new code, but bounded by what the existing kernel allows.

### A.4 Cycle setup

Currently each iteration is ~30-60 min of tt-metal rebuild + ~3 min server warm-up. The handoff session should set up:
- A "kernel-only rebuild" script that doesn't rebuild tt-metal toolchain, just the changed `.cpp`. The `rebuild_tt_xla_stack.sh` precedent in `python/sglang/srt/hardware_backend/tenstorrent/scripts/` is for tt-xla; an analogous script for tt-metal-sglang would help.
- A faster smoke test that doesn't go through the SGLang server — e.g., a standalone pytest in `models/tt_transformers/tests/` that runs decode_forward with `use_prefetcher=True` and times the first 5 decodes.

### Resumption checklist

1. `git apply` the WIP patch (above).
2. Pick A.1 or A.2 as the first kernel target (A.1 is the immediate blocker; A.2 simpler).
3. Implement, build, smoke. Expect 1-3 more rounds of secondary crashes per kernel target.
4. Validate PCC against the non-prefetcher baseline.
5. A/B benchmark: TPOT delta vs `_fixtures/v146_3run_server_prefetcher_off.json` baseline 37.66 ms.

---

## Option B — hand off the WIP patch to a tt-metal kernel engineer

**Scope: a few hours of email/PR thread for handoff; engineer's own time after that.**

Targets: someone familiar with tt-metal's sharded LayerNorm program factory and NoC multicast topology. Internal Tenstorrent engineering, or community tt-metal contributors.

### What to send

1. This handoff document (`docs/platforms/tt_transformers_prefetcher_handoff_2026-05-19.md`).
2. The WIP patch (`patches/tt-metal-prefetcher-enablement-WIP-2026-05-19.patch`).
3. The crash log captured at iter 8 (`/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/bench_3run_pref_v7/Qwen3-8B.server.log`).
4. The breakdown profile (`_fixtures/cpp_device_perf_report_tt_transformers_v6.csv`) showing the prefetcher's target workload (matmul = 41% of decode).
5. The original Tracy recipe (`docs/platforms/tt_metal_profiling_guide.md`) so the engineer can reproduce.

### The clean PR scope (vs the messy WIP)

The first 4 file changes in the WIP patch (gate fix, `worker_start_core` property, RoPE lazy lookup, SDPA start_core) are **upstream-mergeable as standalone bug fixes** that don't depend on the L1/sharded-RMSNorm work. Suggest splitting:

- **PR 1 (`tt-metal-prefetcher-gate-iterate-ring-sizes`)**: just `generator_sglang.py` + the `worker_start_core` property in `prefetcher.py`. Tag as a bug-fix for "prefetcher gate uses ring_size=16 default but the constructor searches wider; gate is more restrictive than the constructor."
- **PR 2 (`tt-metal-rope-sdpa-start-core-prefetcher-aware`)**: `rope.py` + `model_config.py` changes. Tag as "RoPE and SDPA decode hardcode `CoreCoord(1, 0)` which becomes stale once prefetcher carves receivers from the worker grid."
- **PR 3 (or kernel-team issue)**: the `distributed_norm.py` reshard plus the C++ kernel work for non-rectangular / HEIGHT-sharded RMSNorm. This is the big one and should reference this handoff document for context.

---

## Option C — pivot to a different bottleneck

**Scope: 1-3 days per attempt, depending on which target.**

The original profile (Matmul 41%, BinaryNg 20%, Copy 12%, Collectives 13%, Norm 10%) has other levers. The prefetcher is a 5-10 ms lever; here's what else is on the menu:

| target | estimated upside | concrete next step | effort |
|---|---:|---|---|
| **AGMM Linear-topology** | 2-3 ms | the C++ fused-AG program factory hard-asserts no Linear+fuse (`all_gather_async_default_program_factory.cpp:317-318`). Removing the assert AND extending the kernel to compute the Linear-ring scatter pattern. Different kernel, different team. | 3-7 days |
| **Tune un-fused fallback AGMM** | 1-3 ms | `attention.py:811-812` hardcodes `chunks_per_sync=10`, `num_workers_per_link=2`. The AGMM-fused config uses different tuned values (`ATTN_AGMM_CONFIG`). Bring the un-fused defaults closer to the fused config's tuning. | 1 day, low risk |
| **Investigate Copy** | uncertain (estimate was based on suspect data) | The 12% "Copy" line was from extrapolating 12 valid samples out of 962 — the in-trace clock-domain bug made the estimate wrong. Need a re-profile with `TT_METAL_PROFILER_DISPATCH=0` or a tt-metal-level fix for in-trace timing reliability. | unknown; depends on the profile findings |
| **RMSnorm fusion into matmul prologue** | 4-7 ms | Custom matmul kernel that reads weights through RMS-norm in-line, avoiding the separate RMSnorm op altogether. tt-metal C++ work. | 5-10 days |
| **EAGLE-3 acceptance rate** | TPOT/decoded → 1.5-2× via better drafts | Already running on 2× P150a (`tenstorrent-eagle3-working` memory entry). Could push acceptance higher with better draft training or tree-search tuning. | varies (workload-dependent) |

The lower-effort/lower-risk pivots are **"Tune un-fused fallback AGMM"** (1 day, ~1-3 ms) and **"Investigate Copy properly"** (diagnostic, may unlock new attacks). Both can run as their own subagent investigations like the first round.

---

## Reference artifacts

| artifact | path | what's in it |
|---|---|---|
| WIP patch | `python/sglang/srt/hardware_backend/tenstorrent/patches/tt-metal-prefetcher-enablement-WIP-2026-05-19.patch` | 144-line diff across 5 files in `tt-metal-sglang` fork |
| Crash logs | `_fixtures/bench_3run_prefetcher_*/Qwen3-8B.server.log` | Each iteration's server log with TT_FATAL backtraces |
| Bench results | `_fixtures/v146_3run_server_prefetcher_*.json` | A/B baselines and post-patch attempts |
| Profile baseline | `_fixtures/cpp_device_perf_report_tt_transformers_v6.csv` | 32K-row per-op CSV used for the breakdown |
| Profiling guide | `docs/platforms/tt_metal_profiling_guide.md` | How to reproduce the profile (the 4-key recipe) |
| Per-op breakdown | `docs/platforms/tt_transformers_perop_breakdown_2026-05-18.md` | Where the 5-10 ms upside estimate comes from |
| Memory: prefetcher engagement check | `tenstorrent-prefetcher-gated-off-qwen3-8b.md` (if exists; otherwise this handoff is the source) | Why ring_size=16 default fails for Qwen3-8B |
| Memory: tt_transformers TPOT decomposition | `tenstorrent-tt-transformers-tpot-decomposition.md` | 99.9% of TPOT is inside `decode_forward` |
| Memory: tt-metal fork pointer | `tenstorrent-tt-metal-fork.md` | Where the tt-metal-sglang fork lives + branch info |
| Memory: TT-XLA not perf path | `tenstorrent-tt-xla-not-perf-path.md` | Strategic context: tt_transformers is the perf path |

---

## What's specifically NOT in this handoff

- A working enablement of the prefetcher for Qwen3-8B. That requires the kernel work.
- An A/B measurement of the 5-10 ms claim. The number is theoretical (DRAM-bandwidth analysis). Validation requires the kernel work to land first.
- A timeline. Real kernel work depends on the engineer's availability and tt-metal build infrastructure speed.

## My recommendation

If TPOT optimization is genuinely the priority and 5-15 days of kernel work is acceptable: **Option A**, with a kernel engineer fully briefed via this document.

If it's not the priority right now: **Option C**, specifically "Tune un-fused fallback AGMM" — 1 day, ~1-3 ms upside, no kernel changes. Gets a measurable win on the books before investing in the harder paths.

**Option B** is the right move if there's a Tenstorrent engineer available to take it; the standalone-correct portions of the WIP patch are real bug fixes that should land upstream regardless.
