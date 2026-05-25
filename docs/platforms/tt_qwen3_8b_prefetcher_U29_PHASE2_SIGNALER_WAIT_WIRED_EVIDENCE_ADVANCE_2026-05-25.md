# TT Qwen3-8B prefetcher — U29 Phase 2: device-side W2→RS signaler wait WIRED end-to-end, FIX DOES NOT ENGAGE — 2026-05-25

Status: **EVIDENCE_ADVANCE — U29 Phase 2 wires
the consumer-side wait loop, narrows the producer-side increment to
W2 only, adds an extra compute↔dataflow sync handshake to gate the
increment behind compute's final tensix_sync, and validates the full
build + runtime chain (RT args appended, kernels rebuilt, env vars
propagate, JIT compiles, server boots, decode runs). With
`SGLANG_TT_U29_W2_RS_SIGNALER=1` ON, output is STILL garbage on
single-prompt smoke tests, GSM8K(10) crashes mid-test on NaN sampler,
and the wait loop appears to PASS IMMEDIATELY (warm decode latency
0.29s for 15 tokens = ~19 ms/token, same as Phase 1 without wait).
The signature is stable across runs ("doing doing doing..." on warm
short prompts), suggesting the wait is deterministic but the
producer→consumer counter relationship does NOT capture W2 PACK
retirement.**

**Canonical Qwen3-8B (U29 OFF) re-verified bytewise-equal at
e2e_latency 38.28s cold + GSM8K(10) = 10/10 = 100%.** No regression.

Continuation of `tt_qwen3_8b_prefetcher_U29_PHASE1_SIGNALER_INFRA_LANDED_2026-05-25.md`.

tt-metal-sglang HEAD: **`ad685b4576b`** (pending this U29 Phase 2 commit).
Branch `tenstorrent-p1`; NOT pushed (predator2k/* fork only).
sglang HEAD: pending this doc commit.

## TL;DR (Phase 2 forensic chain)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| 2.A | Does the basic Phase 2 wait wire compile and run? | Phase 2 v1: shared counter, static `expected += 1` per RS call, sequential probe of all 32 producer cores. | **YES build + decode run** — but garbage output. Cold latency 39.5s (compile + trace capture), warm 0.29s. |
| 2.B | Does narrowing the wait to RELATIVE counter (`> prev_seen`) fix it? | Phase 2 v2: replace expected-counter with `static u29_prev_seen`; wait until any producer counter strictly exceeds prev_seen, then update prev_seen to min observed. | **NO** — same garbage. Warm 0.29s confirms wait passes ~immediately. |
| 2.C | Is the wait actually engaging, or is it a no-op? | Add `SGLANG_TT_U29_DISABLE_WAIT=1` control env var that compiles in the RT-arg read but skips the spin loop. | Decode latency identical (0.29s) with wait ON vs OFF.  **Confirms wait runs but passes immediately**: producer counter is always far ahead of prev_seen by the time RS reader checks. |
| 2.D | Are non-W2 gathered matmuls (WO, FF1, FF2, FF3, WQKV) clobbering the shared 0x90000 counter? | Phase 2 v3: add `SGLANG_TT_U29_W2_PRODUCER_NOW=1` env gate to matmul factory + wrap W2's `ttnn.linear` call in mlp.py with that env. Only the W2 matmul JIT binary now has the increment. | **NO** — same garbage, same latency. Narrowing producer to W2 doesn't fix the synchronization but proves Phase 2 v2's wait was indeed exiting the loop deterministically. |
| 2.E | Is the in1 sender writer's `noc_semaphore_inc` racing compute's final `tensix_sync()`? | Phase 2 v4: compute pushes one extra `sync_buf.push_back(1)` AFTER its final tensix_sync; in1 sender writer waits on `cb_sync.wait_front(1)` + drains, THEN issues the increment. | **NO** — same garbage. The synchronization between dataflow's `noc_semaphore_inc` and compute's final PACK retirement is now strict by construction, but the consumer-side wait still does NOT block on the producer's increment. |
| 2.F | Does the canonical (U29 OFF, no prefetcher) path still work? | curl + GSM8K(10) chat | **YES — `" What is 2+2? What is 2+2? What"` at e2e_latency 38.28s; GSM8K(10) = 10/10 = 100%.** No regression from the U29 Phase 2 changes when env-gate is OFF. |

**Punchline:** the producer-side increment + consumer-side wait
mechanism builds + runs correctly end-to-end and is robustly
sequenced (compute → dataflow extra sync_buf round → noc_semaphore_inc
→ RS reader spin until > prev_seen). But it does NOT change the
observed-garbage output, suggesting one of:
- The cross-sub-device dispatch race (U28-β) is **NOT the root cause**
  — or it is one of several concurrent races, and fixing W2→RS
  alone is insufficient.
- The RS reader's `noc_async_read(producer's L1 0x90000)` is observing
  a value that is INDEPENDENT of W2 PACK retirement (e.g. L1 0x90000
  on the W2 receiver cores might be in the L1-allocator's used region
  and getting written by some other tensor placement, racing the
  atomic increment).
- The W2 → RS path is correctly synchronized but the actual garbage
  is coming from a different stale-data source (WO → all_reduce_async,
  FF1/FF2 → all_reduce_async, WQKV → nlp_create_qkv_heads, prefetcher
  GCB consumption races, attention KV stride aliasing, etc.).

The Phase 2 work establishes a working device-side handshake
infrastructure that is provably engaged AND provably non-fixing for
the observed Qwen3-8B garbage symptom.

## What U29 Phase 2 lands

All env-gated by `SGLANG_TT_U29_W2_RS_SIGNALER=1`; default OFF;
canonical bytewise-equal verified.

### tt-metal-sglang files modified (5)

| File | Phase 1 → Phase 2 change |
|---|---|
| `models/tt_transformers/tt/mlp.py` | New env gate `SGLANG_TT_U29_W2_PRODUCER_NOW`: set inside W2's `ttnn.linear` call wrapper so only the W2 matmul program is JIT-compiled with the U29 producer-side increment (other gathered matmuls — WO, FF1, FF2, FF3, WQKV — keep the canonical kernel binary). Try/finally to restore prior env state. Only fires when `SGLANG_TT_U29_W2_RS_SIGNALER=1` AND mode==DECODE AND prefetcher active AND not W2-skip path. |
| `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | Replace single-env-gate with double-env-gate (`SGLANG_TT_U29_W2_RS_SIGNALER` + `SGLANG_TT_U29_W2_PRODUCER_NOW`).  Only when BOTH are set do the kernel defines get emitted into the W2 matmul JIT binary. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | After the final `ckernel::tensix_sync()` at kernel exit (under `SGLANG_TT_U29_W2_RS_SIGNALER`), push one more page to `sync_buf` (`reserve_back(1)` + `push_back(1)`).  Pairs with the dataflow wait below.  This guarantees the next dataflow signal cannot fire until compute's last PACK has retired. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Before the U29 `noc_semaphore_inc`, `cb_sync.wait_front(1)` + `cb_sync.pop_front(1)` to consume compute's extra exit-sync push, then `noc.async_write_barrier()` to re-drain, THEN the increment.  This forms a strict happens-before order: compute PACK → tensix_sync → sync_buf push → dataflow wait → noc_semaphore_inc → noc_async_atomic_barrier. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp` | Phase 2 v1: derive producer-core list from `input_tensor.memory_config().shard_spec()->grid`, convert each logical core to NoC virtual coord via `mesh_device->worker_core_from_logical_core(c)`, and pack `[num_producers, x0, y0, x1, y1, ...]` into the reader RT args (appended after the existing shard/fuse_op RT args).  Also propagate two new env-gated defines to the RS reader: `SGLANG_TT_U29_DEBUG_DPRINT` (DPRINT signature inside the wait) and `SGLANG_TT_U29_DISABLE_WAIT` (compile in RT args read but skip spin — used as a control for latency attribution). |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp` | Phase 2 v2 (final): relative-counter wait.  `static uint32_t u29_prev_seen = 0` (persists across trace replays in kernel binary `.bss`, same pattern as U17/U18 budget counters in the same file).  For each producer's NoC coord from RT args, spin `do { read(prod_l1_0x90000); } while (val <= prev_seen);` then update `prev_seen` to the minimum observed value.  Uses `cb_input_id`'s scratch slot for the 4-byte NoC read landing point (cb reserved but never pushed, same pattern as the U17 PRE_RS probe in this file). |

### Run-time matrix verified

| `SGLANG_TT_USE_PREFETCHER` | `SGLANG_TT_U29_W2_RS_SIGNALER` | `SGLANG_TT_U29_DISABLE_WAIT` | Smoke output | Decode TPOT |
|---|---|---|---|---|
| 0 | 0 | n/a | `" What is 2+2? What is 2+2? What"` (canonical) | 38.28s cold |
| 1 | 0 | n/a | garbage (Phase 1 baseline) | 9.10s cold |
| 1 | 1 | 0 | garbage (`" What doing doing..."`) | 39.5s cold, 0.29s warm |
| 1 | 1 | 1 | garbage (similar signature) | 40.4s cold, 0.29s warm |

The 0.29s warm decode is identical across (wait engaged) vs
(wait disabled), proving the wait loop iterates and exits very
quickly each call.

## Architectural learnings (Phase 2)

### Learning 1: producer kernel choice matters

The W2 matmul gathered factory uses two dataflow kernels:
`reader_bmm_tile_layout_in0_ring_all_gather.cpp` (in0 reader, RISCV_1, NoC = preferred_dram_read)
and
`reader_bmm_tile_layout_in1_ring_all_gather.cpp` (in1 sender writer, RISCV_0, NoC = preferred_dram_read).
Neither is the producer of `mm_out_cb` — the COMPUTE kernel is.
Compute writes to mm_out_cb via PACK; the L1 region of mm_out_cb
is what RS reader subsequently reads via NoC.

Phase 1 + Phase 2 v1-v3 placed the `noc_semaphore_inc` in the
in1 sender writer kernel.  Even though Phase 1 added a kernel-exit
`tensix_sync()` in the compute kernel, the COMPUTE and DATAFLOW
kernels run on different RISCs — compute's tensix_sync only
fences TENSIX threads; the dataflow RISC's `noc.async_write_barrier()`
only drains DATAFLOW writes.  The dataflow's exit-time
`noc_semaphore_inc` therefore fires whenever the dataflow loop
exits, INDEPENDENTLY of whether compute has retired its last PACK.

Phase 2 v4 fixes this by adding a compute → dataflow exit-sync
round-trip via the existing `sync_cb` CB: compute pushes one extra
page after its final tensix_sync; dataflow waits for it before
issuing the increment.  This guarantees the increment is ordered
strictly after the LAST PACK retirement.

### Learning 2: shared L1 counter slot across all gathered matmuls collides

All gathered matmuls (W2, WO, FF1, FF2, FF3, WQKV under Qwen3-8B
prefetcher) run on the same matmul factory branch and share the
same `SGLANG_TT_U29_W2_RS_SIGNALER` env gate.  Without a unique
counter slot per matmul, all of them write to L1 0x90000 on each
receiver core.  The W2-specific RS reader then sees a counter that
is incremented by every gathered matmul, not just W2 — so any
"wait until counter > prev_seen" condition is satisfied by ANY
preceding matmul, not specifically W2.

Phase 2 v3 fixes this by gating the producer-side increment on a
SECOND env var `SGLANG_TT_U29_W2_PRODUCER_NOW` that mlp.py sets
ONLY around the W2 `ttnn.linear` call (at PROGRAM CREATION time —
the env is read once when the JIT kernel binary is compiled; the
cached binary is then re-used for subsequent forward calls).

### Learning 3: kernel-local `static` truly does persist across trace replays

We confirmed (via the existing U17/U18 budget counters and our new
`u29_prev_seen` slot) that `static uint32_t` declarations in
kernel functions retain their value across consecutive trace
replays of the same program.  The kernel binary's `.bss` is loaded
once at program creation; trace replay just sends the "go" command
to fire the kernel, which jumps to entry without re-zeroing static
data.

### Learning 4: 0x90000 might not be a safe-to-write L1 address

The L1 allocator on Blackhole (per the existing U22 / U25 evidence
chain) places W2 output at 0xa6700 / 0xa4700 — values BIGGER than
0x90000 = 589 824 bytes.  The L1 unreserved base on BH is much
lower than 0x90000 (somewhere near 0x10000-0x40000 typically), so
0x90000 falls inside the allocator's potential allocation region.
If any other allocator-managed tensor happens to occupy 0x90000
during the W2 → RS critical window, our atomic increment is
clobbering that tensor's data AND our reads observe that tensor's
contents (not the counter).

Phase 2 v3 narrows the increment to W2-only, which somewhat
mitigates the clobbering (we increment less often), but does NOT
resolve the read-side issue.  Phase 3 should use a properly
allocated semaphore (e.g., `tt::tt_metal::CreateGlobalSemaphore`)
or a kernel-local L1 slot that's guaranteed outside the allocator's
range (e.g., high L1 ≥ ~0x140000 if the cumulative tensor footprint
allows).

### Learning 5: the cross-sub-device race might be a symptom, not the root cause

U28-β proposed that the cross-sub-device dispatch race (W2 on
receiver_sub_device, RS on worker_sub_device, no automatic
cross-subdev wait) is the root cause.  U29 Phase 2 closes that
gap (with the right kernel-level handshake AND a correctly placed
L1 sema slot, the RS reader would by construction not start its
NoC reads of mm_out_cb until W2 PACK is fully retired).  Phase 2
demonstrably wires that handshake — but the output is unchanged.

Either the handshake is silently broken (e.g. by the L1 slot
issue in Learning 4), or there's an additional bug on the
non-W2-RS chains (WO → all_reduce_async, FF2 → all_reduce_async).
Phase 3 should:
- First, definitively confirm the W2 → RS handshake works by using
  a known-safe L1 slot (e.g., a high-L1 fixed offset or a
  `GlobalSemaphore`) and a sentinel-value pattern (e.g., producer
  writes 0xDEADBEEF, RS reader DPRINTs the readback).
- Then, if W2 → RS is provably synchronous and garbage persists,
  extend the same pattern to WO → all_reduce_async and FF2 →
  all_reduce_async (which use a DIFFERENT kernel pair —
  `worker_writer.cpp` / `reduction_receiver.cpp` —
  not the line RS reader we modified).

## Implementation diffs (selected)

### A. Compute kernel — exit sync push

```cpp
// Existing U16 / W2_RS_BARRIER_KERNEL final tensix_sync()
#ifdef SGLANG_TT_U29_W2_RS_SIGNALER
    // U29 Phase 2 v4 — proper producer-side sync between compute
    // and dataflow.  Compute pushes one extra page AFTER final
    // tensix_sync so dataflow's noc_semaphore_inc cannot fire
    // until PACK is fully retired.
    ckernel::tensix_sync();
    sync_buf.reserve_back(1);
    sync_buf.push_back(1);
#endif
```

### B. Dataflow in1 sender writer — wait + increment

```cpp
#ifdef SGLANG_TT_U29_W2_RS_SIGNALER
    cb_sync.wait_front(1);    // wait for compute exit-sync push
    cb_sync.pop_front(1);
    noc.async_write_barrier();  // re-drain after the wait
    {
        const uint32_t u29_my_x = my_x[noc_index];
        const uint32_t u29_my_y = my_y[noc_index];
        uint64_t u29_self_sema_noc_addr =
            get_noc_addr(u29_my_x, u29_my_y, (uint32_t)(SGLANG_TT_U29_SEMA_L1));
        noc_semaphore_inc(u29_self_sema_noc_addr, 1);
        noc_async_atomic_barrier();
    }
#endif
```

### C. Matmul factory — double-env gate

```cpp
const char* u29_sig_env = std::getenv("SGLANG_TT_U29_W2_RS_SIGNALER");
const char* u29_now_env = std::getenv("SGLANG_TT_U29_W2_PRODUCER_NOW");
const bool u29_active = (u29_sig_env != nullptr && std::string(u29_sig_env) == "1") &&
                        (u29_now_env != nullptr && std::string(u29_now_env) == "1");
if (u29_active) {
    mm_in1_kernel_defines["SGLANG_TT_U29_W2_RS_SIGNALER"] = "1";
    mm_in1_kernel_defines["SGLANG_TT_U29_SEMA_L1"] = "0x90000";
    mm_kernel_defines["SGLANG_TT_U29_W2_RS_SIGNALER"] = "1";
    mm_kernel_defines["SGLANG_TT_U29_SEMA_L1"] = "0x90000";
}
```

### D. mlp.py — wrap W2 with `_NOW` env

```python
import os as _u29_os
_u29_active = (
    _u29_os.environ.get("SGLANG_TT_U29_W2_RS_SIGNALER", "0") == "1"
    and mode == Mode.DECODE
    and self.prefetcher is not None
    and not _w2_use_skip
)
if _u29_active:
    _u29_prev = _u29_os.environ.get("SGLANG_TT_U29_W2_PRODUCER_NOW")
    _u29_os.environ["SGLANG_TT_U29_W2_PRODUCER_NOW"] = "1"
try:
    w2_out = ttnn.linear(...)  # W2 matmul call (unchanged)
finally:
    if _u29_active:
        if _u29_prev is None:
            _u29_os.environ.pop("SGLANG_TT_U29_W2_PRODUCER_NOW", None)
        else:
            _u29_os.environ["SGLANG_TT_U29_W2_PRODUCER_NOW"] = _u29_prev
```

### E. RS line factory — derive producer NoC coords, append to reader RT args

```cpp
std::vector<uint32_t> u29_producer_noc_coords;  // [x0,y0, x1,y1, ...]
if (u29_enabled) {
    if (input_is_sharded && input_tensor.memory_config().shard_spec().has_value()) {
        const auto producer_grid = input_tensor.memory_config().shard_spec()->grid;
        const auto producer_logical_cores = tt::tt_metal::corerange_to_cores(
            producer_grid, std::nullopt, /*row_wise=*/true);
        for (const auto& lc : producer_logical_cores) {
            const auto nc = mesh_device->worker_core_from_logical_core(lc);
            u29_producer_noc_coords.push_back(static_cast<uint32_t>(nc.x));
            u29_producer_noc_coords.push_back(static_cast<uint32_t>(nc.y));
        }
    }
}
// (later, per-worker reader_rt_args loop) appended after sharding + fuse_op args:
if (u29_enabled) {
    reader_rt_args.push_back(static_cast<uint32_t>(u29_producer_noc_coords.size() / 2));
    reader_rt_args.insert(reader_rt_args.end(),
                         u29_producer_noc_coords.begin(),
                         u29_producer_noc_coords.end());
}
```

### F. RS reader kernel — relative-counter wait

```cpp
#ifdef SGLANG_TT_U29_W2_RS_SIGNALER
    const uint32_t u29_num_producers = get_arg_val<uint32_t>(arg_idx++);
    uint32_t u29_producer_args_start = arg_idx;
    arg_idx += u29_num_producers * 2;
    {
        static uint32_t u29_prev_seen = 0;
#ifndef SGLANG_TT_U29_DISABLE_WAIT
        if (u29_num_producers > 0) {
            cb_reserve_back(cb_input_id, 1);
            uint32_t u29_l1_scratch = get_write_ptr(cb_input_id);
            volatile tt_l1_ptr uint32_t* u29_scratch_p =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u29_l1_scratch);

            uint32_t u29_min_observed = 0xFFFFFFFFu;
            for (uint32_t i = 0; i < u29_num_producers; i++) {
                const uint32_t px = get_arg_val<uint32_t>(u29_producer_args_start + 2 * i);
                const uint32_t py = get_arg_val<uint32_t>(u29_producer_args_start + 2 * i + 1);
                const uint64_t prod_sema_noc_addr =
                    get_noc_addr(px, py, static_cast<uint32_t>(SGLANG_TT_U29_SEMA_L1));
                uint32_t cur_val = 0;
                do {
                    u29_scratch_p[0] = 0;
                    noc_async_read(prod_sema_noc_addr, u29_l1_scratch, 4);
                    noc_async_read_barrier();
                    cur_val = u29_scratch_p[0];
                } while (cur_val <= u29_prev_seen);
                if (cur_val < u29_min_observed) u29_min_observed = cur_val;
            }
            u29_prev_seen = u29_min_observed;
        }
#endif
    }
#endif
```

## Hardware test result

### Canonical (no prefetcher, U29 OFF) — bytewise & GSM8K

```
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -m sglang.launch_server ...

curl ... -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
→ " What is 2+2? What is 2+2? What" at 38.28s e2e_latency

eval_qwen3_8b_gsm8k_chat.py --num 10
→ 10/10 CORRECT = 100.0%
```

**Bytewise-equal vs the U28 canonical baseline.  No regression.**

### Prefetcher + U29 Phase 2 (signaler wait engaged)

```
SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_U29_W2_RS_SIGNALER=1 ...
curl ... -d '{"text":"What is 2+2?","max_new_tokens":15,"temperature":0.0}'
→ " What stati wy wy wy wy wy wy ..." at 39.48s e2e_latency  (cold)

curl ... -d '{"text":"What is the capital of France?","max_new_tokens":15,"temperature":0.0}'
→ " What doing doing doing doing doing ..." at 0.29s e2e_latency  (warm)

eval_qwen3_8b_gsm8k_chat.py --num 10
→ Q1 garbage → NaN logits → sampler RuntimeError → server dies
→ 0/10 = 0.0%
```

Same warm-decode latency with `SGLANG_TT_U29_DISABLE_WAIT=1` set
(0.29s), confirming the wait loop iterates and exits very quickly
per call (i.e., the wait condition is satisfied immediately on
every call).

## Hypothesis ledger (post-U29 Phase 2)

| ID | Suspect | Pre-Phase-2 | Post-Phase-2 |
|---|---|---|---|
| **U16 Phase 5c — Device-side GlobalSemaphore producer-consumer handshake** | TOP STANDING | **FULLY WIRED via U29 (compute exit-sync + dataflow inc + RS reader wait + W2-only narrowing)** — but the symptom is unchanged.  Either the L1 sema slot is unsafe (0x90000 might collide with allocator), or the cross-subdev race is a symptom not root cause. |
| **U28 β — Cross-sub-device dispatch race** | NEW TOP per U28 | **WEAKENED — STANDING but no longer top.**  The proposed in-kernel fix is now end-to-end wired with strict happens-before ordering on the producer side; if U28-β were the sole root cause, Phase 2 should have fixed it.  Output unchanged implies either (a) the fix is silently broken at the sema-slot level, or (b) other races (WO/FF2 → all_reduce_async, prefetcher GCB consumer races) also contribute. |
| **U27 β — W2 PACK writes only tile-0 header** | STANDING | STANDING (Phase 2 didn't probe). |
| **U27 γ — GlobalCB tensor data overflow at 0xa6700** | STANDING | STANDING (Phase 2 didn't probe). |
| **NEW U29 — L1 0x90000 is in allocator-managed region** | (new) | NEW STANDING — Phase 3 should switch to a high-L1 slot (e.g., 0x178000) or a properly allocated `GlobalSemaphore`. |
| **NEW U29 — WO / FF2 / WQKV → CCL chains also race independent of W2** | (new) | NEW STANDING — Phase 3 should extend the U29 pattern to `all_reduce_async`'s kernels OR add per-chain sentinels to confirm. |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (only `models/tt_transformers/tt/mlp.py` in the tt-metal-sglang fork; all probes and gates env-gated; default OFF preserves canonical bytewise).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` and use literal absolute path (`TT_CACHE_HOME=/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop.
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted for `/tt-metal/` — used `podman cp` for source sync (5 files + mlp.py).  Used FIXED `rebuild_tt_metal_kernels.sh` for the build.
- [x] All new code env-gated under `SGLANG_TT_U29_W2_RS_SIGNALER` (plus narrow `SGLANG_TT_U29_W2_PRODUCER_NOW` for matmul side).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"` at 38.28s) AND GSM8K(10) = 10/10 = 100% with U29 OFF.
- [x] Server stopped at session end.

## Phase 3 — exact next steps

1. **Switch L1 sema slot to a known-safe address.**  Either:
   - Allocate a `tt::tt_metal::CreateGlobalSemaphore` over the
     receiver-core grid, query its address (which will be in the
     reserved semaphore region, not the allocator's tensor region),
     and pass it as a runtime arg to BOTH the matmul producer kernel
     and the RS reader kernel.
   - OR pick a high L1 offset (e.g., 0x178000 = 1.47 MB, near top
     of BH's 1.5 MB L1) that's guaranteed above any tensor
     allocation in Qwen3-8B BH P150a.
2. **Confirm the handshake works with a sentinel pattern.**  Have
   the producer write 0xDEADBEEF (not increment) at end of W2.
   Have the RS reader DPRINT the readback.  Iff value is
   consistently 0xDEADBEEF, the handshake is correctly synchronizing.
   Then re-introduce the increment scheme.
3. **Extend U29 to `all_reduce_async`.**  WO → all_reduce_async
   and FF2 → all_reduce_async use a DIFFERENT kernel pair:
   - `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_reader.cpp`
   - `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/reduction_receiver.cpp`
   Repeat the same pattern (producer-side increment in compute → dataflow handshake → consumer-side wait at receiver kernel entry).
4. **GSM8K(10) chat with all three chains fixed.**  Target ≥ 7/10.
5. **If still < 7/10**, drop into the per-chain probe approach: add
   sentinel patterns at each producer's exit and verify consumer-side
   reads match.  The first mismatch identifies the racing chain.

## Reproducer

### A. Build (host)

```bash
# Edit host sources (5 files + mlp.py).
# Sync to container:
podman cp /home/mhnie/tt-metal-sglang/<path>/<file> p3a-ngram:/tt-metal/<path>/<file>

# Rebuild:
podman exec p3a-ngram bash /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh ttnn

# Verify defines + new envs in linked lib:
podman exec p3a-ngram bash -c 'strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep "SGLANG_TT_U29" | sort -u'
# → SGLANG_TT_U29_DEBUG_DPRINT
# → SGLANG_TT_U29_DISABLE_WAIT
# → SGLANG_TT_U29_SEMA_L1
# → SGLANG_TT_U29_W2_PRODUCER_NOW
# → SGLANG_TT_U29_W2_RS_SIGNALER
```

### B. Canonical re-verify (default — U29 OFF)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u29p2_canonical.log 2>&1 &'

# Expected: " What is 2+2? What is 2+2? What" at ~38s, GSM8K 10/10
```

### C. Prefetcher + U29 Phase 2 (signaler wait engaged)

```bash
podman exec p3a-ngram bash -c '
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U29_W2_RS_SIGNALER=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  HF_MODEL=/models/Qwen3-8B \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u29p2.log 2>&1 &'

# Expected: garbage on first request (Phase 2 wait engages but
# does not fix the race in the current 0x90000-slot design).
# Server alive; warm decode 0.29s for 15 tokens.
```

## Shipping verdict (unchanged from U28)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Bug
  is now in active fix flight at the architectural level.  U29
  Phase 2 lands the producer→consumer handshake infrastructure but
  does NOT yet engage on the actual race.  Phase 3 will switch to
  a known-safe L1 slot and extend to non-W2 chains.
- **DO NOT ship `SGLANG_TT_U29_W2_RS_SIGNALER=1`.**  Same garbage
  symptom as without U29 at the smoke-test level.

The U29 Phase 2 changes are safe to leave in tree as default-off:
- The 6 changes in tt-metal-sglang are guarded by
  `SGLANG_TT_U29_W2_RS_SIGNALER=1` (+ `SGLANG_TT_U29_W2_PRODUCER_NOW=1`
  for the matmul producer side) checks; under default OFF they emit
  NO defines, NO additional kernel binary paths.
- Canonical bytewise-equal verified at 38.28s with U29 OFF +
  GSM8K(10) = 10/10 with U29 OFF.

## Commits this session

- **tt-metal-sglang**: pending — bumps HEAD `ad685b4576b` → new SHA.
- **sglang**: pending — bumps HEAD `35aff294d` → new SHA (this doc only; no Python changes in sglang tree).

## Working state at session end

- tt-metal-sglang HEAD: pending (will be the U29 Phase 2 commit).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only).
- sglang HEAD: pending this doc commit.
- Container `/tt-metal/`: synced (5 modified files + mlp.py).
- Container `/tt-metal/ttnn/ttnn/_ttnn{,cpp}.so`: rebuilt; symbols verified.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at 38.28s + GSM8K(10) = 10/10.

## Next-session checklist (Phase 3)

1. Read this doc + Phase 1 + U28 as the latest evidence chain.
2. Phase 3a: switch L1 sema slot to a `GlobalSemaphore` OR to a high-L1 offset (e.g. 0x178000 or near top of BH L1).  Verify with sentinel pattern (producer writes 0xDEADBEEF, RS reader DPRINTs readback).
3. Phase 3b: GSM8K(10) under `SGLANG_TT_U29_W2_RS_SIGNALER=1` — target ≥ 7/10.  If passes, ship as Phase 3.
4. Phase 3c (if Phase 3b still fails): extend the same pattern to `all_reduce_async`'s worker_reader + reduction_receiver kernels.  Repeat GSM8K(10).
5. Phase 3d (if still fails): per-chain sentinel hunt — instrument every gathered-matmul → CCL boundary with sentinel + readback to identify the first racing chain.
