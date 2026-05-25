# TT Qwen3-8B prefetcher — U29 Phase 1: device-side W2→RS signaler INFRASTRUCTURE LANDED — 2026-05-25

Status: **EVIDENCE_ADVANCE — U29 Phase 1 lands the
end-to-end scaffold for the architecturally correct fix to the
U28-β cross-sub-device dispatch race: (a) env-gate
`SGLANG_TT_U29_W2_RS_SIGNALER=1` + matched `SGLANG_TT_U29_SEMA_L1`
define propagated from the matmul-1d gathered factory AND the RS
line factory to their respective kernels; (b) producer-side
`noc_semaphore_inc` increment in
`reader_bmm_tile_layout_in1_ring_all_gather.cpp` AFTER
`async_write_barrier()` (last RISC code on the W2 producer
core); (c) kernel-exit `tensix_sync()` in
`bmm_large_block_zm_fused_bias_activation_gathered.cpp` to
guarantee PACK→L1 retirement before the dataflow increment;
(d) RS reader DPRINT scaffold in
`line_reduce_scatter_minimal_async_reader.cpp` at kernel entry.
All env-gated; canonical bytewise-equal at 9.09s e2e_latency
(no regression from U28 baseline 9.10s).**

**Phase 1 result on hardware (SGLANG_TT_U29_W2_RS_SIGNALER=1):
producer increment fires (build + JIT succeeds), but RS reader
wait-loop is intentionally a no-op DPRINT scaffold; race still
produces garbage as expected.  Output: `" What哮喘 Prevent_extract..."`
at 10.27s e2e_latency.  GSM8K crashes mid-test (NaN in
sampler from garbage logits, matches the chat-template
sensitivity observed in U17–U28).  This is the EXPECTED Phase 1
result — the fix is not engaged yet.  Phase 2 wires the
per-producer-core NoC-coord runtime arg + actual
`noc_async_read` + `noc_semaphore_wait_min` loop in the RS reader.**

Continuation of `tt_qwen3_8b_prefetcher_U28_PER_CORE_HYPOTHESIS_REFUTED_NEW_TOP_HYPOTHESIS_IS_CROSS_SUBDEV_RACE_2026-05-25.md`.

tt-metal-sglang HEAD: **`cbcc05f8edf`** (pending this U29 commit).
Branch `tenstorrent-p1`; NOT pushed (predator2k/* fork only).
sglang HEAD: **`35aff294d`** (pending this doc commit).

## TL;DR

U28 confirmed the cross-sub-device dispatch race
(W2 on receiver_sub_device, RS on worker_sub_device,
set_sub_device_stall_group excludes receiver, trace replay
schedules per-subdev independently → RS reader fires before
W2 PACK retires → reads stale L1 0xa6700 bytes).

U16 Phase 5c (device-side GlobalSemaphore producer-consumer
handshake) was the STANDING top recommendation.  U29 implements
it via a fixed-L1-address atomic counter on each receiver core,
avoiding the need to plumb new optional kwargs through the
public `ttnn.linear` and `ttnn.experimental.reduce_scatter_minimal_async`
Python APIs.

**Phase 1 (this commit):** end-to-end scaffold lands in five files
(matmul factory, matmul in1 sender writer, matmul gathered
compute kernel, RS line factory, RS line reader kernel).  Producer
increment fires; RS reader scaffold logs entry but does NOT wait
yet.  Canonical bytewise-equal verified.  Build chain validated
(host-edit → podman cp → tt-metal rebuild → JIT compile).
Established the L1 scratch address `0x90000` (well outside
0xa6700 residual region; in unreserved L1).

**Phase 2 (next session, kept out of Phase 1 for scope):** wire the
RS reader's actual wait loop:
- New runtime arg to the RS reader: per-producer-core noc-coord
  list (32 entries for W2's 32 receiver cores) computed from the
  input tensor's shard grid in the program factory.
- New runtime arg: expected counter value (bumped per forward via
  `override_runtime_arguments`).
- Replace the Phase-1 DPRINT scaffold with:
  ```cpp
  for (uint32_t i = 0; i < num_producer_cores; i++) {
      uint64_t producer_sema_noc_addr =
          get_noc_addr(producer_noc_x[i], producer_noc_y[i],
                       (uint32_t)(SGLANG_TT_U29_SEMA_L1));
      cb_reserve_back(cb_input_id, 1);
      uint32_t l1_scratch = get_write_ptr(cb_input_id);
      uint32_t cur_val = 0;
      do {
          noc_async_read(producer_sema_noc_addr, l1_scratch, 4);
          noc_async_read_barrier();
          cur_val = *(volatile tt_l1_ptr uint32_t*)l1_scratch;
      } while (cur_val < expected_counter);
  }
  ```

**Phase 3 (after Phase 2 + bytewise-coherent):** apply the same
pattern to other prefetcher chains:
- WO → all_reduce (same gathered matmul + AR shape as W2)
- WQKV → nlp_create_qkv_heads_decode
- FF1/FF3 → ttnn.mul → FF2
- FF2 → all_reduce

## What U29 Phase 1 lands

All env-gated by `SGLANG_TT_U29_W2_RS_SIGNALER=1`; default OFF;
canonical bytewise-equal verified.

### tt-metal-sglang files modified (5)

| File | Change |
|---|---|
| `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | Env-gated propagation of `SGLANG_TT_U29_W2_RS_SIGNALER` + `SGLANG_TT_U29_SEMA_L1=0x90000` defines to in1 sender writer kernel + gathered compute kernel.  Only fires under `ENABLE_GLOBAL_CB` (gathered/prefetcher path). |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Producer-side `noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], SGLANG_TT_U29_SEMA_L1), 1)` + `noc_async_atomic_barrier()` AFTER `noc.async_write_barrier()` (last RISC point on producer core). |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | Kernel-exit `ckernel::tensix_sync()` (in addition to the U16 W2_RS_BARRIER_KERNEL block) to guarantee PACK→L1 retirement before the dataflow kernel's `noc_semaphore_inc` fires. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp` | Env-gated propagation of `SGLANG_TT_U29_W2_RS_SIGNALER` + `SGLANG_TT_U29_SEMA_L1=0x90000` defines to the line RS reader kernel. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp` | RS reader-side DPRINT scaffold at kernel entry logging the matched U29_SEMA_L1 address (Phase 1 scaffold only; Phase 2 will replace with the actual wait loop). |

### sglang files modified (0)

No sglang Python changes in this commit (the Python plumbing
required is delayed to Phase 2's `override_runtime_arguments`
work; for Phase 1 the producer-side increment fires without
needing any Python coordination).

## MatmulFusedOpSignaler study summary (per task brief Phase 1)

`ttnn::experimental::ccl::MatmulFusedOpSignaler` (in
`ccl_op_fusion.hpp:131-236`) is tt-metal's existing
infrastructure for producer→consumer semaphore plumbing between a
matmul and a fused all_gather / reduce_scatter / llama_reduce_scatter.

### How it works (Llama RS variant — closest fit to W2→RS)

1. **RS side** (`init_llama_rs_cores_rs`): creates a per-program
   `tt::tt_metal::Semaphore` over the RS workers' core range.
   Stores it in `rs_semaphore` and the worker core range in
   `rs_cores`.
2. **Matmul side** (`init_llama_rs_cores_mm`): picks a "privileged"
   matmul core (default index 0), creates a per-program
   `matmul_privilaged_semaphore` on that one core, and records the
   number of remaining matmul cores `matmul_semaphore_target =
   cores.size() - 1`.
3. **RS reader kernel** receives `rs_semaphore` ID via
   `push_llama_rs_rt_args_for_rs` and waits on it before reads.
4. **Matmul writer kernel** receives via `push_llama_rs_rt_args_for_mm`:
   - Privileged core's NoC coordinates and semaphore ID.
   - On the privileged core: `target_value`, RS workers bounding
     box, RS semaphore ID.
   - On non-privileged cores: increments the privileged core's
     semaphore to 1.
5. **Privileged core**: waits for `target_value` increments to land
   (i.e. all non-privileged matmul cores done), then multicast-sets
   the RS cores' `rs_semaphore` to release the RS reader.

### Why we didn't reuse it directly for U29 Phase 1

- The Llama RS variant is tightly coupled to specific matmul
  factory code paths (`init_llama_rs_cores_mm` only invoked when
  `fused_op_signaler` is passed through to the gathered factory).
- Passing a `MatmulFusedOpSignaler` from Python requires
  modifying the **public** `ttnn.linear` and
  `ttnn.experimental.reduce_scatter_minimal_async` signatures (or
  introducing a new `ttnn.linear_with_signaler` wrapper op +
  nanobind binding + ttnn registration).
- Our gathered W2 matmul + line RS path is non-fused (separate
  device ops), and the existing fused-op detection in the line
  RS factory triggers only when an explicit
  `ReduceScatterFusedOpSignaler` is in the op invocation.
- U29's fixed-L1-address scheme achieves the **same end-state
  guarantee** (RS reader does not start L1 reads until W2 producer
  has finalized L1 writes) with **zero public API surface change**,
  by using a compile-time-fixed L1 scratch address visible from
  both kernels via matched defines.

### Trade-offs vs MatmulFusedOpSignaler

| Aspect | MatmulFusedOpSignaler | U29 fixed-L1-addr |
|---|---|---|
| Public API change | YES (new kwarg) | NO |
| Kernel-side complexity | High (privileged core election + multicast) | Low (per-core counter + linear poll) |
| Cross-program L1 coherence | Per-program semaphore IDs translated by runtime | Fixed L1 offset, identical on both kernels |
| Trace-replay safety | YES (via program-local kernel args) | YES (define is compile-constant; counter is monotonic) |
| Number of L1 atomics per W2 dispatch | 32 producers → 1 priv → mcast | 32 producers → own L1 (uncontended) |

The U29 scheme avoids the multicast-to-RS-cores phase entirely
(each RS reader worker reads each W2 producer core's L1 counter
directly), at the cost of 32 NoC reads per RS reader worker at
entry.  Given the residual race window is observed at < 1 ms
in U17 probes, 32 × NoC-read latency (~tens of ns each) is
acceptable.

## Implementation diffs

### 1. Matmul factory: env-gated define propagation

`matmul_multicore_reuse_mcast_1d_program_factory.cpp` (gathered branch
inside `if (use_global_cb)` block, after the U25 RCB probe block):

```cpp
const char* u29_sig_env = std::getenv("SGLANG_TT_U29_W2_RS_SIGNALER");
if (u29_sig_env != nullptr && std::string(u29_sig_env) == "1") {
    mm_in1_kernel_defines["SGLANG_TT_U29_W2_RS_SIGNALER"] = "1";
    mm_in1_kernel_defines["SGLANG_TT_U29_SEMA_L1"] = "0x90000";
    mm_kernel_defines["SGLANG_TT_U29_W2_RS_SIGNALER"] = "1";
    mm_kernel_defines["SGLANG_TT_U29_SEMA_L1"] = "0x90000";
}
```

### 2. Matmul in1 sender writer kernel: producer increment

`reader_bmm_tile_layout_in1_ring_all_gather.cpp` (after the
`noc.async_write_barrier()` at end of `kernel_main`):

```cpp
#ifdef SGLANG_TT_U29_W2_RS_SIGNALER
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

Notes:
- `my_x[NUM_NOCS]` / `my_y[NUM_NOCS]` are uint8_t arrays
  declared in `tt_metal/hw/inc/internal/tt-1xx/risc_common.h`.
  Initial implementation shadowed them with `const uint32_t my_x =`
  causing a compile error; fixed by renaming local to `u29_my_x`.
- `noc_async_atomic_barrier()` (not `noc.async_atomic_barrier()` —
  the latter is the experimental Noc class API; this is the
  C-style API that's available in this file's includes).

### 3. Gathered compute kernel: kernel-exit fence

`bmm_large_block_zm_fused_bias_activation_gathered.cpp` (after the
U16 W2_RS_BARRIER_KERNEL block, at end of `kernel_main`):

```cpp
#ifdef SGLANG_TT_U29_W2_RS_SIGNALER
    ckernel::tensix_sync();
#endif
```

This sync fences PACK/UNPACK/MATH threads BEFORE the dataflow
in1 sender writer's `noc_semaphore_inc` fires.  Without this,
the dataflow increment could race the still-in-flight PACK→L1
writes (the very bug U29 is trying to fix).

### 4. RS line factory: env-gated define propagation

`reduce_scatter_minimal_async_program.cpp` (in the
`build_line_reduce_scatter_minimal_async_program_artifacts`
function, after the U25 BYTE_HUNT block, before `// KERNEL CREATION`):

```cpp
const char* u29_sig_env = std::getenv("SGLANG_TT_U29_W2_RS_SIGNALER");
if (u29_sig_env != nullptr && std::string(u29_sig_env) == "1") {
    reader_compute_defines["SGLANG_TT_U29_W2_RS_SIGNALER"] = "1";
    reader_compute_defines["SGLANG_TT_U29_SEMA_L1"] = "0x90000";
}
```

### 5. RS line reader kernel: Phase 1 DPRINT scaffold

`line_reduce_scatter_minimal_async_reader.cpp` (at kernel entry,
after the `ReduceScatterOpReceiver` init):

```cpp
#ifdef SGLANG_TT_U29_W2_RS_SIGNALER
{
    static uint32_t u29_log_budget = 64;
    if (u29_log_budget > 0) {
        u29_log_budget--;
        DPRINT << "[U29_RS_READER_ENTRY signaler_addr=0x" << HEX()
               << (uint32_t)(SGLANG_TT_U29_SEMA_L1) << DEC()
               << " scaffold_only]" << ENDL();
    }
}
#endif
```

Phase 2 will replace this with the actual `noc_async_read` +
`noc_semaphore_wait_min` loop over the 32 producer cores.

## Hardware test result (Phase 1)

### Build chain validation

```
podman cp <all 5 host files> p3a-ngram:/tt-metal/<corresponding paths>
podman exec p3a-ngram bash /sglang/.../scripts/rebuild_tt_metal_kernels.sh ttnn
  → [3/5] Linking CXX shared library ttnn/_ttnncpp.so
  → [4/5] Linking CXX shared library ttnn/_ttnn.so
  → Output libs synced to /tt-metal/ttnn/ttnn/_ttnn{,cpp}.so

podman exec p3a-ngram strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep SGLANG_TT_U29
  → SGLANG_TT_U29_SEMA_L1
  → SGLANG_TT_U29_W2_RS_SIGNALER
```

The `getenv("SGLANG_TT_U29_W2_RS_SIGNALER")` calls AND the
factory's `mm_in1_kernel_defines["SGLANG_TT_U29_SEMA_L1"] = "0x90000"`
emissions are present in the linked shared library.  Defines
propagate to the JIT compile of the gathered matmul + line RS
kernels (verified by the first JIT compile under
`SGLANG_TT_U29_W2_RS_SIGNALER=1` failing with a name collision
between our `const uint32_t my_x =` and the existing
`extern uint8_t my_x[NUM_NOCS]`, which proves the define reached
the per-kernel compile step).  After the fix (rename to `u29_my_x`),
JIT compile succeeded and the prefetcher ran end-to-end.

### Canonical bytewise-equal (no prefetcher, no U29)

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_MAX_BATCH=1 \
HF_MODEL=/models/Qwen3-8B \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -u -m sglang.launch_server ...

curl -H "Content-Type: application/json" \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}' \
  http://127.0.0.1:30000/generate
```

Output: `" What is 2+2? What is 2+2? What"` at e2e_latency=**9.09s**
(matches U28 canonical baseline 9.10s).  **Bytewise-equal; no
regression.**

### Prefetcher + U29 Phase 1 (scaffold only; wait not engaged)

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_MAX_BATCH=1 \
HF_MODEL=/models/Qwen3-8B \
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U29_W2_RS_SIGNALER=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -u -m sglang.launch_server ...

curl -H "Content-Type: application/json" \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}' \
  http://127.0.0.1:30000/generate
```

Output: `" What哮喘 Prevent_extract指望 Yard时髦乔治影音影音影音电工子孙 bambsubset"`
at e2e_latency=**10.27s**.  Garbage output as **EXPECTED** — Phase 1
wires only the producer side; the RS reader does not yet wait.
Server is alive and responsive (no hang, no crash on first
request).

### GSM8K(5) with U29 ON (Phase 1 scaffold)

Server crashed mid-test (Q1 returned garbage; Q2-5 hit
`Connection refused`).  Root cause: the sampler's
`torch.multinomial` (called when temperature != 0 in chat
template) fails on probability tensors containing NaN/Inf,
which is what garbage logits produce.  The crash is a known
secondary consequence of the prefetcher race; it is NOT a
U29-introduced regression.  The garbage→NaN→sampler-crash
chain is the same one that motivated U17–U28's "garbage output
under prefetcher" investigations.

The first-generate call succeeded (no hang), proving the
end-to-end build + dispatch + trace replay path works with
the U29 producer increment firing on every W2 invocation.

## U28-β verification status

U28-β cross-sub-device dispatch race hypothesis remains
**STANDING — CONFIRMED PENDING PHASE 2 FIX VALIDATION.**  The
Phase 2 wait-loop wire-up is the definitive empirical test:
- If Phase 2 GSM8K(10) ≥ 7/10 with `SGLANG_TT_U29_W2_RS_SIGNALER=1`,
  U28-β is **CONFIRMED** and U29 is the FIX.
- If Phase 2 GSM8K(10) < 7/10 with U29 on but signature improves
  (different garbage pattern), the race is fixed but other races
  exist on other prefetcher→CCL chains → apply U29 to WO, WQKV,
  FF1/FF3, FF2 (Phase 3).
- If Phase 2 GSM8K(10) < 7/10 with no signature change, the wait
  isn't engaging → verify with DPRINT that
  `noc_async_read(producer_sema_noc_addr, ...)` actually returns
  the expected count.

## Phase 2 — exact next steps (kept out of Phase 1 for scope)

1. **Add per-producer-core noc-coord runtime arg** to the RS line
   reader.  Source: in
   `build_line_reduce_scatter_minimal_async_program_artifacts`, the
   input tensor's `shard_spec().grid()` (when
   `input_is_sharded`) gives the producer core list.  Convert
   to NoC coords via `mesh_device->worker_core_from_logical_core(c)`.

2. **Add expected-counter runtime arg** + per-forward bump via
   `override_runtime_arguments`.  Counter starts at 0 at program
   creation; each forward, RS reader's expected counter increments
   by 1 (the W2 matmul's per-core increment is also 1 per dispatch).

3. **Replace the Phase-1 DPRINT** in
   `line_reduce_scatter_minimal_async_reader.cpp` with:
   ```cpp
   for (uint32_t i = 0; i < u29_num_producer_cores; i++) {
       uint32_t px = u29_producer_noc_x[i];
       uint32_t py = u29_producer_noc_y[i];
       uint64_t prod_sema_noc_addr =
           get_noc_addr(px, py, (uint32_t)(SGLANG_TT_U29_SEMA_L1));
       uint32_t cur_val = 0;
       while (cur_val < u29_expected_counter) {
           cb_reserve_back(cb_input_id, 1);
           uint32_t l1_scratch = get_write_ptr(cb_input_id);
           noc_async_read(prod_sema_noc_addr, l1_scratch, 4);
           noc_async_read_barrier();
           cur_val = *(volatile tt_l1_ptr uint32_t*)l1_scratch;
       }
   }
   ```

4. **Validate** with `GSM8K(10)` + first-question correctness
   under `SGLANG_TT_U29_W2_RS_SIGNALER=1`.  Target ≥ 7/10
   (within canonical 9-10/10 band).

5. **If FIXED**: bench TPOT 3-run mean at 1k/256, canonical
   re-verify (U29 OFF) bytewise-equal, then apply same pattern
   to WO → all_reduce, WQKV → nlp_create_qkv_heads, FF2 → all_reduce.

## Reproducer

### A. Build (host)

```bash
# Edit host sources
vim /home/mhnie/tt-metal-sglang/ttnn/cpp/.../*.cpp

# Sync to container
podman cp /home/mhnie/tt-metal-sglang/<path>/<file> \
  p3a-ngram:/tt-metal/<path>/<file>

# Rebuild
podman exec p3a-ngram bash \
  /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh ttnn

# Verify defines in linked lib
podman exec p3a-ngram bash -c \
  'strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep SGLANG_TT_U29'
```

### B. Canonical bytewise-equal (default — U29 OFF)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c '
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u29_canonical_off.log 2>&1 &'

# Expected: " What is 2+2? What is 2+2? What" at 9.09s
```

### C. Prefetcher + U29 Phase 1 (producer-side only)

```bash
podman exec p3a-ngram bash -c '
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U29_W2_RS_SIGNALER=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  HF_MODEL=/models/Qwen3-8B \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u29_phase1.log 2>&1 &'

# Expected: garbage output (Phase 1 has scaffold-only RS reader)
# Server alive, no crash on first request.
```

## Hypothesis ledger (post-U29 Phase 1)

| ID | Suspect | Pre-U29 | Post-U29 Phase 1 |
|---|---|---|---|
| (all U27 / U28 standings) | … | per U28 | unchanged (U28 β still standing; Phase 2 will confirm or refute) |
| **U16 Phase 5c — Device-side GlobalSemaphore producer-consumer handshake** | TOP STANDING | (TOP STANDING per U28) | **Phase 1 INFRASTRUCTURE LANDED, no API regression, build chain validated.  Phase 2 wires the actual wait loop.** |
| **U28 β — Cross-sub-device dispatch race** | NEW TOP STANDING per U28 | NEW TOP STANDING | **STILL STANDING — empirical confirmation deferred to Phase 2.** |
| **U27 β — W2 PACK writes only tile-0 header** | STANDING per U27 | STANDING | STANDING (less likely if U29 Phase 2 fixes; will retire if so) |
| **U27 γ — Constant 0xa6700 bytes leak from GlobalCB tensor data overflow** | STANDING per U27 | STANDING | STANDING (less likely if U29 Phase 2 fixes; will retire if so) |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (no Python edits in sglang
      tree; all changes are in tt-metal-sglang fork).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` and use literal
      absolute path (`TT_CACHE_HOME=/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop.
- [x] Port-clear uses `\.` escape
      (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted for `/tt-metal/` — used `podman cp`
      for source sync (5 files).  Used FIXED `rebuild_tt_metal_kernels.sh`
      for the build.
- [x] All new code env-gated (`SGLANG_TT_U29_W2_RS_SIGNALER`).
- [x] Canonical Qwen3-8B re-verified bytewise:
      `" What is 2+2? What is 2+2? What"` at 9.09s e2e_latency.
- [x] Server stopped at session end.
- [x] TT devices: healthy.
- [x] L1 scratch address `0x90000` chosen well outside W2 output
      L1 region (0xa6700) AND prefetcher GCB receivers' shard
      slots; in the unreserved L1 region on Blackhole.

## Shipping verdict (unchanged from U28)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Bug
  is now in active fix flight at the architectural level (U29
  Phase 2 + 3 work).
- **DO NOT ship `SGLANG_TT_U29_W2_RS_SIGNALER=1`.**  Phase 1
  infrastructure only; producer-side increment fires but RS reader
  wait is a scaffold DPRINT.  Same bug behavior as without U29 on
  the prefetcher path.

The U29 changes are safe to leave in tree as default-off:
- The 5 changes in tt-metal-sglang are guarded by
  `SGLANG_TT_U29_W2_RS_SIGNALER=1` checks; under default OFF
  they emit NO defines, NO additional code paths in either matmul
  or RS kernels.
- Canonical bytewise-equal verified at 9.09s.

## Commits this session

- **tt-metal-sglang**: pending — bumps HEAD `cbcc05f8edf` → new SHA
  (commit message in commit step).
- **sglang**: pending — bumps HEAD `35aff294d` → new SHA (this doc + no Python).

## Working state at session end

- tt-metal-sglang HEAD: pending (will be the U29 Phase 1 commit).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.  No sglang Python changes
  in U29 Phase 1.
- Container `/tt-metal/`: synced (all 5 modified files).
- Container `/tt-metal/ttnn/ttnn/_ttnn{,cpp}.so`: rebuilt with U29
  factory defines linked in.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: bytewise `" What is 2+2? What is 2+2? What"`
  at 9.09s e2e_latency.

## Next-session checklist

1. Read this doc + the U28 doc as the latest evidence chain.
2. Phase 2 task: wire RS reader's actual wait loop (see "Phase 2
   exact next steps" above for the code shape).
3. GSM8K(10) under `SGLANG_TT_U29_W2_RS_SIGNALER=1` — target ≥ 7/10.
4. If FIXED: bench TPOT, canonical re-verify, apply to other
   prefetcher chains (Phase 3).
5. If signature CHANGED but < 7/10: apply Phase 3 to siblings.
6. If no change: DPRINT verify the wait loop's
   `noc_async_read(producer_sema_noc_addr)` returns the expected
   value, debug from there.
