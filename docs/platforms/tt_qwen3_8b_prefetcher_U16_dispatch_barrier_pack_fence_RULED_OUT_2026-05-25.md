# TT Qwen3-8B prefetcher — U16: dispatch-cached-path BARRIER + matmul-PACK kernel-exit fence — RULED OUT — 2026-05-25

Status: **BLOCKED — U16 diagnosed the U5 hang to its root cause
(receiver_sub_device runs persistent kernels per `prefetcher.py:451-456`;
`set_sub_device_stall_group([worker only])`; ANY host-side
`Synchronize(receiver_sub_device)` or routing of CCL ops to receiver
deadlocks `finish_nolock` indefinitely), and tried two new device-side
mitigations — (a) forcing the CACHED path of `insert_stall_cmds`
(`tt_metal/impl/program/dispatch.cpp:451`) to emit
`CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER | WAIT_STREAM` like the UNCACHED
path so every trace-replay program gets a global cross-sub-device
sync; and (b) injecting `PACK((ckernel::tensix_sync()))` per-batch and
an unconditional `ckernel::tensix_sync()` before kernel exit in the
gathered matmul compute kernel
(`bmm_large_block_zm_fused_bias_activation_gathered.cpp`) under a
new compile-time define `SGLANG_TT_W2_RS_BARRIER_KERNEL` propagated
from the program factory when `SGLANG_TT_W2_RS_BARRIER=1`. Neither
fix changed the trace-replay garbage output ("What Wizard excer
excer ...", "What 对学生デ ...", "What_iters_iters ..." — all U11
mode-collapse signature). The hypothesis that the bug is fixable
by a heavy-hammer cross-sub-device synchronization is now RULED
OUT. The remaining attack is U16 Phase 5c — a device-side
GlobalSemaphore producer-consumer handshake between W2 matmul's PACK
and reduce_scatter's reader (multi-session C++ scope; mirrors the
existing `all_gather` barrier_semaphore + `MatmulFusedOpSignaler`
infra).**

Continuation of
`tt_qwen3_8b_prefetcher_U15_mlp_w2_consumer_race_2026-05-24.md`
(U15 narrowed bug to MLP W2 -> reduce_scatter trace-replay race;
producer L1 zero per Probe F; consumer reads nonzero per U14).

tt-metal-sglang HEAD: **`30d6af20e3f`** (this U16 commit; all changes
env-gated, default-off; canonical bytewise-equal).
sglang HEAD: pending this doc commit.

## Phase 1 — U5 hang diagnosis (COMPLETED)

Re-launched server with `SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1`,
reproduced the hang at "Allocating device buffers is unsafe due
to the existence of an active trace" post-trace-capture, then took
a `py-spy --native` dump of the wedged scheduler process:

```
pthread_cond_wait                                                  (libc.so.6)
tt::tt_metal::distributed::FDMeshCommandQueue::finish_nolock      (libtt_metal.so)
tt::tt_metal::distributed::MeshCommandQueueBase::enqueue_write_shard_to_sub_grid (libtt_metal.so)
tt::tt_metal::distributed::MeshCommandQueueBase::enqueue_write_mesh_buffer (libtt_metal.so)
tt::tt_metal::GlobalSemaphore::reset_semaphore_value              (libtt_metal.so)
tt::tt_metal::GlobalSemaphore::setup_buffer                       (libtt_metal.so)
tt::tt_metal::CreateGlobalSemaphore                               (libtt_metal.so)
ttnn::global_semaphore::create_global_semaphore                   (_ttnncpp.so)
ttnn::operations::ccl::AllGatherDeviceOperation::AllGatherProgram::create_mesh_workload (_ttnncpp.so)
[...]
ttnn.all_gather                                                   (_ttnncpp.so)
_decode_forward_trace_text@generator.py:1713  (post-trace ttnn.all_gather)
```

The hang is `CreateGlobalSemaphore` inside the post-trace
`ttnn.all_gather(logits)` op (the gather-outside-trace strategy at
generator.py:1715). The L1-alloc path goes through
`enqueue_write_mesh_buffer -> finish_nolock`. finish_nolock waits
on the device's default sub-device stall group.

**Root cause of U5/U16-Phase-2 hang (definitive):**
`prefetcher.py:451-456` sets
`mesh_device.set_sub_device_stall_group([sub_devices_id[-1]])`
where `sub_devices_id[-1]` is the WORKER sub-device. The comment
explicitly says:

> Sender and receiver sub-devices run persistent kernels that never
> generate completion signals; including them in the stall group
> causes finish_nolock({}) — used internally by populate_mesh_buffer,
> cpu(), and blocking buffer ops — to hang indefinitely.

So `Synchronize(receiver_sub_device)` — whether issued by U5's
routing of CCL ops to receiver (which leaves receiver-stream work
pending at trace end), or by U16 Phase 2's explicit
`ttnn.synchronize_device(sub_device_ids=[receiver])` after
`execute_trace` and before the post-trace `all_gather` — ALWAYS
deadlocks. The receiver's persistent kernels never advance a stream
counter, so any wait on receiver's counter is unbounded.

This invalidates every U5 / U16-Phase-2 / U1-Attack-3 fix that
relies on host-side sync of receiver, AND every fix that routes
operations to receiver_sub_device_id.

## Phase 2 — Python-level post-trace receiver drain — RULED OUT

Hypothesis: insert
`ttnn.synchronize_device(self.model_args[i].mesh_device,
sub_device_ids=[prefetcher.receiver_sub_device_id])` after
`execute_trace` in generator.py and before the post-trace
`ttnn.all_gather` to drain receiver-stream work before the
all_gather's `CreateGlobalSemaphore` runs.

Result: HANG, same `pthread_cond_wait -> finish_nolock` stack,
this time at the `Synchronize` call directly (py-spy line
`generator.py:1737`, my newly-inserted synchronize_device line).
Confirmed via py-spy --native:

```
finish_nolock
Synchronize
ttnn::device::device_module(...)::$_18
generator.py:1737
```

This is the SAME root cause as Phase 1. Removed Phase 2 from
generator.py and added an explanatory comment block instead.

## Phase 5a — dispatch.cpp cached-path BARRIER override — RULED OUT

Per U1: the UNCACHED (first-compile) dispatch path emits
`CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER | WAIT_STREAM` (BARRIER is a
GLOBAL sync across ALL sub-devices). The CACHED (trace-replay)
path emits only `WAIT_STREAM` on the current op's OWN sub-device,
leaving a cross-sub-device race where a worker-routed RS can start
BEFORE a receiver-routed matmul has finished.

Fix: under `SGLANG_TT_W2_RS_BARRIER=1`, force the CACHED path
(`tt_metal/impl/program/dispatch.cpp:451`) to ALSO emit
`BARRIER | WAIT_STREAM`. One-time `log_info(LogMetal, ...)` marker
confirms env-read works (`'1' -> u16_force_barrier_cached=true`
when set; `'(unset)' -> false` for canonical).

Hardware result: garbage output unchanged ("What_iters_iters ..."),
e2e_latency 39 s. The cached-path BARRIER is being emitted (verified
via log marker), but the trace-replay race persists. **Conclusion:
the bug is NOT a missing cross-sub-device dispatch sync. Even with
a global BARRIER between every program in the trace replay, output
remains the U11 garbage.**

This rules out a substantial portion of the U1 hypothesis space:
forcing global sync at dispatch level does not close the race.

## Phase 5b — matmul-PACK kernel-exit `tensix_sync()` fence — RULED OUT

Hypothesis: the dispatch-level BARRIER waits for kernel-exit (workers
done), but PACK writes to L1 can be in-flight when the kernel exits.
The dispatcher's "matmul done" stream-counter increment fires before
PACK's L1 writes are coherent. Adding `ckernel::tensix_sync()`
inside the matmul kernel guarantees all tensix-thread ops (including
PACK->L1 writes) retire before the kernel function returns.

Implementation: env-gated propagation of the compile-time define
`SGLANG_TT_W2_RS_BARRIER_KERNEL` from
`matmul_multicore_reuse_mcast_1d_program_factory.cpp` to the
gathered compute kernel
(`bmm_large_block_zm_fused_bias_activation_gathered.cpp`). When
defined, the kernel emits:

```cpp
// Per-batch (inside `for (uint32_t b = 0; b < batch; ++b)` loop):
PACK((ckernel::tensix_sync()));

// Final kernel-exit fence:
ckernel::tensix_sync();
```

Hardware result: garbage output unchanged. **Conclusion: a
PACK-thread tensix fence at kernel exit does not close the race
either. Either the race is between cycles much earlier than
kernel exit, or PACK->L1 coherence is not the actual issue.**

## What this means for the bug

We now have RULED OUT:

1. Producer-side fixes:
   - PACK kernel writes to mm_out_cb (U13 - PACK writes zero)
   - matmul kernel internal state (U14 - end-of-kernel L1 = zero)
   - **NEW: matmul kernel-exit tensix_sync fence (U16 Phase 5b)**
2. Consumer-side fixes:
   - reduce_scatter reader shard mapping (U15 Probe E - identical
     shard_grid as producer; same buffer_address 0xa6700)
3. L1 allocator fixes:
   - L1 allocator overlap (U15 Probe G - 0xa6700 has unique owner)
4. Dispatch ordering fixes:
   - subdev routing of CCL to receiver (U5 - persistent-kernel
     deadlock)
   - subdev routing of CCL + post-trace receiver drain (U16
     Phase 2 - same deadlock)
   - **NEW: dispatch cached-path global BARRIER override (U16
     Phase 5a)**

What's LEFT STANDING:

- U16 Phase 5c — **device-side GlobalSemaphore producer-consumer
  handshake between W2 matmul's PACK and reduce_scatter's reader**.
  Mirrors the existing `all_gather` barrier_semaphore pattern
  (`all_gather_program_factory.cpp:36-44`) and the
  `MatmulFusedOpSignaler` infra (`ccl_op_fusion.hpp:131-236`) used
  by `llama_reduce_scatter`. The matmul writer kernel (running on
  receiver cores or worker cores depending on path) does
  `noc_semaphore_inc(sema, 1)` after its final PACK + writes-flushed;
  the reduce_scatter reader kernel does
  `noc_semaphore_wait_min(sema, expected_matmul_count)` before its
  first `noc_async_read(noc_addr, ...)`. The semaphore lives on
  worker cores so worker stream advances correctly when matmul
  finishes its PACK, regardless of where matmul itself dispatched.

Estimated scope (per U5 doc's own scoping in
`tt_qwen3_8b_prefetcher_U5_subdev_routing_RULED_OUT_2026-05-24.md`):
2-3 sessions of C++ work (matmul writer kernel + reduce_scatter
reader kernel + factory plumbing + device-op-attrs hash for cache
key + Python wiring to create the semaphore and pass it through
both ops' runtime args).

## What U16 lands

All env-gated; canonical (no `SGLANG_TT_W2_RS_BARRIER`,
`SGLANG_TT_PREFETCHER_OUTPUT_BARRIER`, or `SGLANG_TT_USE_PREFETCHER`)
is bytewise-equal to U15:

| File | Change |
|---|---|
| `tt_metal/impl/program/dispatch.cpp` | env-gated cached-path BARRIER override + one-time log_info marker. |
| `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | env-gated propagation of `SGLANG_TT_W2_RS_BARRIER_KERNEL` define to the gathered matmul kernel. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | per-batch + kernel-exit `ckernel::tensix_sync()` under `SGLANG_TT_W2_RS_BARRIER_KERNEL`. |
| `models/tt_transformers/tt/mlp.py` | `_u5_pref_subdev()` explicit note: `SGLANG_TT_W2_RS_BARRIER` does NOT route here (deadlock). |
| `models/tt_transformers/tt/attention.py` | same note. |
| `models/tt_transformers/tt/generator.py` | post-trace receiver-drain code removed; comment block explaining why. |

## Hypothesis ledger (post-U16)

| ID | Suspect | Pre-U16 | Post-U16 |
|---|---|---|---|
| S1-S10, Lead 2-3, U3, U4-A, U4-B | various | RULED OUT | unchanged |
| **U1** | Cross-sub-device WAIT_STREAM gap | LOCATED, 3 fix paths | **U16 Phase 5a (dispatch cached-path global BARRIER) RULED OUT**. The race is NOT fixed by forcing global sync between cached programs. |
| **U5** | Subdev-routing of CCL to receiver | RULED OUT (hangs) | **CONFIRMED-via-py-spy**: receiver runs persistent kernels (prefetcher.py:451-456); ANY routing or Synchronize(receiver) deadlocks `finish_nolock` indefinitely. |
| **U16 Phase 2** | Python `synchronize_device(receiver)` post-trace | (new) | **RULED OUT — same persistent-kernel deadlock as U5.** |
| **U16 Phase 5a** | dispatch.cpp cached-path BARRIER | (new) | **RULED OUT — global cross-sub-device sync between every program in trace replay does not fix the garbage.** |
| **U16 Phase 5b** | matmul-PACK kernel-exit `tensix_sync()` fence | (new) | **RULED OUT — adding per-batch + final tensix_sync inside the matmul compute kernel does not fix the garbage.** |
| **U16 Phase 5c** | Device-side `GlobalSemaphore` producer-consumer handshake (matmul writer increment + RS reader wait) | (new) | **STANDING — TOP PRIORITY, multi-session C++ scope.** |

## Reproducer (working, both directions)

```bash
# U16-positive (W2_RS_BARRIER=1, full U16): garbage output
podman exec p3a-ngram bash -c '\
  PYTHONUNBUFFERED=1 \
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_W2_RS_BARRIER=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u16_phase5c.log 2>&1 &'
# Wait for "fired up", then:
curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text":"What is 2+2?", "sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Output: garbage U11 signature ("What_iters_iters_iters ...")

# Canonical (no prefetcher, no barrier): correct
podman exec p3a-ngram bash -c '\
  PYTHONUNBUFFERED=1 \
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u16_canonical.log 2>&1 &'
# Output: "4. But wait, in some contexts, like" (correct math reasoning).
# e2e_latency=11.7s.
# Log marker: [U16-DISPATCH] cached-path BARRIER override: SGLANG_TT_W2_RS_BARRIER='(unset)' -> u16_force_barrier_cached=false
```

## U17 dispatch recommendation

Drop the cross-sub-device dispatch hypothesis tree entirely. The
remaining viable attack is U16 Phase 5c — device-side
GlobalSemaphore PACK->reader handshake. Implementation outline:

1. **Create a worker-side GlobalSemaphore in Python (mlp.py)**, sized
   for one count per matmul-producer-core per CCL-consumer-core.
   Pass it as a runtime arg to both `ttnn.linear(W2)` and
   `ttnn.experimental.reduce_scatter_minimal_async` via
   `subdevice_id=` already plumbed.
2. **Matmul writer kernel** (running on receiver cores per
   `sub_device_id=receiver` at mlp.py:563): after the FINAL PACK +
   `noc_async_writes_flushed()`, emit
   `noc_semaphore_inc(sema_addr_on_worker_cores, 1)`.
3. **Reduce_scatter reader kernel**
   (`line_reduce_scatter_minimal_async_reader.cpp`): before the
   first `noc_async_read(...)`, emit
   `noc_semaphore_wait_min(sema_local_addr, expected_matmul_count)`.
4. **Factory plumbing**: thread the optional semaphore + expected
   count through `MatmulProgramConfig`'s runtime args + the existing
   `barrier_semaphore` plumbing in
   `reduce_scatter_minimal_async_program.cpp`. Add to device-op-attrs
   hash for program-cache key.

Pattern reference: `all_gather_program_factory.cpp:36-44`
(`barrier_semaphore` init across both producer + consumer); the
existing MatmulFusedOpSignaler in `ccl_op_fusion.hpp:131-236`.

## Working state at session end

- tt-metal-sglang HEAD: **`30d6af20e3f`** (this U16 commit;
  env-gated diagnostic + partial fix; default-off behavior-preserving)
- sglang HEAD: pending this doc commit
- Working trees: clean (tt_metal/third_party/tt_llk/ untracked
  pre-existing)
- `stash@{0,1,2}`: untouched
- Container source synced to HEAD (Python + C++)
- tt-metal libs rebuilt via `rebuild_tt_metal_kernels.sh` (both
  `tt_metal` and `ttnn` targets; also `cp`'d to
  `/tt-metal/tt_metal/libtt_metal.so` and
  `/tt-metal/build_Release/lib/libtt_metal.so` since the build
  output landed only at `/tt-metal/build_Release/tt_metal/libtt_metal.so`)
- `/root/.cache/tt-metal-cache/*` cleared multiple times during
  iteration; current state: probably populated with W2_RS_BARRIER=1
  pre-compiled firmware. Clearing once more is harmless for
  canonical (just adds ~30 s to first model init).
- TT cards: 2x p150a (P300 cluster); reset via `tt-smi -r 0,1`
  3 times during this session due to U5/U16-Phase-2 hangs
- Server: stopped (killed cleanly post-canonical-sanity-check)

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (sglang doc only; tt-metal
      side fully env-gated)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape pattern
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] No regression — canonical preserved (math reasoning intact:
      "4. But wait, in some contexts, like" at 11.7s e2e)
- [x] Probe tensix_sync() applied before L1 reads (kernel-side
      probes inherited from U14)
- [x] All new fixes env-gated (`SGLANG_TT_W2_RS_BARRIER`)

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat >=9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** Trace-
  replay race persists despite U5, U16 Phase 2, U16 Phase 5a, and
  U16 Phase 5b.
- **DO NOT ship `SGLANG_TT_W2_RS_BARRIER=1`.** Does not fix the
  bug; adds cross-sub-device BARRIER overhead per cached program
  for no functional gain.
- **DO NOT ship `SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1`.** Deadlocks
  on receiver-subdev persistent-kernel routing (U5 + U16 Phase 1
  re-confirmation).

## Commits this session

- (tt-metal-sglang) `30d6af20e3f` —
  `prefetcher: U16 dispatch BARRIER + PACK fence — partial fix (BLOCKED)`
- (sglang) `<this doc>` — pending commit
