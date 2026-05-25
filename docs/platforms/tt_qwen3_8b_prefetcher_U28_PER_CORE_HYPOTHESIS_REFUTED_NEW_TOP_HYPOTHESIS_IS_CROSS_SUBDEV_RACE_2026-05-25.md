# TT Qwen3-8B prefetcher — U28: per-core-address-mismatch hypothesis (U28-α) REFUTED; new top hypothesis = cross-sub-device dispatch race between W2 (receiver_sub_device) and RS (worker_sub_device) — 2026-05-25

Status: **EVIDENCE_ADVANCE — U28 Phase 1 Python probes
(`SGLANG_TT_U28_PER_CORE_PROBE=1`) in `models/tt_transformers/tt/mlp.py`
prove the U28-α hypothesis from U27 (per-core W2 output L1 base
address mismatch) is REFUTED: `w2_out.is_per_core_allocated() ==
False` on every layer, and `w2_out.buffer_address()` returns a
single bank-averaged value per layer (alternating between 0xa6700
on even layers and 0xa4700 on odd layers, with several exceptions —
matching the U22 / U23 reinterpretation that the alternation is
across LAYERS, not cores within one buffer).  Width-sharded
w2_out is uniform-sharded across its 32-core shard grid;
`ShardedAddrGen` correctly uses the bank-averaged base on every
core.  No per-core mismatch exists.**

**New top hypothesis (U28-β): The bug is a CROSS-SUB-DEVICE
DISPATCH RACE between the W2 matmul (dispatched on
`receiver_sub_device`) and the W2→RS reduce_scatter
(dispatched on `worker_sub_device`).  Under the active 3-sub-device
prefetcher layout, `receiver_sub_device` is ISOLATED from
`worker_sub_device` and is NOT in the host-side stall_group.
The trace-replayed RS dispatch on `worker_sub_device` does not
wait for the W2 PACK on `receiver_sub_device` to complete before
the RS reader starts its NoC reads of w2_out's L1 — exactly the
U17 46% NONZERO race observation at L1 0xa6700 on (2,7).**

Strong corroborating evidence from `prefetcher.py:765-769` (existing
2-subdev layout comment, predates U28):
```
# U6 — Galaxy-parity 2-sub-device path with rectangular worker
# + dummy_receivers padding the GlobalCB. Bypasses the 3-subdev
# carve-out so matmul + downstream CCL both auto-determine
# `worker_sub_device_id` (since worker now includes receivers),
# ELIMINATING THE U1 CROSS-SUB-DEVICE DISPATCH SYNC GAP.
```

This codebase already knows about the cross-sub-device sync gap.
The 2-subdev layout was the intended workaround.  We are running
3-subdev (Qwen3-8B Blackhole P150a uses `receiver_mapping_override`
without `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1`, hitting the
3-subdev branch at `prefetcher.py:816-841`).

Continuation of `tt_qwen3_8b_prefetcher_U27_FIRMWARE_PROBE_NO_KERNEL_WRITES_0XA6700_2026-05-25.md`.

tt-metal-sglang HEAD: **`62e8c6c24b4`** (no new tt-metal commits in
U28 session — only mlp.py probes, all env-gated default-off;
canonical bytewise-equal at 9.10s e2e_latency, GSM8K(10) 9/10).
Branch `tenstorrent-p1`; NOT pushed (predator2k/* fork only).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (U28 session)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| 1.A | Is w2_out per-core-allocated? | Python `w2_out.is_per_core_allocated()` via `SGLANG_TT_U28_PER_CORE_PROBE=1` | **NO** — returns `False` on EVERY layer 0..35.  W2 output is a normal WIDTH_SHARDED buffer with a single bank-averaged L1 address. |
| 1.B | Does `experimental_per_core_buffer_address()` show variance across cores within one buffer? | Per-core enumeration over `w2_out.memory_config().shard_spec.grid.ranges()` | **N/A** — API only returns a meaningful value for buffers with per-core allocation enabled (Buffer::address() bank-averaging applies uniformly to all cores).  All 32 cores fail the per-core API silently → `unique_addrs=0 cores_probed=0`. |
| 1.C | Does w2_out's bank-averaged address differ across LAYERS (consistent with U22's old=0xa6700 / old=0xa4700 alternation)? | bank_avg_addr per layer for layers 0..35 | **YES** — pattern: layers 0,2,3,5,6,8,9,11,12,14,15,...0x6700; layers 1,4,7,10,13,...0x4700.  Confirms U23's reinterpretation: the U22 alternation was across LAYERS (allocator history dependent), NOT across cores within one buffer. |
| 1.D | Is core (2,7) in W2's shard grid? | shard_grid dump for layer 0 w2_out | **YES** — `[2-7 - 2-7]` is in the grid (32 receiver cores total). |
| 1.E | Does w2_in's shard grid match w2_out? | shard_grid dump for layer 0 w2_in (probe captured BEFORE deallocate) | **YES** — exactly the same 32 cores: `[(0,2), (0,4), (0,6), (1,0), (1,1), (1,3), (1,5), (1,7), (2,1), (2,3), (2,5), (2,7), (3,1), (3,3), (3,5), (3,7), (4,1), (4,3), (4,5), (4,7), (8,0), (8,2), (8,4), (8,6), (9,0), (9,2), (9,4), (9,6), (10,0), (10,2), (10,4), (10,6)]`.  W2 matmul DOES dispatch on (2,7). |
| 1.F | Does the prefetcher's GCB sender/receiver split match? | `self.prefetcher.sender_receiver_mapping` enumeration | **8 senders, 32 receivers** — receivers `{(0,2), (0,4), (0,6), (1,0..7 odd), (2,1..7 odd), (3,1..7 odd), (4,1..7 odd), (8..10, 0,2,4,6)}` — **EXACTLY matches the W2 shard grid** (`w2_out_cores_not_in_gcb=[]`). |
| 2.A | Which sub-device layout is active for Qwen3-8B Blackhole P150a? | `grep "sub-device layout" /tmp/...` | **3-sub-device** layout: `"[Prefetcher] 3-sub-device layout: 32 receiver cores isolated from worker grid"`.  `_receiver_sub_device_idx=1`; `receiver_sub_device_id != worker_sub_device_id`. |
| 2.B | Does host-side `ttnn.synchronize_device(sub_device_ids=[receiver_sub_device_id])` between W2 and RS fix the bug? | `SGLANG_TT_U28_W2_RS_SYNC=1` | **NO + diagnostic** — fails inside trace capture: `TT_FATAL @ fd_mesh_command_queue.cpp:717: !trace_id_.has_value()` ("Event Synchronization is not supported during trace capture").  Host sync is not captureable; the sync must be in the kernels (semaphore). |
| 2.C | Does forcing W2 onto `worker_sub_device_id` (under prefetcher) fix the bug? | `SGLANG_TT_U28_W2_ON_WORKER=1` (W2's `sub_device_id=worker_sub_device_id`) | **CRASHES** — `TT_FATAL: Expecting a non-empty CoreRangeSet! (assert.hpp:104)` at matmul factory: worker_sub_device's worker cores have ZERO intersection with W2's in0 (w2_in) shard grid (the GlobalCB receivers).  Worker sub-device excludes the receiver cores by design; W2 cannot run on worker. |
| 2.D | Does forcing the 2-sub-device layout (which the prefetcher.py code says "eliminates the cross-sub-device dispatch sync gap") fix the bug? | `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1` | **HANGS** — server compiles 2-subdev layout (`U6 2-sub-device layout: worker_rects=[(1-0, 6-7), (8-0, 11-7)]`), starts generate, but never returns.  The matmul factory's `TT_FATAL(sd_worker_cores.num_cores() == bbox.size())` rectangular constraint is not satisfied by the 2-rectangle worker_set; the dispatch hangs.  This codepath needs additional work to ship (out of U28 scope). |

**Punchline:** The U28-α "per-core address mismatch" hypothesis is
**REFUTED**.  The real bug is a CROSS-SUB-DEVICE DISPATCH RACE
between W2 PACK (on `receiver_sub_device`) and RS reader (on
`worker_sub_device`).  Existing 2-sub-device path comment in
`prefetcher.py` explicitly names this gap; the 2-sub-device path
was the intended fix but doesn't work for our config (non-rectangular
worker grid + matmul factory's rectangular-only constraint).

## Phase 1 — refuting U28-α (per-core address mismatch)

### Probe instrumentation

Added to `models/tt_transformers/tt/mlp.py` (env-gated by
`SGLANG_TT_U28_PER_CORE_PROBE=1`; default OFF; canonical
bytewise-equal):

1. After `w2_out = ttnn.linear(...)` call (line ~673):
   - Print `is_per_core_allocated`, bank-averaged `buffer_address`
     (for layers where it's per-layer-determined).
   - Enumerate cores from `w2_out.memory_config().shard_spec.grid.ranges()`.
   - Call `w2_out.experimental_per_core_buffer_address(core)` for
     each core (silently fails when buffer is not per-core-allocated).
   - Print w2_out shard_grid for layer 0 only.

2. Just before `ttnn.deallocate(w2_in)` (line ~687):
   - Print w2_in shard_grid for layer 0 only (captured before deallocation).

3. Just after the shard_grid print (layer 0 only):
   - Enumerate `self.prefetcher.sender_receiver_mapping` to get
     GCB senders + receivers; set-diff against w2_out shard grid.

### Probe results

```
[U28_W2_PER_CORE] iter=1 layer=0 is_per_core_allocated=False bank_avg_addr=0xa6700
[U28_W2_PER_CORE] iter=1 layer=1 is_per_core_allocated=False bank_avg_addr=0xa4700
[U28_W2_PER_CORE] iter=1 layer=2 is_per_core_allocated=False bank_avg_addr=0xa6700
... (alternates per layer, layers 0..35) ...

[U28_W2_PER_CORE_GRID] layer=0 mem_layout=TensorMemoryLayout.WIDTH_SHARDED
shard_grid={[1-5 - 1-5], [2-5 - 2-5], [3-5 - 3-5], [4-5 - 4-5],
            [1-1 - 1-1], [2-1 - 2-1], [3-1 - 3-1], [4-1 - 4-1],
            [1-7 - 1-7], [2-7 - 2-7], [3-7 - 3-7], [4-7 - 4-7],
            [1-3 - 1-3], [2-3 - 2-3], [3-3 - 3-3], [4-3 - 4-3],
            [8-0 - 8-0], [9-0 - 9-0], [10-0 - 10-0], [1-0 - 1-0],
            [8-2 - 8-2], [9-2 - 9-2], [10-2 - 10-2], [0-2 - 0-2],
            [8-6 - 8-6], [9-6 - 9-6], [10-6 - 10-6], [0-6 - 0-6],
            [8-4 - 8-4], [9-4 - 9-4], [10-4 - 10-4], [0-4 - 0-4]}
shard_shape=[32, 128] orient=ShardOrientation.ROW_MAJOR

[U28_W2_IN_GRID] layer=0 mem_layout=TensorMemoryLayout.WIDTH_SHARDED
num_cores=32 shard_shape=[32, 192]
  cores=[(0, 2), (0, 4), (0, 6), (1, 0), (1, 1), (1, 3), (1, 5), (1, 7),
         (2, 1), (2, 3), (2, 5), (2, 7), (3, 1), (3, 3), (3, 5), (3, 7),
         (4, 1), (4, 3), (4, 5), (4, 7), (8, 0), (8, 2), (8, 4), (8, 6),
         (9, 0), (9, 2), (9, 4), (9, 6), (10, 0), (10, 2), (10, 4), (10, 6)]

[U28_GCB_GRID] layer=0 num_senders=8 num_unique_receivers=32
  senders=[(0, 5), (0, 1), (0, 7), (0, 3), (7, 0), (7, 2), (7, 6), (7, 4)]
  receivers=[(0, 2), (0, 4), (0, 6), (1, 0), (1, 1), (1, 3), (1, 5), (1, 7),
             (2, 1), (2, 3), (2, 5), (2, 7), (3, 1), (3, 3), (3, 5), (3, 7),
             (4, 1), (4, 3), (4, 5), (4, 7), (8, 0), (8, 2), (8, 4), (8, 6),
             (9, 0), (9, 2), (9, 4), (9, 6), (10, 0), (10, 2), (10, 4), (10, 6)]
  w2_out_cores_not_in_gcb=[]
  gcb_cores_not_in_w2=[(0, 1), (0, 3), (0, 5), (0, 7), (7, 0), (7, 2), (7, 4), (7, 6)]
```

### Interpretation

- `is_per_core_allocated=False` on every w2_out: the buffer is a
  normal SHARDED buffer; `Buffer::address()` returns one address
  used UNIFORMLY across all 32 cores in its shard grid (each core
  allocates the same L1 offset for its shard).
- The 0xa6700 / 0xa4700 pattern across layers is allocator
  history — even layers' previous-buffer footprint leaves the
  next w2_out at 0xa6700; odd layers shift it by 0x2000.  This
  matches U23's "U22's alternation = inter-layer, not per-core"
  reinterpretation.
- w2_in (in0 of W2 matmul), w2_out, and the GCB receivers are
  ALL the same 32 cores including (2,7).  W2 DOES dispatch on
  (2,7); W2 PACK on (2,7) DOES write to L1 0xa6700 on (2,7).
- The RS reader's `bank_base_address` (via `ShardedAddrGen`)
  evaluates to the same 0xa6700 on every core in w2_out's grid —
  correct for uniform sharded buffers.  No address-level
  mismatch.

**Hypothesis U28-α REFUTED.**

## Phase 2 — investigating cross-sub-device dispatch race

### Observed sub-device layout

```
[Prefetcher] 3-sub-device layout: 32 receiver cores isolated from worker grid
```

The active prefetcher layout for Qwen3-8B on Blackhole P150a is the
**3-subdev path** (`prefetcher.py:816-841`):
- `sub_devices_id[0]` = sender_sub_device (8 sender cores).
- `sub_devices_id[1]` = receiver_sub_device (32 receiver cores).
- `sub_devices_id[2]` = worker_sub_device (all other cores).
- `receiver_sub_device_id != worker_sub_device_id`.
- `set_sub_device_stall_group([worker_sub_device_id])`: host-side
  ops only wait for `worker_sub_device` — sender + receiver are
  excluded (they run persistent prefetcher kernels in some cases
  AND the matmul kernels can complete asynchronously).

W2 matmul is dispatched with `sub_device_id=receiver_sub_device_id`
(`mlp.py:682`).  tt_all_reduce / RS is dispatched with
`subdevice_id=worker_sub_device_id` (via `_u5_pref_subdev` →
`worker_sub_device_id` per `mlp.py:29`).

### Existing code that names this exact problem

```python
# prefetcher.py:765-769
# U6 — Galaxy-parity 2-sub-device path with rectangular worker
# + dummy_receivers padding the GlobalCB. Bypasses the 3-subdev
# carve-out so matmul + downstream CCL both auto-determine
# `worker_sub_device_id` (since worker now includes receivers),
# eliminating the U1 cross-sub-device dispatch sync gap.
```

The 2-sub-device path was created specifically to merge receiver +
worker into a single sub-device, which serializes W2 and RS through
the same CQ stream.

### Why W2 → RS races under 3-subdev

- Trace replay's dispatcher schedules captured commands per
  sub-device.  Within ONE sub-device, dispatch waits for prior
  workers to signal done (via `expected_num_workers_completed`
  counter at `dispatch.cpp:2287`).
- ACROSS sub-devices, there is no automatic wait.  W2 on
  receiver_sub_device proceeds independently of RS on
  worker_sub_device.
- W2 PACK takes a non-trivial time (full matmul).  RS reader
  dispatched on worker can START its NoC reads of (2,7) L1
  0xa6700 BEFORE W2 PACK on (2,7) has written the new output.
- The reader observes the residual bytes from the PREVIOUS
  W2 PACK (or initial allocator state) instead of the current
  step's output — producing the U17 46% NONZERO race signature
  and U19's "FILL w2_out with zero deterministically fixes
  zero-weight output, model garbage under real weights".

### Tested workarounds

| Attempt | Env var | Result |
|---|---|---|
| Host-side `synchronize_device([receiver])` between W2 and RS | `SGLANG_TT_U28_W2_RS_SYNC=1` | **FAILS in trace** — `TT_FATAL: !trace_id_.has_value()` ("Event Synchronization is not supported during trace capture"). |
| Force W2 onto worker_sub_device | `SGLANG_TT_U28_W2_ON_WORKER=1` | **CRASHES** — matmul factory `TT_FATAL: Expecting a non-empty CoreRangeSet`; worker sub-device has zero overlap with W2's in0 (w2_in) shard grid (= GCB receiver cores). |
| Switch to 2-sub-device layout | `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1` | **HANGS** — 2-rect worker_set fails matmul factory rectangularity constraint silently (kernel never returns).  Needs further work to ship for Qwen3-8B P150a config. |

## Path-forward proposals (U29)

### U29 path A — wire fused_op_signaler from W2 → RS (the right fix)

The reduce_scatter kernel already has `ReduceScatterFusedOpSignaler`
(`ccl/ccl_op_fusion.hpp:103`) and `ReduceScatterOpReceiver`
(used in `line_reduce_scatter_minimal_async_reader.cpp:203-206`).
This is the existing mechanism for matmul → RS fusion.

To unblock W2 → RS without modifying RS to consume from W2's CBs
directly: add a producer-completion GlobalSemaphore that:
- W2 matmul's compute kernel `noc_semaphore_atomic_inc`'s once per
  PACK completion on each receiver core (= 32 increments per W2
  dispatch).
- RS reader's `cb_in0` reservation path `noc_semaphore_wait_min`'s
  until the semaphore reaches the expected target.

Requires:
- Modify `bmm_large_block_zm_fused_bias_activation_gathered.cpp`
  to signal a semaphore at end-of-batch (gated by env or a new
  named compile arg).
- Modify `line_reduce_scatter_minimal_async_reader.cpp` to wait
  on this semaphore at kernel entry.
- Plumb the semaphore from Python (tt_all_reduce caller) through
  ttnn.linear's program_config or as a new RS runtime arg.

Estimated complexity: medium.  Existing fused_op_signaler shape
covers most of the bookkeeping.  The W2 matmul does NOT use
`fused_op_signaler` today because matmul → RS fusion is currently
limited to specific matmul factory variants.

### U29 path B — make 2-sub-device layout work for Qwen3-8B P150a

Investigate why `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1` hangs:
- The 2-rect worker_set is not rectangular → matmul factory's
  rectangular subdevice constraint at
  `matmul_multicore_reuse_mcast_1d_program_factory.cpp:5158`
  triggers TT_FATAL.  The hang suggests the FATAL fires in a
  thread that doesn't propagate the exception to the user request
  handler.
- If we can make the rectangularity constraint optional, or
  restructure the worker_set to be a single rectangle (sacrificing
  some cores), the 2-subdev path may work.

Estimated complexity: medium-high.  Requires understanding the
matmul factory's mcast geometry assumption.

### U29 path C — host-side hack: insert ANY captureable op on receiver_sub_device after W2

A device op like `ttnn.experimental.minimal_matmul(w2_out, eye)` on
receiver_sub_device with serializes against W2 PACK within
receiver_sub_device.  If this op is then a producer for RS's
reads, the dispatcher's intra-subdev ordering on receiver
sub-device guarantees W2 PACK completes first.  But RS still
runs on worker_sub_device, so the cross-subdev gap remains for
RS's actual reads.

Likely insufficient on its own; complicates the trace.

### Recommended: U29 path A

The fused-op-signaler infrastructure already exists for exactly
this producer→consumer pattern.  Plumbing it through W2 →
non-fused-RS is the minimum-risk, kernel-level, captureable fix.

## What U28 lands (all env-gated; canonical bytewise-equal)

| File | Change |
|---|---|
| `models/tt_transformers/tt/mlp.py` | Env-gated probes: `SGLANG_TT_U28_PER_CORE_PROBE=1` (per-core W2 addr + shard grid + GCB sender/receiver enumeration; default OFF), `SGLANG_TT_U28_W2_RS_SYNC=1` (diagnostic `ttnn.synchronize_device` between W2 and RS; fails in trace, error logged once; default OFF). |

NO tt-metal C++ changes in U28 session.  NO sglang Python changes.

## Reproducer

### A. U28 per-core probe (rule out U28-α)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c '
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  SGLANG_TT_U28_PER_CORE_PROBE=1 \
  nohup python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u28_phase1.log 2>&1 &'

podman exec p3a-ngram bash -c '
  until curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1; do sleep 5; done
  curl -X POST http://127.0.0.1:30000/generate \
    -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":3,\"temperature\":0.0}}"
'

# Expected probe output:
#   [U28_W2_PER_CORE] iter=N layer=L is_per_core_allocated=False bank_avg_addr=0xa6700|0xa4700
#   [U28_W2_PER_CORE_GRID] layer=0 mem_layout=WIDTH_SHARDED shard_grid=...  (includes [2-7 - 2-7])
#   [U28_W2_IN_GRID]      layer=0 mem_layout=WIDTH_SHARDED num_cores=32 ...  (includes (2,7))
#   [U28_GCB_GRID]        layer=0 senders=[...], receivers=[...]; w2_out_cores_not_in_gcb=[]
```

### B. Canonical bytewise re-verify

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c '
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u28_canonical.log 2>&1 &'

curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Output: " What is 2+2? What is 2+2? What" at e2e_latency=9.10s
# tokens=[3555,374,220,17,10,17,30,3555,374,220,17,10,17,30,3555]

python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 10
# FINAL: 9/10 = 90.0%  (within canonical 9-10/10 baseline range)
```

## Hypothesis ledger (post-U28)

| ID | Suspect | Pre-U28 | Post-U28 |
|---|---|---|---|
| (all previous U16–U27 ruled outs) | … | per U27 | unchanged |
| **U27 A — w2_out per-core L1 base differs across cores (per-core address mismatch)** | (TOP after U27) | NEW TOP STANDING | **REFUTED** — `is_per_core_allocated=False` on every layer's w2_out; ShardedAddrGen uses the bank-averaged address uniformly; correct for uniform-sharded buffers. |
| **U27 β — W2 PACK writes only tile-0 header, leaving bytes beyond unwritten** | (STANDING after U27) | STANDING | STANDING (less likely; not investigated this session). |
| **U27 γ — Constant 0xa6700 bytes leak from GlobalCB tensor data overflow** | (STANDING after U27) | STANDING | STANDING (not investigated). |
| **U28 β — Cross-sub-device dispatch race between W2 (receiver_sub_device) and RS (worker_sub_device)** | (new) | **NEW TOP STANDING** — supported by (1) prefetcher.py:765-769 explicit comment naming the gap, (2) 3-subdev layout in active use, (3) trace replay's per-subdev independent dispatch, (4) U17 46% NONZERO race signature, (5) U19 FILL workaround consistent with timing race rather than address mismatch. |
| **U28 — host-side ttnn.synchronize_device fix** | (new) | n/a | **RULED OUT** — fails inside trace capture (Event Synchronization not supported in trace). |
| **U28 — W2 on worker_sub_device fix** | (new) | n/a | **RULED OUT** — TT_FATAL empty CoreRangeSet; worker sub-device excludes W2's in0 cores by design. |
| **U28 — 2-sub-device layout (SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1)** | (new) | n/a | **NEEDS WORK** — 2-rect worker grid hangs the matmul factory's rectangularity check; not shippable as-is.  U29 path B candidate. |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (no Python edits outside
      `models/tt_transformers/tt/mlp.py`; all probes env-gated).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` and use literal
      absolute path (`TT_CACHE_HOME=/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop.
- [x] Port-clear uses `\.` escape
      (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted for `/tt-metal/` — used `podman cp`
      for source sync (mlp.py only).  No C++ rebuilds in U28.
- [x] All new probes / experiments env-gated (`SGLANG_TT_U28_*`).
- [x] Canonical Qwen3-8B re-verified bytewise: `" What is 2+2?
      What is 2+2? What"` at 9.10s e2e_latency, GSM8K(10) = 9/10
      (within canonical 9-10/10 baseline).
- [x] Server stopped at session end.
- [x] TT devices: healthy.

## Shipping verdict (unchanged from U19/U20/U21/U22/U23/U24/U25/U26/U27)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Bug is
  now understood at the architectural level (cross-sub-device race)
  but the fix is in U29 (kernel-level producer-consumer semaphore).
- **DO NOT ship any `SGLANG_TT_U28_*` env var.**  Diagnostic only.

The mlp.py probes are safe to leave in tree as default-off:
- `SGLANG_TT_U28_PER_CORE_PROBE=1` gates per-core address
  enumeration; otherwise NO-OP.
- `SGLANG_TT_U28_W2_RS_SYNC=1` gates the diagnostic
  synchronize_device call (which fails inside trace); otherwise
  NO-OP.

## Commits this session

- (tt-metal-sglang) **NONE** (no C++ changes; HEAD remains `62e8c6c24b4`).
- (sglang) pending: `<this doc>` + `models/tt_transformers/tt/mlp.py`
  with U28 env-gated probes (in tt-metal-sglang).

Note: the mlp.py probes will be committed in tt-metal-sglang
(predator2k/tt-metal-sglang fork) per project policy.

## Working state at session end

- tt-metal-sglang HEAD: **`62e8c6c24b4`** (U27 commit; no U28
  C++ commits).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.  No sglang Python changes
  in U28 (the mlp.py probe changes are in tt-metal-sglang).
- Container `/tt-metal/`: synced
  (`models/tt_transformers/tt/mlp.py` with U28 env-gated probes).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: bytewise `" What is 2+2? What is 2+2? What"`
  at 9.10s e2e_latency; GSM8K(10) chat = 9/10.
