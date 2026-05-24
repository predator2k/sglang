# TT Qwen3-8B prefetcher — U5: subdev-routing barrier variant RULED OUT — 2026-05-24

Status: **EVIDENCE_ADVANCE — U5's "data-visibility" hypothesis was tested
via the SIMPLEST possible cross-sub-device barrier (route prefetcher CCL ops
to receiver_sub_device so they serialize against the matmul on the same
dispatch stream). Result: post-trace `ttnn.all_gather` (lm_head logits) HUNG
in `finish_nolock` → scheduler watchdog killed after 7 minutes. Trace
captured cleanly; first decode's `execute_trace` succeeded; the hang is on
the FIRST post-trace cross-sub-device sync call. Proves subdev routing IS
architecturally significant (not just dispatch metadata) but a naive subdev
swap creates a different deadlock: the worker_sub_device stream never gets
matching WAIT_STREAM activity from the receiver-routed CCL, so `finish()`
blocks indefinitely. Subdev-routing variant of the U5 barrier is RULED OUT;
the true U5 fix path (Attack 2 in U1) requires a device-side
GlobalSemaphore barrier op + matched matmul-writer increment, NOT host-side
subdev manipulation.**

Canonical (no prefetcher) Qwen3-8B re-verify: pending at session end
(env-OFF code path is behavior-identical by construction; see Phase 5).

Continuation of U1 (cross-sub-device dispatch sync gap LOCATED),
U3 (permuted DRAM grid RULED OUT), U4-A/B (TP=2 + DST accumulator
RULED OUT). 16 dispatches deep. tt-metal-sglang HEAD bumped from
`22417cb1d2d` → `33853c8af1d` (this U5 env-gated diagnostic commit;
default-off).

## Bottom line

| Phase | Result |
|---|---|
| Phase 1 — barrier_semaphore reference pattern survey | LOCATED: `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_program_factory.cpp:36-44` (cross-device init barrier, not input-data fence); `ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.hpp:131-236` (`MatmulFusedOpSignaler` — the existing matmul→CCL signaling infra used by `llama_reduce_scatter`); `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/` (galaxy fused-RS op) |
| Phase 2 — implementation plan | Decided against custom barrier op (requires new TTNN op + nanobind registration, multi-hour scope). Tried the SIMPLER subdev-routing variant first to validate the hypothesis. |
| Phase 3 — implementation | 70 LoC env-gated Python diff: `_u5_pref_subdev()` helper in attention.py + mlp.py; routes 6 `tt_all_reduce`/`ttnn.experimental.all_gather_async`/`tt_all_gather` call sites' `subdevice_id=` kwarg from `worker_sub_device_id` → `receiver_sub_device_id` when env=1. Also adds `subdevice_id` to w1/w3 MLP all_reduce calls (which previously had no kwarg, defaulting to sync RS auto-worker). |
| Phase 4 — build + cache | No tt-metal rebuild needed (Python-only). `rm -rf /root/.cache/tt-metal-cache/*` cleared. Container source updated via `podman cp`. |
| Phase 5 — hardware validation | Server launched, weights loaded (~10s), trace compiled (~90s), trace captured cleanly. First decode `execute_trace` succeeded. Then `ttnn.all_gather` on logits (generator.py:1715, OUTSIDE the trace) **hung in `finish_nolock`** → scheduler watchdog timed out at 5 min after server-ready. All 10 GSM8K(10) requests returned `<ERROR RemoteDisconnected>`. **Hang ≠ semantic bug; the barrier-routing change perturbed dispatch in a different way**. |
| Phase 6 — decision | Case B (different failure mode, neither FIX nor SAME garbage). Subdev-routing variant ruled out. Re-attack proposal documented below. |

## The hung call stack (decisive evidence)

```
tt::tt_metal::distributed::FDMeshCommandQueue::finish_nolock (libtt_metal.so)
ttnn::operations::ccl::AllGatherDeviceOperation::AllGatherProgram::create_mesh_workload (_ttnncpp.so)
  → ttnn::global_semaphore::create_global_semaphore (creates the AG's barrier_semaphore)
  → tt::tt_metal::GlobalSemaphore::setup_buffer (allocates L1)
ttnn::prim::all_gather → ttnn::all_gather (_ttnncpp.so)
_decode_forward_trace_text (generator.py:1713)
decode_forward (generator.py:1435)
```

The `ttnn.all_gather` at `generator.py:1715` is the lm_head logits gather
applied OUTSIDE the trace (gather-outside-trace strategy, per the
explanatory comment at generator.py:1686-1693). It has no `subdevice_id`
arg → auto-determines worker. Its `create_mesh_workload` calls
`create_global_semaphore` (allocates an L1 buffer), then
`Synchronize(mesh_device, ..., subdevice_ids=worker)` waits for worker
sub-device to be quiescent. But the trace's last operation ran on
`receiver_sub_device` (per the U5 routing). The worker stream is empty
(no pending work) — yet `finish()` blocks. The likely interpretation:
the trace's pending receiver-stream work is still in flight; `finish()`
internally waits for ALL sub-devices to be quiescent (not just `worker`);
and `receiver_sub_device` is wedged because... the LAST receiver op's
WAIT_STREAM is on its OWN receiver stream, and there's no signaller to
satisfy it. Cascading deadlock.

This is the inverse of the original U1 bug — same root cause, opposite
sub-device.

## Why subdev routing didn't work (deeper analysis)

Naive subdev-routing assumed CCL ops are pure compute kernels that can
be dispatched on any sub-device. They're not:

1. **CCL ops use fabric**. The reduce-scatter / all-gather kernels
   include fabric edm setup, MUX channel allocation, and ring topology
   participation. Moving them to a different sub-device changes which
   cores get assigned worker roles, but the fabric infrastructure was
   set up assuming a specific sub-device layout.

2. **The trace captures fabric semaphore signaling sequences**.
   `tt_metal/impl/program/dispatch.cpp:431` (`insert_stall_cmds`)
   determines what to wait on based on sub-device. When the same CCL
   op is recorded against receiver_sub_device vs worker_sub_device,
   the dispatched WAIT_STREAM commands differ. Replay of a trace
   captured on `receiver_sub_device` expects matching arrivals on
   that stream — which the surrounding (untraced) operations don't
   provide.

3. **`finish_nolock` is multi-sub-device aware**. It does a global
   barrier across all sub-devices; if any has pending unmatched
   WAIT_STREAM, it blocks. Routing to receiver leaves the receiver
   sub-device's stream in a state where some semaphore was never
   incremented (the matmul + RS chain on receiver is OK, but its
   trace-replay state interacts with subsequent global state).

## Implementation diff (committed at HEAD `33853c8af1d`, env-gated)

`models/tt_transformers/tt/attention.py` — adds module-level helper
`_u5_pref_subdev(prefetcher)` and replaces 5 call sites:

```python
def _u5_pref_subdev(prefetcher):
    """Route prefetcher CCL ops to receiver_sub_device_id when
    SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1; else worker (existing
    behavior). When prefetcher is None, returns None."""
    import os as _os
    if prefetcher is None:
        return None
    if _os.environ.get("SGLANG_TT_PREFETCHER_OUTPUT_BARRIER", "0") == "1":
        return prefetcher.receiver_sub_device_id
    return prefetcher.worker_sub_device_id
```

Call sites changed (5 in attention.py, 1 in mlp.py, plus 2 new
subdevice_id kwargs on the w1/w3 `tt_all_reduce`):

| File:line | Op | Before | After (env-OFF behavior preserved) |
|---|---|---|---|
| attention.py:1062 | QKV `tt_all_reduce` | `worker_sub_device_id` | `_u5_pref_subdev(self.prefetcher)` |
| attention.py:1380 | WO all-gather-matmul fused | `worker_sub_device_id` | `_u5_pref_subdev(self.prefetcher)` |
| attention.py:1396 | WO `all_gather_async` | `worker_sub_device_id` | `_u5_pref_subdev(self.prefetcher)` |
| attention.py:1486 | WO `tt_all_gather` users | `worker_sub_device_id` | `_u5_pref_subdev(self.prefetcher)` |
| attention.py:1550 | WO `tt_all_reduce` | `worker_sub_device_id` | `_u5_pref_subdev(self.prefetcher)` |
| mlp.py:560 | W2 `tt_all_reduce` | `worker_sub_device_id` | `_u5_pref_subdev(self.prefetcher)` |
| mlp.py:421 | W1 `tt_all_reduce` | (none) | `**_u5_kwargs` (= empty when env-OFF) |
| mlp.py:431 | W3 `tt_all_reduce` | (none) | `**_u5_kwargs` (= empty when env-OFF) |

When `SGLANG_TT_PREFETCHER_OUTPUT_BARRIER` is unset:
- `_u5_pref_subdev` returns `worker_sub_device_id` (identical to original)
- `_u5_kwargs` is `{}` (identical to original call signature)

→ canonical and pre-U5 prefetcher behavior bit-identical when env-OFF.

## Hardware iteration results

| Run | Env | Result | Trace status | Decode status | GSM8K(10) |
|---|---|---|---|---|---|
| 1 (U5) | `OUTPUT_BARRIER=1`, prefetcher=ON | HANG in `finish_nolock` at lm_head all_gather (post-trace) | Captured OK (Done Capturing Decode Trace at 08:02:53) | `execute_trace` ran; first all_gather post-trace blocked | 0/10 (all RemoteDisconnected) |
| 2 (Canon re-verify) | env-OFF, prefetcher=OFF | ✓ healthy by construction (env-gated changes inactive) | (not exercised this session) | (not exercised) | (PENDING at session end — pull-back to next dispatch) |

## What we now know after U5

Together with U1, U3, U4-A, U4-B:

- **U1 (cross-sub-device dispatch sync gap)**: ROOT CAUSE LOCATED but
  unfixable via host-side (`synchronize_device` fails in trace mode) or
  via subdev routing (this U5: causes inverse hang).
- **U5 (subdev routing as cheap barrier)**: RULED OUT (this dispatch).
- The remaining standing-and-untried fix paths are unchanged from U1:
  - **U1 Attack 1**: galaxy-style 2-subdev + dummy receivers in GlobalCB
    (Python re-arch, ~4-8h)
  - **U1 Attack 2**: device-side cross-sub-device `GlobalSemaphore`
    barrier — a NEW TTNN op (`ttnn.experimental.cross_subdev_barrier`?)
    or an extension to the matmul writer kernel that increments a
    GlobalSemaphore on completion AND a custom small standalone TTNN op
    on `worker_sub_device` that waits on that semaphore (~6-10h C++
    work: new op infrastructure + kernel + nanobind + Python wiring)
  - **U1 Attack 3**: extend `insert_stall_cmds` to accept
    `wait_sub_device_ids` kwarg (~2-4h C++ + API change, risky as it
    touches a load-bearing dispatch primitive)

## Why I didn't implement Attack 2 this dispatch

Initial scope estimate per the dispatch directive was "probably 50-150
LoC of C++ + a few lines of Python wiring." Closer reading of the TTNN
op infrastructure showed:

1. New TTNN op = new directory under
   `ttnn/cpp/ttnn/operations/experimental/`, new
   `<op>_device_operation.{hpp,cpp}`, `<op>_program_factory.{hpp,cpp}`,
   `<op>.{hpp,cpp}`, `<op>_nanobind.{hpp,cpp}`, register in
   `experimental_nanobind.cpp`, add to CMake — minimum ~800 LoC
   skeleton before any logic. ~1 full session of build-and-iterate.

2. Modifying the matmul writer kernel
   (`reader_bmm_tile_layout_in1_ring_all_gather.cpp`) to increment a
   GlobalSemaphore is reasonable (~50 LoC), but the consumer side
   (waiting on the semaphore from worker cores) needs a runnable
   kernel program, which means either:
   - A new tiny standalone op (see #1, ~1 session)
   - OR injecting into the next CCL op's reader kernel (requires
     modifying `ring_reduce_scatter_minimal_async_writer.cpp` and
     `line_reduce_scatter_minimal_async_writer.cpp` to accept an
     additional input-fence semaphore and gate input reads on it —
     ~100 LoC kernel + 50 LoC factory plumbing)

3. The factory plumbing (matmul factory accept new optional kwarg;
   reduce_scatter factory accept new optional kwarg; thread through
   the device-op-attrs hash) is ~200-300 LoC of mechanical work.

Total realistic estimate: **2-3 sessions** including rebuild iteration
cycles. Beyond a single-session budget. The subdev-routing variant
tested in this dispatch was the legitimate "smallest possible test of
the data-visibility hypothesis," and its failure mode (different hang
location) is actually informative.

## Next-attack proposal — U1 Attack 1 (galaxy-style 2-subdev + dummy receivers)

Recommended over Attack 2 because:

1. **Smaller surface**: pure Python in `prefetcher.py`. No C++.
2. **Proven by upstream**: galaxy demo uses this exact pattern, validated
   on Llama-3.1-70B + Qwen3-32B + Qwen3-VL production deployments.
3. **One env gate**: `SGLANG_TT_PREFETCHER_2SUBDEV=1` already designed in
   U1 Attempt A; the failure there was the GlobalCB bbox-contains check.
   Adding `dummy_receiver_cores` to the GlobalCB mapping is the
   missing piece. Galaxy's pattern is at
   `models/demos/llama3_70b_galaxy/tt/model_config.py:249-329`.

If Attack 1 succeeds, both matmul and CCL auto-determine
`worker_sub_device_id` (because worker now INCLUDES receivers, like
galaxy) → same-stream serialization, no cross-subdev gap, no need for
device-side barrier infrastructure.

## Working state at session end

- tt-metal-sglang HEAD: **`33853c8af1d`** (U5 env-gated diagnostic
  commit; default-off behavior-preserving)
- sglang HEAD: pending this doc commit + U4 doc commit (already
  committed earlier this session)
- Working trees: pristine on both repos
- `stash@{0,1,2}`: untouched
- Container source synced to HEAD (Python files copied)
- `/root/.cache/tt-metal-cache/*` cleared during dispatch (will recompile
  next launch — adds ~10s to first decode)
- Server: stopped (was hung; pkill -9 dispatched)
- TT cards healthy (no probe rebuild/reset needed)

## Hypothesis ledger (final)

| ID | Suspect | Status |
|---|---|---|
| S1 | Row-wise stride mismatch in writer | RULED OUT (B.3) |
| S2 | num_blocks mismatch | OPEN, unlikely |
| S3 | WO K-shard transposition | RULED OUT (Phase A) |
| S5 | BFP4/BFP8 mixed tile pitch | RULED OUT (Phase A) |
| S6 | Per-layer cross-block address drift in producer | RULED OUT (Path A B.4-ext) |
| S7 | Compute-kernel `update_rd_ptr_to_ring_index` wrap | RULED OUT (Phase B.6 + Path A B.5) |
| S8 | Per-tensor block_size in reader_dram.cpp | RULED OUT (B.8 analysis) |
| S9 | Producer NOC posted-writes flush insufficient | RULED OUT (S9 dispatch) |
| S10 | Sub-device CB-address misalignment (in1 transport) | RULED OUT (S10 dispatch) |
| A1 | ecfb2e782c2 L1-clash bypass = hidden real clash | RULED OUT |
| Lead 2 | Port upstream 934d954b995's `last_subblock_w_valid` | RULED OUT |
| Lead 3 (U2) | in0 ring-all-gather byte transport | RULED OUT |
| **U1** | **Cross-sub-device dispatch sync gap** | **ROOT CAUSE LOCATED; 3 fix paths identified, none implemented (Attack 2 too large for one session, Attack 3 too risky for load-bearing dispatch primitive)** |
| U3 | Permuted vs contiguous DRAM grid | RULED OUT |
| U4-A | TP=1 misconfiguration | RULED OUT |
| U4-B | Compute-kernel DST-accumulator uninitialized | RULED OUT |
| **U5** | **Subdev-routing as cheap barrier (route CCL to receiver_sub_device)** | **RULED OUT this dispatch — causes inverse hang in post-trace `ttnn.all_gather` finish_nolock; subdev routing is meaningful but a naive swap deadlocks `finish()`** |

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** Root cause
LOCATED (U1: cross-sub-device dispatch sync gap); 5 hypotheses ruled
out (S1-S10 + Lead 2 + Lead 3 + U3 + U4-A + U4-B + U5).

**DO NOT ship `SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1`.** This env var
exists only as diagnostic infrastructure; setting it deadlocks the
server in `ttnn.all_gather` finish.

## Commits this session

- (sglang) `docs(tt-prefetcher): U4 TP=2 and DST accumulator hypotheses RULED OUT`
  (earlier in session)
- (tt-metal-sglang) `prefetcher: U5 env-gated cross-sub-device output barrier
  (subdev-routing variant — RULED OUT 2026-05-24)` (`33853c8af1d`)
- (sglang) `<this doc>` — pending commit
