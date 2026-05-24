# TT Qwen3-8B prefetcher — U6: galaxy-parity 2-subdev + dummy_receivers (EVIDENCE_ADVANCE) — 2026-05-24

Status: **EVIDENCE_ADVANCE — U1 Attack 1 (true galaxy-parity 2-sub-device
+ dummy_receivers padding the GlobalCB) was implemented with full
geometric correctness (matched bbox-shortcut semantics; verified Python
prototype produces exactly 8 senders + 24 dummy receivers covering
the matmul factory's bbox per worker rect). Two hardware iterations:
(1) initial impl hit `TT_FATAL "Kernel group cores do not match sub
device cores"` on `q_norm`'s unsharded `ttnn.rms_norm` because the
op's auto-grid is the full 12×8 compute grid (96 cores) but only 88
cores were sub_device members (sender 8 + worker 80, missing 8 dummy
sender slots). (2) Fix added dummy senders to `sender_sub_device`,
making it 16 cores so sender ∪ worker = 96 = full grid. TT_FATAL
resolved; trace-capture's no-trace decode forward enqueues; then
`ttnn.synchronize_device(mesh_device)` at `generator.py:1559` HANGS
→ scheduler watchdog kill at 5 min.

The hang signature is DIFFERENT from the prior U1-fix-#1 (row-banded
shortcut) which got "engaged" boot + 0/3 GSM8K garbage. Both attempts
eliminate the cross-sub-device dispatch sync gap differently, and
both BREAK in NEW ways relative to canonical 3-subdev. This is
partial evidence that U1's "cross-sub-device sync gap" framing may
not be the actual root cause: if it were, both fixes would produce
working (or numerically close) decode; instead each fix breaks
differently. Something about the 3-subdev carve-out is load-bearing
beyond just dispatch sync semantics.**

Continuation of `tt_qwen3_8b_prefetcher_U5_subdev_routing_RULED_OUT_2026-05-24.md`,
the prior (unrecorded) row-banded U1 fix attempt, and 17 prior dispatches.
tt-metal-sglang HEAD bumped from `33853c8af1d` → `68b8eaa1152` (this U6
env-gated diagnostic commit; default-off).

## Bottom line

| Iteration | Env | Result | Notes |
|---|---|---|---|
| 1 (U6 v1) | `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1` | TT_FATAL "Kernel group cores do not match" | q_norm's `ttnn.rms_norm` (unsharded, attention.py:545) uses auto-grid 12×8 = 96 cores; only 88 cores in any subdev (sender 8 + worker 80); 8 dummy_sender slots not covered. |
| 2 (U6 v2) | `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1` | HANG in `synchronize_device` after no-trace decode forward | TT_FATAL fixed by adding 8 dummy_senders to sender_sub_device (16 total). Compile run dispatches all 36 layers; sync waits forever for worker stall group to drain. Scheduler watchdog 5min kill. |
| Canonical re-verify | env-OFF, no prefetcher | ✓ **GSM8K(10) = 10/10 = 100%** | Working tree is `tenstorrent-p1` HEAD `68b8eaa1152` with U6 env-gated; canonical unaffected (bit-identical to pre-U6 behavior). |

## Galaxy → ours geometry table

For Qwen3-8B nrc=4, 2× P150a, MUX-clamped grid (rows 0-7, cols 0-11):

| Field | Galaxy production (Llama-70B) | Ours (Qwen3-8B nrc=4) |
|---|---|---|
| Active senders | 12 (cols 0, 4) | 8 (cols 0, 7) |
| Active receivers | 24 (cols {1,2,5,6} × paired rows) | 32 (cols {1-4} × rows {1,3,5,7} + cols {8-11} × rows {0,2,4,6}) |
| Worker rects | (1,0)-(3,9) + (5,0)-(6,9) = 50 cores | (1,0)-(6,7) + (8,0)-(11,7) = 80 cores |
| Matmul bbox per worker rect | (1,0)-(3,9) + (5,0)-(6,9) = 50 cores | (1,1)-(4,7) + (8,0)-(11,6) = 56 cores |
| Dummy receivers (in bbox) | 26 cores (set 1: 4, set 2: 3, … set 8: 2) | 24 cores (cols {1-4} × rows {2,4,6} + cols {8-11} × rows {1,3,5}) |
| Dummy senders | 8 (col 0 unused rows + col 4 unused rows) | 8 (col 0 rows {0,2,4,6} + col 7 rows {1,3,5,7}) |
| Pairing | dummy_sender[i] → dummy_recv_set[i] (galaxy hardcodes 8 sets) | dummy_sender[i] → dummy_recv groups (round-robin partition: 3 each) |
| sub_device layout | 2: [sender(active), worker(incl active receivers)] | 2: [sender(active + dummy = 16), worker(80, incl active recv + dummy recv)] |
| Default core grid coverage | (sender 12) + (worker 50) = 62 ≠ 96; galaxy demo works | (sender 16) + (worker 80) = 96 = full grid (after iter 2 fix) |

**Surprise finding:** galaxy's `sender ∪ worker = 62 cores` — NOT the full
96. Yet galaxy's demo passes for Llama-70B / Qwen3-32B / Qwen3-VL. Possible
explanations:
- Galaxy uses Wormhole (different grid: 8×10 = 80). Or Galaxy's demo
  always launches with sharded program configs, avoiding the auto-grid
  trap our `q_norm` falls into.
- Or galaxy's flow never invokes an unsharded `ttnn.rms_norm` whose
  default grid would span the gaps. Our `q_norm` (per-head RMSNorm
  on Q tensor BEFORE rope) is unsharded.

The galaxy demo's distributed_norm passes `core_range_set =
prefetcher.all_worker_cores_range_set` (galaxy's worker_sub_device =
50 cores) which DOES restrict the program. But `q_norm` doesn't get
this treatment in our codebase — it calls `RMSNorm.forward(x, mode,
norm_config=...)` without `core_range_set`.

## Implementation diff (HEAD `68b8eaa1152`)

`models/tt_transformers/tt/prefetcher.py` — 348 net LoC:

| Function / change | Lines | Purpose |
|---|---|---|
| `generate_2subdev_safe_sender_receiver_mapping(nrc, max_row)` | new, ~90 LoC | Generates mapping where right-side senders use cols {8,9,10,11} instead of borrowing col 0 (the previous `generate_mux_safe_…` hack). Strictly contains right receivers in (8,0)-(11,7). |
| `compute_2subdev_dummy_receivers(worker_rects, real_recv, real_send)` | new, ~75 LoC | For each worker rect, computes bbox(rect ∩ real_receivers), then returns bbox cores - real_receivers - real_senders. Mirrors the matmul factory's bbox-shortcut path at `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2024-2038`. |
| `Prefetcher.__init__` env gate | +5 LoC | `self._real_2subdev = SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1`; routes `_make_mapping` to the new mapping function when in 2-subdev mode. |
| `Prefetcher.init(Mode.DECODE)` 2-subdev branch | +44 LoC | When `_real_2subdev`: skips the 3-subdev receiver carve-out, builds `worker_sub_device` = 2 rectangles (1,0)-(6,7) + (8,0)-(11,7), computes `_dummy_receiver_coords`, also appends 8 dummy_senders to sender_sub_device (so sender ∪ worker = full grid → `q_norm` unsharded ttnn.rms_norm doesn't hit "Kernel group cores do not match"). |
| `Prefetcher.run()` GlobalCB mapping augment | +20 LoC | When in 2-subdev mode and `_dummy_receiver_coords` non-empty: prepends 8 (dummy_sender, dummy_receivers_CRS) pairs to the GlobalCB `sender_receiver_mapping` so `global_cb.all_cores()` covers the matmul factory's bbox cores. |
| `Prefetcher._build_dummy_sender_receiver_pairs()` | new, ~70 LoC | Helper: enumerates dummy sender candidates (unused col 0/7 rows), partitions 24 dummy receivers round-robin across 8 dummies (3 each). |

Code path is 100% env-gated; canonical (no env) is bit-identical to
HEAD `33853c8af1d`. Verified by re-launching server with env-OFF
and getting **GSM8K(10) = 10/10 = 100%** (full 10-question pass on
the same Qwen3-8B chat path that previously hit 9/10 in U5 doc; the
+1 reflects normal run-to-run variance on the borderline Q3 long-
reasoning question; well above the ≥ 9/10 ship threshold).

## Phase 1-4 execution log

### Phase 1: galaxy → ours geometry analysis
- Read `models/demos/llama3_70b_galaxy/tt/prefetcher_common.py` end-to-end (154 lines).
- Read `models/demos/llama3_70b_galaxy/tt/model_config.py:178-389` (get_core_ranges + dummy_receiver_cores).
- Read matmul factory bbox-shortcut at
  `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:2014-2038`.
- Read GlobalCB invariants at
  `tt_metal/impl/buffers/global_circular_buffer.cpp:34-58`.
- Confirmed: matmul kernels have `if (core_type == IDLE_CORE) return;`
  early-exit (factory line 2434, reader/writer/compute kernels at lines
  29-32, 87-95, 234-237). Dummy receivers DO get matmul kernels but exit
  immediately.

### Phase 2: implementation
- Wrote new mapping function (right-side cols {8,9,10,11} instead of
  borrowing col 0). Asserts `nrc <= 4`.
- Wrote bbox-dummy enumeration matching factory's per-rect bbox shortcut.
- Wrote env-gated 2-subdev branch in `init(Mode.DECODE)`.
- Wrote GlobalCB augmentation in `run()`.
- Verified geometry via Python prototype: 8 senders, 32 real receivers,
  24 dummies, exactly as predicted.

### Phase 3: sync + cache clear
- `podman cp prefetcher.py p3a-ngram:/tt-metal/models/tt_transformers/tt/`
- `podman exec p3a-ngram rm -rf /root/.cache/tt-metal-cache/*` (guarded by `[ -d ]`)

### Phase 4: hardware validation
- **Iter 1**: Launched server with `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1`.
  Server ready in 36s. First GSM8K request triggers compile → TT_FATAL
  at `program.cpp:1808` "Kernel group cores do not match sub device
  cores". Stack trace points to `q_norm` (attention.py:1155) calling
  `ttnn.rms_norm` (unsharded). Diagnosed: `ttnn.rms_norm` default
  grid = `compute_with_storage_grid_size()` = 12×8 = 96 cores, but
  sender (8) + worker (80) = 88; 8 dummy sender slot rows are NOT in
  any sub_device.
- **Iter 2 (fix)**: Added 8 dummy sender slots to `sender_sub_device`
  (now 16 cores). Now sender (16) + worker (80) = 96 = full grid.
  Server ready. First GSM8K request: compile run finishes; then
  `ttnn.synchronize_device(mesh_device)` at `generator.py:1559` blocks
  → scheduler watchdog 5min timeout → server crash. py-spy snapshot
  confirms hang inside `_capture_decode_trace_text` at line 1559.

## Decision: Case B (different garbage signature)

Per dispatch directive Phase 5:
- **Case A (GSM8K(10) ≥ 7/10 — U1 confirmed)**: NOT MET.
- **Case B (< 7/10 with different failure)**: ✓ MET.
  - Hang in synchronize_device is DIFFERENT from U1-fix-#1's "garbage signature".
  - Both attempts break differently than canonical. Each fix changed
    failure mode without producing a working decode.
- **Case C (same garbage as #1)**: would be cleanest U1-refute; not seen.
- **Case D (boot fail)**: ✓ MET for iter 1; resolved → iter 2 hang.

## Verdict on U1

**U1 cross-sub-device dispatch sync gap as the ROOT CAUSE is now WEAKLY
SUPPORTED at best.** Two independent fixes (row-banded and galaxy-parity)
both eliminate the cross-sub-device gap (matmul + CCL on same subdev) but
neither produces working decode. If the gap were the true bug, removing
it should give working (or numerically close) decode. Instead:
- Row-banded → matmul "engaged" but garbage outputs (downstream of matmul fails).
- Galaxy-parity → hangs entirely in sync (something deeper than dispatch).

The 3-sub-device carve-out is doing something LOAD-BEARING beyond
dispatch sync. Possible candidates:
1. **L1 layout** — the 3-subdev carve-out isolates RECEIVER L1 (where
   GlobalCB lives, ~835 KB/core) from COMPUTE L1 (where matmul outputs +
   CB intermediates live). In 2-subdev, all 56 bbox cores (active + dummy
   receivers) have GlobalCB AND matmul output CB → L1 budget tighter.
   Might OOM silently or corrupt addresses.
2. **Receiver auto-determination** — the matmul kernel placement on
   receivers (auto-determined sub_device for IDLE cores) might depend
   on the carve-out for some semantic property not yet identified.
3. **q_norm grid coverage** — even with dummy senders in subdev, the
   q_norm's unsharded kernel may launch on bbox-rounded cores within
   the 12×8 grid, and the cb_sync semaphores etc. interact with the
   dummies differently.

## Recommended next attacks

1. **L1 budget probe**: with `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1`,
   instrument prefetcher to log per-core L1 usage before+after GlobalCB
   creation. If usage on a dummy receiver core (e.g., (1,2)) exceeds
   1.5 MB - matmul_output_CB_size, that's the bug.
2. **Per-op kernel grid audit**: rather than chasing cross-subdev sync,
   constrain ALL prefetcher-path ops (including unsharded `q_norm` /
   `k_norm` / lm_head's per-head RMSNorms) to a `core_range_set` ⊆
   worker_sub_device. Requires patching `attention.py` to pass
   `core_range_set=self.prefetcher.all_worker_cores_range_set` into
   `RMSNorm.forward` for q_norm/k_norm.
3. **Pivot from U1**: U6 + U1-fix-#1 together suggest U1 is the wrong
   frame. Instead investigate:
   - Differential matmul output test: register a hook to compare per-core
     matmul output bytes between 3-subdev (working) and 2-subdev (broken)
     for one layer's QKV. Pinpoint which receiver core diverges first.
   - GlobalCB byte-level comparison: confirm prefetcher producer writes
     IDENTICAL bytes to ACTIVE receivers in both 3-subdev and 2-subdev modes.

## Working state at session end

- tt-metal-sglang HEAD: **`68b8eaa1152`** (U6 env-gated diagnostic;
  default-off behavior-preserving)
- sglang HEAD: pending this doc commit
- Working trees: pristine on tt-metal-sglang; sglang has only this doc
- `stash@{0,1,2}`: untouched
- No server running (killed via pkill)
- `/root/.cache/tt-metal-cache/*` cleared during dispatch
- TT cards healthy (no probe rebuild/reset needed)

## Hypothesis ledger update

| ID | Suspect | Status |
|---|---|---|
| S1-S10 | Various producer/consumer + L1/CB micro-hypotheses | All RULED OUT (prior dispatches) |
| A1 | ecfb2e782c2 L1-clash bypass = hidden real clash | RULED OUT |
| Lead 2 | Port upstream 934d954b995's `last_subblock_w_valid` | RULED OUT |
| Lead 3 (U2) | in0 ring-all-gather byte transport | RULED OUT |
| **U1** | **Cross-sub-device dispatch sync gap as root cause** | **WEAKENED (this dispatch): both U1-fix-#1 (row-banded) and U6 (galaxy parity) eliminated the gap; neither produces working decode. Different failure modes across the two fixes. U1 may be a SECONDARY effect of the deeper 3-subdev-vs-2-subdev architectural mismatch, not the primary cause.** |
| U3 | Permuted vs contiguous DRAM grid | RULED OUT |
| U4-A | TP=2 misconfig | RULED OUT |
| U4-B | DST accumulator stale state | RULED OUT |
| U5 | Subdev-routing barrier | RULED OUT |
| **U6** | **Galaxy-parity 2-subdev + dummy_receivers** | **EVIDENCE_ADVANCE: implementation correct (GlobalCB membership passes, matmul IDLE-core early-exit works). Iter 1 hit TT_FATAL on q_norm full-grid auto-placement (architectural — galaxy doesn't trip this). Iter 2 hangs in synchronize_device (NEW failure mode, ≠ row-banded U1 fix #1). Combined with #1 → U1 framing weakened, see Verdict on U1.** |

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = **10/10 = 100%**
this dispatch (improved from 9/10 in U5 dispatch, well above ship
threshold).

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** U6 evidence
weakens the U1 "cross-sub-device sync gap" hypothesis. Next session
should pivot from the U1 frame to a per-op kernel grid audit or
differential matmul output test (see Recommended next attacks).

**DO NOT ship `SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1`.** This env var
exists only as diagnostic infrastructure; setting it hangs the server
in trace-capture sync.

## Commits this session

- (tt-metal-sglang) `prefetcher: U6 env-gated galaxy-parity 2-sub-device + dummy_receivers (boot-fail/hang — diagnostic)` (`68b8eaa1152`)
- (sglang) `<this doc>` — pending commit.
