# TT Qwen3-8B prefetcher — fork-patch suspect ranking (research only) — 2026-05-23

Status: **RESEARCH ONLY — no source edits, no server runs. Identifies which of
the 61 commits on top of tt-metal base `89686ee78d` could plausibly have
regressed prefetcher correctness, and flags upstream fixes our fork is missing.
The single strongest signal is an upstream commit we do NOT have:
`934d954b995` "[Bug Fix] DRAM Matmul - Fix Compute Reading Un-pushed Data"
(2026-05-21) — a literal description of our symptom on a sibling matmul factory.**

## Fork base + commit count

- Base: `89686ee78d` (upstream `tenstorrent/tt-metal`, 2026-05-12) — confirmed
  ancestor of `upstream/main`.
- Fork branch: `tenstorrent-p1`, HEAD `c1e3437b78e` (B.8 SKIP-reroute).
- Commits on top of base: **61** (60 + 1 EAGLE Python flag).
- Upstream commits since base: **347** (we are 11 days behind).

## Suspect commits — files that match the dispatch suspect-path list

A "suspect" commit touches `ttnn/cpp/ttnn/operations/prefetcher/**`,
`ttnn/cpp/ttnn/operations/matmul/**`, `tt_metal/impl/sub_device/**`,
`tt_metal/impl/program/**`, `tt_metal/impl/trace/**`,
`tt_metal/distributed/**`, `tt_metal/impl/device/device.cpp`, or the global-CB
machinery. (Pure Python touches `models/tt_transformers/tt/prefetcher.py` etc.
are tracked separately as "Python-prefetcher-config" — they cannot regress
on-device kernel correctness but can change L1 geometry, CB sizing, sub-device
layout, or stall-group composition.)

| Rank | SHA | Subject | C++ files touched | Why suspect | Plausibility |
|------|-----|---------|-------------------|-------------|--------------|
| 1 | `ecfb2e782c2` | perf(prefetcher): restore cascade-WIP A.3 baseline | `tt_metal/api/{program,mesh_device}.hpp`, `tt_metal/impl/program/{program,program_impl}.{cpp,hpp}`, `tt_metal/impl/sub_device/{sub_device_manager,sub_device_manager_tracker}.{cpp,hpp}`, `tt_metal/distributed/{fd_mesh_command_queue,mesh_device,mesh_device_impl}.{cpp,hpp}`, `tt_metal/impl/trace/dispatch.hpp`, `tt_metal/third_party/{tracy,umd}` submodule bumps, 3× embeddings program factory | Adds `global_program_` flag that bypasses `determine_sub_device_ids` core-membership validation; skips ALL `lowest_occupied_compute_l1_address` checks when DEFAULT manager is active; remaps trace sub_device_ids; bumps umd & tracy. Also reorganizes prefetcher Python (3-sub-device layout, receiver_sub_device_id, dynamic_worker_core_grid → 16-core rect, decode-manager cache flush). Touches every S10-relevant surface. | **HIGH** |
| 2 | `c2d7f735c09` | fix(dispatch): layer-18 + 18.5 WAIT_STREAM | `tt_metal/distributed/fd_mesh_command_queue.cpp`, `tt_metal/impl/trace/dispatch.cpp` | Adds `wait_sub_device_ids` overload to `issue_trace_commands`; rewrites the WAIT_STREAM logic for "remapped" traces (substitutes captured stream index, drops N_tensix/N_eth additions). This is dispatch-critical and can stale-data the matmul if a single corner case mishandles a non-remapped sub-device. | **HIGH** |
| 3 | `31d55797b56` | fix(prefetcher): stream-0/N mismatch cross-manager trace replay | `tt_metal/distributed/fd_mesh_command_queue.cpp`, `tt_metal/distributed/mesh_trace.hpp`, `tt_metal/impl/sub_device/sub_device_manager_tracker.cpp` | Introduces `captured_sub_device_ids` field, post-trace counter swap, `register_default_trace_on_active_manager`. Direct prerequisite for the Layer-18 stream rerouting. If any sub-device id arithmetic is off-by-one, all matmul programs running on the worker stream get wrong expected-completion counters. | **HIGH** |
| 4 | `a46a236be73` | A.3 extras: optional core_range_set kwarg through layer_norm/rms_norm | `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_*.{cpp,hpp}`, `ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.{cpp,hpp}`, `rmsnorm_nanobind.cpp` | Adds optional `CoreRangeSet` kwarg through layernorm/rmsnorm. Self-described as behavior-preserving when no caller passes the kwarg, but `ecfb2e782c2` DOES wire up callers via `prefetcher_norm_grid`. If the threading skips a default-grid path on the prefetcher's RMSNorm, the norm output (consumed by every downstream matmul) could be wrong on a subset of cores. | **MED** |
| 5 | `cb4a6b678f9` / `f38959dbc38` | ag_matmul Stage 1 lift+revert | `all_gather_async_default_program_factory.cpp` | Lifted and immediately re-asserted a `TT_FATAL`. Net diff = zero, but worth audit-confirming the revert is byte-exact (it is — checked). | **LOW** (net zero) |
| 6 | `09eab40b879` / `8cc248854fb` / `cf05cc85779` | P3a.2 P300 grid clamp | `tt_metal/impl/device/device.cpp` | Clamps `compute_with_storage_grid_size().y` to ≤8 when `FabricTensixConfig != DISABLED`. Affects every consumer that reads the device grid (incl. matmul auto-configs and prefetcher.py's MUX detection). On 2× P150a the gate is `fabric_tensix != DISABLED` — needs verification that P150a runs with `DISABLED` (no MUX) so the clamp is bypassed for the Qwen3-8B target. If MUX is unexpectedly ON, the matmul's allowed-worker-cores grid is silently 12×8 instead of 12×10 and any matmul that was sized for 12×10 will produce wrong tile counts. | **MED** |
| 7 | `081215f6ae4` | prefetcher partial-BF16 + GlobalCB cross-product sizing | (Python only) | Changes `insert_tensor`'s `max_tensor_block_size` from per-tensor product to cross-product max. The canonical BFP8 path is uniform-dtype so cross-product == per-tensor product → numerically a no-op on Qwen3-8B canonical. Still worth verifying the running fork has this commit's value match what the C++ factory expects. | **LOW** |
| 8 | `81ad47c3918` | 3-sub-device layout | (Python only) | Switches from 2-sub-device to 3-sub-device layout (sender / receiver / worker) for DECODE. Receiver cores are excluded from the worker grid (subtracted via `subtract`). If the subtraction produces a non-rectangular `compute_only` set, downstream matmul auto-configs may pick odd shapes. `dynamic_worker_core_grid` in `ecfb2e782c2` works around this by hardcoding `(5,0)-(6,7)` — fragile. | **MED** |
| 9 | `4acf2659d86` / `a3246ce4d3e` | MUX-safe sender/receiver mapping | (Python only) | Adds `generate_mux_safe_sender_receiver_mapping`, excludes core `(0,0)` from receivers, auto-detects MUX via `is_blackhole()`. Hardcodes `max_row=7`. Wrong on non-MUX Blackhole — but the user is on 2× P150a where MUX detection should fire on `is_blackhole()` regardless. | **LOW** (config knob) |
| 10 | `9b32e945e87` | standalone bug fixes (lazy worker_start_core, etc.) | (Python only) | Lazy `worker_start_core` read in SDPA config + output mem config; gates `is_prefetcher_supported` iteration over `legal_receiver_cores`. Adds `worker_start_core` property. Self-described as behavior-preserving with `use_prefetcher=False`; for `use_prefetcher=True` it changes the SDPA grid origin from hardcoded `(1,0)` to whatever `all_worker_cores_range_set.ranges()[0].start` returns. On 2× P150a Qwen3-8B this becomes `(5,0)` (the `dynamic_worker_core_grid` rect). | **MED** |

## Critical upstream fixes the fork is MISSING

Of the 347 upstream commits since base, these directly intersect the suspect-paths.
We are 11 days behind upstream as of 2026-05-23.

| Upstream SHA | Date | Subject | Relevance |
|---|---|---|---|
| **`934d954b995`** | **2026-05-21** | **[Bug Fix] DRAM Matmul - Fix Compute Reading Un-pushed Data** | **Symptom-matching upstream fix**. Compute kernel was reading more from `cb_in1` than the reader pushed (`per_core_N_compute > per_core_N_in1_sender`), masked by the writer dropping padded columns and the CB being oversized. Patches `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp` + `matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp` + `bmm_large_block_zm_fused_bias_activation.cpp` to track `last_subblock_w_valid`. **Does NOT touch the `_1d_program_factory.cpp` / `_ring_all_gather.cpp` / `_gathered.cpp` path Qwen3-8B prefetcher uses, but proves this exact bug class exists in tt-metal matmul kernels.** A sibling bug in the gathered factory is plausible. |
| `ddfdb039c88` | 2026-05-15 | Fix bad optional access in CCL matmul paths | `matmul_multi_core_reuse_mcast_1d_optimized_helper` + `MeshWorkload` factory unconditionally read `allowed_worker_cores.value()` (added in `ebaaa579351`). CCL ops (`all_gather_matmul_async`, `reduce_scatter_matmul`) call these directly without `normalize_program_config`. Our fork has neither `ebaaa579351` nor the fix → this particular regression cannot manifest here. |
| `ebaaa579351` | 2026-05-15 | Add allowed_worker_cores to matmul program configs | We don't have it; backward-compatible per upstream summary. |
| `875d24c6e71` | 2026-05-19 | Add dst_full_sync_en / throttle propagation to mm factories | Bug fix; missed throttle propagation in some matmul factories. Could silently affect dst-register synchronization. |
| `7685d31873c` | 2026-05-18 | Fix bad optional access for fused matmul callers | Same family as `ddfdb039c88`. |
| `be3d3fccab0` | 2026-05-18 | [Descriptor Migration] prefetcher | Rewrites `DramPrefetcherProgramFactory` → `DramPrefetcherOperation::create_descriptor` returning `ProgramDescriptor`. Functionally equivalent per upstream but our fork is on the LEGACY pre-descriptor path. We have NOT validated the new path on Blackhole; potentially safer to STAY on legacy until the descriptor migration is independently verified on prefetcher hardware. |
| `a67eb83be3c` | 2026-05-13 | Move new device apis out of experimental | Header-move only; include-path changes. No functional risk. |

## Per-suspect deep-dive + how-to-test

### Suspect #1 — `ecfb2e782c2` (cascade-WIP A.3 baseline restore)

What it does (C++):
1. Adds `Program::set_global_program(bool)` + `is_global_program()`.
2. In `determine_sub_device_ids` (`tt_metal/impl/program/program.cpp:1809`):
   - Lifts the `num_intersections == num_cores` TT_FATAL when `global_program_` is true.
   - If `global_program_`, **collapses `used_sub_device_ids` to `{SubDeviceId{0}}`** regardless of what the kernel groups actually intersect.
3. In `lowest_occupied_compute_l1_address`
   (`sub_device_manager_tracker.cpp:227`): **early-returns `std::nullopt`** whenever the DEFAULT manager is active and `sub_device_ids` is non-empty. This entirely disables L1 clash detection on the DEFAULT-active path, including for programs whose kernel groups overlap GlobalCB receiver cores.
4. Embedding factories get rewritten to use `get_sub_device_stall_group().back()` rather than `compute_with_storage_grid_size`.
5. `tracy` and `umd` submodules are bumped.

Prefetcher relevance: HIGH. (2) and (3) explicitly weaken core-membership and L1-clash validation. If a Qwen3-8B matmul program ever gets created with `global_program_=true` (or runs while DEFAULT is active but the DECODE manager still holds the GlobalCB pages), the upstream guard that would have refused to bind cb_in1 to the wrong L1 address is gone.

How to test (cheapest first):
- **Test A** (10 min, no rebuild): `git revert --no-commit ecfb2e782c2` on a probe branch, then run `git status` only — confirm the revert applies cleanly. If it does, follow with build + GSM8K(10) under `SGLANG_TT_USE_PREFETCHER=1`. If GSM8K ≥ 7/10, **this patch (or its dependencies) is the culprit**.
- **Test B** (surgical, 30 min rebuild): keep all Python changes from `ecfb2e782c2` but revert ONLY the `lowest_occupied_compute_l1_address` early-return (drop the `if (!sub_device_ids.empty() && default == active) return nullopt;` lines). Rebuild + run. If we hit a real L1 clash assertion, that confirms the suppression was hiding a real bug.
- **Test C**: revert ONLY the `global_program_` collapse in `determine_sub_device_ids` (force back to assertion). If Qwen3-8B canonical (no prefetcher) still passes, then prefetcher-enabled crashing in determine_sub_device_ids would be the smoking gun.

### Suspect #2 — `c2d7f735c09` (Layer-18 WAIT_STREAM)

What it does: adds a `wait_sub_device_ids` optional override to `issue_trace_commands` so that for "remapped" traces the WAIT_STREAM uses the captured stream index instead of the remapped one, AND drops the `+N_tensix/+N_eth` addition for remapped traces (because go-signal acks land on the remapped stream, not the captured one).

Prefetcher relevance: HIGH if Qwen3-8B uses cross-manager trace replay. The phase B.9 doc confirms the failure point IS during trace replay (`_v25_suspend_prefetcher` flow). If the `is_remapped` branch incorrectly fires for any non-remapped trace (e.g. when `descriptor->captured_sub_device_ids` is mistakenly non-empty), the WAIT counter shortens and the consumer matmul reads stale CB pages.

How to test: revert `c2d7f735c09` + `31d55797b56` together (they're a pair) on a probe branch. Rebuild. The phase B.9 doc indicates pre-Layer-18 runs HUNG on run-2 — if reverting brings the hang back without the wrong-bytes failure, that's evidence the dispatch fixes added new behavior that drives the wrong-bytes bug. If reverting yields a different symptom (hang, not garbage), the dispatch fixes are NOT the bug.

### Suspect #3 — `31d55797b56` (stream-0/N mismatch root patch)

Same surface as #2; tested together.

### Suspect #4 — `a46a236be73` (layer_norm/rms_norm core_range_set kwarg)

The kwarg is read first in `LayerNormMultiCoreProgramFactory::create_descriptor`:
```cpp
CoreRangeSet requested_cores =
    operation_attributes.core_range_set.has_value()
        ? operation_attributes.core_range_set.value()
        : (core_range_set.has_value() ? core_range_set.value() : default_core_range(device));
```
`ecfb2e782c2` wires this through prefetcher's `dynamic_worker_core_grid(16)` (16 cores at `(5,0)-(6,7)`). If the residual+norm shard splits 32-tile rows across 16 cores instead of the original 32, the norm output's per-core stride differs from what the downstream matmul expects — and the matmul reads from wrong L1 offsets.

How to test: probe-branch revert `a46a236be73`; rebuild; run prefetcher. If layer-norm crashes because `_force_unsharded` / `_pc_w2` configs in `distributed_norm.py` expect the old grid, that confirms the norm geometry is load-bearing. Then either (a) carry forward `a46a236be73` AND audit the actual shard widths byte-by-byte, or (b) revert the Python `prefetcher_norm_grid = CoreGrid(y=8, x=2)` change in tandem.

## Cross-reference with prior S1-S9 hypotheses

- **S6** (cross-block address drift in producer): Path A's B.4-extended dump used the LEGACY prefetcher path — same as our fork. Upstream `be3d3fccab0` (descriptor migration) does not change the address-encoding logic. Not implicated by any fork commit.
- **S7** (compute kernel `update_rd_ptr_to_ring_index` wrap): ruled out in B.6 + Path A B.5. No fork commit touches the gathered compute kernel.
- **S9** (NOC posted-writes flush): ruled out in S9 doc. The S9 surgical test made writer_l1.cpp non-posted; the non-posted regime produced the SAME NaN crash. No fork commit touches writer_l1.cpp.
- **S10** (sub-device CB-address misalignment): **DIRECTLY touched by `ecfb2e782c2`** which weakens the very `lowest_occupied_compute_l1_address` clash check S10 hypothesizes is failing. If `ecfb2e782c2`'s early-return is suppressing a real clash, that IS S10's failure mode. S10 verification (next session per phase B.9 doc) and reverting `ecfb2e782c2`'s C++ portion both attack the same surface.

## Recommended top-3 attacks for next subagent

### Attack 1 (highest leverage, ~3h end-to-end) — revert `ecfb2e782c2` C++ only

On a probe branch, keep all Python changes from `ecfb2e782c2` (they're load-bearing for the bench-verified TPOT regression fix) but revert the C++ surfaces:
- `tt_metal/impl/program/program.cpp` (the `global_program_` flag + collapse)
- `tt_metal/impl/sub_device/sub_device_manager_tracker.cpp` (the `lowest_occupied_compute_l1_address` early-return)
- `tt_metal/api/tt-metalium/program.hpp` (`set_global_program`)
- `tt_metal/impl/program/program_impl.hpp` (`global_program_` field)
- 3× embedding program factory revert
- DO NOT revert the tracy/umd bumps — too risky (other Blackhole bring-up depends).

Rebuild. Run GSM8K(10) chat with `SGLANG_TT_USE_PREFETCHER=1`. **Result interpretation**:
- ≥ 7/10: this is the culprit; the prefetcher bug is the suppressed L1 clash detection.
- 0/10 with a CHANGE in failure signature: still contributing; narrow further.
- 0/10 same NaN signature: not load-bearing for the bug; move on.

### Attack 2 (~2h, dispatch-side) — revert `31d55797b56` + `c2d7f735c09` together

Probe-branch revert both Layer-15/16/18 dispatch patches. Expect run-2 to HANG (as documented pre-fix). If it does NOT hang and the prefetcher succeeds, the dispatch patches are the regressor. If it hangs (most likely), they are correctness-positive and we keep them.

### Attack 3 (~4h, upstream rebase test) — cherry-pick `934d954b995` onto current fork HEAD

Even though `934d954b995` doesn't touch the gathered factory directly, cherry-picking onto fork HEAD verifies (a) we can carry it (no API drift), and (b) whether the bug class exists in our exact path. If clean: write a sibling patch for `matmul_multicore_reuse_mcast_1d_program_factory.cpp` (the 1D / ring-gathered factory used by Qwen3-8B prefetcher) following the same "track `per_core_N_compute` vs `per_core_N_in1_sender` and narrow last subblock" pattern. The Qwen3-8B prefetcher's per-tensor block_size differs between WQKV / W1 / W3 / WO / W2 (BFP4 vs BFP8 padding), so the padding skew that triggers the dram_sharded bug should similarly trigger the gathered bug.

## Conclusion

**Does any fork patch look like the culprit?**

The strongest signal is `ecfb2e782c2` (cascade-WIP restore). It bumps `umd` + `tracy` submodules, adds an opt-out flag (`global_program_`) that disables sub-device validation, and **disables L1 clash detection (`lowest_occupied_compute_l1_address` early-return) on the DEFAULT-manager-active path** — which is exactly the path the v25 prefetcher-suspend flow re-enters for chat-detect prefills and (per phase B.9) for first-decode crashes. The C++ portion of this commit was justified by the BENCH speedup (37 → 27 ms TPOT) but its correctness gating was a separate concern that this dispatch is now revealing.

A second strong signal is `c2d7f735c09` + `31d55797b56` (Layer-18 dispatch). They were added to fix a separate hang and CANNOT be wholesale-reverted without regressing the hang. Their REMAINING risk is a corner case where remapped/captured tracking off-by-ones the consumer matmul's stale-CB read.

The strongest standalone external signal is upstream `934d954b995` — a literal symptom-match bug ("compute reads un-pushed data") on a sibling DRAM-sharded matmul factory, fixed 2 days ago. Our fork does not have it AND it doesn't fix our exact factory, but it proves the bug class exists.

**Single highest-leverage next attack**: Attack 1 (revert C++ portion of `ecfb2e782c2`, keep all Python). The Python is load-bearing for boot; the C++ is the suspect surface that uniquely intersects S10.
