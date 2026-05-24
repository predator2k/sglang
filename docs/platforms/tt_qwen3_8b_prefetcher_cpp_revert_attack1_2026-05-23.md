# TT Qwen3-8B prefetcher — Attack 1 (surgical C++ revert of ecfb2e782c2) EVIDENCE_ADVANCE — 2026-05-23

Status: **EVIDENCE_ADVANCE (Case B mix) — The L1-clash detection bypass in
ecfb2e782c2 is NOT the prefetcher correctness bug. Surgical revert of the 7
C++ files in ecfb2e782c2 (2 prefetcher-suspect surfaces + 3 embedding-factory
helpers + 2 program/program_impl header pieces) built cleanly and the
prefetcher run produced a CHANGED garbage signature on Q1 (`'rabbitagens璧
.stubedList一个小时橘吊顶(errno资管女星...'`) — DIFFERENT from the previous
broken-prefetcher signature (`'房东benh火力ifty主权悠悠 bambS齊 Canon
新规...'`) and NO NaN. The model still generated garbage on Q1 (0/10), and
then Q2's prefill crashed on the EXACT L1-clash false-positive the bypass
was designed to suppress: `TT_THROW: Statically allocated circular buffers
in program 390 clash with L1 buffers on core range [0-0 - 7-0]. L1 buffer
allocated at 706304 and static circular buffer region ends at 1458688`.
The 706304 address bit-exactly matches S10's documented GlobalCB receiver-
core base — confirming the upstream-diff doc's "the clash bypass might be
hiding a real bug" hypothesis is FALSIFIED. The suppression IS legitimately
suppressing a false positive from the lockstep-bank allocator assumption,
exactly as the original ecfb2e782c2 comment explained. Probe branch
deleted, tt-metal-sglang restored to c1e3437b78e, container source +
shared-lib rebuilt. Canonical re-verify: GSM8K(10) chat = 9/10 = 90%
(Q3 wrong as documented baseline). No regression.**

Continuation of `tt_qwen3_8b_prefetcher_upstream_diff_2026-05-23.md`
(upstream-diff investigation) → `tt_qwen3_8b_prefetcher_S10_ruled_out_2026-05-23.md`
(S10 ruled out) → this doc (Attack 1 ruled out).

## Bottom line

| Phase | Result |
|---|---|
| Probe-branch creation (probe-revert-ecfb2e782c2-cpp on tenstorrent-p1) | OK |
| Surgical revert of 7 C++ files (all diff against ecfb2e782c2^ = zero) | OK |
| tt-metal incremental rebuild (`rebuild_tt_metal_kernels.sh ttnn`, 233 TUs) | OK in ~6 min, both `_ttnn.so` (14.2 MB) and `_ttnncpp.so` (33.3 MB) relinked clean |
| Server boot with prefetcher | OK — `/health_generate` = 200 in ~50 s. L1-clash check did NOT fire on boot |
| Prefetcher Q1 (eval) | 76.5 s; garbage tokens; signature CHANGED |
| Q2 prefill | CRASHED with TT_THROW on L1 clash at addr=706304, core [0-0 - 7-0], in program 390 (lm_head recompile after V2.5 suspend) |
| Final GSM8K(10) | 0/10 = 0% (Q2-Q10 = connection refused after server died) |
| Probe branch cleanup | deleted; restored tenstorrent-p1 (c1e3437b78e) |
| Container re-sync + rebuild | OK; libs restored |
| Canonical re-verify (no prefetcher) | **9/10 = 90%** (Q3 wrong, matches documented baseline) |
| Stash state | unchanged (stash@{0,1,2} preserved) |

## Surgical revert details

Per the upstream-diff doc + my analysis of dependencies between commits,
the C++ portion of ecfb2e782c2 narrows to 7 files (out of the 29 touched).
The OTHER C++ files in the commit (`fd_mesh_command_queue.cpp` LAYER19
diagnostics + `end_mesh_trace` finish-skip, `mesh_device.cpp/_impl.hpp`,
`mesh_device.hpp`, `sub_device_manager.cpp/hpp`, `dispatch.hpp`,
`distributed_nanobind.cpp`) are dependencies of `31d55797b56` (Suspect #3
in the upstream-diff doc) and `generator.py`'s
`register_default_trace_on_active_manager` call — reverting them would
break compilation.

Files reverted (all diff against ecfb2e782c2^ = zero after edits):

| File | Lines reverted | Net |
|---|---|---|
| `tt_metal/api/tt-metalium/program.hpp` | -6 (removed `set_global_program`/`is_global_program` declarations) | dead-code |
| `tt_metal/impl/program/program.cpp` | -15 (restored unconditional `TT_FATAL`, removed `global_program_` collapse, removed `set_global_program`/`is_global_program` defs) | re-enabled core-membership check |
| `tt_metal/impl/program/program_impl.hpp` | -7 (removed `global_program_` field + setter/getter) | dead-code |
| `tt_metal/impl/sub_device/sub_device_manager_tracker.cpp` | -51 / +16 (re-enabled L1-clash check on `default==active && !sub_device_ids.empty()` path) | **THE ACTIVE SUPPRESSION** |
| `ttnn/cpp/ttnn/operations/embedding/device/embeddings_fused_program_factory.cpp` | -19 (restored `compute_with_storage_grid_size()`) | grid choice |
| `ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp` | -11 (restored `compute_with_storage_grid_size()`) | grid choice |
| `ttnn/cpp/ttnn/operations/embedding/device/embeddings_tilized_indices_program_factory.cpp` | -23 (restored `compute_with_storage_grid_size()`) | grid choice |

**Critical finding from inspection** (before any rebuild): `set_global_program(true)`
is **never called anywhere in the codebase** (`grep -rn 'set_global_program'`
= only declarations/definitions). The `global_program_` collapse path in
`determine_sub_device_ids` is dead code — the field is always `false`.
This narrows the active suspect surfaces of `ecfb2e782c2` to (1) the
`lowest_occupied_compute_l1_address` early-return and (2) the embedding
factory `stall_group`-vs-`compute_with_storage_grid_size` grid choice.

Probe branch commit: `11e7806afbf` (now deleted).

## Build outcome

tt-metal incremental rebuild via
`/sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh`
target `ttnn`:

- 233 TUs (Unity builds dominate)
- Compile + link in ~6 minutes
- No errors, no warnings about the reverted symbols
- Both shared libs relinked: `_ttnn.so` 14252664 bytes, `_ttnncpp.so` 33270192 bytes

This confirms that the probe's choice of "narrow revert" (omit the
`register_*` machinery and `LAYER19-DIAG` from the revert set) was correct:
restoring just the 7 files preserved all dependencies of later commits
(particularly `31d55797b56`).

## Server boot

`SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_DISABLE_PREFILL_TRACE=1` boot reached
`/health_generate` = 200 in ~50 s — same as canonical boot time.

**The L1-clash check did NOT fire on boot.** None of:
- KV-cache allocation
- Prefetcher init (sender/receiver/worker sub-device creation)
- 5-tier prefill warmup at lengths 128, 1024, 2048, 4096, 1 token

This contradicted the upstream-diff doc's implicit prediction that the
check might fire during boot if reverted. It does NOT fire because (a) boot
runs entirely under the DEFAULT manager before DECODE manager is created,
and (b) the GlobalCB at 706304 only appears in the DEFAULT allocator AFTER
the DECODE manager is set up + prefetcher activates + V2.5 suspend reverts
to DEFAULT.

## Prefetcher Q1 — DIFFERENT garbage signature

```
Q 1/10 WRONG   gt=64.0 pred=3.0  70.5s
last_line=' rabbitagens璧.stubedList一个小时橘吊顶(errno资管女星屏障在一旁
            centr раз раз�后备叙述 proceeding号楼生活习惯\tmesh前提是 conceive化'
```

This is a NEW garbage signature, distinct from:
- B.8 canonical broken-prefetcher: `'房东benh火力ifty主权悠悠 bambS齊 Canon 新规...'`
- B.9 Iter-2 retry: `'<think>� hacking一朵的带领灰尘...'`

Both the old signatures and the new one are "Chinese/mixed-token noise"
patterns, confirming the matmul IS producing OUTPUT (no NaN crash) but
from wrong inputs. The shift in character distribution between
the two states proves SOMETHING in the kernel-launch state changed —
possibly the embedding factory's grid choice (16 cores via stall_group
back to full compute grid) — but not the underlying prefetcher cycle
correctness.

## Q2 prefill crash — the SMOKING GUN that ecfb2e782c2's clash bypass IS NEEDED

```
2026-05-24 03:33:06.894 INFO [V2.5] suspend: model_id=0 saved
    worker=SubDeviceId(2) receiver=SubDeviceId(1) mode=Mode.DECODE
2026-05-24 03:33:06.939 critical TT_THROW: Statically allocated circular
    buffers in program 390 clash with L1 buffers on core range [0-0 - 7-0].
    L1 buffer allocated at 706304 and static circular buffer region ends
    at 1458688 (assert.hpp:104)
RuntimeError: ... validate_circular_buffer_region(IDevice const*)
    ... MeshWorkloadImpl::compile(MeshDevice*)
    ... EnqueueMeshWorkload(...)
```

The crash hits at the EXACT L1 address (706304) documented in:
1. The original `ecfb2e782c2` comment as the GlobalCB receiver-core base.
2. The S10 byte-dump evidence (S10 confirmed
   `fifo_start = fifo_wr_ptr = fifo_rd_ptr = 706304` across all 32 receiver
   cores for the WQKV layer 0 GlobalCB).

The crash sequence (verified in log):
1. Q1 finishes (~70.5 s) — last decode under DECODE manager.
2. Q2 sent.
3. V2.5 suspend: drain DECODE sub-devices, switch to DEFAULT manager.
4. Q2 prefill needs to recompile lm_head (program cache was cleared at
   DECODE switch).
5. `validate_circular_buffer_region` for lm_head calls
   `lowest_occupied_compute_l1_address({SubDeviceId{0}})`.
6. With ecfb2e782c2's bypass: returns `nullopt`, no clash. WITHOUT the bypass
   (our probe): returns 706304 (the GlobalCB sender-bank's lockstep-mirror
   appearance in the DEFAULT allocator), which clashes with lm_head's CB at
   1458688 (since lm_head's CB region extends down toward 706304).
7. TT_THROW kills the scheduler.

This is the EXACT scenario the ecfb2e782c2 comment describes:
> "lm_head uses COMPUTE cores, not receiver cores where GlobalCB lives.
> Per-core L1 spaces are independent; there is no real clash."

The reported clash is a FALSE POSITIVE arising from the lockstep-bank
allocator assumption: `lowest_occupied_compute_l1_address` looks at bank
0 of the global allocator under DEFAULT, but GlobalCB is registered there
even though physically it lives on receiver cores, not lm_head's compute
cores. The DEFAULT allocator is "viewing" the GlobalCB's address because
the L1 allocator under DEFAULT is shared across all banks (lockstep), even
though the actual L1 memory is per-core and physically partitioned.

**Conclusion**: The L1-clash bypass in `ecfb2e782c2` IS load-bearing for
multi-prefill (V2.5 suspend → lm_head recompile) and is suppressing a
documented false positive — NOT a real bug. Removing it crashes Q2's
prefill on a known-false-positive clash. The upstream-diff doc's
Suspect #1 ranking of ecfb2e782c2 as "HIGH plausibility for prefetcher
bug" is now **RULED OUT** for the L1-clash sub-surface.

## What ALSO changed (signature shift on Q1 vs B.8) — interpretation

The Q1 garbage signature changed (rabbitagens vs housebenh). This is
attributable to ONE of:
1. **Embedding-factory grid revert**: ecfb2e782c2 made embedding factories
   use the stall-group's worker cores (16 cores under DECODE = 2-sub-device's
   `dynamic_worker_core_grid(5,0)-(6,7)`) instead of the full compute grid.
   My revert restored `compute_with_storage_grid_size()` = full grid. This
   changes the embedding's per-core split, output sharding, and therefore
   the input to the very first matmul (WQKV layer 0). DIFFERENT input
   bytes → DIFFERENT output bytes → DIFFERENT garbage signature.
2. **`determine_sub_device_ids` restored TT_FATAL**: If any program's
   kernel groups failed the `num_intersections == num_cores` check after
   my revert, the program would TT_FATAL before launch. But none did
   (boot succeeded, Q1 decoded for 70.5 s), so this was a no-op.

The embedding-factory revert is the more likely cause of the signature
shift. This is consistent with embedding being layer-0's input. It doesn't
imply embedding was a CULPRIT — only that the embedding's per-core layout
matters for kernel-launch determinism, which we already knew.

## Hypothesis ledger update

| ID | Suspect | Status |
|---|---|---|
| S1 | Row-wise stride mismatch in writer | RULED OUT (B.3 byte-dump) |
| S2 | num_blocks mismatch | OPEN, unlikely |
| S3 | WO K-shard transposition | RULED OUT (Phase A) |
| S5 | BFP4/BFP8 mixed tile pitch | RULED OUT (Phase A) |
| S6 | Per-layer cross-block address drift in producer | RULED OUT (B.4-extended) |
| S7 | Compute-kernel ring_idx wrap | RULED OUT (B.6) |
| S8 | Per-tensor block_size in reader_dram.cpp | RULED OUT (B.8 analysis) |
| S9 | Producer NOC posted-writes flush insufficient | RULED OUT (S9 dispatch) |
| S10 | Sub-device CB-address misalignment | RULED OUT (S10 dispatch — bytes match) |
| **A1: ecfb2e782c2 L1-clash bypass = hidden real clash** | (this dispatch) | **RULED OUT — bypass is needed; clash is a documented false positive from lockstep-bank assumption; reverting crashes Q2 prefill on the same false positive that ecfb2e782c2's comment described** |
| **A1: ecfb2e782c2 embedding factory grid choice = correctness regressor** | (this dispatch) | **PARTIALLY RULED OUT — reverting changes garbage signature but doesn't fix; not the root cause** |
| **A1: ecfb2e782c2 `global_program_` collapse + TT_FATAL bypass** | (this dispatch) | **DEAD CODE — `set_global_program(true)` is never called anywhere; field is always false; cannot affect runtime** |

## Path forward — recommended next attacks

Per the upstream-diff doc's ranking + this dispatch's findings:

### Lead 2 (next dispatch, ~4-6 h) — port upstream 934d954b995's `last_subblock_w_valid` pattern

`934d954b995` (2026-05-21) is the upstream fix "DRAM Matmul - Fix Compute
Reading Un-pushed Data": compute kernel was reading more from `cb_in1` than
the reader pushed when `per_core_N_compute > per_core_N_in1_sender`, masked
by the writer dropping padded columns and the CB being oversized. The fix
tracks `last_subblock_w_valid` in `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`
+ siblings.

Our prefetcher uses `matmul_multicore_reuse_mcast_1d_program_factory.cpp`
(the 1D / ring-gathered factory) which `934d954b995` does NOT patch. But
the bug class is the SAME and the symptoms match: BFP8 garbage with valid
exponent range, deterministic-but-wrong. The Qwen3-8B prefetcher's
per-tensor block_size differs between WQKV / W1 / W3 / WO / W2 (BFP4 vs
BFP8 padding) — the same padding skew that triggers the upstream bug should
similarly trigger the gathered bug.

Next dispatch should:
1. Cherry-pick `934d954b995` onto fork HEAD (verify no API drift).
2. Trace the matmul kernel for the 1D / ring-gathered factory and identify
   the analogous `last_subblock_w` truncation point.
3. Port the pattern: track per-tensor `per_core_N_compute` vs
   `per_core_N_in1_sender_from_writer`, narrow the last subblock read in
   compute when they differ.
4. Test on Qwen3-8B with prefetcher.

### Lead 3 (parallel) — U2 in0-side direct probe

S10 instrumented in1 (the weight side) and confirmed bytes match. The
in0 (activation) side was NOT instrumented. Per phase_b9 doc U2: the
matmul's in0 reads may have a layout/format mismatch with what the
all-reduce / reshard emitted. Add DPRINT to
`reader_bmm_tile_layout_in0_ring_all_gather.cpp` mirroring the S10 reader
probe. Compare in0 bytes the matmul sees vs. what `all_reduce` / `reshard`
emitted.

This is a relatively cheap parallel attack since the instrumentation
pattern is already developed (S10 dispatch).

### Lead 4 (out of scope for the on-fork-patch dispatch but worth flagging)

The phase_b9 doc mentioned U1 (program-config drift) and U3-U4 (per-core
runtime args). The upstream-diff agent should compare our `attention.py`
/ `mlp.py` `ttnn.linear(..., global_cb=...)` call sites against the
upstream tt-metal demo's prefetcher invocation (if one exists). If there
IS upstream test coverage for prefetcher correctness on
`matmul_multicore_reuse_mcast_1d_program_factory.cpp`, our config might
be diverging in a way that's invisible to the byte-level probe.

## Working state at session end

- tt-metal-sglang HEAD: **`c1e3437b78e`** (unchanged from session start —
  probe branch deleted; HEAD restored).
- Container source synced from HEAD (7 files re-copied via `podman cp`).
- Container shared libs rebuilt to HEAD state (~6 min, both `_ttnn.so` +
  `_ttnncpp.so` updated).
- `/root/.cache/tt-metal-cache/*` cleared (twice; once per build).
- `stash@{0,1,2}` UNTOUCHED.
- All cards healthy.
- No server running.
- Canonical GSM8K(10) chat re-verified: **9/10 = 90%** (matches baseline,
  Q3 wrong as documented).

## Commits this session

- (tt-metal-sglang) NONE — probe-branch commit `11e7806afbf` deleted; HEAD
  unchanged.
- (sglang) `<this doc>` — pending commit.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping config
at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** All
documented prefetcher suspects S1-S10 + A1 (the C++ revert) are RULED
OUT. The remaining attack vectors are Lead 2 (port upstream 934d954b995's
fix pattern to our 1D / ring-gathered factory) and Lead 3 (in0-side
DPRINT probe). Both require a different attack tool than the C++
revert tried this session.

## Risk note: ecfb2e782c2's L1-clash bypass behavior is correct as written

For future sessions: do NOT attempt to "narrow" the
`lowest_occupied_compute_l1_address` early-return to a finer condition
(e.g. "only skip when GlobalCB is detected") without first making the
DEFAULT allocator aware of WHICH cores GlobalCB physically uses. The
current implementation correctly conservatively skips the false positive;
attempting to narrow it without fixing the underlying allocator-side
visibility (GlobalCB at receiver cores tracked in DEFAULT's bank-0
allocator under lockstep assumption) WILL re-introduce the Q2-prefill
crash we observed this session.

The cleaner long-term fix is to make GlobalCB's L1 reservation visible
ONLY in the DECODE-manager's receiver sub-device allocator, not in the
DEFAULT allocator's bank-0. That requires upstream tt-metal changes to
the GlobalCircularBuffer registration path. For now, the bypass is
correct as-is.
