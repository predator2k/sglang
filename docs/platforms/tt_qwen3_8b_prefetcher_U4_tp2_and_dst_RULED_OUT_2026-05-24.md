# TT Qwen3-8B prefetcher — U4: TP=2 + DST-accumulator hypotheses RULED OUT — 2026-05-24

Status: **EVIDENCE_ADVANCE — U4-A (TP=2 misconfiguration) is architecturally
impossible: SGLang's TT platform layer rejects `--tp-size 2` at startup
because the TT-internal TP=2 lives inside the ttnn mesh device and is
hidden from the SGLang scheduler. U4-B (DST-accumulator stale-state in
the gathered compute kernel) is ruled out by code analysis: the same
`mm_block_init` → `tile_regs_acquire` → `matmul_block` →
`tile_regs_commit/release` lifecycle is used by both the gathered
prefetcher kernel and the non-gathered canonical-path kernel, both
backed by the LLK section-rotation mechanism that gives each
`tile_regs_acquire` a fresh DST section. Canonical works at GSM8K(10) =
9/10 with the same primitives, so cross-launch DST staleness cannot be
the prefetcher-specific bug. Working tree pristine; HEAD unchanged at
`22417cb1d2d`; canonical not re-verified this dispatch (no
prefetcher-side changes that could regress it).**

Continuation of `tt_qwen3_8b_prefetcher_U1_cross_subdev_sync_2026-05-23.md`,
`tt_qwen3_8b_prefetcher_phase_b8_2026-05-23.md`, and
`tt_qwen3_8b_prefetcher_path_a_b4ext_2026-05-23.md`. The dispatch
attempted two simultaneous new hypotheses after 14 prior dispatches.
Both ruled out without running the proposed remediation.

## Bottom line

| Hypothesis | Test | Result |
|---|---|---|
| **U4-A**: `--tp-size 1` is a misconfiguration; the prefetcher was
designed for galaxy/T3000 multi-device and breaks at single-device TP | Launch server with `--tp-size 2` | **FAIL AT STARTUP — `platform.py:153` raises `ValueError: TT backend requires --tp 1 (TT-internal TP=2 is hidden)`.** SGLang scheduler sees TP=1 by design; tt-metal mesh_device handles the actual 2-Blackhole TP internally. There is no Python surface where `--tp-size 2` could ever boot, so the hypothesis is **architecturally vacuous**. |
| **U4-B**: gathered compute kernel's DST accumulator holds stale values across program launches; bug = missing reset | Read `bmm_large_block_zm_fused_bias_activation_gathered.cpp` + cross-reference `bmm_large_block_zm_fused_bias_activation.cpp` + LLK `reg_api.h` + `matmul.h` | **RULED OUT by code analysis (this dispatch, no probe needed).** Both kernels use `mm_block_init` (which calls `llk_pack_dest_init` resetting DST section), `tile_regs_acquire`/`tile_regs_commit`/`tile_regs_release` (LLK section rotation gives each acquire a fresh DST section), and `matmul_block` (hardware FPU zeroes DST on first call of a sequence). The canonical path uses the non-gathered kernel and works at GSM8K(10) = 9/10 — if cross-launch DST staleness were the bug, canonical would also fail. The DST mechanism is shared and known-good. |

## U4-A — Code-level refutation

`/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/platform.py:140-157`:

```python
# TP visibility: SGLang sees tp_size = 1 in P1. The "TP=2" mesh is
# entirely inside ttnn's mesh_device — invisible to SGLang. Users
# invoke with --tp 1 (or omit, defaulting to 1). If they explicitly
# pass --tp 2, raise at startup rather than silently override.
if server_args.tp_size not in (None, 1):
    raise ValueError(
        f"TT backend requires --tp 1 (TT-internal TP=2 is hidden); "
        f"got --tp {server_args.tp_size}."
    )
server_args.tp_size = 1
```

The 2× Blackhole P150a hardware already runs TP=2 via the
tt-metal mesh device (`mesh_device.get_num_devices() == 2`). The
`--tp-size 1` value at the SGLang layer means "one logical scheduler
worker," not "one Tenstorrent device." The prefetcher's
`generate_mux_safe_sender_receiver_mapping` is already configured
for the 2-device mesh; the matmul `ShardTensor2dMesh(dims=(0,1),
mesh_shape=(1,2))` distributes weights across both Blackholes.

**Captured failure log** (`/tmp/u4a_tp2_test.log`):

```
File "/sglang/python/sglang/srt/hardware_backend/tenstorrent/platform.py",
  line 153, in apply_server_args_defaults
    raise ValueError(
ValueError: TT backend requires --tp 1 (TT-internal TP=2 is hidden);
  got --tp 2.
```

No fix path exists at the SGLang Python surface for U4-A: removing
the guard would not enable the prefetcher to work differently — it
would just let SGLang try to spawn TWO scheduler workers per
TT-internal device, which the rest of the codebase is not wired
to support (`tp_worker.py:238` etc.).

## U4-B — Code-level refutation

The hypothesis: the matmul's `tile_regs_acquire` does not zero DST,
so `matmul_block(... dst_index=0, ...)` (with semantics "DST += C")
adds the first compute onto stale DST contents from a previous
program launch / previous matmul block. Over 36 layers, drift
explodes to ~10²⁰.

### Refutation point 1 — DST section rotation

`/home/mhnie/tt-metal-sglang/tt_metal/hw/inc/api/compute/reg_api.h`:

```cpp
ALWI void tile_regs_acquire() {
    MATH((llk_math_wait_for_dest_available()));
}
ALWI void tile_regs_commit() {
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
}
ALWI void tile_regs_release() {
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}
```

`llk_math_wait_for_dest_available` blocks MATH until the next DST
section is free (PACK has signaled `dest_section_done`). It returns
a fresh, unused DST section — not the same physical tiles as the
prior subblock. The `_section_done` calls advance the section
pointer; section sizing (typically 8 tiles in half-sync) lets MATH
and PACK pipeline two sections at a time.

So **each `tile_regs_acquire` returns a CLEAN section** that has
just been packed-and-released by the prior iteration. The
`matmul_block(... dst_index=0, ...)` writes into THIS section's
tile 0, which was last touched by the matmul HW op (which on
first-tile semantics emits a write, not an add). There's no
cross-section stale data.

### Refutation point 2 — `mm_block_init` resets pack state per launch

`/home/mhnie/tt-metal-sglang/tt_metal/hw/inc/api/compute/matmul.h:233-271`
(`mm_block_init`):

```cpp
PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out_cb_id)));
PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));   // <-- resets DST init state
PACK((llk_pack_init<false, false>(out_cb_id)));
```

This is called once per gathered-kernel `kernel_main()` invocation
(`bmm_large_block_zm_fused_bias_activation_gathered.cpp:256`). The
`llk_pack_dest_init` resets pack-side DST tracking; combined with
section rotation, the FIRST `tile_regs_acquire` of the launch gets
a known-clean section.

### Refutation point 3 — non-gathered kernel uses the IDENTICAL pattern

`/home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
uses the same `mm_block_init` + `tile_regs_acquire` + `matmul_block`
+ `tile_regs_commit`/`tile_regs_release` lifecycle. The two kernels
differ ONLY in:

| Aspect | Gathered (prefetcher) | Non-gathered (canonical) |
|---|---|---|
| `ENABLE_GLOBAL_CB` compile-time gate | Defined (sets up ring_idx, in1_cb_start_addr math) | Not defined |
| `in1_cb` reader semantics | Reads from c_31 receiver CB (filled by `ttnn.dram_prefetcher`) | Reads from in1_cb filled by reader_in1.cpp |
| `update_rd_ptr_to_ring_index` post-rotation per launch | Yes | No |
| `mm_out_cb_ids` indexed by `b` (batch dim) | Yes | Single mm_out_cb_id |

**None of these differences touch DST register lifecycle.** The
gathered kernel's spill path (lines 425-444) has the EXACT same
`tile_regs_acquire`/`tile_regs_commit`/`tile_regs_wait`/`tile_regs_release`
sequence as the non-gathered version (lines 300-397). The last_out
path (lines 385-423) similarly mirrors the non-gathered last_out
(lines 342-375).

### Refutation point 4 — canonical works

GSM8K(10) chat = **9/10 = 90%** for canonical Qwen3-8B (no
prefetcher), verified at end of every prior session including
U1, B.8, Path A. The canonical path is exercised on EVERY layer's
WQKV, WO, FF1, FF3, FF2 matmuls — all using the same DST mechanism.
If DST cross-launch staleness were a thing, this would be visibly
broken. It isn't.

### Refutation point 5 — DPRINT probe deferred (analysis sufficient)

Per Path A's lessons-learned: in-kernel DPRINT in
`bmm_large_block_zm_fused_bias_activation_gathered.cpp` is fragile.
First iteration produced 838K lines and hung the host print server.
Gating to `(x==1 && y==1)` cut volume to ~34K but the cross-comparison
goal was extracted only during kernel-compile and trace-capture phases
(decode warmup still hung).

Given the four code-level refutations above, a fresh DPRINT iteration
on DST would consume hours of build-and-run time without changing the
verdict: **the DST mechanism is the SAME mechanism the canonical path
uses successfully**. The bug must be elsewhere.

## Why the U1 result is still the strongest standing hypothesis

After this dispatch's two refutations, the **only standing hypothesis
with a code-cited root cause and a concrete fix path** remains:

**U1 — cross-sub-device dispatch serialization gap**
(`tt_metal/impl/program/dispatch.cpp:431` `insert_stall_cmds`): the
matmul kernel runs on `receiver_sub_device`, the immediately-
following all_reduce / CCL op runs on `worker_sub_device`. Each
program emits `WAIT_STREAM` only on its OWN sub-device's dispatch
stream in the CACHED dispatch path. The UNCACHED path emits a
`BARRIER` (global sync), which is why the FIRST decode forward sometimes
looks reasonable, then mode-collapses once the dispatch transitions
to cached. Bytes-on-device for in0+in1 are verified correct (Path A
B.4-ext + Lead 3), so the bug must be that the all_reduce reads
stale matmul output L1 bytes before the matmul finishes writing.

The recommended fix (per U1 doc) is one of:
- Attack 1 — galaxy-style 2-subdev layout + dummy receivers in
  GlobalCB (4-8h Python rework)
- Attack 2 — device-side cross-sub-device GlobalSemaphore barrier
  (2-4h C++ work)
- Attack 3 — extend `insert_stall_cmds` to accept a
  `wait_sub_device_ids` kwarg on all_reduce/CCL (1-2h C++ + API
  change)

These are all out of U4's scope (U4 scope explicitly forbade SGLang
Python and tt_transformers/tt/ edits). The next session should pick
ONE of those attacks and implement it.

## What was done in this dispatch

1. Read U1, B.8, Path A, correctness docs in full to confirm
   understanding of prior state.
2. Issued the U4-A test command (`--tp-size 2`) and captured the
   startup `ValueError` from `platform.py:153` — confirms U4-A
   architecturally impossible.
3. Read `bmm_large_block_zm_fused_bias_activation_gathered.cpp` end-
   to-end (487 lines).
4. Read `bmm_large_block_zm_fused_bias_activation.cpp` (non-gathered
   canonical kernel) for differential.
5. Read `matmul.h` (mm_init, mm_block_init, matmul_block, matmul_tiles).
6. Read `reg_api.h` (tile_regs_acquire/commit/wait/release, the DST
   section rotation primitive).
7. Verified factory point of call: `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2409`
   selects the gathered kernel only when GlobalCB is active (i.e.,
   prefetcher path). Canonical uses the non-gathered kernel via the
   same factory at lines 743, 1652, 4461.
8. Verified `git log --oneline tenstorrent-p1 ^upstream/main` includes
   NO local edits to either matmul kernel — both are upstream-pristine.

## What was NOT done

- Did NOT run a fresh DPRINT probe on DST. Code-level refutation
  is decisive without it (DST mechanism is shared with the working
  canonical path).
- Did NOT modify any kernel or Python file.
- Did NOT regress canonical: no edits made.
- Did NOT progress past U4 to the U1 fix attempts (out of scope per
  U4 plan; the next session should pick from U1 Attack 1/2/3).

## Updated hypothesis ledger

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
| Lead 3 (U2) | in0 ring-all-gather byte transport | RULED OUT (in0 bytes correct, D0/D1 match) |
| **U1** | **Cross-sub-device dispatch sync gap** | **ROOT CAUSE LOCATED; fix paths identified (Attack 1/2/3); NOT YET IMPLEMENTED** |
| U3 | Permuted vs contiguous DRAM grid | RULED OUT |
| **U4-A** | **TP=1 misconfiguration** | **RULED OUT (this dispatch — architecturally impossible at SGLang surface)** |
| **U4-B** | **Compute-kernel DST-accumulator uninitialized** | **RULED OUT (this dispatch — DST mechanism shared with working canonical path)** |

## Working state at session end

- tt-metal-sglang HEAD: **`22417cb1d2d`** (unchanged from session start)
- sglang HEAD: pending this doc + commit message
- Working trees: pristine on both repos
- `stash@{0,1,2}`: untouched (NEVER popped per CLAUDE.md rule)
- No server running
- Cache not cleared (no kernel rebuild this session)
- TT cards healthy (no probe rebuild/run)

## Recommended next attack

Pick **U1 Attack 2** (device-side GlobalSemaphore barrier between
receiver_sub_device matmul and worker_sub_device all_reduce). Reasoning:

- Attack 1 (galaxy 2-subdev + dummy receivers): biggest scope (~4-8h),
  requires understanding galaxy's dummy-receiver enumeration logic
  for our Blackhole MUX-clamped grid.
- Attack 2 (GlobalSemaphore barrier): smallest behavior-preserving
  surface (~2-4h C++ work in tt-metal-sglang). The existing pattern at
  `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_program_factory.cpp:36-44`
  shows how barrier_semaphores are created — apply the same pattern
  between the gathered matmul's final `noc_async_writes_flushed` and
  the all_reduce's first kernel launch.
- Attack 3 (extend `insert_stall_cmds` with `wait_sub_device_ids`):
  cleanest fix conceptually but requires modifying a load-bearing
  dispatch primitive AND every CCL op's public signature. Higher
  risk of upstream-incompat surprises.

If Attack 2 lands a working fix (GSM8K(10) ≥ 7/10 prefetcher),
canonical baseline still 9/10, then the 1.66× TPOT win is shippable.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** Root cause
LOCATED (U1: cross-sub-device dispatch sync gap); fix paths identified
but not yet implemented. U4 ruled out two new hypotheses (TP=1
misconfig + compute-kernel DST stale-state); U1 remains the strongest
standing hypothesis with a concrete code citation and three actionable
fix attacks.

## Commits this session

- (tt-metal-sglang) **NONE** — no kernel modifications this dispatch.
- (sglang) `<this doc>` — pending commit.
