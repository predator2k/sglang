# TT Qwen3-8B prefetcher — Path A (B.4-extended layer probe + compute-kernel dump) — 2026-05-23

Status: **EVIDENCE_ADVANCE — S6 (per-layer cross-block address drift in
producer) RULED OUT. Byte transport from producer to receiver verified
correct for layers {0, 1, 17, 35} of the 36-layer cycle under FULL
prefetcher (5 tensors). Compute-kernel rd_ptr reads land on valid BFP8
exponent bytes (not garbage) once the L1-alignment-unit shift is applied.
Canonical (no prefetcher) Qwen3-8B re-verified 9/10 = 90% on GSM8K(10)
chat at session end. Tt-metal-sglang HEAD unchanged at
`b81f15001ca`; working tree clean.**

Continuation of `tt_qwen3_8b_prefetcher_phase_b67_2026-05-23.md`
(Path A track). Executes Step B.4 (extended byte-dump for layers >0)
AND Step B.5 (compute-kernel UNPACK-side instrumentation) of the
phase_b67 next-session plan in a single dispatch.

## Bottom line

| Test | What it probes | Result |
|---|---|---|
| **B.4-extended** | producer→receiver byte transport for L ∈ {0,1,17,35} of 36-layer cycle | **Bytes match for ALL 4 probed layers — S6 RULED OUT** |
| **B.5** | compute-kernel `update_rd_ptr_to_ring_index` post-rotation bytes the matmul actually consumes | **Reads land on valid BFP8 exponent bytes (0x79..0x7c range) — no garbage. Rotation math arithmetically correct.** |
| Canonical regression | GSM8K(10) chat post-revert | **9/10 = 90%** (unchanged baseline) |

## Setup

* tt-metal-sglang HEAD: `b81f15001ca` (Phase B.7 zero-tensor guard).
* Path A is C++-only (matmul reader_bmm_tile_layout, matmul
  compute bmm_large_block_zm_fused_bias_activation_gathered,
  prefetcher writer_l1). Per dispatch scope-rule:
  Path A does not touch `models/tt_transformers/tt/*.py` (Path B
  surface) or `python/sglang/**`.
* Stash@{0} (Phase B.3 byte-dump scaffold) applied with `git stash
  apply` (NOT pop). Preserved.

## Instrumentation summary

Extended Phase B.3's `stash@{0}` scaffold with:

1. **Producer (`writer_l1.cpp`)** — gate dump on `layer ∈ {0,1,17,35} &&
   t == 0 && block ∈ {0,1,15,31}`. Tag each line with `L{n} T{m} B{k}`.
2. **Matmul reader (`reader_bmm_tile_layout_in1_ring_all_gather.cpp`)**
   — gate dumps to a single receiver core
   `(get_absolute_logical_x()==1 && get_absolute_logical_y()==1)`,
   add per-launch ordinal (static within kernel ELF) so the dump
   sequence can be correlated with producer's layer/tensor cycle.
   Existing dumps preserved (in1_rd@, +1088, +2176, +row_sz).
3. **Matmul compute
   (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`)** — NEW
   per-launch dumps gated to same single core. Uses **`DPRINT_UNPACK(DPRINT
   << ... << ENDL());`** per `tt_metal/hw/inc/api/debug/dprint.h:50`
   (the broken `UNPACK((DPRINT << ...))` form from phase_b67 was
   NOT used). Dumps:
   * pre-rotation `ring_idx`, `cb_start`, `rd_ptr_pre` (both aligned-unit
     and byte forms)
   * post-rotation `rd_ptr_post` (both aligned-unit and byte forms)
   * first 16 bytes at `rd_ptr_post << 4` (byte-form pointer) — i.e.
     the actual bytes the matmul will unpack
   * end-of-tensor `rd_ptr_for_next_tensor` (where the NEXT layer's
     matmul launch will start)

### Key gotcha (fixed during dispatch)

LocalCBInterface's `fifo_rd_ptr` is stored in **L1-alignment units**,
not bytes. The byte address is `fifo_rd_ptr << cb_addr_shift` (=`<< 4`
on Blackhole). Cf. `tt_metal/hw/inc/api/debug/dprint_tile.h:17`:
```c
#define CB_RD_PTR(id) (get_local_cb_interface(id).fifo_rd_ptr << cb_addr_shift)
```
The first iteration of the compute-kernel dump used `rd_ptr_post`
directly as a byte pointer and read what looked like RISC-V opcode
bytes (`fb010113 5212223 ...`) — actually L1 instruction memory at
the unshifted low offset (e.g. 62912). Fixed by adding `<< 4` for
all reads through `LocalCBInterface` rd_ptr values.

### DPRINT volume gating (lesson)

First iteration without core-gating produced 838K lines and **hung
the server in the prefill warmup** (host print server backpressure).
Single-core (`x==1 && y==1`) gating cut volume to ~34K lines (still
no successful decode warmup — see below), but the cross-comparison
goal was achieved during the kernel-compile and trace-capture phases
(which both run all 36 layers × all 5 tensors). For pure correctness
probing this is sufficient; for ALSO running a successful decode we
would need to gate further (e.g. only dump for ord ∈ {0,1,17,35}, or
disable producer-side dumps entirely, or limit
`TT_METAL_DPRINT_CORES` to the single core instead of `all`).

## Layer-by-layer dump comparison table (decisive S6 evidence)

WQKV ring matmul: `h=4 w=3 row_sz=3264` (BFP8, 12 tiles/block).
Producer core `0:0-1:BR` is paired with receiver core `0:1-1:BR`
(per Phase B.3 producer-receiver mapping). Reading the first uint32
at producer's `local_cb_addr` and at receiver's `in1_rd@` for the
WQKV matmul, ring_idx=4 (= receiver `0:1-1`'s assigned per-core ring
index for this op):

| Layer | Producer src@ (cycles 3 slots) | Producer first 16 bytes | Receiver MM_R in1_rd@ (advances per cycle) | Receiver first 16 bytes | Match |
|---|---|---|---|---|---|
| **L0**  | `src@706304`  | `7a7a7a79 79797979 7a7a7979 7a7a7979` | `in1_rd@706304`  | `7a7a7a79 79797979 7a7a7979 7a7a7979` | **✓** |
| **L1**  | `src@810752`  | `7a7b7a7a 7a7a7b7a 7a797a7a 7a7a7a7a` | `in1_rd@1489664` | `7a7b7a7a 7a7a7b7a 7a797a7a 7a7a7a7a` | **✓** |
| **L17** | `src@915200`  | `7b7b7b7a 7a7b7c7b 7b7b7b7b 7b7b7a7b` | `in1_rd@1332992` | `7b7b7b7a 7a7b7c7b 7b7b7b7b 7b7b7a7b` | **✓** |
| **L35** | `src@915200`  | `7a7a7a7b 7a7a7a7a 7a7a7a7a 7a7a7a7a` | `in1_rd@1097984` | `7a7a7a7b 7a7a7a7a 7a7a7a7a 7a7a7a7a` | **✓** |

Producer slot rotates through `{706304, 810752, 915200}` per the
triple-buffer (`total_num_blocks_in_buffer = 3` in
`dram_prefetcher_program_factory.cpp:126`) with delta = 104448 bytes
between slots. Layer N starts at slot `(2N) mod 3` due to
`num_blocks=32` and `32 mod 3 = 2` advance per layer.

Receiver in1_rd@ advances per-cycle by 78336 bytes (= 6 ×
block_size_bytes) because the matmul's local-CB view aligned to the
remote c_31 advances by the BLOCK size each launch, but the cycle
also touches the OTHER 4 tensors between consecutive WQKV launches
(WO 8-tile, W1 24-tile BFP4, W3 24-tile BFP4, W2 24-tile BFP8), so the
WQKV-only inter-launch delta is a function of all 5 tensors' block
sizes summed minus WQKV's own.

**Conclusion: producer→receiver byte transport is correct for all 4
probed layers under the FULL 5-tensor prefetcher cycle.** Phase B.3
established this at L0 under WQKV-only SKIP config (since-invalidated
by Phase B.7 reroute-bug finding); Path A re-establishes it
**without the SKIP confound** for 4 spaced layers, and for the
load-bearing FULL config.

## Compute-kernel byte-read evidence (decisive S7-supplement)

For 0:1-1:TR0 (compute), at ring_idx=4, post-rotation `rd_ptr_byte`
typically lands at `cb_start_byte + 3264` (= ring_idx × block_size_bytes
/ L1_alignment × L1_alignment = 4 × 13056/16 × 16 = 13056 bytes after
cb_start; or in aligned units 47408 = 44144 + 3264 = 706304/16 +
4×13056/16). Bytes read at that address:

```
0:1-1:TR0: MM_C ord=0 ring_idx=4 ... rd_ptr_pre_byte@706304
0:1-1:TR0: MM_C ord=0 rd_ptr_post_aligned=47408 rd_ptr_post_byte@758528
0:1-1:TR0: MM_C ord=0 bytes@rd_ptr_byte [0..3]=7a797a79 797a7a7a 79797a7a 79797979
```

The bytes are in the BFP8-exponent range (0x79..0x7c), matching the
expected layout for the first 64-byte exponent prefix of a BFP8 tile.
No garbage. No NaN. The pre-rotation `rd_ptr_pre_byte@706304` matches
the receiver's `in1_rd@706304` AND matches the producer's `src@706304`
— all three agree.

`update_rd_ptr_to_ring_index`'s arithmetic is correct: it advances the
rd_ptr by exactly `ring_idx × in1_block_size_bytes / L1_ALIGNMENT`
aligned units (= `ring_idx × in1_block_size_bytes` bytes after
`<< 4`), which is the position of block N out of num_blocks blocks
in the c_31 receiver FIFO. Phase B.6's surgical bypass (return early
from this function) showed the bug isn't here. Path A's
instrumentation confirms by direct observation that the post-rotation
rd_ptr lands on valid weight bytes.

## Updated suspect ranking

| ID | Suspect | Status |
|---|---|---|
| S1 | Row-wise stride mismatch in writer (block-0 level) | RULED OUT (B.3) |
| S2 | num_blocks mismatch | UNCHANGED — open but unlikely |
| S3 | WO K-shard transposition | RULED OUT (A) |
| S5 | BFP4 vs BFP8 tile-pitch mix | RULED OUT (A) |
| **S6** | **Per-layer cross-block address drift in producer** | **RULED OUT (this dispatch, B.4-extended)** |
| **S7** | **Compute-kernel `update_rd_ptr_to_ring_index` wrap math** | **RULED OUT (B.6 + this dispatch's B.5 direct observation)** |
| **S8** | **Tile-metadata / data_format mismatch** | **OPEN — strongest standing suspect** |
| **S9** | **Producer NOC posted-writes flush insufficient** | **OPEN** |
| **S10** | **Sub-device CB-address misalignment** | **OPEN** |
| NEW | SKIP_* matmul reroute config (separate bug from B.7) | OPEN |

## What this dispatch did NOT do

* Did not land a fix. With S6+S7 ruled out, the remaining suspects
  (S8/S9/S10) need different probes than per-layer byte comparison.
* Did not get GSM8K(10) under prefetcher ON. The DPRINT-instrumented
  server hung at first prefill warmup due to print-server backpressure
  even with single-core gating (the dump fires once per matmul launch
  × ~180 launches × 2 cores produces enough lines that the prefill
  cycle, which has its own decode warmup, stalls). All correctness
  signals were extracted from the kernel-compile + trace-capture
  phases of the boot, which DO run all 36 layers × all 5 tensors
  with the full prefetcher dataflow.
* Did not investigate the SKIP_* reroute bug (separate per Phase B.7).

## Path-A next attack: tighter probes for S8 / S9 / S10

### S8 — tile-metadata / data_format mismatch (highest leverage)

The prefetcher GlobalCB is sized with `max_tile_size_df` = BFP8 (1088 B)
in `dram_prefetcher_program_factory.cpp:103-105`. The matmul remote_cb
for W1/W3 is configured with `in1_data_format = BFP4` (576 B) in
`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2155`. The producer
WRITES `curr_block_size_per_receiver = curr_block_num_tiles ×
curr_single_tile_sizes / num_receivers` BYTES per receiver per block
(writer_l1.cpp:48); for BFP4 this is `48 × 576 / 4 = 6912 B`.

But the producer's c_0 reader (reader_dram.cpp:95) advances
`l1_write_addr += max_block_size` between blocks where `max_block_size
= max_tile_size × max_block_tiles = 1088 × 48 = 52224 B`. For BFP4
blocks, actual data is `48 × 576 = 27648 B`, leaving a 24576-byte
GAP filled with whatever was there before.

If the writer's reads use the OLD layout assumption (reading 13056 B
per receiver from c_0, but the BFP4 block only occupies 6912 B per
receiver), the writer's bytes 6913..13056 are STALE/GARBAGE. These
get shipped to c_31 alongside the valid BFP4 bytes.

**Concrete next probe**: add a producer-side DPRINT comparing
`curr_block_size_per_receiver` (what writer thinks each receiver gets)
vs the actual reader-DRAM-written size at that address. If they
diverge for BFP4 tensors, S8 is confirmed; the fix is in
`writer_l1.cpp` to use per-tensor sizing (matching reader_dram), or
to fix reader_dram to not skip ahead.

### S9 — producer NOC posted-writes flush

writer_l1.cpp:118 has `noc_async_posted_writes_flushed()` after
`remote_cb_push_back_and_write_pages`. The matmul consumer's
`remote_cb_wait_front(remote_cb_id, num_blocks)` is a semaphore wait
that should serialize after all producer pushes. If the semaphore
increment is NOT properly fenced after the actual data writes, the
matmul may read before bytes land.

**Concrete next probe**: replace `noc_async_posted_writes_flushed()`
with the stronger `noc_async_writes_flushed()` (which fences on
unposted writes too) and rerun GSM8K. If it passes, S9 is confirmed.

### S10 — sub-device CB address misalignment

The matmul uses `sub_device_id = receiver_sub_device_id`. The remote
c_31 CB is allocated in the GlobalCB region. If the matmul's
`align_local_cbs_to_remote_cb` math computes a different L1 base address
than the prefetcher writer's NOC-target address, bytes go to the wrong
place. Phase 1 audit already showed the LOGICAL addresses align (both
@ 706304); Path A confirms the bytes there match. So S10 is unlikely
unless there's a NOC-virtual vs L1-physical translation bug.

**Concrete next probe**: dump `get_remote_cb_interface(c_31)`'s
fifo_start_addr from BOTH the writer's perspective and the matmul
reader's perspective. If they differ, S10 is real.

## Working state at end of session

* tt-metal-sglang HEAD: `b81f15001ca` (unchanged from B.7 session).
* C++ kernel tree pristine (all Path A instrumentation reverted via
  `git checkout HEAD -- ...`). Working tree clean.
* Container's `/tt-metal/ttnn/cpp/.../{compute,dataflow}/...` matches
  pristine (sync'd via `podman cp` after revert).
* Kernel cache cleared at `/root/.cache/tt-metal-cache/`.
* `stash@{0,1,2}` untouched (per CLAUDE.md no-stash-manipulation rule;
  `apply` only, never `pop` or `drop`).
* DPRINT artifact preserved at host `/tmp/path_a_dprint_v2_artifact.log`
  (3 MB, 34630 lines).
* Canonical GSM8K(10) chat re-verified at session end: **9/10 = 90%**
  with `--context-length 4096 --max-new 3000`.
* All cards healthy.
* No server running.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B until S8/S9/S10
probing identifies the data-format / synchronization / sub-device
mismatch and a localized correction lands.

## Commits this session

* (sglang) `<this doc>` — pending.
* (tt-metal-sglang) NO commits. Working tree reverted to pristine
  HEAD per Path A scope-rule (no SGLang core changes; instrumentation
  is throwaway). `stash@{0..2}` preserved for next session.
