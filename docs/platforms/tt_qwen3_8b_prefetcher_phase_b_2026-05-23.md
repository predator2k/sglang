# TT Qwen3-8B prefetcher — Phase B (SKIP matrix + C++ byte-dump) — 2026-05-23

Status: **PARTIAL — Phase B.1 SKIP matrix DONE (decisive: bug is
intra-tensor, not cross-tensor); Phase B.3 byte-dump DONE (decisive:
producer→consumer byte transport is CORRECT; bug is elsewhere — likely
matmul-config / synchronization / per-block-K transition. NO fix
landed. Canonical shipping path UNCHANGED, GSM8K(10) re-verified
10/10 = 100% post-revert.**

Continuation of `tt_qwen3_8b_prefetcher_phase_a_2026-05-23.md`.

## Phase B.1 — SKIP matrix completion (decisive intra-tensor probe)

Goal: discriminate single-tensor vs cross-tensor bug.

| # | Config (active prefetched weights) | GSM8K(10) | Q1 garbage pattern |
|---|---|---|---|
| 1 | `SKIP_W2=1` only (WQKV+WO+W1+W3) | **0/10** | "观摩 thankfully心意风尚风尚 unveiled Nast..." |
| 2 | `SKIP_W1=W3=W2=1` (attn-only, WQKV+WO) | **0/10** | "<think>률晚上']=$从而晚上...uranuran..." |
| 3 | `SKIP_WO=W1=W3=W2=1` (single-tensor, WQKV ONLY) | **0/10** | "erle尋狭窄状态下两级ricing世界各国..." |
| 4 | `SKIP_WQKV=W1=W3=W2=1` (single-tensor, WO ONLY) | **0/10** | "发展格局 enrollmentaatReuse世代..." |

Every prefetcher config trips
`RuntimeError: probability tensor contains either inf, nan or
element < 0` and the scheduler exits within Q1-Q2.

**Decisive verdict**: Bug is **intra-tensor**. Even a SINGLE prefetched
tensor (1 layer cycle × 1 tensor × 32 blocks) reproduces deterministic
corruption with a unique signature per tensor. Each tensor's distinct
shape (WQKV: K=4096 N=3072; WO: K=2048 N=4096; W1/W3 BFP4 different;
W2 BFP8 different) yields a distinct garbage signature.

This rules out S6 (cross-tensor cursor drift) as a sufficient
explanation: single-tensor cycles have no cross-tensor transitions, yet
still fail. The bug must live in the producer/consumer pipeline of a
SINGLE tensor's blocks, or in the matmul's interpretation of bytes
in c_31.

## Phase B.3 — Byte-dump (decisive byte-transport probe)

Approach: Add DPRINT to both producer (`writer_l1.cpp`) and consumer
(matmul `reader_bmm_tile_layout_in1_ring_all_gather.cpp`) at the
load-bearing point. Use WQKV-only config (Config 3 above) so producer
cycle has exactly ONE tensor per layer.

### Instrumentation (now reverted; kept in git stash@{0})

- **Producer** dumps for layer 0 tensor 0 blocks {0, 1, 15, 31}:
  - First 4 uint32s at `local_cb_addr` (= start of producer's c_0 read pointer for that block)
  - First 4 uint32s at `local_cb_addr + 1088` (= start of tile 1, after tile 0's exponent prefix + mantissa)
  - First 4 uint32s at `local_cb_addr + coal_ps (3264)` (= 3-tile-row boundary, where receiver 0's row 1 starts)
  - First 4 uint32s at `local_cb_addr + next_block_row_stride (13056)` (= cross-receiver boundary)
  - Plus printf-of (coal_ps, coal_np, bh_tiles, btiles, tile_sz, blk_sz_per_rx, num_rx, nblk).

- **Consumer** dumps for batch iter b=0:
  - First 4 uint32s at `cb_in1.get_read_ptr()` (= start of in1 view into c_31)
  - First 4 uint32s at +1088, +2176 (= tile 1, tile 2 starts in row 0)
  - First 4 uint32s at +row_stride (= row 1 start)
  - Plus printf-of (in1_block_h, in1_block_w, num_blocks, tile_sz).

Knobs:
- `TT_METAL_DPRINT_CORES=all`
- `TT_METAL_DPRINT_FILE=/tmp/dprint_wqkv.log`
- `TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1`

Kernel cache must be cleared (`rm -rf /root/.cache/tt-metal-cache/*`)
before each DPRINT iteration — tt-metal aggressively reuses compiled
kernels by source hash, and DPRINT-injected kernels do not always
recompile cleanly without a clean cache.

### Parameters extracted from DPRINT (WQKV, Qwen3-8B, 2× P150a):

| Param | Value | Meaning |
|---|---|---|
| `num_layers` | 36 | Qwen3-8B decoder layers |
| `num_tensors` | 1 | (with WQKV-only SKIP config) |
| `num_blocks` | 32 | = ring_size = 8 senders × 4 receivers per sender |
| `num_receivers` | 4 | receivers per sender |
| `tile_size_bytes` | 1088 | BFP8 tile (64 exp + 1024 mantissa) |
| `block_num_tiles` | 48 | 48 tiles per producer-block per sender |
| `block_height_in_tiles` | 4 | 4 tile rows per block |
| `coalesced_page_size` | 3264 | 3 tiles wide × 1088 bytes (per-receiver chunk per row) |
| `coalesced_num_pages` | 1 | one coalesced page covers a receiver's width |
| `block_size_per_receiver` | 13056 | 4 rows × 3 tiles × 1088 bytes |
| Consumer `in1_block_h` | 4 | matches producer `bh_tiles` ✓ |
| Consumer `in1_block_w` | 3 | matches producer's per-receiver tile-width (12 ÷ 4) ✓ |

All geometry is **consistent** between producer and consumer.
`btiles * tile_sz / num_rx = 48 × 1088 / 4 = 13056 = blk_sz_per_rx` ✓.
`bh_tiles × coal_np × coal_ps = 4 × 1 × 3264 = 13056` ✓.

### Byte-level comparison (decisive finding)

Producer-receiver core pairing (matched by `_make_mapping`):

| Producer core | Receiver core | Producer src@1124096 [0..3] | Consumer in1_rd@1124096 [0..3] | Match |
|---|---|---|---|---|
| `1:7-0:BR` | `1:8-0:BR` | `7a7a7a7a 7a7a7a7a 7a7a7a79 797a797a` | `7a7a7a7a 7a7a7a7a 7a7a7a79 797a797a` | **✓** |
| `0:7-0:BR` | `0:8-0:BR` | `7a7a7a7a 7a7a7b7a 7a7a7a7a 7a7a7a7a` | `7a7a7a7a 7a7a7b7a 7a7a7a7a 7a7a7a7a` | **✓** |
| `0:7-2:BR` | `0:8-2:BR` | `7a7a797a 797a7a7b 7a79797a 7a7a7a7a` | `7a7a797a 797a7a7b 7a79797a 7a7a7a7a` | **✓** |
| `1:0-1:BR` | `1:1-1:BR` | `7b7a7a7a 7a7a7b7a 7b7b7a7a 7a7b7a7a` | `7b7a7a7a 7a7a7b7a 7b7b7a7a 7a7b7a7a` | **✓** |
| `0:0-1:BR` | `0:1-1:BR` | `7a7a7a79 79797979 7a7a7979 7a7a7979` | `7a7a7a79 79797979 7a7a7979 7a7a7979` | **✓** |

**Bytes match at all probed offsets (0, +1088, +2176, +row_stride).**
Same for producer-blocks 1, 15, 31. The producer's local-CB c_0 triple-
buffers (`total_num_blocks_in_buffer = 3`), so addresses cycle through
`{1124096, 1176320, ?}` — but the receiver's c_31 is configured
with `set_page_size(in1_block_size_bytes)` and writes contiguously.

The bytes themselves look like BFP8 exponent bytes (all in `0x79..0x7c`
range, typical for trained-weight FP exponents around 2^-7 magnitudes),
which is the expected layout for the first 64 bytes of a BFP8 tile.

**Verdict: the byte transport is CORRECT.** Bytes at the producer's
push source land at the receiver's c_31 in the right address with the
right values for block 0 of layer 0. The Phase 1-audit S1 hypothesis
(row-wise stride mismatch in writer) is **RULED OUT** at the first-block
level.

### Why bytes match but matmul still produces garbage

Strong candidates remaining (NOT investigated this session):

1. **S6.r1 — Per-layer cross-block address drift** in
   producer. Producer's local-CB c_0 wraps every 3 blocks
   (`total_num_blocks_in_buffer = 3`). Block 0, 3, 6, … all read from
   the same c_0 address. The mapping from a c_0 read-ptr to the
   correct DRAM page must be perfect. If the reader miscomputes the
   DRAM offset for block N>0, the producer's c_0 contents are
   wrong but the writer happily ships them to the receiver — which
   sees consistent (but wrong) bytes.

2. **S7 — Matmul-config / compute-kernel `update_rd_ptr_to_ring_index`
   wrap arithmetic**. The compute kernel ENABLE_GLOBAL_CB path does:

   ```cpp
   UNPACK((update_rd_ptr_to_ring_index(
       in1_cb_id, in1_block_size_bytes, ring_size, in1_tensor_split)));
   ```

   at end-of-batch (line 482 in `bmm_large_block_zm_fused_bias_activation_gathered.cpp`).
   If `in1_tensor_split` is misdetected or the wrap math is off by
   one tile, every subsequent layer's WQKV read reads from offset
   `±tile_sz` of the intended position.

3. **S8 — Tile metadata mismatch**. `set_data_format(in1_data_format)`
   on both the prefetcher remote_cb (program_factory line 165, with
   `max_tile_size_df` = data format of the LARGEST tile) AND the
   matmul remote_cb (program_factory line 2155, with `in1_data_format`
   = the matmul's expected format). When prefetcher has BFP8+BFP4 mix,
   `max_tile_size_df = BF8` but a BFP4-using matmul expects BFP4. The
   single-tensor case removes this drift — but BFP4 W1+W3 still
   reproduces, which suggests S8 alone isn't the issue.

4. **S9 — Producer-side `noc_async_posted_writes_flushed` is
   insufficient for the matmul's compute-kernel-side cb_in1
   `get_read_ptr` semantics**. The producer uses POSTED writes
   (`skip_ptr_update=true`) for perf. If the matmul reads c_31 before
   the producer's writes have completed, it sees stale data. But
   `remote_cb_wait_front` should block on the semaphore count
   incremented by the writer.

5. **S10 — Sub-device / sub-grid CB-address misalignment**. The
   prefetcher uses 3 sub-devices (worker, receiver, prefetcher); each
   sub-device has its own L1 partition and CB-address space. If the
   matmul's c_31 is configured on a sub-device whose CB-address differs
   from where the producer thinks it's writing, bytes land at the
   wrong physical L1 location. This is hard to validate without
   instrumenting `get_remote_cb_interface`'s NOC address lookups.

## What this session did NOT do

- Did not fix the bug. Phase B.3 byte-dump consumed ~1 h after
  Phase B.1 (~45 min), leaving ~2 h remaining out of the 8 h cap.
  Phases C+D (fix + validate) need a localized hypothesis to attack —
  the byte-dump ruled out the strongest standing suspect (S1) but did
  not surface a new one.
- Did not extend the byte-dump to LATER LAYERS / LATER TENSORS in the
  cycle. The single-tensor probe only covered layer 0 tensor 0 of a
  36-layer × 1-tensor cycle. The bug may surface at layer N>0 or at
  tensor boundaries.
- Did not instrument the **compute kernel's actual UNPACK_TILE** path
  to see what bytes the matmul **actually consumes** after the
  `update_rd_ptr_to_ring_index` rotation. The reader-side DPRINT
  shows what bytes are in c_31 BEFORE the compute kernel's rotation;
  if the rotation is off-by-N, the compute kernel reads different
  bytes than the reader-side DPRINT shows.

## Phase B.3+ recommended attack (next session)

### B.4 — Extended byte-dump: later layers and tensor transitions

Repeat Phase B.3 instrumentation with the following extensions:

1. Dump producer for layer N ∈ {0, 1, 17, 35} (= start, mid, end of
   36-layer cycle) tensor 0 block 0.
2. Dump consumer at matching points (need to compute when the matmul
   for layer N runs — count `b` iterations and matmul launches).
3. With FULL 5-tensor config: dump for tensor transitions (e.g. end
   of WQKV block 31 → start of WO block 0 in same layer).

If layer-0 bytes match but layer-N bytes don't, the producer's
DRAM-offset arithmetic for block N is wrong (S6.r1).

### B.5 — Instrument the compute kernel directly

Add DPRINT in `bmm_large_block_zm_fused_bias_activation_gathered.cpp`
after `update_rd_ptr_to_ring_index` to dump the ACTUAL rd_ptr the
compute kernel uses for tile reads. Compare to the producer's
fifo_wr_ptr advance.

### B.6 — Bypass the wrap arithmetic

Hack-patch the compute kernel to set `ring_idx=0` always (bypassing
`update_rd_ptr_to_ring_index`'s rotation). If the corruption persists,
the wrap math is not the bug. If the corruption resolves AND the
matmul becomes correct for layer 0 only (because layers 1+ would now
read the wrong block), we've localized the bug to the wrap math.

### B.7 — Validate with a non-prefetcher ring matmul

The SKIP_* knobs use `prefetch=False, num_global_cb_receivers=1`. This
configures the matmul with ring topology but tells it NOT to read from
GlobalCB. Verify this config gives 9-10/10 GSM8K on its own (without
ANY prefetcher running). If it DOES, the matmul ring config is fine.
If it doesn't, the matmul ring config itself has a bug independent of
the prefetcher.

## Working state at end of session

- tt-metal-sglang HEAD: `af126cf5e7b` (unchanged from Phase A — no C++ landed; byte-dump in `stash@{0}`).
- sglang HEAD: `936b14097` + `<this Phase B doc>` (commit pending).
- Canonical baseline: GSM8K(10) chat = **10/10** (verified 2026-05-23).
- All cards healthy.
- Cache cleared.
- No server running.

## Shipping verdict (unchanged from Phase A)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B until B.4-B.7
identify the layer-N or wrap-arithmetic mismatch and a localized
correction lands in
`tt-metal-sglang/ttnn/cpp/ttnn/operations/{prefetcher,matmul}/`.

## Commits

- (this session, sglang) `docs(tt-prefetcher): Phase B — SKIP matrix decisive + byte-dump rules out S1 (2026-05-23)`
- tt-metal-sglang stash@{0}: byte-dump WIP (not committed; revertable by
  `git stash drop stash@{0}` or pop into a new probing branch).
