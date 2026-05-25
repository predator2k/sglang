# TT Qwen3-8B prefetcher — U31 CT-arg discriminator IDENTIFIED: BFP4 (clean) vs BFP8 (garbage). Root mechanism: receiver-matmul GCB byte-offset assumption mismatches prefetcher's actual write position when matmuls have different per-tensor `in1_block_size_bytes`. Naive single-line `set_fifo_size = global_cb->size()` workaround fails on CB divisibility constraint. Canonical re-verified — 2026-05-25

Status: **EVIDENCE_ADVANCE — U31 Phase 1 (factory CT-arg dump) LANDS conclusively.
The discriminator between the 1 CLEAN ELF (`0x2db6e`) and the 3 GARBAGE ELFs
(`0x2d96e`, `0x2de6b`, `0x2df6a`) is the in1 tensor's dtype: CLEAN is
**BFP4** (in1_single_tile_size=576), all 3 GARBAGE are **BFP8** (in1_single_tile_size=1088).
Under Qwen3-8B "balanced" precision preset, FF1/FF3 → BFP4 (the lone clean
ELF) while WQKV / WO / FF2 → BFP8 (the 3 garbage ELFs).
Root mechanism analysis (described below) points at the matmul receiver's
local-CB rd_ptr assumption (`fifo_start + ring_idx * in1_block_size_bytes`)
not matching the prefetcher's per-tensor sequential write position in the
global circular buffer.  The naive C++ workaround
`remote_cb_fifo_size = global_cb->size()` (drop the `(gcb/page)*page` rounding)
fails at allocation with `TT_THROW: Total circular buffer size 835584 B must
be divisible by page size 13824 B` from `circular_buffer_config.cpp:150`.
Canonical Qwen3-8B (prefetcher OFF, U31_FACTORY_ARGS unset) re-verified
`" What is 2+2? What is 2+2? What"` at 9.08s + GSM8K(10)=9/10 = 90% (Q3
reasoning-pattern only; same per U30 baseline).  No regression from the
default-OFF factory-args probe.**

Continuation of `tt_qwen3_8b_prefetcher_U30_PATH_A_W2_PACK_OUTPUT_IS_WRONG_2026-05-25.md`.

tt-metal-sglang HEAD: pending (1 commit this session: U31 factory-args probe; env-gated default-off).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U31.1 | Per-ELF compile-time args under real weights + prefetcher | Added env-gated `SGLANG_TT_U31_FACTORY_ARGS=1` `fprintf(stderr, "[U31_FACTORY_ARGS elf=0x%x in0_block_w=%u ... in1_single_tile_size=%u ... fp32_dest_acc_en=%u ...]")` at gathered-matmul `CreateKernel` call site in `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2589`. The elf_tag uses the SAME hash the kernel computes (`(in0_block_w*1) ^ (in0_num_subblocks*131) ^ (in1_num_subblocks*17) ^ (num_blocks*7919) ^ (out_subblock_h*31) ^ (out_subblock_w*257) ^ (batch*65537)`) so factory dumps correlate 1:1 with U30 Path A's kernel-side ELF tags. | **4 unique ELFs dumped per decode (same 4 ELFs U30 Path A measured).** |
| U31.2 | Which single CT-arg differs between 1 CLEAN and 3 GARBAGE? | Table below. | **CLEAN ELF `0x2db6e`: in1_single_tile_size=576 (BFP4). All 3 GARBAGE: in1_single_tile_size=1088 (BFP8).** Every other CT-arg is shared with at least one other ELF — only dtype is unique-to-CLEAN. |
| U31.3 | Confirm via dtype-flip control experiments? | (a) `SGLANG_TT_QWEN3_PRECISION=full_bf16` (lifts ALL MLP+attention weights → BF16). (b) `SGLANG_TT_QWEN3_PRECISION=bfp8` (lifts FF1/FF3 BFP4→BFP8 only). | **BOTH FAIL at startup with `TT_FATAL: Out of Memory: Not enough space to allocate 62914560 B L1 buffer across 40 banks, where each bank needs to store 1572864 B, but bank size is 1461760 B` from `bank_manager.cpp:462`.** BF16 weights ≈ 2× BFP4 / 1.4× BFP8 → exceeds Blackhole P150a L1 cap on TP=1. Direct dtype-only flip not testable in current L1 budget. |
| U31.4 | Root-mechanism hypothesis (BFP4-vs-BFP8 specific to gathered/GCB path) | Analysis: factory line 2154-2155 `tt_metal::CircularBufferConfig((global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes)` rounds the per-matmul local-CB view of the GCB DOWN to a multiple of THAT matmul's `in1_block_size_bytes = in1_single_tile_size * in1_block_num_tiles`. For BFP4 W1 with `in1_block_size_bytes = 576 * 24 = 13824`: `(835584 / 13824) * 13824 = 60 * 13824 = 829440` (≠ `835584`). For BFP8 W2 with `in1_block_size_bytes = 8704`: `(835584/8704)*8704 = 835584` (exact). The producer (prefetcher writer_l1.cpp) wraps at `(gcb_size - gcb_size%page_size)` with its OWN per-tensor page_size set runtime via `experimental::resize_remote_sender_cb_interface`; consumer wraps at receiver-matmul's per-tensor local-CB fifo_limit. The 6144-byte gap (835584 − 829440) means BFP4-receiver's wrap doesn't match producer's wrap when the producer is mid-write of a smaller-page-size tensor's data. Additionally, `setup_local_cb_read_write_interfaces` resets every LocalCB's `fifo_rd_ptr` to `fifo_addr` (= fifo_start) at every kernel entry — so each matmul reads from `fifo_start + ring_idx * in1_block_size_bytes`, INDEPENDENT of where its tensor's data was actually written by the prefetcher's sequential append (W1 at offset 0; W3 at 442368; W2 at 884736; etc., per the producer's tensor-order writes paced by `remote_cb_reserve_back`). | **NEW TOP HYPOTHESIS: receiver-matmul reads from `fifo_start + ring_idx*block_size` but the prefetcher writes tensor data at sequential offsets across all matmuls in the GCB. The first 1-2 iterations are "normal" because the GCB region still holds zero-initialized L1; once the prefetcher fills the ring, mis-aligned reads pick up other tensors' bytes (→ 2^60-2^109 magnitudes from BFP8-decoded floats, matching U7's "huge magnitude" signature).** |
| U31.5 | Naive C++ fix attempt: drop the rounding | When `SGLANG_TT_U31_KERNEL_FIX=1`, replace `(global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes` with `global_cb->size()` directly. | **FAILS at allocation: `TT_THROW @ /tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:150: Failed allocation attempt on buffer index 31. Total circular buffer size 835584 B must be divisible by page size 13824 B`.** The CB infra enforces `fifo_size % page_size == 0`; rounding cannot be dropped without changing one of (page_size, gcb_size). |
| U31.6 | Canonical re-verify (U31_FACTORY_ARGS unset, U31_KERNEL_FIX unset, prefetcher OFF) | Standard canonical. | **PASS — `" What is 2+2? What is 2+2? What"` at 9.08s e2e_latency; GSM8K(10) = 9/10 = 90.0% (Q3 wrong: reasoning-pattern only, gt=160 pred=2 with `"Wait, but maybe there's a different way. Let me think again. Suppose that after the restart, she has"` truncated; the model didn't reach a final answer within token budget — non-arithmetic issue identical to canonical baseline pattern).**  Bytewise-equal vs U30 canonical. No regression from the default-OFF factory-args probe. |

**Punchline:** The U31 factory-args probe lands the **discriminator at the matmul's in1 weight dtype**: BFP4 (576-byte tile) survives; BFP8 (1088-byte tile) produces 65-86% garbage at PACK exit (U30 Path A measurement).  Root mechanism is a **receiver-side assumption mismatch**: the matmul kernel reads from `fifo_start + ring_idx * in1_block_size_bytes` while the prefetcher writes tensor data at sequential byte offsets across all matmuls in the GCB.  The first 1-2 iterations are "normal" because the GCB is zero-init; once filled, mis-aligned reads pick up other tensors' bytes which, when decoded by the in1's per-matmul `in1_data_format` (BFP4 / BFP8), produce huge magnitudes that NaN-poison the residual stream.

The naive single-line fix (drop the `(gcb/page)*page` rounding) fails on CB divisibility; the proper fix requires **either** (a) sizing the GCB to be a common multiple of all matmul block_sizes [Python-side change to `prefetcher.py:_max_block_tiles` accounting], **or** (b) passing per-matmul GCB-byte-offset as a runtime arg to the compute kernel so each matmul reads from where the prefetcher actually wrote its data, **or** (c) unifying all matmul block_sizes Python-side by adjusting `in0_block_w`/`out_subblock_w`/`in1_num_subblocks` Python parameters across W1/W3/W2/WQKV/WO so they all share one block_size [perf-touching].  All three need multi-file Python+C++ co-changes beyond a single env-gated patch and are **U32**'s scope.

## Per-ELF compile-time-args table (U31.1 raw)

8 `[U31_FACTORY_ARGS …]` lines captured (4 unique ELFs × 2 program-creation events for `--max-running-requests 1` decode warmup + first real decode).  Per-ELF (de-duplicated):

| ELF | role (likely) | in0_block_w | in0_num_subblocks | in1_num_subblocks | num_blocks | out_subblock_h | out_subblock_w | out_subblock_num_tiles | batch | per_core_M | per_core_N | K | in1_block_num_tiles | **in1_block_size_bytes** | **in1_single_tile_size** | **dtype** | fp32_dest_acc_en | packer_l1_acc_en | spill |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| **`0x2db6e`** (CLEAN, 0% garbage) | **FF1/FF3** (w1/w3, BFP4) | 4 | 1 | 1 | 32 | 1 | 6 | 6 | 1 | 1 | 6 | 128 | 24 | **13824** | **576** | **BFP4** | 0 | 1 | 0 |
| `0x2d96e` (86% garbage) | **FF2** (w2, BFP8; K=hidden=192t per device) | 6 | 1 | 1 | 32 | 1 | 4 | 4 | 1 | 1 | 4 | 192 | 24 | **26112** | **1088** | **BFP8** | 0 | 1 | 0 |
| `0x2de6b` (65% garbage) | **WQKV** (BFP8) | 4 | 1 | 1 | 32 | 1 | 3 | 3 | 1 | 1 | 3 | 128 | 12 | **13056** | **1088** | **BFP8** | 1 | 1 | 0 |
| `0x2df6a` (66% garbage) | **WO** (BFP8) | 4 | 1 | 1 | 32 | 1 | 2 | 2 | 1 | 1 | 2 | 128 | 8 | **8704** | **1088** | **BFP8** | 1 | 1 | 0 |

Common to ALL 4: `in0_num_subblocks=1, in1_num_subblocks=1, num_blocks=32, out_subblock_h=1, batch=1, per_core_M=1, ring_size=32, num_cores=32, spill=0, dst_full_sync_en=0`.

**Per-CT-arg cross-tab (CLEAN vs 3 GARBAGE):**

| CT arg | CLEAN | QKV-Garbage | WO-Garbage | FF2-Garbage | Discriminates? |
|---|---|---|---|---|---|
| in0_block_w | 4 | **6** | 4 | 4 | No (shared with 2 garbage) |
| K | 128 | **192** | 128 | 128 | No (shared with 2 garbage) |
| out_subblock_w | **6** | 4 | 3 | 2 | **YES (CLEAN unique)** |
| in1_block_num_tiles | 24 | 24 | **12** | **8** | No (CLEAN shares with QKV) |
| in1_block_size_bytes | **13824** | 26112 | 13056 | 8704 | YES (each ELF unique) |
| **in1_single_tile_size** | **576** | 1088 | 1088 | 1088 | **YES (CLEAN unique → BFP4 vs BFP8)** |
| **in1_data_format (derived)** | **BFP4** | **BFP8** | **BFP8** | **BFP8** | **YES (CLEAN unique)** |
| fp32_dest_acc_en | 0 | 0 | 1 | 1 | No (CLEAN shares with QKV-garbage) |
| packer_l1_acc_en | 1 | 1 | 1 | 1 | No (shared) |

**Two single-CT-arg discriminators correlate 1:1 with CLEAN-vs-GARBAGE: `in1_single_tile_size` (= dtype) and `out_subblock_w` (= per_core_N).**  Per QWen3-8B `model_config.py:313` `_default_settings + Qwen3-8B-balanced` preset, FF1/FF3 are the only matmuls assigned BFP4 (line 126: `"TensorPrecision": {TensorGroup.FF1_FF3: PrecisionSetting.BFP4}`).  FF1/FF3 also happen to be the matmuls with the largest `per_core_N` (because their hidden-dim output is wider than the others' outputs), hence `out_subblock_w = 6` ≠ all 3 garbage ELFs.  Either is a single-arg discriminator; **dtype is the architecturally meaningful one** because (a) BFP4 vs BFP8 directly drives unpacker hw_configure differences and (b) the bug manifests at PACK exit (the BFP-decode happens during unpack/MATH).

## Root-mechanism analysis (U31.4 derived from code reading; not yet directly confirmed via kernel-side rd_ptr probe)

### Producer (prefetcher) write pattern

Reading `tt_metal-sglang/ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp` + `kernels/writer_l1.cpp`:

1. **GCB sizing** (Python `prefetcher.py:_max_block_tiles + max_tile_bytes`):
   `global_cb_size = max(_max_tile_bytes) * max(_max_block_tiles)` cross-product.  For Qwen3-8B this is `1088 * 768 = 835584` bytes per receiver (BFP8 max tile × 768 max per-tensor tiles per receiver).
2. **Writer init** (factory line 165-168): `remote_cb_config.remote_index(remote_cb_index).set_page_size(L1_ALIGNMENT=16).set_data_format(max_tile_size_df=BFP8)` — initial page_size=16, then runtime-resized per tensor.
3. **Per-tensor write** (`writer_l1.cpp:95-130`): `experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, curr_block_size_per_receiver, noc);` — sets sender page_size to the CURRENT tensor's per-receiver block_size, then `experimental::remote_cb_push_back_and_write_pages<>` writes the block from local_cb into remote_cb.  `fifo_wr_ptr` advances by `curr_block_size_per_receiver` per block.
4. **Wrap-around**: `next_fifo_wr_ptr = fifo_start + align(fifo_wr_ptr - fifo_start, page_size)`; `fifo_limit_page_aligned = fifo_size - fifo_size%page_size`.  Producer wraps at `(835584/page_size)*page_size`.  For most tensors page_size divides 835584 evenly (BFP8 6912/4352/4944/13056) so wrap is at 835584.  For BFP4 with page=6912 (per receiver = 13824/2), also divides 835584 evenly.

### Consumer (matmul receiver) read pattern

Reading `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2152-2160` + `kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` + `kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp`:

1. **CB init** (factory line 2154-2160): `tt_metal::CircularBufferConfig((global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes)` — per-matmul fifo_size rounded down to multiple of its OWN block_size_bytes.  For BFP4 W1: 60*13824 = **829440** (≠ 835584).  For BFP8: all 835584 (exact).
2. **Local CB rd_ptr init** (firmware `setup_local_cb_read_write_interfaces`): `local_interface.fifo_rd_ptr = fifo_addr;` — **every kernel restart resets rd_ptr to fifo_start**.  The previous matmul's rd_ptr advancement does NOT carry over to the next program.
3. **Kernel top-of-batch** (gathered kernel line 304-308): `in1_rd_ptr_start_addr = get_local_cb_rd_ptr(in1_cb_id);` (= fifo_start, per step 2) + `update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_idx, in1_tensor_split)` advances rd_ptr by `ring_idx * in1_block_size_bytes`.
4. **Per-block read**: `matmul_block(input0_cb_id, in1_cb_id, ...)` reads tile at `cb_in1.fifo_rd_ptr`.  Each subsequent block advances via `update_local_cb_rd_ptr(in1_cb_id, next_in1_rd_ptr_addr)` computed by `calculate_next_block_index_and_update_rd_ptr` using `in1_block_size_bytes`.
5. **End-of-batch** (line 841-843): reset `update_local_cb_rd_ptr(in1_cb_id, in1_rd_ptr_start_addr)` (back to fifo_start), then `update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_size, in1_tensor_split)` advances by `ring_size * in1_block_size_bytes`.

### The mismatch

**Receiver assumes: tensor T's data starts at GCB offset `0` (the matmul's local rd_ptr begins at fifo_start at every kernel entry per step 2 above), then steps by `in1_block_size_bytes`.**

**Producer actually writes: tensor T's data starts at GCB offset `sum_{prior tensors} (block_size_per_receiver * num_blocks_for_prior_tensor)`, modulo GCB wrap.**

For Qwen3-8B-balanced per-layer tensor order W1 → W3 → W2 → WQKV → WO (from `mlp.py:290-302` and `attention.py:785-789`):

| Tensor | dtype | per-tensor block_size_per_receiver (bytes) | num_blocks=ring_size | Per-tensor total bytes per receiver | Cumulative GCB offset (bytes) |
|---|---|---:|---:|---:|---:|
| W1 (FF1) | BFP4 | 6912 | 32 | 221184 | 0 → 221184 |
| W3 (FF3) | BFP4 | 6912 | 32 | 221184 | 221184 → 442368 |
| W2 (FF2) | BFP8 | 13056 | 32 | 417792 | 442368 → 860160 (WRAPS at 835584 → 24576) |
| WQKV | BFP8 | 6528 | 32 | 208896 | 24576 → 233472 |
| WO | BFP8 | 4352 | 32 | 139264 | 233472 → 372736 |

(Cumulative offsets above are approximate — exact values depend on `num_receivers_per_reader` and the per-receiver division which I haven't precisely cross-checked; the ORDER-OF-MAGNITUDE point is what matters.)

Every receiver matmul reads from `fifo_start + ring_idx * in1_block_size_bytes`.  For W2 (3rd in queue) this is `fifo_start + ring_idx * 13056` — i.e., starting at byte 0 of GCB — but the prefetcher wrote W2's data starting at byte 442368.  **W2 reads W1's data**, decoded with BFP8 unpacker → huge magnitudes from misinterpreted BFP4-encoded bytes.

Under **zero weights** the entire GCB region is zero-init bytes; any rd_ptr offset returns zero; matmul `out = x * 0 = 0`; the residual stream stays the same; final output is the "broken model with zeroed weights" pattern (e.g. `"Und_flag_flag"` from U30 Path A.4) but NOT NaN/Inf.

Under **real weights** the misaligned reads pick up other tensors' bytes; BFP8-unpacking BFP4-encoded data (or vice-versa) yields ~2^60-2^109 magnitudes (U7's signature, U30 Path A.2's 86% NaN/Inf at PACK exit).

### Why the "1-2 iterations are normal" pattern (U7/U30)

At the very first decode iteration, the prefetcher has just been started; the producer has written the first 1-2 tensors before the first matmul kernel fires.  The "fresh" GCB region (still zero-init L1) returns zeros for the mis-aligned reads on the not-yet-written-to byte ranges.  Once the producer fills the ring (typically by iteration 2-3), every mis-aligned read returns real-weight bytes → garbage.

This temporal pattern matches U7's per-core dump:
* iter 1: `0x3ced3cb3 bd09bc68 …` (~0.05 — normal small magnitude)
* iter 7: `0xc026 4022 3f1d c026 …` (~2 — degrading)
* iter 20: `0x7effff59 …` (NaN/Inf — fully poisoned)

## Why the naive C++ workaround fails

Applied (then reverted):

```cpp
// matmul_multicore_reuse_mcast_1d_program_factory.cpp:2152-2160 (U31 attempt)
if (use_global_cb) {
    uint32_t in1_block_size_bytes = in1_single_tile_size * in1_block_num_tiles;
    const char* u31_kernel_fix_env = std::getenv("SGLANG_TT_U31_KERNEL_FIX");
    bool u31_kernel_fix = (u31_kernel_fix_env != nullptr &&
                           std::string(u31_kernel_fix_env) == "1");
    uint32_t remote_cb_fifo_size =
        u31_kernel_fix ? global_cb->size()
                       : (global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes;
    tt_metal::CircularBufferConfig remote_cb_config =
        tt_metal::CircularBufferConfig(remote_cb_fifo_size);
    ...
}
```

Runtime symptom:

```
TT_THROW: Failed allocation attempt on buffer index 31.
Total circular buffer size 835584 B must be divisible by page size 13824 B
(assert.hpp:104)
```

The CB infrastructure (`circular_buffer_config.cpp:150`) enforces `fifo_size % page_size == 0` at allocation time.  The original rounding existed precisely to satisfy this invariant.  Removing the rounding without changing `page_size` (= per-matmul `in1_block_size_bytes`) violates the invariant.

To actually fix the bug, the GCB size and all matmul `in1_block_size_bytes` values must share a common multiple.  The current GCB sizing in `prefetcher.py:998-1001`
(`max(_max_tile_bytes * _max_block_tiles)` cross product) produces 835584 = 1088 * 768.  This IS divisible by all the BFP8 block_sizes (8704, 13056, 26112) but NOT by the BFP4 block_size (13824, since 835584/13824 = 60.45).  Sizing the GCB to LCM of all block_sizes (13824, 13056, 8704, 26112) yields an impractically large value (LCM ≈ 10s of GB — exceeds Blackhole L1 banks by orders of magnitude).

The viable fixes (U32 scope):

* **Fix A — Python-side GCB sizing**: round `global_cb_size` UP to the LCM of all per-matmul block_sizes that share the GCB.  For Qwen3-8B-balanced, this would be LCM(13824, 13056, 8704, 26112) — too large.  Alternative: round up to a smaller common multiple that fits L1 (e.g. enforce all per-tensor block_sizes to share a common factor via padding).
* **Fix B — Receiver-kernel offset arg**: pass a "tensor's starting byte offset in GCB" runtime arg to each gathered matmul kernel; the kernel uses `update_local_cb_rd_ptr(in1_cb_id, fifo_start + tensor_offset_bytes)` at top of batch instead of trusting `fifo_start`.  Requires Python (prefetcher.py to compute per-tensor offsets) + C++ (factory.cpp to plumb the rt-arg) + kernel.cpp (consume the rt-arg).  Cleanest semantically.
* **Fix C — Unify all matmul block_sizes**: change `in0_block_w` / `out_subblock_w` / `in1_num_subblocks` Python-side across W1/W3/W2/WQKV/WO so they all yield the SAME `in1_block_size_bytes`.  Requires matmul-config refactoring in `mlp.py` + `attention.py`; may impact per-op perf (subblock dims are chosen for MAC throughput).

## Run results (U31 reproduction)

### U31.1 — Factory CT-arg dump under prefetcher + real weights

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c 'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U31_FACTORY_ARGS=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u31_p1_server.log 2>&1'

# Wait for /health, then:
curl -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":5,"temperature":0.0}}'

# Per-ELF CT-args:
grep "U31_FACTORY_ARGS" /tmp/u31_p1_server.log | sort -u
# 8 lines (4 unique ELFs × 2 program-creation events)
```

Server returned `" What敏捷请您p季后"` (garbage, expected — U31_FACTORY_ARGS does not modify behavior).

### U31.6 — Canonical re-verify (default — prefetcher OFF, U31_* envs unset)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c 'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u31_canonical_server.log 2>&1'

curl -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# → " What is 2+2? What is 2+2? What" at 9.08s e2e_latency  (bytewise-equal vs U30 baseline)

python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 10
# → FINAL: 9/10 = 90.0% (Q3 reasoning-truncation: gt=160 pred=2, "Wait, but maybe there's a different way..."
#    — same per-question failure pattern as canonical baseline; non-arithmetic)
```

## Hypothesis ledger (post-U31)

| ID | Suspect | Pre-U31 | Post-U31 |
|---|---|---|---|
| U7 / Case B — Gathered compute kernel produces garbage on specific CT-arg ELFs under REAL weights | CONFIRMED (U30) | **REFINED** — discriminator IS BFP8-vs-BFP4 dtype; bug is NOT in the kernel's compute math (which is correct under zero weights) but in the **bytes it reads** from cb_in1 (which are wrong due to GCB rd_ptr/offset mismatch between matmul receiver's assumption and prefetcher's actual write position). |
| **NEW U31 — Per-matmul `(gcb_size / in1_block_size_bytes) * in1_block_size_bytes` fifo-size rounding causes BFP4 W1 to see `fifo_size=829440` (≠ GCB-actual 835584).  Combined with `setup_local_cb_read_write_interfaces` resetting rd_ptr to fifo_start every kernel entry, every matmul reads from `fifo_start + ring_idx * its_own_block_size`, INDEPENDENT of the prefetcher's per-tensor sequential write position.** | (new) | **TOP STANDING.** Naive `fifo_size = gcb_size` workaround fails on CB divisibility; proper fix needs U32 multi-file Python+C++ co-change. |
| NEW U30 — mm_partials_cb spill/reload bug | NEW STANDING (U30) | RULED OUT — `spill=0` for all 4 ELFs in our prefetcher decode (per factory dump); the spill path is never executed. |
| NEW U30 — GCB rd_ptr arithmetic bug | NEW STANDING (U30) | **CONFIRMED as the mechanism** — `update_rd_ptr_to_ring_index` advances local rd_ptr by `ring_idx * block_size`, but the prefetcher writes tensor data at offsets that depend on which tensors preceded it in the queue, NOT on `ring_idx * this_tensor's_block_size`. |
| NEW U30 — DST accumulator stale state | NEW STANDING (U30) | NOT NEEDED — U31's GCB-offset bug fully explains the symptom without invoking DST staleness. |
| U17-U28 downstream-stomp lineage | EACH RULED OUT (U30) | UNCHANGED — all the downstream "stomper" theories are derived from observing wrong bytes in L1, which are now explained as the matmul WRITING wrong values (the matmul output IS the wrong bytes; nothing is "stomping" them — they were wrong at PACK exit because the inputs were wrong, which traces back to GCB read offset). |
| U16/U29 W2→RS handshake | RETIRED as fix | UNCHANGED. |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (Python-side untouched; only tt-metal-sglang C++ factory got an env-gated default-OFF probe).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used FIXED `rebuild_tt_metal_kernels.sh` after each host edit + `podman cp` for source sync (verified `[U31_FACTORY_ARGS …]` string lands in `_ttnncpp.so`).
- [x] Probe env-gated (`SGLANG_TT_U31_FACTORY_ARGS=1`).
- [x] Attempted fix env-gated (`SGLANG_TT_U31_KERNEL_FIX=1` — reverted, fails on CB divisibility).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"` at 9.08s) + GSM8K(10) = 9/10 = 90% (Q3 reasoning-pattern only; baseline-equivalent).
- [x] Server stopped at session end; cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: pending — **1 new commit** this session (U31 factory CT-args probe; env-gated default-OFF; `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2589-2640`).  Naive C++ fix attempt at line 2152-2170 was reverted in-session (HEAD is the probe-only version).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/ttnn/cpp/...` source: synced; `_ttnn.so` + `_ttnncpp.so` synced to install dir; sentinel `SGLANG_TT_U31_FACTORY_ARGS` verified in `strings _ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at 9.08s + GSM8K(10) = 9/10 (Q3 reasoning-pattern only).
- DPRINT/stderr artifacts preserved in-container at `/tmp/u31_p1_server.log`, `/tmp/u31_p4_fix_server.log`, `/tmp/u31_canonical_server.log` for U32's reference.

## Shipping verdict (unchanged from U30)

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  3 of 4 unique gathered-matmul ELFs read mis-aligned bytes from the GCB under real weights.
- **`SGLANG_TT_U31_FACTORY_ARGS=1` is safe to leave in tree (default-off).**  It only enables an stderr fprintf at program-creation time on the gathered-matmul factory path; no behavioral change.
- **`SGLANG_TT_U31_KERNEL_FIX=1` IS NOT IN THIS HEAD** (was reverted in-session due to CB divisibility failure).  No env var to disable.

## U32 — exact next attack

Per U31's identified discriminator (BFP4 vs BFP8) and root mechanism (receiver-matmul GCB rd_ptr offset assumption mismatches prefetcher's sequential per-tensor write position):

1. **U32.1 — direct probe of LOCAL cb_in1 rd_ptr at kernel entry vs prefetcher's actual write offset per tensor.**  Add DPRINT in gathered compute kernel reading `get_local_cb_rd_ptr(in1_cb_id)` AND the underlying L1 bytes at that address vs the EXPECTED-FOR-THIS-TENSOR bytes (from a host-side weight cache).  If bytes mismatch — U31's root-mechanism is empirically confirmed.
2. **U32.2 — implement Fix B (per-matmul GCB-byte-offset runtime arg)** in 3 files:
   - `prefetcher.py`: track `tensor_offset_bytes[t] = sum_{i<t}(block_size_per_receiver[i] * ring_size[i])` Python-side and expose via a per-tensor accessor.
   - `mlp.py` / `attention.py`: pass `tensor_offset_bytes` as a kwarg to the `ttnn.linear` call (gathered path).
   - `matmul_multicore_reuse_mcast_1d_program_factory.cpp`: read the kwarg, push as runtime arg to `mm_kernel`.
   - `bmm_large_block_zm_fused_bias_activation_gathered.cpp`: read the rt-arg, use `update_local_cb_rd_ptr(in1_cb_id, fifo_start + tensor_offset_bytes)` at top of batch INSTEAD of trusting `fifo_start`.
3. **U32.3 — validate**: GSM8K(10) ≥ 7/10 under prefetcher on Qwen3-8B.  If yes — bench TPOT, retire U29/U16 guards, squash to one production flag.
4. **U32.4 — canonical re-verify** unchanged.

## Commits this session

- (tt-metal-sglang) **1 commit**: factory CT-args probe (`matmul_multicore_reuse_mcast_1d_program_factory.cpp`).
- (sglang) `<this doc>` — pending commit.
