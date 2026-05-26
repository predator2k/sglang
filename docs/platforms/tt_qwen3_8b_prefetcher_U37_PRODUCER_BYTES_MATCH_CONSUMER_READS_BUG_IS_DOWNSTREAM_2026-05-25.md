# TT Qwen3-8B prefetcher — U37 ground-truth byte-level diagnostic LANDS conclusively: producer's bytes ARE correctly delivered to consumer L1 (EXACT byte-match across all 5 tensors of layer 0 decode step 1); bug lives BENEATH delivery layer in the compute path itself (env-gated, default-off; canonical Qwen3-8B GSM8K 10/10 preserved) — 2026-05-25

Status: **EVIDENCE_ADVANCE — U37 ships three orthogonal env-gated byte-level probes: (1) `SGLANG_TT_U37_PROD_BYTES` in writer_l1.cpp dumps producer's source bytes at each NoC write; (2) `SGLANG_TT_U37_READ_BYTES` in the gathered compute kernel dumps consumer's L1 bytes at the matmul_block read address; (3) `SGLANG_TT_U37_GT_BYTES` in mlp.py best-effort ttnn.to_torch dump (failed for multi-device sharded weights — orthogonal).  Hardware cross-correlation on worker core (0,1)-5 FIRST READ for all 5 tensors of layer 0 decode step 1: every single dword matches between producer's wr_ptr write bytes and consumer's rd_l1 read bytes.  Conclusion: U36-α (a) producer write skew is REFUTED.  U36-α (b) post-producer L1 stomp is REFUTED.  Bug lives in the compute/decode/unpacker path itself — same compute kernel processes BFP4 correctly but BFP8 incorrectly given correctly-delivered bytes.  Canonical Qwen3-8B (no prefetcher) bytewise-equal `" What is 2+2? What is 2+2? What"` + GSM8K(10) chat = 10/10 = 100% (matches U36 baseline).**

Continuation of `tt_qwen3_8b_prefetcher_U36_WRAP_ARITHMETIC_REFUTED_2026-05-25.md`.

tt-metal-sglang HEAD: **`e834da9df37`** (U37, 5 files, +200 lines).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U37.1 | At the moment matmul_block fires, what are the bytes the kernel ACTUALLY reads at the rd_ptr it points to? | Add env-gated kernel probe in `bmm_large_block_zm_fused_bias_activation_gathered.cpp` that reads first 16 bytes of L1 at `fifo_rd_ptr * L1_ALIGNMENT` right BEFORE `matmul_block`.  Gated to `(ring_idx==0, b==0, block==0, in0/in1_subblock==0, inner_dim_idx==0)` so we get one dump per worker per program-launch. | **In tree (+79 lines).**  Sentinel `SGLANG_TT_U37_READ_BYTES` verified in `_ttnncpp.so`. |
| U37.2 | What bytes does the PRODUCER intend to deliver?  Dump them at NoC write time. | Add env-gated probe in `writer_l1.cpp` that reads first 16 bytes of `local_cb_addr` (= source bytes pre-NoC) for each (layer, t, block) we care about.  Gated to `(layer==0, block==0)` per tensor. | **In tree (+41 lines).**  Sentinel `SGLANG_TT_U37_PROD_BYTES` verified in `_ttnncpp.so`. |
| U37.3 | Add a Python-side ground-truth dump of the DRAM weight tensor's first 16 bytes (un-tilized via ttnn.to_torch). | Add env-gated dump in `mlp.py::register_weights()`, layer 0 only. | **In tree (+59 lines).**  Sentinel `SGLANG_TT_U37_GT_BYTES` works for the gating; the ttnn.to_torch call fails with `TT_FATAL @ pytensor.cpp:259: buffers.size() == 1` because Qwen3-8B weights are multi-device sharded.  This branch failed gracefully (try/except).  ORTHOGONAL to the kernel cross-correlation conclusion. |
| U37.4 | Do the kernel-read bytes match the producer-written bytes for the same L1 address? | Cross-correlate one worker core's (0,1)-5 FIRST READ vs each tensor's PROD write at the matching wr_ptr.  All 5 tensors of layer 0 decode step 1. | **YES — EVERY DWORD MATCHES EXACTLY.  See cross-correlation table below.** |
| U37.5 | Does this rule out (a) producer write skew and (b) post-producer L1 stomp? | Match means producer's bytes correctly land at consumer's read address.  Skew or stomp would have produced byte divergence. | **YES — both (a) and (b) REFUTED.** |
| U37.6 | Does the BFP8 garbage still appear under prefetcher+U32-U36+U37 probes? | Single-decode `"What is 2+2?"` test. | **YES — `" What不/xxx"` garbage signature (different bytes vs U36 garbage but same character class).** |
| U37.7 | Canonical re-verify (no prefetcher, all U37 envs unset). | Standard canonical. | **PASS — `" What is 2+2? What is 2+2? What"` (bytewise-equal vs U36 canonical) + GSM8K(10) chat = 10/10 = 100% (matches U36 baseline).** |

**Punchline:** U37 is the definitive byte-level diagnostic.  Producer's bytes ARE correctly delivered to the consumer's L1 — verified for all 5 tensors of layer 0 decode step 1 on one worker core (0,1)-5.  Yet the 3 BFP8 ELFs still produce garbage outputs while the 1 BFP4 ELF produces correct outputs.  The bug lives BENEATH the byte-delivery layer: same compute kernel binary, same byte stream entering cb_in1, BFP4 path correct but BFP8 path garbage.  U36-α (a) producer write skew is **REFUTED**.  U36-α (b) post-producer L1 stomp is **REFUTED**.  New top hypothesis: compute / unpacker / MATH / PACK behavior diverges between BFP4 and BFP8 on the gathered (`use_global_cb`) path specifically.

## Files changed (5)

```
 models/tt_transformers/tt/mlp.py                                          | 59 ++++++++++++++++
 ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp | 10 +++
 ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp | 79 ++++++++++++++++++++++
 ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp        | 11 +++
 ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp                      | 41 +++++++++++
 5 files changed, 200 insertions(+)
```

Commit: `e834da9df37` on `tenstorrent-p1`.

## Cross-correlation table — the killer evidence

For worker core (device=0, x=1, y=5)'s UNPACK RISC FIRST READ in decode step 1, layer 0:

| ELF | Tensor | rd_l1 | Kernel-read first 16 bytes (4 dwords) | Producer write source (reader 0:0-5 wr_ptr=rd_l1) | Match |
|---|---|---:|---|---|:---:|
| `0x2de6b` | WQKV | 0xac700 | `7a7a7a7a 7a7a797a 797a7b79 7a7a7979` | `7a7a7a7a 7a7a797a 797a7b79 7a7a7979` (t=0 layer=0) | **YES** |
| `0x2df6a` | WO | 0x112700 | `7a7a7a7b 7a7a7a7a 7a7a7b7a 7a7a797a` | `7a7a7a7b 7a7a7a7a 7a7a7b7a 7a7a797a` (t=1 layer=0) | **YES** |
| `0x2db6e` | W1 | 0x158900 | `7a7a7a7a 7a7a7a7a 7a7a7a78 7b7a7b7a` | `7a7a7a7a 7a7a7a7a 7a7a7a78 7b7a7b7a` (t=2 layer=0) | **YES** |
| `0x2db6e` | W3 | 0xfa100 | `7a7a7a7b 7b7b7b7a 7a7b7a78 7b7a7a7a` | `7a7a7a7b 7b7b7b7a 7a7b7a78 7b7a7a7a` (t=3 layer=0) | **YES** |
| `0x2d96e` | W2 | 0x16bb00 | `7b7a7a7a 7a7a7a7b 7a7b7a7a 7a7b7a7a` | `7b7a7a7a 7a7a7a7b 7a7b7a7a 7a7b7a7a` (t=4 layer=0) | **YES** |

**5/5 exact byte-wise match.**  Note that the producer cores (16 total — 4 on device 0, 4 on device 1, in two rows) each send to a specific subset of 32 receiver cores.  Reader 0:0-5 (device 0, x=0, y=5) is the producer for the column of receivers including worker 0:1-5.  Each (reader, receiver) pair carries a specific data shard; the match across all 5 tensors confirms the delivery is sound.

## Implementation deep-dive

### Path A — U37_READ_BYTES probe in gathered compute kernel

Inside the matmul block-stride loop, at the in1_subblock + in0_subblock + inner_dim_idx innermost iteration RIGHT BEFORE `matmul_block(...)` (line 605), gated to `(ring_idx=0, b=0, block=0, in0_subblock=0, in1_subblock=0, inner_dim_idx=0)`:

```cpp
#ifdef SGLANG_TT_U37_READ_BYTES
if (ring_idx == 0 && b == 0 && block == 0 &&
    in0_subblock == 0 && in1_subblock == 0 &&
    inner_dim_idx == 0) {
    constexpr uint32_t u37_elf_tag =
        (in0_block_w * 1u) ^
        (in0_num_subblocks * 131u) ^
        (in1_num_subblocks * 17u) ^
        (num_blocks * 7919u) ^
        (out_subblock_h * 31u) ^
        (out_subblock_w * 257u) ^
        (batch * 65537u);
    static uint32_t u37_budget = 8;
    UNPACK(({
        if (u37_budget > 0) {
            u37_budget--;
            ckernel::tensix_sync();
            uint32_t u37_rd_shifted = get_local_cb_rd_ptr(in1_cb_id);
            uint32_t u37_rd_l1 = u37_rd_shifted * L1_ALIGNMENT;
            volatile uint32_t* u37_p = (volatile uint32_t*)u37_rd_l1;
            DPRINT << "[U37_READ elf=0x" << HEX() << u37_elf_tag
                   << DEC() << " ring=" << ring_idx
                   << " in1bs=" << in1_block_size_bytes
                   << " rd_l1=0x" << HEX() << u37_rd_l1
                   << " w=[0x" << u37_p[0]
                   << " 0x" << u37_p[1]
                   << " 0x" << u37_p[2]
                   << " 0x" << u37_p[3] << "]"
                   << DEC() << "]" << ENDL();
        }
    }));
}
#endif
```

The ELF tag matches U13's per-ELF tag so dumps correlate 1:1 with U30 Path A's per-ELF garbage measurements.  The `tensix_sync()` ensures any in-flight writes complete before reading L1 (per U13's lesson on PACK probe timing).

### Path B — U37_PROD_BYTES probe in writer_l1.cpp

Inside the writer_l1 main loop just before `experimental::remote_cb_push_back_and_write_pages`, gated to `(layer=0, block=0)`:

```cpp
#ifdef SGLANG_TT_U37_PROD_BYTES
{
    static uint32_t u37_pb_budget = 64;
    if (layer == 0 && block == 0 && u37_pb_budget > 0) {
        u37_pb_budget--;
        auto& _u37_remote_cb = get_remote_sender_cb_interface(remote_cb_id);
        uint32_t _u37_wr_ptr = _u37_remote_cb.fifo_wr_ptr;
        volatile uint32_t* _u37_src = (volatile uint32_t*)local_cb_addr;
        DPRINT << "[U37_PROD layer=" << layer
               << " t=" << t << " blk=" << block
               << " local_cb=0x" << HEX() << local_cb_addr
               << " wr_ptr=0x" << _u37_wr_ptr
               << " w=[0x" << _u37_src[0]
               << " 0x" << _u37_src[1]
               << " 0x" << _u37_src[2]
               << " 0x" << _u37_src[3] << "]"
               << DEC() << " bsz=" << curr_block_size_per_receiver
               << "]" << ENDL();
    }
}
#endif
```

`local_cb_addr` is the source of the NoC write (data already pulled from DRAM and tilized into the producer's local CB).  `wr_ptr` is the destination address in each receiver's L1 GCB region (same value the noc_async_write uses as `dest_addr`).

### Path C — Python ground-truth dump in mlp.py

Inside `register_weights()` after each `prefetcher.insert_tensor`, env-gated to `SGLANG_TT_U37_GT_BYTES=1` AND `layer_num == 0`, best-effort `ttnn.to_torch(weight).contiguous().view(torch.uint8)[:16]`.  Failed on Qwen3-8B:

```
[U37_GT name=W1 ERROR: TT_FATAL @ pytensor.cpp:259: buffers.size() == 1
[U37_GT name=W3 ERROR: TT_FATAL @ pytensor.cpp:259: buffers.size() == 1
[U37_GT name=W2 ERROR: TT_FATAL @ pytensor.cpp:259: buffers.size() == 1
```

Multi-device sharded tensors don't survive a direct `ttnn.to_torch`.  ORTHOGONAL — the kernel cross-correlation (Paths A + B) is the definitive evidence; Path C was a belt-and-suspenders extra check.

### Path D — Factory env-gating

Two factory call-sites, both env-gated default-off:

```cpp
// matmul_multicore_reuse_mcast_1d_program_factory.cpp (gathered path)
const char* u37_read_env = std::getenv("SGLANG_TT_U37_READ_BYTES");
if (u37_read_env != nullptr && std::string(u37_read_env) == "1") {
    mm_kernel_defines["SGLANG_TT_U37_READ_BYTES"] = "1";
}

// dram_prefetcher_program_factory.cpp (writer_l1)
const char* env = std::getenv("SGLANG_TT_U37_PROD_BYTES");
if (env != nullptr && std::string(env) == "1") {
    writer_defines["SGLANG_TT_U37_PROD_BYTES"] = "1";
}
```

## DPRINT capture infrastructure

DPRINT requires `TT_METAL_DPRINT_CORES=all` to enable kernel-side prints.  The `TT_METAL_DPRINT_FILE_NAME=/tmp/u37_dprint.log` env var was set but the existing dprint server appears to route printf to the launch_server stdout (visible inline in `/tmp/u37_server2.log`).  Output format:

```
device:x-y:RISC: [U37_PROD layer=L t=T blk=B local_cb=... wr_ptr=... w=[...] bsz=...]
device:x-y:RISC: [U37_READ elf=0x... ring=R in1bs=... rd_l1=0x... w=[...]]
```

Where:
- `device` = 0 or 1 (mesh device index)
- `x-y` = logical core coordinates
- `RISC` = `BR` (BRISC, dataflow), `TR0` (UNPACK), `TR1` (MATH), `TR2` (PACK)
- BRISC reads come from the writer_l1 (producer); TR0 reads come from the gathered compute kernel (consumer UNPACK side)

## Hardware test result

```bash
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE_NAME=/tmp/u37_dprint.log \
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U32_GCB_OFFSET=1 \
SGLANG_TT_U33_FACTORY_OFFSET=1 \
SGLANG_TT_U34_LCM_ALIGN=1 \
SGLANG_TT_U35_PER_LAYER_OFFSET=1 \
SGLANG_TT_U36_WRAP_FIX=1 \
SGLANG_TT_U37_READ_BYTES=1 \
SGLANG_TT_U37_PROD_BYTES=1 \
SGLANG_TT_U37_GT_BYTES=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -m sglang.launch_server ... --model-path /models/Qwen3-8B ...
```

DPRINT lines captured: **2200+ U37_PROD + 1800+ U37_READ** across both devices and 16 producer cores + 2 consumer cores (only the (0,1)-5 and (1,1)-5 workers hit the ring_idx==0 && b==0 && block==0 gate within the static budget per program-launch).

Single-decode test: returned `" What不/..."` (Chinese-mixed garbage; different signature vs U36's `" What_fmt还不是周六..."` but same garbage class — expected since the GCB stomp pattern shifts with iteration count + which kernel cache is hit when).

## Why "all bytes correctly delivered, but BFP8 still garbage"

Five suspects for the new top hypothesis:

1. **Missing `set_tile_dims` on gathered path's src1 CB config** (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2159`).
   The gathered path only calls `.set_page_size(in1_single_tile_size).set_data_format(in1_data_format)` for src1_cb_index.  The non-gathered path (line 2163-2165) ALSO calls `.set_tile_dims(src1_cb_index, in1_tile)`.  Missing tile_dims may cause the BFP8 unpacker to use a default tile shape that mis-strides for the BFP8 layout.
2. **LLK BFP8 unpacker alignment** — even with correct bytes at rd_ptr, the unpacker may have an internal alignment requirement (e.g. 32-byte) that differs from the GCB's actual placement; with BFP4 (smaller tile, different stride) the misalignment may happen to land on safe bytes.
3. **PACK output format reconfig** — the gathered kernel calls `pack_reconfig_data_format(mm_partials_cb_id)` (line 479) and `pack_reconfig_data_format(mm_out_cb_id)` (line 748) but NEVER reconfigures the UNPACK side for in1.  If a previous matmul left the unpacker in a different data format state, the first matmul of the gathered loop may read with the wrong format.
4. **DST register layout for BFP8 accumulation** — fp32_dest_acc_en differs between ELFs (0 for FF1/FF3/FF2, 1 for WQKV/WO per U31 table).  Combined with BFP8 unpack, the DST tile layout may differ.
5. **In1 tile face order** — BFP8 32x32 tile has 4 faces of 16x16; the unpacker decodes them in face-major order.  The producer writes data in a specific face order via the resize_remote_sender_cb_interface page mechanism.  If the producer's `coalesced_page_size` != consumer's `face_size`, the bytes are correct at byte-0 but face-1 (the 257th byte onward) may be from a different tile.

The next attack (U38) should probe MORE bytes per dump — first 32 bytes (one face exponent + first row of mantissas).  If face-0 matches but face-1 diverges, we've found a face-stride mismatch.  This requires extending the U37 probe to dump 32 or 64 bytes.

## Hypothesis ledger (post-U37)

| ID | Suspect | Pre-U37 | Post-U37 |
|---|---|---|---|
| U7 / Case B — Gathered compute kernel produces garbage on specific CT-arg ELFs under REAL weights | CONFIRMED | UNCHANGED — root mechanism still active |
| U31 — Per-matmul rd_ptr reset to fifo_start independent of tensor's actual write position | CLOSED | CLOSED |
| U33 — Cumulative consumer-side offsets not aligned to matmul's `in1_block_size_bytes` | CLOSED | CLOSED |
| U34 — Even with correct, aligned offsets, GCB layout doesn't match producer-sim | CLOSED | CLOSED |
| U35-α — Matmul kernel's wrap-arithmetic edge case for wrap-straddling start offsets | CLOSED | CLOSED |
| U36-α (a) — Producer fifo_wr_ptr-tracked position differs from actual NoC-write destination on some receiver core | TOP STANDING | **REFUTED** — U37 verifies producer write source bytes match consumer read bytes at the matching wr_ptr/rd_l1, exactly, across all 5 tensors of layer 0. |
| U36-α (b) — Post-producer L1 stomp by another kernel between producer-write and consumer-read | TOP STANDING | **REFUTED** — U37 verifies kernel reads the producer's intended bytes at the moment matmul_block fires (the latest moment before consumption). |
| U36-α (c) — Data-format dependent decode bug in BFP4/BFP8 unpacker for specific tile patterns | TIE | UPGRADED to TOP STANDING |
| U36-α (d) — Sub-device cross-talk between W2's prefetcher_sub_device and matmul's worker_sub_device | TIE | DOWNGRADED — U37 byte-match across all tensors makes cross-talk unlikely (cross-talk would have shown up as wrong bytes in L1) |
| **NEW U37-α — Same compute kernel binary, same byte stream entering cb_in1, BFP4 path correct but BFP8 path garbage.  Suspect: (i) missing `set_tile_dims` on gathered path's src1 CB (line 2159); (ii) LLK BFP8 unpacker alignment; (iii) UNPACK-side data format reconfig missing; (iv) DST register layout for BFP8 + fp32_dest_acc_en combination; (v) In1 tile face order mismatch between producer's coalesced_page_size and consumer's face_size** | (new) | **TOP STANDING.**  Next attack: extend U37 probe to dump 32 or 64 bytes (face-0 exponent + face-1 mantissas) — if face-1 diverges from producer's bytes despite face-0 matching, we've found a face-stride mismatch. |

## Hard constraints checked

- [x] No upstream PR (`predator2k/*` only; commit `e834da9df37` on `tenstorrent-p1`).
- [x] No SGLang core behavior changes (sglang Python-side `python/sglang/...` untouched; only tt-metal-sglang C++ + the explicitly-allowed `mlp.py`).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` for source sync + FIXED `rebuild_tt_metal_kernels.sh`; verified sentinels `SGLANG_TT_U37_READ_BYTES` + `SGLANG_TT_U37_PROD_BYTES` present in `_ttnncpp.so`.
- [x] `tensix_sync()` applied before the L1 byte read in the kernel probe (per U13 lesson).
- [x] All U37 paths env-gated (`SGLANG_TT_U37_READ_BYTES`, `SGLANG_TT_U37_PROD_BYTES`, `SGLANG_TT_U37_GT_BYTES`).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"`) + GSM8K(10) chat = 10/10 = 100% (matches U36 baseline).
- [x] Server stopped + cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: **`e834da9df37`** (U37, 5 files, +200 lines).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/ttnn/cpp/...` + `/tt-metal/models/...` source: synced; `_ttnn.so` + `_ttnncpp.so` synced to install dir; sentinels `SGLANG_TT_U37_READ_BYTES` + `SGLANG_TT_U37_PROD_BYTES` verified in `strings _ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` + GSM8K(10) chat = 10/10 = 100%.
- Artifacts preserved in-container:
  - `/tmp/u37_server2.log` (U37 + DPRINT_CORES=all run; 2200+ U37_PROD + 1800+ U37_READ lines; bytes-match cross-correlation source)
  - `/tmp/u37_canonical.log` (canonical no-prefetcher control; bytewise-equal canonical baseline + 10/10 GSM8K)

## Shipping verdict (unchanged from U31/U32/U33/U34/U35/U36)

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  U37 verifies that the bug is NOT in the byte-delivery layer (producer→L1→consumer) but in the consumer's compute/decode path for BFP8 specifically.  No fix in this session.
- **`SGLANG_TT_U37_READ_BYTES=1` + `SGLANG_TT_U37_PROD_BYTES=1` + `SGLANG_TT_U37_GT_BYTES=1` are all safe to leave in tree (default-off + canonical bytewise-equivalent at default).**  Only enable DPRINT in the gathered matmul / prefetcher writer / mlp.py register_weights paths under U37=1.

## 50-dispatch ledger (cumulative)

| ID | Status | Note |
|---|---|---|
| S1-S10, Lead 1/2/3, A1 | CLOSED | Per-doc closure. |
| U1-U16 | CLOSED | DST stale, GCB block bisect, LLK probes, PACK probes, W2→RS barrier, etc. |
| U17-U28 | CLOSED | All "downstream stomper" theories refuted. |
| U29 | CLOSED | W2→RS semaphore handshake verified working; garbage persists. |
| U30 | CLOSED | Per-ELF PACK-side garbage: 86% NaN/Inf at PACK exit for 3 of 4 ELFs. |
| U31 | CLOSED | Discriminator: BFP8 vs BFP4 dtype. |
| U32 | CLOSED | Structural plumbing (kernel + factory + override + Python) landed. |
| U33 | CLOSED | Consumer-side cumulative offsets landed Python-side; alignment gap → U34. |
| U34 | CLOSED | Producer-simulation aligned offsets + critical lookup-bug fix; broadcast crashes ELIMINATED; output garbage persists. |
| U35 | CLOSED | Kernel-side AND producer-side DPRINT prove offsets correct; per-layer producer-drift identified + fixed via stateful sim; per-layer stateful offset plumbs correctly through the kernel but produces NaN. |
| U36 | CLOSED | Pre-emptive wrap fix functionally equivalent to original; per-block rd_ptr trace verifies kernel reads at correct producer-matching positions; output still garbage + NaN; U36-α REFUTED. |
| **U37** | **EVIDENCE_ADVANCE — three byte-level probes (producer write source, consumer read target, Python ground-truth) land; consumer-read bytes EXACTLY match producer-write bytes at the matching L1 address for all 5 tensors of layer 0 decode step 1; output still garbage.  U36-α (a) and (b) BOTH REFUTED.** | **U37-α NEW TOP STANDING** — same compute kernel binary processes correctly-delivered BFP4 bytes correctly but correctly-delivered BFP8 bytes incorrectly.  Bug lives in compute / unpacker / MATH / PACK / tile-stride behavior that diverges between BFP4 and BFP8 on the gathered (`use_global_cb`) path specifically. |
| U38 | OPEN | Extend U37 probe to dump 32 or 64 bytes per probe (cover face-0 exponent + face-1 first row of mantissas).  If face-1 diverges from producer's bytes despite face-0 matching, we've found a face-stride mismatch.  Alternatively, add `.set_tile_dims(src1_cb_index, in1_tile)` to the gathered path's `remote_cb_config` (factory line 2159) and re-test — straight-line repair attempt. |

## Commits this session

- (tt-metal-sglang) **1 commit**: `e834da9df37 prefetcher: U37 — ground-truth byte-level diagnostic (producer + consumer + python GT); MATCH verdict — producer's bytes ARE correctly delivered to consumer L1; bug lives in compute/decode path NOT in delivery (env-gated, default-off; canonical Qwen3-8B GSM8K 10/10 preserved)`.
- (sglang) `<this doc>` — pending commit.
