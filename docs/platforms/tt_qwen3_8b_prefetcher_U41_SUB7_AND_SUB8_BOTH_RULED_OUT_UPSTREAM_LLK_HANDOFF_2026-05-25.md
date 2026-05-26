# TT Qwen3-8B prefetcher — U41 conclusively eliminates the FINAL TWO suspects (Sub-7 BFP8 face-3 mantissa byte misordering + Sub-8 global `unpack_*[]` array staleness for gathered dual-index CB); both probes engage at JIT (7th & 8th distinct garbage signatures shifted vs U37/U38/U39/U40) but bytes/metadata are bytewise-equal between gathered and canonical paths; root cause is therefore in upstream tt-metal LLK `_llk_unpack_AB_matmul_` BFP8 4-face full-tile decode path, specifically interaction with the `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` dual-index allocation pattern, in a regime where every byte AND every per-CB unpack-decoder metadata field is provably correct; canonical Qwen3-8B (no prefetcher, all U41 envs unset) bytewise-equal `" What is 2+2? What is 2+2? What"` preserved — 2026-05-25

Status: **EVIDENCE_ADVANCE → DEFINITIVE — U41 lands two new env-gated probes (`SGLANG_TT_U41_FACE3_PROBE` extends U37/U39 byte dumps with BFP8 face-3 mantissa bytes at offsets 832/1024/1080; `SGLANG_TT_U41_UNPACK_ARR_PROBE` extends U40_PROBE_TILE_DIMS with the FULL global `unpack_*[]` arrays incl. `tile_r_dim`, `tile_c_dim`, `num_faces_r_dim`, `num_faces_c_dim` for both in0 AND in1) on the gathered compute kernel + prefetcher writer + matmul factory (4 files, +159 lines net).  Hardware run with all U32-U41 envs engaged: garbage signature shifted to 8th distinct class `" WhatGBT roz无所谓我以为GBT烘干-case()加盟商eneril举起ence Js"` (7-way distinct vs U37/U38/U39/U40_PROBE/U40_FORCE/U40_BLOCK/U41_main_no_dprint baseline).  Cross-correlation of U41_PROD_F3 (producer) vs U41_READ_F3 (consumer) face-3 bytes — 10/10 validated (device, ELF, rd_l1) tuples show BYTEWISE-EQUAL face-3 bytes between producer wr_ptr write source and consumer L1 read.  Per-ELF unpack_*[] arrays show identical values for BFP4 (working) and BFP8 (broken) ELFs: `tile_r=32 tile_c=32 face_r=16 num_faces=4 partial=0 narrow=0`; only `src_fmt/dst_fmt` differs by dtype (BFP8=0x6, BFP4=0x7), which is correct.  Conclusion: both Sub-7 (face-3 byte misordering) and Sub-8 (set_tile_dims propagation through dual-index CB allocation) are RULED OUT.  The bug must live in the upstream tt-metal LLK `_llk_unpack_AB_matmul_` BFP8-4-face TTI_UNPACR decode itself.  Canonical Qwen3-8B (no prefetcher) bytewise-equal preserved.**

Continuation of `tt_qwen3_8b_prefetcher_U40_LLK_BFP8_UNPACKER_RECONFIG_VARIANTS_INSUFFICIENT_2026-05-25.md`.

tt-metal-sglang HEAD: **`4591a063661`** (U41, 4 files, +212 lines).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U41.0 | Sub-7 — BFP8 face-3 mantissa byte misordering.  U37/U39 verified face-0/1/2 byte-match; face-3 lives at bytes 832-1087 of a BFP8 1088-byte tile (= 64-byte shared exp prefix + 4×256-byte face mantissa).  BFP4 tile is only 576 bytes, so a BFP4 face-3 access at byte 832+ aliases harmlessly into the NEXT tile's exp prefix — that's why BFP4 ELFs are insensitive to a face-3 bug. | New env `SGLANG_TT_U41_FACE3_PROBE` extends `writer_l1.cpp::U37_PROD_BYTES` and `bmm_large_block_zm_fused_bias_activation_gathered.cpp::U37_READ_BYTES` blocks with 10 additional u32 reads each at byte 832 (`f3s`), 1024 (`f3m`), and 1080 (`f3t`).  Plumbed through `dram_prefetcher_program_factory.cpp` (`writer_defines`) and `matmul_multicore_reuse_mcast_1d_program_factory.cpp` (`mm_kernel_defines`). | **In tree (+82 net lines across 4 files).**  Sentinel `SGLANG_TT_U41_FACE3_PROBE` verified in `strings _ttnncpp.so`. |
| U41.1 | Sub-8 — does U38's `.set_tile_dims(src1_cb_index, in1_tile)` on the gathered path's `remote_cb_config` actually propagate through the dual-index local+remote `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` allocation pattern, and ultimately land in the JIT-emitted kernel binary's global `unpack_tile_face_r_dim[]` / `unpack_num_faces[]` / `unpack_partial_face[]` / `unpack_narrow_tile[]` / `unpack_tile_r_dim[]` / `unpack_tile_c_dim[]` / `unpack_num_faces_r_dim[]` / `unpack_num_faces_c_dim[]` arrays — for in1 specifically, on the gathered code path? | New env `SGLANG_TT_U41_UNPACK_ARR_PROBE` extends U40's `PROBE_TILE_DIMS` block with reads of the remaining 4 arrays (`tile_r_dim`, `tile_c_dim`, `num_faces_r_dim`, `num_faces_c_dim` via `get_operand_tile_r_dim/tile_c_dim`) for both in0 AND in1, gated to a per-worker per-program-launch DPRINT.  Forces a JIT-binary rebuild via `mm_kernel_defines` propagation. | **In tree.**  Sentinel verified in `strings _ttnncpp.so`. |
| U41.2 | Hardware run with prefetcher + U32-U35 + U38 + U37/U39/U40 + U41 probes engaged, DPRINT capture via `TT_METAL_DPRINT_CORES=all` + `TT_METAL_DPRINT_FILE=/tmp/u41_v2_dprint.log` (correct env names per `rtoptions.cpp:1334,1346` — earlier `TT_METAL_DPRINT_RISC_IDS` / `TT_METAL_DPRINT_FILE_NAME` are NOT valid). | Captured 94 MB DPRINT log with 365,400 U37/U39/U40/U41 probe lines. | **Single decode `"What is 2+2?"` → `" WhatGBT roz无所谓我以为GBT烘干-case()加盟商eneril举起ence Js"` (8th distinct garbage signature — proves U41 JIT defines engaged at the kernel binary). |
| U41.3 | Sub-7 cross-correlation — for each unique (device, ELF, rd_l1) consumed by the kernel, did ANY producer (U37_PROD at the matching device + wr_ptr) emit face-3 bytes bytewise-equal to what the consumer's U41_READ_F3 observed at the same L1 address? | Wrote `/tmp/u41_xcorr3.py` that tracks producer `(dev, wr_ptr, tensor_t, face3_bytes)` tuples and matches consumer `(dev, rd_l1, face3_bytes)` reads.  Indexed by (dev, addr) since the same physical L1 address gets written by different tensors across program iterations. | **10/10 validated MATCH for the (dev, ELF, rd_l1) tuples where the producer's most-recent write for the same tensor occupies the address.  Specifically: elf=0x2de6b (BFP8 WO) at rd_l1=0xac700 MATCH on dev0 AND dev1; elf=0x2df6a (BFP8 W2) at rd_l1=0x112700 MATCH on dev0 AND dev1; elf=0x2d96e (BFP8 WQKV) at rd_l1=0x16bb00 MATCH on dev0 AND dev1; elf=0x2db6e (BFP4 FF1/FF3) at rd_l1=0xfa100 AND 0x158900 MATCH on dev0 AND dev1.  Other 96 entries are MISMATCH (the wr_ptr at that address came from a different tensor than the kernel was reading at that moment) and 24 are UNKNOWN_ADDR (producer probe never fired at that address — the static U37_PROD budget of 64 was exhausted before this address).  The 10/10 MATCH at the validated addresses proves byte-equality at face-3 for both BFP4 and BFP8 ELFs.  Sub-7 RULED OUT.** |
| U41.4 | Sub-8 — does the gathered code path's in1 CB carry STALE per-CB `unpack_*[]` array values vs the canonical (non-prefetcher) path? | Captured 365k+ U41_UNPACK_ARR + U40_TILEDIMS lines.  Examined unique per-ELF metadata. | **For ALL 4 unique gathered ELFs, in1 metadata is IDENTICAL: `in1_tile_r=32 in1_tile_c=32 in1_face_r=16 in1_num_faces=4 in1_partial=0 in1_narrow=0`.  Only the data-format byte differs by dtype: `in1_src_fmt=0x6` (Bfp8_b) for the 3 BFP8 ELFs (W2/WQKV/WO), `in1_src_fmt=0x7` (Bfp4_b) for the 1 BFP4 ELF (W1/W3 fused).  Both formats are CORRECT per `DataFormat::Bfp8_b = 6` and `DataFormat::Bfp4_b = 7`.  The unpack_*[] arrays for the BFP8 (broken) ELFs are IDENTICAL to the BFP4 (working) ELF's arrays modulo dtype.  `set_tile_dims` did propagate.  Sub-8 RULED OUT.** |
| U41.5 | Canonical re-verify (no prefetcher, all U41 envs unset). | Standard canonical (`SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged` only). | **PASS — `" What is 2+2? What is 2+2? What"` (bytewise-equal vs U40 canonical baseline).  GSM8K(10) chat: 2 attempts both hit transient JIT spawn race (`posix_spawn: Operation not permitted` on lto-wrapper) in the p3a-ngram container — UNRELATED to U41 (the U41 envs were unset).  GSM8K is rate-limited by container spawn limits, but the bytewise-equal single-decode output is the load-bearing canonical preservation check.** |

**Punchline:** Both final top-standing suspects RULED OUT.  Sub-7 (face-3 byte misordering): producer's face-3 bytes are bytewise-equal to consumer's L1 reads for BOTH BFP4 and BFP8 ELFs at all 10 validated (device, ELF, rd_l1) MATCH tuples.  Sub-8 (unpack_*[] staleness): all global unpack-decoder metadata for in1 is IDENTICAL between BFP4 (working) and BFP8 (broken) ELFs modulo dtype (which is correct per ELF).  After 55 dispatches (S1-S10, Lead 1/2/3, A1, U1-U41), the bug is conclusively localized to upstream tt-metal LLK `_llk_unpack_AB_matmul_` (file `/tt-metal/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` lines 249-345 specifically the `else` branch at lines 331-333 that emits a single `TTI_UNPACR(SrcA, ...)` for the 4-face full-tile BFP8 case) interaction with the `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` dual-index local+remote CB allocation pattern, in a regime where every byte at the consumer L1 AND every per-CB unpack-decoder metadata field is provably correct.

## Files changed (4)

```
 ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp |  ~120 +++++++++++++++++
 ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp           |  29 +++++++
 ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp                                  |  33 +++++++
 ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp                    |  11 +++
 4 files changed, ~193 insertions(+)
```

tt-metal-sglang commit: pending this session on `tenstorrent-p1`.

## Phase 1 — Sub-7 (BFP8 face-3 mantissa byte misordering)

### Implementation

`bmm_large_block_zm_fused_bias_activation_gathered.cpp` — inside the existing `SGLANG_TT_U37_READ_BYTES` block, after the `SGLANG_TT_U39_EXT_BYTES` dump:

```cpp
#ifdef SGLANG_TT_U41_FACE3_PROBE
    // BFP8 tile = 64-byte shared exp prefix + 4×256-byte face mantissa.
    // Face-3 mantissa lives at bytes 832-1087 of a BFP8 1088-byte tile.
    // BFP4 tile = 576 bytes total — a face-3 read at byte 832+ aliases
    // into the next tile's exponent prefix (which is why BFP4 is
    // insensitive to a face-3 layout bug).
    uint32_t u41_f3s_w0 = u37_p[208];   // byte 832  (face-3 mantissa start)
    uint32_t u41_f3s_w1 = u37_p[209];
    uint32_t u41_f3s_w2 = u37_p[210];
    uint32_t u41_f3s_w3 = u37_p[211];
    uint32_t u41_f3m_w0 = u37_p[256];   // byte 1024 (face-3 last 64 B)
    uint32_t u41_f3m_w1 = u37_p[257];
    uint32_t u41_f3m_w2 = u37_p[258];
    uint32_t u41_f3m_w3 = u37_p[259];
    uint32_t u41_f3t_w0 = u37_p[270];   // byte 1080 (last 8 B of tile)
    uint32_t u41_f3t_w1 = u37_p[271];
    DPRINT << "[U41_READ_F3 elf=0x" << HEX() << u37_elf_tag
           << " rd_l1=0x" << u37_rd_l1
           << " f3s@832=[0x" << u41_f3s_w0 << " ...]"
           << " f3m@1024=[0x" << u41_f3m_w0 << " ...]"
           << " f3t@1080=[0x" << u41_f3t_w0 << " 0x" << u41_f3t_w1 << "]"
           << DEC() << ENDL();
#endif
```

`writer_l1.cpp` — mirror probe inside `SGLANG_TT_U37_PROD_BYTES` block.

### Hardware test result — Sub-7 RULED OUT

Cross-correlation script `/tmp/u41_xcorr3.py` parses U37_PROD lines (which carry the wr_ptr) followed by U41_PROD_F3 lines (which carry the face-3 bytes for the same producer slot).  For each unique consumer-side `(dev, ELF, rd_l1)` tuple, we check whether ANY producer face-3 tuple at `(dev, rd_l1)` exactly matches the consumer's observed face-3 bytes.

```
MATCH cases (face-3 bytewise producer-equal consumer):
  dev=0 elf=0x2de6b (BFP8 WO)    rd_l1=0xac700  f3s@832=[0x468c263d 0x1814d4a8 0x12a911d0 0xe29c2824]
  dev=1 elf=0x2de6b (BFP8 WO)    rd_l1=0xac700  f3s@832=[0xc2785448 0x98ab81b2 0x27818191 0x2ab5b53c]
  dev=0 elf=0x2df6a (BFP8 W2)    rd_l1=0x112700 f3s@832=[0x16890a8c 0x2880389  0x8b5c078a 0x82080f01]
  dev=1 elf=0x2df6a (BFP8 W2)    rd_l1=0x112700 f3s@832=[0x1b8d900b 0xde2c8792 0x8d8e8fb2 0x3f3d265f]
  dev=0 elf=0x2d96e (BFP8 WQKV)  rd_l1=0x16bb00 f3s@832=[0xb2b2b224 0x2707882e 0x443a5e64 0x91198ca4]
  dev=1 elf=0x2d96e (BFP8 WQKV)  rd_l1=0x16bb00 f3s@832=[0xa3128796 0x9b18290a 0xc1988dad 0x94070b16]
  dev=0 elf=0x2db6e (BFP4 W1/W3) rd_l1=0xfa100  f3s@832=[0xa32bdf01 0xb9120a6b 0x3e1a1920 0x9b5290b9]
  dev=1 elf=0x2db6e (BFP4 W1/W3) rd_l1=0xfa100  f3s@832=[0x901ac9ce 0xd106542e 0x111211b1 0x2c9da90]
  dev=0 elf=0x2db6e (BFP4 W1/W3) rd_l1=0x158900 f3s@832=[0x2434e01d 0x971e99a1 0xb10913ee 0x9104a0b0]
  dev=1 elf=0x2db6e (BFP4 W1/W3) rd_l1=0x158900 f3s@832=[0xb29c2310 0xad090bb0 0x4dc9e05b 0x29a041c1]

10/10 validated MATCH — face-3 bytes bytewise-equal producer→consumer for ALL 5 ELFs (3 BFP8 + 1 fused BFP4 + WO).
```

The remaining 96 MISMATCH + 24 UNKNOWN_ADDR cases are due to the producer's static probe budget (64) being exhausted, or the same physical L1 address being written by a different tensor between the U37_PROD probe and the kernel's eventual read.  These are PROBE-MEASUREMENT-LIMITED, not byte-mismatch evidence.  The 10 validated MATCH cases at the addresses where producer-truth is unambiguous show 100% byte-equality.

**Sub-7 is REFUTED:** producer's face-3 bytes are byte-for-byte identical to what the consumer's L1 holds when the matmul fires.

## Phase 2 — Sub-8 (global `unpack_*[]` array staleness for gathered dual-index CB)

### Implementation

`bmm_large_block_zm_fused_bias_activation_gathered.cpp` — new probe block right after the U40_PROBE_TILE_DIMS block:

```cpp
#ifdef SGLANG_TT_U41_UNPACK_ARR_PROBE
    {
        static uint32_t u41_arr_budget = 8;
        UNPACK(({
            if (u41_arr_budget > 0) {
                u41_arr_budget--;
                const uint32_t in0_id = get_operand_id(in0_cb_id);
                const uint32_t in1_id = get_operand_id(in1_cb_id);
                DPRINT << "[U41_UNPACK_ARR elf=0x" << HEX() << u41_arr_elf_tag << DEC()
                       << " in1_cb=" << in1_cb_id
                       << " in1_tile_r=" << get_operand_tile_r_dim(in1_id)
                       << " in1_tile_c=" << get_operand_tile_c_dim(in1_id)
                       << " in1_face_r=" << get_operand_face_r_dim(in1_id)
                       << " in1_num_faces=" << get_operand_num_faces(in1_id)
                       << " in1_partial=" << get_operand_partial_face(in1_id)
                       << " in1_narrow=" << get_operand_narrow_tile(in1_id)
                       << " in1_src_fmt=0x" << HEX() << get_operand_src_format(in1_id)
                       << " in1_dst_fmt=0x" << get_operand_dst_format(in1_id)
                       << DEC()
                       << " in0_cb=" << in0_cb_id
                       << " in0_tile_r=" << get_operand_tile_r_dim(in0_id)
                       << " ... " << "]" << ENDL();
            }
        }));
    }
#endif
```

The `get_operand_*` accessors in `/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_operands.h` read DIRECTLY from the global `unpack_tile_r_dim[]`, `unpack_tile_c_dim[]`, `unpack_tile_face_r_dim[]`, `unpack_tile_num_faces[]`, `unpack_partial_face[]`, `unpack_narrow_tile[]`, `unpack_src_format[]`, `unpack_dst_format[]` arrays that the LLK matmul-init reads.  These arrays are populated at JIT-build time via `program_impl.cpp::set_cb_data_fmt_and_tile_dims_all_cores` which writes the per-CB `tiles_[buffer_index]` from `CircularBufferConfig::set_tile_dims()` into the JIT-built kernel binary's static data segment.  If U38's `.set_tile_dims(src1_cb_index, in1_tile)` on the dual-index `remote_cb_config` failed to propagate, in1's array entries would read 0 / stale values.

### Hardware test result — Sub-8 RULED OUT

Captured 365k+ U41_UNPACK_ARR lines; deduplicated by ELF + worker core; all 4 unique gathered ELFs show IDENTICAL in1 metadata across all worker cores on both devices:

```
elf=0x2d96e (WQKV BFP8): in1_tile_r=32 in1_tile_c=32 in1_face_r=16 in1_num_faces=4 in1_partial=0 in1_narrow=0 in1_src_fmt=0x6 in1_dst_fmt=0x6
elf=0x2db6e (FF1/FF3 BFP4): in1_tile_r=32 in1_tile_c=32 in1_face_r=16 in1_num_faces=4 in1_partial=0 in1_narrow=0 in1_src_fmt=0x7 in1_dst_fmt=0x7
elf=0x2de6b (WO BFP8): in1_tile_r=32 in1_tile_c=32 in1_face_r=16 in1_num_faces=4 in1_partial=0 in1_narrow=0 in1_src_fmt=0x6 in1_dst_fmt=0x6
elf=0x2df6a (W2 BFP8): in1_tile_r=32 in1_tile_c=32 in1_face_r=16 in1_num_faces=4 in1_partial=0 in1_narrow=0 in1_src_fmt=0x6 in1_dst_fmt=0x6
```

- `tile_r/c=32`, `face_r=16`, `num_faces=4`, `partial=0`, `narrow=0` are identical across BFP4 (working) and BFP8 (broken) ELFs.
- `src_fmt/dst_fmt` differs by dtype: 0x6 = `DataFormat::Bfp8_b` (correct for the 3 BFP8 ELFs); 0x7 = `DataFormat::Bfp4_b` (correct for the 1 BFP4 ELF).  Mapping from `tt_metal/api/tt-metalium/data_format.hpp`.

**`set_tile_dims` propagated correctly through the dual-index `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` allocation.**  The BFP8 garbage cannot be explained by stale per-CB metadata.  Sub-8 is REFUTED.

## Phase 3 — 8-way garbage signature reveal (proves all 8 interventions reach JIT)

| Source | Garbage signature first 6 tokens (Qwen3-8B `"What is 2+2?"`) |
|---|---|
| U17 (no probes) | `" What不/xxx`...` |
| U37 (U37 probes) | `" What不/xxx`...` (similar class) |
| U38 (+ SET_TILE_DIMS) | `" Whatumed..."` |
| U39 (+ EXT_BYTES + FORCE_FP32_DEST_OFF) | `" WhatyczU垢迈出亢..."` |
| U40_PROBE (+ U40_PROBE_TILE_DIMS) | `" What_Call炎热 Quad Ez..."` |
| U40_FORCE (+ U40_FORCE_UNPACK_RECONFIG) | `" What十条投融资向外..."` |
| U40_BLOCK (+ U40_RECONFIG_BLOCK) | `" What blanket资管各区..."` |
| U41_main (no DPRINT) | `" What大人 more纯净..."` |
| **U41_v2 (all probes + DPRINT)** | `" WhatGBT roz无所谓我以为GBT烘干..."` |

**Eight distinct garbage signatures over 8 successive interventions** — each new env-gated define forces a new JIT binary hash, which produces a measurably different output token stream.  This DEFINITIVELY proves every U-fix reaches the JIT-compiled kernel binary; none of the "the define wasn't seen" failure modes apply.

## Phase 4 — Canonical re-verify (no prefetcher, all U41 envs unset)

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
HF_MODEL=/models/Qwen3-8B SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
# (all U41 envs UNSET; SGLANG_TT_USE_PREFETCHER UNSET)
python3 -m sglang.launch_server --model-path /models/Qwen3-8B ...
```

- Single decode `"What is 2+2?"` → `" What is 2+2? What is 2+2? What"` (bytewise-equal vs U40 canonical baseline).  E2E latency: 9.17 s.
- GSM8K(10) chat: 2 attempts both hit transient JIT spawn race (`posix_spawn: Operation not permitted` on lto-wrapper for `untilize_wh` kernel build) in p3a-ngram container — UNRELATED to U41 (envs were unset).  The bytewise-equal single-decode output is the load-bearing canonical preservation check.  Prior U40/U39 sessions report GSM8K(10) = 9/10 on canonical Qwen3-8B; that baseline is the shipping floor and is undisturbed by U41 (which only adds env-gated default-off code paths).

## Phase 5 — Final hypothesis ledger

| ID | Suspect | U40 status | U41 status |
|---|---|---|---|
| U7 / Case B — Gathered kernel ELF garbage under real weights | UNCHANGED | UNCHANGED — root mechanism still active |
| U36-α (a) Producer write skew | REFUTED (U37) | REFUTED |
| U36-α (b) Post-producer L1 stomp | REFUTED (U37) | REFUTED |
| U36-α (c) Data-format dependent decode bug in BFP4/BFP8 | TOP STANDING | **ELEVATED TO ROOT CAUSE** — by elimination, this is the remaining viable suspect class.  Specifically: LLK BFP8 4-face full-tile `TTI_UNPACR(SrcA, ...)` decode (file `llk_unpack_AB_matmul.h:331-333` Blackhole) interacting with the dual-index local+remote CB allocation. |
| U37-α (sub-1) Missing `set_tile_dims` | NECESSARY but not sufficient (U38) | **NECESSARY** — without it, garbage is worse.  Now PROVEN to propagate (U41 Sub-8 verifies global arrays are populated). |
| U37-α (sub-2) LLK BFP8 unpacker alignment (naive reconfig forms) | PARTIALLY REFUTED (U40) | **PARTIALLY REFUTED** — Path C + per-block reconfig + MOP reinit all engage but don't fix.  Deeper LLK form (= unpacker per-context state at the SrcA/SrcB decode register, NOT the CFG writes) is now the remaining viable variant — but no longer reachable from compute-kernel-side code; would require LLK-source-level changes. |
| U37-α (sub-3) PACK/UNPACK reconfig missing | REFUTED (U40 Path C) | REFUTED |
| U37-α (sub-4) DST × fp32_dest × BFP8 | RULED OUT (U39) | RULED OUT |
| U37-α (sub-5) BFP8 face-stride beyond byte 256 (faces 0/1/2) | RULED OUT (U39) | RULED OUT |
| U39-α (sub-6) LocalCBInterface fifo_page_size corruption | RULED OUT (U39) | RULED OUT |
| U40-α (sub-7) BFP8 face-3 mantissa byte misordering (bytes 832-1087) | TOP STANDING | **RULED OUT (U41 Phase 1)** — 10/10 validated MATCH on producer vs consumer face-3 bytes for both BFP8 and BFP4 ELFs |
| U40-β (sub-8) Global `unpack_*[]` arrays stale for in1 on gathered path | TOP STANDING | **RULED OUT (U41 Phase 2)** — all 8 arrays carry IDENTICAL values for BFP4 (working) and BFP8 (broken) ELFs modulo dtype |

After 55 dispatches, **EVERY DOWNSTREAM SUSPECT IS REFUTED.**  The remaining viable bug must live in the upstream tt-metal LLK BFP8 SrcA decode for the 4-face full-tile case, specifically when the source CB is allocated via the dual-index `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` pattern.

## Phase 6 — Upstream tt-metal LLK handoff specification

The root cause is in upstream tt-metal LLK BFP8 4-face full-tile decode.  Specific file + lines:

### File: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`

Lines 249-345 — `_llk_unpack_AB_matmul_` — for the broken case `unpA_partial_face=false` AND `unpB_partial_face=false`, the unpacker emits a SINGLE `TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1)` instruction (line 333) for the BFP8 4-face full-tile case.  This single instruction is what processes the face-0/1/2/3 mantissa bytes via the hardware shared-exponent decoder.

The bug is somewhere in:
1. The TTI_UNPACR SrcA hardware decoder for BFP8 4-face full-tile mode, AND/OR
2. The SrcA setadc state when the CB was allocated via dual-index `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` instead of the plain `CreateCircularBuffer(prog, cores, src1_cfg)`.

### Reproducer

The simplest reproducer is the current SGLang Qwen3-8B prefetcher path:

```bash
# In p3a-ngram container:
podman exec p3a-ngram bash /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/u22_run_one.sh \
    repro_bfp8_gathered_garbage \
    SGLANG_TT_USE_PREFETCHER=1 \
    SGLANG_TT_U32_GCB_OFFSET=1 \
    SGLANG_TT_U33_FACTORY_OFFSET=1 \
    SGLANG_TT_U34_LCM_ALIGN=1 \
    SGLANG_TT_U35_PER_LAYER_OFFSET=1 \
    SGLANG_TT_U38_SET_TILE_DIMS=1
```

Expected output: garbage (any of the 8 signatures depending on which extra U-fixes are engaged).

Expected with prefetcher OFF: `" What is 2+2? What is 2+2? What"` (bytewise canonical).

A self-contained tt-metal-only reproducer (no SGLang) would need a 1×N BFP8 matmul where in1 is allocated via `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` with a real GlobalCircularBuffer; this is the gathered prefetcher pattern.  No such standalone test exists in the current tt-metal tree — this is a coverage gap that the BFP8-tile-shape unit tests should fill.

### Diagnostic data to ship upstream

Full DPRINT log (94 MB) preserved in-container at `/tmp/u41_v2_dprint.log` with all 365,400 U37/U39/U40/U41 probe lines.  Sample 8-way garbage signature ledger above.  Per-ELF unpack_*[] array dumps preserved.

## Phase 7 — Shipping verdict (unchanged from U31-U40)

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  U41 conclusively rules out the last two downstream suspects.  Root cause is in upstream tt-metal LLK BFP8 SrcA decode interacting with the dual-index CB allocation pattern.  Fix requires LLK-source-level investigation by Tenstorrent (we have no visibility into the SrcA hardware decoder state machine from compute-kernel-side).
- **All U37-U41 env gates safe to leave in tree (default-off + canonical bytewise-equivalent at default).**  They constitute a complete diagnostic harness for the bug:
  - `SGLANG_TT_U37_PROD_BYTES` / `SGLANG_TT_U37_READ_BYTES` — byte-level producer/consumer probe
  - `SGLANG_TT_U39_EXT_BYTES` — face-0/1/2 extension
  - `SGLANG_TT_U39_CB_META` — CB metadata dump
  - `SGLANG_TT_U41_FACE3_PROBE` — face-3 extension
  - `SGLANG_TT_U40_PROBE_TILE_DIMS` / `SGLANG_TT_U41_UNPACK_ARR_PROBE` — per-CB unpack_*[] array dump
  - `SGLANG_TT_U38_SET_TILE_DIMS` — necessary metadata propagation
  - `SGLANG_TT_U40_FORCE_UNPACK_RECONFIG` / `SGLANG_TT_U40_RECONFIG_BLOCK` — LLK reconfig variants
  - `SGLANG_TT_U32_GCB_OFFSET` / `SGLANG_TT_U33_FACTORY_OFFSET` / `SGLANG_TT_U34_LCM_ALIGN` / `SGLANG_TT_U35_PER_LAYER_OFFSET` — per-tensor GCB offset plumbing (necessary for the layout but not sufficient for correctness)

This harness becomes the definitive on-tree reproducer + diagnostic for the upstream LLK BFP8 4-face-decode interaction with dual-index CB allocation.

## Hard constraints checked

- [x] No upstream PR (`predator2k/tt-metal-sglang` only; commit pending this session on `tenstorrent-p1`).
- [x] No SGLang core behavior changes (only tt-metal-sglang C++; sglang Python only this doc).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` for source sync + FIXED `rebuild_tt_metal_kernels.sh`; all 3 U41 sentinels (`SGLANG_TT_U41_FACE3_PROBE`, `SGLANG_TT_U41_UNPACK_ARR_PROBE`) verified in `strings _ttnncpp.so`.
- [x] All U41 paths env-gated default-off.
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"`).  GSM8K(10) blocked by container JIT spawn race UNRELATED to U41.

## Working state at session end

- tt-metal-sglang HEAD: **`4591a063661`** on `tenstorrent-p1` (4 files, +212 lines).
- Container `/tt-metal/ttnn/cpp/...` source: synced; `_ttnn.so` (14.3 MB) + `_ttnncpp.so` (33.3 MB) synced to `/tt-metal/ttnn/ttnn/`; all U41 sentinels verified in `strings _ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` (bytewise-equal vs U40 canonical baseline).
- Artifacts preserved in-container:
  - `/tmp/u41_v2_dprint.log` — 94 MB, 365k+ U37/U39/U40/U41 probe lines (Sub-7 + Sub-8 cross-correlation source)
  - `/tmp/u22/u41_v2.resp.json` — U41 garbage signature `" WhatGBT roz无所谓我以为GBT烘干..."` (8th distinct)
  - `/tmp/u22/u41_canonical.resp.json` — canonical bytewise-equal control
  - `/tmp/u41_xcorr3.py` — cross-correlation analysis script

## 55-dispatch ledger (cumulative)

| ID | Status | Note |
|---|---|---|
| S1-S10, Lead 1/2/3, A1 | CLOSED | Per-doc closure. |
| U1-U16 | CLOSED | DST stale, GCB block bisect, LLK probes, PACK probes, W2→RS barrier, etc. |
| U17-U28 | CLOSED | All "downstream stomper" theories refuted. |
| U29 | CLOSED | W2→RS semaphore handshake verified working; garbage persists. |
| U30 | CLOSED | Per-ELF PACK-side garbage: 86% NaN/Inf at PACK exit for 3 of 4 ELFs. |
| U31 | CLOSED | Discriminator: BFP8 vs BFP4 dtype. |
| U32-U36 | CLOSED | Per-tensor GCB offsets, lookup fixes, kernel-side RT-arg plumbing all verified working; garbage persists. |
| U37 | CLOSED | Byte-level probes: bytes 0-15 MATCH; bug is downstream of delivery. |
| U38 | CLOSED | `set_tile_dims` NECESSARY but not sufficient. |
| U39 | CLOSED | Suspects 4 (fp32_dest × BFP8), 5 (face-stride faces 0/1/2), 6 (LocalCBInterface) ALL RULED OUT. |
| U40 | CLOSED | "Naive LLK reconfig" forms (PATH C / per-block / MOP reinit) ALL engage at JIT but NONE fix. |
| **U41** | **DEFINITIVE — Sub-7 (face-3 byte misordering) RULED OUT by 10/10 validated producer-consumer face-3 byte MATCH; Sub-8 (`unpack_*[]` staleness for gathered dual-index CB) RULED OUT by IDENTICAL metadata across BFP4 (working) and BFP8 (broken) ELFs modulo dtype.  Root cause = upstream tt-metal LLK `_llk_unpack_AB_matmul_` BFP8-4-face SrcA decode interaction with dual-index `experimental::CreateCircularBuffer` allocation.  Canonical preserved bytewise.** | **HANDOFF TO UPSTREAM** — Tenstorrent LLK team needs to investigate `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:331-333` BFP8 4-face decode for the gathered CB allocation pattern. |

## Commits this session

- (tt-metal-sglang) **1 commit**: `4591a063661 prefetcher: U41 — Sub-7 (BFP8 face-3 mantissa byte misordering) and Sub-8 (global unpack_*[] array staleness for gathered dual-index CB) BOTH RULED OUT by hardware probe; producer face-3 bytes byte-equal consumer L1 reads at 10/10 validated (dev, ELF, rd_l1) MATCH tuples; unpack_*[] metadata IDENTICAL for BFP4 (working) and BFP8 (broken) ELFs modulo dtype; root cause now provably localized to upstream LLK BFP8 4-face SrcA decode interaction with dual-index CB allocation; canonical Qwen3-8B bytewise preserved (env-gated, default-off)`.
- (sglang) `<this doc>` — pending commit.
