# Upstream bug report — tt-metal LLK BFP8 SrcA decode produces garbage when in1 CB is allocated via `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` (gathered prefetcher path)

**Date:** 2026-05-26
**Reporter:** SGLang Tenstorrent port team (fork: `predator2k/tt-metal`, branch `tenstorrent-p1`)
**Affected hardware:** 2 × Blackhole P150a in a single host (mesh shape 1×2)
**Affected codepath:** `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` (the `_llk_unpack_AB_matmul_` MOP, BFP8 4-face full-tile SrcA `else` branch at lines 333-336) when the in1 source CB is allocated through the dual-index pattern `tt_metal::experimental::CreateCircularBuffer(program, cores, remote_cfg, *global_cb)` used by `tt_metal::experimental::dram_prefetcher` + `matmul_multicore_reuse_mcast_1d`.

---

## 1. Executive summary

When `matmul_multicore_reuse_mcast_1d` is run with `use_global_cb = true` (i.e. in1 weights are streamed in by `dram_prefetcher` through a `GlobalCircularBuffer` allocated via `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)`), every BFP8 weight matmul in the model produces silently corrupted output, while every BFP4 weight matmul on the same path produces bit-correct output. Under real Qwen3-8B weights the corrupted matmul output reaches the residual stream at magnitude ~2^60-2^109, which propagates to logits at ~2e19-2e20 and NaN-poisons the sampler. The canonical (no-prefetcher) matmul path on the same hardware, with the same weights, the same compute kernel binary signature, and the same in1 dtype, produces correct output (Qwen3-8B GSM8K-10 chat = 9/10, TPOT ≈ 27 ms). Under zero-valued weights the corrupted path correctly outputs zeros (sum(x · 0) = 0), so the activation pipeline and the producer/consumer byte delivery are not at fault. After 57 dispatches we have empirically ruled out every byte-delivery, per-CB metadata, and THCON_SEC0/SEC1 cfg-register-level mechanism. The remaining surface is the silicon-level unpacker state machine (SrcA AllowedClient handshake, SrcBank/SrcRow ownership, ADC counters, and the MOP replay buffer programmed at `_llk_unpack_AB_matmul_init_` time) — none of which are reachable from kernel-side C++ or any cfg-register override.

## 2. Environment

| Item | Value |
|---|---|
| Hardware | 2 × Blackhole P150a (board ids `000004033191415e`, `0000040331914160`); mesh shape (1, 2) |
| Firmware bundle | `19.6.0.0` (cm_fw `0.28.0.0`, dm_app_fw `0.22.0.0`) |
| KMD | `TT-KMD 2.8.0` |
| Host OS | Ubuntu 24.04.4 LTS, Linux 6.17.0-23-generic, Python 3.12.3 |
| Container | `podman p3a-ngram` (in-container tt-metal build; not bind-mounted; sources sync'd via `podman cp`) |
| tt-metal HEAD | `b723bd4648c` on private fork `predator2k/tt-metal`, branch `tenstorrent-p1` |
| tt-metal base | `c0da7f83197` (upstream `tenstorrent/tt-metal` main) + **146** commits ahead (55 of them probe/diagnostic, all env-gated default-off) |
| Active tt-llk tree | `tt_metal/tt-llk/` (linked by `tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:80-87` and `tt_metal/hw/CMakeLists.txt:383-384`); `tt_metal/third_party/tt_llk/` is an orphan 2025 mirror and is NOT linked |
| Model | Qwen/Qwen3-8B (HF instruct variant, weights at `/models/Qwen3-8B`); local SGLang tt port |
| Compile-time precision | Qwen3-8B-balanced preset: FF1+FF3 fused = BFP4; WQKV, WO, FF2 = BFP8 |

## 3. Bug description

### 3.1 Data path

```
HuggingFace weights (BF16, DRAM)
        │
        ▼
ttnn.from_torch → prefetcher.insert_tensor → GlobalCircularBuffer  (per-receiver L1 buffer, 835584 B in our config)
        │  writer_l1.cpp:
        │    experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, curr_block_size_per_receiver, noc);
        │    experimental::remote_cb_push_back_and_write_pages<skip_ptr_update>(...);
        ▼
GlobalCB (L1 of every receiver core)
        │   matmul_multicore_reuse_mcast_1d_program_factory.cpp:2152-2170:
        │     remote_cb_config.remote_index(c_31).set_page_size(in1_block_size_bytes).set_data_format(in1_data_format);
        │     remote_cb_config.index(src1_cb_index).set_page_size(in1_single_tile_size).set_data_format(in1_data_format);
        │     cb_src1 = tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *global_cb);
        ▼
gathered compute kernel (bmm_large_block_zm_fused_bias_activation_gathered.cpp)
        │   mm_init/mm_block_init → matmul_block(in0_cb, in1_cb, ...)
        │   → invokes _llk_unpack_AB_matmul_(...) which issues TTI_UNPACR(SrcA, ...) for the
        │   full-tile 4-face BFP8 case via the MOP replay buffer
        ▼
mm_out_cb (BF16/BFP8 packed output) → reduce_scatter → residual → next layer
```

The same `gathered` compute kernel is JIT-built four times in a Qwen3-8B decode step — one per distinct compile-time-arg tuple (ELF):

| ELF tag | Tensor role | `in1_data_format` | `in1_single_tile_size` (B) | `in1_block_size_bytes` (B) | `in1_block_num_tiles` | K | per_core_N | Behaviour |
|---|---|---|---:|---:|---:|---:|---:|---|
| `0x2db6e` | FF1+FF3 fused | **BFP4** (0x7) | **576** | 13824 | 24 | 128 | 6 | **CLEAN** — bit-correct output |
| `0x2d96e` | WQKV          | **BFP8** (0x6) | **1088** | 13056 | 12 | 128 | 3 | **GARBAGE** — ~2^60-2^109 magnitude |
| `0x2de6b` | WO            | **BFP8** (0x6) | **1088** |  8704 |  8 | 128 | 2 | **GARBAGE** |
| `0x2df6a` | FF2           | **BFP8** (0x6) | **1088** | 26112 | 24 | 192 | 4 | **GARBAGE** |

Sole single-CT-arg discriminator across the 4 ELFs: `in1_single_tile_size` (= dtype). All other shape parameters (`in0_block_w`, `num_blocks=32`, `out_subblock_h=1`, `batch=1`, `per_core_M=1`, `ring_size=32`, `num_cores=32`, `spill=0`, `dst_full_sync_en=0`) are either shared with at least one garbage ELF or trivially the same across all 4.

### 3.2 Symptom

With prefetcher ON and a single `"What is 2+2?"` decode (Qwen3-8B, greedy, max_new_tokens=15):

- **Expected** (canonical, no prefetcher): `" What is 2+2? What is 2+2? What"`
- **Actual** (prefetcher on): a sequence of unrelated tokens (the exact pattern depends on which env-gated probes are active because each new define triggers a fresh JIT binary hash; we have observed 8 distinct garbage signatures across U37-U41 interventions, e.g. `" WhatGBT roz无所谓我以为GBT烘干..."`) and on continuation past token ~2 the sampler returns NaN logits and crashes.

Magnitude of the matmul output, measured at PACK exit (U30 Path A): 65-86% of tiles contain Inf/NaN; mean absolute magnitude ~2^60-2^109. BFP4 ELF: <0.1% Inf/NaN.

### 3.3 What works

- Canonical Qwen3-8B (no prefetcher): bit-exact output, GSM8K-10 chat = 9-10/10, TPOT ≈ 27 ms.
- The BFP4 ELF (`0x2db6e`, FF1+FF3 fused) on the prefetcher path: bit-correct output (verified bytewise vs reference).
- Galaxy Llama-3 70B demo (`models/demos/llama3_70b_galaxy/`) uses the same `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` pattern at scale and is reported correct. Their config presumably differs in a way that avoids whatever silicon regime the BFP8 4-face decode lands in here; a diff between their gathered-matmul config and ours is a strong next investigation step.

### 3.4 What is broken

Every BFP8 matmul on the gathered (`use_global_cb=true`) path — only when in1 is sourced through `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)`. Same BFP8 weights, same compute kernel, same hardware, fed by the plain `CreateCircularBuffer(prog, cores, src1_cfg)` path: correct.

## 4. What was conclusively ruled out

Each item below is one of `[H]` hardware-empirical or `[S]` source-static. Probe identifiers refer to env-gated DPRINT instrumentation that ships in branch `tenstorrent-p1` (see §7).

| # | Eliminated suspect | Evidence | Dispatch |
|---|---|---|---|
| 1 | Producer write skew (NoC delivery wrong) | `[H]` `SGLANG_TT_U37_PROD_BYTES` (writer_l1.cpp:123) and `SGLANG_TT_U37_READ_BYTES` (gathered compute kernel, in-tree): producer bytes byte-equal consumer L1 reads at the matching `(device, wr_ptr, rd_l1)` tuples for **5/5** tensors of layer-0 decode step 1 (WQKV, WO, W1, W3, W2; see U37 table). | U37 |
| 2 | Post-producer L1 stomp (downstream kernel overwriting cb_in1) | `[H]` Same probes — consumer reads observed *immediately before* `matmul_block` fires; bytes are still the producer's. Also U17-U28 ruled out every dataflow/dispatcher/fabric/EDM/RS reader/writer/W2 dataflow kernel suspect individually. | U17-U28, U37 |
| 3 | Cross-subdevice WAIT_STREAM race (W2 → RS handshake) | `[H]` U29 Phase 3 — installed `GlobalSemaphore`-backed handshake; sentinel `0xdeadbeef` confirmed received in 2880/2880 iterations. Garbage unchanged. | U29 |
| 4 | DST register stale state | `[H]` U9, U39 (Suspect 4 `SGLANG_TT_U39_FORCE_FP32_DEST_OFF=1` flipped `fp32_dest_acc_en` uniformly to 0). Garbage signature shifted (proving the override engaged) but BFP8 still produces NaN. | U9, U39 |
| 5 | L1 allocator overlap of GCB region | `[H]` U25, U26, U27 — adjacent-buffer ownership probe + firmware-level per-kernel L1 0xa6700 entry/exit probe: the suspect address is constant pre-existing data; no kernel writes it. | U25-U27 |
| 6 | BFP8 face-3 mantissa byte misordering (bytes 832-1087 of the 1088 B tile) | `[H]` `SGLANG_TT_U41_FACE3_PROBE` — extended U37 probes with face-3 reads at byte offsets 832, 1024, 1080. Cross-correlation `/tmp/u41_xcorr3.py`: **10/10 validated MATCH** for producer face-3 bytes vs consumer L1 face-3 bytes across all 5 tensors / both devices / 3 face windows. The BFP4 path also matched 10/10 — so face-3 byte ordering is a true non-issue regardless of dtype. | U41 |
| 7 | Per-CB unpack metadata staleness (`unpack_tile_r_dim[]`, `unpack_face_r_dim[]`, `unpack_num_faces[]`, `unpack_partial_face[]`, `unpack_narrow_tile[]`, `unpack_tile_c_dim[]`, `unpack_num_faces_r_dim[]`, `unpack_num_faces_c_dim[]`, `unpack_src_format[]`, `unpack_dst_format[]`) | `[H]` `SGLANG_TT_U41_UNPACK_ARR_PROBE` — read all 10 arrays for in0 + in1 on every ELF. All 4 ELFs report `tile_r=32 tile_c=32 face_r=16 num_faces=4 partial=0 narrow=0`. The only discriminator is the legitimate `in1_src_fmt/dst_fmt` (BFP8=0x6 vs BFP4=0x7). U38's `.set_tile_dims(src1_cb_index, in1_tile)` on `remote_cb_config` propagates correctly through the dual-index allocation. | U38, U41 |
| 8 | `mm_init` / `mm_block_init` / `set_tile_dims` propagation gap | `[H]` U38 added the missing `.set_tile_dims(src1_cb_index, in1_tile)` (env-gated `SGLANG_TT_U38_SET_TILE_DIMS`); U41 verified it propagated into the global arrays. Necessary; not sufficient. | U38, U41 |
| 9 | Unpacker `WrapAddr` (`Unpack_limit_address` / `Unpack_fifo_size`) per UNPACR_Regular.md:199-207 | `[S]` U42 exhaustive grep across `/home/mhnie/tt-metal-sglang/`: only writer is `tests/tt_metal/.../writer_config_reg.cpp:75` (a config-write unit test). `unpack_config_t` zero-inits both (`tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:851-852`, comments "Set dynamically" but no caller does). `[H]` U43: runtime DPRINT of `THCON_SEC0_REG2_Unpack_limit_address_ADDR32=74` and `_Unpack_fifo_size_ADDR32=75` on every ELF — both are 0 at HW level. WrapAddr reduces to `addr -= 0` for every positive L1 address. | U42, U43 |
| 10 | Multi-context format override (`REG2_Ovrd_data_format`, `REG7_Unpack_data_format_cntx0`) | `[H]` U43: both = 0 on every ELF (BFP4 and BFP8 alike). InDataFormat is sourced from TileDescriptor word 0 bits[3:0] per `UNPACR_Regular.md:86-92` (the `else` branch). | U43 |
| 11 | `REG2_Force_shared_exp` accidentally set | `[H]` U43: = 0 on every ELF. Unpacker reads per-tile shared exp from L1 (correct). | U43 |
| 12 | `TileDescriptor.NoBFPExpSection` | `[H]` U43: = 0 on every ELF — exponent section IS allocated for both BFP4 and BFP8 per `UNPACR_Regular.md:128-132`. | U43 |
| 13 | `TileDescriptor.DigestSize` / `IsUncompressed` / `ZDim` (num_faces) | `[H]` U43: `DigestSize=0`, `IsUncompressed=1`, `ZDim=4` on every ELF (BFP4 and BFP8 identical). | U43 |
| 14 | `REG3_Base_address` / `REG7_Offset_address` arithmetic mismatch | `[H]` U43: `REG7_Offset_address=0` on every ELF; `REG3` base differs per-core legitimately (shared-L1 reservation per core), but the *shape* of the address arithmetic is identical across ELFs. | U43 |
| 15 | LLK reconfig variants (`reconfig_data_format_srca/srcb`, MOP reinit, per-block reconfig) | `[H]` U40 Paths C / per-block / MOP reinit (`SGLANG_TT_U40_FORCE_UNPACK_RECONFIG`, `SGLANG_TT_U40_RECONFIG_BLOCK`): each one engages at the JIT binary (proven by distinct garbage signature shifts) but none fix BFP8. U43 explains why: the post-reconfig cfg state is already what the reconfig would write. | U40, U43 |
| 16 | LocalCBInterface `fifo_page_size` corruption | `[H]` U39 (Suspect 6) `SGLANG_TT_U39_CB_META` dumps `fifo_page_size` from the local CB interface: 1088 for BFP8, 576 for BFP4, 2048 for BF16 — all correct. | U39 |
| 17 | BFP8 face-stride beyond byte 256 for faces 0/1/2 | `[H]` U39 (Suspect 5) `SGLANG_TT_U39_EXT_BYTES` — extended U37 byte dumps from 16 to 64 B; face-0/1/2 boundaries (bytes 64, 320, 576) match producer-to-consumer for all 5 tensors. | U39 |
| 18 | Dispatcher / launch_msg / kernel_config writes overwriting GCB region | `[H]` U22 Path A traced dispatcher writes; none land in the suspect range. | U22 |
| 19 | Prefetcher writer_l1 self-stomp | `[H]` U18 / U21 addr probes (`SGLANG_TT_U18_PREFETCHER_WRITE_PROBE`, `SGLANG_TT_U21_PREFETCHER_ADDR_PROBE`): writer never writes outside its tensor's intended slot. | U18, U21 |
| 20 | Fabric / EDM ack writes | `[H]` U24 fabric_erisc_router addr probe rules out writes into the suspect L1 range. | U24 |

### Cross-correlation byte-match table (U37 — the killer evidence for delivery)

Worker core (device=0, x=1, y=5), UNPACK RISC FIRST READ, layer 0 decode step 1:

| ELF | Tensor | rd_l1 | Kernel-read first 16 B (4 dwords) | Producer write source | Match |
|---|---|---|---|---|:---:|
| `0x2de6b` | WQKV | `0xac700` | `7a7a7a7a 7a7a797a 797a7b79 7a7a7979` | `7a7a7a7a 7a7a797a 797a7b79 7a7a7979` | YES |
| `0x2df6a` | WO   | `0x112700` | `7a7a7a7b 7a7a7a7a 7a7a7b7a 7a7a797a` | `7a7a7a7b 7a7a7a7a 7a7a7b7a 7a7a797a` | YES |
| `0x2db6e` | W1   | `0x158900` | `7a7a7a7a 7a7a7a7a 7a7a7a78 7b7a7b7a` | `7a7a7a7a 7a7a7a7a 7a7a7a78 7b7a7b7a` | YES |
| `0x2db6e` | W3   | `0xfa100`  | `7a7a7a7b 7b7b7b7a 7a7b7a78 7b7a7a7a` | `7a7a7a7b 7b7b7b7a 7a7b7a78 7b7a7a7a` | YES |
| `0x2d96e` | W2   | `0x16bb00` | `7b7a7a7a 7a7a7a7b 7a7b7a7a 7a7b7a7a` | `7b7a7a7a 7a7a7a7b 7a7b7a7a 7a7b7a7a` | YES |

### Per-ELF THCON_SEC0 cfg snapshot (U43 — the killer evidence for the cfg-register surface)

Captured live via `SGLANG_TT_U43_CFG_DUMP=1` (92,159 DPRINT lines across all worker cores; 258 unique `(elf, td, cfg)` tuples after dedup; per-(elf, cfg-shape) signature invariant across cores).

| ELF | Tensor | InDataFormat | TileDesc[0] | TileDesc[1] (Y/Z) | TileDesc[3] (Digest) | REG2 cfg[0] | REG2 cfg[1] | REG2 cfg[2] (limit_addr) | REG2 cfg[3] (fifo_size) | REG7 off/fmt |
|---|---|---:|---|---|---:|---|---|---:|---:|---:|
| `0x2db6e` | W1+W3 (**BFP4 CLEAN**) | **0x7** | 0x17 | 0x40001 | 0 | 0x27 | 0xf000f | **0** | **0** | 0 |
| `0x2d96e` | WQKV  (**BFP8 GARBAGE**) | **0x6** | 0x16 | 0x40001 | 0 | 0x26 | 0xf000f | **0** | **0** | 0 |
| `0x2de6b` | WO    (**BFP8 GARBAGE**) | **0x6** | 0x16 | 0x40001 | 0 | 0x26 | 0xf000f | **0** | **0** | 0 |
| `0x2df6a` | FF2   (**BFP8 GARBAGE**) | **0x6** | 0x16 | 0x40001 | 0 | 0x26 | 0xf000f | **0** | **0** | 0 |

The three BFP8 ELFs have **byte-identical** THCON_SEC0 cfg state (modulo legitimate per-core REG3 base addresses). The BFP4 ELF differs only in the legitimate `InDataFormat`/`Out_data_format` bits (0x7 vs 0x6). The bug is invariant under the entire kernel-side-visible cfg-register surface.

## 5. Suspected mechanism

### 5.1 Suspect code

`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` lines 333-336 — the BFP8 4-face full-tile SrcA `else` branch inside `_llk_unpack_AB_matmul_`:

```cpp
        else
        {
            if (unpA_partial_face) {
                // … partial-face path: TTI_UNPACR_NOP + 2× TTI_UNPACR + TTI_SETADCZW
            }
            else
            {
                TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/,
                           p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);   // line 335
            }
        }

        TT_MOP(0, (reuse_a ? ct_dim : rt_dim) - 1, unp_cfg_context == 0 ? 0 : 0xff);            // line 339
```

The full-tile non-partial-face case emits a single `TTI_UNPACR(SrcA, ...)` and lets the unpacker's internal state machine and MOP replay buffer fan it out across all 4 faces. This is the only branch that fires for our BFP4 (clean) ELF AND for all three BFP8 (garbage) ELFs.

### 5.2 Allocation site (the "gathered" / `use_global_cb` path)

`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:2152-2170`:

```cpp
if (use_global_cb) {
    uint32_t in1_block_size_bytes = in1_single_tile_size * in1_block_num_tiles;
    tt_metal::CircularBufferConfig remote_cb_config =
        tt_metal::CircularBufferConfig((global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes);
    remote_cb_config.remote_index(remote_cb_index)
        .set_page_size(in1_block_size_bytes)
        .set_data_format(in1_data_format);
    remote_cb_config.index(src1_cb_index)
        .set_page_size(in1_single_tile_size)
        .set_data_format(in1_data_format);
    // U38 (SGLang) optionally adds: .set_tile_dims(src1_cb_index, in1_tile)
    cb_src1 = tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *global_cb);
}
```

The "dual-index" pattern: `remote_index(c_31)` programs the remote-write end (driven by the prefetcher), and `index(src1_cb_index)` programs the local-read end (read by the unpacker). The single backing buffer is `*global_cb`, a `GlobalCircularBuffer` allocated by `dram_prefetcher_program_factory.cpp:169`:

```cpp
tt::tt_metal::experimental::CreateCircularBuffer(program, reader_core_range, remote_cb_config, global_cb);
```

with `remote_cb_size = global_cb.size()` and an initial `set_page_size(L1_ALIGNMENT=16)` that is later runtime-resized per tensor via `writer_l1.cpp:95`:

```cpp
experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, curr_block_size_per_receiver, noc);
```

before each `experimental::remote_cb_push_back_and_write_pages<>` (writer_l1.cpp:236).

### 5.3 Hypothesis

After 57 dispatches, every byte at the consumer L1 (U37 + U41), every per-CB `unpack_*[]` metadata field (U41), and every THCON_SEC0/SEC1 unpacker cfg register visible from the kernel side (U43) is provably correct or benign. The bug must therefore live in unpacker state that is NOT capturable from cfg-register reads or per-CB metadata. The most plausible candidates, in order:

1. **SrcA AllowedClient handshake / SrcBank ownership under the BFP8 4-face fast path.** Per `UNPACR_Regular.md:354-356`, the unpacker spins on `SrcA[Bank].AllowedClient == SrcClient::Unpackers` before issuing decoded datums. The MOP at line 339 fans out a packed `TTI_UNPACR(SrcA, ...)` programmed at `_llk_unpack_AB_matmul_init_` time. If the replay-buffer entries were programmed for a tile-stride that matches BFP4 (576 B) but is wrong for BFP8 (1088 B) under the dual-index allocation's L1 reservation, the unpacker would issue fetches with the wrong stride. BFP4 happens to "work" because its 576 B tile fits inside the BFP8 1088 B address window (any over-read aliases into the next tile's exp prefix, which decodes harmlessly for the small offset).

2. **ADC counter state leakage.** Per `UNPACR_Regular.md:139-140` the ADCs track input position. `_llk_unpack_AB_matmul_` does NOT reset ADCs between consecutive matmul invocations within the same MOP loop. If an earlier matmul left an ADC at an offset valid for BFP4 but stale for BFP8, the next BFP8 matmul would compute `FirstDatum` from a wrong base. ADCs are not cfg registers and are not capturable from kernel-side cfg-reads.

3. **MOP replay buffer programmed at init-time captures the first ELF's tile geometry.** `mm_block_init` runs once per kernel binary launch. If the JIT-compiled binary is shared across all 4 ELFs (because the dual-index allocator places them in a single pool), the replay-buffer entries baked at init time might encode the first-launched ELF's `WhichUnpacker` + `Ch0/Ch1` increments. Subsequent ELFs with different `in1_block_size_bytes` would issue UNPACR instructions to wrong addresses.

(Candidate 1 is the strongest fit to the dtype-only asymmetry plus the byte-equality / cfg-equality results.)

The Galaxy Llama-3 70B demo (`models/demos/llama3_70b_galaxy/`) uses the same `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` allocation pattern at scale with BFP8 weights, reportedly without this corruption — strongly suggesting some specific config knob (tile-stride, subblock dims, ring_size, num_cores) keeps the silicon in a regime that avoids the failing path. A side-by-side diff of their gathered-matmul config against ours is the most economical first probe Tenstorrent could run.

## 6. Reproducer

### 6.1 In SGLang (what we observed)

In a host with 2 × Blackhole P150a + tt-metal-sglang `b723bd4648c` installed in the container (default-off probes; no probe envs needed to reproduce the bug — the probes are only for instrumenting it):

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c 'CACHE=/root/.cache/tt-metal-cache; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

# BROKEN: prefetcher ON
podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/repro_broken.log 2>&1'

# wait for /health, then:
curl -sX POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Actual: garbage tokens, then NaN sampler crash on continuation.
```

Control (canonical — same command minus `SGLANG_TT_USE_PREFETCHER=1`):

```bash
podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server ... > /tmp/repro_clean.log 2>&1'
curl -sX POST http://127.0.0.1:30000/generate -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Expected: " What is 2+2? What is 2+2? What"
```

GSM8K-10 chat on canonical = 9-10/10; on prefetcher = 0/10 (NaN sampler crash within question 1).

### 6.2 Minimal standalone tt-metal test (strongly recommended next deliverable)

We do not currently have a standalone reproducer that does not depend on SGLang + Qwen3-8B. The cleanest place to build one is `tests/tt_metal/tt_metal/test_kernels/` using the existing `dram_prefetcher` op + `matmul_multicore_reuse_mcast_1d`. Outline:

```cpp
// Pseudocode — Tenstorrent LLK test infra is the best place to flesh this out.

// 1. Bring up a (1, 2) Blackhole P150a mesh.
auto mesh = ttnn::MeshDevice::create({1, 2});

// 2. Construct a GlobalCircularBuffer sized like Qwen3-8B-balanced (835584 B/receiver works;
//    smaller sizes likely also reproduce as long as in1_single_tile_size = 1088 and
//    `(gcb.size() / in1_block_size_bytes) * in1_block_size_bytes == gcb.size()`).
auto global_cb = ttnn::experimental::CreateGlobalCircularBuffer(mesh, /*size=*/835584, …);

// 3. Insert at least ONE BFP8 weight tile filled with KNOWN, NON-ZERO bytes (e.g. all 0x7a).
//    Use dram_prefetcher to stream it into the GCB.
auto W = ttnn::from_torch(torch::full({32, 32}, 0.5, torch::kBFloat16),
                          ttnn::DataType::BFLOAT8_B, /*tile_layout*/ true);
ttnn::experimental::dram_prefetcher::insert_tensor(global_cb, W);

// 4. Run a single gathered matmul:
auto out = ttnn::matmul(
    /*in0=*/ones_like_activations,
    /*in1=*/W,
    /*program_config=*/MatmulMultiCoreReuseMultiCast1DProgramConfig{
        /*compute_with_storage_grid_size=*/...,
        /*in0_block_w=*/4, /*out_subblock_h=*/1, /*out_subblock_w=*/2,
        /*per_core_M=*/1, /*per_core_N=*/2, /*fuse_batch=*/true, /*mcast_in0=*/true,
    },
    /*use_global_cb=*/true);

// 5. Compare to torch reference.
auto ref = torch::matmul(ones_torch, W_torch);   // ~32 (sum of 1 × 0.5 across 64 cols)
auto got = ttnn::to_torch(out);                  // expected: ~32; actual: ~2^60
EXPECT_NEAR(got.abs().max().item<float>(), 32.0, 1e-2);   // FAILS
```

Expected: with BFP8 in1, the matmul output has magnitude many orders of magnitude larger than the torch reference. With BFP4 (change `from_torch` dtype + page_size to 576), the assertion passes.

If a standalone reproducer is helpful for Tenstorrent's internal investigation we can collaborate on building one against the team's preferred test scaffold.

## 7. Probe inventory (env-gated, default-off; in tree on branch `tenstorrent-p1`)

All probes default to off, and the canonical (no-prefetcher) path is bytewise-identical with all probe envs unset. Each `SGLANG_TT_U*` variable below activates the corresponding DPRINT block via the matmul factory at `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2536-2640` and/or the prefetcher writer factory at `dram_prefetcher_program_factory.cpp:214-274`.

| Env var | What it dumps | Source location |
|---|---|---|
| `SGLANG_TT_U37_PROD_BYTES` | Producer's 16-byte source at each `remote_cb_push_back_and_write_pages` call, layer 0 only. | `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp:123` |
| `SGLANG_TT_U37_READ_BYTES` | Consumer's 16-byte L1 read at `cb_in1.fifo_rd_ptr * L1_ALIGNMENT` right before `matmul_block`, gated to `(ring_idx=0, b=0, block=0, in0/in1_subblock=0, inner_dim_idx=0)`. | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` |
| `SGLANG_TT_U39_EXT_BYTES` | Extends U37 to 64 bytes per probe — covers face-0 exp prefix + face-1 mantissa start (byte 256-byte 576). | same files |
| `SGLANG_TT_U39_CB_META` | Dumps `LocalCBInterface.fifo_page_size`. | gathered compute kernel |
| `SGLANG_TT_U39_FORCE_FP32_DEST_OFF` | Unconditionally forces `fp32_dest_acc_en = 0` (probes Suspect 4). | matmul factory line 2725 |
| `SGLANG_TT_U40_PROBE_TILE_DIMS` | Per-(ring, block) DPRINT of in0/in1 `tile_r_dim`, `tile_c_dim`, `face_r_dim`, `num_faces`, `partial`, `narrow`, `src_fmt`, `dst_fmt`. | gathered compute kernel |
| `SGLANG_TT_U40_FORCE_UNPACK_RECONFIG` | Path C — emit a `reconfig_data_format_srca/srcb` after `mm_block_init`. | gathered compute kernel |
| `SGLANG_TT_U40_RECONFIG_BLOCK` | Most aggressive — full unpack reconfig per block. | gathered compute kernel |
| `SGLANG_TT_U41_FACE3_PROBE` | Extends U37/U39 dumps with face-3 mantissa reads at bytes 832, 1024, 1080. | writer_l1.cpp:201 + gathered compute kernel |
| `SGLANG_TT_U41_UNPACK_ARR_PROBE` | DPRINT the full per-CB `unpack_tile_r_dim[]` / `unpack_tile_c_dim[]` / `unpack_face_r_dim[]` / `unpack_num_faces[]` / `unpack_partial_face[]` / `unpack_narrow_tile[]` / `unpack_num_faces_r_dim[]` / `unpack_num_faces_c_dim[]` / `unpack_src_format[]` / `unpack_dst_format[]` arrays for in0 + in1. | gathered compute kernel |
| `SGLANG_TT_U43_CFG_DUMP` | Runtime DPRINT of THCON_SEC0 + THCON_SEC1 `REG0_TileDescriptor`, `REG2_*` (4 words including `Out_data_format` / `Throttle_mode` / `Force_shared_exp` / `Ovrd_data_format` / `Unpack_limit_address` / `Unpack_fifo_size`), `REG3_Base_address`, `REG7_Offset_address` / `Unpack_data_format_cntx0` immediately after `mm_block_init` → `_llk_unpack_hw_configure_`. | gathered compute kernel; matmul factory line 2637 |
| `SGLANG_TT_U38_SET_TILE_DIMS` | Adds the missing `.set_tile_dims(src1_cb_index, in1_tile)` call on `remote_cb_config` (NECESSARY metadata propagation, not sufficient). | matmul factory line 2166-2169 |
| `SGLANG_TT_U32_GCB_OFFSET` / `SGLANG_TT_U33_FACTORY_OFFSET` / `SGLANG_TT_U34_LCM_ALIGN` / `SGLANG_TT_U35_PER_LAYER_OFFSET` | Per-tensor GCB byte-offset plumbing (Python → C++ rt-arg → kernel `update_local_cb_rd_ptr`) — necessary plumbing under the dual-index layout, not sufficient for correctness. | `prefetcher.py`, matmul factory, gathered compute kernel |
| `SGLANG_TT_U18_PREFETCHER_WRITE_PROBE` / `SGLANG_TT_U21_PREFETCHER_ADDR_PROBE` / `SGLANG_TT_U25_RCB_PROBE` | Earlier downstream-stomper probes (ruled out). | writer_l1.cpp lines 40, 46, 255 |
| `SGLANG_TT_U31_FACTORY_ARGS` | Dumps per-ELF compile-time args at `CreateKernel` time on the gathered matmul factory. | matmul factory line 2589 |

DPRINT capture envs (provided by tt-metal itself, not added by us): `TT_METAL_DPRINT_CORES=all` and `TT_METAL_DPRINT_FILE=/tmp/x.log` (note: `TT_METAL_DPRINT_FILE_NAME` is NOT a valid env name; the correct one is `TT_METAL_DPRINT_FILE` per `rtoptions.cpp:1334,1346`).

## 8. Workarounds attempted (none fixed it)

| Attempt | Outcome |
|---|---|
| Cross-subdevice `GlobalSemaphore` handshake (U29 Phase 3) | Sentinel handshake correct, garbage unchanged. |
| Per-tensor GCB byte offset (U32-U35) | Necessary plumbing for the per-layer write pattern; garbage unchanged. |
| `.set_tile_dims(src1_cb_index, in1_tile)` on `remote_cb_config` (U38) | Garbage signature shifted (proves the JIT engaged); BFP8 still NaN. |
| Force `fp32_dest_acc_en = 0` uniformly (U39 Suspect 4) | Garbage signature shifted; BFP8 still NaN. |
| `reconfig_data_format_srca/srcb` post-`mm_block_init` (U40 Path C) | Garbage signature shifted; BFP8 still NaN. |
| Full unpack reconfig per block (U40 Path block-reconfig) | Garbage signature shifted; BFP8 still NaN. |
| `Unpack_limit_address = UINT32_MAX` (would disable WrapAddr) | Skipped — WrapAddr is already de-facto disabled (both regs = 0 per U42 + U43); forcing nonzero would only change behavior, not fix it. |
| Drop the per-matmul `(gcb_size / in1_block_size_bytes) * in1_block_size_bytes` rounding (U31 kernel-fix) | Fails at allocation: `Total circular buffer size 835584 B must be divisible by page size 13824 B` (`circular_buffer_config.cpp:150`). |

## 9. Suggested next debugging steps for Tenstorrent

1. **Galaxy-diff.** Side-by-side the `dram_prefetcher` + `matmul_multicore_reuse_mcast_1d` config used by `models/demos/llama3_70b_galaxy/` (which works with BFP8 + `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)`) vs the four Qwen3-8B configs above. Likely differences live in `in0_block_w`, `out_subblock_w`, `in1_num_subblocks`, `ring_size`, or `num_cores`. The single CT-arg that makes Galaxy land in a working silicon regime would be the key.
2. **Standalone tt-llk reproducer.** Build the minimal-reproducer outline in §6.2 inside `tt-llk/tests/` (or wherever the LLK team's preferred test scaffold lives). Once reproducible without SGLang dependencies, it can be bisected independently.
3. **tt-llk bisection.** Bisect against `tt-llk` commit history: does an older revision handle BFP8 + the dual-index `remote_cfg + *global_cb` allocation correctly? The active path is `tt_metal/tt-llk/tt_llk_blackhole/` (NOT the orphan `tt_metal/third_party/tt_llk/`); both contain the same structural `else { TTI_UNPACR(SrcA, ...); }` branch.
4. **Tensix ISA disasm comparison.** Disassemble the UNPACR microcode actually emitted on the broken (gathered) vs working (canonical) paths, against the same in1 BFP8 weights. The MOP replay buffer programmed at `_llk_unpack_AB_matmul_init_` time is the highest-value target — its replay-buffer contents are not user-visible at runtime today but should be inspectable from a debugger.
5. **Tensix debug-status / ADC inspection.** After the failing matmul fires, read `TENSIX_DEBUG` + ADC channel registers to confirm or refute Hypothesis 5.3(2) (ADC counter leakage).
6. **SrcA AllowedClient trace.** Per `UNPACR_Regular.md:354-356`, the unpacker spins on `SrcA[Bank].AllowedClient`. If silicon trace shows the unpacker proceeding before the bank is truly owned (race on the `ALLOW_CLIENT` write from MATH), that would directly evidence Hypothesis 5.3(1).
7. **REG10 (newer enable-gated WrapAddr) sanity-check.** `THCON_SEC0_REG10_Unpack_limit_address_ADDR32=104` + `_Unpack_fifo_size_ADDR32=105` + `_Unpack_limit_address_en_ADDR32=105:17` were also confirmed by U42 to never be written. If a future silicon design relies on REG10 being explicitly programmed, this would surface here.

## 10. Reproducing the diagnostic harness — commit + push

| Repo | Where to fetch the probes | Branch | HEAD |
|---|---|---|---|
| tt-metal (private fork) | `git@github.com:predator2k/tt-metal.git` | `tenstorrent-p1` | `b723bd4648c` (146 commits ahead of upstream main `c0da7f83197`; 55 are probes/diagnostic, all env-gated default-off) |
| sglang (private fork) | `git@github.com:predator2k/sglang.git` | `tenstorrent-p1` | this doc + `99f90922a` (U43) |

We can push these forks to a public mirror, or extract the probe patch as a single diff against upstream tt-metal `main`, on request. The full investigation ledger (29 docs, S1-S10 → U1-U43) lives under `sglang/docs/platforms/tt_qwen3_8b_prefetcher_*.md`.

Key in-tree files (all on tt-metal-sglang `tenstorrent-p1`):

- Suspect LLK code — `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:333-336`
- Dual-index CB allocator — `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:2152-2170`
- Prefetcher GlobalCB writer — `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp:160-169` and `kernels/writer_l1.cpp:95-236`
- ISA spec we relied on — `sglang/docs/platforms/tt-isa-docs/UNPACR_Regular.md` (lines 86-92, 128-132, 139-140, 199-207, 304-313, 354-356)
- Format encoding table — `sglang/docs/platforms/tt-isa-docs/Unpackers_FormatConversion.md`

## 11. Contact / next steps

**Reporter contact:** TBD — to be filled in by the user before handoff.
**Preferred channel:** GitHub issue on `tenstorrent/tt-metal`, or direct email to the LLK team if that is preferred for hardware-bug reports of this depth.
**Hardware availability:** the 2 × Blackhole P150a setup that produced every empirical result here is still online; we can run additional probes on request (turn-around ~30-45 min per env-gated probe iteration, dominated by full tt-metal rebuild + JIT spawn warmup in the p3a-ngram container).
**Repo access:** we can grant Tenstorrent LLK engineers read access to `predator2k/tt-metal-sglang` and `predator2k/sglang` (currently private). The probe diff against upstream tt-metal main can also be sent as a single patch on request.

---

### Open uncertainties (called out for the LLK team)

- **5.3(1) vs 5.3(2) vs 5.3(3).** We have ruled out everything reachable from the kernel-side cfg surface, but we cannot rank the three remaining silicon-state hypotheses without access to Tensix debug-status / ADC reads. The asymmetry of "BFP4 works, BFP8 garbage, every cfg register identical" most strongly fits (1) (tile-stride mismatch in the MOP replay buffer), but (2) and (3) cannot be excluded.
- **Galaxy vs Qwen3-8B config delta.** We assert that the Llama-3 70B Galaxy demo uses the same dual-index CB allocation pattern with BFP8 weights and is correct — based on code-read of `models/demos/llama3_70b_galaxy/`. We have not run it ourselves on our hardware (different mesh shape requirement), so this assertion is source-static, not hardware-verified.
- **Container-level JIT spawn race.** Two of two GSM8K(10) attempts under U41/U42 hit `posix_spawn: Operation not permitted` on the `lto-wrapper` build for `untilize_wh`. This is an unrelated p3a-ngram container limit, not a bug in tt-metal, and not believed to interact with the LLK BFP8 bug. Reported here for completeness so it doesn't show up as "missing baseline" when Tenstorrent re-runs.
- **Standalone reproducer not yet built by us.** §6.2 is an outline only. If the LLK team would prefer we collaborate on flesh-out, we can — but we believe the failing path is internal enough that Tenstorrent's existing `tests/ttnn/unit_tests/operations/matmul/test_matmul_1d.cpp` or `tt-llk/tests/llk_matmul/` infrastructure is the better starting point.
