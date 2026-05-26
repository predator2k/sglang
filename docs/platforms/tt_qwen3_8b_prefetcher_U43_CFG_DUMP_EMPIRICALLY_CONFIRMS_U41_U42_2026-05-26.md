# TT Qwen3-8B prefetcher — U43 ships an env-gated `SGLANG_TT_U43_CFG_DUMP` runtime DPRINT of the suspect THCON_SEC0/SEC1 unpacker cfg registers (TileDescriptor + REG2 unpack_config + REG3 base + REG7 offset/cntx-fmt) inside the gathered compute kernel right after `mm_block_init` runs `_llk_unpack_hw_configure_`, captures ALL FOUR ELFs (BFP4-clean `0x2db6e` = W1+W3 fused, BFP8-garbage `0x2d96e` = WQKV, `0x2de6b` = WO, `0x2df6a` = W2) on a single live prefetcher run against Qwen3-8B on 2× Blackhole p150a (output `" What\tService_pressition"` — confirmed prefetcher garbage signature) — and **EMPIRICALLY CONFIRMS U42's static-analysis verdict**: `Unpack_limit_address = 0` and `Unpack_fifo_size = 0` on EVERY ELF at runtime (WrapAddr is provably de-facto disabled per the ISA functional model); `REG2_Force_shared_exp = 0`; `REG2_Ovrd_data_format = 0`; `REG7_Unpack_data_format_cntx0 = 0` (no per-context format override); `TileDescriptor.NoBFPExpSection = 0` on both BFP4 and BFP8 paths; `TileDescriptor.DigestSize = 0`; `TileDescriptor.IsUncompressed = 1`; `TileDescriptor.ZDim = 4` (num_faces) — IDENTICAL across BFP4-clean and BFP8-garbage ELFs. The ONLY discriminator across BFP4 vs BFP8 ELFs is the legitimate per-tensor `InDataFormat` (0x7=BFP4 vs 0x6=BFP8) + `Out_data_format` (matching) — exactly the values the data should carry. The 3 BFP8 ELFs (WQKV/WO/W2) have IDENTICAL THCON_SEC0 cfg state (modulo per-core base addresses) yet ALL produce garbage; the BFP4 ELF (W1+W3) has THE SAME state modulo the legitimate format-bit difference. Every cfg-register-level discriminator hypothesis is REFUTED. Bug is silicon-level: invariant under the entire set of THCON_SEC0/SEC1 cfg registers — must live in the unpacker's internal state machine (SrcA/SrcB AllowedClient handshake, SrcBank/SrcRow ownership, ADC counters) which IS NOT capturable from cfg-register reads. Active-vs-orphan LLK tree diff documented (U41/U40 line references to `third_party/tt_llk/...:331-335` correspond to the ORPHAN 2025 tree; the active `tt-llk/...:333-336` carries the SAME structural code with an added 16x16 assert and the `_llk_unpack_configure_addresses_(address_b, address_a, cfg)` call) — same bug surface in both trees. tt-metal-sglang commit pending; sglang commit pending; canonical Qwen3-8B baseline UNDISTURBED (probe is env-gated default-off AND `use_global_cb`-only).

Status: **EVIDENCE_ADVANCE — every cfg-register-level discriminator HYPOTHESIS REFUTED EMPIRICALLY; bug confirmed to live BELOW the cfg-register API surface (silicon state machine); production shipping config remains canonical `tt_transformers_paged` (no prefetcher). U41 verdict (upstream tt-metal LLK BFP8 4-face SrcA decode × dual-index CB allocation, NOT addressable from sglang's plug-in surface) stands.**

Continuation of `tt_qwen3_8b_prefetcher_U42_WRAP_HYPOTHESIS_REFUTED_BY_STATIC_ANALYSIS_2026-05-25.md`.

tt-metal-sglang HEAD: **`4591a063661`** + this U43 probe (pending commit).
sglang HEAD: **`556e99311`** + this doc (pending commit).

## TL;DR per-ELF runtime cfg-register table

Captured live via env-gated `SGLANG_TT_U43_CFG_DUMP=1` against the running gathered prefetcher kernels on Qwen3-8B / Blackhole p150a / `tt_transformers_paged`. All 4 ELFs hit; **92,159** total U43 DPRINT lines across all worker cores; **258** unique (elf, TileDesc, cfg) tuples after dedup; the per-(elf, cfg-shape) signature is invariant across cores (only the base addresses vary per-core, as expected).

### SEC0 (= weight side = SrcA = Unpacker 0 — the BFP8/BFP4 path that diverges)

| ELF | Tensor | InDataFormat | TileDesc[0] | TileDesc[1] (Y/Z) | TileDesc[2] (W/Blobs) | TileDesc[3] (Digest) | cfg[0] | cfg[1] (compress/force_exp) | cfg[2] (limit_addr) | cfg[3] (fifo_size) | off_fmt |
|---|---|---:|---|---|---:|---:|---|---|---:|---:|---:|
| **0x2db6e** | W1+W3 fused (**BFP4 clean**) | **7 (BFP4)** | 0x17 | 0x40001 | 0 | 0 | 0x27 | 0xf000f | **0** | **0** | 0 |
| 0x2d96e | WQKV (**BFP8 garbage**)        | **6 (BFP8)** | 0x16 | 0x40001 | 0 | 0 | 0x26 | 0xf000f | **0** | **0** | 0 |
| 0x2de6b | WO (**BFP8 garbage**)          | **6 (BFP8)** | 0x16 | 0x40001 | 0 | 0 | 0x26 | 0xf000f | **0** | **0** | 0 |
| 0x2df6a | W2 (**BFP8 garbage**)          | **6 (BFP8)** | 0x16 | 0x40001 | 0 | 0 | 0x26 | 0xf000f | **0** | **0** | 0 |

Field decode (cfg word indexing = `THCON_SEC0_REG2_Out_data_format_ADDR32 + i` for i in 0..3; td indexing = `THCON_SEC0_REG0_TileDescriptor_ADDR32 + i`):

- **TileDesc word 0** (bits [31:0]):
  - bits [3:0] = `InDataFormat`
    - 0x6 = `BFP8` (per `Unpackers_FormatConversion.md` encoding table 0b01??/0b??10)
    - 0x7 = `BFP4` (per same table 0b01??/0b??11)
  - bit [4] = `IsUncompressed` = 1 (both)
  - bit [5] = `NoBFPExpSection` = **0** (both → exponent section IS present for both formats)
  - bits [10:8] = `BlobsPerXYPlane` = 0
  - bits [31:16] = `XDim` = 0 (SEC0 = unpA → per `cunpack_common.h:819` SEC0 leaves XDim=0 because unpA uses per-context x_dim from `REG5_Tile_x_dim_cntx0`)
- **TileDesc word 1**: bits [7:0] = `YDim` = 1; bits [23:16] = `ZDim` = 4 (4 faces per tile). Both BFP4 and BFP8 carry ZDim=4.
- **TileDesc word 2**: WDim = 0 → effectively 1.
- **TileDesc word 3**: `DigestSize` (bits [31:24]) = 0 → tile header is `(1+0)*16 = 16` bytes.
- **cfg word 0** (THCON_SEC0_REG2 word 120, bits [31:0]):
  - bits [3:0] = `Out_data_format`
    - 0x6 = BFP8; 0x7 = BFP4 — matches InDataFormat (`cunpack_common.h:842 config.f.out_data_format = unpA_dst_format_masked`)
  - bits [5:4] = `Throttle_mode` = 0b10 = 2 (x4 speed) for both
  - bits [7:6] = `Context_count` = 0
  - bit [8] = `Haloize_mode` = 0
  - bit [9] = `Tileize_mode` = 0
  - bit [10] = `Unpack_Src_Reg_Set_Upd` = 0
  - bit [11] = `Unpack_If_Sel` = 0
  - bits [13:12] = `Upsample_rate` = 0
  - bit [14] = **`Ovrd_data_format` = 0** → no multi-context format override is engaged → InDataFormat is sourced from `ConfigDescriptor.InDataFormat` (the TileDescriptor word 0 bits [3:0]) per UNPACR_Regular.md lines 86-92.
  - bit [15] = `Upsample_and_interleave` = 0
  - bits [31:16] = `Shift_amount_cntx0..3` = 0
- **cfg word 1** (THCON_SEC0_REG2 word 121):
  - bits [3:0] = `Disable_zero_compress_cntx0..3` = 0xf (zero-compression disabled for all 4 contexts, correct since data is uncompressed)
  - bits [7:4] = `Unpack_if_sel_cntx0..3` = 0
  - bit [8] = **`Force_shared_exp` = 0** → unpacker uses the per-tile shared-exp prefix bytes from L1 (NOT a forced fixed exponent) per UNPACR_Regular.md lines 304-313.
  - bits [11:9] = `Context_count_non_log2` = 0
  - bit [12] = `Context_count_non_log2_en` = 0
  - bits [19:16] = `Disable_zero_compress_cntx4..7` = 0xf
  - bits [23:20] = `Unpack_if_sel_cntx4..7` = 0
- **cfg word 2** (THCON_SEC0_REG2 word 122):
  - bits [16:0] = **`Unpack_limit_address` = 0** → EMPIRICALLY CONFIRMS U42 static-analysis verdict.
- **cfg word 3** (THCON_SEC0_REG2 word 123):
  - bits [16:0] = **`Unpack_fifo_size` = 0** → EMPIRICALLY CONFIRMS U42 static-analysis verdict.
  - WrapAddr lambda from UNPACR_Regular.md:199-207 reduces to `addr -= 0 * 16 = 0` → no-op for every positive L1 address. **The proposed BFP8-vs-BFP4 wrap mechanism cannot fire.**
- **off_fmt** (THCON_SEC0_REG7 word 140, overlap of `REG7_Offset_address` AND `REG7_Unpack_data_format_cntx0`):
  - = 0 on every ELF → `REG7_Offset_address = 0` (no addressable base offset) AND `REG7_Unpack_data_format_cntx0 = 0` (no per-context format override). Combined with `Ovrd_data_format = 0` above, the entire multi-context format path is INACTIVE. InDataFormat comes purely from TileDescriptor word 0 bits [3:0].

### SEC1 (= activation side = SrcB = Unpacker 1)

| ELF | Tensor | TileDesc word 0 | cfg word 0 | cfg[1] | cfg[2] | cfg[3] | off_fmt |
|---|---|---|---|---|---:|---:|---:|
| 0x2db6e | W1+W3 (BFP4 clean) | 0x1000015 | 0x25 | 0xf000f | 0 | 0 | 0 |
| 0x2d96e | WQKV (BFP8 garbage) | **0x1000016** | **0x26** | 0xf000f | 0 | 0 | 0 |
| 0x2de6b | WO (BFP8 garbage)   | 0x1000015 | 0x25 | 0xf000f | 0 | 0 | 0 |
| 0x2df6a | W2 (BFP8 garbage)   | 0x1000015 | 0x25 | 0xf000f | 0 | 0 | 0 |

- **TileDesc word 0 = `0x100001X`**: bits [31:16] = 0x0100 → XDim = 256 = 16×16 (one full face). bits [3:0] = activation format: WQKV uses BFP8 activations (0x6), the other three use BF16 activations (0x5). This is a discriminator AMONG SEC1 but NOT a discriminator between BFP4-clean and BFP8-garbage SEC0 paths.
- **cfg word 0**: `Out_data_format` = 0x6 (BFP8) for WQKV, 0x5 (BF16) for the other three. Matches InDataFormat — legitimate format-bits.
- All other SEC1 cfg fields = identical to SEC0 (`Force_shared_exp=0`, `Ovrd_data_format=0`, `Unpack_limit_address=0`, `Unpack_fifo_size=0`, `Disable_zero_compress=0xf`, etc.). No discriminator.

## What this proves

1. **U42's static-analysis verdict is empirically validated.** Under live `SGLANG_TT_USE_PREFETCHER=1` operation on Qwen3-8B / Blackhole, `Unpack_limit_address = 0` and `Unpack_fifo_size = 0` on EVERY ELF (BFP4-clean and BFP8-garbage alike). The WrapAddr lambda from UNPACR_Regular.md:199-207 is a no-op in practice. The U42 "subtract 0 → unchanged" prediction holds at the hardware level.

2. **Every other cfg-register-level discriminator hypothesis from the U43 plan is REFUTED EMPIRICALLY:**
   - (a) `REG2_Force_shared_exp` — `0` on both BFP4 and BFP8 paths → unpacker reads per-tile shared exp from L1 (correct behavior, not the forced-exp path). REFUTED.
   - (b) `REG2_Ovrd_data_format + REG7_Unpack_data_format_cntx` — both `0` → multi-context format override is INACTIVE → InDataFormat is sourced from TileDescriptor word 0 bits [3:0] (per UNPACR_Regular.md line 90 `else` branch). REFUTED.
   - (c) `THCON_SEC0/SEC1.REG3_Base_address + REG7_Offset_address` — `REG7_Offset_address = 0` for all ELFs; the per-tile address is `REG3_Base_address` (the ELF's tensor base) without any added offset. The base addresses themselves vary per-core (as expected for shared-L1 reservation) but the SHAPE of the address arithmetic is identical across ELFs. REFUTED — no SEC1-vs-SEC0 mismatch in offset application.
   - (d) `TileDescriptor.DigestSize` — `0` for ALL ELFs → tile header size = `(1+0)*16 = 16` bytes (the standard BFP8/BFP4 shared-exp prefix). Identical across formats. REFUTED.
   - (e) `TileDescriptor.NoBFPExpSection` — `0` for ALL ELFs (bit 5 of word 0 is 0) → exponent section IS allocated for both BFP4 and BFP8. Note: per UNPACR_Regular.md lines 128-132, this bit is ONLY consulted for non-BFP8 formats (BFP8/BFP8a always allocate exp section); for BFP4 it must be 0 to enable the exponent fetch — which it IS. REFUTED.
   - (f) `MultiContextMode` state leakage from prior matmul — `Context_count = 0`, `Context_count_non_log2_en = 0`, `Ovrd_data_format = 0`, `REG7_Unpack_data_format_cntx0 = 0` on every ELF. MultiContextMode is structurally not engaged on this code path. REFUTED.

3. **The 3 BFP8 ELFs (WQKV, WO, W2) have IDENTICAL THCON_SEC0 cfg state at the register level** (modulo per-core `REG3_Base_address` which is a legitimate per-core offset into the prefetcher's shared L1 region). Yet ALL 3 produce garbage. The 1 BFP4 ELF (W1+W3 fused) differs only in `InDataFormat = 7` vs `6` and `Out_data_format = 7` vs `6` — i.e., the legitimate per-tensor format bits. Same DRIVER (unpacker MOP), same cfg state apart from the format-bit difference, yet ONE produces correct output and THREE produce garbage. The bug is invariant under the entire set of THCON_SEC0/SEC1 cfg registers visible from the kernel side.

## Phase 4 — Active-vs-orphan LLK tree diff

Confirmed: the active build path links against `tt_metal/tt-llk/tt_llk_blackhole/` per `tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:80-87` and `tt_metal/hw/CMakeLists.txt:383-384`. The `tt_metal/third_party/tt_llk/tt_llk_blackhole/` tree is the older 2025 mirror; never linked. Key differences:

- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` (active, ©2026, 348 lines, 15954 B) vs. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` (orphan, ©2025, 15733 B). Diff: active adds 2 lines (the 16x16 matmul assert at lines 202-203). Both versions have the SAME `_llk_unpack_AB_matmul_` structure: `_llk_unpack_configure_addresses_(address_b, address_a, cfg)` writing the matmul's `address_b` (in0/activations) to THCON_SEC0 and `address_a` (in1/weights) to THCON_SEC1 inside the helper (which the helper docstring labels with the OPPOSITE convention — see `tt-llk/.../llk_unpack_common.h:218-243`). Net: weights → SEC0 base for matmul; SrcA (Unpacker 0) reads SEC0 → consistent.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB.h` (active, ©2026) vs. orphan (©2025). Diff: active uses `ckernel::TensorShape` parameter; orphan uses `num_faces + narrow_tile` parameters. Both expose the same MOP-config behavior; not on the matmul path.
- All SFPU + math files differ between trees — but those are not on the matmul-unpack path.

Verdict: any U40/U41 line references to `third_party/tt_llk/.../llk_unpack_AB_matmul.h:331-335` correspond to the orphan 2025 tree. The active tree's equivalent code lives at `tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:333-336` (the `else` branch emitting `TTI_UNPACR(SrcA, ...)` for the full-tile non-partial-face case). Structurally identical; bug surface is the same in both versions.

## Phase 3 — Why no fix is shipped

The U43 plan specified that, once a discriminator is identified, an env-gated override of the suspect cfg register should be attempted. **No discriminator was identified** (Phase 2 reduced to the empty set after Phase 1 capture). Therefore Phase 3's "attempt a fix targeting the discriminator" has nothing to act on. Attempting to override e.g. `Unpack_limit_address` or `Force_shared_exp` would FLIP the value from 0 to non-zero on every ELF, which would change behavior in a way the LLK has never been tested under — high regression risk vs. zero chance of fixing the underlying issue.

## What U43 also empirically verified (incidentally)

- **U37 + U41 byte-equality** continues to hold: the producer writes the right bytes; the consumer reads the right bytes; the bug is not byte delivery.
- **U38 `set_tile_dims`** is engaged on the gathered path — `TileDescriptor.IsUncompressed = 1`, `YDim = 1`, `ZDim = 4` (= num_faces) match the BFP8 per-CB metadata.
- **U40 `force_unpack_reconfig`** had been an attempt to force a `reconfig_data_format_srca/srcb` post-`mm_block_init`. Per U40 it engaged at the JIT binary but didn't fix the bug. U43 now explains why: the post-reconfig cfg state IS already correct (same InDataFormat / Out_data_format / Throttle / etc. as what the LLK programmed at init time); there is nothing wrong with the cfg state for the reconfig to "fix".

## Hard constraints checked

- [x] No upstream PR.
- [x] No SGLang core behavior changes. The only sglang change is this doc.
- [x] No destructive `rm` without `[ -n "$VAR" ]` guard (cache cleanup uses literal `/root/.cache/tt-metal-cache` with the variable-set guard).
- [x] No stash pop/drop.
- [x] Port-clear with `\.` escape (`pkill -9 -f "sglang\.launch_server.*--port 30000"`).
- [x] HF_MODEL=/models/Qwen3-8B (local container path).
- [x] Container NOT bind-mounted — sources sync'd via `podman cp` then rebuilt via fixed `rebuild_tt_metal_kernels.sh ttnn`.
- [x] Cache clear used literal absolute path.
- [x] `tt_metal/third_party/tt_llk/` NOT touched.
- [x] Canonical Qwen3-8B path tested under U43 envs: probe is gated on `use_global_cb` AND `SGLANG_TT_U43_CFG_DUMP=1` — canonical run with `SGLANG_TT_U43_CFG_DUMP=1` but no `SGLANG_TT_USE_PREFETCHER=1` produces ZERO U43 lines (probe inactive) AND correct output `" What is 2+2"` for the standard `"What is 2+2?"` prompt.
- [x] Probe is env-gated default-off.

## Working state at session end

- tt-metal-sglang HEAD: **`4591a063661`** + 1 commit pending (U43 cfg-dump probe).
- sglang HEAD: **`556e99311`** + 1 commit pending (this doc).
- Container `/tt-metal/` source: synced via `podman cp`, includes U43 probe.
- `_ttnn.so` / `_ttnncpp.so`: rebuilt (incremental ttnn target), include U43 wiring (verified via `strings ... | grep SGLANG_TT_U43_CFG_DUMP`).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped (post-canonical-verify).
- Canonical re-verify: PASS — `" What is 2+2"` echoed correctly under all probe envs unset for prefetcher.
- Artifacts in-container:
  - `/tmp/u43.log` (10.6 MB; 92,159 U43 DPRINT lines from the prefetcher run; 258 unique (elf, td, cfg) tuples after dedup).
  - `/tmp/u43_canonical.log` (178 B; zero U43 lines since gated on `use_global_cb`).
  - `/tmp/u43/run.log` (prefetcher launch + decode log).
  - `/tmp/u43/canonical.run.log` (canonical launch + decode log).

## Commits this session

- (tt-metal-sglang) **1 commit pending** — U43 cfg-dump probe in gathered compute kernel + factory wiring (env-gated `SGLANG_TT_U43_CFG_DUMP`).
- (sglang) **1 commit pending** — this doc.

## Refined 57-dispatch ledger

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
| U41 | CLOSED | Sub-7 (face-3 byte misordering) RULED OUT by 10/10 producer-consumer face-3 byte MATCH; Sub-8 (unpack_*[] staleness) RULED OUT by identical metadata across BFP4/BFP8 modulo dtype. Root cause = upstream tt-metal LLK BFP8 4-face SrcA decode × dual-index CB allocation. |
| U42 | CLOSED | `Unpack_limit_address` / `Unpack_fifo_size` WrapAddr hypothesis REFUTED BY STATIC ANALYSIS without hardware. Both regs = 0 throughout production code paths. |
| **U43** | **CLOSED — Runtime DPRINT of THCON_SEC0/SEC1 unpacker cfg registers on a live prefetcher run EMPIRICALLY CONFIRMS U42 (both regs = 0 at HW level) AND REFUTES every other cfg-register-level discriminator hypothesis (Force_shared_exp=0, Ovrd_data_format=0, Unpack_data_format_cntx=0, NoBFPExpSection=0, DigestSize=0, IsUncompressed=1, ZDim=4 — IDENTICAL across BFP4-clean and BFP8-garbage ELFs). The 3 BFP8 ELFs have IDENTICAL THCON_SEC0 cfg state and ALL produce garbage; the BFP4 ELF differs only in the legitimate `InDataFormat` (7 vs 6) and `Out_data_format` (matching) bits and produces correct output. Bug is invariant under all THCON_SEC0/SEC1 cfg registers visible from the kernel-side cfg-read API. Must live in silicon state machine (SrcA AllowedClient handshake, SrcBank ownership, ADC counters) NOT capturable from cfg-reads. U41 verdict stands.** | **U43 closes the hardware-instrumentation phase. Every cfg-register-level mechanism has been empirically ruled out. Upstream Tenstorrent LLK / silicon-state investigation is the only forward path.** |

## Phase 9 — What WOULD a useful U44 hypothesis look like?

Given U37-U43 have ruled out:
- byte delivery (U37, U41 — 10/10 producer-consumer match at all 5 ELFs + face-3 windows);
- per-CB `unpack_*[]` metadata (U41 — identical for BFP4 working / BFP8 broken);
- naive LLK reconfig variants (U40 — Path C / per-block / MOP reinit all engage but don't fix);
- `fp32_dest` × BFP8 (U39);
- face-stride faces 0/1/2 (U39);
- LocalCBInterface `fifo_page_size` (U39);
- face-3 mantissa byte misordering (U41 — 10/10 byte-match);
- the `WrapAddr` config-register hypothesis (U42 static, U43 runtime — both regs = 0);
- every other cfg-register-level hypothesis (U43 — Force_shared_exp, Ovrd_data_format, Unpack_data_format_cntx, NoBFPExpSection, DigestSize, IsUncompressed, ZDim ALL identical across BFP4 and BFP8 ELFs);

…the remaining viable hypothesis classes are NOT addressable from compute-kernel-side C++ or from any cfg-register override. They are below the cfg-API surface and require Tenstorrent's silicon-state investigation:

1. **SrcA AllowedClient handshake race under the BFP8 4-face decode path.** Per `UNPACR_Regular.md:354-356`, the unpacker waits for `SrcA[Bank].AllowedClient == SrcClient::Unpackers` before issuing decoded datums. The MOP fast-path emits packed `TTI_UNPACR(SrcA, ...)` instructions out of a replay buffer programmed at init time. The replay buffer assumes a specific tile-stride that may differ between BFP4 (576 B / tile) and BFP8 (1088 B / tile) under the dual-index CB allocation's L1 reservation pattern. Not addressable without Tenstorrent silicon-trace.

2. **ADC counter state leakage between matmuls.** Per `UNPACR_Regular.md:139-140`, ADC channels track input position. `_llk_unpack_AB_matmul_` doesn't reset ADCs between consecutive matmul calls within the same MOP loop. If the ADC state from a prior matmul leaks into a later matmul's `FirstDatum` calculation, the unpacker would fetch from the wrong offset within the tile. The cfg registers show NO leakage indicator, but ADCs are NOT cfg registers — they're auto-incrementable address counters internal to the unpacker. Not addressable from the kernel-side cfg-read API.

3. **TTI_UNPACR replay buffer programmed at init-time captures the FIRST ELF's tile geometry.** `mm_block_init` runs once at the top of the gathered kernel and programs the replay buffer with a specific `WhichUnpacker` + `Ch0/Ch1 increments`. If the replay buffer is shared across all 4 ELFs (because the JIT-compiled binary is reused), and the BFP8 ELFs' replay-buffer entries were originally programmed for the BFP4 ELF's tile geometry (or vice versa), the unpacker would issue UNPACR instructions that fetch from the wrong addresses. This is plausible only if the dual-index CB allocation causes ALL 4 ELFs to share a single replay buffer, which is internal to the LLK and not addressable from sglang.

None of these are addressable from sglang's plug-in surface. **U41 handoff to upstream Tenstorrent remains the only viable forward path.**

## Final handoff summary

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B on Blackhole.** After 57 dispatches (S1-S10, Lead 1/2/3, A1, U1-U43), every downstream, config-register-level, and HW-instrumentable suspect is refuted. The remaining viable bug lives in upstream tt-metal LLK BFP8 4-face SrcA decode (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:333-336`, the `else` branch emitting a single `TTI_UNPACR(SrcA, ...)` for the BFP8 4-face full-tile case) interacting with the dual-index `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` allocation pattern, in a regime where EVERY byte at the consumer L1 (U37, U41), EVERY per-CB `unpack_*[]` metadata field (U41), AND EVERY THCON_SEC0/SEC1 unpacker cfg register (U43) is provably correct or benign. This needs Tenstorrent LLK-source-level + silicon-state investigation; sglang has no further leverage.

Production shipping config remains canonical `tt_transformers_paged` (no prefetcher) at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9/10. The U37-U43 env-gated diagnostic harness (`SGLANG_TT_U37_PROD_BYTES`, `SGLANG_TT_U37_READ_BYTES`, `SGLANG_TT_U39_EXT_BYTES`, `SGLANG_TT_U39_CB_META`, `SGLANG_TT_U41_FACE3_PROBE`, `SGLANG_TT_U40_PROBE_TILE_DIMS`, `SGLANG_TT_U41_UNPACK_ARR_PROBE`, `SGLANG_TT_U38_SET_TILE_DIMS`, `SGLANG_TT_U40_FORCE_UNPACK_RECONFIG`, `SGLANG_TT_U40_RECONFIG_BLOCK`, `SGLANG_TT_U32_GCB_OFFSET`, `SGLANG_TT_U33_FACTORY_OFFSET`, `SGLANG_TT_U34_LCM_ALIGN`, `SGLANG_TT_U35_PER_LAYER_OFFSET`, **`SGLANG_TT_U43_CFG_DUMP`**) remains the on-tree definitive reproducer + diagnostic for the upstream Tenstorrent team.
