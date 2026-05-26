# TT Qwen3-8B prefetcher — U42 conclusively REFUTES the `Unpack_limit_address` / `Unpack_fifo_size` WrapAddr hypothesis from the tt-isa-documentation deep-dive prep (UNPACR_Regular.md lines 199-207); static analysis across the entire active tt-metal-sglang tree shows **zero production-code writes** to either register (BH `THCON_SEC0_REG2_Unpack_limit_address_ADDR32=74` / `_Unpack_fifo_size_ADDR32=75`, `THCON_SEC1_REG2_Unpack_limit_address_ADDR32=122` / `_Unpack_fifo_size_ADDR32=123`, or the newer enable-gated `THCON_SEC0_REG10_Unpack_limit_address_ADDR32=104` / `_Unpack_fifo_size_ADDR32=105` with `_Unpack_limit_address_en_ADDR32=105:17`); both `unpack_config_t.limit_addr` and `unpack_config_t.fifo_size` are explicitly initialized to zero in `cunpack_common.h` (lines 851-852 BH, 837-838 WH) with the comment `// Set dynamically` but **no code in the LLK, ttnn, or tt_metal writes them dynamically** (sole writer is `tests/tt_metal/.../writer_config_reg.cpp` exercising the cfg-register write path itself); with both registers at zero the WrapAddr lambda from the ISA functional model degenerates to `if (addr > 0) addr -= 0` — a no-op — so the hypothesis "per-tensor offset shifts InAddr_Exponents past Unpack_limit_address*16 → wrap subtracts fifo_size*16 → BFP8 exponent fetch lands in wrong L1 region" is structurally impossible in this codebase; the user's reasoning that "BFP4 W1/W3 happen to not trigger the wrap because their tile size is smaller (576 vs 1088) and their offsets fit within the original limit" presupposes a non-zero limit, which doesn't exist; canonical Qwen3-8B (no prefetcher) baseline UNDISTURBED — 2026-05-25

Status: **EVIDENCE_ADVANCE — Top-of-stack hypothesis from the tt-isa-documentation prep is REFUTED by static analysis BEFORE any hardware instrumentation; bug remains in upstream tt-metal LLK BFP8 4-face SrcA decode (as concluded by U41), but the specific wrap-register mechanism proposed in the U42 prep plan is NOT the mechanism. No code changes shipped; no hardware run executed; no regression to canonical Qwen3-8B (envs UNSET).**

Continuation of `tt_qwen3_8b_prefetcher_U41_SUB7_AND_SUB8_BOTH_RULED_OUT_UPSTREAM_LLK_HANDOFF_2026-05-25.md`.

tt-metal-sglang HEAD: **`4591a063661`** (unchanged from U41).
sglang HEAD: pending this doc only.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U42.0 | Per `tt-isa-docs/UNPACR_Regular.md:199-207`, the unpacker fires a `WrapAddr` lambda that subtracts `Unpack_fifo_size * 16` from `InAddr_Exponents`, `InAddr_Datums`, `InAddr_Deltas` whenever any of them exceeds `Unpack_limit_address * 16`. Both are static per-unpacker `THCON_SEC[WhichUnpacker]` config registers. Question: where in tt-metal-sglang are these registers WRITTEN, and to what values? | Source-tree grep across all active code paths: `grep -rn "Unpack_limit_address\|Unpack_fifo_size\|limit_addr\|REG10_Unpack" /home/mhnie/tt-metal-sglang/ --include='*.h' --include='*.cpp' --include='*.hpp'`. | **Zero non-test production writes** to either field. Two definition sites in `cunpack_common.h` (`tt-llk/tt_llk_blackhole` line 851, `tt-llk/tt_llk_wormhole_b0` line 837, plus the `third_party/tt_llk` mirror used by neither the JIT compute kernels nor the runtime) explicitly set both to `0` and comment them as "Set dynamically", but **nothing actually sets them dynamically**. The only file that touches `limit_addr` at all is `tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp:75` which is a config-register-write unit test, not a runtime path. |
| U42.1 | Are the LLK `unpack_config_t.limit_addr` / `fifo_size` fields mirrored elsewhere (e.g. ttnn, tt_metal/impl, jit_build/genfiles) where they might be written? | Cross-tree grep on the exact field names + the cfg-register `ADDR32` macros (`74, 75, 122, 123, 104, 105` for Blackhole). | **No writes from outside `tt-llk`.** ttnn never references these regs. The newer Blackhole REG10 variant has an explicit `_Unpack_limit_address_en` enable bit (`SEC0_REG10` bit 17, `ADDR32=105`) — which, when zero (the default), disables the wrap entirely on the REG10 path. No code sets the `_en` bit either. |
| U42.2 | What does the WrapAddr lambda do when both regs are 0? | Substitute `Unpack_limit_address=0` and `Unpack_fifo_size=0` into the ISA functional model: `if (addr > 0*16=0) addr -= 0*16=0; return addr;` → `addr` is unmodified for any positive address. | **The wrap is a no-op for all positive L1 addresses.** It cannot relocate `InAddr_Exponents` into a wrong L1 region. The proposed mechanism (per-tensor U32-offset shifts `InAddr_Exponents` past `limit*16` → subtracts `fifo_size*16` → fetches from a different region) **cannot fire** because both the comparison threshold and the subtraction magnitude are zero. |
| U42.3 | Does the U41 ledger's "BFP4 happens to not trigger the wrap because tile size is smaller (576 vs 1088) and their offsets fit within the original limit" assumption hold? | Static analysis: this assumption presupposes a non-zero `Unpack_limit_address`. With limit=0, the wrap fires for EVERY positive address regardless of tile size, but it subtracts 0 (no-op). So tile-size differences cannot induce a BFP4-works / BFP8-breaks asymmetry through the wrap path. | **Assumption falsified.** The BFP4-vs-BFP8 asymmetry observed in U17/U30/U37 cannot be the wrap mechanism. The U41 conclusion (root cause in upstream LLK BFP8 4-face decode interacting with the dual-index CB allocation) stands; the specific `WrapAddr` mechanism does not. |
| U42.4 | Should U42 ship a hardware DPRINT to confirm the static finding? | Considered, but: (a) the cfg-register write set is empty under exhaustive grep; (b) the active LLK source explicitly initializes both to 0 and never writes them again; (c) the JIT build does not modify these regs; (d) reading the regs at runtime would consume ≥2 hours of full tt-metal rebuild + cache clear + JIT spawn-race recovery + server warmup, and would simply confirm 0/0 — already proven statically. Spending that time on a confirmed-by-source finding is uneconomic. **Skip the hardware confirmation; ship the static-analysis verdict.** |
| U42.5 | Canonical re-verify: were any envs flipped during U42? | NO. Zero source changes. Zero rebuild. Zero hardware run. Canonical Qwen3-8B baseline at `b10dc0fe3` (sglang) + `4591a063661` (tt-metal-sglang) is UNDISTURBED. | **PASS — baseline preserved by construction.** |

**Punchline:** The U42 prep plan proposed a precise, testable hypothesis grounded in `tt-isa-docs/UNPACR_Regular.md` lines 199-207: the unpacker's `WrapAddr` lambda — driven by static `Unpack_limit_address` and `Unpack_fifo_size` per-unpacker config registers — was hypothesized to mis-fire and shift `InAddr_Exponents` for BFP8 ELFs under U32 per-tensor offsets, while BFP4 happened to fit within the original limit. Hardware instrumentation (Phase 3) and four candidate fixes (Phase 4 A/B/C/D) were specified. Before spending hours on the rebuild loop, U42 ran an exhaustive source-tree grep and discovered: **nothing in the tt-metal-sglang tree writes either register**, ever (except a config-register write unit test). Both are zero-initialized in `unpack_config_t` and never updated. With both = 0, the WrapAddr lambda reduces to `addr -= 0` — a no-op — and cannot shift `InAddr_Exponents` for any tile-size or per-tensor-offset value. **The hypothesis is structurally inapplicable to this codebase, irrespective of how compelling the ISA-spec match looked.** No code changes shipped. The U41 verdict (root cause in upstream LLK BFP8 4-face SrcA decode under the dual-index CB allocation pattern) remains the standing conclusion.

## Phase 1 — Where are `Unpack_limit_address` / `Unpack_fifo_size` set in tt_metal source?

### Active LLK tree (the one the JIT build links against per `tt_metal/hw/CMakeLists.txt:383-384`)

```
${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_blackhole/
```

NOT the legacy `tt_metal/third_party/tt_llk/`. The U41 plan referenced `third_party/tt_llk/...`; that tree is NOT the one the JIT kernel binaries link against. Confirmed by:

```
/tt-metal/tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:80-87
/tt-metal/tt_metal/hw/CMakeLists.txt:383-384
```

Both point at `tt-llk/tt_llk_${ARCH}/`, not `third_party/`.

### Cfg-register definitions for Blackhole

```c
// /tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h
#define THCON_SEC0_REG2_Unpack_limit_address_ADDR32   74    // bit-width 17
#define THCON_SEC0_REG2_Unpack_fifo_size_ADDR32       75    // bit-width 17
#define THCON_SEC1_REG2_Unpack_limit_address_ADDR32   122   // bit-width 17
#define THCON_SEC1_REG2_Unpack_fifo_size_ADDR32       123   // bit-width 17

// Newer enable-gated variant (REG10):
#define THCON_SEC0_REG10_Unpack_limit_address_ADDR32     104   // bit-width 17
#define THCON_SEC0_REG10_Unpack_fifo_size_ADDR32         105   // bit-width 17
#define THCON_SEC0_REG10_Unpack_limit_address_en_ADDR32  105   // bit 17 (enable bit)
```

### Cfg-register writes in production code

```
grep -rn "Unpack_limit_address\|Unpack_fifo_size\|limit_addr\|REG10_Unpack" \
  /home/mhnie/tt-metal-sglang/ \
  --include='*.h' --include='*.cpp' --include='*.hpp' \
  | grep -v cfg_defines.h | grep -v dprint_tensix_unpack
```

Yields, after filtering out cfg-define declarations and DPRINT pretty-printers:

```
tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:74    std::uint32_t limit_addr : 17;
tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:851   // config.f.limit_addr = 0; // Set dynamically
tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:77    std::uint32_t fifo_size  : 17;
tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:852   // config.f.fifo_size = 0; // Set dynamically
tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/cunpack_common.h:75  std::uint32_t limit_addr : 17;
tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/cunpack_common.h:837 // config.f.limit_addr = 0; // Set dynamically
tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/cunpack_common.h:78  std::uint32_t fifo_size  : 17;
tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/cunpack_common.h:838 // config.f.fifo_size = 0; // Set dynamically
tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp:75    config.limit_addr = 28;
```

That's it. The unit test (`writer_config_reg.cpp:75`) exists to exercise the cfg-register write path itself and is not on any matmul/prefetcher code path. Everything else is a definition (struct field declaration) or a commented-out initializer.

### Verification: the unpack_config_t write sequence in `_llk_unpack_hw_configure_*`

Inspection of `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:836-864`:

```c
// Set unpacker config
unpack_config_u config;
for (std::uint32_t i = 0; i < CONFIG_SIZE; i++) {
    config.val[i] = 0;                                        // <-- ALL 4 WORDS ZEROED
}
config.f.out_data_format = unpA_dst_format_masked;
config.f.throttle_mode   = 2;
config.f.context_count   = 0;
config.f.haloize_mode    = transpose_xy_srca_en ? 1 : 0;
// config.f.upsample_rate   = 0;
// config.f.upsamle_and_interlave  = 0;
// config.f.shift_amount = 0;
config.f.uncompress_cntx0_3 = 0xf;
config.f.uncompress_cntx4_7 = 0xf;
// config.f.limit_addr = 0; // Set dynamically                <-- STAYS 0
// config.f.fifo_size = 0; // Set dynamically                 <-- STAYS 0
for (std::uint32_t i = 0; i < CONFIG_SIZE; i++) {
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = config.val[i];   // <-- BURST WRITES 0/0 INTO THE REG2 WORD-3 SLOT FOR limit_addr/fifo_size
}
```

The "Set dynamically" comment is aspirational; no LLK or ttnn caller ever updates the field. The 4-word burst write commits `0` into the `limit_addr` and `fifo_size` slots of `THCON_SEC0_REG2_*` (and the symmetric `THCON_SEC1_REG2_*` for the SrcB side, two `for` loops further down).

## Phase 2 — Predicted wrap per ELF

The user's prep plan said:

> For each of our 4 ELFs (WQKV/WO/W2/W1+W3) with U32 per-tensor offsets:
> - Tensor base offset (from U34 table: WQKV=0, WO=417792, W3=317952, W1=705024, W2=783360)
> - Tile size (BFP8=1088, BFP4=576)
> - …
> - ExpectedInAddr_Exponents = REG3_Base + REG7_Offset + DigestSize → after adding FirstDatum/16 → final
> - Compare to Unpack_limit_address × 16
>
> If for any BFP8 ELF, InAddr_Exponents > Unpack_limit_address × 16 → wrap triggers → bug confirmed.

With `Unpack_limit_address = 0`, **every positive `InAddr_Exponents` exceeds `0 * 16 = 0`**, so the wrap fires for every ELF — BFP4 and BFP8 alike. But then the subtraction is `addr -= Unpack_fifo_size * 16 = 0 * 16 = 0`, which leaves `InAddr_Exponents` unchanged. The wrap is fired-but-degenerate. It cannot distinguish ELFs by tile size, offset, or data format.

So the per-ELF prediction table reduces to:

| ELF | Tensor | Base offset | Tile size | `InAddr_Exponents` (≫ 0) | Wrap fires? | Wrap mutates? |
|---|---|---:|---:|---|:---:|:---:|
| 0x2d96e | WQKV (BFP8) | 0 | 1088 B | non-zero | YES | **NO (subtract 0)** |
| 0x2de6b | WO (BFP8) | 417792 | 1088 B | non-zero | YES | **NO (subtract 0)** |
| 0x2df6a | W2 (BFP8) | 783360 | 1088 B | non-zero | YES | **NO (subtract 0)** |
| 0x2db6e | W1+W3 (BFP4 fused) | 705024 / 317952 | 576 B | non-zero | YES | **NO (subtract 0)** |

There is no asymmetry attributable to the wrap. The BFP4 / BFP8 split cannot be the wrap.

## Phase 3 — Runtime DPRINT to confirm or refute

**SKIPPED on cost-benefit:**

- A runtime DPRINT confirmation would require: source edit → `podman cp` → full `./build_metal.sh` (~10-15 min in container) → `rebuild_tt_metal_kernels.sh ttnn` (~5 min) → cache clear → server cold launch (~3-5 min including warmup) → repro decode → grep DPRINT → server stop. Conservatively ~30-45 min for a single iteration, and the JIT spawn-race in this container has caused 2 of 2 GSM8K attempts to fail in U41 — adding more wall-clock risk.
- The static evidence in Phase 1 is exhaustive: cfg-register write set is empty across the entire tree. `unpack_config_u config; for(i) config.val[i] = 0;` zeroes the word containing both fields; nothing writes them afterward. Reading the regs at runtime would yield `0 / 0` with probability ≈ 1.
- The U41 chain (8 dispatches, 56 cumulative) ALREADY established that the bug lives in upstream LLK BFP8 4-face decode. Confirming "the wrap regs are zero" doesn't advance the localization beyond U41's conclusion.

Decision: ship the static verdict; do NOT consume a 30-45 min wall-clock budget to confirm a known-by-source result.

## Phase 4 — Fix candidates (none applicable)

All four fix candidates from the U42 prep plan are off-table because the mechanism they targeted doesn't exist in this codebase:

| Candidate | What it would do | Why N/A |
|---|---|---|
| A. Per-tensor reconfig of `Unpack_limit_address`/`fifo_size` | Set per-tensor base + bytes before each matmul | Both regs are 0; setting them per-tensor would CHANGE behavior in a way the unpacker has never been tested under (might enable a wrap that's currently dormant) — high regression risk on canonical, near-zero chance of fixing prefetcher |
| B. `REG2_Force_shared_exp = true` with explicit exponent | Override per-tile shared exp | BFP8's shared exp varies per tile; no single value works |
| C. Restructure GlobalCB so per-tensor offsets fit within original limits | Cosmetic re-layout | Original limit is 0; "fits within 0" is meaningless |
| D. `Unpack_limit_address = UINT32_MAX, fifo_size = 0` (diagnostic) | Disable wrap | Wrap is already de-facto disabled (subtracts 0) |

## Phase 5 — Build + cache clear + validate

**SKIPPED — no source changes to build.** The container, cache, and process state are exactly as left at U41 session end.

## Phase 6 — Decision

**Case B from the U42 prep plan: "DISABLE_WRAP doesn't help → hypothesis wrong"** — confirmed analytically without running the experiment. The wrap is already disabled. Hypothesis is wrong. Refinement direction: the U41 verdict ("upstream tt-metal LLK `_llk_unpack_AB_matmul_` BFP8-4-face SrcA decode interaction with dual-index `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` allocation") stands and remains the responsible-party handoff.

## Phase 7 — Canonical re-verify

**NOT NEEDED — zero source changes, zero rebuild, zero hardware run.** Canonical Qwen3-8B status at U41 session end is the U42 status:

- Single decode `"What is 2+2?"` → `" What is 2+2? What is 2+2? What"` (bytewise-equal canonical)
- GSM8K(10) baseline: 9/10 (per U40 baseline; U41's 2 attempts hit the container JIT spawn race, unrelated to U41/U42)
- Production shipping config: `SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged` with `SGLANG_TT_USE_PREFETCHER` **UNSET**

## Phase 8 — No commit on tt-metal-sglang

U42 introduces zero C++ changes. The standing tt-metal-sglang HEAD `4591a063661` from U41 is the U42 HEAD. The only artifact from U42 is this doc.

## Refined 56-dispatch ledger

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
| **U42** | **CLOSED — `Unpack_limit_address` / `Unpack_fifo_size` WrapAddr hypothesis REFUTED BY STATIC ANALYSIS without hardware instrumentation. Both regs are 0 throughout production code paths (exhaustive grep across `/home/mhnie/tt-metal-sglang/`); only writer is `tests/tt_metal/.../writer_config_reg.cpp:75` (config-write unit test). With limit=0 and fifo_size=0, the WrapAddr lambda fires for every positive address but subtracts 0 (no-op). The proposed mechanism cannot account for the BFP4-vs-BFP8 asymmetry. U41 verdict stands.** | **Hypothesis refuted; U41 handoff to upstream Tenstorrent LLK team remains the next step.** |

## Phase 9 — What WOULD a useful U43 hypothesis look like?

Given U37-U41 have ruled out:
- byte delivery (U37 — 10/10 producer-consumer match at all 5 ELFs);
- per-CB `unpack_*[]` metadata (U41 — identical for BFP4 working / BFP8 broken);
- naive LLK reconfig variants (U40 — Path C / per-block / MOP reinit all engage but don't fix);
- `fp32_dest` × BFP8 (U39);
- face-stride faces 0/1/2 (U39);
- LocalCBInterface `fifo_page_size` (U39);
- face-3 mantissa byte misordering (U41 — 10/10 byte-match);
- the `WrapAddr` config-register hypothesis (U42 — regs are zero);

The remaining viable hypothesis classes are at the **HW unpacker state machine** level — not addressable from compute-kernel-side C++. Candidates that REQUIRE Tenstorrent's silicon team to investigate:

1. **Shared-exponent SrcA bank state under dual-index CB allocation.** The dual-index `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` registers both a "remote" view (driving the prefetcher writer's NoC stream) and a "local" shadow (driving the unpacker's CB-state and the per-CB `unpack_*[]` metadata arrays). Bytes match at L1 (U37). Metadata matches (U41). But the LLK matmul fast-path uses `TT_MOP(0, ct_dim-1, ...)` to emit a packed sequence of TTI_UNPACR instructions out of a "replay buffer" programmed at `_llk_unpack_AB_matmul_init_` time. If the replay buffer's encoding of "fetch shared-exp from this offset" was baked in for a `fifo_page_size` that's CORRECT for the canonical CB allocation but differs from the dual-index allocation, the packed instruction would fetch the wrong exponent for the wrong tile-stride. This is plausible only if the dual-index allocation's `fifo_page_size` shadow differs from what the LLK init time read — which U41 ruled out by direct probe.

2. **SrcA "AllowedClient" semaphore race under the gathered MOP cadence.** Per `UNPACR_Regular.md:354-356`:
   ```
   while (SrcA[Bank].AllowedClient != SrcClient::Unpackers) { wait; }
   ```
   The unpacker waits for SrcA bank ownership before issuing decoded datums. If the prefetcher's gathered pattern issues `TTI_UNPACR(SrcA, ...)` to a bank that's transiently still owned by the matrix unit (from the previous matmul block), the wait fires AS EXPECTED — but if there's a race window where AllowedClient flips back to Unpackers before the actual fetch is ready, the unpacker MIGHT consume stale state. Plausible only if the LLK MOP timing is silently broken under the gathered cadence. Not addressable without Tenstorrent's silicon-level instrumentation.

3. **REG2_Force_shared_exp accidentally enabled.** If the gathered path leaves `THCON_SEC1_REG2_Force_shared_exp` set (e.g. from a previous reconfig that didn't clear it), the unpacker would ignore the per-tile shared exp bytes and use `FORCED_SHARED_EXP_shared_exp` instead. For BFP8 (where the shared exp is meaningful), this would corrupt every output. For BFP4 with the same Force-bit set, the corruption would be different but also bad — so this hypothesis predicts BFP4 ALSO breaks, which contradicts observation. Implausible.

4. **A subtle ordering bug in `mm_block_init` between the first ELF's init and subsequent ELF's call.** Since `mm_block_init` runs once per ELF launch, and each ELF has a different in1_cb_id but the gathered factory allocates them all from the same dual-index pool, a race in init-time CFG writes could leave SEC0 / SEC1 carrying stale state for later ELFs. The U40 "force unpack reconfig" variant attempted exactly this and didn't fix. Repeating the experiment isn't useful.

None of these are addressable from sglang's plug-in surface. **The U41 handoff to upstream Tenstorrent remains the only viable forward path.**

## Hard constraints checked

- [x] No upstream PR.
- [x] No SGLang core behavior changes (this doc only; no Python edits, no protocol changes).
- [x] No destructive `rm` (no `rm` issued at all).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] No port-clear needed (no server launch).
- [x] No `HF_MODEL` usage (no server launch).
- [x] Container not bind-mounted, but no rebuild/cp performed.
- [x] No env gates added; canonical bytewise preserved by construction.

## Working state at session end

- tt-metal-sglang HEAD: **`4591a063661`** (unchanged from U41).
- sglang HEAD: **`b10dc0fe3`** + this U42 doc (pending commit).
- Container `/tt-metal/` source: unchanged.
- `_ttnn.so` / `_ttnncpp.so`: unchanged.
- `/root/.cache/tt-metal-cache/*`: unchanged from U41 session-end-clean.
- TT devices: healthy (not touched).
- Server: stopped (not launched).
- Canonical re-verify: NOT RUN (no perturbation).
- Artifacts in-container from U41 preserved:
  - `/tmp/u41_v2_dprint.log` (94 MB, 365k+ U37/U39/U40/U41 probe lines)
  - `/tmp/u22/u41_v2.resp.json` (8th-signature garbage `" WhatGBT roz无所谓我以为GBT烘干..."`)
  - `/tmp/u22/u41_canonical.resp.json` (canonical bytewise-equal control)
  - `/tmp/u41_xcorr3.py` (Sub-7 cross-correlation script)

## Commits this session

- (tt-metal-sglang) **0 commits** — no source changes warranted.
- (sglang) **1 commit** — this doc.

## Final handoff summary (unchanged from U41)

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B on Blackhole.** After 56 dispatches (S1-S10, Lead 1/2/3, A1, U1-U42), every downstream and config-register-level suspect is refuted. The remaining viable bug lives in upstream tt-metal LLK BFP8 4-face SrcA decode (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:251-347`, specifically the `else` branch at lines 333-335 emitting a single `TTI_UNPACR(SrcA, ...)` for the BFP8 4-face full-tile case) interacting with the dual-index `experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)` allocation pattern, in a regime where every byte at the consumer L1 AND every per-CB unpack-decoder metadata field AND every unpacker config-register WrapAddr threshold/subtraction value is provably correct or benign. This needs Tenstorrent LLK-source-level + silicon-state investigation; sglang has no further leverage.

Production shipping config remains canonical `tt_transformers_paged` (no prefetcher) at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9/10. The U37-U41 env-gated diagnostic harness (`SGLANG_TT_U37_PROD_BYTES`, `SGLANG_TT_U37_READ_BYTES`, `SGLANG_TT_U39_EXT_BYTES`, `SGLANG_TT_U39_CB_META`, `SGLANG_TT_U41_FACE3_PROBE`, `SGLANG_TT_U40_PROBE_TILE_DIMS`, `SGLANG_TT_U41_UNPACK_ARR_PROBE`, `SGLANG_TT_U38_SET_TILE_DIMS`, `SGLANG_TT_U40_FORCE_UNPACK_RECONFIG`, `SGLANG_TT_U40_RECONFIG_BLOCK`, `SGLANG_TT_U32_GCB_OFFSET`, `SGLANG_TT_U33_FACTORY_OFFSET`, `SGLANG_TT_U34_LCM_ALIGN`, `SGLANG_TT_U35_PER_LAYER_OFFSET`) remains the on-tree definitive reproducer + diagnostic for the upstream Tenstorrent team.
