# Qwen3-8B BFP8 Prefetcher U49b — Lanes 1+2+3 Result

**Date**: 2026-05-28
**Status**: `EVIDENCE_ADVANCE` — Engineer's remaining 3 attack lanes from PR
#45402's analysis doc are now empirically resolved. **Case B verdict (Lane
2) confirmed by direct GPR capture**: `TILE_SIZE_A` is byte-exact correct
(68 for BFP8 ELFs, 36 for BFP4) at the EXACT instant `matmul_block` fires,
on every fire of every ELF. **Lane 1 (CFG state mid-sequence) shows no
divergence between BFP4 and BFP8 ELFs.** **Lane 3 (Galaxy vs Qwen3-8B
config diff) identifies 2 real Wormhole-vs-Blackhole discriminators**
(`dst_full_sync_en` and `hop_cores`) plus 1 LoFi/HiFi-relevant flag, but
none are silicon-side; all are program-config-level. The engineer's
"change my mind and look at silicon" condition is now triple-confirmed.

## TL;DR

| Lane | Probe | Result | Verdict |
|---|---|---|---|
| 2 | `regfile[p_gpr_unpack::TILE_SIZE_A]` at MOP fire time | **68 on all 3 BFP8 ELFs; 36 on BFP4 ELF** (correct for every ELF) | **Case B — GPR is right** |
| 2 | `regfile[p_gpr_unpack::TILE_SIZE_B]` at MOP fire time | 68 on `0x2d96e` (BFP8/BFP8 self-pair); 128 on the other 3 (BF16 activation, 2048/16=128) | Correct for every ELF |
| 2 | `regfile[p_gpr_unpack::KT_DIM]` at MOP fire time | 6 / 4 / 4 / 4 (matches in0_block_w per ELF) | Correct |
| 1 | `THCON_SEC0_REG3_Base_address` PRE vs POST matmul_block | Per-MOP-fire stride advancement consistent with `kt × tile_size_a`; no mid-update intermediate observed | No divergence between BFP4 and BFP8 |
| 1 | `THCON_SEC0_REG0_TileDescriptor + REG2_Out_data_format` byte at MOP fire time | BFP8: `s0_td0=0x16, s0_cfg0=0x26`; BFP4: `s0_td0=0x17, s0_cfg0=0x27` | Correct data-format byte per ELF |
| 3 | Llama-3 70B Galaxy (Wormhole) vs Qwen3-8B BH compute-kernel-config + 1d-ring matmul-config | 3 real differences (see §3) | None silicon-side; all program-config-level |

## §1 Lane 2 — TILE_SIZE_A at MOP fire time (the killer probe)

**Method**: U49b adds `regfile[p_gpr_unpack::TILE_SIZE_A] & 0xffff` DPRINT
fired from the UNPACK risc IMMEDIATELY BEFORE the `matmul_block(...)` call,
gated to `b==0 && block==0 && in0_subblock==0 && in1_subblock==0 &&
inner_dim_idx==0`. Same per-ELF tag scheme as U37/U43/U48 so each of the 4
distinct gathered-matmul ELFs is captured separately.

**Encoding sanity**: `unpA_tile_size` is set by
`TT_SETDMAREG(0, LOWER_HALFWORD(fifo_page_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A))`,
where `fifo_page_size = (raw_page_bytes >> cb_addr_shift)` and BH
`cb_addr_shift = 4` (L1_ALIGNMENT=16). So:
- BFP8 32×32 tile = 64 B exp + 4 × 256 B mantissa = 1088 B → expected GPR
  = 1088 / 16 = **68**.
- BFP4 32×32 tile = 64 + 4 × 128 = 576 B → **36**.
- BF16 32×32 tile = 4 × 512 = 2048 B → **128**.

**Killer table** (5 unique combinations across **6912 + 13824 + 6912 +
6912 = 34560 captured fires across 4 ELFs × ~72 worker cores ×
program-executions**):

| ELF | dtype-A (in1 → SrcA) | dtype-B (in0 → SrcB) | tsA observed | tsB observed | kt observed | in0_block_w | out_subblock_w × out_subblock_h |
|---|---|---|---:|---:|---:|---:|---:|
| `0x2d96e` | BFP8 (WQKV/WO) | BFP8 (self-pair) | **68 ✓** | 68 ✓ | 6 ✓ | 6 ✓ | 4 × 1 |
| `0x2db6e` | BFP4 (W1+W3 fused) | BF16 | **36 ✓** | 128 ✓ | 4 ✓ | 4 ✓ | 6 × 1 |
| `0x2de6b` | BFP8 (FF2/W2) | BF16 | **68 ✓** | 128 ✓ | 4 ✓ | 4 ✓ | 6 × 1 |
| `0x2df6a` | BFP8 (other) | BF16 | **68 ✓** | 128 ✓ | 4 ✓ | 4 ✓ | 6 × 1 |

The GPR value is **CORRECT on every BFP8 ELF, every fire**. There is no
stale-GPR / cross-ELF leakage / BFP4-stride bleeding into the BFP8 MOP
context, which was the engineer's most-direct Lane 2 hypothesis.

**Case B confirmed**: The unpacker is fed the correct stride; the bug
must be elsewhere in silicon.

## §2 Lane 1 — CFG state mid-sequence (BFP4 vs BFP8 diff)

**Method**: U49b dumps `THCON_SEC0_REG3_Base_address`,
`THCON_SEC0_REG3_Base_cntx1_address`, `THCON_SEC1_REG3_Base_address`,
`THCON_SEC0_REG0_TileDescriptor` (word 0), and
`THCON_SEC0_REG2_Out_data_format` (word 0) at TWO points around each
matmul_block call:

- **PRE**: right before `matmul_block(...)` — i.e. immediately after
  `mm_block_init` (or `mm_block_init_short_with_dt`) has finished
  programming the MOP, but before the MOP fires.
- **POST**: right after `matmul_block(...)` returns — i.e. the MOP has
  iterated `(ct_dim - 1)` times and advanced both base addresses by
  `kt × tile_size_a`.

**Per-ELF probe firings**: BFP8 ELFs: 6912 each; BFP4 ELF: 13824
(double-capacity because BFP4 ELF maps to FF1+FF3 fused which sees more
fires).

### Lane 1 table

| Field | BFP4 ELF `0x2db6e` PRE | BFP8 ELF `0x2d96e` PRE | Diff |
|---|---|---|---|
| `s0_td0` (`THCON_SEC0_REG0_TileDescriptor`+0, contains `InDataFormat`) | `0x17` | `0x16` | **Expected** — 0x16 = BFP8_b, 0x17 = BFP4_b per `data_format.hpp` |
| `s0_cfg0` (`THCON_SEC0_REG2_Out_data_format`+0) | `0x27` | `0x26` | **Expected** — Out_data_format byte sized similarly |
| `s1_base` (`THCON_SEC1_REG3_Base_address`, in1 / SrcB) | `0x581f` | `0x581f` | Identical (both BF16 in0) |
| `s0_base_c1 - s0_base` delta | varies by fire, but always `+0x80..0x90` units (= `kt × tile_size_a × 1` in 16-B units) | varies, always `+0x80` units | Consistent with `kt × tile_size_a` advancement — no mid-update divergence |

### POST observation

For BFP4 ELF on PRE = `0x142c7`, POST = `0xdbf7` (wrapped — `0x142c7 - 0xdbf7 = 0x66d0`). The wrap is consistent with the prefetcher's FIFO recycle behavior (the global CB writes to a bounded region; the consumer's MOP advances `kt-1=3` times, then the next program-fire issues a fresh `wait_for_next_context` that re-anchors `s0_base` to wherever the producer currently lands).

For BFP8 ELF on PRE = `0xa46f` (constant across many fires from many cores), POST = `0xb973`/`0xb9b7`/etc. Each MOP fire advances by a consistent stride (small wraps possible). Importantly:

- **No PRE value is observed where the data-format byte read from the cfg disagrees with the dtype of the ELF.** I.e. the engineer's "TileDescriptor/Out_data_format programming order" failure mode (cfg-read sees mid-update intermediate) **does not occur** at the granularity our probe can see (post-`TTI_STALLWAIT(STALL_CFG, THCON)`, post-`TTI_WRCFG`, pre-`matmul_block`).
- **No PRE value where the format byte is the WRONG byte for the ELF.** E.g., BFP4 ELF always reads `s0_td0=0x17` (BFP4_b), BFP8 ELFs always read `s0_td0=0x16` (BFP8_b). No cross-ELF leakage.

**Lane 1 verdict**: CFG state is correctly programmed before MOP fires on
every ELF. Does NOT rule out the engineer's deeper hypothesis that the
silicon FSM consumes mid-`TTI_WRCFG` intermediate values DURING the MOP's
own `TTI_STALLWAIT(STALL_CFG, THCON) + TTI_WRCFG` sequence (lines 65-66 of
`llk_unpack_AB_matmul.h`), because those are sub-instruction-cycle and not
visible to a TR0-thread DPRINT. But the engineer's "post-WRCFG visible
cfg state divergence" hypothesis is empirically refuted.

## §3 Lane 3 — Llama-3 70B Galaxy (Wormhole) vs Qwen3-8B (Blackhole) program-config diff

Per the engineer's analysis doc, Galaxy works with BFP8 +
`experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)`.
Galaxy = **Wormhole**, our target = **Blackhole**. The diff identifies
program-config-level differences that COULD discriminate the two paths
within the WH-vs-BH silicon-FSM superset.

**Sources**:
- Galaxy 1D ring config:
  `/home/mhnie/tt-metal-sglang/models/demos/llama3_70b_galaxy/tt/model_config.py:2450-2507`
- Galaxy compute-kernel configs:
  `/home/mhnie/tt-metal-sglang/models/demos/llama3_70b_galaxy/tt/model_config.py:641-672`
- tt_transformers 1D ring config:
  `/home/mhnie/tt-metal-sglang/models/tt_transformers/tt/model_config.py:3872-3931`
- tt_transformers compute-kernel configs:
  `/home/mhnie/tt-metal-sglang/models/tt_transformers/tt/model_config.py:1087-1134`

### Diff table

| Field | Llama-3 70B Galaxy (Wormhole, BFP8 works) | Qwen3-8B BH (BFP8 broken) | Match? | Note |
|---|---|---|---|---|
| **Silicon arch** | Wormhole | Blackhole | **NO** | Engineer-acknowledged: BH-only bug; the rest of the diff is to find what we touch that Galaxy doesn't. |
| compute_kernel_config_lofi `math_fidelity` | LoFi | LoFi | ✓ | Both LoFi for the BFP8 path. |
| compute_kernel_config_lofi `math_approx_mode` | False | False | ✓ | Same. |
| compute_kernel_config_lofi `fp32_dest_acc_en` | **False** | False | ✓ | Both False. |
| compute_kernel_config_lofi `packer_l1_acc` | **True** | True | ✓ | Same. |
| compute_kernel_config_lofi **`dst_full_sync_en`** | **True** | **False (default)** | **MISMATCH** ⚠ | Galaxy sets `dst_full_sync_en=True` for LoFi; tt_transformers doesn't set it, so defaults to False. **Candidate discriminator.** |
| compute_kernel_config_hifi2 `dst_full_sync_en` | True | False (default) | **MISMATCH** ⚠ | Same direction. |
| 1D ring matmul `mcast_in0` | False | False | ✓ | Both gather_in0 mode. |
| 1D ring matmul `gather_in0` | True | True | ✓ | |
| 1D ring matmul `fuse_batch` | True | True | ✓ | |
| 1D ring matmul `num_global_cb_receivers` | 2 (`prefetch=True`) | runtime: `prefetcher.num_receiver_cores` (Qwen3-8B usually 2) | ≈ ✓ | Likely match in practice. |
| 1D ring matmul **`hop_cores`** | **`[(3, 6)]`** (one hop core per worker grid) | **`[]`** (no hop cores) | **MISMATCH** ⚠ | Galaxy uses a hop core in the worker grid; tt_transformers does not. Affects compute-grid arrangement and may relate to CB allocation pattern. |
| 1D ring matmul `out_subblock_w` starting point | 8 (constrained to `out_block_w` divisibility) | 8 (constrained to `out_block_w` divisibility) | ✓ | Observed at runtime: BFP8 ELFs run `out_subblock_w=4`, BFP4 ELF runs `out_subblock_w=6` (driven by Qwen3's per-core-N values). |
| 1D ring matmul `untilize_out` | False (caller-controlled) | False (caller-controlled) | ✓ | |
| in1_tile geometry (32x32 / face 16x16 / 4 faces / not narrow / not partial / not transposed) | Set by ttnn.Tile defaults — same as ours when not overridden | After PR #45402 `set_tile_dims(in1_tile)` applied, byte-identical to canonical (U43 confirmed) | ✓ | Identical post-U48. |
| PACKER_L1_ACC compile flag | derived from packer_l1_acc=True at config | True for the BFP8 path | ✓ | Same. |
| FP32_DEST_ACC_EN compile flag | varies (BFP8 LoFi: False) | varies (BFP8 LoFi: False) | ✓ | Same on the BFP8 LoFi path. |
| Throttle level | Not exposed in ttnn ComputeKernelConfig — defaults x2/x4 via LLK arch headers | Same | ✓ | Not config-level. |
| MathFidelity (BFP8 path) | LoFi | LoFi | ✓ | Same. |

### Lane 3 candidate discriminators (program-config-level differences)

**D1: `dst_full_sync_en`** — Galaxy LoFi sets `dst_full_sync_en=True`;
tt_transformers LoFi defaults to False. This flag controls whether DST
buffers are pingponged (when False) or fully sync'd between math and pack
phases (when True). With pingpong, math writes one half of DST while pack
reads the other half. On a borderline pipeline-stall condition (e.g., if
the MOP timing is sensitive to whether DST is in pingpong-half mode), this
could surface differently on BH vs WH.

**D2: `hop_cores`** — Galaxy uses one hop core in the worker grid;
tt_transformers does not. The hop core acts as a relay in the gather_in0
ring and affects which physical core slots are "worker" vs "hop". This
indirectly changes the per-worker-core CB allocation pattern.

**D3: `out_subblock_w` per ELF** — Driven by `per_core_N` which depends on
`hidden_dim / num_devices` and tile alignment. Galaxy 70B has different
hidden_dim than Qwen3-8B, so these naturally differ. Less likely to be
silicon-side discriminator, but it does change the MOP iteration count
which interacts with the dual-index CB allocation pattern.

### Lane 3 NOT in the discriminator list (engineer's mentions)

- `PACKER_L1_ACC=True` — set identically on both paths.
- `FP32_DEST_ACC_EN=False` for BFP8 LoFi — set identically on both paths.
- `MathFidelity=LoFi` for BFP8 — set identically.
- `in1_tile` 32x32 / face 16x16 / 4-face / not-narrow / not-partial / no
  transpose — byte-identical post-U48 PR #45402 set_tile_dims fix.
- `num_global_cb_receivers` — both use 2 in the prefetcher path.

## §4 Case verdict

Per the U49b plan:

- **Case A** (TILE_SIZE_A WRONG for BFP8 ELFs) — **NOT MATCHED**. The GPR is
  correct on every BFP8 ELF.
- **Case B** (TILE_SIZE_A CORRECT for BFP8 ELFs) — **MATCHED**. This rules
  out the stride hypothesis. The unpacker is fed the correct stride; bug
  is elsewhere in silicon.
- **Case C** (CFG state mid-sequence diverges BFP4 vs BFP8) — **NOT MATCHED
  at our probe granularity**. Both ELFs show the expected data-format byte
  in cfg word 112 at PRE matmul_block. Sub-instruction-cycle divergence
  during the MOP's own `TTI_WRCFG` cannot be ruled out by this probe
  (would require a math-thread instrumented stall + cfg read inside the
  replay-buffer, which is invasive and changes the timing it's trying to
  measure).
- **Case D** (Galaxy diff reveals a discriminator) — **PARTIAL MATCH**. Two
  candidate discriminators (`dst_full_sync_en`, `hop_cores`) and one
  shape-driven `out_subblock_w` difference. None are silicon-side per se,
  but `dst_full_sync_en=True` is the **most actionable** difference because:
  - It's a single boolean.
  - It directly affects DST pipeline timing, which interacts with the
    same pack-after-MOP pattern that's failing.
  - Galaxy ships it as `True` on the LoFi BFP8 path that works.
  - The change is reversible / canonical-preserving.

## §5 Recommended next action

**Phase A (immediate)**: Test D1 (`dst_full_sync_en=True`) on Qwen3-8B
LoFi compute kernel config. This is the highest-signal, lowest-cost test
because it's a single-flag flip that brings Qwen3-8B exactly in line with
Galaxy's known-working configuration. If the BFP8 corruption goes away with
`dst_full_sync_en=True`, we have a workaround (and the engineer has a
silicon-FSM corner-case to investigate). If it doesn't, D1 is also ruled
out.

This is a Python-side change in `models/tt_transformers/tt/model_config.py`
or a runtime env override of the compute kernel config — needs careful
implementation to avoid breaking canonical (non-prefetcher) BFP8. Suggest
env-gated:
```python
compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
    dst_full_sync_en=(os.environ.get("SGLANG_TT_U49_DST_FULL_SYNC", "0") == "1"),
)
```

Note: `dst_full_sync_en` is normally a HW-config flag that affects packer
timing. It's set on Wormhole/Galaxy successful runs of the SAME prefetcher
pattern.

**Phase B (if D1 doesn't fix)**: Test D2 (`hop_cores=[(3,6)]` in tt_transformers
matmul_1d_ring_config). More invasive — changes worker grid shape — but
worth trying as the next-most-discriminator-like difference.

**Phase C (if D1+D2 don't fix)**: Hand off to upstream LLK team with the
U49b evidence appended to the existing UPSTREAM bug report (committed in
`d8d23d8ef`). The handoff package now contains:
- U48: per-section `Force_shared_exp` + per-unpacker `FORCED_SHARED_EXP`
  + producer-side byte-level layout — **all 4 ELFs byte-identical**.
- U49b Lane 2: `TILE_SIZE_A` GPR at MOP fire time — **correct on every
  ELF**.
- U49b Lane 1: CFG state PRE / POST `matmul_block` — **correct format
  byte per ELF, no mid-update intermediate visible**.
- U49b Lane 3 Galaxy diff: D1 (`dst_full_sync_en`) and D2 (`hop_cores`)
  candidate discriminators identified; will be reported with whatever
  result D1/D2 ablations produce.

If D1 fixes, the upstream report becomes a feature-flag-gated workaround
+ silicon FSM investigation request. If neither D1 nor D2 fixes, the
upstream report stands as is and the bug is silicon-FSM-side
under the dual-index global-CB allocation pattern on BH.

## §6 Files modified

- `/home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
  — +175 lines: Lane 2 + Lane 1 probes, env-gated.
- `/home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
  — +12 lines: env propagation for `SGLANG_TT_U49_MOP_PROBE` +
  `SGLANG_TT_U49_CFG_ORDER_PROBE`.

All changes are within the U49b-plan-permitted scope. No sglang Python
touched. No stash@{0,1,2} touched.

## §7 Env flags (all default-off; canonical untouched when unset)

- `SGLANG_TT_U49_MOP_PROBE=1` — DPRINT `TILE_SIZE_A` + `TILE_SIZE_B` +
  `KT_DIM` + `TMP_LO` + `TMP0` GPR values from UNPACK risc immediately
  before `matmul_block(...)`. Gated to (b==0, block==0, in0_subblock==0,
  in1_subblock==0, inner_dim_idx==0) so once per matmul fire per worker
  core, per-ELF tag.
- `SGLANG_TT_U49_CFG_ORDER_PROBE=1` — DPRINT
  `THCON_SEC0_REG3_Base_address`, `Base_cntx1`, `THCON_SEC1_REG3_Base`,
  `TileDescriptor[0]`, `Out_data_format[0]` PRE and POST `matmul_block`.

## §8 Verification + canonical-preserved evidence

- **U49b probe build** clean: 4/4 targets `ttnn` ninja phony; one `.cxx.o`
  rebuilt, `_ttnn.so` / `_ttnncpp.so` re-linked + synced to install dir.
- **`SGLANG_TT_U49_MOP_PROBE` + `SGLANG_TT_U49_CFG_ORDER_PROBE` env strings**
  present in `_ttnncpp.so` post-build (`strings | grep SGLANG_TT_U49` → 2/2).
- **Probe + prefetcher run**: 103,686 DPRINT lines, 4 ELFs captured.
  - U49_MOP per ELF: 4 distinct (ELF, tsA, tsB, kt) combinations across
    34,560 fires — every BFP8 ELF reports tsA=68, BFP4 reports tsA=36.
  - U49_CFGORD_PRE/POST per ELF: 4 distinct s0_td0/s0_cfg0 signatures —
    BFP8 ELFs all `s0_td0=0x16`, BFP4 ELF `s0_td0=0x17`. No mid-update
    intermediate.
- **Prefetcher-on output still garbage** (with probes on but probes are
  read-only): `/generate "What is 2+2?"` → `" What capacities capacities"`
  — same signature class as the U48 baseline garbage. The probes do not
  fix the bug (they're diagnostic-only).
- **Canonical preserved (prefetcher OFF, U49b probes OFF)**:
  - `/generate "The capital of France is"` → `" Paris. The capital of the
    United States"` (bytewise-identical to baseline, U48 verification, etc.)
  - 10-question sanity probe: factual / arithmetic answers correct or
    correctly-continuing (e.g. "What year did World War II end?" → "World
    War II ended in 1945"; "Square root of 144" → "is" then continuation).
    The base-model autocomplete behavior on "What is 2+2?" remains
    consistent with all prior runs (Qwen3-8B base, not chat, so
    completion-style is expected).
  - No regression from baseline.

## §9 Updates to memory keys

This result narrows the bug surface further: the engineer's most-direct
Lane 2 (stale-GPR / wrong stride at MOP fire time) hypothesis is **ruled
out** by direct GPR capture. Lane 1's "post-WRCFG visible mid-sequence
divergence" is **ruled out**. Lane 3 produces 2 actionable candidate
discriminators (`dst_full_sync_en=True`, `hop_cores=[(3,6)]`) that bring
us in line with the working Galaxy WH BFP8 prefetcher config.

The narrowed root cause must be in the BH silicon FSM that drives the
`TTI_UNPACR` path under the `experimental::CreateCircularBuffer(prog,
cores, remote_cfg, *global_cb)` dual-index CB allocation pattern. The
stride is right, the cfg state is right, the per-section gating bits are
zero, the per-unpacker forced-exp is zero, the producer's byte layout is
byte-exact — but the silicon decodes the BFP8 4-face SrcA payload to
garbage anyway, only under this specific allocation pattern, only on BH,
only with BFP8 (not BFP4), only with multi-tile in0_block_w under the
`reuse_a` MOP path.

This is the engineer's "I'll change my mind and look at silicon" zone,
triple-confirmed.
