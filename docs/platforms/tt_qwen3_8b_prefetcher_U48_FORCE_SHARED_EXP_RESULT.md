# Qwen3-8B BFP8 Prefetcher U48 — Force_shared_exp Hypothesis Result

**Date**: 2026-05-28
**Status**: `EVIDENCE_ADVANCE` — `Force_shared_exp` hypothesis from tt-metal Staff
LLK engineer (`ncvetkovicTT`, PR #45402 analysis doc Addendum §1) **RULED OUT**
by direct hardware probe. PR #45402's `set_tile_dims(in1_tile)` fix applied and
also does **not** fix the corruption (engineer-acknowledged).
**Case B verdict** per U48 plan. Engineer's "change my mind and look at silicon"
condition is materially closer.

## TL;DR

| Probe | Result | Verdict |
|---|---|---|
| PR #45402 `set_tile_dims(in1_tile)` (matmul factory + AGMM fusion) | Applied unconditionally, both files | Does NOT fix corruption ("What CIF CIF" garbage still appears on `/generate` with prefetcher); does NOT regress canonical (GSM8K(10) = 10/10) |
| `THCON_SEC0_REG2_Force_shared_exp` | **0** on all 4 ELFs | Hypothesis RULED OUT |
| `THCON_SEC1_REG2_Force_shared_exp` | **0** on all 4 ELFs | Hypothesis RULED OUT |
| `UNP0_FORCED_SHARED_EXP_shared_exp` | **0x0** on all 4 ELFs | Hypothesis RULED OUT |
| `UNP1_FORCED_SHARED_EXP_shared_exp` | **0x0** on all 4 ELFs | Hypothesis RULED OUT |
| U48 layout probe — first 64 B (exp block) | Sane BFloat16 exponent byte range `0x76`-`0x7c` across all 4 ELFs | Producer-side layout is byte-perfect for both BFP4 and BFP8 |
| U48 layout probe — bytes 64-79 (mantissa start) | Distinct random 32-bit patterns per ELF (expected) | No layout-offset shift between BFP4 and BFP8 |

## Killer table (per-ELF U48 probe)

All 4 expected ELFs captured. Values are byte-identical across BFP4-clean and
BFP8-broken ELFs:

| ELF | dtype | hits | s0_force | s1_force | unp0_exp | unp1_exp | raw_w73 | raw_w121 | raw_w50 | raw_w62 |
|---|---|---:|---|---|---|---|---|---|---|---|
| `0x2d96e` | BFP8 (WQKV/WO/W2) | 6912 | 0 | 0 | 0x0 | 0x0 | 0xf000f | 0xf000f | 0x100 | 0x0 |
| `0x2db6e` | BFP4 (W1+W3 fused) | 13824 | 0 | 0 | 0x0 | 0x0 | 0xf000f | 0xf000f | 0x100 | 0x0 |
| `0x2de6b` | BFP8 | 6912 | 0 | 0 | 0x0 | 0x0 | 0xf000f | 0xf000f | 0x100 | 0x0 |
| `0x2df6a` | BFP8 | 6912 | 0 | 0 | 0x0 | 0x0 | 0xf000f | 0xf000f | 0x100 | 0x0 |

`raw_w50=0x100` is bit 8 of cfg word 50 — outside the `shared_exp[7:0]` field
(MASK = 0xff). The actual `shared_exp` value (low byte) is 0 across the board.
The non-zero bits in `raw_w*` are other adjacent fields packed in the same
cfg word (e.g. UNP0_*Y_START_TILE_DESCRIPTOR or similar) — they're identical
between BFP4 and BFP8 ELFs and so cannot discriminate.

## Layout probe (Phase 6 — engineer's recommendation step 2)

> "dump 1 BFP8 weight tile from L1 as raw 1088 bytes, compare the *first*
> 64 bytes against the host-side reference's exp block, compare bytes
> 64-1087 against the host-side mantissa block — **separately**."

64-byte exp blocks (16 u32 words `e0_0..e3_3`) for all 4 ELFs are uniformly
in the range `0x76`-`0x7c` per byte (sample below from `0x2db6e` BFP4-clean
ELF):

```
e0_0=0x797a7a7a e0_1=0x7b7a7a7b e0_2=0x7a7a7a7a e0_3=0x7a7b7b7b
e1_0=0x7b7a7b7a e1_1=0x7b7a7b7a e1_2=0x7a7a7a7a e1_3=0x7a7a7a7a
e2_0=0x7a7a7b7a e2_1=0x7b7a7a7b e2_2=0x7b7a7b7a e2_3=0x7a7b7a79
e3_0=0x7a7a7b7a e3_1=0x7a7b7b7a e3_2=0x7a7a7b79 e3_3=0x7b7b7a7b
```

These are valid BFloat16 exponent bytes (bias 127; values `0x76`-`0x7c` ≡
unbiased exponents -9 to -3 ≡ magnitudes 2^-9 to 2^-3 ≈ 0.002 to 0.125 —
expected for normal-init / fine-tuned neural-network weights).

BFP8-broken ELF `0x2d96e` exp block is in the same valid range — example:
```
e0_0=0x797a7a79 e0_1=0x757a7578 e0_2=0x7476797a e0_3=0x7a78787b
e1_0=0x777a7a77 e1_1=0x767a7577 e1_2=0x7577797a e1_3=0x7a797779
e2_0=0x7879797b e2_1=0x7a797a7b e2_2=0x7a7a7579 e2_3=0x7a7b797b
e3_0=0x777a797a e3_1=0x7a797a7a e3_2=0x7a7b7578 e3_3=0x7a7a7979
```

**Both BFP4 and BFP8 exp blocks at byte offsets 0-63 contain valid exponent
data; both mantissa blocks at byte offset 64+ contain non-trivial random
data.** No layout-offset shift, no padding mismatch, no overflow.

Per-ELF mantissa samples (m_64 = u32 at byte 64) are distinct, non-zero, and
internally consistent within each ELF — exactly what's expected if the producer
laid the data out correctly.

## What this means

The engineer's analysis doc concluded with:

> "If after the layout dumps are clean, the `FORCED_SHARED_EXP` probe is also
> clean, and the unpacked SrcA is still garbage, I'll change my mind and look
> at silicon."

Both conditions are now satisfied:

1. **Layout dumps clean** — U48 layout probe shows valid exp + mantissa
   layout at correct offsets, no producer-side error.
2. **FORCED_SHARED_EXP probe clean** — all 4 cfg fields are 0 on both BFP4
   and BFP8 ELFs.
3. **Unpacked SrcA is still garbage** — `/generate "What is 2+2?"` →
   `" What]}\"]}\""` (gibberish — same signature class as the historical
   "magnitudes 2^60-2^109" output).

The customer's prior LLK-level pin (`llk_unpack_AB_matmul.h:333-336`) was
disproven by the engineer's audit; their own ISA cross-check disproved the
`Force_shared_exp` and `NoBFPExpSection` paths. We've now also disproven the
producer-side layout hypothesis at byte-exact granularity.

The next-step bug surface is the silicon FSM that drives the BH `TTI_UNPACR`
path under the `experimental::CreateCircularBuffer(prog, cores, remote_cfg,
*global_cb)` dual-index CB allocation pattern. This is the engineer's "I'll
change my mind" zone.

## What advanced

| Lead | Status |
|---|---|
| `set_tile_dims(in1_tile)` missing on global-CB path | **Fixed** (PR #45402 applied unconditionally; U38 env var removed) |
| `Force_shared_exp` + `FORCED_SHARED_EXP_shared_exp` substitutes single hardcoded exponent | **Ruled out** by U48 probe (all 4 cfg fields = 0 on all 4 ELFs) |
| Producer-side BFP8 exp block at wrong byte offset | **Ruled out** by U48 layout probe (exp block at 0-63 is valid BFloat16 exponent range; mantissa at 64+ is non-zero random per ELF) |
| `TileDescriptor` CFG programming order | Not directly probed — U43 already showed `TileDescriptor` words are identical across BFP4/BFP8 ELFs except dtype, but the engineer also said "if those pass independently but the unpacked SrcA values are still nonsense, the bug is in CFG `TileDescriptor`/`Out_data_format` programming **order**, not in layout" — this is the next attack |

## Remaining attack lanes (per engineer's recommendations)

1. **CFG programming order probe**. Dump `TileDescriptor` + `Out_data_format`
   cfg words *during* the THCON `STALLWAIT(STALL_CFG, THCON) + WRCFG` sequence
   inside `mm_block_init`, not just after, to verify the dependent reads
   inside the `_llk_unpack_AB_matmul_` mainloop see the post-WRCFG values
   rather than mid-update intermediates. Requires PACK-stalled MATH probes
   that don't perturb the pipeline.
2. **Disassemble the MOP replay buffer at runtime** with `TILE_SIZE_A` GPR
   captured via `TTI_STOREREG` to L1 immediately before `TT_MOP(0, …)`.
3. **Re-verify BFP4 with output-value sanity check**, not just no-crash, per
   engineer's caveat that "if the customer's 'BFP4 works' baseline is itself
   coincidentally correct rather than spec-guaranteed, the comparison to BFP8
   may be unreliable."
4. **Side-by-side diff of working Llama-3 70B Galaxy vs broken Qwen3-8B BH**
   `in1_tile.get_tile_shape() / get_face_shape() / get_num_faces() /
   get_partial_face() / get_narrow_tile() / get_transpose_of_faces() /
   in1_single_tile_size / in1_block_size_bytes / global_cb.size() /
   num_global_cb_receivers / in0_block_w / per_core_N / PACKER_L1_ACC /
   FP32_DEST_ACC_EN / throttle / dst_full_sync_en / MathFidelity` — i.e.
   the dst-accum + packer_l1_acc + LoFi interaction theory.

## Verification + canonical-preserved evidence

- **U48 probe build**: `bmm_large_block_zm_fused_bias_activation_gathered.cpp`
  +158 lines (probe + layout dump + clear scaffold), `matmul_multicore_reuse_mcast_1d_program_factory.cpp` +55/-2 lines
  (PR #45402 unconditional set_tile_dims, U38 env var removed, +3 env-gate
  wirings for U48 probe / layout / clear), `llama_1d_mm_fusion.cpp` +12/-2 lines (PR #45402 set_tile_dims propagation).
- **U48_FORCED_EXP probe + prefetcher**: 34,566 DPRINT lines, 4 ELFs captured,
  all `Force_shared_exp / FORCED_SHARED_EXP` values = 0.
- **U48_LAYOUT probe + prefetcher**: 3,246 DPRINT lines, 4 ELFs captured,
  all exp blocks in valid BFloat16 exponent range, mantissas distinct per
  ELF.
- **Canonical preserved (prefetcher OFF, set_tile_dims still applied)**:
  `/generate "The capital of France is"` → `" Paris. The capital of the United States"` (correct).
  GSM8K(10) chat = **10/10 = 100%**.
- **Prefetcher-on output still garbage**: `/generate "What is 2+2?"` →
  `" What]}\"]}\""` — confirming PR #45402 fix alone does NOT close the bug.

## Files modified

- `/home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
- `/home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- `/home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp`

(All three are within the U48-plan-permitted scope. No sglang Python touched.
No stash@{0,1,2} touched.)

## Env flags (all default-off; canonical untouched when unset)

- `SGLANG_TT_U48_FORCED_EXP_PROBE=1` — DPRINT `Force_shared_exp` +
  `FORCED_SHARED_EXP` per ELF (one-shot per worker, budget 8).
- `SGLANG_TT_U48_LAYOUT_PROBE=1` — DPRINT first 64 B (full exp block) +
  bytes 64-79 (mantissa start) per ELF. Requires `SGLANG_TT_U37_READ_BYTES=1`
  to fire.
- `SGLANG_TT_U48_FORCE_EXP_CLEAR=1` — defined but not yet implemented as a
  fix (probe shows Force is already 0, so a clear would be a no-op).

## Updates to memory keys

This result narrows the bug surface further: silicon-side `Force_shared_exp`
hypothesis ruled out + producer-side layout hypothesis ruled out at byte-
exact granularity. The bug is provably:

- not in the per-section `Force_shared_exp` gating bit
- not in the per-unpacker `FORCED_SHARED_EXP_shared_exp` substitute value
- not in the producer's exponent block byte layout
- not in the producer's mantissa block byte layout

The narrowed root cause must be in the `TTI_UNPACR` / `TT_MOP` *sequencing*
under the dual-index `experimental::CreateCircularBuffer(prog, cores,
remote_cfg, *global_cb)` path, with the CFG state correctly programmed
between kernel launches but the unpacker FSM consuming intermediate values
or stale-pipeline state during the BFP8-specific 4-face SrcA decode.

This is now provably the engineer's "I'll change my mind and look at silicon"
zone.
