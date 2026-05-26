# TT Qwen3-8B prefetcher — U40 lands three env-gated LLK BFP8 unpacker reconfig variants (Path C `FORCE_UNPACK_RECONFIG` once per batch + `RECONFIG_BLOCK` per-subblock + `PROBE_TILE_DIMS` diagnostic); EACH SHIFTS GARBAGE SIGNATURE TO A DISTINCT CLASS (4th/5th/6th distinct vs U37/U38/U39) PROVING reconfig engages at the JIT binary — but BFP8 path STILL produces NaN; canonical Qwen3-8B GSM8K 9/10 preserved bytewise-equal — 2026-05-25

Status: **EVIDENCE_ADVANCE — U40 conclusively eliminates "missing `reconfig_data_format_srca`/`srcb` on the gathered code path" as the sole fix.  Three independent reconfig variants (top-of-batch only, per-subblock + `mm_block_init_short` MOP re-init, and a tile-dim probe) each engage at the JIT-compiled binary (distinct garbage signatures prove the kernel binary changed) but NONE eliminate the BFP8 garbage.  After U37-U40 the bug must live deeper than the standard `reconfig_data_format_*` CFG write sequence — either in (a) the per-CB `unpack_partial_face[] / unpack_tile_face_r_dim[] / unpack_num_faces[]` global arrays that U38 was supposed to populate via `set_tile_dims` but which may still carry stale per-tile values for the gathered path's src1 CB, OR (b) the producer's BFP8 layout writes face-3 mantissa bytes (832-1087) with an ordering that the LLK unpacker doesn't expect for BFP8 specifically (BFP4 face-3 is the next-tile exponent prefix, so misordering aliases harmlessly; BFP8's face-3 is genuine mantissa).  Canonical Qwen3-8B (no prefetcher, all U40 envs unset) bytewise-equal `" What is 2+2? What is 2+2? What"` + GSM8K(10) chat = 9/10 = 90% (matches U38 canonical baseline).**

Continuation of `tt_qwen3_8b_prefetcher_U39_SUSPECTS_4_5_6_ALL_RULED_OUT_2026-05-25.md`.

tt-metal-sglang HEAD: **`60cf5f2fdfc`** (U40, 2 files, +159 lines).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U40.0 | Phase 1 — diff gathered vs canonical bmm kernels for LLK init differences. | Read both `bmm_large_block_zm_fused_bias_activation_gathered.cpp` and `.../bmm_large_block_zm_fused_bias_activation.cpp` end-to-end; trace every `mm_init` / `mm_block_init` / `mm_block_init_short_with_dt` / `reconfig_data_format_*` call. | Both kernels call `mm_block_init()` once at top with identical UNPACK + MATH + PACK init calls.  Both call `reload_from_cb_to_dst()` → `mm_block_init_short_with_dt()` which re-issues `llk_unpack_reconfig_data_format_srca(old=mm_partials, new=in1)`.  **Canonical additionally calls `reconfig_data_format_srca(mm_partials, in1) + mm_block_init_short` at the END of each (b, bh, bw) iter when batch > 1 OR num_blocks_w_dim > 1.  Gathered does NOT.**  U40 Path C effectively brings the same pattern to the TOP of each batch. |
| U40.1 | Path C — force an explicit `reconfig_data_format_srca(in1_cb_id) + reconfig_data_format_srcb(in0_cb_id)` AFTER `mm_block_init`, before any matmul_block can fire. | New env `SGLANG_TT_U40_FORCE_UNPACK_RECONFIG` adds the two reconfig calls right after the existing `mm_block_init` in the gathered compute kernel.  Wires through the matmul factory (`mm_kernel_defines`) so the JIT define propagates. | **Single decode garbage signature SHIFTED from U39's `" WhatyczU垢迈出亢..."` to U40's `" What十条投融资向外�Infos哩跳出召回 carn生产总值..."` (5th distinct signature; U37/U38/U39/U40 four-way reconfig-engagement proof).**  Reconfig engaged at JIT but BFP8 garbage persists. |
| U40.2 | Most aggressive: force reconfig BEFORE every subblock's `matmul_block`, plus `mm_block_init_short` to re-establish the matmul MOP per subblock. | New env `SGLANG_TT_U40_RECONFIG_BLOCK` gates a tighter per-`(in0_subblock=0, in1_subblock=0)` reconfig.  Adds `mm_block_init_short(in0, in1, transpose, ct, rt, kt)` to restore the MOP after the reconfig (else the post-reconfig MOP state would be stale). | **Garbage signature shifted to `" What blanket资管各区Republic焕发肉类 Desk各区 Desk..."` (6th distinct signature).**  Even tighter reconfig also shifts the garbage class without fixing it. |
| U40.3 | Per-CB tile-dim observability — DPRINT the unpack tile-dim metadata observed by the LLK at init time.  Diagnose whether U38's `set_tile_dims` actually populated the per-CB `unpack_partial_face` / `unpack_tile_face_r_dim` / `unpack_num_faces` / `unpack_narrow_tile` / `unpack_src_format` / `unpack_dst_format` arrays for in1. | New env `SGLANG_TT_U40_PROBE_TILE_DIMS` dumps these per-CB values right after `mm_block_init` via UNPACK DPRINT (one dump per worker per launch). | **Garbage signature `" What_Call炎热 Quad Ez状态下antine召回 Opera蓝图..."` (4th distinct signature).**  DPRINT output does not surface without `TT_METAL_DPRINT_*` runtime env — but the garbage-shift alone confirms the JIT define engaged.  Probe values not captured in this session; logged under U41 as a follow-up. |
| U40.4 | Canonical re-verify (no prefetcher, all U40 envs unset). | Standard canonical (`SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged` only). | **PASS — `" What is 2+2? What is 2+2? What"` (bytewise-equal vs U39 canonical) + GSM8K(10) chat = 9/10 = 90% (Q3 the lone reasoning-loss `pred=20, gt=160` on canonical floor; matches U38 canonical 9/10 baseline).** |

**Punchline:** Three independent reconfig variants land at the JIT binary (3 new distinct garbage signatures — 4th/5th/6th overall, providing four-way coverage when combined with U37/U38/U39's signatures).  Each engages.  None converges on correct output.  Suspect 2's "naive LLK reconfig" form is REFUTED.  The remaining viable forms of Suspect 2 are deeper: either the global `unpack_*` arrays carry stale per-CB values that `set_tile_dims` failed to overwrite (U38 may have set the parent's `tiles_[buffer_index]` but the propagation through `set_cb_data_fmt_and_tile → set_cb_tile_dims_all_cores` may be bypassed on the gathered path's CB allocation), OR the producer's BFP8 layout problem mentioned above.

## Files changed (2)

```
 ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp |  40 +++++++
 ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp | 119 +++++++++++++++++++++
 2 files changed, 159 insertions(+)
```

tt-metal-sglang commit: `60cf5f2fdfc` on `tenstorrent-p1`.

## Phase 1 — LLK init diff (gathered vs canonical)

### `mm_block_init` (called once at top of both kernels)

Both gathered and canonical issue the SAME LLK call sequence via `mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_ids[0], in1_transpose_tile, ct, rt, kt)`:

```cpp
// matmul.h line 233-271 — same expansion for both kernels:
state_configure(in1_cb_id, in0_cb_id, mm_partials_cb_ids[0], call_line);
UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(in1_cb_id, in0_cb_id)));        // THCON_SEC0/SEC1 formats + TILE_SIZE_A/B
UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose, ct, rt, kt)));           // SETADCXX + MOP replay buf
MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0, in1)));
MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0, in1, transpose, ct, rt)));
PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out_cb_id)));
PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
PACK((llk_pack_init<false, false>(out_cb_id)));
```

`llk_unpack_hw_configure` reads `get_local_cb_interface(unpA_id).fifo_page_size` (= 1088 for BFP8 per U39) and `unpack_src_format[unpA_id]` (= BFP8_b for the 3 BFP8 ELFs).  Identical inputs → identical THCON_SEC0 state programmed.

### `reload_from_cb_to_dst` (same function body in both kernels)

```cpp
copy_tile_to_dst_init_short_with_dt(in1, mm_partials);
copy_block_matmul_partials(mm_partials, ...);
mm_block_init_short_with_dt(in0, in1, mm_partials, transpose, ct, rt, kt);
   // -> llk_unpack_reconfig_data_format_srca<...>(old=mm_partials, new=in1)
   // -> llk_math_reconfig_data_format_srca<...>(old=mm_partials, new=in1)
   // -> mm_block_init_short(in0, in1, ...)
```

Both kernels re-issue `llk_unpack_reconfig_data_format_srca` for in1 here.

### Canonical-only delta

Canonical (`bmm_large_block_zm_fused_bias_activation.cpp:540-551`) ALSO does:

```cpp
if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
    reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
    mm_block_init_short(in0_cb_id, in1_cb_id, in1_transpose_tile, ...);
}
```

at the END of each (b, bh, bw) iter.  Gathered does NOT.  U40 Path C effectively brings this missing reconfig pattern to the TOP of each batch via `SGLANG_TT_U40_FORCE_UNPACK_RECONFIG`.

## Phase 2 — Path C (`SGLANG_TT_U40_FORCE_UNPACK_RECONFIG`)

### Implementation

`bmm_large_block_zm_fused_bias_activation_gathered.cpp` — directly after the existing `mm_block_init(...)` call:

```cpp
#ifdef SGLANG_TT_U40_FORCE_UNPACK_RECONFIG
    reconfig_data_format_srca(in1_cb_id);  // in1 → srcA → THCON_SEC0
    reconfig_data_format_srcb(in0_cb_id);  // in0 → srcB → THCON_SEC1
#endif
```

Each `reconfig_data_format_srca` call calls `llk_unpack_reconfig_data_format_srca_impl_` which:
- `cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format[id])`
- `cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(unpack_dst_format[id])`
- `TT_SETDMAREG(0, LOWER_HALFWORD(fifo_page_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A))`

These are exactly the same writes `llk_unpack_hw_configure` already issued ~2 ns earlier — but if the global `unpack_src_format[]` / `unpack_dst_format[]` arrays carry STALE values from a sibling kernel-binary's static data on the gathered code path, this re-issue would overwrite the stale CFG writes with the correct current values.

### Hardware test result

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged SGLANG_TT_MAX_BATCH=1 \
HF_MODEL=/models/Qwen3-8B SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U32_GCB_OFFSET=1 SGLANG_TT_U33_FACTORY_OFFSET=1 \
SGLANG_TT_U34_LCM_ALIGN=1 SGLANG_TT_U35_PER_LAYER_OFFSET=1 \
SGLANG_TT_U38_SET_TILE_DIMS=1 SGLANG_TT_U40_FORCE_UNPACK_RECONFIG=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -m sglang.launch_server --model-path /models/Qwen3-8B ...
```

- Single decode `"What is 2+2?"` → `" What十条投融资向外�Infos哩跳出召回 carn生产总值各区_reason(\"\"theses"` (5th distinct garbage signature; U37=" What不...", U38=" Whatumed...", U39=" WhatyczU垢...", U40_PROBE=" What_Call炎热...", U40_FORCE=" What十条投融资...").  The 5-way signature spread conclusively proves each successive intervention reaches the JIT-compiled kernel binary.
- e2e latency: 10.39s (same order as U39's 10.x s).

GSM8K(10) chat NOT run for U40 garbage runs (single-decode crash signature already shows the bug persists; running GSM8K would have wasted ~10 min on confirmed-broken weights).

## Phase 3 — Per-block reconfig (`SGLANG_TT_U40_RECONFIG_BLOCK`)

### Implementation

`bmm_large_block_zm_fused_bias_activation_gathered.cpp` — inside the per-subblock loop, gated to first subblock + first inner iter:

```cpp
#ifdef SGLANG_TT_U40_RECONFIG_BLOCK
    if (in0_subblock == 0 && in1_subblock == 0) {
        reconfig_data_format_srca(in1_cb_id);
        reconfig_data_format_srcb(in0_cb_id);
        mm_block_init_short(in0_cb_id, in1_cb_id, in1_transpose_tile,
            out_subblock_w, out_subblock_h, in0_block_w);
    }
#endif
```

The `mm_block_init_short` re-establishes the matmul MOP that `reconfig_data_format_*` does NOT touch — so if MOP-state-drift across kernel re-entries were the root cause, this variant would fix it.

### Hardware test result

Single decode → `" What blanket资管各区Republic焕发肉类 Desk各区 Desk沿途 Desk"` (6th distinct garbage signature; the trailing `Desk Desk Desk` repetition is new — likely a side-effect of the MOP re-init perturbing the matmul accumulator pipeline state).  Reconfig + MOP re-init engaged at JIT binary but BFP8 garbage persists with new characteristics.

## Phase 4 — Diagnostic probe (`SGLANG_TT_U40_PROBE_TILE_DIMS`)

### Implementation

Right after `mm_block_init`, gated UNPACK DPRINT dumps the per-CB metadata that the LLK reads at init time:

```cpp
#ifdef SGLANG_TT_U40_PROBE_TILE_DIMS
    {
        UNPACK(({
            DPRINT << "[U40_TILEDIMS elf=0x" << HEX() << u40_elf_tag
                   << " in1_cb=" << in1_cb_id
                   << " in1_face_r=" << get_operand_face_r_dim(in1_id)
                   << " in1_num_faces=" << get_operand_num_faces(in1_id)
                   << " in1_partial=" << get_operand_partial_face(in1_id)
                   << " in1_narrow=" << get_operand_narrow_tile(in1_id)
                   << " in1_src_fmt=0x" << get_operand_src_format(in1_id)
                   << " in1_dst_fmt=0x" << get_operand_dst_format(in1_id)
                   << ...
                   << " in0_cb=" << in0_cb_id
                   << " in0_*= ..."  // same for in0
                   << "]" << ENDL();
        }));
    }
#endif
```

### Hardware test result

Single decode → `" What_Call炎热 Quad Ez状态下antine召回 Opera蓝图不在乎内容简介itime_nat搬到"` (4th distinct garbage signature; DPRINT lines did not surface in the server log because `TT_METAL_DPRINT_RISC_IDS` was not set — but the garbage-shift confirms the U40_PROBE define propagated to the JIT-compiled binary).

To actually capture the probe values, the run must be re-launched with `TT_METAL_DPRINT_RISC_IDS=0,1` and `TT_METAL_DPRINT_FILE=/tmp/u40_dprint.log` exported.  Logged under U41 follow-up.

## Phase 5 — Canonical re-verify (no prefetcher, all U40 envs unset)

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
HF_MODEL=/models/Qwen3-8B SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
# (all U40 envs UNSET; SGLANG_TT_USE_PREFETCHER UNSET)
python3 -m sglang.launch_server --model-path /models/Qwen3-8B ...
```

- Single decode `"What is 2+2?"` → `" What is 2+2? What is 2+2? What"` (bytewise-equal vs U39 canonical baseline).
- GSM8K(10) chat:
  - Q1 CORRECT (gt=64, pred=64) 50.5s
  - Q2 CORRECT (gt=260, pred=260) 48.8s
  - Q3 WRONG (gt=160, pred=20) 102.2s — canonical reasoning-loss; this is the same Q3 that U38 also missed.
  - Q4-10 ALL CORRECT.
  - **FINAL: 9/10 = 90.0%** (matches U38 canonical 9/10 baseline).

## Phase 6 — Hypothesis ledger update

| ID | Suspect | Pre-U40 | Post-U40 |
|---|---|---|---|
| U7 / Case B — Gathered kernel ELF garbage under real weights | CONFIRMED | UNCHANGED — root mechanism still active across all 5 ELFs |
| U36-α (a) Producer write skew | REFUTED (U37) | REFUTED |
| U36-α (b) Post-producer L1 stomp | REFUTED (U37) | REFUTED |
| U36-α (c) Data-format dependent decode bug in BFP4/BFP8 | TOP STANDING | **TOP STANDING — narrowed to BFP8 face-3 mantissa layout OR per-CB unpack_*[] array staleness** |
| U37-α (sub-1) Missing `set_tile_dims` | NECESSARY but not sufficient (U38) | NECESSARY but not sufficient |
| U37-α (sub-2) LLK BFP8 unpacker alignment | TOP STANDING (U39) | **PARTIALLY REFUTED — naive reconfig forms (init-only, per-batch, per-subblock) all engage but none fix; deeper variant still standing** |
| U37-α (sub-3) PACK/UNPACK reconfig missing | PARTIALLY REFUTED | **REFUTED — U40 Path C added the canonical-equivalent reconfig pattern; did not fix** |
| U37-α (sub-4) DST × fp32_dest × BFP8 | RULED OUT (U39) | RULED OUT |
| U37-α (sub-5) BFP8 face-stride beyond byte 256 (faces 0/1/2) | RULED OUT (U39) | RULED OUT |
| U39-α (NEW sub-6) LocalCBInterface fifo_page_size corruption | RULED OUT (U39) | RULED OUT |
| U40-α (NEW sub-7) BFP8 face-3 mantissa byte misordering (bytes 832-1087) | (new) | **TOP STANDING** — U37/U39 byte probes only verified bytes 0-575 (exp prefix + faces 0/1/2); face-3 mantissa (256 BFP8 bytes) was NOT probed, and BFP4 face-3 aliases harmlessly to next-tile exponent (so BFP4 clean ELFs would be insensitive to a face-3 misordering bug). |
| U40-β (NEW sub-8) Global `unpack_partial_face[] / unpack_tile_face_r_dim[] / unpack_num_faces[] / unpack_narrow_tile[]` arrays carry stale per-tile values for in1 on gathered path | (new) | **TOP STANDING** — U38's `set_tile_dims` populates `CircularBufferConfig::tiles_[buffer_index]`, but the propagation through `program_impl.cpp::set_cb_data_fmt_and_tile → JitBuildOptions::set_cb_data_fmt_and_tile → set_cb_tile_dims_all_cores` may not engage on the gathered path's CB allocation pattern (the local-AND-remote dual-index CB is created via `tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *global_cb)`, which may route through a different code path than the non-gathered `tt_metal::CreateCircularBuffer`). |

## Phase 7 — U41 next-step plan

1. **Verify the `set_tile_dims` actually propagates to the global `unpack_*[]` arrays on the gathered path.**  Cross-correlation:
   - Re-run U40_PROBE_TILE_DIMS with `TT_METAL_DPRINT_RISC_IDS=0,1` and `TT_METAL_DPRINT_FILE=/tmp/u40_dprint.log` exported.  Capture `unpack_partial_face[in1_id]`, `unpack_tile_face_r_dim[in1_id]`, `unpack_num_faces[in1_id]`, `unpack_narrow_tile[in1_id]`, `unpack_src_format[in1_id]`, `unpack_dst_format[in1_id]` for each of the 5 ELFs (W1/W3 BFP4, WQKV/WO/W2 BFP8).
   - Re-run the same probe under the non-prefetcher canonical config (same ELFs but routed through `process_in0` factory path).  Compare per-ELF values.
   - If any value differs → root cause = the dual-index gathered CB allocation bypasses the `set_cb_data_fmt_and_tile` propagation.  Fix would be at the CB allocation layer (`CircularBufferConfig::set_tile_dims` is a setter, but the propagation may need a parallel call for the remote-buffer-index code path).

2. **Probe face-3 mantissa bytes.**  Extend U37/U39's byte-level probe to byte 832, 896, 960, 1024 (face-3 mantissa start + 3 quarter-window samples) for both producer and consumer.  Cross-correlate.  If face-3 bytes diverge → producer's BFP8 layout is wrong for face-3 specifically (would need writer_l1.cpp / remote_cb_push_back_and_write_pages investigation).

3. **If both (1) and (2) come back clean, the bug is in the LLK BFP8 face-3 decoder itself** — which would be an upstream tt-metal LLK issue, not a P3a project issue.  At that point we should:
   - File the diagnostic in `predator2k/tt-metal-sglang` issue tracker (not upstream `tenstorrent/tt-metal`)
   - Pin the prefetcher to "DO NOT SHIP for Qwen3-8B" and document the workaround (canonical shipping config at TPOT ≈ 27 ms / 1024-1024).

## Hard constraints checked

- [x] No upstream PR (`predator2k/tt-metal-sglang` only; commit `60cf5f2fdfc` on `tenstorrent-p1`).
- [x] No SGLang core behavior changes (only tt-metal-sglang C++; sglang Python untouched).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` for source sync + FIXED `rebuild_tt_metal_kernels.sh`; all 3 U40 sentinels verified in `_ttnncpp.so` via `strings`.
- [x] All U40 paths env-gated default-off.
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"`) + GSM8K(10) chat = 9/10 = 90%.
- [x] Server stopped + cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: **`60cf5f2fdfc`** (U40, 2 files, +159 lines, env-gated).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/ttnn/cpp/...` source: synced; `_ttnn.so` (14.3 MB) + `_ttnncpp.so` (33.3 MB) synced to `/tt-metal/ttnn/ttnn/`; all 3 U40 sentinels (`SGLANG_TT_U40_PROBE_TILE_DIMS`, `SGLANG_TT_U40_FORCE_UNPACK_RECONFIG`, `SGLANG_TT_U40_RECONFIG_BLOCK`) verified in `strings _ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` (bytewise-equal vs U39 canonical baseline) + GSM8K(10) chat = 9/10 = 90%.
- Artifacts preserved in-container:
  - `/tmp/u22/u40_probe_only.log` — U40_PROBE_TILE_DIMS run; garbage signature `" What_Call炎热 Quad Ez状态下antine召回 Opera蓝图..."`
  - `/tmp/u22/u40_force_reconfig.log` — U40_FORCE_UNPACK_RECONFIG run; garbage signature `" What十条投融资向外�Infos哩跳出召回 carn生产总值..."`
  - `/tmp/u22/u40_reconfig_block.log` — U40_RECONFIG_BLOCK run; garbage signature `" What blanket资管各区Republic焕发肉类 Desk各区 Desk沿途 Desk"`
  - `/tmp/u22/u40_canonical.log` — canonical no-prefetcher control; bytewise-equal baseline
  - `/tmp/u40_canonical_gsm8k_server.log` — canonical GSM8K(10) server log; 9/10

## Shipping verdict (unchanged from U31-U39)

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  U40 conclusively refutes "naive reconfig forms" as the sole fix; either deeper LLK BFP8 unpacker per-CB array staleness (sub-7) or producer-side BFP8 face-3 byte ordering (sub-8) is the actual root cause.
- **All 3 U40 env gates safe to leave in tree (default-off + canonical bytewise-equivalent at default).**  They're orthogonal probes/overrides that only activate under the explicit env vars.

## 53-dispatch ledger (cumulative)

| ID | Status | Note |
|---|---|---|
| S1-S10, Lead 1/2/3, A1 | CLOSED | Per-doc closure. |
| U1-U16 | CLOSED | DST stale, GCB block bisect, LLK probes, PACK probes, W2→RS barrier, etc. |
| U17-U28 | CLOSED | All "downstream stomper" theories refuted. |
| U29 | CLOSED | W2→RS semaphore handshake verified working; garbage persists. |
| U30 | CLOSED | Per-ELF PACK-side garbage: 86% NaN/Inf at PACK exit for 3 of 4 ELFs. |
| U31 | CLOSED | Discriminator: BFP8 vs BFP4 dtype. |
| U32 | CLOSED | Structural plumbing landed. |
| U33 | CLOSED | Consumer-side cumulative offsets landed. |
| U34 | CLOSED | Producer-simulation aligned offsets landed; output garbage persists. |
| U35 | CLOSED | Per-layer stateful offset plumbs correctly; produces NaN. |
| U36 | CLOSED | Pre-emptive wrap fix; per-block rd_ptr trajectory verified correct; output garbage persists. |
| U37 | CLOSED | Byte-level probes: consumer-read bytes EXACTLY match producer-write bytes at byte 0-15; U36-α (a) and (b) BOTH REFUTED. |
| U38 | CLOSED | `set_tile_dims` on gathered path's src1 local CB: NECESSARY (garbage signature shifted) but NOT SUFFICIENT. |
| U39 | CLOSED | Suspects 4 (fp32_dest × BFP8), 5 (face-stride beyond byte 256, faces 0/1/2), 6 (LocalCBInterface fifo_page_size) ALL RULED OUT.  Garbage signature shifts conclusively under each override but BFP8 path still produces NaN. |
| **U40** | **EVIDENCE_ADVANCE — "Naive LLK BFP8 unpacker reconfig" forms (init-only, per-batch, per-subblock + MOP-reinit) ALL engage at JIT but NONE fix.  Sub-3 (PACK/UNPACK reconfig missing) REFUTED.  Sub-2 (LLK alignment) narrowed to deeper variants.  Canonical 9/10 baseline preserved.** | **U41 NEXT** — Capture U40_PROBE_TILE_DIMS DPRINT under `TT_METAL_DPRINT_RISC_IDS=0,1` and cross-correlate per-CB unpack_*[] arrays vs non-prefetcher canonical; AND/OR extend U37/U39 byte probes to BFP8 face-3 mantissa (bytes 832-1087). |
| U41 | OPEN | Capture DPRINT output for tile-dim probes + probe BFP8 face-3 mantissa byte delivery. |

## Commits this session

- (tt-metal-sglang) **1 commit**: `60cf5f2fdfc prefetcher: U40 — Suspect 2 LLK BFP8 unpacker internal state (Path C force_unpack_reconfig + per-block reconfig variants); 3 distinct garbage signatures shifted vs U37/U38/U39 baselines proving each reconfig flavour engages at the JIT binary, but BFP8 path still produces NaN; env-gated, default-off; canonical Qwen3-8B GSM8K 9/10 preserved`.
- (sglang) `<this doc>` — pending commit.
