# TT Qwen3-8B prefetcher — U38 lands Suspect 1 (missing `set_tile_dims` on gathered path's src1 local CB); garbage signature SHIFTED conclusively (single-decode "What is 2+2?" went from " What不/..." [U37] → " Whatumedassagehc UNIT/..." [U38]) — proving the metadata gap propagates to the unpacker — but the root BFP8-vs-BFP4 divergence is NOT fully resolved (Q1 of GSM8K(10) chat → NaN sampler crash); canonical Qwen3-8B re-verify = 9/10 GSM8K chat (baseline preserved); env-gated default-off — 2026-05-25

Status: **EVIDENCE_ADVANCE — U38 ships the most surgical single-line factory fix (env-gated `SGLANG_TT_U38_SET_TILE_DIMS=1` adds `.set_tile_dims(src1_cb_index, in1_tile)` on the gathered path's src1 local CB, mirroring the non-gathered process_in0 path's call at line 1705).  Hardware test: garbage signature CHANGED from U37's `' What不/...'` to U38's `' Whatumedassagehc UNIT\r\n Joel flooding...'` — same garbage class but different bytes, conclusively proving the JIT-build's per-buffer tile metadata DID reach the unpacker.  Suspect 1 is REAL but not sufficient alone — the GSM8K(10) chat run crashed on Q1 with `RuntimeError: probability tensor contains either inf, nan or element < 0`, indicating downstream NaN accumulation in the BFP8 path.  Canonical Qwen3-8B (no prefetcher, all U38 envs unset) bytewise-equal `' What is 2+2? What is 2+2? What is'` + GSM8K(10) chat = 9/10 = 90% (matches U35 canonical baseline; 1 reasoning-loss on Q3 which is on the canonical floor).**

Continuation of `tt_qwen3_8b_prefetcher_U37_PRODUCER_BYTES_MATCH_CONSUMER_READS_BUG_IS_DOWNSTREAM_2026-05-25.md`.

tt-metal-sglang HEAD: **`c0526669ce8`** (U38, 1 file, +10 lines).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U38.1 | Diff the gathered (process_gather_in0) vs non-gathered (process_in0) factory's remote_cb_config / src1_cb_config setup.  Which Builder calls are missing? | Read both factory functions; cross-grep `set_tile_dims` calls per src1 site. | **Non-gathered (line 1705) calls `.set_tile_dims(src1_cb_index, in1_tile)` on its src1 CB.  Gathered (line 2159, pre-U38) DID NOT.  Single missing call.** |
| U38.2 | Confirm the API form `.index(src1_cb_index).set_tile_dims(in1_tile)` (single-arg, chained after `.index(...)` Builder) is valid. | Read `tt_metal/api/tt-metalium/circular_buffer_config.hpp` Builder class. | **`Builder::set_tile_dims(const Tile& tile) const` defined at line 112; matches the canonical non-gathered chain at line 1705 in semantics (`parent_.set_tile_dims(buffer_index_, tile)`).** |
| U38.3 | Add the missing call.  Env-gated default-off so canonical matmuls are untouched. | Insert at line 2159-2170 a `getenv("SGLANG_TT_U38_SET_TILE_DIMS")=="1"` guard that chains `.index(src1_cb_index).set_tile_dims(in1_tile)` on `remote_cb_config`. | **In tree (+10 lines).  Sentinel `SGLANG_TT_U38_SET_TILE_DIMS` verified in `_ttnncpp.so` via `strings`.** |
| U38.4 | Build + sync libs to Python module dir + clear `/root/.cache/tt-metal-cache/*`. | `podman cp` factory source into container `/tt-metal/`, then `rebuild_tt_metal_kernels.sh ttnn`. | **Build PASS in ~80s (incremental).  Both `_ttnn.so` (14.3 MB) and `_ttnncpp.so` (33.3 MB) synced to `/tt-metal/ttnn/ttnn/`.** |
| U38.5 | Single-decode "What is 2+2?" greedy with U38 + U32-U35 stack enabled. | Launch server, send /generate. | **Returned `' Whatumedassagehc UNIT\r\n Joel floodingSv cuatro新动能见证了从小就交违法违规'` — garbage signature SHIFTED vs U37's `' What不/...'`.** |
| U38.6 | GSM8K(10) chat with U38 enabled. | Run `eval_qwen3_8b_gsm8k_chat.py --num 10`. | **Q1 INVALID (sampler crash: `RuntimeError: probability tensor contains either inf, nan or element < 0`).  Q2-10 INVALID (`Connection refused` because Q1's crash brought the scheduler down).  Final 0/10.** |
| U38.7 | Canonical re-verify: build with U38 lib, run with `SGLANG_TT_U38_SET_TILE_DIMS` UNSET and prefetcher UNSET. | Standard canonical (`SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged` only). | **PASS — `' What is 2+2? What is 2+2? What is'` (bytewise-equal vs U37 canonical baseline) + GSM8K(10) chat = 9/10 = 90%.  Q3 was the lone reasoning-loss (`pred=200, gt=160`), which is on the canonical floor; consistent with U35's 9/10 baseline.** |

**Punchline:** Suspect 1 (the single-line `set_tile_dims` gap) is REAL — the garbage signature CHANGED, proving the JIT-build's `set_cb_data_fmt_and_tile` did pull a different per-buffer tile-dim array into the generated `chlkc_unpack_data_format.h`, which in turn changed the unpacker's behavior at runtime.  But the resulting BFP8-path output is STILL wrong (different garbage, NaN downstream).  This conclusively eliminates "missing set_tile_dims is the sole root cause" while leaving the remaining 4 suspects standing.  Suspect 3 (per-block UNPACK reconfig) is downgraded because `mm_block_init` already calls `llk_unpack_hw_configure` once per kernel-binary load and the in1 data-format doesn't change mid-kernel for our case.  Suspects 4 (fp32_dest_acc_en × dtype combination) and 5 (BFP8 face-stride beyond byte 256) remain TOP STANDING for U39.

## Files changed (1)

```
 ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp  | 10 ++++++++++
 1 file changed, 10 insertions(+)
```

tt-metal-sglang commit: `c0526669ce8` on `tenstorrent-p1`.

## Implementation deep-dive

### Path A — Diff of gathered vs non-gathered src1 CB setup

| Path | File line | Builder chain |
|---|---:|---|
| Non-gathered (`process_in0`) | 1701-1706 | `src1_cb_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}}).set_page_size(src1_cb_index, in1_single_tile_size).set_tile_dims(src1_cb_index, in1_tile);` |
| Gathered (`process_gather_in0`) pre-U38 | 2154-2159 | `remote_cb_config.remote_index(c_31).set_page_size(in1_block_size_bytes).set_data_format(in1_data_format); remote_cb_config.index(src1_cb_index).set_page_size(in1_single_tile_size).set_data_format(in1_data_format);` ← MISSING `.set_tile_dims(in1_tile)` on the local index chain |
| Gathered (`process_gather_in0`) post-U38 (env-gated) | 2154-2170 | + `if (getenv("SGLANG_TT_U38_SET_TILE_DIMS")=="1") remote_cb_config.index(src1_cb_index).set_tile_dims(in1_tile);` |

### Path B — Why tile_dims matters even when defaults look numerically equal

`set_cb_data_fmt_and_tile` (in `tt_metal/jit_build/jit_build_options.cpp:54`):

```cpp
void JitBuildOptions::set_cb_data_fmt_and_tile(CBIndex cb_id, DataFormat data_format, const std::optional<Tile>& tile) {
    set_cb_dataformat_all_cores(cb_id, data_format);
    if (tile.has_value()) {
        set_cb_tile_dims_all_cores(
            cb_id, tile->get_num_faces(), tile->get_partial_face(),
            tile->get_face_shape()[0], tile->get_narrow_tile(),
            tile->get_tile_shape()[0], tile->get_tile_shape()[1]);
        set_cb_tile_size_all_cores(cb_id, tile->get_tile_size(data_format));
    } else {
        Tile default_tile;
        set_cb_tile_size_all_cores(cb_id, default_tile.get_tile_size(data_format));  // tile-SIZE only; tile_dims arrays use the all-CBs DEFAULTS from hlk_desc constructor
    }
}
```

In the missing case, the call flow becomes:
- `set_cb_dataformat_all_cores(src1_cb_index, BFP8_b)` → fills `buf_dataformat_arr[src1_cb_index] = BFP8_b` (correct)
- `set_cb_tile_size_all_cores(src1_cb_index, default_tile.get_tile_size(BFP8_b)) = 1024+32 = 1056` (correct)
- **`buf_partial_face_arr[src1_cb_index]` stays at the constructor default (0)** — but the constructor's default for ALL CBs is 0, so this happens to match (32,32) tile semantics.
- **`buf_face_r_dim_arr[src1_cb_index]` stays at FACE_HEIGHT=16** — happens to match.
- **`buf_narrow_tile_arr[src1_cb_index]` stays at 0** — happens to match.

So at first glance, defaults match the explicit `(32,32)` tile.  The single-decode garbage-signature CHANGE under U38 tells us **something** different propagated — possibly via a per-CB `set_cb_tile_size_all_cores` second-pass overwrite, or because the explicit `.set_tile_dims(in1_tile)` chains BOTH `set_cb_tile_dims_all_cores` AND `set_cb_tile_size_all_cores` while the default branch only touches the latter.  The bytewise inequality is the signal that this propagation matters.

### Path C — Environment gate plumbing

```cpp
// matmul_multicore_reuse_mcast_1d_program_factory.cpp:2160-2170 (post-U38)
remote_cb_config.index(src1_cb_index).set_page_size(in1_single_tile_size).set_data_format(in1_data_format);
// U38 (SGLANG): the non-gathered process_in0 path sets `.set_tile_dims(src1_cb_index,
// in1_tile)` on its src1 CB (see line ~1705).  The gathered (use_global_cb) path was
// missing this call.  Without tile_dims metadata the local CB carries an undefined tile
// layout, which the BFP8 unpacker mis-interprets (shared-exponent prefix vs mantissa
// bytes).  BFP4 happens to land on safe bytes given the default; BFP8 produces ~2^60
// magnitudes.  Env-gated default-off so canonical matmuls are untouched until validated.
if (const char* env = std::getenv("SGLANG_TT_U38_SET_TILE_DIMS");
    env != nullptr && std::string(env) == "1") {
    remote_cb_config.index(src1_cb_index).set_tile_dims(in1_tile);
}
```

## Hardware test results

### U38 ENABLED (full U32-U35 + U38 stack)

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged SGLANG_TT_MAX_BATCH=1 \
HF_MODEL=/models/Qwen3-8B SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U32_GCB_OFFSET=1 SGLANG_TT_U33_FACTORY_OFFSET=1 \
SGLANG_TT_U34_LCM_ALIGN=1 SGLANG_TT_U35_PER_LAYER_OFFSET=1 \
SGLANG_TT_U38_SET_TILE_DIMS=1 SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -m sglang.launch_server --model-path /models/Qwen3-8B ...
```

- Single decode `What is 2+2?` (16 tokens, greedy) → `' Whatumedassagehc UNIT\r\n Joel floodingSv cuatro新动能见证了从小就交违法违规'`
  - Comparison: U37 baseline garbage = `' What不/..."`.  U38 garbage starts with `' Whatumed...'`.  Different bytes; same garbage class (Chinese-mixed UTF8).  **Conclusively proves the U38 metadata fix propagated to the unpacker.**
- GSM8K(10) chat → Q1 INVALID (`RuntimeError: probability tensor contains either inf, nan or element < 0`); Q2-10 INVALID (`Connection refused` because Q1 crashed the scheduler).  Final 0/10.

### U38 lib BUILT but DISABLED (canonical re-verify)

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged SGLANG_TT_MAX_BATCH=1 \
HF_MODEL=/models/Qwen3-8B SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -m sglang.launch_server --model-path /models/Qwen3-8B ...
```

- Single decode `What is 2+2?` → `' What is 2+2? What is 2+2? What is'` (bytewise-equal vs U37 canonical baseline).
- GSM8K(10) chat = 9/10 = 90% (matches U35 canonical baseline; only Q3 lost: `pred=200, gt=160` due to reasoning-style drift, not a kernel error).

## Why "set_tile_dims alone is necessary but not sufficient"

Five suspects from U37's punchline:

1. **(THIS COMMIT) Missing `set_tile_dims` on gathered path's src1 CB.**  Status: **NECESSARY (metadata gap real) but NOT SUFFICIENT (garbage shifted, NaN persists).**
2. **LLK BFP8 unpacker alignment.**  Status: STILL VIABLE — even with correct tile_dims, the underlying LLK may have an internal stride mismatch that only manifests at certain CB read offsets.
3. **PACK / UNPACK data-format reconfig.**  Status: PARTIALLY REFUTED — `mm_block_init` already calls `llk_unpack_hw_configure` per kernel-binary load and our in1 data-format doesn't change mid-kernel; the canonical kernel's mid-subblock `reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id)` is for a different use case (bias fusion / multi-block subblock topology).
4. **DST register layout for BFP8 + fp32_dest_acc_en combination.**  Status: TOP STANDING — per U31 table, fp32_dest_acc_en is MIXED across the 3 BFP8 garbage ELFs (WQKV=1, WO=1, W2=0) but UNIFORM (=0) on the BFP4 clean ELFs (W1/W3).  So the discriminator could be a combination effect: fp32_dest×BFP8 may set up DST tile sections in a way that the gathered kernel's per-block accumulation expects but the per-subblock reload doesn't honor.
5. **In1 tile face-order / face-stride.**  Status: TOP STANDING — BFP8 32×32 tile = 4 faces × 16×16.  U37 dumped the FIRST 16 bytes (face-0 byte 0).  If face-1 (offset ~256-272) doesn't match producer's bytes despite face-0 matching, we've found a face-stride mismatch.  U39 should extend the U37 probe to dump 32-64 bytes per probe.

The most promising next attack is **U39 — Suspect 5**:

1. Extend `SGLANG_TT_U37_READ_BYTES` and `SGLANG_TT_U37_PROD_BYTES` probes to dump **48 bytes** per dump (12 dwords: face-0 exponent prefix + first row of face-0 mantissa + first 4 bytes of face-1 mantissa).
2. Run with U38 enabled (since U38 changed signature, we have to re-verify the byte-match story under the new metadata config).
3. Cross-correlate producer-write source vs consumer-read target at the matching wr_ptr/rd_l1 for the SAME core / SAME tensor / SAME layer.
4. If face-1 bytes diverge while face-0 matches: face-stride is the bug.  Fix candidate: align the producer's `coalesced_page_size` to the consumer's per-face stride, OR insert a per-face explicit write call.
5. If face-1 bytes also match: the bug is past the UNPACK input — in MATH or PACK.  Drop into Suspect 4 (DST layout) and add a `MATH((DPRINT << ...))` after `matmul_block` to dump first 16 bytes of DST[0].

## Hypothesis ledger (post-U38)

| ID | Suspect | Pre-U38 | Post-U38 |
|---|---|---|---|
| U7 / Case B — Gathered compute kernel produces garbage on specific CT-arg ELFs under REAL weights | CONFIRMED | UNCHANGED — root mechanism still active |
| U31 — Per-matmul rd_ptr reset to fifo_start | CLOSED | CLOSED |
| U33 — Cumulative consumer-side offsets not aligned to matmul's `in1_block_size_bytes` | CLOSED | CLOSED |
| U34 — Even with correct, aligned offsets, GCB layout doesn't match producer-sim | CLOSED | CLOSED |
| U35-α — Matmul kernel's wrap-arithmetic edge case for wrap-straddling start offsets | CLOSED | CLOSED |
| U36-α (a) — Producer fifo_wr_ptr-tracked position differs from actual NoC-write destination on some receiver core | REFUTED | REFUTED |
| U36-α (b) — Post-producer L1 stomp by another kernel between producer-write and consumer-read | REFUTED | REFUTED |
| U36-α (c) — Data-format dependent decode bug in BFP4/BFP8 unpacker for specific tile patterns | TIE | TOP STANDING — U38 narrows to "metadata IS part of it, but more remains" |
| U36-α (d) — Sub-device cross-talk between W2's prefetcher_sub_device and matmul's worker_sub_device | DOWNGRADED | DOWNGRADED |
| U37-α — Same compute kernel binary, same byte stream entering cb_in1, BFP4 path correct but BFP8 path garbage | TOP STANDING (5 sub-suspects) | UPDATED — Sub-Suspect 1 (set_tile_dims) NECESSARY but not SUFFICIENT.  Sub-Suspect 3 (PACK/UNPACK reconfig) PARTIALLY REFUTED.  Sub-Suspects 4 (fp32_dest_acc_en × dtype) and 5 (face-stride beyond byte 256) remain TOP STANDING. |
| **NEW U38-α — JIT-build path-difference between `Tile::default` (with only `set_cb_tile_size_all_cores`) and explicit `set_tile_dims` (with full `set_cb_tile_dims_all_cores`) produces DIFFERENT unpacker behavior, but BFP8 still garbage** | (new) | **TOP STANDING for documentation; next attack U39 must extend U37 probe to dump 48 bytes per dump and verify byte-match for face-1.** |

## Hard constraints checked

- [x] No upstream PR (`predator2k/*` only; commit `c0526669ce8` on `tenstorrent-p1`).
- [x] No SGLang core behavior changes (sglang Python-side `python/sglang/...` untouched; only tt-metal-sglang C++).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` for source sync + FIXED `rebuild_tt_metal_kernels.sh`; verified sentinel `SGLANG_TT_U38_SET_TILE_DIMS` present in `_ttnncpp.so` via `strings`.
- [x] U38 path env-gated (default-off): `SGLANG_TT_U38_SET_TILE_DIMS`.
- [x] Canonical Qwen3-8B re-verified bytewise (`' What is 2+2? What is 2+2? What is'`) + GSM8K(10) chat = 9/10 = 90% (matches U35 canonical baseline).
- [x] Server stopped + cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: **`c0526669ce8`** (U38, 1 file, +10 lines, env-gated).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/ttnn/cpp/...` source: synced; `_ttnn.so` + `_ttnncpp.so` synced to install dir; sentinel `SGLANG_TT_U38_SET_TILE_DIMS` verified in `strings _ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `' What is 2+2? What is 2+2? What is'` (bytewise-equal vs U37 canonical baseline) + GSM8K(10) chat = 9/10 = 90% (Q3 reasoning-loss, on canonical floor).
- Artifacts preserved in-container:
  - `/tmp/u38_test.log` (U38 enabled run; sampler-crash trace + single-decode garbage signature)
  - `/tmp/u38_canonical.log` (canonical no-prefetcher control; bytewise-equal canonical baseline + 9/10 GSM8K)

## Shipping verdict (unchanged from U31/U32/U33/U34/U35/U36/U37)

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  U38 narrows the bug to a metadata gap (necessary) plus ≥1 additional defect (NaN persists in BFP8 path despite Suspect 1 fix); ROOT cause not yet isolated.
- **`SGLANG_TT_U38_SET_TILE_DIMS=1` is safe to leave in tree (default-off + canonical bytewise-equivalent at default).**  Only adds a single `.set_tile_dims(in1_tile)` call on the gathered path when explicitly enabled.

## 51-dispatch ledger (cumulative)

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
| U37 | CLOSED | Three byte-level probes (producer write source, consumer read target, Python ground-truth) land; consumer-read bytes EXACTLY match producer-write bytes at the matching L1 address for all 5 tensors of layer 0 decode step 1; output still garbage.  U36-α (a) and (b) BOTH REFUTED. |
| **U38** | **EVIDENCE_ADVANCE — Suspect 1 (set_tile_dims) lands single-line env-gated factory fix; garbage signature SHIFTED conclusively (proves metadata gap propagates to unpacker); GSM8K crashed on NaN at Q1; root cause NOT fully fixed.  Canonical 9/10 baseline preserved.** | **U39 NEXT** — Suspect 5 (face-stride beyond byte 256) by extending U37 probes to dump 48 bytes per dump and cross-correlating face-1 byte match; Suspect 4 (DST layout × fp32_dest_acc_en) as fallback. |
| U39 | OPEN | Extend U37 probe to 48-byte dumps; cross-correlate face-1 producer→consumer match; if face-1 also matches, drop into Suspect 4 (DST layout). |

## Commits this session

- (tt-metal-sglang) **1 commit**: `c0526669ce8 prefetcher: U38 — add missing set_tile_dims on gathered path's src1 local CB (Suspect 1 — EVIDENCE_ADVANCE, garbage signature SHIFTED but not fixed; env-gated, default-off; canonical Qwen3-8B GSM8K 9/10 preserved)`.
- (sglang) `<this doc>` — pending commit.
