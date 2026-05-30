# Codex handoff — Qwen3-8B BFP8 prefetcher matmul corruption

**Date:** 2026-05-30
**For:** Codex (or any fresh agent) picking up the BFP8 prefetcher corruption investigation
**Status:** 61 Claude dispatches (U1–U51) + Tenstorrent PR #45402 + 4 PR comments. Bug NOT fixed. Canonical (no-prefetcher) ships fine. One high-value lead remains untested on hardware.

---

## 0. READ THIS FIRST — your execution environment

A prior Codex rescue attempt **died 3 times** with:

```
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

Codex's `exec_command` runs inside a `bwrap` (bubblewrap) sandbox that tries to create a loopback network namespace. **This host forbids that syscall**, so `exec_command` fails before the shell even starts — you cannot run `pwd`, `ls`, `git`, anything.

**Workaround that lets you make progress:**
- Use your native `read_file` / `write_file` / `list_directory` tools for ALL local file operations on `/home/mhnie/sglang` and `/home/mhnie/tt-metal-sglang`. Those do not go through `bwrap`.
- For hardware (build / test on the TT devices), you cannot run `podman` if `exec_command` is sandboxed. If even `podman exec` fails under `bwrap`, you are **read/write-only** — you can analyze code and write a patch, but cannot build or test it. In that case: write the patch + a precise test recipe, and hand the build/test back to the human operator (who has un-sandboxed Bash).
- Do NOT burn time retrying `exec_command` for `pwd`/`ls` — if the first call returns the `bwrap` error, it will fail for the whole session. Pivot to file-tools-only mode immediately.

If you (Codex) genuinely cannot run any shell command, your deliverable is: **a precise root-cause analysis + a concrete env-gated patch (written to disk) + an exact build/test recipe for the operator.** That is still extremely valuable.

---

## 1. The bug (one paragraph)

On 2× Tenstorrent Blackhole P150a (single host, ttnn mesh shape `(1,2)`), serving **Qwen3-8B** through SGLang with the DRAM prefetcher enabled (`SGLANG_TT_USE_PREFETCHER=1`), every **BFP8** weight matmul on the `gather_in0` / `use_global_cb=true` ring-matmul path produces silently corrupted output with magnitude **~2^60–2^109**. Every **BFP4** matmul on the *same* code path is bit-correct. Under real weights the BFP8 garbage NaN-poisons the sampler → mode-collapsed gibberish, then `RuntimeError: probability tensor contains inf/nan`. The canonical path (prefetcher off, weights read from DRAM directly) is bit-correct: **GSM8K(10) chat = 9–10/10, TPOT ≈ 27 ms / 1024-1024**. The prefetcher would give a ~1.66× TPOT win (≈16 ms) if correct.

**The discriminator is dtype/tile-size, nothing else.** Per-ELF compile-time-arg table (all 4 gathered-matmul ELFs in the Qwen3-8B prefetcher path):

| ELF | matmul | dtype | `in1_single_tile_size` | result |
|---|---|---|---:|---|
| `0x2d96e` | WQKV | BFP8 | 1088 | **garbage** |
| `0x2de6b` | WO | BFP8 | 1088 | **garbage** |
| `0x2df6a` | W2 | BFP8 | 1088 | **garbage** |
| `0x2db6e` | FF1+FF3 (fused) | BFP4 | 576 | **clean** |

Same gathered compute-kernel binary, same factory code path; only the in1 dtype/tile-size differs. BFP4 = smaller tile (576 B = 64 B shared-exp + 512 B mantissa). BFP8 = larger tile (1088 B = 64 B shared-exp + 1024 B mantissa).

The magnitude signature (2^60–2^109) is the fingerprint of **exponent corruption** — mantissas paired with a wildly-wrong shared exponent. (Tenstorrent's Staff LLK engineer agrees: see §4.)

---

## 2. Repos, hardware, build/test

- **tt-metal fork:** `/home/mhnie/tt-metal-sglang`, branch `tenstorrent-p1`, HEAD `890498aa9ad`. Public mirror: `github.com/predator2k/tt-metal` (branch `tenstorrent-p1`, pushed). Upstream remote `upstream` = `tenstorrent/tt-metal`.
- **sglang fork:** `/home/mhnie/sglang`, branch `tenstorrent-p1`, HEAD `3966e0b0d`. Public mirror: `github.com/predator2k/sglang`.
- **Active LLK tree:** `tt_metal/tt-llk/` (NOT the orphan `tt_metal/third_party/tt_llk/`). Confirmed via `tt_metal/jit_build/build.cpp` + `tt_metal/jit_build/fake_kernels_target/CMakeLists.txt`.
- **Container is NOT bind-mounted.** Host `/home/mhnie/tt-metal-sglang` ≠ container `/tt-metal`. After editing a host file you MUST `podman cp` it in.

Operator build/test recipe (run by the human if Codex is sandboxed):
```bash
# Sync a C++ or Python edit into the container:
podman cp /home/mhnie/tt-metal-sglang/<path> p3a-ngram:/tt-metal/<path>
# Rebuild C++ (incremental, ~5-15 min):
bash /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh ttnn
# Clear JIT kernel cache (mandatory between kernel edits):
podman exec p3a-ngram bash -c 'rm -rf /root/.cache/tt-metal-cache/* 2>/dev/null'
# Kill any stale server (the \. escape is load-bearing — pkill self-terminates without it):
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
# Launch with prefetcher ON:
podman exec p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=Qwen/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  <YOUR_FIX_ENV_VAR=1> \
  nohup python3 -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/server.log 2>&1 &'
# Wait for /health_generate=200, then:
podman exec p3a-ngram python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 10
# >=7/10 == fixed. Then re-verify canonical (drop SGLANG_TT_USE_PREFETCHER + your fix) must stay >=9/10.
```

**Hard constraints:** push only to `predator2k/*` forks (no upstream PRs); any fix MUST be env-gated default-off so canonical is preserved; never regress canonical; never `rm -rf $VAR/*` without `[ -n "$VAR" ]` guard; never `git stash pop`/`drop` (3 WIP stashes preserved on tt-metal-sglang).

---

## 3. What is RULED OUT (empirically, on hardware — do NOT re-chase)

| # | Hypothesis | How ruled out |
|---|---|---|
| — | Byte transport producer→consumer L1 | U37+U41: byte-exact 10/10 match incl. face-3 mantissa (bytes 832–1087) |
| — | All THCON_SEC0/SEC1 unpacker cfg registers | U43: identical between BFP4-clean and BFP8-garbage ELFs (92K-line runtime DPRINT) |
| — | `Force_shared_exp` + `UNP0/UNP1_FORCED_SHARED_EXP` | U48: all zero across all 4 ELFs (this was the engineer's top suspect — refuted) |
| — | `TILE_SIZE_A` GPR at MOP fire time | U49b: correct (68 for BFP8, 36 for BFP4) — captured at the instant `matmul_block` runs, 34560 fires |
| — | `set_tile_dims` gap on global-CB local index | U38 / PR #45402: applied, preserves canonical, does NOT fix (defaults already match 32×32 4-face) |
| — | `dst_full_sync_en=True` (Galaxy diff D1) | U50: REFUTED — makes it WORSE (NaN). On BH it poisons; on WH Galaxy it's canonical |
| — | `hop_cores=[(3,6)]` (Galaxy diff D2) | U50: hangs BH silicon (that hop-core is WH-8×8-specific) |
| — | CFG programming order (STALLWAIT+WRCFG) | U49b: no post-WRCFG divergence visible |
| — | Cross-subdev WAIT_STREAM race | U16/U29: signaler handshake correct, didn't fix |
| — | DST register stale state | U9/U39 |
| — | L1 allocator overlap / "stomp" | U17–U28: there is no stomp; matmul writes its own garbage |
| — | fabric/EDM ack writes, dispatcher launch_msg/kernel_config writes | U22/U24 |
| — | per-tensor GCB byte-offset, wrap arithmetic, num_layers drift | U32–U36 |
| — | Standalone reproducer (no SGLang stack) | U46/U51: CLEAN even with inter-op eltwise + 500 trace replays + real HF weights. Bug needs the full SGLang stack |

**Important consequence of U46/U51:** the bug does NOT reproduce in an isolated tt-metal pytest that runs the same 5 ring matmuls through the same prefetcher GlobalCB with deterministic/real weights and 500 trace replays. Something in the *full* production context (CCL + RMSNorm + RoPE + SDPA between matmuls, OR 36-layer `prefetcher.run()` driven by the SGLang sub-device scheduler, OR prefill+decode concurrency) is required. The standalone scaffold **cannot** drive `prefetcher.run()` with `num_layers>1` (it hangs — needs production sub-device scheduling).

---

## 4. Tenstorrent's response — PR #45402 (READ THIS)

`tenstorrent/tt-metal#45402` (open, by `ncvetkovicTT`, a Staff LLK engineer) is the direct downstream of our handoff report. View it:
```
gh pr view 45402 --repo tenstorrent/tt-metal --json body
gh pr diff 45402 --repo tenstorrent/tt-metal     # includes his analysis doc qwen3_8b_bfp8_prefetcher_llk_analysis.md
```
(If you can't run `gh`, the analysis is summarized below and quoted in our `tt_qwen3_8b_prefetcher_U48_*` doc.)

His conclusions:
1. **"The bug is not in LLK."** The `llk_unpack_AB_matmul.h:333-336` branch we pinned emits a single data-format-agnostic `TTI_UNPACR(SrcA,…)`; all behavior is CFG-driven, and CFG is verified correct. Same branch serves the working BFP4 case.
2. He found + fixed the missing `.set_tile_dims()` (the PR) but explicitly: **"I do not believe this fix is the root cause."** (Confirmed by us — applied, doesn't fix.)
3. **"Where I think the actual bug is" — his suspect #1 (still UNTESTED on hardware):**
   > "If the prefetcher's `dram_prefetcher_program_factory.cpp` chooses `max_tile_size_df` and `max_tile_size` over the tensor set (lines 100–103), and any tensor in the set has a different dtype or tile shape from `in1` on the consumer, then bytes can be byte-identical between writer and reader yet **the exponent block sits at the wrong offset relative to the consumer's mantissa block.** Worth dumping the actual L1 contents of one full tile and asking 'is the exp block where THCON `TileDescriptor` says it is?' — not 'do the bytes match what the producer wrote'."
4. His "I'll change my mind and look at silicon" condition — we satisfied it (layout clean, FORCED_SHARED_EXP clean, SrcA still garbage) in U48. But **his suspect #1 layout/stride mismatch was never directly tested** — U37/U41 proved *byte equality*, not *exponent-at-correct-offset*.

---

## 5. THE LEAD TO CHASE — GCB mixed-dtype stride / exponent-offset mismatch

This is the highest-value untested hypothesis. Here is the exact code I (Claude) traced before handing off.

### Producer side — `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp`

```cpp
// lines 102-108: GCB SLOT sizing uses the MAX across the mixed BFP4+BFP8 set
uint32_t max_block_tiles  = *std::max_element(tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());
uint32_t max_tile_size    = *std::max_element(tensor_tile_sizes.begin(), tensor_tile_sizes.end());   // = 1088 (BFP8)
tt::DataFormat max_tile_size_df = tensor_data_formats[max_tile_size_tensor_idx];                      // = BFP8
uint32_t max_block_size_per_reader_core = max_tile_size * max_block_tiles;

// line 128: the SENDER's staging CB uses uniform max_tile_size slots, max_tile_size_df format
uint32_t reader_cb_single_tile_size = max_tile_size;   // 1088 for ALL tensors incl. BFP4

// lines 304-306: BUT coalesced page sizes are computed PER-TENSOR
auto [coalesced_page_size, coalesced_num_page] = get_max_page_size_and_num_pages(
    max_page_size, block_width_in_tiles / num_receivers_per_reader, tt::tile_size(tensor_data_formats[t]));  // per-tensor tile_size: 576 BFP4 / 1088 BFP8
```

So the producer's GCB slot stride is uniform `max_tile_size=1088`, but each tensor is written with its own per-tensor coalesced page size.

### Consumer side — `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (process_gather_in0, around line 2149)

```cpp
if (use_global_cb) {
    uint32_t in1_block_size_bytes = in1_single_tile_size * in1_block_num_tiles;   // per-tensor: 1088*N (BFP8) or 576*N (BFP4)
    tt_metal::CircularBufferConfig remote_cb_config =
        tt_metal::CircularBufferConfig((global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes);
    remote_cb_config.remote_index(remote_cb_index)        // c_31
        .set_page_size(in1_block_size_bytes)              // per-tensor BLOCK size
        .set_data_format(in1_data_format);                // per-tensor dtype
    remote_cb_config.index(src1_cb_index)                 // local view
        .set_page_size(in1_single_tile_size)              // per-tensor TILE size (1088 BFP8 / 576 BFP4)
        .set_data_format(in1_data_format)
        .set_tile_dims(in1_tile);                         // PR #45402 fix (applied)
    cb_src1 = tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *global_cb);
}
```

### The hypothesis to confirm or kill

The producer's GCB **slot stride is `max_tile_size=1088` uniformly**, and the GCB's data format is `max_tile_size_df=BFP8`. The consumer reads each tensor at its **own** `in1_single_tile_size` and `in1_data_format`.

- For **BFP8** tensors: consumer tile size (1088) == producer slot stride (1088). ✓ aligned. **Yet these are the GARBAGE ones.**
- For **BFP4** tensors: consumer tile size (576) != producer slot stride (1088). Mismatch. **Yet these are CLEAN.**

This is the OPPOSITE of the naive "padding mismatch" story → so the simple version of the hypothesis is likely wrong, OR there's a subtler interaction (e.g. the GCB receiver CB inherits `max_tile_size_df=BFP8`'s implied `aligned_exp_size` for ALL tensors; when the consumer reads a BFP8 tensor at the actual 1088 stride but the GlobalCB's *receiver-side* page/exp-block geometry was set up from a DIFFERENT tensor's parameters, the per-face exponent offset within the 1088-byte tile could be off).

**Critical un-run experiment (the engineer's exact ask):** a *layout-aware* probe (not byte-equality). For one BFP8 weight tile in the consumer's L1 view, dump:
1. bytes 0–63 (the 64-byte shared-exponent block) — compare against the host torch reference's exponent block
2. bytes 64–1087 (the 1024-byte mantissa block) — compare against host mantissa block
…**separately**. If (1) passes but (2) passes too yet unpacked SrcA is still garbage → CFG `TileDescriptor`/`Out_data_format` *programming order*. If (1) is at the wrong offset → the GCB exp-block-offset mismatch is confirmed.

Also worth checking directly:
- Does the GlobalCB's *receiver-side* CB descriptor (the `*global_cb` object passed to `CreateCircularBuffer`) carry a single global page-size/data-format (BFP8 max), and does that override the per-tensor `in1_single_tile_size` the consumer thinks it's using? Trace `tt_metal/impl/buffers/global_circular_buffer.cpp` + how `experimental::CreateCircularBuffer(program, cores, remote_cfg, *global_cb)` reconciles the remote_cfg page-size with the GlobalCB's own.
- `Tile::get_tile_size` (`tt_metal/impl/data_format/tile.cpp:~70`): `aligned_exp_size = round_up(face_shape[0]*num_faces, l1_alignment)`. On BH `l1_alignment=64`: BFP8→1088, BFP4→576. Confirm BOTH paths compute this per-tensor and neither inherits a default `BFLOAT8_B_TILE_HW=1088` constant where it shouldn't (`tt_hlk_desc` constructor default — the engineer flagged this as suspect #2).

### Files to read for this lead
- `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp` (lines 24-30, 100-180, 290-345)
- `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (process_gather_in0, ~lines 2140-2180; and the per-tensor sizing ~141, 183)
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
- `tt_metal/impl/buffers/global_circular_buffer.cpp` + `.hpp`
- `tt_metal/impl/data_format/tile.cpp`
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` + `llk_unpack_common.h`
- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h`

---

## 6. Other untested lanes (lower priority)

1. **MOP replay-buffer runtime disassembly** — `TTI_STOREREG` the actual rd_ptr/base-address GPRs to L1 just before `TT_MOP(0,…)` in the gathered compute kernel, read back host-side. (U49b captured `TILE_SIZE_A`; did not capture the base-address GPR sequence the MOP body actually advances.)
2. **BFP4 output-value sanity** — the engineer warns Blackhole `UnpackRowWidth` for BFP2/4 is documented "not yet characterized" (`tt-isa-documentation .../UNPACR_Regular.md:210-214`). Our "BFP4 works" baseline might be coincidentally correct rather than spec-guaranteed. Re-verify BFP4 numerical output against a torch reference, not just "no NaN".
3. **Galaxy (WH) vs Qwen3-8B (BH) full config diff** — U49b found only D1 (`dst_full_sync_en`) + D2 (`hop_cores`) differ, both refuted. But Galaxy = Wormhole, Qwen3-8B = Blackhole; the working-vs-broken split may be purely the WH-vs-BH silicon delta on this exact path, which points back to silicon.
4. **Per-tensor exp-block probe inside the SrcA register file** — needs Tensix debug-status / logic-analyzer; this is Tenstorrent-internal-tooling territory.

---

## 7. ISA reference (already fetched locally)

`/home/mhnie/sglang/docs/platforms/tt-isa-docs/` has the relevant Blackhole/Wormhole ISA docs we pulled:
- `UNPACR_Regular.md` (the unpacker functional model — lines 103-207 address computation, 126-133 BFP exp-section gating, 199-207 WrapAddr)
- `Unpackers_FormatConversion.md` (the BFP exponent-array footnote: "Exponents usually come from a separate array in L1, one exponent per 16 datums")
- `SrcASrcB.md`, `BackendConfiguration.md`, `ADCs.md`

---

## 8. Full evidence chain (the 29 dispatch docs)

All in `/home/mhnie/sglang/docs/platforms/tt_qwen3_8b_prefetcher_*.md`. Most-relevant-recent first:
- `..._UPSTREAM_BUG_REPORT_2026-05-26.md` — the master report (also the §6.0 from-scratch clone/build/repro recipe)
- `..._U51_PRODUCTION_INGREDIENTS_RULED_OUT_2026-05-28.md`
- `..._U50_DST_FULL_SYNC_RULED_OUT_2026-05-28.md`
- `..._U49b_LANE_1_2_3_RESULT.md`
- `..._U48_FORCE_SHARED_EXP_RESULT.md`
- earlier U1–U43 docs (each names what it ruled out)

The standalone (negative) reproducer: `tt-metal-sglang/tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BFP8_corruption_BH.py` (1112 lines, all PASS — needs full SGLang stack to repro).

---

## 9. Your deliverable

1. Confirm or kill the §5 GCB exp-offset hypothesis by reading the producer/consumer/GlobalCB/Tile code carefully (you can do this with file tools even if shell is sandboxed).
2. If you find a concrete mismatch: write an **env-gated, default-off** patch (suggest env var `SGLANG_TT_U52_GCB_EXP_FIX`) to the relevant factory/kernel file on `/home/mhnie/tt-metal-sglang`, and a precise build/test recipe (§2).
3. If shell works for you: build, test on hardware (GSM8K ≥7/10 = fixed; canonical stays ≥9/10), and report numbers.
4. If you conclude it's genuinely silicon (not addressable from the program-factory / kernel / CFG layer), say so with independent reasoning — that itself closes the lead and strengthens the upstream handoff.

Push code only to `predator2k/*`. Document your result in a new `docs/platforms/tt_qwen3_8b_prefetcher_U52_*.md`.
