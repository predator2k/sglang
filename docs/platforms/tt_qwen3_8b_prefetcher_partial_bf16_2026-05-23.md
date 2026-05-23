# TT Qwen3-8B prefetcher — partial-BF16 ablation (2026-05-23)

Status: **DEFINITIVE NEGATIVE — no partial-BF16 weight class fits the
prefetcher's GlobalCB on 2× P150a.** Prefetcher correctness for Qwen3-8B
is now confirmed to require tt-metal C++ work (per-bank L1 cap raise,
GlobalCB layout change, or BFP8 tile-routing fix in `ttnn.dram_prefetcher`).

This document supersedes the *fix-attempt* portion of
`tt_qwen3_8b_prefetcher_correctness_2026-05-23.md` and
`tt_qwen3_8b_prefetcher_logit_probe_2026-05-23.md` — both attempts kept
hope that "partial BF16 in the right weight class" might thread the
needle of "fits the GlobalCB" + "doesn't hit the BFP8 corruption." This
session proves it cannot.

## Bottom line

Canonical (no prefetcher) remains the production shipping config:
TPOT ≈ 27 ms / 1024-1024, GSM8K(10) chat-format = 9/10 = 90%
(re-verified this session as a sanity check).

| Variant            | Fits L1? | TT_FATAL or boot OK?       | GSM8K(10) | TPOT  | Verdict |
|--------------------|----------|----------------------------|-----------|-------|---------|
| `wqkv_bf16`        | NO       | L1 OOM at GlobalCB create  | 0/10 (crash) | n/a   | FAIL    |
| `wo_bf16`          | NO       | L1 OOM at GlobalCB create  | 0/10 (crash) | n/a   | FAIL    |
| `mlp_bf16`         | NO       | L1 OOM at GlobalCB create  | 0/10 (crash) | n/a   | FAIL    |
| `attn_bf16`        | NO       | L1 OOM at GlobalCB create  | 0/10 (crash) | n/a   | FAIL    |
| `full_bf16` (prior)| NO       | L1 OOM at GlobalCB create  | 0/10 (crash) | n/a   | FAIL    |
| canonical (sanity) | yes      | OK                         | **9/10**  | 27 ms | ship    |

Every BF16 lift that includes ANY single matmul weight class hits the
same hard wall: **a 768-tile BF16 block needs 1,572,864 bytes per L1
bank, but the per-bank cap is 1,461,760 bytes** (`Out of Memory:
Not enough space to allocate 62914560 B L1 buffer across 40 banks,
where each bank needs to store 1572864 B, but bank size is 1461760 B`).

This is an L1 bank-capacity ceiling enforced by tt-metal's
`bank_manager.cpp:462`. It is independent of which weight class is
lifted; the GlobalCB tile-pitch is determined by the LARGEST tile size
across all prefetched tensors × the LARGEST block-tile count across all
prefetched tensors (C++ cross-product at
`dram_prefetcher_program_factory.cpp:108`). Once any prefetched tensor
goes BF16, every BFP8 tensor with a large block-tile count (e.g. the
MLP FF1 at 768 tiles) is multiplied by the BF16 tile pitch (2048 bytes
vs 1088 for BFP8), and the product blows the per-bank cap.

## Reproduction matrix

For each variant, the launch sequence is identical (only
`SGLANG_TT_QWEN3_PRECISION` changes):

```bash
podman exec p3a-ngram bash -c '\
    pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c '\
    SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
    SGLANG_TT_MAX_BATCH=1 \
    HF_MODEL=Qwen/Qwen3-8B \
    SGLANG_TT_USE_PREFETCHER=1 \
    SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
    SGLANG_TT_QWEN3_PRECISION=<VARIANT> \
    nohup python3 -m sglang.launch_server \
        --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
        --device tenstorrent --context-length 4096 \
        --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
        --skip-server-warmup --max-running-requests 1 --trust-remote-code \
        --attention-backend torch_native > /tmp/qwen3_8b_<variant>.log 2>&1 &'
# wait for /get_model_info to 200
podman exec p3a-ngram python3 \
    /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 10
```

All four `*_bf16` variants reach `/get_model_info=200` at server boot
(weight-load and model-init succeed — the L1 trip is at the FIRST
decode forward, not at model construction). The chat-format gate then
hits `RemoteDisconnected` on Q1 as the server scheduler dies with
`SIGQUIT`.

Server crash signature (identical across all four variants):

```
[DRAM Prefetcher] Creating global CB with size: 1572864
TT_FATAL: Out of Memory: Not enough space to allocate 62914560 B L1 buffer
  across 40 banks, where each bank needs to store 1572864 B,
  but bank size is 1461760 B (allocated: 28928 B, free: 1432832 B,
  largest free block: 1430784 B)
RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/allocator/bank_manager.cpp:462: false
```

The CB requested size is the C++ cross-product
`max_tile_bytes * max_block_tiles`:

* `max_tile_bytes` = 2048 (BF16) for any variant that lifts any weight
  to BF16
* `max_block_tiles` = 768 (MLP FF1 at hidden_dim=12288, per-device =
  6144, ring_size=16, → 4096/32 × 6144/32 / 16 = 128 × 192 / 16 = 768
  block tiles; BFP8 doesn't change tile count, only tile size)
* Product = 2048 × 768 = **1,572,864 bytes per bank**
* L1 bank cap on 2× P150a (40 banks, ring_size=16) = 1,461,760 bytes
  → fails by ~7.6%.

## Confirming the negative

### Python-side CB-sizing bug fixed (no-op on canonical path)

While instrumenting, I found a Python ↔ C++ mismatch in the GlobalCB
sizing path that initially masked the L1 OOM as a confusing TT_FATAL
("largest tensor 1572864 must fit in global cb 835584"). The original
`prefetcher.insert_tensor` computed
`max_tensor_block_size = max(per_tensor_tile_bytes * per_tensor_block_tiles)`
— i.e. **per-tensor** product. But the C++ runtime check at
`dram_prefetcher_program_factory.cpp:108` uses the **cross-product**
`max(tile_bytes) * max(block_tiles)` across ALL tensors. When all
weights are the same dtype (canonical BFP8, or full BF16), the two
formulas agree. When BF16 and BFP8 are mixed (partial-BF16 modes), the
Python under-sizes the CB by exactly `bytes_in_tile[bf16] /
bytes_in_tile[bfp8] = 2048 / 1088 ≈ 1.88×`.

Fix: track `_max_tile_bytes` and `_max_block_tiles` separately and use
the cross-product to size the CB
(`tt-metal-sglang/models/tt_transformers/tt/prefetcher.py:729-749`).
Verified no-op on canonical-BFP8 path (GSM8K(10) = 9/10 = 90% sanity
check passes unchanged).

With the fix, all four partial-BF16 variants ROLLOVER from the
old TT_FATAL ("largest tensor must fit in global cb") to the
TRUE underlying failure: L1 OOM on the requested cross-product size.
This proves the L1 cap is the real architectural ceiling, not the
Python sizing math.

### The four new precision presets are kept in tree

* `wqkv_bf16` — only WQKV in BF16
* `wo_bf16` — only WO in BF16
* `mlp_bf16` — only MLP (FF1/FF3 + FF2) in BF16
* `attn_bf16` — WQKV + WO in BF16 (q_norm/k_norm not in TensorGroup
  enum, so RMSNorm weights stay on default path; documented in preset
  comment)

All four are env-gated behind `SGLANG_TT_QWEN3_PRECISION=<name>` so
they are zero-cost on the canonical shipping path. They are preserved
in tree as repro scaffolding for the next session — when (and only
when) the tt-metal-side L1 cap or layout fix lands, these presets
become the immediate validation matrix.

## Architectural attack vectors (tt-metal C++ scope)

To enable any prefetcher path that mixes BF16 with BFP8 on 2× P150a,
one of the following must change in tt-metal C++:

1. **Raise per-L1-bank cap above 1.572 MB** (currently 1.461 MB on
   2× P150a with ring_size=16). Likely requires either fewer banks
   (lower ring_size, hurting decode parallelism) or a different L1
   sub-device carve-up. File:
   `tt_metal/impl/allocator/bank_manager.cpp` and the sub-device init
   chain that sets `bank_size`.
2. **Per-tensor CB sizing in `ttnn.dram_prefetcher`**: change the
   GlobalCB pitch from `max(tile_size) * max(block_tiles)` cross-product
   to per-tensor `tile_size * block_tiles`. This requires the matmul
   consumer side (`matmul_multicore_reuse_mcast_1d_program_factory.cpp`)
   to know per-tensor tile pitch on read, not a uniform CB pitch.
   Substantial cross-op redesign.
3. **Reduce MLP FF1 block-tile count below 768**: change ring matmul
   tile config so the MLP per-core block is smaller. This is a
   matmul-config change in
   `models/tt_transformers/tt/model_config.py:matmul_1d_ring_config`
   AND requires the ring matmul kernel to support a smaller block
   shape, which currently hits the validator at
   `matmul_device_operation.cpp:514-518` constraints.
4. **Fix the BFP8 shared-exponent tile-routing bug** in
   `ttnn.dram_prefetcher` so canonical-BFP8 prefetcher (which DOES fit
   L1) is correct. The 2026-05-23 logit-probe doc fingered this as the
   underlying corruption — see `tt_qwen3_8b_prefetcher_logit_probe_2026-05-23.md`
   for the ATTACK-2 evidence. If the BFP8 routing bug is fixed in
   C++, the all-BFP8 prefetcher would ship at TPOT ≈ 16-20 ms (1.66×)
   with no L1 issues; partial-BF16 modes become unnecessary.

Option 4 is by far the cleanest path. The partial-BF16 ablation in
this session was the last Python-side hypothesis; it's now closed.

## Files touched this session

1. `tt-metal-sglang/models/tt_transformers/tt/model_config.py` —
   added 4 env-gated presets (`wqkv_bf16`, `wo_bf16`, `mlp_bf16`,
   `attn_bf16`) under `SGLANG_TT_QWEN3_PRECISION`. Zero-cost on
   canonical path.
2. `tt-metal-sglang/models/tt_transformers/tt/prefetcher.py` —
   `insert_tensor` GlobalCB-sizing math fixed to match C++ cross-product
   semantics. No-op on uniform-dtype paths; uncovered the TRUE L1 OOM
   on partial-BF16 paths (previously masked as
   `largest tensor must fit in global cb` Python/C++ mismatch).
3. `docs/platforms/tt_qwen3_8b_prefetcher_partial_bf16_2026-05-23.md` —
   this document.

## Updated shipping verdict (unchanged from prior session)

**DO NOT ship the prefetcher under `SGLANG_TT_USE_PREFETCHER=1` for
Qwen3-8B until tt-metal C++ work lands.**

The 1.66× decode-speed win is real (TPOT 16-20 ms vs canonical 27 ms)
but correctness requires either fixing the BFP8 corruption in
`ttnn.dram_prefetcher` (cleanest) or raising the per-bank L1 cap above
1.572 MB to make BF16 partial lifts viable. Both are tt-metal C++
deliverables. The Python preset surface in this repo is now exhausted.

Canonical (`SGLANG_TT_USE_PREFETCHER` unset/0) remains the production
configuration at TPOT ≈ 27 ms / 1024-1024 with GSM8K(10) = 9/10 = 90%
(re-verified 2026-05-23 in this session as a sanity check after the
prefetcher.py CB-sizing edit).
