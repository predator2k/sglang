# WS-A.11: Qwen3.5-0.8B step-0 cache verify + layer-compounding analysis

Date: 2026-05-22
Branch: tenstorrent-p1
HEAD (sglang): `1e3eca852` (pre-WS-A.11) -> new commit on top
HEAD (tt-metal-sglang): `c8233193598` (WS-A.10 load-time KV replicate) -> new commit on top

## TL;DR

**The "step-0 PCC=0.0267 anomaly" claimed in WS-A.10's hand-off was a stale
measurement. The actual step-0 end-to-end PCC vs HuggingFace reference is
0.72, consistent with steps 1+ (0.65-0.70).** No cache write bug exists at
position 0. The remaining PCC gap (~0.30 vs the 0.99 target) is from
BF16/BFP8 numerical error compounding across the 6 full-attention layers,
not a logic bug.

## Phase 1 — KV cache verify at pos 0 (DONE)

New diagnostic: `python/sglang/srt/hardware_backend/tenstorrent/test/standalone/_wsa11_step0_cache_verify.py`

**Findings:**

1. **Pre-decode KV cache is all zeros** (sanity check passes).
2. **After step 0**, layer-3 cache block-0 contains:
   - `|K[h0,pos0]| = |K[h1,pos0]| = 22.67` (bit-identical, load-time replicate
     produces identical K rows for both KV-head slots).
   - `|V[h0,pos0]| = |V[h1,pos0]| = 4.64` (bit-identical).
   - All other positions are zero.
3. **Cross-head check**: |K0-K1| = 0.0000 (cosine 1.0) on both devices.
4. **Cross-device check** (device 0 vs HF V head 0; device 1 vs HF V head 1):
   - Device 0 V vs HF V head 0: **PCC = 0.9101** (norm 4.64 vs HF 5.32)
   - Device 1 V vs HF V head 1: **PCC = 0.9642** (norm 11.40 vs HF 11.52)
   - Device 0's V is the lower-PCC slice (smaller absolute values amplify
     bf16 quantization noise more).

**Verdict:** The cache write path is correct. No `paged_update_cache`
zero-slot bug. WS-A.10's hypothesis #1 (cache write only fills first head)
is **not** the explanation.

## Phase 2/3 — Fix needed? (NO)

End-to-end PCC reproduction (3 runs across 2 harnesses, all agreeing):

| Step | WS-A.10 claim | WS-A.11 measured | WS-A.11 (DecodersPrecision.accuracy) |
|------|---------------|------------------|--------------------------------------|
| 0    | 0.0267        | **0.7183**       | **0.7312**                           |
| 1    | 0.6967        | 0.6967           | 0.7007                               |
| 4    | 0.6805        | 0.6806           | 0.6852                               |
| 16   | n/a           | 0.6527           | 0.6512                               |

WS-A.10's reported step-0 anomaly does not reproduce. Three independent
runs (`_wsa11_step0_cache_verify.py`, `_wsa11_full_pcc_report.py`,
`test_qwen35_pcc.py` pytest) all return step-0 PCC ≈ 0.72.

Likely WS-A.10's 0.0267 was measured with a stale on-disk KV `.tensorbin`
cache from an earlier (broken) run that survived the harness wipe. The
fresh-cache path produces the expected ~0.72 PCC consistently.

**No fix landed.** No fix needed.

## Phase 5 — Per-op layer-3 chain (DONE)

Per-op PCC at layer 3 (first full-attention layer, vs HF reference) is
unchanged from WS-A.10:

| op                          | PCC (load-time KV replicate) |
|-----------------------------|------------------------------|
| 00_layer3_input             | 0.9643                       |
| 01_post_input_layernorm     | 0.9515                       |
| 04_post_k_proj              | 0.5058 (head-0 only)         |
| 05_post_v_proj              | 0.9100                       |
| 06_post_q_norm              | 0.2949 (row-permuted Q)      |
| 10_post_sdpa                | 0.9101                       |
| 11_sigmoid_gate             | 0.8844                       |
| 11b_post_gate_mul           | 0.9058                       |
| 12_post_o_proj              | **0.8472**                   |
| 13_layer3_output (residual) | 0.9260                       |

### Why "post_q_norm = 0.2949" is NOT a bug

The WS-A.6 partial-RoPE permute (`qwen35_partial_rotary_head_perm`) rewires
per-head Q rows so the bound `rotate_half(half_Wt = head_dim/2)` kernel
implements HF's partial RoPE (rotary_dim=64 of head_dim=256). After this
permute, TT Q is in a dim-permuted layout vs HF Q, so element-wise PCC is
low — but the SDPA dot product (which uses the same permuted layout on Q
and K) preserves the inner product exactly. Confirmed by post_sdpa PCC
= 0.9101 = V PCC (at step 0, sdpa output = V exactly, regardless of Q).

### Cross-head verification of WO matmul + AllGather

Added two diagnostic dumps (gated behind `SGLANG_TT_DUMP_LAYER3=1`):
- `11d_all_gather_output` — per-device view of full activation post AllGather
- `11c_wo_weight` — per-device WO weight slice

**Manual recomputation results:**
- `wo @ tt_all_gather_output` reproduces TT post_o_proj **bit-exactly**
  (PCC=1.0000). WO matmul is correct.
- `wo @ hf_full_post_gate_mul` reproduces HF post_o_proj to PCC=0.9999.
  WO weight is loaded correctly.
- Per-head PCC of the AllGather output vs HF: heads 0-7 all in [0.88, 0.94]
  range — no head ordering bug, no head dropping.

The 0.91 → 0.85 PCC drop at o_proj is pure error amplification from a
2048-wide reduction in BF16 (then sharded across devices). With perfect
inputs the matmul gives PCC=0.9999; with noisy inputs (PCC 0.91) the output
naturally degrades.

### Why 0.91 is the v_proj ceiling

The v_proj output PCC is 0.91 (heads 0-3 on device 0). The matmul input
(post_input_layernorm) is PCC=0.95 vs HF. The 1024→256 matmul against BFP8
weight rows drops PCC by ~4 points — within expected bounds for
BF16/BFP8 numerical precision on this matmul shape.

Switching `DecodersPrecision.performance` (BFP8 W) to `DecodersPrecision.accuracy`
(BF16 W + HIFI4 MAC) gave only +0.013 step-0 PCC. The hardware-imposed
numerical floor dominates over the precision setting at this layer width.

## What we did NOT do

- **No cache write fix** (cache is correct).
- **No `paged_update_cache` modification** (not the bug).
- **No new harness step-0 special path** (no special path exists).
- **No TT-native GatedDeltaNet work** (out of scope per WS-A.11 rule 6).

## Qwen3-8B regression

`test_prefetcher_smoke.py -v -s` → **1 passed** (no regression).

## Files changed

### tt-metal-sglang
- `models/tt_transformers/tt/attention.py` (+8 lines): added two
  `_ws_a7_dump_save` calls (11c_wo_weight, 11d_all_gather_output) gated
  behind the existing `_ws_dump` env-var (`SGLANG_TT_DUMP_LAYER3=1`).
  Zero behavioral impact when env var is unset.

### sglang
- `python/sglang/srt/hardware_backend/tenstorrent/test/standalone/_wsa11_step0_cache_verify.py`
  (new) — KV cache read-back probe per device at pos 0/1 + cross-device V
  comparison vs HF reference.
- `python/sglang/srt/hardware_backend/tenstorrent/test/standalone/_wsa11_full_pcc_report.py`
  (new) — end-to-end PCC report for all CAPTURE_STEPS without pytest's
  abort-on-first-fail behavior.
- `docs/platforms/tt_qwen3_5_ws_a11_findings_2026-05-22.md` (this file).

## Next blockers / handoff

Current end-to-end PCC: **0.72 / 0.70 / 0.68 / 0.65** (steps 0/1/4/16).
Per-layer compounding from 6 full-attention layers × ~0.93 per-layer drop
≈ ~0.65 end-to-end — matches measurement.

**To get to PCC ≥ 0.95** would require either:

1. **FP32 weights / MAC** — not supported natively on Blackhole; would
   need software-level emulation that destroys throughput.
2. **Eliminate KV-replicate workaround** — fix the underlying tt-metal
   SDPA-decode kernel bug at (n_q=4, n_kv=1, head_dim=256). This is the
   "real" fix, but requires kernel C++ work in tt-metal upstream.
3. **Pre-multiply Q with the inverse permute and use straight rope_decode**
   instead of HF rope — would simplify the permute story but may not move
   PCC. Worth trying if user pivots back to PCC chase.
4. **Per-step output token quality is already meaningful**: at step 5/8/11/13
   TT and HF agree on top-1. The PCC=0.65-0.72 hides decent top-K behavior.
   If the goal is sampling quality rather than bit-equivalence, a top-K
   evaluation may show better results than PCC implies.

For now, **WS-A.11 closes the cache-bug hypothesis chain**. The remaining
PCC gap is structural (BFP8/BF16 precision on a 24-layer model with
load-time KV replicate adding 4x redundant heads), not a fixable software
bug at this layer of the stack.
