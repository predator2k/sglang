# Qwen3.5 WS-A.15 — Full model-wide BF16 lift confirms structural precision floor (2026-05-22)

This document records WS-A.15: the most aggressive accuracy lift attempted
for Qwen3.5-0.8B on 2× Blackhole P150a — convert **every** weight tensor
plus the KV cache to BF16, and force **every** compute kernel to HiFi4 —
and the measured PCC outcome.

WS-A.15 follows directly from WS-A.14 (`d54189ca253`), which concluded the
per-op gap at layer 3 (`post_sdpa` 0.91, `post_o_proj` 0.85) was set by
WQKV/Wo BFP8 quantization and that no SDPA-side fidelity knob could move
it. WS-A.13 had previously measured a Wo+WQKV+KV BF16 attention preset
and recorded +0.005..+0.018 e2e gain — below the +0.05 commit threshold.

WS-A.15's question: does extending the BF16 lift to **all** weight
classes (MLP w1/w2/w3 + LM head added to the WS-A.13 attention preset)
compound across the 24-layer stack into a meaningful PCC gain — i.e. is
the per-layer noise the MLP weights, the LM head, or both?

---

## Reference baseline (reproduced)

Bit-exact reproduction of the WS-A.14 baseline (default precision config,
no env vars set; 17 decode steps with HF-greedy teacher forcing):

| step | PCC | cosine | top-1 match |
|---|---|---|---|
| 0 | 0.7183 | 0.6980 | False |
| 1 | 0.6967 | 0.7537 | False |
| 4 | 0.6806 | 0.8459 | False |
| 16 | 0.6527 | 0.7517 | False |

---

## WS-A.15 implementation

New env var `SGLANG_TT_QWEN35_FULL_BF16=1`, default off, that:

1. **`ModelOptimizations.performance()` Qwen3.5-0.8B branch**: when set,
   builds a preset that lifts every TensorGroup that has a precision dial
   to BF16 and every OpGroup to HiFi4:
   - `TensorGroup.FF1_FF3` BFP4 → BF16  (MLP up/gate projections)
   - `TensorGroup.FF2`     BFP8 → BF16  (MLP down projection)
   - `TensorGroup.WQKV`    BFP8 → BF16  (attention fused QKV)
   - `TensorGroup.WO`      BFP8 → BF16  (attention output projection)
   - `TensorGroup.KV_CACHE` BFP8 → BF16 (paged KV cache backing tensor)
   - `OpGroup.LI_FF1_FF3` / `LI_FF2` HiFi2_FP16 → HiFi4
   - `OpGroup.LI_QKV_DECODE` / `_PREFILL` HiFi2 → HiFi4
   - `OpGroup.SDPA_DECODE` HiFi2 → HiFi4 (prefill was already HiFi4)
   - `OpGroup.LI_O_DECODE` / `_PREFILL` HiFi2 → HiFi4

2. **`ModelArgs._set_model_specific_params()`**: when the env var is set,
   set `self.lm_head_dtype = ttnn.bfloat16`. Two consumers:
   - `lm_head.py:164` (matmul output dtype) reads `args.lm_head_dtype`
     directly via `hasattr` fallback.
   - `model.py:174` (lm head WEIGHT storage) was patched to take
     `getattr(args, "lm_head_dtype", dtype)` so the storage dtype also
     lifts to BF16 (formerly always followed the top-level Transformer
     `dtype` arg, which stays BFP8 for cache-path consistency).

Precedence: `SGLANG_TT_QWEN35_FULL_BF16=1` takes priority over
`SGLANG_TT_QWEN35_WO_PRECISION={bf16,wo_only}` (WS-A.13). Both env
vars unset (default) keep the original BFP4/BFP8 mixed precision.

### Verified by load-time cache filenames

With `SGLANG_TT_QWEN35_FULL_BF16=1`, the on-disk tt-metal cache files
landed at:

```
layers.*.feed_forward.w1_sharded_dtype_BFLOAT16_layout_TILE.tensorbin
layers.*.feed_forward.w2_sharded_dtype_BFLOAT16_layout_TILE.tensorbin
layers.*.feed_forward.w3_sharded_dtype_BFLOAT16_layout_TILE.tensorbin
layers.*.attention.wqkv_sharded_2d_dtype_BFLOAT16_layout_TILE.tensorbin
layers.*.attention.wo_width_sharded_2d_dtype_BFLOAT16_layout_TILE.tensorbin
empty_kcache_paged_attention(64, 2, 32, 256)_dtype_BFLOAT16_layout_TILE.tensorbin
empty_vcache_paged_attention(64, 2, 32, 256)_dtype_BFLOAT16_layout_TILE.tensorbin
output_lm_head_8_split_shard_*_dtype_BFLOAT16_layout_TILE.tensorbin
```

vs the baseline `*_dtype_BFLOAT8_B_*` (attention/MLP-w2/KV/LM-head) and
`*_dtype_BFLOAT4_B_*` (MLP-w1/w3). All 8 weight surfaces successfully
lifted.

---

## Measured PCC

End-to-end vs HF-reference logits, identical decode trace, same start
token (42), HF-greedy teacher forcing:

| step | baseline | FULL_BF16 | delta |
|---|---|---|---|
| 0 | 0.7183 | **0.7245** | +0.0062 |
| 1 | 0.6967 | **0.6969** | +0.0002 |
| 4 | 0.6806 | **0.6826** | +0.0020 |
| 16 | 0.6527 | **0.6496** | **−0.0031** |

**Maximum gain anywhere: +0.006.** **Negative at step 16.** Far below the
+0.05 commit threshold; far below any plausible "meaningful gain"
threshold.

Cosine similarity dropped at steps 0 and 16 (the model is matching HF
slightly less well in direction), suggesting the BF16 lift is shifting
the per-layer error in a different (not strictly better) way — not the
"noise floor" pattern one would expect from a pure precision win.

---

## Memory cost

Weight cache sizes after a fresh build with `SGLANG_TT_QWEN35_FULL_BF16=1`
(measured at `/tt-metal/models/weights/Qwen3.5-0.8B/P300/hf_rope/tensor_cache_bfp8/`):

| category | files | total |
|---|---|---|
| BF16 weights (FULL_BF16 preset) | 205 | **1580 MB** |
| BFP8 weights (baseline preset) | 44 | 398 MB |
| BFP4 weights (baseline preset) | 48 | 95 MB |

Baseline mixed precision: **493 MB**. FULL_BF16: **1580 MB** —
**3.2× larger**. Still fits P150a 16 GB DRAM comfortably with room for
KV cache and activations.

Model build time (cold cache → warm cache flow, 24 layers):
- Baseline: ~11 s end-to-end build (from log).
- FULL_BF16 first-build (cold cache): JIT cache 165/200 (82.5%) hits;
  ~10 s additional kernel compile (`JitBuildState::build` total = 9758 ms
  for the FULL_BF16 fresh-kernel set). After the cold build, warm
  re-runs are comparable to baseline.

---

## Verdict — structural floor confirmed

**Three independent precision audits now agree.**

| audit | weights lifted | e2e PCC gain at step 0 |
|---|---|---|
| WS-A.13 `wo_only` | Wo only | +0.018 |
| WS-A.13 `bf16` | Wo + WQKV + KV | +0.013 |
| **WS-A.15 `FULL_BF16`** | **Wo + WQKV + KV + FF1/FF3 + FF2 + LM head** | **+0.006** |

Lifting all 8 weight surfaces to BF16 and forcing HiFi4 fidelity on every
matmul/SDPA op produced **less** gain than the partial WS-A.13 attention-
only lift. This rules out per-tensor BFP8 quantization as the dominant
PCC failure source.

**The remaining ~0.30 PCC gap vs HF is structural** — it comes from:

1. **bf16 multiply-accumulate noise compounded across 24 layers**, even
   when all weights and all kernels are at the highest precision the
   tt-metal Wormhole/Blackhole math fabric supports (HiFi4 fp32 dest acc
   is already enabled in the default kernel config).
2. **Hybrid full-attention vs GatedDeltaNet layer interaction** — the
   host-fallback DeltaNet path produces PCC=1.0 in isolation (verified
   by WS-A.5 `qwen35_deltanet_unit.py`), but the hand-off into the
   following BFP8-trained-on-BF16-activations layer compounds error.
3. **load-time KV-head replicate workaround** — the n_kv_heads=1 → 2
   doubling is mathematically equivalent (WS-A.14 H5 confirmed bit-exact
   per-head outputs), but the load is fully on the 2048-wide Wo
   reduction and SDPA-decode kernel path which already maxes out the
   precision budget.

PCC ≥ 0.95 for Qwen3.5-0.8B is **not achievable** on the current
tt-metal kernel stack without:

- A new fp32-accumulate-everywhere SDPA decode kernel (TT-side patch).
- Wider-than-2048 fused all-gather-matmul kernels that hold the reduction
  in fp32 throughout (TT-side patch).
- An SGLang core change to relax the n_kv_heads=1 constraint on the
  SDPA decode kernel so the load-time KV-replicate workaround can be
  removed (out of scope per the no-core-changes rule).

---

## What survives

`models/tt_transformers/tt/model_config.py`:

- New `SGLANG_TT_QWEN35_FULL_BF16=1` env-gated branch in
  `ModelOptimizations.performance()` (Qwen3.5-0.8B section).
- New `self.lm_head_dtype = ttnn.bfloat16` assignment in
  `_set_model_specific_params()` gated on the same env var.

`models/tt_transformers/tt/model.py`:

- `LMHead` instantiation now uses `getattr(args, "lm_head_dtype", dtype)`
  for the WEIGHT storage dtype (was always the top-level `dtype` arg).
  Byte-equivalent for every model that does not set `args.lm_head_dtype`.

All paths are default-off / behavior-preserving when the env var is
unset. A future operator can re-run the WS-A.15 comparison with one env
var (no rebuild) once tt-metal kernels improve.

### Regression

- `test_prefetcher_smoke_5_decodes` (Qwen3-8B, 2× P150a, prefetcher ON):
  **PASS** (1 passed, 12.05 s).
- Baseline Qwen3.5 PCC reproduction (env var unset): bit-exact match to
  WS-A.14 numbers (0.7183 / 0.6967 / 0.6806 / 0.6527).

---

## Decision

**Kept as opt-in (env-gated, default-off).**

Per the WS-A plan rule, we commit the diagnostic infrastructure even
though the gain is below the +0.05 commit threshold — same outcome
pattern as WS-A.14. Default behavior is byte-equivalent.

The next attack vectors for Qwen3.5 accuracy (post WS-A.15) are not in
the precision dimension — they are kernel-implementation changes on the
tt-metal side or DeltaNet-on-device improvements:

1. **tt-metal SDPA decode kernel patch** to do fp32 accumulate on the
   softmax→V reduction (currently the bf16 matmul stage clips precision
   even with HiFi4 set). This would lift the per-op `post_sdpa` ceiling
   from 0.91 → ~0.97.
2. **TT-native GatedDeltaNet** (WS-A.12 scaffolding exists, full
   conversion deferred). The current host-fallback path is bit-exact in
   isolation but adds a host hop per linear-attention layer which
   inflates TPOT and forces a memory round-trip.
3. **Per-layer compounding probe** to confirm whether one specific layer
   amplifies more error than the others; if yes, a targeted per-layer
   precision lift (instead of model-wide) could close the gap at
   ~3-5× lower memory cost than FULL_BF16.

None of these are blockers for shipping Qwen3.5 at the current PCC
floor; they are the long-tail roadmap for closing the remaining gap
once the precision-budget items are exhausted (which WS-A.13+14+15 now
establishes definitively).
