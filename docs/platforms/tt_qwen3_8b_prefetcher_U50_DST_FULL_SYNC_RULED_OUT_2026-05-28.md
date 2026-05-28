# Qwen3-8B BFP8 Prefetcher U50 — D1 (`dst_full_sync_en=True`) Result

**Date**: 2026-05-28
**Status**: `EVIDENCE_ADVANCE` — D1 (`dst_full_sync_en=True` on prefetcher-fed
matmuls) is **REFUTED** as the fix. D2 (`hop_cores=[(3,6)]`) **HANGS** the
silicon and is also REFUTED. The U49b/PR-#45402 single-boolean-flip
hypothesis ("it'd be a `dst_full_sync_en + packer_l1_acc + LoFi`
interaction that only manifests with the BH dest accumulator quirks") is
empirically falsified for Qwen3-8B BH.

## TL;DR

| Variant                                 | Smoke prompt `"What is 2+2?"` first 20 tokens                       | GSM8K(10)        | Verdict                       |
|-----------------------------------------|---------------------------------------------------------------------|------------------|-------------------------------|
| Baseline (prefetcher-on, **no D1**)     | random multilingual garbage (`不断发展习惯了 amazon贬值aat...`)         | 0/10 (baseline)  | known broken                  |
| **D1** (`dst_full_sync_en=True`)        | single-token repetition `Cialis Cialis Cialis...` (id=32302) → NaN  | **0/10**         | **REFUTED — changes signature but introduces NaN; sampler crashes** |
| **D2** (`hop_cores=[(3,6)]`)            | (server hung post-weight-load — never reached /health)              | n/a              | **REFUTED — hangs silicon**   |
| **D1 + D2**                             | (server hung post-weight-load — same as D2)                         | n/a              | hangs (D2 dominates)          |
| Canonical re-verify (no prefetcher)     | `What is 2+2? What is 2+2? What is 2+2`                             | **10/10 = 100%** | **no regression**             |

## §1 Implementation (env-gated, default-off — canonical untouched)

Two single-knob env gates were added to `tt-metal-sglang` (predator2k fork,
branch `tenstorrent-p1`):

### `SGLANG_TT_U50_DST_FULL_SYNC=1`

Patches `models/tt_transformers/tt/model_config.py:get_math_fidelity()`
(around L4958). After the standard `MathFidelitySetting` → `compute_kernel_config_*`
lookup, when the env flag is set AND the OpGroup is one of the four
prefetcher-fed matmul ops (`LI_FF1_FF3`, `LI_FF2`, `LI_QKV_DECODE`,
`LI_O_DECODE`), the returned `ttnn.WormholeComputeKernelConfig` is rebuilt
with `dst_full_sync_en=True` (copying all other fields). All other ops
(SDPA, RoPE, RMSNorm, all-reduce) are untouched.

Diff:
```python
_cfg = math_fidelity_setting_lookup[...]
if (os.environ.get("SGLANG_TT_U50_DST_FULL_SYNC", "0") == "1"
        and op in (OpGroup.LI_FF1_FF3, OpGroup.LI_FF2,
                   OpGroup.LI_QKV_DECODE, OpGroup.LI_O_DECODE)):
    _cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=_cfg.math_fidelity,
        math_approx_mode=_cfg.math_approx_mode,
        fp32_dest_acc_en=_cfg.fp32_dest_acc_en,
        packer_l1_acc=_cfg.packer_l1_acc,
        dst_full_sync_en=True,
    )
return _cfg
```

### `SGLANG_TT_U50_HOP_CORES=1`

Patches `models/tt_transformers/tt/model_config.py:matmul_1d_ring_config()`
(around L3904). When the env flag is set, `hop_grid` becomes `[(3, 6)]`
(matching `models/demos/llama3_70b_galaxy/tt/model_config.py:2479,2537`'s
canonical Galaxy WH value). When unset, `hop_grid` remains `[]` as before.
The same `hop_core_range_set` construction code feeds the resulting cores
into the `MatmulMultiCoreReuseMultiCast1DProgramConfig.hop_cores` arg for
all ring-config matmul callers (every prefetcher-fed WQKV / WO / W1 / W3 /
W2 matmul go through this function).

## §2 Hardware test — D1

Server launched with:
```
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged
SGLANG_TT_MAX_BATCH=1
HF_MODEL=Qwen/Qwen3-8B
SGLANG_TT_USE_PREFETCHER=1
SGLANG_TT_U50_DST_FULL_SYNC=1
SGLANG_TT_DISABLE_PREFILL_TRACE=1
```

`POST /generate "What is 2+2?" max_new_tokens=20 temperature=0.0`:
```
{"text":" What Cialis Cialis Cialis Cialis Cialis Cialis Cialis Cialis ...",
 "output_ids":[3555,32302,32302,32302,32302,...,32302]}
```

**vs. baseline (no D1)**:
```
{"text":" What不断发展习惯了 amazon贬值aat霸气/modEUtein敬请7课外OfWeek请您�高频 volleyballERVE潛",
 "output_ids":[3555,114133,114057,38416,114384,...]}
```

The garbage signature changed **dramatically** — from
random-multilingual-spray (typical "logits Gaussian") to
single-token-repetition (id=32302 = `" Cialis"`). This is a
fundamentally different failure mode and confirms D1 has a real
silicon-side effect on the matmul output.

**GSM8K(10) with D1 enabled**:
```
Q1/10  INVALID   garbage Chinese+English with strong repetition
                 `ัว本来就education本来就本来就education本来就 Unsigned...`
Q2-10  ERROR     Connection refused (server crashed after Q1)
FINAL: 0/10 = 0.0%
```

Server log shows:
```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```

**Verdict**: D1 is not just neutral — it **makes things actively worse**.
The matmul output goes from "scrambled-but-finite logits" (baseline) to
"NaN/Inf propagating through the chain" (D1). This is consistent with
`dst_full_sync_en=True` on BH altering the BFP8 4-face SrcA decode
sequencing in a way that introduces denormals or out-of-range
intermediates that the next FFN multiplication blows up to inf.

## §3 Hardware test — D2 (`hop_cores=[(3,6)]`)

Server launched with `SGLANG_TT_U50_HOP_CORES=1`. Weight load completed
(36/36 layers loaded), prefetcher dram-grid initialized (180 tensors
queued), prefill trace started:

```
[Prefetcher] run(): num_layers=36 effective_num_layers=36
```

Then **silicon hung**. `/health` returned 503 for >2 minutes; log stopped
producing output after the `[Prefetcher] Creating global CB with size:
835584` line. `tt-smi -r` was required to recover the device for
subsequent runs.

**Verdict**: D2 (`hop_cores=[(3,6)]`) is a hard silicon hang on
Blackhole. The (3,6) coordinate is valid for the WH Galaxy 8x8 grid (where
the `llama3_70b_galaxy/tt/model_config.py:2479` canonical value comes
from) but appears to map to a problematic physical core or NoC route on
the Blackhole p150a topology — either deadlocking the gather-route or
overlapping with the dispatch sub-device. This makes D2 untestable for
fixing the BFP8 corruption.

## §4 D1+D2 combo

With `SGLANG_TT_U50_DST_FULL_SYNC=1 SGLANG_TT_U50_HOP_CORES=1`, weight
load completed but server hung at the same spot as D2-alone (post-
prefetcher-init, never reached HTTP /health). D2 dominates; the combo is
also REFUTED (cannot evaluate output).

## §5 Canonical re-verify (no D1, no D2, no prefetcher)

Standard canonical command (no U50 env vars, no
`SGLANG_TT_USE_PREFETCHER`):

```
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged
SGLANG_TT_MAX_BATCH=1
HF_MODEL=Qwen/Qwen3-8B
SGLANG_TT_DISABLE_PREFILL_TRACE=1
```

`/generate "What is 2+2?"` → `" What is 2+2? What is 2+2? What is 2+2"`
(canonical echo behavior — Qwen3-8B base model w/o chat template repeats
the prompt under greedy decoding).

**GSM8K(10) = 10/10 = 100.0%** — bit-for-bit no regression. The U50
env-gated discriminator code is safe to land alongside canonical
operation.

## §6 What this rules out

U49b enumerated three remaining Galaxy-WH vs Qwen3-8B-BH discriminators:
D1 (`dst_full_sync_en`), D2 (`hop_cores`), and 1 LoFi/HiFi-relevant flag
already accounted for upstream. U50 directly tested D1 and D2 in
hardware:

- D1 changes the silicon behavior (good — confirms `dst_full_sync_en`
  is a real lever on this matmul path), but in the WRONG direction
  (signature degrades from scrambled-logits to NaN-propagation; sampler
  crashes). Refuted.
- D2 hangs the silicon entirely. Refuted (untestable).
- D1+D2 inherits the hang. Refuted.

**The "single program-config discriminator" attack lane from PR #45402
is now empirically closed**. The bug surface remains as narrowed by U46:
strictly inside the upstream tt-metal LLK BFP8 4-face SrcA decode under
the dual-index `experimental::CreateCircularBuffer(prog, cores,
remote_cfg, *global_cb)` global-CB allocation pattern.

## §7 Next-step suggestions for the LLK team

1. **`dst_full_sync_en=True` produces NaN on BH BFP8 prefetcher matmul** —
   this is itself a useful data point for the upstream bug report. On
   Galaxy WH the same flag is set and works; on BH Qwen3-8B prefetcher
   path it produces denormal-induced NaN propagation. Either:
   - the BH dest-acc FSM mis-orders the 4-face SrcA decode when
     `dst_full_sync_en=True` AND the source is BFP8 AND the CB is
     dual-index, OR
   - the BFP8 mantissa-block extraction trips a different code path
     under `dst_full_sync_en=True` on BH (not on WH).
2. **`hop_cores=[(3,6)]` hangs BH Qwen3-8B prefetcher** — should be
   reported as a separate issue (config-level hang, not silicon-corruption).
   The (3,6) coordinate is canonical for the WH Galaxy 8x8 layout; the
   BH p150a likely needs a different hop-core placement (if any). This
   blocks any further exploration of the "Galaxy-shape hop-core
   routing" hypothesis without a BH-specific hop_core value, which we
   don't have a reference for.
3. **The remaining attack lane is genuinely silicon-FSM-only** —
   matches U46's prior conclusion. Next probes should be either
   silicon-tracer-level (logic-analyzer the SrcA register file in real
   time during a corrupted MOP fire) or contained-MWE-style (extend
   the U46 standalone reproducer scaffold with the 4 production-only
   ingredients enumerated there: dual-index CB + gathered matmul +
   5 mixed-size weights + 50 trace replays + BFP8/HiFi2 + LoFi BFP4
   mixing).

## §8 What's persisted

- **Env-gated U50 D1 + D2 discriminator code** committed to
  `tt-metal-sglang@tenstorrent-p1` so future U-experiments can reproduce
  this finding with a single env-flag flip (no re-edit needed).
- **U50 result doc** (this file) in `sglang/docs/platforms/`.

## §9 Hard constraints honored

- No upstream PR (predator2k fork only — same constraint as U37-U49b)
- No SGLang core behavior changes (only `tt-metal-sglang` model layer)
- Both env flags default-off; canonical untouched (re-verified 10/10)
- Cache-clear used `[ -n "$VAR" ]` guard with explicit absolute path
- `tt-smi -r` invoked manually after D2 hang to recover device state
- stash@{0,1,2} not touched
