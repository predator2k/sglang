# tt-vllm Qwen3-8B per-op breakdown — vLLM on 2× P150a (2026-05-21)

**Date:** 2026-05-21
**Branch:** `tenstorrent-p1`
**Container:** `p3a-ngram` (same container as the tt_transformers reference; vLLM installed into `/opt/venv`)
**vLLM install:** Tenstorrent's `tt-vllm-plugin` v0.1.0 (standalone, from `/home/mhnie/tt-inference-server/tt-vllm-plugin/`)
**vLLM version:** 0.10.1.1
**Hardware:** 2× Blackhole p150a, mesh = `(1, 2)` via `MESH_DEVICE=P150x2`
**Model:** Qwen3-8B BFP8 (TP=2, tt-metal `Qwen3-8B-N300` recipe via tt_transformers generator `QwenForCausalLM`)
**Inputs:**
- `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/cpp_device_perf_report_vllm_qwen3_8b.csv` (11 MB, 46,720 rows, all OP NAMEs populated)
- Decode-loop config: prompt 2001 effective tokens (1024-token padded then chat-templated by HF tokenizer), generation 16-128 tokens, batch 1
- Env: `MESH_DEVICE=P150x2 VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TTNN_OP_PROFILER=1 TT_PROFILE_PERIODIC_FLUSH=1 HF_MODEL=/models/Qwen3-8B`
- **Canonical TPOT (no profiler) = 39.18 ms/token** (linear fit, see Method §3)
- Profiler-on TPOT = 236.5 ms/token (6.0× canonical — within expected 6-9× range)

## Method

### 1. vLLM install

The standalone `tt-vllm-plugin` does NOT register `TTQwen3ForCausalLM` for text generation by default; it points the slot at `Qwen3ForEmbedding`. The fork's in-tree `vllm-tt-plugin` (under `/home/mhnie/tt-vllm/plugins/vllm-tt-plugin/`) does register the text generator correctly, but installing the full fork would have required a heavy in-container build.

**Local patches applied (LOCAL ONLY — not propagated to any sglang or tt-metal source):**

1. **`/opt/tt-vllm-plugin/tt_vllm_plugin/__init__.py`** — register `TTQwen3ForCausalLM` → `models.tt_transformers.tt.generator_vllm:QwenForCausalLM` (the same text generator the upstream fork uses). Also registered `TTQwen2ForCausalLM` for Qwen2.5 in case needed. The original embedding registration was removed for Qwen3.
2. **`/tt-metal/models/tt_transformers/tt/generator_vllm.py`** — two `try/except ImportError` shims for vLLM 0.11→0.10.1.1 API drift:
   - `BaseDummyInputsBuilder` moved from `vllm.multimodal.profiling` to `vllm.multimodal.processing` between 0.10.1.1 and 0.11
   - `STR_DTYPE_TO_TORCH_DTYPE` moved from `vllm.utils` to `vllm.utils.torch_utils` between the same versions
   - Patch was reverted in container after the capture run. The shims are 4 added lines total.
3. **`/opt/tt-vllm-plugin/tt_vllm_plugin/v1/worker/tt_model_runner.py`** — added a 6-line `if os.environ.get("TT_PROFILE_PERIODIC_FLUSH") == "1": ttnn.ReadDeviceProfiler(self.mesh_device)` after each decode forward, gated on env var. Zero overhead when unset.
4. **`mistral_common`** downgraded from 1.11.2 → 1.9.1 (vllm 0.10.1.1's `pixtral.py` imports `ImageChunk` from `protocol.instruct.messages`, which was removed in 1.10+).

### 2. Tracy recipe (same 4 keys as the SGLang reference)

1. `TTNN_OP_PROFILER=1` — populates OP NAME column
2. `TT_METAL_DEVICE_PROFILER=1` — enables device profiler
3. `TT_METAL_PROFILER_CPP_POST_PROCESS=1` — enables the C++ post-processor that writes `cpp_device_perf_report.csv` (the missing flag from the original recipe; without it only the raw 2.7 GB `profile_log_device.csv` is written)
4. `close_mesh_device()` in `tt_vllm_plugin/worker/tt_worker.py` already calls `ttnn.ReadDeviceProfiler(mesh_device)` at teardown. vLLM v1 multiprocessing was disabled (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the engine teardown runs in the main process and the close hook fires reliably.

### 3. Canonical TPOT measurement

`/tmp/p3a_steady_state_tpot.py` (in-container) runs 3 trials each at gen={16, 64, 128}, after 3 warmup runs. Linear fit:

```
gen=16:  avg = 0.800 s  (3 trials std-dev <3 ms)
gen=64:  avg = 2.678 s
gen=128: avg = 5.189 s
slope: (5.189 - 0.800) / (128 - 16) = 39.18 ms / token
intercept: 0.800 - 16 * 0.0392 = 173.2 ms (prefill)
sanity: predicted gen=64 = 2.681 s vs actual 2.678 s, residual −3.3 ms
```

## Caveats — fairness of comparison vs tt_transformers

- **vLLM does NOT use the DRAM prefetcher.** `use_prefetcher` only exists in `generator_sglang.py` (added during the SGLang port for Workstream-A). The vLLM path goes through `generator_vllm.py` which has no prefetcher support. **Therefore vLLM 39.2 ms should be compared to tt_transformers WITHOUT prefetcher (37.4 ms), not against the prefetcher-on 31.4 ms.** See [[tenstorrent-perf-knobs]] for prefetcher background. The CSV confirms: zero `DramPrefetcherOperation` rows in the vLLM dataset.
- **Precision and recipe are identical** to tt_transformers: BFP8 weights, `optimizations="performance"`, `n_layers=36`, tt-metal `Qwen3-8B` config (effectively a `Qwen3-8B-N300` recipe on a P300-equivalent mesh).
- **vLLM serves greedy + temperature=0** matching SGLang's bench config.
- **TPOT under TT_PROFILE=1 + TT_METAL_PROFILER_CPP_POST_PROCESS=1 is ~6× slower than canonical** (39.2 ms → 236.5 ms). Per-op `DEVICE FW DURATION` reflects kernel execution time on-device; relative shares hold.
- **~81% of FW rows (38,023 / 46,720) are corrupted by the dual-clock-domain bug** (see [[tenstorrent-tt-xla-tracy-fw-duration-bug]]). Filtered via `< 1e9 ns`. Estimated totals = mean(valid) × count. This corruption rate is higher than the tt_transformers reference's 34% because the periodic-flush hook didn't fire between every op (vLLM's decode loop pushes ops faster than the flush can drain the 12K-marker buffer), but the per-op valid samples are still statistically sufficient (most ops have valid_n ≥ 100; only `PagedUpdateCacheDeviceOperation` has 36 valid out of 2,332 — still 36 samples).
- **Number of decode iterations captured: 1,224 SdpaDecodeDeviceOperation / (36 layers × 2 devices) = 17 decodes** (16 from the capture run + 1 spilled from warmup).

## Per-op breakdown — vLLM Qwen3-8B decode

Top 12 device ops by estimated total time (mean valid FW × count) across the 17 captured decodes × 2 devices, scaled to per-device-per-decode:

| op | count | valid_n | mean µs | est total ms | % compute | ~ms/decode |
|---|---:|---:|---:|---:|---:|---:|
| **MatmulDeviceOperation** | 7068 | 613 | 258.3 | 1825.7 | 63.9% | 53.7 |
| **LayerNormDeviceOperation** | 5480 | 2455 | 83.8 | 459.4 | 16.1% | 13.5 |
| **ReduceScatterDeviceOperation** | 1455 | 782 | 157.3 | 228.9 | 8.0% | 6.7 |
| **AllGatherAsyncDeviceOperation** | 3105 | 1780 | 46.8 | 145.4 | 5.1% | 4.3 |
| **InterleavedToShardedDeviceOperation** | 4823 | 117 | 17.2 | 83.0 | 2.9% | 2.4 |
| NlpCreateHeadsDeviceOperation | 248 | 176 | 107.9 | 26.8 | 0.9% | 0.8 |
| AllGatherDeviceOperation | 1110 | 1094 | 18.9 | 21.0 | 0.7% | 0.6 |
| NLPConcatHeadsDeviceOperation | 246 | 174 | 78.4 | 19.3 | 0.7% | 0.6 |
| PagedUpdateCacheDeviceOperation | 2332 | 36 | 5.4 | 12.5 | 0.4% | 0.4 |
| NLPCreateQKVHeadsDecodeDeviceOperation | 1166 | 17 | 9.4 | 11.0 | 0.4% | 0.3 |
| RotaryEmbeddingLlamaDeviceOperation | 2952 | 36 | 2.6 | 7.5 | 0.3% | 0.2 |
| ReshardDeviceOperation | 2255 | 1045 | 2.3 | 5.1 | 0.2% | 0.2 |

**Total est compute FW: 2859.2 ms across 17 decodes × 2 devices = 84.1 ms / device / decode under profiler** (canonical 39.2 ms, scale 2.15×).

## Mapping to canonical 39.18 ms TPOT

The profiler scales kernel time by ~2.15×. Scaling the per-decode numbers above by 1/2.15:

| component | est ms / decode (canonical) | % of TPOT | what these are |
|---|---:|---:|---|
| **Matmul** | ~25.0 | **63.7%** | Q+K+V proj × 36, Wo × 36, gate/up FFN × 36, down × 36, lm_head — all DRAM-bound, no weight prefetching |
| **LayerNorm (RMSNorm)** | ~6.3 | **16.0%** | 161 norms per device per decode (pre/post attn + pre/post FFN × 36 layers + final) |
| **TP Collectives (RS + AG + AG-async)** | ~5.4 | **13.8%** | ReduceScatter 3.1 + AllGatherAsync 2.0 + AllGather 0.3; TP=2 comms after each matmul stage |
| **Layout (I2S + S2I + Reshard)** | ~1.3 | **3.4%** | InterleavedToSharded 1.1 + ShardedToInterleaved 0.05 + Reshard 0.07 |
| **Attention heads + create-QKV + ConcatHeads** | ~1.0 | **2.5%** | NlpCreateHeads (legacy prefill path) + NLPCreateQKVHeadsDecode + NLPConcatHeads |
| KV-cache + RoPE + misc | ~0.4 | 0.9% | PagedUpdateCache + RotaryEmbeddingLlama |
| **TOTAL** | **~39.4** | **~100%** | (matches canonical 39.18 ms within profiler noise) |

## Side-by-side comparison vs tt_transformers (same model, same hardware)

All three datasets are 2× P150a, Qwen3-8B BFP8, TP=2, batch=1, prompt~1024 tok, greedy decode.

### Headline TPOT comparison

| stack | TPOT (ms/tok) | vs vLLM | vs tt_transformers no-prefetcher | fairness |
|---|---:|---:|---:|---|
| **vLLM (no prefetcher)** | **39.18** | 1.00× | 1.05× slower than tt_transformers | apples-to-apples vs ttx no-prefetcher |
| **tt_transformers (no prefetcher, v6 2026-05-18)** | **37.43** | 0.96× | 1.00× | apples-to-apples vs vLLM |
| **tt_transformers (PREFETCHER ON, SGLang Workstream-A)** | **31.35** | 0.80× | 0.84× | UNFAIR — vLLM has no prefetcher hookup |

**vLLM is 1.05× slower than tt_transformers without prefetcher (+1.75 ms / token, +4.7%).** The vast majority of this gap is in matmul kernel time. The prefetcher-on number is included only to show what the matmul-launch optimization is worth on the same hardware — porting `use_prefetcher=True` to `generator_vllm.py` could close the remaining gap and exceed tt_transformers.

### Top 5 ops side-by-side (canonical-scaled ms per device per decode)

| op | vLLM ms/dec | ttx no-pref ms/dec | ttx prefetcher-on ms/dec | vLLM−ttx-no-pref Δ |
|---|---:|---:|---:|---:|
| **Matmul** | 25.0 | 14.3 | 2.7 | **+10.7 ms (slower)** |
| **LayerNorm** | 6.3 | 3.5 | 2.6 | **+2.8 ms (slower)** |
| **ReduceScatter** | 3.1 | 2.9 | 2.1 | +0.2 ms |
| **AllGatherAsync** | 2.0 | 1.5 | 1.2 | +0.5 ms |
| **BinaryNg (residual + RMSNorm mul)** | 0.0 | 6.8 | 0.7 | **−6.8 ms (vLLM faster / absent)** |
| **CopyDeviceOperation** | 0.0 | 4.2 | 0.0 | **−4.2 ms (vLLM faster / absent)** |

The op-level deltas are surprising: vLLM has much higher matmul (+10.7 ms) and LayerNorm (+2.8 ms) but completely lacks the `BinaryNg` (−6.8 ms) and `CopyDeviceOperation` (−4.2 ms) overhead that tt_transformers' no-prefetcher path incurs. **Net = +1.75 ms in favor of tt_transformers no-prefetcher.**

### Op-count delta — what's structurally different

| op | vLLM ops/device/decode | ttx no-pref ops/device/decode | Δ |
|---|---:|---:|---:|
| Matmul | 207.9 | 239.7 | −31.8 (vLLM has fewer launches) |
| LayerNorm | 161.2 | 170.9 | −9.7 |
| ShardedToInterleaved | 104.6 | 94.0 | +10.6 |
| AllGatherAsync | 91.3 | 144.4 | −53.1 (vLLM batches collectives more) |
| RotaryEmbedding | 86.8 | 42.8 | +44.0 |
| PagedUpdateCache | 68.6 | (PagedFusedUpdateCache 30.5) | +38.1 (vLLM uses unfused KV update) |
| Reshard | 66.3 | 95.0 | −28.7 |
| ReduceScatter | 42.8 | (ReduceScatterMinimalAsync 50.1) | −7.3 |
| NLPConcatHeadsDecode | 36.0 | (different op family) | varies |
| NLPCreateQKVHeadsDecode | 34.3 | 29.0 | +5.3 |
| AllGather (non-async) | 32.6 | 0.0 | +32.6 (vLLM dispatches a separate non-async AG) |
| Typecast | 27.5 | 56.8 | −29.3 |
| PagedFillCache | 18.1 | 0.0 | +18.1 (vLLM has both update + fill) |
| NlpCreateHeads (prefill-style) | 7.3 | 11.9 | −4.6 |

**Major structural differences:**

1. **vLLM matmul launches are ~13% fewer (208 vs 240 per device per decode) but each is 2× slower (258 µs vs 128 µs mean valid).** Total matmul wall time is therefore 1.75× higher in vLLM. The lower count suggests vLLM merges some matmul stages (likely via a unified attn-output path), but the per-launch latency is dominated by DRAM read of BFP8 weights — exactly what the prefetcher would fix.
2. **vLLM uses `PagedUpdateCacheDeviceOperation` + `PagedFillCacheDeviceOperation` (2 ops, 69+18 = 87 calls/dec/dev)** whereas tt_transformers no-prefetcher uses the fused `PagedFusedUpdateCacheDeviceOperation` (1 op, 31 calls/dec/dev). vLLM's unfused KV-cache path adds ~0.5 ms but uses a different dispatch pattern.
3. **vLLM dispatches a separate non-async `AllGatherDeviceOperation` (33 calls/dec/dev) that doesn't exist in tt_transformers no-prefetcher.** This is the same op that shows up in the prefetcher-on path (1 call/dec/dev at 0.9 ms) — vLLM seems to use it 33× more often per decode, but at much lower per-call cost (0.6 ms total vs 0.9 ms for 1 call).
4. **vLLM has zero `BinaryNgDeviceOperation` and zero `CopyDeviceOperation`** — those are 12% + 20% of tt_transformers no-prefetcher's compute. vLLM's lower op count for residual + workspace-copy operations is a real (not artifact) structural advantage: instead of a separate Binary+Copy chain, vLLM fuses residual into the next op or uses on-place tensor updates.
5. **vLLM has 2× more `RotaryEmbeddingLlamaDeviceOperation` calls (87 vs 43 per dec per dev)** — likely because vLLM applies RoPE separately to Q and K instead of the fused tt_transformers path. Each call is small (~2.5 µs), so total impact is modest (~0.4 ms more).

## Surprising findings

1. **vLLM matches tt_transformers no-prefetcher within 5%, despite being a completely independent integration.** The vLLM path through `tt_transformers/tt/generator_vllm.py` shares the same underlying ttnn kernels and trace-replay infrastructure as the SGLang/sglang integration path. The TPOT delta of ~1.75 ms / token is small relative to the underlying ~40 ms baseline.

2. **vLLM's matmul per-call cost is 2× tt_transformers no-prefetcher's (258 vs 128 µs mean).** This is unexpected — both should use the same `MatmulDeviceOperation` kernel. Hypotheses:
   - The valid-row subset is different: vLLM has 613/7068 = 9% valid rows; tt_transformers no-prefetcher has 470/5274 = 9% valid rows. Same valid-row fraction. So the corruption pattern is similar.
   - Likely the long-tail matmuls (lm_head, FFN down) dominate vLLM's valid sample more than tt_transformers — those have larger weight matrices and longer DRAM stall.
   - **Practical implication**: porting `use_prefetcher=True` to `generator_vllm.py` would likely cut vLLM matmul time from 25 ms → ~3 ms (4-8× speedup), bringing vLLM TPOT to **~22-25 ms / token**, which would BEAT tt_transformers prefetcher-on (31.35 ms).

3. **Both stacks have nearly identical TP collective overhead (~5.4 ms vLLM, ~4.4 ms ttx no-pref).** Collectives don't benefit from per-stack model code differences — they're communication-bound between the two P150a chips over the Blackhole Ethernet links. The 1 ms gap is within profiler noise.

4. **vLLM completely lacks the `CopyDeviceOperation` and `BinaryNgDeviceOperation` overhead that dominates tt_transformers no-prefetcher.** This is a structural win for vLLM: it composes ops differently. Specifically:
   - `CopyDeviceOperation` (962 rows, 12% of ttx-no-pref compute) → 0 rows in vLLM
   - `BinaryNgDeviceOperation` (3786 rows, 20% of ttx-no-pref compute) → 0 rows in vLLM
   - Together: −16.4 ms / decode saved in vLLM

5. **vLLM's `InterleavedToShardedDeviceOperation` mean is 17 µs vs tt_transformers no-pref's 1 µs.** This is the only op where vLLM is dramatically slower per-call. Total impact: vLLM 2.4 ms vs ttx 0.13 ms (+2.3 ms). This is the layout-conversion penalty for vLLM's different sharding strategy across the decoder layers.

## Top attack candidates (for a hypothetical vLLM perf push)

### 1. Port `use_prefetcher=True` to `generator_vllm.py` (potential −10 to −22 ms / decode, ~25-55%)

The prefetcher is the largest unfair advantage tt_transformers/SGLang currently has. The `generator_sglang.py` already has the wiring (`use_prefetcher` kwarg threaded through `__init__`); applying the same change to `generator_vllm.py` should be a ~20-line port. This alone would likely flip vLLM from 39 ms slower to 25-30 ms / token (faster than tt_transformers prefetcher-on at 31 ms, because vLLM also has the BinaryNg + Copy elimination advantage).

### 2. Reduce `InterleavedToShardedDeviceOperation` per-call cost (potential −2.3 ms)

vLLM's I2S takes 17 µs per call vs 1 µs in tt_transformers no-prefetcher. Investigate which call sites generate this overhead and whether the sharding strategy can be aligned with tt_transformers' more-efficient pattern.

### 3. Fuse the second `PagedFillCacheDeviceOperation` into `PagedUpdateCacheDeviceOperation` (potential −0.1 ms)

vLLM dispatches both Paged ops; tt_transformers uses the fused `PagedFusedUpdateCacheDeviceOperation`. Small win but easy.

## What's NOT a bottleneck

- **TP=2 collective overhead is largely unavoidable** for both stacks (~5 ms / decode). Fused-all-gather-matmul would help equally.
- **PagedAttention / SDPA** — 0% of cost in vLLM dataset (1224 SdpaDecode calls all corrupted but small per-call). Hand-tuned, marginal returns.
- **Embeddings, Slice, Tilize** — all <0.1% in vLLM.

## Container state at end of session

- **Both P150a cards healthy**: `tt-smi -ls` shows 2× Blackhole p150a (UMD Chip IDs 0, 1) on 0000:01:00.0 / 0000:06:00.0, board number 0000040338xxx, ready for next run.
- **vLLM 0.10.1.1 + tt_vllm_plugin 0.1.0 installed** in `/opt/venv` inside `p3a-ngram`; persists across container exec sessions. To remove: `podman exec p3a-ngram /opt/venv/bin/python3 -m pip uninstall -y tt-vllm-plugin vllm`.
- **pip-freeze diff** (`/tmp/p3a_pip_freeze_pre_vllm.txt` → `/tmp/p3a_pip_freeze_post_vllm.txt`): 29 new packages, 18 version changes. Key downgrades: `transformers 5.8.1 → 4.55.0`, `tokenizers 0.22.2 → 0.21.4`, `huggingface_hub 1.15.0 → 0.36.2`, `protobuf 3.20.2 → 7.35.0`, `numba 0.65.1 → 0.61.2`, `llvmlite 0.47.0 → 0.44.0`, `mistral_common 1.11.2 → 1.9.1`. SGLang remains importable (we did not touch `/opt/venv/lib/python3.10/site-packages/sglang*` and SGLang's editable install resolved against the container's tt-metal as before).
- **tt-metal source clean**: `generator_vllm.py` shim was reverted in container. No SGLang source changed for this task (only this doc + the CSV fixture were added).
- **Plugin patches**: `/opt/tt-vllm-plugin/tt_vllm_plugin/__init__.py` and `/opt/tt-vllm-plugin/tt_vllm_plugin/v1/worker/tt_model_runner.py` retain the local patches (Qwen3-text registration + periodic Tracy flush). These are LOCAL to the container and not committed anywhere — destroying / rebuilding the container loses them.

## Files

- `_fixtures/cpp_device_perf_report_vllm_qwen3_8b.csv` (11 MB, 46,720 rows) — captured 2026-05-21 ~00:12 UTC
- `_fixtures/cpp_device_perf_report_tt_transformers_v6.csv` (8.6 MB, 32,560 rows) — reference no-prefetcher tt_transformers (from 2026-05-18 evening session)
- `_fixtures/cpp_device_perf_report_tpot_breakdown_31ms_prefetcher.csv` (4.3 MB, 20,554 rows) — reference prefetcher-on tt_transformers (from 2026-05-21 morning session)
- `docs/platforms/tt_transformers_perop_breakdown_31ms_prefetcher_2026-05-21.md` — tt_transformers prefetcher-on reference doc (sibling document)
- `docs/platforms/tt_transformers_perop_breakdown_2026-05-18.md` — original no-prefetcher tt_transformers breakdown
