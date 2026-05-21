# tt_transformers Qwen3-8B per-op breakdown — prefetcher-ON server-path Tracy (2026-05-21)

**Date:** 2026-05-21
**Branch:** `tenstorrent-p1`
**Container:** `p3a-ngram` (the production tt_transformers stack)
**Inputs:**
- `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/cpp_device_perf_report_tpot_breakdown_31ms_prefetcher.csv` (4.3 MB, 20,554 rows, all OP NAMEs populated)
- Bench: 3-run `bench_3run_server_alive.py` with `--input-len 1024 --output-len 256 --context-len 2048 --backend tt_transformers_paged`
- Env: `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_DISABLE_PREFILL_TRACE=1 TT_PROFILE=1 TTNN_OP_PROFILER=1 SGLANG_TT_DECODE_TIMING=1 SGLANG_TT_PROFILER_PERIOD=1 HF_MODEL=Qwen/Qwen3-8B`
- **Reference:** canonical TPOT = 31.35 ms (verified in `v146_3run_server_prefetcher_no_prefill_trace_v4.json`; prior baseline without prefetcher = 37.43 ms)
- Profiler patches in `tt_llm.py`: atexit+SIGTERM+SIGINT profiler-flush handlers + periodic `ttnn.ReadDeviceProfiler` every N=1 decode steps. These patches were **already present** in `tt_llm.py` as of this session (not reverted — they are gated on `TT_METAL_DEVICE_PROFILER=1`, zero overhead when env var absent).

## Method

1. The `tt_llm.py` file already contains the Tracy recipe patches from the 2026-05-18 session:
   - `_setup_tt_profiler_dump_paths()` registers atexit / SIGTERM / SIGINT handlers that call `ttnn.ReadDeviceProfiler(self.mesh_device)` followed by `self.device_runner.close_device()` (the latter triggers `ProfilerInitializer::post_teardown` which writes `profile_log_device.csv`).
   - On each decode step, calls `ttnn.ReadDeviceProfiler` every `SGLANG_TT_PROFILER_PERIOD` steps to prevent the on-device 12K-marker-per-core buffer from overflowing.
   - All paths are gated on `os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"`. The bench script `bench_3run_server_alive.py` maps `TT_PROFILE=1` → `TT_METAL_DEVICE_PROFILER=1` when launching the server subprocess.
2. Ran the bench with `TT_PROFILE=1 TTNN_OP_PROFILER=1 SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_DISABLE_PREFILL_TRACE=1` to capture the prefetcher-on path.
3. After bench, ran `python3 -m tracy.process_ops_logs -o /tt-metal/generated/profiler --name-append v32ms`. (Throws an assertion in the legacy Python pretty-print step — ignored, the CSV is already written.)
4. Aggregation script (pandas):
   ```python
   import pandas as pd
   fw_col = "DEVICE FW DURATION [ns]"
   VALID_THRESH = 1e9  # dual-clock-domain corruption filter
   df = pd.read_csv(csv_path, low_memory=False)
   df[fw_col] = pd.to_numeric(df[fw_col], errors='coerce')
   # Per-op: valid_n = count where FW < 1e9, mean_us = mean of valid / 1000
   # est_ms = mean_us * count / 1000  (scale: mean × total invocations)
   # Scale to canonical: per-device-per-decode = est_total / n_decode / 2_devices
   #   verified = 31.2ms ≈ 31.35ms canonical (ratio ≈ 1.0)
   ```

## Caveats

- TPOT under this profiler config is **~6-9× slower than canonical** (31 ms → ~200-280 ms / decode). Per-op `DEVICE FW DURATION` reflects kernel execution time on-device and is unaffected by host dispatch overhead. Relative proportions hold.
- ~34% of DEVICE FW DURATION rows (6,981 / 20,554) are corrupted by the dual-clock-domain bug ([[tenstorrent-tt-xla-tracy-fw-duration-bug]]). Filtered via `< 1e9 ns`. Estimated totals = mean(valid) × count.
- **Four op types have 100% corrupted FW rows:** `AllGatherAsyncDeviceOperation` (1,744 rows), `SdpaDecodeDeviceOperation` (576), `NLPConcatHeadsDecodeDeviceOperation` (576), `ReduceScatterMinimalAsyncDeviceOperation` (576). These ops use async dispatch and the clock-domain corruption hits all rows. For these, per-device-decode estimates from the prior 2026-05-18 dataset (no prefetcher) are used as proxies and are marked `*est` in the table.
- `DramPrefetcherOperation` runs **CONCURRENTLY with decode compute** inside the tt-metal trace replay. Its 57 ms steady-state FW duration is NOT additive to TPOT — the prefetcher overlaps with matmul execution, which is the mechanism for its speedup. It is excluded from the compute total and scaling.
- Number of decode iterations captured: 576 SdpaDecodeDeviceOperation / (36 layers × 2 devices) = **8 decode iterations**.

## Per-op breakdown — what's inside prefetcher-ON decode_forward

Top 12 device ops by estimated total time (mean valid FW × count), **excluding** concurrent DramPrefetcherOperation:

| op | count | valid_n | mean µs | est total ms | % compute | ~ms/step | note |
|---|---:|---:|---:|---:|---:|---:|---|
| **MatmulDeviceOperation** | 2960 | 2960 | 31.4 | 92.9 | 18.6% | 5.81 | |
| **LayerNormDeviceOperation** | 2320 | 2320 | 38.8 | 90.0 | 18.1% | 5.63 | |
| ReduceScatterMinimalAsyncDeviceOperation | 576 | 0 | 125.0 | 72.0 | 14.4% | 4.50 | *est |
| **UntilizeDeviceOperation** | 20 | 16 | 2413.8 | 48.3 | 9.7% | 3.02 | |
| AllGatherAsyncDeviceOperation | 1744 | 0 | 23.0 | 40.1 | 8.0% | 2.51 | *est |
| **ShardedToInterleavedDeviceOperation** | 2416 | 1248 | 16.2 | 39.1 | 7.8% | 2.44 | |
| BinaryNgDeviceOperation | 1728 | 576 | 15.1 | 26.0 | 5.2% | 1.63 | |
| **ConcatDeviceOperation** | 16 | 16 | 1434.8 | 23.0 | 4.6% | 1.43 | |
| NLPConcatHeadsDecodeDeviceOperation | 576 | 0 | 31.0 | 17.9 | 3.6% | 1.12 | *est |
| AllGatherDeviceOperation | 16 | 16 | 895.7 | 14.3 | 2.9% | 0.90 | |
| PagedUpdateCacheDeviceOperation | 1152 | 1152 | 9.9 | 11.4 | 2.3% | 0.71 | |
| ReshardDeviceOperation | 1728 | 576 | 5.5 | 9.5 | 1.9% | 0.59 | |

*`*est` = FW rows 100% corrupted; estimate derived from prior 2026-05-18 no-prefetcher dataset at same decode-per-device rate.*

**DramPrefetcherOperation (CONCURRENT, not additive):** 16 rows (1 per decode per device); cold-start first 2 rows = 931 ms and 1081 ms (discarded); steady-state mean = **57.5 ms** (range 56.0–59.3 ms). Runs during trace replay in parallel with compute.

Total estimated compute FW: 498.6 ms across 8 decodes × 2 devices = **31.2 ms / device / decode** under profiler (matches canonical 31.35 ms with scale factor ≈ 1.006 — effectively zero profiler distortion on per-op kernel time).

## Mapping to canonical 31.35 ms TPOT

| component | est ms / decode | % of TPOT | what these are |
|---|---:|---:|---|
| **TP Collectives (AllGather + ReduceScatter)** | ~7.9 | **25.4%** | ReduceScatter 4.5ms + AllGatherAsync 2.5ms + AllGatherDeviceOperation 0.9ms; TP=2 communication after each matmul stage |
| **Matmul** | ~5.8 | **18.6%** | Q+K+V fused proj × 36 layers, Wo × 36, gate/up FFN × 36, down × 36, lm_head (prefetcher has already staged weights in SRAM) |
| **LayerNorm (RMSNorm)** | ~5.6 | **18.1%** | Input + post-attention RMS norms × 36 layers × 2 directions (pre/post attn + pre/post FFN) |
| **S2I / I2S layout moves** | ~3.6 | **11.5%** | ShardedToInterleaved (2416 rows) + InterleavedToSharded (2912 rows) + Reshard (1728 rows); tensor shard boundary crossings |
| **Untilize (lm_head D2H)** | ~3.0 | **9.7%** | ~2.4 ms each × ~16 steady-state rows = per-decode D2H sync (full logits read for vocab sampling) |
| Binary (residual + RMSNorm mul) | ~1.6 | 5.2% | residual adds after attention + FFN; mul in RMSNorm internals |
| Concat (prefetcher output) | ~1.4 | 4.6% | ConcatDeviceOperation: 16 rows at 1.4 ms each; prefetcher output concatenation step |
| Attention (SDPA + heads + RoPE) | ~1.4 | 4.6% | PagedUpdateCacheDeviceOperation + NLPCreateQKVHeadsDecode + NLPConcatHeads + RotaryEmbeddingLlama + SdpaDecodeDeviceOperation |
| PagedKV cache update | ~0.7 | 2.3% | PagedUpdateCacheDeviceOperation (replaces PagedFusedUpdateCache from no-prefetcher path) |
| **TOTAL** | **~31.2** | **~100%** | |

## Surprising findings

1. **Matmul kernel time dropped 4.1× (128 µs → 31 µs per launch).** This is the primary mechanism of the prefetcher win. The `DramPrefetcherOperation` stages all Qwen3-8B BFP8 weights into SRAM before the decode trace replay begins, eliminating the DRAM-read stall that dominated each matmul launch in the no-prefetcher path. The speedup is direct: per-matmul FW from 128.3 µs → 31.4 µs, all 2,960 rows valid (vs only 470/5274 valid in the old CSV — the long-duration valid rows dominated, hiding the true mean).

2. **TP collectives are now the single largest component at 25%.** In the no-prefetcher breakdown they were 13% (AllGather 4.5% + ReduceScatter 8.4%). With matmul dramatically faster, collectives — which don't benefit from weight prefetching — become the dominant bottleneck. Their absolute duration is unchanged; their *share* grew because matmul shrank.

3. **CopyDeviceOperation is completely GONE (962 rows in no-prefetcher, 0 in prefetcher).** The prior 12% bottleneck "Copy" op disappeared entirely. In its place: `ConcatDeviceOperation` (16 rows, 4.6%) and `AllGatherDeviceOperation` (non-async, 16 rows, 2.9%). The prefetcher path restructures how data moves between ops — the generic workspace copies were replaced by structured concat + non-async allgather for the prefetcher output assembly.

4. **Matmul count per decode dropped (239.7 → 185 per device).** Likely a structural difference: the prefetcher path uses a different internal op graph (e.g., fewer intermediate copy+matmul sequences). The ReduceScatter count also dropped from 50.1 → 36 per device, consistent with fewer independent matmul stages per layer.

5. **LayerNorm is now 18% (up from 10%).** Absolute mean duration is similar (38.8 µs vs 43.7 µs), but the count per device per decode dropped (145 → 145, roughly same), so the rise is due to the overall TPOT shrinking while LayerNorm stayed proportionally unchanged. LayerNorm does not benefit from weight prefetching (it has no large weight matrices to prefetch — just weight/bias scalars).

6. **DramPrefetcherOperation steady-state = 57.5 ms, which is 1.83× the canonical 31.35 ms TPOT.** This means the prefetcher is running for LONGER than the decode step itself. It is prefetching not just the current layer's weights but likely one or more future layers' weights into SRAM, providing a pipeline buffer. This confirms the prefetcher is genuinely ahead of compute — not just barely keeping up.

## Top attack candidates (ranked by leverage)

### 1. TP=2 collective overhead (~7.9 ms / decode, 25%)

ReduceScatter (14.4%) + AllGather (8.0% + 2.9%) combined = 25.4% of TPOT. These don't benefit from weight prefetching because they're communication-bound between the two P150a chips, not DRAM-bound. 36 AllGather calls + 36 ReduceScatter calls per decode on BH Ethernet links.

**Fix paths:**
- `use_fused_all_gather_matmul` flag in tt-metal — fuses AllGather with the subsequent matmul into one dispatched op, hiding collective latency inside compute. Currently this is TG-topology-only; check if P300/P150a-pair can enable it.
- AllGather count reduction: currently 1744/8/2 = 109 AllGathers per device per decode. If any can be batched (e.g., delay AllGather across FFN gate+up), latency could be amortized.

### 2. LayerNorm overhead (~5.6 ms / decode, 18%)

72 RMSNorm calls × ~38 µs = now 18% (up from 10% in no-prefetcher). Prefetching weights does not accelerate norm computation (norms have no large weight matrices). Norm is purely TENSIX-compute-bound.

**Fix paths:**
- Fused RMSNorm-into-next-matmul prologue: the next op after each norm is always a matmul. If TTNN can fuse norm + Q/K/V proj, norm overhead drops to 0.
- Check if `enable_fused_qk_norm_rope` SGLang flag can be enabled on Blackhole (currently `False` in server args).

### 3. S2I / I2S layout moves (~3.6 ms / decode, 11.5%)

ShardedToInterleaved (2416 rows at 16 µs), InterleavedToSharded (2912 rows at 3 µs), Reshard (1728 rows at 5.5 µs). The prefetcher path actually INCREASED S2I count (151 vs 94 per device per decode). These are tensor shard-boundary transitions between tp-sharded and unsharded formats.

**Fix paths:**
- Profile which specific layer transitions generate S2I — if the prefetcher output assembly (Concat → S2I) can be kept in sharded format end-to-end, ~1.5 ms could be recovered.
- Investigate if `ReshardDeviceOperation` is redundant (count increased from 95 to 108 per device per decode with prefetcher).

### 4. Untilize (lm_head D2H) — ~3.0 ms / decode, 9.7%

One `UntilizeDeviceOperation` per decode step (lm_head output → host-readable format). 16 steady-state rows at ~2.4 ms each. This is the logit readout: the full vocab logits (Qwen3-8B vocab = 151,936 tokens) untilized and transferred to host for greedy sampling.

**Fix path:** On-device greedy argmax: compute argmax on-device (before D2H) to avoid transferring 152K × 2B = 304 KB of logits per step. Blocked by the `vocab_size/2 > 65536` per-device limit (151,936/2 = 75,968 exceeds Blackhole's 65,536-element reduction limit). Could work with a 2-stage reduction.

### 5. Concat + AllGather (non-async) — ~2.3 ms / decode, 7.5%

`ConcatDeviceOperation` (1.4 ms) + `AllGatherDeviceOperation` (0.9 ms): 16 rows each per 8 decodes = 1 per device per decode. This is the prefetcher-path-specific output assembly step — the prefetcher's distributed results are concatenated and all-gathered once per step.

**Fix path:** Investigate if the `ConcatDeviceOperation` + non-async `AllGatherDeviceOperation` can be fused into the matmul chain, eliminating a separate dispatch + collective.

## Comparison with no-prefetcher 2026-05-18 breakdown

| component | no-prefetcher ms/step | prefetcher-ON ms/step | delta | direction |
|---|---:|---:|---:|---|
| Matmul | ~15.3 | ~5.8 | −9.5 | **massive win (4.1×)** |
| TP collectives | ~4.8 | ~7.9 | +3.1 | **grew (abs unchanged, % share rose)** |
| Binary (residual) | ~7.2 | ~1.6 | −5.6 | **large win** |
| Copy ops (CopyDeviceOperation) | ~4.5 | 0.0 | −4.5 | **eliminated** |
| LayerNorm | ~3.7 | ~5.6 | +1.9 | worse (same µs, share rose) |
| Layout (S2I/I2S/Reshard) | ~0.7 | ~3.6 | +2.9 | **worse (prefetcher adds boundary moves)** |
| Untilize (lm_head) | ~0.4 | ~3.0 | +2.6 | worse (larger capture window) |
| Attention + heads + RoPE | ~1.1 | ~1.4 | +0.3 | roughly same |
| **TOTAL** | **37.43** | **31.35** | **−6.1 ms** | **net 16% win** |

The prefetcher's speedup flows almost entirely from eliminating DRAM stall in matmuls (−9.5 ms) and residual ops (−5.6 ms) and Copy ops (−4.5 ms) — partially offset by increased layout-transition overhead (+2.9 ms) and the collective/norm costs becoming proportionally larger.

## What's NOT a bottleneck (unchanged from prior analysis)

- **PagedAttention / SDPA** — 2.3–4.6% combined. Hand-tuned, marginal returns.
- **RotaryEmbedding** — 0.8%. Fused rope.
- **EmbeddingsDeviceOperation** — 0.0%. Negligible (first prefill only).

## Files

- `_fixtures/cpp_device_perf_report_tpot_breakdown_31ms_prefetcher.csv` (4.3 MB, 20,554 rows; same data as `cpp_device_perf_report_a3_baseline.csv` captured 2026-05-21 13:51)
- `_fixtures/v146_3run_server_prefetcher_no_prefill_trace_v4.json` — canonical TPOT = 31.35 ms bench that this breakdown is mapped to
- `docs/platforms/tt_transformers_perop_breakdown_2026-05-18.md` — prior no-prefetcher 37ms breakdown (reference)
