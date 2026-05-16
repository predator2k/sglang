# v140: TT vLLM vs SGLang Decode Gap Analysis

## Question
Why does TT vLLM get 30.7ms/decode vs our 36.35ms/decode on Qwen3-8B?

## Method
Inspected the TT vLLM Docker image `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e867533-22be241`
without running inference (firmware incompatible). Compared all model layer files
against our tt-metal fork (`/home/mhnie/tt-metal-sglang/`, branch `tenstorrent-p1`,
base commit `89686ee78d`).

## Key Finding: Not an Apples-to-Apples Comparison

The vLLM container is built for **`ARCH_NAME=wormhole_b0`** (N300 hardware).
Our deployment runs on **Blackhole P150a** (P300_X2 MUX topology).

The two hardware platforms have fundamentally different kernel dispatch and memory
hierarchies. Almost every decode-critical difference traces back to Blackhole
workarounds we added to make tt-metal run on P300_X2 MUX at all.

## tt-metal Version Relationship

| Source | Commit | Relationship |
|--------|--------|-------------|
| vLLM image | `e867533fc55` | Older |
| Our fork base | `89686ee78d` | ~4012 commits newer than e867533 |

Both commits exist in the same upstream history. Our fork is significantly more recent.

## Root Causes (5 identified)

### 1. MLP Fidelity: HIFI2_FP16 vs LOFI (~3-5ms estimated)

**vLLM (all non-Qwen2.5 models including Qwen3-8B):**
```python
# Falls through to default else branch — no Qwen3-8B specific block exists
settings = {
    "TensorPrecision": {TensorGroup.FF1_FF3: PrecisionSetting.BFP4},
    "OpFidelity": {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI},
}
```
LOFI = `MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True`

**Our fork (Qwen3-8B "balanced" mode, the default):**
```python
inst = cls({
    "TensorPrecision": {TensorGroup.FF1_FF3: PrecisionSetting.BFP4},
    "OpFidelity": {OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16},
})
```
HIFI2_FP16 = `MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True`

**Impact:** HIFI2 has higher compute cost than LOFI. With BFP4 weights, the MLP
FF1/FF3 matmuls execute with full HiFi2 fidelity instead of LoFi. This affects
36 layers x 2 matmuls per layer = 72 matmul ops per decode step. This is likely
the single largest contributor to the ~5.65ms gap.

**Fix:** Set `SGLANG_TT_QWEN3_PRECISION=lofi` (needs implementation) or change
the default "balanced" mode to match vLLM's LOFI. Accuracy impact must be verified.

### 2. Blackhole DRAM-routing workarounds (~1-2ms estimated)

Our P3a.2 patches force multiple decode-path memory configs from L1-sharded to
DRAM-interleaved on Blackhole:

| Config | vLLM (Wormhole) | Ours (Blackhole) |
|--------|----------------|-------------------|
| `get_attn_input_mem_config` (decode) | `WIDTH_SHARDED` in L1 | `DRAM_MEMORY_CONFIG` |
| `get_attn_qkv_mm_mem_config` (decode) | `WIDTH_SHARDED` output | `DRAM_MEMORY_CONFIG` |
| `get_mlp_ff1_3_mem_config` (decode) | `WIDTH_SHARDED` output | `DRAM_MEMORY_CONFIG` |
| `get_mlp_ff2_mem_config` (decode) | `WIDTH_SHARDED` output | `DRAM_MEMORY_CONFIG` |
| `distributed_norm` (decode) | `in_sharded=True, out_sharded=True` | `_force_unsharded=True` + DRAM move |

Each DRAM roundtrip adds latency. With ~36 decoder layers, each paying this cost
for norm + QKV + WO + FF1/FF3 + FF2, the cumulative impact is significant.

**Why:** Blackhole P300_X2 MUX dispatch rejects WIDTH_SHARDED inputs/outputs in
`dram_matmul_config` and `matmul_device_operation.cpp`. These are known hardware
limitations of the Blackhole grid topology (8x8 max compute grid vs Wormhole's 8x8 default).

### 3. Norm unsharded fallback (~0.5-1ms estimated)

In `distributed_norm.py`, our Blackhole patch adds:
```python
_force_unsharded = not self.args.is_multichip or _is_bh_p3a2()
if _force_unsharded:
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
x = self.norm(x, mode=mode, in_sharded=False, out_sharded=False, ...)
```

The vLLM version runs the sharded path:
```python
x = self.norm(x, mode=mode, in_sharded=(mode == Mode.DECODE), out_sharded=(mode == Mode.DECODE), ...)
```

Sharded RMS norm on L1 is significantly faster than unsharded norm reading from DRAM.

### 4. Attention QKV sharded-to-interleaved patch (~0.3ms estimated)

Our attention.py adds a P3a.2 patch for the QKV fused matmul output:
```python
# Our version: conditional handling
if xqkv_fused_sharded.is_sharded():
    xqkv_fused = ttnn.sharded_to_interleaved(...)
else:
    xqkv_fused = ttnn.to_memory_config(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG)
    if xqkv_fused.dtype != ttnn.bfloat16:
        xqkv_fused = ttnn.typecast(xqkv_fused, ttnn.bfloat16)
```

vLLM version always does the direct sharded-to-interleaved (no conditional).
The extra branch + potential typecast adds overhead.

### 5. model.py logits DRAM move (~0.1ms estimated)

Our model.py adds an extra DRAM move at the end of prefill/decode logits:
```python
logits = self.lm_head(x)
logits = ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # OUR ADDITION
```

The vLLM version returns logits directly from lm_head without this extra move.

## What is NOT Different (ruled out)

- **Rope implementation:** Both use mllama rope (use_hf_rope=False by default)
- **use_qk_fused:** Both default to True for non-multimodal on multi-chip
- **CCL config (chunks_per_sync, num_workers_per_link):** Identical values
- **Compute kernel config definitions:** Same WormholeComputeKernelConfig values
  for LOFI, HIFI2, HIFI2_FP16, HIFI4
- **Decode trace capture flow:** Same pattern (compile run, then capture, then execute)
- **Generator.decode_forward signature:** Identical interface (both call
  `decode_forward` with tokens, start_pos, page_table, kv_cache, enable_trace)
- **decode_forward_text method rename:** Our fork patches
  `super().decode_forward_text()` to `super().decode_forward()` in
  generator_sglang.py but this is a fix, not a perf difference
- **read_decode_output / process_decode_output_host:** Nearly identical

## Actionable Next Steps

### Quick win: Test LOFI mode for MLP (~3-5ms)
Add a `lofi` option to `SGLANG_TT_QWEN3_PRECISION` and benchmark. If accuracy
holds, this alone may close 60-90% of the gap.

### Medium-term: Blackhole L1-sharded matmul enablement
The Blackhole DRAM workarounds are the structural penalty. As upstream tt-metal
improves Blackhole support (our base is already 4K commits newer), these workarounds
may become unnecessary. Track:
- `dram_matmul_config` accepting WIDTH_SHARDED on Blackhole
- `matmul_device_operation.cpp` grid checks for P300_X2

### Long-term: Native Blackhole DRAM-sharded norm + matmul pipeline
Optimal decode on Blackhole requires matmuls that stay in L1 with proper sharding,
avoiding DRAM roundtrips per layer.

## Estimated Breakdown

| Source | Estimated Impact | Fixable? |
|--------|-----------------|----------|
| MLP HIFI2_FP16 vs LOFI | ~3-5ms | Yes (env var) |
| DRAM-routing workarounds | ~1-2ms | Needs upstream |
| Norm unsharded fallback | ~0.5-1ms | Needs upstream |
| QKV conditional branch | ~0.3ms | Possibly |
| Logits DRAM move | ~0.1ms | Yes |
| **Total estimated** | **~5-8ms** | |
| **Actual gap** | **5.65ms** | |

## Files Compared

| File | Lines diff (approx) | Decode-relevant? |
|------|---------------------|-----------------|
| model_config.py | 351+ | YES (precision, mem configs) |
| generator.py | 600+ | YES (prefetcher, trace) |
| attention.py | 300+ | YES (rope, QKV, sharding) |
| mlp.py | 20+ | YES (minimal_matmul, conditional) |
| model.py | 300+ | YES (batched prefill, logits) |
| decoder.py | 20+ | Minor |
| distributed_norm.py | 14+ | YES (unsharded fallback) |
| lm_head.py | 6+ | Minor |
| rope.py | 360+ | No (HfRope unused in decode) |
| embedding.py | 3 | No |
| generator_sglang.py | 50+ | Minor (prefetcher) |
| generator_vllm.py | 300+ | N/A (vLLM-specific bridge) |
