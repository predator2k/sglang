# EAGLE-3 on 2× Tenstorrent Blackhole P150a — Maintainer Guide

## Quick Start

```bash
bash python/sglang/srt/hardware_backend/tenstorrent/scripts/repro_eagle3_2xp150a.sh
```

Single launch serves **both** `/generate` and `/v1/chat/completions` correctly.

## What This Directory Contains

Evidence JSON files documenting the path from broken (v73) to unified-mode-working (v111-v113). Each file captures one iteration's measured behavior:

| File | What it shows |
|---|---|
| `FINAL_STATE.md` | Production launch recipes + known behavior |
| `v87_per_request.json` | First non-zero acceptance (3-aux-layer capture unlocks accept_rate > 0) |
| `v88_bonus_only_paris.json` | First factually correct content ("Paris") |
| `v89_perturb.json` | Multi-fact reasoning with per-position embed perturbation |
| `v90_v91_perturb_ceiling.json` | Perturbation magnitude doesn't matter — captured signal dominates |
| `v92_per_pos_multi_decode.json` | Per-position multi-decode breakthrough |
| `v92b_longer_coherent.json` | 48-token sustained coherence |
| `v93_optimal_config.json` | `dtn=3 num_steps=2` config sweep |
| `v94_minimal_spec.json` | `dtn=2 num_steps=1` peak: 9.6 tok/s |
| `v94b_python_endurance.json` | 96-token Wikipedia-quality (Guido van Rossum facts) |
| `v95_suite_results.json` | 8-prompt diverse-domain suite: 7/8 correct |
| `v99_chat_template_limitation.json` | Chat-template loop diagnosed |
| `v102_repro_results.json` | Reproducibility validation |
| `v111_unified_mode.json` | **Unified mode breakthrough — prefill-time chat detection** |
| `v112_alternating_validation.json` | 4 alternating GEN/CHAT requests, all correct |
| `v113_full_reasoning_chain.json` | Chat completions delivers Tokyo with full reasoning |

## Architecture Overview

The work spans three layers:

### 1. SGLang plugin (`python/sglang/srt/hardware_backend/tenstorrent/`)
- `tt_eagle_kernels.py` — torch fallbacks for `sgl_build_tree_kernel_efficient` and `verify_tree_greedy`. Bit-exact vs canonical test fixture.
- `tt_eagle_backend.py` — `TTMultiStepDraftBackend` with TorchNativeAttnBackend delegation and `extend_prefix_lens` reconcile.
- `models/tt_llm.py` —
  - Aux-layer wrapping at decoder layers `[2, mid, N-3]` (the layer_ids EAGLE-3's `set_eagle3_layers_to_capture` defaults to).
  - Per-position multi-decode in `_call_prefill_for_verify` for real per-position 3-aux-layer hidden states.
  - Prefill-time chat-template detection (scans for Qwen3 tokens `{151644, 151645, 151648}`).
  - Verify auto-mode reads chat flag and engages tree-mask via `_skip_self_attention` on attention layers.
- `tp_worker.py` — wires `install_tt_eagle_kernels()` before super init.
- `models/registry.py` — lazy-registers `LlamaForCausalLMEagle3` to dodge a layers/utils circular-import.

### 2. tt-metal patches (`predator2k/tt-metal` branch `tenstorrent-p1`)
- `device.cpp::compute_with_storage_grid_size()` — grid-y clamp gates on `FabricTensixConfig::MUX` (was the wrong gate originally).
- `model_config.py::dram_matmul_config()` — Blackhole multichip falls back to (8,8) matmul to avoid DRAM-sharded's 8×10 grid request.
- `model_config.py::get_attn_input_mem_config / get_attn_qkv_mm_mem_config / get_mlp_*_mem_config / get_mlp_binary_mult_mem_config` — switch WIDTH_SHARDED to DRAM on Blackhole multichip.
- `distributed_norm.py::_force_unsharded` — extended to Blackhole multichip.
- `attention.py` — `_skip_self_attention` flag (tree-mask) is wired up (the patch was applied in P3a.1 NGRAM work but never used until v103).

### 3. Documentation
- `docs/platforms/tenstorrent_design.md` — broader design doc.
- Memory `tenstorrent-eagle3-working.md` — auto-loaded into future Claude sessions.

## Configuration

Recommended for production:
```
--speculative-algorithm EAGLE3
--speculative-draft-model-path /models/qwen3_8b_eagle3
--speculative-eagle-topk 1
--speculative-num-steps 1
--speculative-num-draft-tokens 2
--max-running-requests 1
--context-length 2048
--mem-fraction-static 0.5
--attention-backend torch_native
--disable-cuda-graph
--enable-metrics
```

Environment:
```
SGLANG_PLATFORM=tenstorrent
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged
SGLANG_TT_SPEC_DRAFT_PATH=/models/qwen3_8b_eagle3
SGLANG_TT_SPEC_DRAFT_BACKEND=cpu
```

Tree-mask override (optional, default `auto` is correct):
```
SGLANG_TT_EAGLE_TREE_MASK={auto,0,1}
```

## Measured Behavior

| Endpoint | tok/s | accept_rate | Sample output |
|---|---|---|---|
| `/generate` ("capital of Japan") | 9.33 | 0.476 | "Tokyo, and the capital..." |
| `/generate` ("WW2 ended") | 7.83 | 0.240 | "1945. The United States and Soviet Union..." |
| `/generate` ("Python tech") | 8.34 | 0.343 | "used for a wide range of applications..." |
| `/v1/chat/completions` ("capital of Japan", 200 tok) | n/a | n/a | Full reasoning chain → Tokyo |

`/generate` suite: **8.24 tok/s avg, 7/8 prompts factually correct.**

## Known Limitations

- Token doublings ("the the", "Tokyo Tokyo") visible in output. Doesn't break semantic correctness. Root cause is chain-attention without per-position tree-mask SDPA; a proper kernel patch in `paged_scaled_dot_product_attention_decode` would fix this. Out of session scope.
- Throughput limited by 6× decode calls per verify (multi-decode is required for real per-position hidden states). Trace mode incompatible with our intermediate-tensor capture pattern; recovering trace-speed requires adding aux-hidden as a persistent output in tt-metal's prefill trace machinery (analog of `process_hidden_states_after_prefill_trace`).

## How to Validate

```bash
# Full smoke test (3 prompts, ~30s):
bash scripts/repro_eagle3_2xp150a.sh

# Diverse 8-prompt suite (~2 min):
python3 /tmp/v95_suite.py     # if /tmp/v95_suite.py exists (check git history)

# Single chat probe:
podman exec p3a-ngram bash -lc 'curl -s -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"/models/Qwen3-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Japan?\"}],\"max_tokens\":200,\"temperature\":0.0}"'
```

## History

110+ versioned iterations (v73 → v113) on `predator2k/sglang tenstorrent-p1`. Major commits:
- `675052c60` — EAGLE-3 boots end-to-end (v73)
- `6591b7cdf` — 3-aux-layer capture, first accept > 0 (v87)
- `e0dfc7eb9` — per-position multi-decode (v92)
- `dc047a513` — dtn=2 minimal config 9.6 tok/s (v94)
- `1328d5e4b` — 8-prompt diverse suite 7/8 correct (v95)
- `6de169dc3` — prefill-time chat detection (v111, **unified mode**)
- `3143fa82b` — full reasoning chain demo (v113)
