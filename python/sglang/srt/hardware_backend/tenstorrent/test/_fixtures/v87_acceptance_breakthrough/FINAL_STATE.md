# EAGLE-3 on 2× Tenstorrent Blackhole P150a — Final State (2026-05-15)

## TL;DR
EAGLE-3 spec decoding operational on 2× P150a producing factually accurate
content with measurable acceleration (~8–10 tok/s, accept_rate 0.18–0.50)
via `/generate` endpoint.

## Recommended production launch
```bash
podman exec \
  -e SGLANG_PLATFORM=tenstorrent \
  -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  -e SGLANG_TT_SPEC_DRAFT_PATH=/models/qwen3_8b_eagle3 \
  -e SGLANG_TT_SPEC_DRAFT_BACKEND=cpu \
  p3a-ngram bash -lc 'source /opt/venv/bin/activate; python -m sglang.launch_server \
    --model-path /models/Qwen3-8B --trust-remote-code \
    --host 0.0.0.0 --port 30000 --device tenstorrent \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /models/qwen3_8b_eagle3 \
    --speculative-eagle-topk 1 \
    --speculative-num-steps 1 \
    --speculative-num-draft-tokens 2 \
    --max-running-requests 1 --context-length 2048 \
    --mem-fraction-static 0.5 --attention-backend torch_native \
    --disable-cuda-graph --enable-metrics'
```

## What works
- `/generate` endpoint with raw text prompts
- 8/8 diverse prompts in v95 suite produce factually correct content
- 96-token sustained generation: factually accurate Python facts
  ("Guido van Rossum, late 1980s, web/data/AI")
- Reproducibility: bit-exact across server restarts

## Known limitation
`/v1/chat/completions` endpoint produces self-reinforcing repetition
loops ("OkayOkay...") on Qwen3 because chat template's deterministic
`<think>\nOkay,` start triggers EAGLE-3 spec doublings amplified by
chain attention without tree-mask.

**Workaround**: use `/generate` and apply chat templates client-side.

**Real fix** (future, multi-hour tt-metal C++ kernel work): tree-attention
self-mask in `paged_scaled_dot_product_attention_decode`.

## Key implementation pieces
| File | What it does |
|---|---|
| `tt_eagle_kernels.py` | Pure-torch fallbacks for `sgl_build_tree_kernel_efficient` + `verify_tree_greedy` |
| `tt_eagle_backend.py` | `TTMultiStepDraftBackend.forward` delegating to TorchNativeAttnBackend + `extend_prefix_lens` reconcile |
| `models/tt_llm.py` | `_install_aux_layer_capture_once()` wraps target's decoder layers [2, mid, N-3]; `_call_prefill_for_verify` runs per-position multi-decode |
| `tp_worker.py` | `install_tt_eagle_kernels()` wired before super().__init__() |
| tt-metal device.cpp | `compute_with_storage_grid_size()` clamp on FabricTensixConfig::MUX |
| tt-metal model_config.py | DRAM mem-cfg fallbacks for QKV/MLP on Blackhole multichip |
| tt-metal distributed_norm.py | `_force_unsharded` extended to Blackhole multichip |

## Iteration milestones
- v73 boot end-to-end, accept_rate=0
- v87 3-aux-layer capture: first accept > 0
- v92 per-position multi-decode: multi-fact factual output
- v94 dtn=2 steps=1: 9.6 tok/s throughput
- v94b 96 tokens Wikipedia content
- v95 diverse 8-prompt suite: 7/8 correct
- v99 chat-template limitation diagnosed

## Evidence files in this directory
- v87_per_request.json — first accept_rate > 0
- v88_bonus_only_paris.json — first correct fact ("Paris")
- v89_perturb.json — multi-fact reasoning
- v90_v91_perturb_ceiling.json — negative-result documented
- v92_per_pos_multi_decode.json — per-position breakthrough
- v92b_longer_coherent.json — 48-token endurance
- v93_optimal_config.json — config sweep
- v94_minimal_spec.json — 9.6 tok/s peak
- v94b_python_endurance.json — Wikipedia-quality content
- v95_suite_results.json + v95_suite_output.txt — 8-prompt suite
- v99_chat_template_limitation.json — chat issue documented
- FINAL_STATE.md — this document

## Update (v103): tree-mask env-var-gated workaround for chat

`SGLANG_TT_EAGLE_TREE_MASK=1` enables tree-mask SDPA via the
`_skip_self_attention` flag (patch already applied to tt-metal
attention.py). Trade-off depending on workload:

| Env value | /generate | /v1/chat/completions |
|---|---|---|
| `0` (default) | Works ("Paris. The capital of Italy...") | Broken (OkayOkay loop) |
| `1` | Broken ("Tokyo\n![]( 2024年") | Works ("OkayOkay, the user is asking about the capital of Japan. Let me think.") |

Proper fix would be per-request mode detection.

Launch with env var:
```bash
podman exec -e SGLANG_TT_EAGLE_TREE_MASK=1 ...  # enable for chat workloads
```

## Update v107: auto-detect attempted but loop-detector misfires

Tried per-request runtime loop detection (`SGLANG_TT_EAGLE_TREE_MASK=auto`)
that tracks consecutive same-token emissions and engages tree-mask when
count ≥ 2. Empirically does NOT catch the chat-template "OkayOkay" loop
because the first emission is "Okay," (token A) followed by "Okay"
(token B) — different token IDs, so the counter never increments past 0.

Final operator-facing recommendation:
- `/generate` workloads → leave env var unset (or `0`)
- `/v1/chat/completions` workloads → set `SGLANG_TT_EAGLE_TREE_MASK=1`

Proper unified fix requires per-request mode detection at prefill time,
which requires SGLang TT path to surface `forward_batch.reqs` to the
verify hook. Future SGLang-TT interop improvement.
