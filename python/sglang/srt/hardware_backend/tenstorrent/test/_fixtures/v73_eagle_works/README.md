# EAGLE-3 on 2× P150a — Working v73 (2026-05-15) — **HISTORICAL**

> **Superseded by v87 → v94 → v111-v113.** This fixture captures the
> initial end-to-end EAGLE-3 boot milestone (real chat completions, but
> accept_rate ≈ 0 because hidden_states was a bf16-zero stub). Subsequent
> iterations unblocked acceptance (v87: 3-aux-layer capture), found the
> optimal spec config (v94: `num_steps=1 num_draft_tokens=2`, 9.6 tok/s),
> and added unified-mode prefill-time chat-template detection (v111).
>
> For current production launch and the canonical EAGLE-3 example, use:
> ```
> bash python/sglang/srt/hardware_backend/tenstorrent/scripts/repro_eagle3_2xp150a.sh [generate|chat|both]
> ```
> Maintainer guide: `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v87_acceptance_breakthrough/README.md`.
>
> The launch config shown below uses the pre-v94 spec tuning
> (`num_steps=5 num_draft_tokens=6`); don't copy it for production use.

End-to-end EAGLE-3 speculative decoding on 2× Tenstorrent Blackhole P150a
producing real Qwen3-8B chat completions.

## Reproducibility

```
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
    --speculative-eagle-topk 1 --speculative-num-steps 5 --speculative-num-draft-tokens 6 \
    --max-running-requests 1 --context-length 2048 --mem-fraction-static 0.5 \
    --attention-backend torch_native --disable-cuda-graph'
```

## Evidence

- `v73_gen_metrics.json` — chat completion response, 16 tokens, "Say hi." → "Okay, the user said 'Say hi.' So I need to respond"
- `server_info.json` — server args showing EAGLE3 + qwen3_8b_eagle3 draft
- Prior gen: "Hi" → "Okay, the user just said" (8 tokens)

## The breakthrough patches (T2.2.M, T2.2.N, plus reconcile)

1. `tt_eagle_kernels.py` (new) — pure-torch fallbacks for
   `sgl_build_tree_kernel_efficient` and `verify_tree_greedy`, injected into
   `eagle_utils.py` namespace at worker init. Bit-exact vs canonical test.
2. `tt_metal/impl/device/device.cpp` — `compute_with_storage_grid_size` clamp
   now gates on `MetalContext::get_fabric_tensix_config() != DISABLED` (was
   `DispatchCoreType::ETH` which never matched P300 under MUX).
3. `model_config.py` — extend single-chip `is_blackhole()` fallbacks to:
   - `dram_matmul_config` → 8×8 matmul instead of DRAM-sharded (avoids the
     `get_optimal_dram_bank_to_reader_assignment` 8×10 grid request)
   - `get_attn_input_mem_config` DECODE → DRAM (avoids WIDTH_SHARDED)
   - `get_attn_qkv_mm_mem_config` DECODE → DRAM (avoids WIDTH_SHARDED output)
   - `get_mlp_ff1_3_mem_config` / `get_mlp_ff2_mem_config` DECODE → DRAM
   - `get_mlp_binary_mult_mem_config` DECODE → DRAM
4. `distributed_norm.py` — `_force_unsharded` now also fires on Blackhole
   multichip (RMSNorm sharded path fails because input is now DRAM).
5. `tt_eagle_backend.py` — `TTMultiStepDraftBackend.forward` delegates to
   inner `TorchNativeAttnBackend`; also reconciles `extend_prefix_lens` to
   `seq_lens - extend_seq_lens` when SGLang's metadata is stale.
6. `tt_llm.py:_call_prefill_for_verify` — returns `hidden_states` stub
   (bf16 zeros, shape `[bs*draft_token_num, hidden_size]`) so EAGLE's
   `spec_info.hidden_states[accept_index]` doesn't crash on NoneType.

## Known limitations (expected, deferred to follow-up)

- Target hidden states are not extractable from the TT device, so the
  draft re-feed runs on zero hidden_states stubs. Acceptance rate will be
  near zero — every generated token comes from the bonus (target's own
  prediction). EAGLE pipeline runs but doesn't actually accelerate yet.
- Throughput ~3.5s per token at this config (mostly the per-verify
  target-decode trace replay, ~3s of that is fixed overhead).
- The `dram_matmul_config` → 8×8 fallback gives up the DRAM-sharded
  matmul perf optimization. Recovering it requires fixing
  `get_optimal_dram_bank_to_reader_assignment` to clamp to the 12×8
  grid on P300 under MUX.
