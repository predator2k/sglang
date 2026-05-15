# Status vs. user goals (2026-05-15, post-cron-iteration 2)

User goals (cron-loop directive):
1. **Without EAGLE spec, beat Tenstorrent's own offering** on Qwen3
2. **With EAGLE spec, beat without-spec perf** (spec must be net-positive)

## Numbers on file (single-user, ctx=2048, --max-running-requests 1)

| Config | TTFT (ms) | TPOT (ms) | tok/s | Source |
|---|---|---|---|---|
| Qwen3-8B no-spec, 2x P150a | 47–51 | 36 | **27.4** | v115 |
| Qwen3-8B no-spec, 1x P150a | 55–59 | 36 | 26.8 | v116 |
| Qwen3-8B EAGLE-3 path B | 80 | 100 | 8–10 | v114b |
| Qwen3-1.7B no-spec, 2x P150a | 23–26 | 22 | **45.9** | v116 |
| tt-metal Llama-3.1-8B / P150 single (PERF.md) | 76 | 30 | 33.6 | tt_transformers PERF.md |
| tt-metal Llama-3.2-1B / N150 single (PERF.md) | 26 | 11.4 | 87.8 | tt_transformers PERF.md |

## Goal 1 status

- Qwen3-1.7B is at 45.9 tok/s. tt-metal's closest comparable (no Qwen3-1.7B in PERF.md) is Llama-3.2-3B on N300 = 68 tok/s. Our setup (2x P150a, Qwen3-1.7B, SGLang) is below that ceiling by ~33%.
- Qwen3-8B at 27.4 tok/s. tt-metal ref Llama-3.1-8B P150 single = 33.6 tok/s. We're 18% below that ceiling. SGLang adds ~6 ms/tok overhead vs raw tt-metal demo.
- The user cited "50+ tok/s" without specifying model/config. **No Tenstorrent-published number for our exact setup (Qwen3-8B, 2× P150a, SGLang) hits 50+ tok/s.** The 50+ figure likely refers to: smaller model, raw tt-metal demo (no SGLang scheduler), or batched serving (we're at batch=1).

## Goal 2 status

EAGLE-3 spec on Qwen3-8B is **3× SLOWER** than no-spec baseline (8 vs 27.4). The cause is unambiguous (v115):
- Per verify cycle = 2× decode_forward at `enable_trace=False`
- enable_trace=False is mandatory: the Python aux-capture wrapper is trace-incompatible
- 150ms/cycle ÷ 1.5 accept_length = 100 ms/token = 10 tok/s

To beat 27.4 tok/s baseline, spec would need TPOT ≤ 36 ms. With current architecture's 2 decodes × 75 ms = 150 ms per cycle, even 100% acceptance (2 tok/cycle) gives 75 ms/token = 13 tok/s — STILL slower than baseline.

The only architectural path: **collapse to 1 decode per verify cycle** (path A: single `prefill_forward` returning all dn positions). Blocked because tt_transformers' `prefill_forward_text` returns last-token logits only.

## What I tried this cron-loop arc

1. **v116**: Mapped Qwen3-8B baseline (27.4), Qwen3-1.7B baseline (45.9), single-chip 8B (26.8). Confirmed TP=2 not the bottleneck.
2. **v117**: Identified DRAM Prefetcher gap (tt-metal demos gate Prefetcher on "Llama"+"8B" string match). Plumbed `SGLANG_TT_USE_PREFETCHER` env-gated (default off) through to tt-metal's `initialize_sglang_model`. Added Qwen3-1.7B/Qwen3-8B to VERIFIED_MODEL_CONFIGS. **Blocked**: prefetcher sender cores at row 9 outside MUX-clamped grid; L1 CB clash with norm op on (0,0).
3. **v118 (this)**: Tried page-size 32 (rejected by tt-metal tile alignment), back-to-back warm probes (stable 45–46 tok/s for Qwen3-1.7B).

## What would actually close the gaps (each is a focused session, not a cron tick)

### Goal 1 path (push Qwen3-1.7B past 50 tok/s):

A. **Get the Prefetcher working for Qwen3-1.7B.** Requires either:
   - Custom prefetcher core mapping that fits MUX-clamped 12×8 grid (i.e., drop the row-9 sender pair, sacrifice 2 DRAM banks of prefetch capacity).
   - Coordinate the prefetcher's static L1 CB region with the model's per-op L1 allocator (sub_device API plumbing).
   
B. **Add per-decoder JSON config for Qwen3-1.7B** (model_params/Qwen3-1.7B/performance_decoder_config.json). Mirror Llama-3.1-8B pattern that keeps last decoder at safe precision and uses BFP4 FF1_FF3 elsewhere. Likely small (1–3%) effect on perf but might also enable currently-skipped fused kernels.

C. **Per-token SGLang overhead reduction** (~5 ms/tok currently). Hard to find more headroom here without architectural changes.

### Goal 2 path (spec > no-spec):

A. **Single prefill_forward verify** (path A from v114_tpot_analysis). Requires modification to tt-metal's `prefill_forward_text` or `process_output_prefill` to expose per-position logits at all dn positions (currently slices to last token only). Then `_call_prefill_for_verify` becomes 1 forward call instead of dn.

B. **Trace-compatible aux capture** (path C). Move the layer-output capture from a Python wrapper to a tt-metal trace machinery hook (analog of `process_hidden_states_after_prefill_trace`). Re-enables `enable_trace=True` for verify decode. ~3× decode speedup.

C. **Or: combine the v94 spec config (dn=2 num_steps=1) with the smaller Qwen3-1.7B model**, then train an EAGLE-3 draft for it. No public 1.7B EAGLE-3 draft exists. Out of scope for a perf cron loop.

## Honest verdict

Both user goals are reachable but require **multi-hour focused tt-metal kernel engineering**, not a 15-minute cron-loop iteration. Each iteration of the cron has now found a different blocker confirming this conclusion. The plumbing/scaffolding for the right fixes is in place (env-gates default-off so production is unaffected) — what's missing is the kernel work to unblock them.

Hardware free, server stopped after each iteration. All commits pushed.
