# EAGLE-3 TPOT/TTFT Optimization — Measured State and Concrete Paths

## Measured (v114, 2026-05-15)

- TTFT ~80ms (short 5-6 token prompts; Qwen3-8B prefill bucket=128 padded)
- TPOT 104–135 ms/tok (= 7.4–9.6 tok/s end-to-end)
- TPOT tracks inversely with `spec_accept_length`: `TPOT ≈ per_cycle_ms / accept_length`
- Per-verify-cycle wall time ~150–170 ms

## Why TPOT is this high

`models/tt_llm.py:_call_prefill_for_verify` runs `draft_token_num` (=2)
sequential `decode_forward` calls per verify cycle, each with `enable_trace=False`,
and each layer-output capture issues a **host-blocking** read via
`ttnn.get_device_tensors(out)` + `to_torch(shard)` × tp_shards.

Per cycle, for dn=2 and 3 aux layers and TP=2:

| Cost line                          | Approx ms |
|---|---|
| 2× `decode_forward` (no trace)     | 130–150   |
| 2× 3 layers × 2 shards `to_torch`  | 15–30     |
| Tree-mask flag toggle + bookkeeping| <2        |

The `enable_trace=False` is mandatory today: the layer-capture wrapper is
Python code calling `to_torch` *inside* the layer chain. Under trace
replay, Python wrappers don't re-fire — they're called only at trace
capture — so subsequent replays would read stale aux. The capture
pattern is fundamentally incompatible with the current tt_transformers
trace mechanism.

## Concrete optimization paths (in order of leverage)

### A. Replace multi-decode with single prefill_forward (HIGHEST IMPACT)

`prefill_forward` processes dn tokens in one ttnn call. Layer output
wrapping would naturally produce a `[bs, dn, hidden]` capture per layer
instead of `[bs, 1, hidden]` per decode call. One ttnn call instead of
two ~halves the dominant compute cost.

Open questions to resolve before implementing:
- `prefill_forward` currently returns only the last position's logits
  (squeezed). The verify path needs per-position logits for all dn
  positions. Either expose a `return_all_logits` flag, or capture the
  pre-lm-head hidden at all dn positions and run lm_head host-side
  (cheap: dn × `[hidden] @ [hidden, vocab]` matmul).
- KV cache writes happen for positions [seq_lens, seq_lens+dn-1] —
  identical to current multi-decode, so cache management is unchanged.
- Aux capture wrapper would need to handle the `[bs, dn, hidden]` shape
  in `_read_captured_aux_concat_host` (currently assumes `[bs, hidden]`).

Estimated win: TPOT ~104ms → ~55–70ms (15–20 tok/s).

### B. Deferred batched aux read with ttnn.clone (MODERATE IMPACT)

Current wrapper reads each layer-output to host inline, blocking the
next layer. Replace inline `to_torch` with `ttnn.clone(out)` (device-to-
device, fast), and read all 3 cloned tensors at the end of
`decode_forward`. Eliminates the inline serialization.

Risk: the cloned device tensors must outlive `decode_forward`. ttnn's
ref-counted allocator should preserve them as long as Python holds a
reference, but worth probing.

Estimated win: TPOT ~104ms → ~85–95ms (10.5–11.7 tok/s).

### C. Trace-compatible aux capture in tt-metal trace machinery (LARGE)

Analog of `process_hidden_states_after_prefill_trace`: thread a
persistent host-readable output buffer through the trace setup so each
trace replay populates it. Re-enables `enable_trace=True` for the
verify decode and combines naturally with path A or current multi-
decode. This is multi-file tt-metal kernel work.

Estimated win: `enable_trace=True` brings per-decode wall time from
~70ms toward ~25–35ms (per v74-baseline reference). Combined with A,
TPOT ~104ms → ~30–40ms (25–33 tok/s).

### D. TTFT — short prompts paying for 128-token prefill bucket

Prefill bucket padding for prompts < 128 tokens. Currently 80ms TTFT
absorbs the full bucket cost. Adding a smaller bucket (e.g. 32) to
`tt_transformers`' prefill warmup set would let short prompts hit a
matching bucket; the saved ms would be visible only for short prompts
but the user-perceived TTFT improvement is real for chat-style
single-turn probes.

Estimated win: short-prompt TTFT 80ms → ~35–50ms.

## Not attempted in this iteration

All four paths require either substantial verify-path restructuring
(A, B) or tt-metal C++ work (C, D). Each is genuinely risky against
the working v111-v113 state — a regression would silently break
acceptance. Proper attempt requires a focused session with isolated
testing per path, not a cron-loop iteration where each firing has
~15 min before the next.

Baseline to beat: 8.24 tok/s avg across v95 8-prompt suite,
accept_rate 0.18–0.50, factually correct on 7/8 prompts.
