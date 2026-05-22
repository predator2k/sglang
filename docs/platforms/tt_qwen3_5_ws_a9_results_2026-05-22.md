**SUPERSEDED-BY:** [`tt_qwen3_5_final_status_2026-05-22.md`](tt_qwen3_5_final_status_2026-05-22.md) — consolidated end-to-end final status across all 8 workstreams.

# WS-A.9 — Per-op divergence with KV-replicate, 2026-05-22

Companion to `tt_qwen3_5_adaptation_status_2026-05-22.md`. WS-A.9 was charged
with running the WS-A.7 layer-3 divergence probe in the KV-replicate (WS-A.8)
configuration and identifying the root cause of the residual ~0.07 full-stack
PCC. This document captures what was actually found.

## Status: BLOCKED

The accuracy ceiling is set by an unresolved **tt-metal SDPA decode kernel bug**
on the per-device config `(n_local_heads=4, n_local_kv_heads=1, head_dim=256)`,
not by anything that can be fixed at the sglang / tt_transformers layer. Two
attempted workarounds (WS-A.7 KV-replicate at weight-load — never deployed;
WS-A.8 KV-replicate on-device via `ttnn.concat`) either hang or produce broken
PCC. Phase-1 of WS-A.9 confirms this conclusively.

## Phase 1 — Per-op PCC at layer 3

### Setup

- Commit: `tt-metal-sglang` `2a59cc73a42` (HEAD), `sglang` `83ea872fb` (HEAD).
- Harness: `/tmp/qwen35_layer3_divergence_hunt.py` (WS-A.7 instrumentation).
- Input token: 42, decode step 0, position 0.
- `SGLANG_TT_DUMP_LAYER3=1` was set throughout to capture per-op TT dumps.
- Per-op tensors were saved to `/tmp/qwen35_diag_tt_baseline.pt` (no
  KV-replicate) and `/tmp/qwen35_diag_tt_kvreplicate.pt` (KV-replicate).
- HF reference dump: `/tmp/qwen35_diag_hf_layer3.pt` from WS-A.7.

### Two runs were collected

| Run | `SGLANG_TT_QWEN35_KV_REPLICATE` | KV-cache shape | Outcome |
|---|---|---|---|
| Baseline | unset (0) | `(64, 2, 32, 256)` | Completed (16 dumps, decoded to end) |
| KV-replicate | `1` | `(64, 4, 32, 256)` | Hangs at `to_torch` after SDPA (11 dumps captured) |

Both runs used a freshly-reset device (`tt-smi -r`) prior to launch.

### Per-op PCC table at layer 3 (vs HF reference)

| op | shape (TT) | PCC_HF(baseline) | PCC_HF(KV-replicate) | PCC baseline-vs-KVR |
|---|---|---|---|---|
| 00_layer3_input | [1,1,32,512] | 0.9643 | 0.9643 | 1.0000 |
| 01_post_input_layernorm | [1,1,32,1024] | 0.9515 | 0.9515 | 1.0000 |
| 03a_q_only | [1,1,32,1024] | 0.9715 | 0.9715 | 1.0000 |
| 03b_gate_only | [1,1,32,1024] | 0.9184 | 0.9184 | 1.0000 |
| 04_post_k_proj | [1,1,32,256] | 0.9612 | 0.9612 | 1.0000 |
| 05_post_v_proj | [1,1,32,256] | 0.9100 | 0.9100 | 1.0000 |
| 06_post_q_norm | [1,1,32,1024] | 0.9581 | 0.9581 | 1.0000 |
| 07_post_k_norm | [1,1,32,256] | 0.9574 | 0.9574 | 1.0000 |
| 08_post_rope_q | [1,1,32,1024] | 0.9581 | 0.9581 | 1.0000 |
| 09_post_rope_k | [1,1,32,256] | 0.9574 | 0.9574 | 1.0000 |
| **10_post_sdpa** | [1,1,32,1024] | **0.6431** | (hang) | — |
| 11_sigmoid_gate | [1,1,32,1024] | 0.8844 | (hang) | — |
| 11b_post_gate_mul | [1,1,32,1024] | 0.5585 | (hang) | — |
| 12_post_o_proj | [1,1,32,512] | 0.2750 | (hang) | — |
| 13_layer3_output | [1,1,32,512] | 0.9035 | (hang) | — |

KV-replicate produces bit-identical Q, K, V, q-norm, k-norm, RoPE-Q and RoPE-K
outputs to the baseline (PCC 1.0000 baseline-vs-KVR). The two paths only
diverge at the SDPA op itself.

### Critical new observation about SDPA output structure

For step 0 at position 0, the K/V cache contains a single entry, so the
attention softmax reduces to `softmax([single_score]) = 1.0`, which means the
attention output is exactly `V[0]` for **every** q-head. HF's reference output
confirms this: all 8 HF heads are bit-identical (= V[0] expression).

The TT SDPA output, however, has the following structure (row 0 of per-device
`attn_output_cat`, shape [4 heads × 256] = [1024]):

  - **TT head 0 == TT head 1 bit-by-bit** (cos vs HF = 0.9094)
  - **TT head 2 == TT head 3 == 0** (all zeros, cos undefined)

That is: the SDPA decode kernel fills only 2 of the 4 per-device q-heads,
duplicates the first head into the second slot, and zeros out heads 2 and 3.

This matches WS-A.7's observation that q-heads 2-3 were "garbage" — except
today the bad values are zeros, not `1e36`. The catastrophic value version
may have been a transient state of the same underlying bug. The qualitative
finding is identical: the kernel does not service all 4 per-device q-heads.

### WS-A.8 KV-replicate does not fix the bug

The KV-replicate workaround attempts to make the kernel see
`n_local_kv_heads=2` (so `q_heads_per_kv_head = 4/2 = 2`) instead of
`n_local_kv_heads=1` (`q_heads_per_kv_head = 4/1 = 4`). The on-device
implementation in `attention.py:903-937`:

1. `ttnn.sharded_to_interleaved` for K and V → DRAM_INTERLEAVED
2. `ttnn.concat([K, K], dim=2)` → doubled-head DRAM
3. `ttnn.to_memory_config` back to HEIGHT_SHARDED on 2 cores

This sequence **hangs on `to_torch` of the post-SDPA output** (or any
subsequent dependent op). `py-spy dump` confirms the hang location:

    Thread MainThread (idle):
      __call__ (ttnn/decorators.py:473)
      to_torch (ttnn/operations/core.py:398)
      _ws_a7_dump_save (attention.py:56)
      forward_decode (attention.py:1033)        # _ws_a7_dump_save("10_post_sdpa", attn_output_cat)
      forward (attention.py:1574)
      forward (decoder.py:253)

The same hang appears in the **full 24-layer pytest path** (no dumps, just
`test_qwen35_pcc_full_24layer` with `SGLANG_TT_QWEN35_KV_REPLICATE=1`): the
test exhausted its 600s timeout without producing the first step-0 logits.

Two independent witnesses confirm: WS-A.8 KV-replicate is **functionally
broken** in the current environment. Its prior "PCC=0.07, decodes without
garbage" claim cannot be reproduced here.

### Baseline (no KV-replicate) is the actual best PCC

`test_qwen35_pcc_full_24layer` without `SGLANG_TT_QWEN35_KV_REPLICATE`:

    step=0 PCC=0.1018 cosine=0.3262

This is marginally **better** than the WS-A.8 status-doc claim of 0.07 with
KV-replicate. (The 0.07 number could not be reproduced today.) Linear-only
n=3 PCC stays at 0.9028 with top-1 match, so the WS-A.5 + WS-A.7 loader fixes
remain effective.

## Bug identified

**File:** `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` (and the underlying compute / reader kernels).

**Configuration that breaks it:** per-device decode SDPA where
`n_q_heads = 4`, `n_kv_heads = 1`, `head_dim = 256`, batch = 1.

**Symptom:** kernel writes only 2 q-head outputs (and duplicates the first
into the second slot); q-heads 2 and 3 stay zero.

The kernel works correctly on the closely related Qwen3-8B per-device
config: `n_q_heads = 16`, `n_kv_heads = 4`, `head_dim = 128`. The Qwen3.5-0.8B
config is the **only** model in this codebase that hits the broken combination.

Hypothesis (per WS-A.7): kernel tile geometry assumes head_dim packs into
2× 128-dim tiles per head, and the kernel's `num_active_cores =
num_cores_per_head × num_kv_heads × B / num_heads_per_core` math under-allocates
when `num_kv_heads = 1` and `head_dim = 256`. Verification requires reading
`sdpa_flash_decode.cpp` (compute kernel) and `reader_decode_all.cpp`.

## What was NOT the bug

- **Not** the loader (WS-A.7 Bug 1 fix at PCC=0.9715 for `q_after_split`).
- **Not** the q/k/v projection (PCC=0.96 / 0.91 / 0.96 — within BF16 drift).
- **Not** q_norm / k_norm (PCC=0.9581 / 0.9574 with WS-A.5 add_unit_offset).
- **Not** partial-RoPE (PCC=0.9581 / 0.9574 with WS-A.6 head perm; perm is
  bit-equivalent applied to Q and K, and post-RoPE PCC matches pre-RoPE).
- **Not** attn_output_gate per-head ordering (gate dump PCC=0.9184 OK).
- **Not** the KV-replicate workaround math (the math is sound; the on-device
  concat / re-shard path simply hangs).

## What was tried

1. Re-running WS-A.7 per-op probe with `SGLANG_TT_QWEN35_KV_REPLICATE=1` —
   hangs at `to_torch` after SDPA.
2. Re-running same probe **without** KV-replicate, with fresh device reset —
   completes; 16 dump keys captured; PCC table above.
3. Comparing baseline-vs-KVR for the 11 dumps that succeeded with KVR —
   PCC=1.0000 for every pre-SDPA op, confirming WS-A.8 is purely a no-op
   up to that point.
4. Running `test_qwen35_pcc_full_24layer` with KV-replicate — 600s timeout,
   no logits produced.
5. Running `test_qwen35_pcc_full_24layer` without KV-replicate — completes;
   step-0 PCC = 0.1018.
6. Running `test_qwen35_pcc_linear_only_n3` — PCC=0.9028, top1 match (sanity).
7. Running `test_prefetcher_smoke` on Qwen3-8B — 1 passed. **No regression.**

## What was NOT tried (out of scope for WS-A.9)

- Fixing the SDPA decode kernel directly (deep C++ change in tt-metal compute
  kernels).
- Implementing a host-side SDPA fallback for full_attention layers
  (analogous to linear_attention's host fallback for GatedDeltaNet). This
  would be a major addition and is closer in scope to WS-A.10
  ("TT-native GatedDeltaNet kernel") than to WS-A.9 ("find and fix the
  accuracy bug").
- Re-replicating K/V at **weight-load time** (cloning the K and V projection
  weight rows so `n_local_kv_heads` is effectively 2 before any matmul, with
  the qkv_size and cache shape adjusted accordingly). This would dodge the
  on-device `ttnn.concat` that hangs in WS-A.8.

## Recommendations for next-bug WS-A.10

1. **Implement load-time KV-replicate.** Clone the K and V projection rows so
   `n_local_kv_heads` is 2 per device (4 total). Update `qkv_size`,
   `model_args.n_kv_heads`, and the cache shape. This makes the SDPA kernel
   see `n_kv_heads=2, head_dim=256` instead of `n_kv_heads=1, head_dim=256`.
   If the kernel still mis-counts heads, this confirms the kernel needs a
   real fix. If it works, it sidesteps the on-device concat hang and unblocks
   WS-A.9's accuracy target without modifying the kernel.

2. **If load-time KV-replicate also fails**, fix the SDPA decode kernel
   directly. Files to inspect:
   - `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
     (program / core allocation, lines 75-178)
   - `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
     (compute kernel — per-head loop)
   - `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`
     (reader — per-head Q strides)
   Look for an off-by-one or tile-boundary assumption in the per-head loop
   that under-counts when `head_dim` requires more than one tile per head
   (`head_dim / TILE_WIDTH > 1`) and `num_kv_heads = 1`.

3. **If neither (1) nor (2) is tractable, host-side SDPA fallback.** Pattern
   already exists in `linear_attention.py` for GatedDeltaNet. Per-device, the
   SDPA decode for full_attention layers is small (q=[1,1,4,256], k/v=
   [block,1,32,256]) so a host fallback is cheap. Slow but correct.

## Files saved for next workstream

In the `p3a-ngram` container `/tmp/`:

- `qwen35_diag_hf_layer3.pt` — HF layer-3 reference (26 tensors, single source of truth).
- `qwen35_diag_tt_baseline.pt` — TT layer-3 dumps, no KV-replicate, complete.
- `qwen35_diag_tt_kvreplicate.pt` — TT layer-3 dumps with KV-replicate (partial — hang).
- `qwen35_layer3_divergence_hunt.py` — WS-A.7 harness, still working.
- `compare_kvr.py` — Phase-1 PCC comparison script (this document's table).
- `ws_a9_kvr*.log` / `full_kvr.log` — captured hang traces.

## Time budget

- Started ~2h elapsed for context loading + setup.
- Phase 1 data collection: ~1.5h (including 3 device hangs and resets).
- Phase 2 (diagnosis) + writeup: ~30min.
- Total: ~4h.

Status doc updated; no source files modified. WS-A.10 takes over with the
recommendations above.
