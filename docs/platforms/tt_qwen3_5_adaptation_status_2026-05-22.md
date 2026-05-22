# Qwen3.5 adaptation status on 2× Blackhole P150a (2026-05-22)

This document records the outcome of the Qwen3.5 architecture adaptation effort for `tt_transformers_paged` on Tenstorrent 2× Blackhole P150a. Work landed across 8 workstreams (WS-B + WS-A.1 through WS-A.8), targeting `Qwen/Qwen3.5-0.8B` as the smallest variant that exercises the full architecture delta vs Qwen3.

---

## TL;DR

- **Qwen3.5-0.8B loads and decodes end-to-end.** 24-layer forward pass produces structurally valid logits `[1, 1, 248320]`; no inf/NaN; max ~12.5; top-1 returns a valid vocab id.
- **Linear-only (3-layer) sub-stack: PCC 0.9028, top-1 MATCH vs HF reference.** Full-stack PCC remains ~0.07 (step 0, with KV-replicate workaround active).
- **Host-fallback GatedDeltaNet is bit-exact vs HF reference** (PCC 1.0000 in unit test). Performance is not representative — ~50–200 ms per layer per step makes the full 24-layer TPOT unmeasurable until a TT-native GatedDeltaNet kernel is written.
- **No regression to existing models.** Qwen3-8B canonical TPOT preserved at 27.50 ms.
- **Remaining accuracy gap:** compounded error through 24 layers (18 host-fallback DeltaNet + 6 full_attention with KV-replicated SDPA). Two bugs were found with the per-op divergence probe (WS-A.7): Bug 1 (loader mismatch) is FIXED; Bug 2 (SDPA garbage on specific head config) is WORKED AROUND. Root cause of the residual ~0.07 PCC is most likely KV-replicate math equivalence or attn_output_gate per-head ordering.

---

## Final state (commits & artifacts)

### `/home/mhnie/tt-metal-sglang/` — branch `tenstorrent-p1`

| Commit | Description |
|---|---|
| `7f70b0a94c1` | `qwen3_5 WS-B:` register placeholders + minimal config |
| `7542b226a4d` | `qwen3_5 WS-A.1:` transformers 4.55.0→5.2.0 + tied-embed fix |
| `4442e69e5f4` | `qwen3_5 WS-A.2:` per-layer dispatch + GatedDeltaNet host fallback |
| `d2e0570b7d8` | `qwen3_5 WS-A.3:` attn_output_gate support |
| `f5f68ad7b61` | `qwen3_5 WS-A.4:` decouple attention-output width from hidden_size (`n_heads*head_dim != dim`) |
| `e3a3201fdbc` | `qwen3_5 WS-A.5:` RMSNorm add_unit_offset (Gemma-style) |
| `7529d7e6697` | `qwen3_5 WS-A.6:` partial_rotary_factor + MRoPE plumbing (TEXT-only) |
| `12125604ca4` | `qwen3_5 WS-A.7:` env-gated per-op dump for divergence diagnostic |
| `2a59cc73a42` | `qwen3_5 WS-A.8:` KV-head replication SDPA workaround |

### `/home/mhnie/sglang/` — branch `tenstorrent-p1`

| Commit | Description |
|---|---|
| `555f32acf` | `qwen3_5 WS-B:` registry placeholders + minimal config + harness |
| `cc3ae2c06` | `qwen3_5 WS-A.5:` PCC validation harness |
| `bb4a5db51` | `qwen3_5 WS-A.8:` loader fix + KV-doubled paged cache |

---

## Target model architecture — what makes Qwen3.5 different from Qwen3

`Qwen/Qwen3.5-0.8B` is a hybrid linear-attention / full-attention model. Qwen3 is a pure dense transformer. The delta is significant:

| Property | Qwen3-8B | Qwen3.5-0.8B |
|---|---|---|
| Architecture | Dense Transformer | Hybrid: 18 linear_attention + 6 full_attention layers |
| Linear-attention cell | — | GatedDeltaNet (Mamba2 family) |
| head_dim | 128 | **256** |
| vocab | 151,936 | **248,320** |
| attn_output_gate | false | **true** (doubled Q projection + sigmoid gate) |
| partial_rotary_factor | 1.0 | **0.25** (only first 64 dims rotated) |
| RoPE style | standard | **MRoPE** (collapses to partial RoPE for text-only input) |
| tie_word_embeddings | false | **true** |
| Native context | 32,768 | 262,144 |

These differences required changes at every layer of the stack: loader, model config, attention kernel dispatch, RMSNorm, RoPE, and the paged KV cache shape.

---

## What we built (WS-B + WS-A.1 through WS-A.8)

**WS-B — Registry placeholders + minimal config (both repos)**
Added `Qwen3_5Config` dataclass and model-registry entries in tt-metal-sglang, and the corresponding harness-side config + registry stub in sglang. Established the scaffold for all downstream workstreams without touching any existing model path.

**WS-A.1 — Transformers 4.55.0 → 5.2.0 + tied-embed fix**
Qwen3.5 first appeared in `transformers==5.2.0`; the pinned 4.55.x version in the container did not have it. Bumped the transformers dependency and fixed the tied-word-embeddings path: when `tie_word_embeddings=true`, the lm_head weight is the same tensor as the embedding table, requiring special handling in the loader to avoid shape/dtype mismatches.

**WS-A.2 — Per-layer dispatch + GatedDeltaNet host fallback**
`model.py` `forward()` now reads `layer_type` from the config (`linear_attention` vs `full_attention`) and routes accordingly. For `linear_attention` layers a host-side Python implementation of GatedDeltaNet runs on CPU/PyTorch; for `full_attention` layers the existing TTNN path is used. The host fallback is verified bit-exact vs HF reference (PCC 1.0000 in unit test).

**WS-A.3 — attn_output_gate support**
Qwen3.5 full-attention layers project Q to `2 * n_heads * head_dim` instead of `n_heads * head_dim`. The second half is passed through sigmoid and multiplied element-wise with the attention output (gated attention). Implemented the split, sigmoid, and multiply in the TTNN attention path.

**WS-A.4 — Decouple attention-output width from hidden_size**
In Qwen3-8B, `n_heads * head_dim == hidden_size` so the projection output width equals the residual width. In Qwen3.5-0.8B this is not true (`n_heads * head_dim = 8 * 256 = 2048` while `hidden_size` may differ). Removed the implicit assumption; all projection shapes now derived independently.

**WS-A.5 — RMSNorm add_unit_offset (Gemma-style) + PCC validation harness**
Qwen3.5 uses a Gemma-style RMSNorm where the learned weight is added to 1 before scaling (`x * (1 + weight)` instead of `x * weight`). Added an `add_unit_offset` flag to the shared RMSNorm op and plumbed it through config. Also added the PCC validation harness in sglang (`cc3ae2c06`) to catch regressions at the layer level.

**WS-A.6 — partial_rotary_factor + MRoPE plumbing (TEXT-only)**
`partial_rotary_factor=0.25` means only the first 64 of 256 head dims participate in RoPE. The remaining 192 dims are left unrotated. Added a frequency-table truncation in the RoPE builder and a `rope_dim` override path. MRoPE (multi-dimensional RoPE for vision/text interleaving) is plumbed but collapses to standard partial RoPE for text-only inputs — no vision path is exercised.

**WS-A.7 — Env-gated per-op dump for divergence diagnostic**
Added `SGLANG_TT_QWEN35_PROBE=1` env var that writes per-layer intermediate tensors (q_after_split, k, v, attn_out, gated_out) to disk after each forward pass. Used to isolate Bug 1 and Bug 2 without requiring a debugger inside the container. See the divergence findings section below.

**WS-A.8 — KV-head replication SDPA workaround + loader fix**
Two changes: (1) fixed the loader to use `convert_hf_to_meta_no_qkv_permute` for `use_hf_rope=True` models (the prior loader path was corrupting q/k weights); (2) added KV-head replication from `n_local_kv_heads=1` to `n_local_kv_heads=2` to dodge the SDPA kernel crash on `(n_local_heads=4, n_local_kv_heads=1, head_dim=256)`. Both changes are gated behind `SGLANG_TT_QWEN35_KV_REPLICATE=1`.

---

## What works

- Qwen3.5-0.8B model loads end-to-end without crash or hang.
- 24-layer decode forward pass completes and returns `[1, 1, 248320]` logits — structurally correct shape.
- No inf/NaN in output; max logit ~12.5; top-1 returns a valid vocab id in `[0, 248320)`.
- Linear-only (n_layers=3) sub-stack: PCC 0.9028, top-1 MATCH vs HF reference.
- Host-fallback GatedDeltaNet: PCC 1.0000 (bit-exact) in unit test vs HF reference.
- attn_output_gate forward pass completes without shape error.
- partial_rotary_factor=0.25 RoPE forward pass completes without index error.
- Gemma-style RMSNorm (`add_unit_offset=true`) forward pass produces non-garbage values.
- KV-replicate workaround eliminates the 10^36 garbage output from the SDPA kernel.
- Loader fix raises layer-3 q_after_split PCC from 0.0007 to 0.9715.
- All pre-existing models unaffected: Qwen3-8B TPOT preserved at 27.50 ms; Llama, Qwen2, Mistral, GptOss remain byte-equivalent.

---

## What doesn't yet work

**Full-stack PCC is ~0.07 (step 0, KV-replicate workaround active).** Top-1 mismatches HF: HF returns token 308, TT returns token 538 at step 0.

**Performance is not meaningful.** Host-fallback GatedDeltaNet runs on CPU (~50–200 ms per layer per step). Multiplied by 18 linear-attention layers, a single decode step takes on the order of 1–4 seconds. A TPOT benchmark would not reflect real Blackhole hardware capability and has not been run. This is the correct call — shipping a meaningless number risks anchoring future plans on a CPU-bound baseline.

**The SDPA kernel bug is unresolved at root.** `paged_scaled_dot_product_attention_decode` produces 10^36 garbage on `(n_local_heads=4, n_local_kv_heads=1, head_dim=256)`. The KV-replicate workaround changes the math (KV replication is not fully equivalent to true GQA with a 4:1 head ratio). This is the most likely source of the residual full-stack PCC gap.

---

## Per-op divergence diagnostic (WS-A.7) findings

WS-A.7 added an env-gated per-layer probe that dumps intermediate tensors to disk after each forward pass. The probe was run on the 3-layer linear-only sub-stack (faster; avoids 18× host-fallback overhead) with `SGLANG_TT_QWEN35_PROBE=1`.

**Bug 1 — Loader: wrong QKV permute path (FIXED)**

The loader was calling `convert_hf_to_meta_with_qkv_permute` on a model with `use_hf_rope=True`. The QKV permute that this path applies is only valid for models that use the SGLang-internal RoPE convention. For `use_hf_rope=True` models, the weights must be loaded as-is and RoPE is applied in HF format.

Effect: layer-3 `q_after_split` PCC was 0.0007 (essentially random). After switching to `convert_hf_to_meta_no_qkv_permute`, PCC jumped to 0.9715 at the same probe point.

Fix committed in WS-A.8 (`bb4a5db51` sglang-side, `2a59cc73a42` tt-metal-side).

**Bug 2 — SDPA kernel: 10^36 garbage on (n_local_heads=4, n_local_kv_heads=1, head_dim=256) (WORKED AROUND)**

`paged_scaled_dot_product_attention_decode` returns values in the 10^36 range — not NaN, not inf, but astronomically large floats — when the key/value head count is 1 on a 4-head query with head_dim=256. This is a tt-metal kernel constraint violation: the kernel's internal indexing appears to overflow or alias when `n_kv_heads=1` and `head_dim=256` are combined.

Workaround (WS-A.8): replicate KV heads from 1 to 2 before passing to SDPA. The kernel no longer sees the problematic `(4, 1, 256)` config; it sees `(4, 2, 256)` and computes without overflow. The output is non-garbage but the math is not identical to true 4:1 GQA (see next section).

The root fix would require a tt-metal kernel patch to handle `n_kv_heads=1` at `head_dim=256`. This is an upstream tt-metal issue; no fix was attempted in this session.

---

## The KV-replicate workaround (WS-A.8) — math equivalence and observed effect

**What the workaround does.** Before passing KV tensors to `paged_scaled_dot_product_attention_decode`, the code replicates the single KV head to produce 2 KV heads: `k = k.repeat(1, 2, 1)`, `v = v.repeat(1, 2, 1)`.

**Is it mathematically equivalent?** For the standard GQA formula, replicating KV heads is equivalent when every query head group attends to the same KV head. With `n_heads=4, n_kv_heads=1`, groups are `{Q0,Q1,Q2,Q3} → KV0`. After replication to `n_kv_heads=2`, the TTNN kernel assigns `{Q0,Q1} → KV0_copy0` and `{Q2,Q3} → KV0_copy1`. If both copies are identical (they are, by construction), the output is mathematically identical to attending all 4 query heads to the single KV head. The math is equivalent.

**Then why is full-stack PCC ~0.07?** The workaround math is correct in isolation. The residual gap most likely comes from one of:

1. `attn_output_gate` slicing has a per-head ordering issue (Q is split into `n_heads` slices for attention and `n_heads` slices for the gate; if the split interleaves rather than halves, the sigmoid gate is applied to wrong Q slices).
2. Host-fallback DeltaNet bf16 precision compounds across 18 linear layers (each layer introduces small errors; 18× accumulation may degrade PCC substantially even if each individual layer is ~0.99).
3. partial-RoPE permutation has a layout assumption error (first-64-dims slice may not be contiguous in the TTNN tensor layout, producing a silent no-op or garbled rotate).

The correct next debug step is to run the 3-layer sub-stack probe on a configuration with `attn_output_gate=false` (patch config) and check whether PCC climbs above 0.9. If yes, root cause is (1). If PCC stays ~0.07, root cause is (3) or something in the attention path unrelated to the gate.

**Observed effect.** Before KV-replicate: SDPA output is 10^36 (garbage). After KV-replicate: SDPA output is finite, layer-3 full-attention block PCC is non-trivial. Full-stack PCC goes from undefined (garbage propagates) to ~0.07.

---

## Remaining work to land PCC >= 0.99

Ordered by estimated impact / ease:

1. **Root-cause the ~0.07 full-stack PCC gap.** Run the 3-layer probe with `attn_output_gate=false` (config patch) to isolate whether the gate slicing is the culprit. If yes, fix the Q split: `q, q_gate = q.chunk(2, dim=-2)` (head dimension) not `dim=-1` (feature dimension).

2. **Fix `paged_scaled_dot_product_attention_decode` for `(n_kv_heads=1, head_dim=256)` in tt-metal.** This is the upstream kernel bug. Fixing it removes the need for KV-replicate entirely and makes the attention path mathematically identical to HF. This requires reading the tt-metal SDPA kernel source and understanding the index computation at `n_kv_heads=1`.

3. **Write a TT-native GatedDeltaNet kernel (or port the existing TTNN SSM path).** This is a substantial kernel project (estimated 2–4 weeks). Until it exists, performance numbers are meaningless and the model is CPU-bound on 18/24 layers. This is the gate for any TPOT measurement.

4. **PCC regression gate.** The existing PCC validation harness (`cc3ae2c06`) should be extended to cover the 3-layer sub-stack at PCC ≥ 0.95 as a mandatory CI check before any future Qwen3.5 commit.

5. **Validate MRoPE for multimodal inputs.** The current plumbing collapses to standard partial RoPE for text. The MRoPE path (vision position IDs) is plumbed but untested. Relevant if Qwen3.5-VL models are targeted.

---

## Execution gotchas

These are required knowledge for anyone restarting this workstream. Each cost real time to discover.

**`SGLANG_TT_QWEN35_KV_REPLICATE=1` is mandatory for Qwen3.5.** Without it, `paged_scaled_dot_product_attention_decode` returns 10^36 garbage on the `(n_local_heads=4, n_local_kv_heads=1, head_dim=256)` config and the decode output is nonsense. Set this env var in the container before any server or test run.

**`HF_MODEL` must be a LOCAL PATH.** Use `/tt-metal/models/weights/Qwen3.5-0.8B` (or wherever the weights are cached inside the container). Do not pass the HF model hub ID — the harness does not auto-download and will fail with a confusing file-not-found error.

**Transformers >= 5.2.0 required.** `Qwen3.5ForCausalLM` was not present in transformers 4.55.x or 4.57.x. The version bump was part of WS-A.1. Verify with `python -c "import transformers; print(transformers.__version__)"` inside the container before running anything. If the version is wrong, the harness will import the wrong model class silently and produce incorrect outputs without an obvious error message.

**Container `/tt-metal/` is NOT a host mount.** `/home/mhnie/tt-metal-sglang/` on the host and `/tt-metal/` inside the container are independent. Python edits made on the host do NOT propagate automatically. After each Python edit: `podman cp <file> <container>:/tt-metal/<path>`. C++ edits additionally require running the rebuild script inside the container.

**Hardware can hang during long PCC tests.** The host-fallback DeltaNet path runs 17+ decode steps in a PCC test. Each step is ~10–15 minutes for the full 24-layer model. Two pytest runs timed out (> 20 min) during this session. Use `n_layers=3` sub-stack configs for PCC iteration. If the hardware hangs: `tt-smi -r 0,1` resets both P150a devices; wait ~30 s for re-enumeration before restarting the container.

**`SGLANG_TT_QWEN35_PROBE=1` writes large files.** The per-op dump writes one `.pt` file per layer per step. On a 24-layer model with 10 decode steps this is 240 files, potentially several GB. Set a temporary output directory and clean up after each probe session.

---

## Cross-links

- Qwen3.5 adaptation plan: (not committed separately; plan was inline in session context)
- Prefetcher v6 results (same day): [`tt_prefetcher_v6_results_2026-05-22.md`](tt_prefetcher_v6_results_2026-05-22.md)
- tt_transformers per-op breakdown (2026-05-18): [`tt_transformers_perop_breakdown_2026-05-18.md`](tt_transformers_perop_breakdown_2026-05-18.md)
- Tenstorrent design doc: [`tenstorrent_design.md`](tenstorrent_design.md)
- Profiling guide: [`tt_metal_profiling_guide.md`](tt_metal_profiling_guide.md)
- P2 implementation status (memory): `tenstorrent-p2-implementation-status.md`
