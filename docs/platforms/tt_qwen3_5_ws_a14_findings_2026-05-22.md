**SUPERSEDED-BY:** [`tt_qwen3_5_final_status_2026-05-22.md`](tt_qwen3_5_final_status_2026-05-22.md) — consolidated end-to-end final status across all 8 workstreams.

# Qwen3.5 WS-A.14 — SDPA + post-SDPA precision floor confirmed (2026-05-22)

This document records the WS-A.14 probe sweep, the per-op attribution at
layer 3, and the conclusion that the remaining per-op PCC gap at SDPA and
post-`o_proj` is **structural** for the current default precision
configuration.

WS-A.14 follows directly from WS-A.13 (`1c1ca9e2bb8`), which measured the
Wo precision lift (BFP8 → BF16, +HIFI4) and recorded only +0.005..0.018
end-to-end gain — below the +0.05 commit threshold.

The work is purely diagnostic: env-gated probes were added to
`attention.py` so future operators can re-run the same sweep without
modifying production code. No new behavior; default-off, all paths
zero-cost when env vars are unset.

---

## Reference baseline (reproduced)

Per-op layer-3 PCC at step 0 (Qwen3.5-0.8B, n_layers=4, 2× P150a mesh,
load-time KV-replicate on):

| op | PCC |
|---|---|
| 04 post_k_proj | 0.5058 |
| 05 post_v_proj | 0.9100 |
| 06 post_q_norm | 0.9581 |
| **10 post_sdpa** | **0.9101** |
| 11b post_gate_mul | 0.9058 |
| **12 post_o_proj** | **0.8472** |
| 13 layer3_output | 0.9260 |

End-to-end PCC (4-layer probe is bit-identical to full 24-layer for the
attention path; uses `_wsa11_full_pcc_report.py` for the full stack):

| step | PCC | cosine | top-1 match |
|---|---|---|---|
| 0 | 0.7183 | 0.6980 | False |
| 1 | 0.6967 | 0.7537 | False |
| 4 | 0.6806 | 0.8459 | False |
| 16 | 0.6527 | 0.7517 | False |

---

## Probes (Phase 1)

All probes are env-gated, default-off, zero behavior change when unset.

| env var | hypothesis | effect at layer 3 |
|---|---|---|
| `SGLANG_TT_QWEN35_WSA14_SDPA_HIFI4=1` | SDPA HiFi4 (H1) | post_sdpa: 0.9101 → 0.9101; post_o_proj: 0.8472 → 0.8464 |
| `SGLANG_TT_QWEN35_WSA14_SDPA_FP32ACC=1` | SDPA fp32 dest acc only (H2) | bit-identical baseline (default already has fp32 dest acc) |
| `SGLANG_TT_QWEN35_WSA14_WO_HIFI4=1` | Wo matmul HiFi4 (BFP8 weights kept) (H3) | post_o_proj: 0.8472 → 0.8468 |
| `SGLANG_TT_QWEN35_WSA14_CCL_DTYPE=bf16` | All-reduce CCL bf16 lift (H3a) | no effect — Qwen3.5 uses the `use_fused_all_gather_matmul` path (no all_reduce) |
| `SGLANG_TT_QWEN35_WSA14_PRINT_MEMCFG=1` | Print attn_out + gate memcfg before multiply (H4) | diagnostic only |
| `SGLANG_TT_QWEN35_WSA14_HEAD_PROBE=1` | Dump SDPA output per-head (H5) | diagnostic only |
| `SGLANG_TT_QWEN35_WO_PRECISION=bf16` (from WS-A.13) | full BF16 attention | post_o_proj: 0.8472 → 0.8432 |
| `SGLANG_TT_QWEN35_WO_PRECISION=wo_only` (from WS-A.13) | Wo BF16 + HIFI4 only | post_o_proj: 0.8472 → 0.8428 |

H6 (attention scale): `self.scale = head_dim**-0.5 = 1/16 = 0.0625`. HF
default with `query_pre_attn_scalar=None` uses the same value. ✅

---

## Attribution (Phase 2 analysis — the surprising part)

### SDPA at step 0 IS V

At step 0, the KV cache holds exactly one token. softmax of a single
attention score is 1, so SDPA output for any q-head attending to its
kv-group equals V[g, 0] exactly.

Measured at row 0, head 0:

|  | TT L2 | HF L2 | norm ratio | PCC |
|---|---|---|---|---|
| `post_v_proj[h0]` | 4.621 | 5.319 | 0.869 | 0.9100 |
| `post_sdpa[h0]` | 4.462 | 5.319 | 0.839 | 0.9101 |

**The `post_sdpa` PCC of 0.9101 matches `post_v_proj` PCC of 0.9100 to 4
decimal places.** The "SDPA error" at layer 3 is _entirely_ the
V-projection error propagating through softmax-of-1 × V. SDPA itself is
operating correctly. No SDPA-side precision knob can fix this — only
lifting WQKV precision (BFP8 → BF16) can move it, and WS-A.13 measured
that gain as +0.005..0.018 e2e (below threshold).

### post_sdpa → post_o_proj loses 0.06 PCC entirely inside the matmul

For Qwen3.5-0.8B (no prefetcher, P300 Linear topology), `attention.py`
takes the `use_fused_all_gather_matmul=True` branch. Subselects to:
`all_gather(dim=3) + linear(wo)`. **No all_reduce** in this path.

Measured chain:

| dump | PCC | notes |
|---|---|---|
| `11b_post_gate_mul` (per-device, 1024-wide) | 0.9058 | row 0 |
| `11d_all_gather_output` (full 2048-wide, BF16) | 0.9057 / 0.9153 | first half vs HF[:1024]; second half vs HF[1024:] |
| `12_post_o_proj` (per-device, 512-wide) | 0.8472 | drop of 0.0585 |

The all_gather is **bit-exact** for row 0 (max abs diff = 0.0, PCC =
1.0). The 0.0585 PCC drop happens entirely inside `ttnn.linear(wo)`.
That matmul is BFP8 weights × BF16 activations with HIFI2 +
fp32_dest_acc_en + L1 acc, reducing across k=2048. The reduction
amplifies the already-noisy activation (PCC 0.9058) into PCC 0.8472.
Lifting weights to BF16 (WS-A.13's `wo_only` mode) actually nudged
layer-3 per-op slightly the wrong way (−0.004), because the layer-3
input precision already dominates the matmul reduction error.

### Head-replica equality (H5) — KV-replicate is mathematically clean

After SDPA, TT q-heads 0..3 on device 0 produce **bit-identical output**
(max abs diff = 0.0 across the 4 heads). KV-replicate is doing exactly
what it claims to do; the SDPA kernel correctly assigns each q-head to
its kv-group; no averaging artifact.

---

## Verdict

**No probe moves layer-3 per-op PCC by ≥ +0.02.**

The layer-3 PCC floor (post_sdpa = 0.91, post_o_proj = 0.85) is set by:

1. **WQKV BFP8 quantization** of V → 0.91 ceiling at SDPA.
2. **Wo BFP8 weights × 2048-wide reduction** amplifying the already-
   noisy activation → 0.85 at post_o_proj.

Both ceilings move only with BF16 lift (already audited in WS-A.13, gain
+0.005..0.018 e2e — below the +0.05 commit threshold).

The end-to-end 0.72 PCC vs HF 1.0 is the compounded effect of ~28
attention layers each operating at this per-layer 0.91 floor, plus the
GatedDeltaNet path.

**Phase 2 (apply wins): no wins to apply.** Per the WS-A.14 plan rule, we
commit only the diagnostic infrastructure (env-gated probes) so future
work can re-evaluate without modifying production code.

---

## What survives

`models/tt_transformers/tt/attention.py` gains:

- `_ws_a14_sdpa_override_kernel_cfg(default_cfg)` — H1/H2 SDPA override.
- `_ws_a14_wo_kernel_cfg(default_cfg)` — H3 Wo HiFi4 override.
- `_ws_a14_print_memcfg_enabled()` — H4 memcfg pre-multiply print.
- `_ws_a14_head_probe_enabled()` — H5 per-head SDPA dump.
- `SGLANG_TT_QWEN35_WSA14_CCL_DTYPE` env var for path-2 all_reduce dtype
  override (no effect on Qwen3.5-0.8B; useful for future Qwen3.5
  configs without fused-AGMM).
- New diagnostic dump `12pre_wo_matmul_out` (path-2 only).

All are default-off; baseline behavior is byte-equivalent.

Regression: `test_prefetcher_smoke_5_decodes` (Qwen3-8B) PASS.

---

## Next attack vectors (post WS-A.14)

1. **WQKV BF16 + Wo BF16 + HIFI4** as a permanent default for Qwen3.5
   (the `bf16` preset in `model_config.py`). Measured e2e gain is small
   (+0.018), and there is a memory + perf cost — but the layer-3 V
   ceiling can only move with this. Worth re-measuring once other
   sources of per-layer noise are eliminated.
2. **Per-layer compounding probe**: dump post_o_proj at layers 3, 9,
   15, 21 to confirm or deny that the 0.85 per-op PCC at layer 3
   compounds linearly down the stack. If yes, the e2e fix is "raise the
   per-layer floor"; if no, there is a single bad layer somewhere
   amplifying.
3. **Tt-metal kernel patch** for the BFP8 matmul with HiFi4 fp32 acc on
   2048-wide reductions. Likely the cleanest fix to lift the Wo matmul
   precision without changing weight storage.
4. **Investigate paged_update_cache write precision**: at later decode
   steps, BFP8 KV cache write+read may compound V noise. Probe by
   loading BF16 KV cache (`SGLANG_TT_QWEN35_WO_PRECISION=bf16` already
   exercises this; per-op delta at step 0 was small but step-N>1 was
   not measured here).
