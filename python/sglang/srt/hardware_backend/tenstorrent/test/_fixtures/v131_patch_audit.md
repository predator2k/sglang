# v131 — Fork Patch Audit for Rebase

**Date:** 2026-05-15
**Base:** `89686ee78d` | **Target:** `e867533fc55` | **Branch:** `tenstorrent-p1`

## Summary

| Category    | Count | Description                                      |
|-------------|-------|--------------------------------------------------|
| CHERRY-PICK | 7     | Direct cherry-pick onto rebased target            |
| ADAPT       | 3     | Needs manual adaptation due to upstream API drift |
| DROP        | 8     | P3a.2 single-chip workarounds superseded by 2-chip topology or upstream fixes |
| SKIP        | 4     | Prefetcher experiments (not needed for P3b BFP4 goal) |

## Patch-by-Patch Classification

### CHERRY-PICK (apply directly)

| # | SHA        | Message                                                        | Files                    | Rationale                                            |
|---|------------|----------------------------------------------------------------|--------------------------|------------------------------------------------------|
| 1 | `6f5817e7` | compat(common): soft-import AutoModelForVision2Seq             | llama_models.py          | e867533 still has hard import; our soft-import needed |
| 2 | `a0b16d1a` | fix(generator_sglang): decode_forward_text doesn't exist       | generator_sglang.py      | e867533 still calls decode_forward_text; fix needed   |
| 3 | `8cc0c7ac` | _skip_self_attention flag for EAGLE tree-mask emulation        | attention.py             | Not in upstream; EAGLE feature needed                 |
| 4 | `ba0d960e` | use_prefetcher through initialize_sglang_model + Qwen3 configs | generator_sglang.py, prefetcher.py | Not in upstream; SGLang integration needed    |
| 5 | `1bb2f6b4` | Qwen3-1.7B/8B MAX_PREFILL_CHUNK_SIZE entries                  | model_config.py          | Entries not in e867533; P300 key for our hardware     |
| 6 | `4e0ae491` | Qwen3-8B balanced mode — BFP4 MLP + HIFI2 fidelity           | model_config.py          | Core BFP4 precision config for Qwen3-8B              |
| 7 | `bbf9607a` | 6 experimental Qwen3-8B precision modes for BFP4 quality sweep | model_config.py          | LOFI mode needed for max throughput                   |

### ADAPT (manual rework needed)

| # | SHA        | Message                                                        | Files                    | Rationale                                            |
|---|------------|----------------------------------------------------------------|--------------------------|------------------------------------------------------|
| 8 | `477facd2` | rope_theta fallback for Llama-3 and Qwen3 on transformers 5.x | model_config.py          | Same area changed upstream; may need context adjustment |
| 9 | `6186fa28` | add Qwen3-8B to high-precision exception list                 | model_config.py          | The exception list context may differ; check upstream list |
| 10| `cf05cc87` | EAGLE-3 enablement for Blackhole P300_X2 (MUX dispatch)       | distributed_norm.py, model_config.py, device.cpp | device.cpp API changed; distributed_norm may differ |

### DROP (superseded or not needed for 2xP150a)

| # | SHA        | Message                                                        | Files                    | Rationale                                            |
|---|------------|----------------------------------------------------------------|--------------------------|------------------------------------------------------|
| 11| `09eab40b` | P300 / Qwen3-8B grid + LM head workarounds                   | model_config.py, device.cpp | Single-chip P300 workaround; 2xP150a uses MUX path  |
| 12| `ea7b7f30` | strip GQA padding from Q/K/V before norm + SDPA               | attention.py             | Single-chip Qwen3 on P300; not hit in TP=2 path     |
| 13| `8cc24885` | conditional grid-y clamp + DECODE LM head fix                 | model_config.py, device.cpp | Grid clamp for single P300; superseded by patch 10 ADAPT |
| 14| `6843c6c6` | DECODE-mode unsharded fallbacks on single-chip                | attention.py, distributed_norm.py, model_config.py | Single-chip only; our TP=2 path doesn't hit these |
| 15| `1ec8126e` | MLP + QKV DECODE single-chip DRAM mem-configs                 | model_config.py          | Single-chip P300 mem config; not used in TP=2        |
| 16| `07cd4a84` | clamp QKV prefill grid to (8,8) for Blackhole MUX dispatch   | model_config.py          | Subsumed into patch 10 ADAPT (EAGLE-3 enablement)    |
| 17| `4415b573` | fused-AG for num_devices>=2                                    | model_config.py          | May already be upstream; check during ADAPT phase    |
| 18| `a843391c` | prefer larger ring size to reduce L1 per receiver core        | prefetcher.py            | Prefetcher tuning; not needed for P3b baseline       |

### SKIP (prefetcher experimental, defer to later)

| # | SHA        | Message                                                        | Files                    | Rationale                                            |
|---|------------|----------------------------------------------------------------|--------------------------|------------------------------------------------------|
| 19| `4acf2659` | MUX-safe sender/receiver mapping                              | prefetcher.py            | Prefetcher MUX experiment; defer                     |
| 20| `a3246ce4` | detect MUX via is_blackhole(), not grid.y                     | prefetcher.py            | Prefetcher MUX experiment; defer                     |
| 21| `81ad4791` | 3-sub-device layout isolates receivers from workers           | prefetcher.py            | Prefetcher sub-device experiment; defer              |
| 22| `8afd0ecb` | CoreRange uses .start/.end not .start_coord/.end_coord        | prefetcher.py            | Prefetcher API compat fix; defer                     |

## Cherry-pick Order

Apply in dependency order on top of `e867533`:

1. `6f5817e7` — soft-import (no deps)
2. `477facd2` — rope_theta (ADAPT, model_config.py)
3. `6186fa28` — high-precision exception (ADAPT, model_config.py — depends on 2)
4. `a0b16d1a` — decode_forward fix (generator_sglang.py)
5. `ba0d960e` — use_prefetcher plumbing (generator_sglang.py — depends on 4)
6. `1bb2f6b4` — MAX_PREFILL_CHUNK_SIZE entries (model_config.py)
7. `4e0ae491` — Qwen3-8B balanced mode (model_config.py — depends on 3)
8. `bbf9607a` — 6 precision modes (model_config.py — depends on 7)
9. `8cc0c7ac` — _skip_self_attention (attention.py)
10. `cf05cc87` — EAGLE-3 enablement (ADAPT — device.cpp API changed)

## Notes

- The `device.cpp` grid clamp (patch 10 ADAPT) is the most complex adaptation. The target commit changed `compute_with_storage_grid_size()` to use `MetalContext::instance()` directly instead of `MetalEnvAccessor`. The clamp logic itself is identical; only the surrounding code differs.
- The existing `rebase-e867533` branch already has patches 6 (MAX_PREFILL) applied. We will create a new `rebase-p3b` branch from scratch for cleanliness.
- Prefetcher patches (19-22) can be re-applied post-P3b if prefetcher brings measurable throughput gains on the rebased metal.
