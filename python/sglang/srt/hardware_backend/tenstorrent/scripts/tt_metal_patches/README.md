> **STATUS 2026-05-15**: These patches have been migrated to a proper
> tt-metal fork at `predator2k/tt-metal` on branch `tenstorrent-p1` (base
> commit `89686ee78d`). The fork has additional commits beyond patches
> 01–03: the `_skip_self_attention` flag (sketched in patch 04) is now a
> real commit and is **actively used by EAGLE-3 verify** in the no-trace
> path (`_call_prefill_for_verify`, `enable_trace=False`), which sidesteps
> patch 04's original trace-capture blocker. Additional Blackhole-multichip
> patches (DRAM-mem-config fallbacks for QKV/MLP, `_force_unsharded` for
> RMSNorm, FabricTensixConfig-gated grid-y clamp) also live on the fork
> branch — see memory `tenstorrent-tt-metal-fork.md` for the index.
>
> The patches in this directory remain for backwards compatibility (still
> applied via the apply-loop below for now), but new work should target
> the fork.

# tt-metal patches for P2a.1 plugin-absorbed path

These patches are applied to the **tt-metal source inside the running container**
(`/tt-metal/...`) before launching the SGLang server with
`SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged`. They unblock issues found
during P2a.1 §9.1 smoke validation on 2× Tenstorrent Blackhole p150a:

| # | File | Fix | Why |
|---|---|---|---|
| 01 | `models/common/llama_models.py` | Soft-import `AutoModelForVision2Seq` etc. | Removed in `transformers` 5.x; not needed for text-only Llama path |
| 02 | `models/tt_transformers/tt/model_config.py` | Soft-import + `rope_theta` fallback for Llama-3.x + nested-dict handling for Qwen3 `rope_parameters` | (a) `transformers` 5.x `LlamaConfig.to_dict()` drops `rope_theta` from top-level; Llama-3 standard value is 500000.0. (b) `Qwen3ForCausalLM` stores `rope_theta` inside a nested `rope_parameters` dict — `text_config.get("rope_theta")` returns `None`; must read `text_config["rope_parameters"]["rope_theta"]` instead. Discovered in P2b T4.2 Qwen3-8B smoke. |
| 03 | `models/tt_transformers/tt/generator_sglang.py` | Fix `super().decode_forward_text()` → `super().decode_forward()` (4 sites) | `decode_forward_text` does not exist on `Generator`; the correct text-decode entry point is `decode_forward`. Verified empirically in P2a.0 Q1/Q2 evidence (`phase0_signature_evidence.txt`). This is an **upstream tt-metal bug** that we report separately. |
| 04 | `models/tt_transformers/tt/attention.py` | Add `_skip_self_attention` flag that uses `cur_pos-1` instead of `cur_pos` as the SDPA attention bound | Emulates a per-position tree-mask root self-mask without a full content-aware tree-mask kernel. Originally drafted for P3a.1 NGRAM tree-mask exploration; now used by EAGLE-3 verify in the no-trace path (the trace-capture blocker only fires when running under `trace_replay`, which the EAGLE verify hook does not). |

## How to apply

Inside the container (after `podman run ... local-tt-metal:dev`):

```bash
cd /tt-metal
for p in /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/tt_metal_patches/0*.patch; do
  echo "--- applying $(basename $p) ---"
  patch -p1 < "$p"
done
```

Patch 03 (`decode_forward`) should be **upstreamed** to Tenstorrent (`tt-metal`
repository), as it's a real bug affecting any SGLang user of `generator_sglang.py`.
Track the upstream submission in `/REBASE_TARGETS.md` (repo root).

## Pinned tt-metal commit

These patches are against tt-metal commit `89686ee7` (UMD bump 2026-05-12),
the same commit used to build `local-tt-metal:dev` (sha256:`973e972bddf5`).

If the tt-metal image is rebuilt at a different commit, patches may need
regeneration via `cd /tt-metal && git diff <path> > patch.diff` for each
file.
