# tt-metal patches for P2a.1 plugin-absorbed path

These patches are applied to the **tt-metal source inside the running container**
(`/tt-metal/...`) before launching the SGLang server with
`SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged`. They unblock issues found
during P2a.1 §9.1 smoke validation on 2× Tenstorrent Blackhole p150a:

| # | File | Fix | Why |
|---|---|---|---|
| 01 | `models/common/llama_models.py` | Soft-import `AutoModelForVision2Seq` etc. | Removed in `transformers` 5.x; not needed for text-only Llama path |
| 02 | `models/tt_transformers/tt/model_config.py` | Same soft-import + fallback `rope_theta=500000.0` for Llama-3.x | `transformers` 5.x `LlamaConfig.to_dict()` drops `rope_theta` from the top-level config; Llama-3 standard value is 500000.0 |
| 03 | `models/tt_transformers/tt/generator_sglang.py` | Fix `super().decode_forward_text()` → `super().decode_forward()` (4 sites) | `decode_forward_text` does not exist on `Generator`; the correct text-decode entry point is `decode_forward`. Verified empirically in P2a.0 Q1/Q2 evidence (`phase0_signature_evidence.txt`). This is an **upstream tt-metal bug** that we report separately. |

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
Track the upstream submission in `REBASE_TARGETS.md`.

## Pinned tt-metal commit

These patches are against tt-metal commit `89686ee7` (UMD bump 2026-05-12),
the same commit used to build `local-tt-metal:dev` (sha256:`973e972bddf5`).

If the tt-metal image is rebuilt at a different commit, patches may need
regeneration via `cd /tt-metal && git diff <path> > patch.diff` for each
file.
