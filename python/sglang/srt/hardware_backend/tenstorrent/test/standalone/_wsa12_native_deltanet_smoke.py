"""WS-A.12 smoke: TT-native GatedDeltaNet vs host-fallback.

Builds Qwen3.5-0.8B twice (once host-fallback, once TT-native via env), runs
the same 5 decode steps for token 42, and compares logits row-0 between the
two paths. Also re-loads the HF reference pickle for an absolute PCC anchor.

Run inside p3a-ngram container:
    PYTHONPATH=/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone:\
              /sglang/python:$PYTHONPATH \
    HF_MODEL=/tt-metal/models/weights/Qwen3.5-0.8B TRANSFORMERS_OFFLINE=1 \
    python3 _wsa12_native_deltanet_smoke.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import torch

# Sentinel — we drive each mode by setting the env var BEFORE importing
# tt_transformers (and re-importing if needed). Each subprocess captures
# one configuration so module state can't leak.

START_TOKEN = 42
CAPTURE_STEPS = [0, 1, 4]
MAX_STEPS = max(CAPTURE_STEPS) + 1
HF_REF_PATH = Path("/tmp/qwen35_hf_ref_logits.pkl")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    a_c = a_f - a_f.mean()
    b_c = b_f - b_f.mean()
    num = (a_c * b_c).sum()
    den = (a_c.norm() * b_c.norm()).clamp(min=1e-8)
    return float(num / den)


def run_one_mode(native_enabled: bool, hf_chosen: list[int]) -> dict[int, torch.Tensor]:
    """Build model in selected mode and capture per-step logits."""
    os.environ["SGLANG_TT_QWEN35_DELTANET_NATIVE"] = "1" if native_enabled else "0"

    # Late imports so the env var is honored
    import ttnn
    from _prefetcher_harness import build_paged_kv_cache, decode_one_step
    from _qwen35_harness import build_qwen35_model, close_mesh_device_with_fabric, open_2x_blackhole_mesh

    print(f"[mode={'native' if native_enabled else 'host'}] opening 2x Blackhole mesh...", flush=True)
    mesh = open_2x_blackhole_mesh()
    out = {}
    try:
        model, args = build_qwen35_model(mesh, install_loader_shims=True)
        kv, page = build_paged_kv_cache(model, args)
        for step in range(MAX_STEPS):
            in_tok = START_TOKEN if step == 0 else hf_chosen[step - 1]
            logits = decode_one_step(
                model, args, step=step, token_id=in_tok,
                kv_cache=kv, page_table_host=page,
            )
            if step in CAPTURE_STEPS:
                out[step] = logits[0, 0].float().cpu()
                print(
                    f"  step={step} top1={int(logits[0, 0].argmax())} "
                    f"|logits|max={float(logits.abs().max()):.2f} "
                    f"|logits|mean={float(logits.abs().mean()):.4f}",
                    flush=True,
                )
    finally:
        close_mesh_device_with_fabric(mesh)
    return out


def main():
    if not HF_REF_PATH.exists():
        print(f"FATAL: {HF_REF_PATH} missing; generate with /tmp/qwen35_hf_reference.py")
        sys.exit(2)
    with open(HF_REF_PATH, "rb") as f:
        ref = pickle.load(f)
    hf_logits = ref["logits"]
    hf_chosen = ref["meta"]["chosen_tokens"]

    mode = sys.argv[1] if len(sys.argv) > 1 else "native"
    native = (mode == "native")

    print(f"[ws-a.12 smoke] running mode={mode}", flush=True)
    out = run_one_mode(native, hf_chosen)

    # Save under the matching key for cross-run compare
    out_path = Path(f"/tmp/qwen35_wsa12_{mode}_logits.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"[ws-a.12 smoke] saved {out_path}", flush=True)

    # PCC vs HF
    print(f"\n[ws-a.12 smoke] PCC vs HF reference (mode={mode}):")
    for s in CAPTURE_STEPS:
        if s not in out or s not in hf_logits:
            print(f"  step={s} MISSING")
            continue
        # Both are full vocab; trim to min len
        a, b = out[s], hf_logits[s]
        n = min(a.numel(), b.numel())
        p = pcc(a[:n], b[:n])
        cos = torch.nn.functional.cosine_similarity(a[:n].view(1, -1), b[:n].view(1, -1)).item()
        top1_match = int(a[:n].argmax()) == int(b[:n].argmax())
        print(f"  step={s} PCC={p:.4f} cos={cos:.4f} top1_match={top1_match}")


if __name__ == "__main__":
    main()
