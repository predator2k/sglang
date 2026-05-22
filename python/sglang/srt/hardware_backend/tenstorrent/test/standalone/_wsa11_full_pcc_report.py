"""WS-A.11 full step-by-step PCC report for Qwen3.5-0.8B.

Runs decode for MAX_STEPS=17, captures logits at steps 0/1/4/16, compares to
HF reference, prints a clean PCC table. Does NOT abort on first PCC fail
(unlike pytest), so we see every step.
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import torch

sys.path.insert(
    0,
    "/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone",
)

import ttnn
from _prefetcher_harness import build_paged_kv_cache, decode_one_step
from _qwen35_harness import (
    build_qwen35_model,
    close_mesh_device_with_fabric,
    open_2x_blackhole_mesh,
)


CAPTURE_STEPS = [0, 1, 4, 16]
MAX_STEPS = max(CAPTURE_STEPS) + 1
START_TOKEN = 42
HF_REF = "/tmp/qwen35_hf_ref_logits.pkl"


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    a_c = a - a.mean()
    b_c = b - b.mean()
    den = (a_c.norm() * b_c.norm()).clamp(min=1e-12)
    return float((a_c * b_c).sum() / den)


def cossim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.view(1, -1).float(), b.view(1, -1).float()
    ).item())


def main() -> int:
    with open(HF_REF, "rb") as f:
        ref = pickle.load(f)
    hf_chosen = ref["meta"]["chosen_tokens"]
    print(f"HF chosen tokens (first 5): {hf_chosen[:5]}")
    print(f"HF chosen tokens (steps 0,1,4,16): "
          f"{[hf_chosen[i] for i in CAPTURE_STEPS]}")

    mesh = open_2x_blackhole_mesh()
    tt_logits = {}
    try:
        t_load = time.time()
        model, args = build_qwen35_model(mesh, install_loader_shims=True)
        kv, page = build_paged_kv_cache(model, args)
        print(f"Model+cache built in {time.time()-t_load:.1f}s "
              f"(n_kv_heads={args.n_kv_heads}, "
              f"n_local_kv={args.n_kv_heads//args.num_devices})")

        for step in range(MAX_STEPS):
            in_tok = START_TOKEN if step == 0 else hf_chosen[step - 1]
            t0 = time.time()
            logits = decode_one_step(
                model, args, step=step, token_id=in_tok,
                kv_cache=kv, page_table_host=page,
            )
            dt = time.time() - t0
            top1 = int(logits[0, 0].argmax().item())
            hf_top1 = hf_chosen[step] if step < len(hf_chosen) else -1
            print(f"step={step:2d} in_tok={in_tok:6d} tt_top1={top1:6d} "
                  f"hf_top1={hf_top1:6d} match={top1==hf_top1} dt={dt:.2f}s")
            if step in CAPTURE_STEPS:
                tt_logits[step] = logits[0, 0].float().cpu()
    finally:
        try:
            close_mesh_device_with_fabric(mesh)
        except Exception as e:
            print(f"MESH_CLOSE_WARN: {e}")

    print()
    print("=" * 68)
    print(f"{'step':>5}  {'PCC':>8}  {'cosine':>8}  {'tt_top1':>10}  "
          f"{'hf_top1':>10}  {'top1_match':>10}")
    for s in CAPTURE_STEPS:
        if s not in tt_logits:
            continue
        a = ref["logits"][s]
        b = tt_logits[s]
        p = pcc(a, b)
        c = cossim(a, b)
        tt_t1 = int(b.argmax().item())
        hf_t1 = int(a.argmax().item())
        print(f"{s:>5d}  {p:>8.4f}  {c:>8.4f}  {tt_t1:>10d}  "
              f"{hf_t1:>10d}  {str(tt_t1==hf_t1):>10}")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    sys.exit(main())
