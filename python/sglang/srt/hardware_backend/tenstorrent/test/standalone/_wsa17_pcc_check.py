"""WS-A.17 PCC check — trace replay vs eager host fallback.

Runs N steps in each mode (separate processes to keep mesh state clean)
and compares the per-step argmax tokens + per-step logit PCC.

Baseline = host fallback (HOST_FALLBACK mode = eager torch DeltaNet).  The
TT-native eager path (B=32 vs assert B==1) has been broken since
WS-A.12 — proved by the host-vs-native warm avg being identical and by the
``assert B == 1`` in ``_tt_native_delta_net_step``.  Host fallback is the
only correct eager baseline available.

PCC metric: Pearson correlation across the full vocab dimension at each
step.  Target: > 0.95 (trace introduces bf16 quantization vs the host's
fp32 reference; bit-exact is not expected).
"""

from __future__ import annotations

import os
import sys

import torch


def run_mode(trace: bool, n_steps: int = 6):
    if trace:
        os.environ["SGLANG_TT_QWEN35_DELTANET_NATIVE"] = "1"
        os.environ["SGLANG_TT_QWEN35_TRACE"] = "1"
    else:
        os.environ["SGLANG_TT_QWEN35_DELTANET_NATIVE"] = "0"
        os.environ["SGLANG_TT_QWEN35_TRACE"] = "0"

    from _prefetcher_harness import build_paged_kv_cache
    from _qwen35_harness import build_qwen35_model, close_mesh_device_with_fabric, open_2x_blackhole_mesh
    from _qwen35_trace_harness import make_decode_runner

    mesh = open_2x_blackhole_mesh()
    tokens = []
    logits_list = []
    try:
        model, args = build_qwen35_model(mesh, install_loader_shims=True)
        kv, page = build_paged_kv_cache(model, args)
        runner = make_decode_runner(model, args, kv, page)
        prev = 42
        for step in range(n_steps):
            logits = runner(step=step, token_id=prev)
            tokens.append(int(logits[0, 0].argmax()))
            logits_list.append(logits.detach().clone())
            prev = tokens[-1]
    finally:
        close_mesh_device_with_fabric(mesh)
    return tokens, logits_list


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    a_c = a_f - a_f.mean()
    b_c = b_f - b_f.mean()
    denom = (a_c.norm() * b_c.norm()).clamp_min(1e-12)
    return float((a_c @ b_c) / denom)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "trace"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    is_trace = mode == "trace"
    tokens, logits_list = run_mode(trace=is_trace, n_steps=n_steps)
    print(f"\n[{mode}] tokens (first 6): {tokens[:6]}", flush=True)

    if out_path:
        torch.save({"tokens": tokens, "logits": [t.cpu() for t in logits_list]}, out_path)
        print(f"[{mode}] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
