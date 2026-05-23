"""WS-A.12 perf compare: TT-native vs host-fallback DeltaNet.

Measures per-step decode wall time at step=10 (warm), 20 steps, both modes.
Runs as a single process per mode to keep mesh state clean.
"""

from __future__ import annotations

import os
import sys
import time

import torch


def run_mode(native: bool, n_steps: int = 20):
    os.environ["SGLANG_TT_QWEN35_DELTANET_NATIVE"] = "1" if native else "0"
    import ttnn
    from _prefetcher_harness import build_paged_kv_cache, decode_one_step
    from _qwen35_harness import build_qwen35_model, close_mesh_device_with_fabric, open_2x_blackhole_mesh
    # WS-A.17: pick the trace-wrapped runner when SGLANG_TT_QWEN35_TRACE=1.
    from _qwen35_trace_harness import make_decode_runner, _trace_env_on

    mesh = open_2x_blackhole_mesh()
    times = []
    try:
        model, args = build_qwen35_model(mesh, install_loader_shims=True)
        kv, page = build_paged_kv_cache(model, args)
        runner = make_decode_runner(model, args, kv, page)
        trace_on = _trace_env_on() and native
        prev = 42
        for step in range(n_steps):
            t0 = time.perf_counter()
            logits = runner(step=step, token_id=prev)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            prev = int(logits[0, 0].argmax())
    finally:
        close_mesh_device_with_fabric(mesh)

    # report
    label = "native" if native else "host"
    if trace_on:
        label += "+trace"
    print(f"\n[{label}] step times (ms):", flush=True)
    for i, t in enumerate(times):
        print(f"  step={i:2d}  {1000*t:.1f} ms", flush=True)
    warm = times[5:]
    if warm:
        avg = sum(warm) / len(warm)
        print(f"\n[{label}] WARM avg (steps 5-{len(times)-1}): {1000*avg:.1f} ms", flush=True)
    return times


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "native"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run_mode(mode == "native", n_steps)


if __name__ == "__main__":
    main()
