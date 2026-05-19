"""Standalone prefetcher smoke test.

Runs Qwen3-8B decode 5 times with use_prefetcher=True
and reports per-step latency. Bypasses the ~3 min SGLang
server warm-up by going straight through ttnn.

Run inside the p3a-ngram container:
    podman exec p3a-ngram bash -c 'cd /tt-metal && \
        SGLANG_TT_USE_PREFETCHER=1 \
        pytest /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone/test_prefetcher_smoke.py -s -v'

Phase A.4 of the Option A prefetcher plan.
"""

import time

import pytest

try:
    from _prefetcher_harness import (
        build_prefetcher_on_model,
        close_mesh_device_with_fabric,
        decode_one_step,
        open_2x_blackhole_mesh,
    )

    TT_METAL_AVAILABLE = True
except ImportError:
    TT_METAL_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not TT_METAL_AVAILABLE,
    reason="tt-metal not importable; run inside p3a-ngram container",
)


@pytest.fixture(scope="module")
def mesh_device():
    device = open_2x_blackhole_mesh()
    yield device
    close_mesh_device_with_fabric(device)


@pytest.fixture(scope="module")
def model_with_prefetcher(mesh_device):
    return build_prefetcher_on_model(mesh_device)


def test_prefetcher_smoke_5_decodes(model_with_prefetcher):
    model, model_args = model_with_prefetcher

    latencies_ms = []
    for step in range(5):
        t0 = time.perf_counter()
        logits = decode_one_step(model, model_args, step=step)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        assert logits is not None, f"decode returned None at step {step}"

    print(f"\nPer-step latencies (ms): {latencies_ms}")
    print(f"Median: {sorted(latencies_ms)[len(latencies_ms) // 2]:.2f} ms")

    for step, lat in enumerate(latencies_ms):
        assert lat < 500.0, f"Step {step} latency {lat:.2f} ms is implausible"
