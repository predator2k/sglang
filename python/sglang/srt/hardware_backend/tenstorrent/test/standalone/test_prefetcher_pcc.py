"""PCC test: prefetcher path must match prefetcher-OFF baseline.

Loads the baseline fixture from prefetcher_pcc_baseline.pt
(captured by capture_pcc_baseline.py) and verifies that running
with use_prefetcher=True produces logits with per-position
cosine similarity >= 0.99 at decode steps 0, 5, 10.

Run inside p3a-ngram (prefetcher enabled):
    podman exec p3a-ngram bash -c 'cd /tt-metal && \
        PYTHONPATH=/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone:$PYTHONPATH \
        SGLANG_TT_USE_PREFETCHER=1 \
        pytest /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone/test_prefetcher_pcc.py -s -v'

Phase A.4 of the Option A plan; reused across A.3/A.1/A.2.
"""

import pytest
import torch

try:
    import ttnn
    from _prefetcher_harness import (
        build_paged_kv_cache,
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

PCC_THRESHOLD = 0.99
FIXTURE_PATH = "/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/prefetcher_pcc_baseline.pt"
CAPTURE_STEPS = [0, 5, 10]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-position cosine similarity. Both tensors must be the same shape."""
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    sims = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return float(sims.mean())


@pytest.fixture(scope="module")
def baseline_logits():
    return torch.load(FIXTURE_PATH)


@pytest.fixture(scope="module")
def mesh_device():
    device = open_2x_blackhole_mesh()
    yield device
    close_mesh_device_with_fabric(device)


@pytest.fixture(scope="module")
def model_with_prefetcher(mesh_device):
    model, model_args = build_prefetcher_on_model(mesh_device)
    kv_cache, page_table_host = build_paged_kv_cache(model, model_args)
    return model, model_args, kv_cache, page_table_host


def test_prefetcher_pcc(model_with_prefetcher, baseline_logits):
    model, model_args, kv_cache, page_table_host = model_with_prefetcher

    max_step = max(CAPTURE_STEPS)
    failures = []
    for step in range(max_step + 1):
        logits = decode_one_step(
            model, model_args, step=step,
            kv_cache=kv_cache, page_table_host=page_table_host,
        )
        if step in CAPTURE_STEPS:
            actual = logits.detach().cpu().float()
            expected = baseline_logits[step]

            pcc = cosine_similarity(actual, expected)
            print(f"Step {step}: PCC = {pcc:.6f}")
            if pcc < PCC_THRESHOLD:
                failures.append((step, pcc))

    if failures:
        msg = "\n".join(f"  step {s}: PCC = {p:.4f} (< {PCC_THRESHOLD})" for s, p in failures)
        pytest.fail(f"Prefetcher path fails PCC at:\n{msg}")
