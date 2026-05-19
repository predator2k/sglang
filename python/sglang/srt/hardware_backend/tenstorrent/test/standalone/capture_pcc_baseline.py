"""Capture prefetcher-OFF decode logits at steps 0, 5, 10.

Saves to _fixtures/prefetcher_pcc_baseline.pt. Used by
test_prefetcher_pcc.py to verify the prefetcher path is
numerically equivalent.

Run inside p3a-ngram (prefetcher disabled):
    podman exec p3a-ngram bash -c 'cd /tt-metal && \
        PYTHONPATH=/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone:$PYTHONPATH \
        SGLANG_TT_USE_PREFETCHER=0 \
        python /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone/capture_pcc_baseline.py'

Phase A.4 of the Option A prefetcher plan.
"""

import torch
import ttnn

from _prefetcher_harness import (
    build_paged_kv_cache,
    build_prefetcher_off_model,
    close_mesh_device_with_fabric,
    decode_one_step,
    open_2x_blackhole_mesh,
)


CAPTURE_STEPS = [0, 5, 10]
FIXTURE_PATH = "/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/prefetcher_pcc_baseline.pt"


def main():
    mesh_device = open_2x_blackhole_mesh()
    try:
        model, model_args = build_prefetcher_off_model(mesh_device)

        # initialize_sglang_text_transformer always uses use_paged_kv_cache=True,
        # so layer.attention.layer_past is never set. Allocate a paged KV cache
        # and pass it through every decode step.
        kv_cache, page_table_host = build_paged_kv_cache(model, model_args)

        captured = {}
        max_step = max(CAPTURE_STEPS)
        for step in range(max_step + 1):
            logits = decode_one_step(
                model, model_args, step=step,
                kv_cache=kv_cache, page_table_host=page_table_host,
            )
            if step in CAPTURE_STEPS:
                captured[step] = logits.detach().cpu().float().clone()
                print(f"Captured step {step}: logits shape {captured[step].shape}")

        torch.save(captured, FIXTURE_PATH)
        print(f"\nSaved baseline to {FIXTURE_PATH}")
    finally:
        close_mesh_device_with_fabric(mesh_device)


if __name__ == "__main__":
    main()
