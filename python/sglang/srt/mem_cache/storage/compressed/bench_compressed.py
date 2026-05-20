# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Standalone micro-benchmark for the compressed HiCache backend.

Runs the storage backend round-trip on either:
  - a real KV-cache page dumped by the kvcache_comp prototype
    (``--kv-dir /mnt/ssd/mhnie/kvcache_comp/data/llama3.1_8b_ctx8192``), or
  - a synthetic BF16 tensor of a chosen shape (``--synthetic --shape ...``).

Reports per-page compression ratio, set/get latency, set/get throughput,
and verifies lossless round-trip bit-for-bit.

Run with::

    python -m sglang.srt.mem_cache.storage.compressed.bench_compressed \\
        --kv-dir /mnt/ssd/mhnie/kvcache_comp/data/llama3.1_8b_ctx8192 \\
        --layers 1,16,31 --zstd-level 1

This is a developer tool — it bypasses HiCacheStorageConfig wiring and
talks to the backend directly so you can iterate on the codec without
spinning up an inference server.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.compressed.hicache_compressed import (
    CompressedHiCacheFile,
)


def make_backend(tmpdir: str, zstd_level: int, layout: str) -> CompressedHiCacheFile:
    cfg = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="bench",
        extra_config={"zstd_level": zstd_level, "layout": layout},
    )
    return CompressedHiCacheFile(cfg, file_path=tmpdir)


def bench_tensor(name: str, t: torch.Tensor, backend: CompressedHiCacheFile, repeats: int = 3) -> dict:
    """Round-trip + timing for a single tensor. Returns a stats dict."""
    nbytes = t.numel() * t.element_size()

    # warm-up + time set
    best_set = float("inf")
    for _ in range(repeats):
        backend_path = os.path.join(backend.file_path, f"{name}{backend.config_suffix}.bin")
        if os.path.exists(backend_path):
            os.remove(backend_path)
        t0 = time.perf_counter()
        ok = backend.set(name, t)
        dt = time.perf_counter() - t0
        assert ok, f"set failed for {name}"
        best_set = min(best_set, dt)

    blob_size = os.path.getsize(backend_path)

    # time get + verify lossless
    best_get = float("inf")
    decoded = torch.empty_like(t)
    for _ in range(repeats):
        decoded.zero_()
        t0 = time.perf_counter()
        ret = backend.get(name, decoded)
        dt = time.perf_counter() - t0
        assert ret is not None, f"get failed for {name}"
        best_get = min(best_get, dt)

    # bit-equal check (using uint8 view to bypass any dtype-specific equality semantics)
    if not torch.equal(t.view(torch.uint8), decoded.view(torch.uint8)):
        raise RuntimeError(f"LOSSY round-trip for {name}")

    return {
        "name": name,
        "shape": tuple(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "orig_bytes": nbytes,
        "blob_bytes": blob_size,  # includes 24-byte header
        "ratio": nbytes / max(blob_size, 1),
        "set_ms": best_set * 1000,
        "get_ms": best_get * 1000,
        "set_mbps": (nbytes / 1e6) / best_set,
        "get_mbps": (nbytes / 1e6) / best_get,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kv-dir",
        default=None,
        help="path to a kvcache_comp dump dir (with layer*.pt and meta.json)",
    )
    ap.add_argument(
        "--layers",
        default=None,
        help="comma-separated layer indices to test (default: 3 sampled, "
        "skipping layer 0 to avoid the V outlier)",
    )
    ap.add_argument("--synthetic", action="store_true", help="ignore --kv-dir, use a synthetic tensor")
    ap.add_argument("--shape", default="8,8192,128", help="synthetic shape, e.g. 8,8192,128")
    ap.add_argument("--zstd-level", type=int, default=1)
    ap.add_argument("--layout", default="auto", choices=["auto", "sem_split", "raw"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="sglang_kvcomp_bench_") as tmpdir:
        backend = make_backend(tmpdir, args.zstd_level, args.layout)
        rows = []

        if args.synthetic or not args.kv_dir:
            shape = tuple(int(x) for x in args.shape.split(","))
            t = torch.randn(*shape, dtype=torch.bfloat16) * 0.1
            rows.append(bench_tensor("synthetic", t, backend, args.repeats))
        else:
            kv_dir = Path(args.kv_dir)
            meta = json.load(open(kv_dir / "meta.json"))
            full_layers = meta.get("full_attn_layers") or list(range(meta["num_layers"]))
            if args.layers:
                layer_set = [int(x) for x in args.layers.split(",")]
            else:
                usable = [l for l in full_layers if l != 0]
                if len(usable) >= 3:
                    layer_set = [usable[0], usable[len(usable) // 2], usable[-1]]
                else:
                    layer_set = usable
            print(f"Loading {len(layer_set) * 2} tensors from {kv_dir}…")
            for li in layer_set:
                for kv in ("K", "V"):
                    t = torch.load(kv_dir / f"layer{li:02d}_{kv}.pt", weights_only=True)
                    rows.append(bench_tensor(f"L{li}_{kv}", t, backend, args.repeats))

        # report
        print()
        print(f"  CompressedHiCacheFile  zstd-{args.zstd_level}  layout={args.layout}")
        print()
        print(
            f"  {'name':>10s}  {'shape':<24s}  {'dtype':<8s}  "
            f"{'orig MB':>8s}  {'comp MB':>8s}  {'ratio':>6s}  "
            f"{'set MB/s':>10s}  {'get MB/s':>10s}"
        )
        for r in rows:
            print(
                f"  {r['name']:>10s}  {str(r['shape']):<24s}  {r['dtype']:<8s}  "
                f"{r['orig_bytes']/1e6:8.2f}  {r['blob_bytes']/1e6:8.2f}  {r['ratio']:6.3f}  "
                f"{r['set_mbps']:10.1f}  {r['get_mbps']:10.1f}"
            )

        # aggregate
        if len(rows) > 1:
            total_orig = sum(r["orig_bytes"] for r in rows)
            total_comp = sum(r["blob_bytes"] for r in rows)
            mean_set = sum(r["set_mbps"] for r in rows) / len(rows)
            mean_get = sum(r["get_mbps"] for r in rows) / len(rows)
            print()
            print(
                f"  aggregate orig={total_orig/1e6:.1f} MB "
                f"comp={total_comp/1e6:.1f} MB ratio={total_orig/max(total_comp,1):.3f}x  "
                f"set={mean_set:.1f} MB/s  get={mean_get:.1f} MB/s"
            )

        stats = backend.get_stats()
        print(f"  backend stats: {stats}")


if __name__ == "__main__":
    main()
