# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Unit tests for CompressedHiCacheFile.

Run::

    pytest python/sglang/srt/mem_cache/storage/compressed/test_hicache_compressed.py -v

These tests do not require a GPU or a model; they construct synthetic
tensors that mimic real KV-cache pages.
"""

from __future__ import annotations

import tempfile

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.compressed.hicache_compressed import (
    CompressedHiCacheFile,
    _sem_split_bf16_like,
    _sem_unsplit_bf16_like,
)


def _cfg(extra: dict | None = None) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="testmodel",
        extra_config=extra,
    )


# ---------------------------------------------------------------------------
# Layout transform: pure-bytes round-trip
# ---------------------------------------------------------------------------

def test_sem_split_round_trip_bf16():
    torch.manual_seed(0)
    t = torch.randn(8, 1024, 128, dtype=torch.bfloat16) * 0.5
    raw = t.contiguous().view(torch.uint8).numpy().tobytes()
    split = _sem_split_bf16_like(raw)
    assert len(split) == (t.numel() + 7) // 8 + 2 * t.numel()
    recovered = _sem_unsplit_bf16_like(split, t.numel())
    assert recovered == raw


def test_sem_split_round_trip_fp16():
    torch.manual_seed(0)
    t = torch.randn(4, 512, 64, dtype=torch.float16)
    raw = t.contiguous().view(torch.uint8).numpy().tobytes()
    split = _sem_split_bf16_like(raw)
    recovered = _sem_unsplit_bf16_like(split, t.numel())
    assert recovered == raw


# ---------------------------------------------------------------------------
# End-to-end set/get round-trip — exhaustive lossless check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(8, 1024, 128), (16, 32, 64), (1024,)])
def test_set_get_lossless(tmp_path, dtype, shape):
    torch.manual_seed(123)
    t = (torch.randn(*shape) * 0.3).to(dtype)
    backend = CompressedHiCacheFile(_cfg(), file_path=str(tmp_path))

    assert backend.set("page0", t)
    assert backend.exists("page0")

    target = torch.zeros_like(t)
    out = backend.get("page0", target)
    assert out is not None
    # uint8 view equality avoids dtype-specific equality quirks (e.g. NaN handling)
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))


def test_set_get_lossless_realistic_distribution(tmp_path):
    """Typical KV cache values are clustered tightly. Make sure we still round-trip."""
    torch.manual_seed(7)
    # Approximate the layer-0 V distribution: tiny std, near zero
    t = (torch.randn(8, 8192, 128) * 0.04).to(torch.bfloat16)
    backend = CompressedHiCacheFile(_cfg(), file_path=str(tmp_path))
    assert backend.set("layer0_V", t)
    target = torch.zeros_like(t)
    out = backend.get("layer0_V", target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))

    stats = backend.get_stats()
    # Tightly-clustered distribution should compress significantly — at least 1.4x.
    assert stats["write_ratio"] > 1.4, f"unexpectedly low write_ratio: {stats}"


def test_batch_round_trip(tmp_path):
    torch.manual_seed(0)
    tensors = [
        (torch.randn(2, 256, 64) * 0.1).to(torch.bfloat16) for _ in range(4)
    ]
    keys = [f"page{i}" for i in range(len(tensors))]
    backend = CompressedHiCacheFile(_cfg(), file_path=str(tmp_path))

    assert backend.batch_set(keys, tensors)

    targets = [torch.zeros_like(x) for x in tensors]
    outs = backend.batch_get(keys, targets)
    for orig, dec in zip(tensors, outs):
        assert dec is not None
        assert torch.equal(orig.view(torch.uint8), dec.view(torch.uint8))


def test_force_layout_raw_still_lossless(tmp_path):
    """The `layout=raw` mode skips sem-split and just runs zstd directly."""
    torch.manual_seed(1)
    t = (torch.randn(2, 512, 64) * 0.1).to(torch.bfloat16)
    backend = CompressedHiCacheFile(_cfg(extra={"layout": "raw", "zstd_level": 3}), file_path=str(tmp_path))
    assert backend.set("p", t)
    target = torch.zeros_like(t)
    out = backend.get("p", target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))


def test_get_missing_returns_none(tmp_path):
    backend = CompressedHiCacheFile(_cfg(), file_path=str(tmp_path))
    target = torch.zeros(4, 4, dtype=torch.bfloat16)
    assert backend.get("never_written", target) is None


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

def test_factory_registered():
    """`compressed_file` must be a known backend."""
    assert "compressed_file" in StorageBackendFactory._registry


def test_factory_creates_instance(tmp_path, monkeypatch):
    # Direct the backend at our tmp dir via the env var the parent uses.
    monkeypatch.setenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", str(tmp_path))
    backend = StorageBackendFactory.create_backend(
        "compressed_file", _cfg(extra={"zstd_level": 3}), mem_pool_host=None
    )
    assert isinstance(backend, CompressedHiCacheFile)
    assert backend.zstd_level == 3
