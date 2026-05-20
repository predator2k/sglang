# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Unit tests for CompressedHiCacheFile.

Run::

    pytest python/sglang/srt/mem_cache/storage/compressed/test_hicache_compressed.py -v
"""

from __future__ import annotations

import textwrap

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.compressed.hicache_compressed import (
    CompressedHiCacheFile,
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
# Single-slab end-to-end (default path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(8, 1024, 128), (16, 32, 64), (1024,)])
def test_set_get_lossless(tmp_path, dtype, shape):
    torch.manual_seed(123)
    t = (torch.randn(*shape) * 0.3).to(dtype)
    backend = CompressedHiCacheFile(_cfg(), file_path=str(tmp_path))
    assert backend.set("page0", t)
    target = torch.zeros_like(t)
    out = backend.get("page0", target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))


def test_set_get_lossless_realistic_distribution(tmp_path):
    """Layer-0 V is tightly clustered; should still round-trip exactly."""
    torch.manual_seed(7)
    t = (torch.randn(8, 8192, 128) * 0.04).to(torch.bfloat16)
    backend = CompressedHiCacheFile(_cfg(), file_path=str(tmp_path))
    assert backend.set("layer0_V", t)
    target = torch.zeros_like(t)
    out = backend.get("layer0_V", target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))


def test_batch_round_trip(tmp_path):
    torch.manual_seed(0)
    tensors = [(torch.randn(2, 256, 64) * 0.1).to(torch.bfloat16) for _ in range(4)]
    keys = [f"page{i}" for i in range(len(tensors))]
    backend = CompressedHiCacheFile(_cfg(), file_path=str(tmp_path))
    assert backend.batch_set(keys, tensors)
    targets = [torch.zeros_like(x) for x in tensors]
    outs = backend.batch_get(keys, targets)
    for orig, dec in zip(tensors, outs):
        assert dec is not None
        assert torch.equal(orig.view(torch.uint8), dec.view(torch.uint8))


# ---------------------------------------------------------------------------
# Codec / layout knobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "codec",
    [
        "zstd",
        "zstd_chunked",
        pytest.param("blosc2", marks=pytest.mark.skipif(__import__("importlib").util.find_spec("blosc2") is None, reason="blosc2 not installed")),
        pytest.param("lz4", marks=pytest.mark.skipif(__import__("importlib").util.find_spec("lz4") is None, reason="lz4 not installed")),
        pytest.param("isal_gzip", marks=pytest.mark.skipif(__import__("importlib").util.find_spec("isal") is None, reason="isal not installed")),
    ],
)
def test_codec_round_trip(tmp_path, codec):
    torch.manual_seed(0)
    t = (torch.randn(8, 1024, 128) * 0.1).to(torch.bfloat16)
    backend = CompressedHiCacheFile(
        _cfg(extra={"codec": codec, "compression_level": 1}),
        file_path=str(tmp_path),
    )
    assert backend.set("k", t)
    target = torch.zeros_like(t)
    out = backend.get("k", target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))


@pytest.mark.parametrize(
    "layout",
    ["raw", "channel_major", "byte_hi_lo", "byte_hi_lo_channel",
     "sem_split", "sem_split_channel", "zipnn_channel_delta", "bit_plane"],
)
def test_layout_round_trip(tmp_path, layout):
    torch.manual_seed(0)
    t = (torch.randn(8, 1024, 128) * 0.1).to(torch.bfloat16)
    backend = CompressedHiCacheFile(
        _cfg(extra={"layout": layout, "channels": 8 * 128, "compression_level": 1}),
        file_path=str(tmp_path),
    )
    assert backend.set("k", t)
    target = torch.zeros_like(t)
    out = backend.get("k", target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))


def test_force_layout_raw_still_lossless(tmp_path):
    torch.manual_seed(1)
    t = (torch.randn(2, 512, 64) * 0.1).to(torch.bfloat16)
    backend = CompressedHiCacheFile(
        _cfg(extra={"layout": "raw", "compression_level": 3}),
        file_path=str(tmp_path),
    )
    assert backend.set("p", t)
    target = torch.zeros_like(t)
    out = backend.get("p", target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), t.view(torch.uint8))


def test_strict_dtype_rejects_mismatch(tmp_path):
    backend = CompressedHiCacheFile(
        _cfg(extra={"strict_dtype": True}),
        file_path=str(tmp_path),
    )
    t = (torch.randn(64, 64) * 0.1).to(torch.bfloat16)
    assert backend.set("k", t)
    target = torch.zeros(64, 64, dtype=torch.float16)
    assert backend.get("k", target) is None


def test_lax_dtype_allows_mismatch(tmp_path):
    backend = CompressedHiCacheFile(
        _cfg(extra={"strict_dtype": False}),
        file_path=str(tmp_path),
    )
    t = (torch.randn(64, 64) * 0.1).to(torch.bfloat16)
    assert backend.set("k", t)
    target = torch.zeros(64, 64, dtype=torch.float16)
    assert backend.get("k", target) is not None


def test_get_missing_returns_none(tmp_path):
    backend = CompressedHiCacheFile(_cfg(), file_path=str(tmp_path))
    target = torch.zeros(4, 4, dtype=torch.bfloat16)
    assert backend.get("never_written", target) is None


# ---------------------------------------------------------------------------
# Multi-slab + YAML policy
# ---------------------------------------------------------------------------


def test_multi_slab_yaml_policy(tmp_path):
    """A YAML policy with per-layer rules round-trips losslessly across slabs."""
    L, P, H, D = 4, 8, 4, 64
    torch.manual_seed(0)
    page = (torch.randn(2, L, P, H, D) * 0.1).to(torch.bfloat16)
    flat = page.contiguous().view(-1).clone()

    yaml_path = tmp_path / "policy.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            default:
              codec: zstd
              compression_level: 1
              layout: sem_split_channel
            rules:
              - layers: [0]
                K: { codec: zstd, compression_level: 1, layout: sem_split_channel }
                V: { codec: zstd, compression_level: 9, layout: sem_split_channel }
              - layers: "1-2"
                codec: zstd
                compression_level: 3
                layout: byte_hi_lo_channel
              - layers: [3]
                codec: lz4
                layout: byte_hi_lo_channel
            """
        ).strip()
    )

    backend = CompressedHiCacheFile(
        _cfg(extra={"profiles_yaml": str(yaml_path), "slab_threads": 2}),
        file_path=str(tmp_path),
    )

    class _Pool:
        head_num = H
        head_dim = D
        layer_num = L
        page_size = P
        layout = "layer_first"

    backend.register_mem_pool_host(_Pool())

    assert backend.set("page0", flat)
    target = torch.zeros_like(flat)
    out = backend.get("page0", target)
    assert out is not None
    assert torch.equal(flat.view(torch.uint8), target.view(torch.uint8))


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_registered():
    assert "compressed_file" in StorageBackendFactory._registry


def test_factory_creates_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", str(tmp_path))
    backend = StorageBackendFactory.create_backend(
        "compressed_file", _cfg(extra={"compression_level": 3}), mem_pool_host=None
    )
    assert isinstance(backend, CompressedHiCacheFile)
    assert backend.policy.default.compression_level == 3
