# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Compressed file backend for HiCache (KV cache prefix offload).

This is a prototype that demonstrates lossless KV cache compression on the
CPU host side. It inherits the file-backed storage and only overrides the
per-page set/get path to insert a layout transform + zstd codec.

The transform is sign/exponent/mantissa byte splitting (ZipNN-style) for
BF16/FP16 tensors. zstd level 1 is the documented sweet spot — see
``../../docs_new/source/developer_guide/kv_cache_compression.md`` and the
ablation in the standalone kvcache_comp prototype.

File format on disk::

    bytes 0..7   magic         = b"SGLKVCMP"
    byte  8      version       = 1
    byte  9      layout_kind   = 0 raw, 1 sem_split (BF16/FP16), 2 sem_split (FP8 placeholder)
    byte  10     dtype_kind    = 0 unknown/raw, 1 bf16, 2 fp16, 3 fp8_e4m3, 4 fp8_e5m2
    byte  11     zstd_level    = uint8
    bytes 12..15 reserved
    bytes 16..23 orig_bytes    = uint64 little-endian
    bytes 24..   zstd payload

The decoder uses the target tensor's shape/dtype, so no shape is stored in
the file. Only the *number of bytes* is recorded so a short-read can be
detected.
"""

from __future__ import annotations

import logging
import os
import struct
from typing import Any, List, Optional

import numpy as np
import torch

try:
    import zstandard as zstd
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "compressed_file backend requires `zstandard`. "
        "Install with: pip install zstandard"
    ) from e

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

_MAGIC = b"SGLKVCMP"
_VERSION = 1
_HEADER_FMT = "<8sBBBB4sQ"  # 8s + 4B + 4s + Q = 8+1+1+1+1+4+8 = 24 bytes
_HEADER_LEN = struct.calcsize(_HEADER_FMT)
assert _HEADER_LEN == 24

# layout_kind codes
_LK_RAW = 0
_LK_SEM_SPLIT = 1

# dtype_kind codes
_DT_UNKNOWN = 0
_DT_BF16 = 1
_DT_FP16 = 2
_DT_FP8_E4M3 = 3
_DT_FP8_E5M2 = 4

_TORCH_TO_DTKIND = {
    torch.bfloat16: _DT_BF16,
    torch.float16: _DT_FP16,
}
if hasattr(torch, "float8_e4m3fn"):
    _TORCH_TO_DTKIND[torch.float8_e4m3fn] = _DT_FP8_E4M3
if hasattr(torch, "float8_e5m2"):
    _TORCH_TO_DTKIND[torch.float8_e5m2] = _DT_FP8_E5M2


def _pack_header(layout_kind: int, dtype_kind: int, zstd_level: int, orig_bytes: int) -> bytes:
    return struct.pack(
        _HEADER_FMT, _MAGIC, _VERSION, layout_kind, dtype_kind, zstd_level, b"\x00" * 4, orig_bytes
    )


def _unpack_header(buf: bytes) -> tuple[int, int, int, int]:
    magic, version, layout_kind, dtype_kind, zstd_level, _reserved, orig_bytes = struct.unpack(
        _HEADER_FMT, buf[:_HEADER_LEN]
    )
    if magic != _MAGIC:
        raise ValueError(f"bad magic in compressed page: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported compressed-page version {version}")
    return layout_kind, dtype_kind, zstd_level, orig_bytes


# ----------------------------------------------------------------------------
# Layout transform (lossless, dtype-aware)
# ----------------------------------------------------------------------------


def _sem_split_bf16_like(raw_bytes: bytes) -> bytes:
    """Split a BF16/FP16 byte stream into (sign 1b, exp 8b, mant 7b) streams.

    BF16 / FP16 share the same total width (16 bits) but different exp/mantissa
    splits. We deliberately use the BF16 split (1 sign + 8 exp + 7 mant) for
    both — it's still lossless because we treat the value as a uint16 and
    extract specific bit positions. Decoder mirrors this.
    """
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    sign = ((u16 >> 15) & 1).astype(np.uint8)
    exp = ((u16 >> 7) & 0xFF).astype(np.uint8)
    mant = (u16 & 0x7F).astype(np.uint8)
    sign_packed = np.packbits(sign, bitorder="little")
    return sign_packed.tobytes() + exp.tobytes() + mant.tobytes()


def _sem_unsplit_bf16_like(split_bytes: bytes, n_values: int) -> bytes:
    """Inverse of _sem_split_bf16_like. Returns the original uint16 byte stream."""
    n_sign_bytes = (n_values + 7) // 8
    sign_packed = np.frombuffer(split_bytes[:n_sign_bytes], dtype=np.uint8)
    sign = np.unpackbits(sign_packed, bitorder="little")[:n_values]
    exp = np.frombuffer(split_bytes[n_sign_bytes : n_sign_bytes + n_values], dtype=np.uint8)
    mant = np.frombuffer(split_bytes[n_sign_bytes + n_values : n_sign_bytes + 2 * n_values], dtype=np.uint8)
    u16 = (
        (sign.astype(np.uint16) << 15)
        | (exp.astype(np.uint16) << 7)
        | mant.astype(np.uint16)
    )
    return u16.tobytes()


def _choose_layout(dtype: torch.dtype) -> tuple[int, int]:
    """Return (layout_kind, dtype_kind) for the given tensor dtype."""
    if dtype in (torch.bfloat16, torch.float16):
        return _LK_SEM_SPLIT, _TORCH_TO_DTKIND[dtype]
    # Unknown dtypes (FP8, FP32, INT8 …) fall back to raw zstd. Still lossless,
    # just lower ratio. We add proper splits in follow-ups.
    return _LK_RAW, _TORCH_TO_DTKIND.get(dtype, _DT_UNKNOWN)


# ----------------------------------------------------------------------------
# Compressed file backend
# ----------------------------------------------------------------------------


class CompressedHiCacheFile(HiCacheFile):
    """HiCacheFile variant that compresses each page with sem_split + zstd-1.

    Drop-in replacement for the ``"file"`` storage backend. Pages are stored
    as a 24-byte header plus a zstd-compressed payload. Read path is in-place:
    decompressed bytes are copied into the caller-provided ``target_location``.

    Configurable via :class:`HiCacheStorageConfig.extra_config`::

        extra_config = {
            "zstd_level": 1,            # 1..22; 1 is the measured sweet spot
            "layout": "sem_split",      # or "raw" for an A/B baseline
        }
    """

    DEFAULT_ZSTD_LEVEL = 1

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache_compressed",
    ) -> None:
        super().__init__(storage_config, file_path=file_path)
        extra = storage_config.extra_config or {}
        self.zstd_level = int(extra.get("zstd_level", self.DEFAULT_ZSTD_LEVEL))
        self.force_layout = extra.get("layout", "auto")  # auto | sem_split | raw

        # Pre-build a compressor; the decompressor is stateless enough to share.
        self._cctx = zstd.ZstdCompressor(level=self.zstd_level)
        self._dctx = zstd.ZstdDecompressor()
        logger.info(
            "CompressedHiCacheFile ready at %s (zstd_level=%d, layout=%s)",
            self.file_path,
            self.zstd_level,
            self.force_layout,
        )

        # Lightweight aggregate stats — useful for ablation runs.
        self._bytes_written_orig = 0
        self._bytes_written_comp = 0
        self._bytes_read_orig = 0
        self._bytes_read_comp = 0

    # ------------------------------------------------------------------ get

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            with open(tensor_path, "rb", buffering=0) as f:
                blob = f.read()
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from CompressedHiCacheFile storage.")
            return None

        if len(blob) < _HEADER_LEN:
            logger.error(f"Compressed page {key} truncated ({len(blob)} bytes)")
            return None

        layout_kind, dtype_kind, _zstd_level, orig_bytes = _unpack_header(blob)
        payload = self._dctx.decompress(blob[_HEADER_LEN:])
        if len(payload) != _expected_payload_size(layout_kind, orig_bytes, target_location):
            logger.error(
                "Compressed page %s payload size mismatch: header.orig=%d, decompressed_payload=%d",
                key,
                orig_bytes,
                len(payload),
            )
            return None

        # Reconstruct raw byte stream of length orig_bytes.
        if layout_kind == _LK_SEM_SPLIT:
            n_values = orig_bytes // 2
            raw_bytes = _sem_unsplit_bf16_like(payload, n_values)
        elif layout_kind == _LK_RAW:
            raw_bytes = payload
        else:
            logger.error(f"Unsupported layout_kind {layout_kind} for key {key}")
            return None

        if len(raw_bytes) != orig_bytes:
            logger.error(
                "Decompressed raw_bytes length %d != header.orig_bytes %d for key %s",
                len(raw_bytes),
                orig_bytes,
                key,
            )
            return None

        expected = target_location.numel() * target_location.element_size()
        if expected != orig_bytes:
            logger.error(
                "target_location bytes %d != stored orig_bytes %d for key %s",
                expected,
                orig_bytes,
                key,
            )
            return None

        # numpy memoryview slice-assign doesn't support ndim > 1, so flatten the
        # uint8 view first. The underlying storage is shared with target_location
        # because we asked for a contiguous view, so this is still in-place.
        arr_flat = target_location.view(torch.uint8).contiguous().numpy().reshape(-1)
        arr_flat[:] = np.frombuffer(raw_bytes, dtype=np.uint8)

        self._bytes_read_orig += orig_bytes
        self._bytes_read_comp += len(blob) - _HEADER_LEN
        return target_location

    # ------------------------------------------------------------------ set

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if value is None:
            logger.error("CompressedHiCacheFile.set called with value=None")
            return False
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")

        layout_kind, dtype_kind = _choose_layout(value.dtype)
        if self.force_layout == "raw":
            layout_kind = _LK_RAW
        elif self.force_layout == "sem_split" and value.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            layout_kind = _LK_SEM_SPLIT

        raw_bytes = value.contiguous().view(torch.uint8).numpy().tobytes()
        orig_bytes = len(raw_bytes)

        if layout_kind == _LK_SEM_SPLIT:
            payload = _sem_split_bf16_like(raw_bytes)
        else:
            payload = raw_bytes

        comp = self._cctx.compress(payload)
        header = _pack_header(layout_kind, dtype_kind, self.zstd_level, orig_bytes)

        try:
            tmp_path = tensor_path + ".tmp"
            with open(tmp_path, "wb", buffering=0) as f:
                f.write(header)
                f.write(comp)
            os.replace(tmp_path, tensor_path)
        except Exception as e:
            logger.error(f"Failed to save compressed tensor {key}: {e}")
            return False

        self._bytes_written_orig += orig_bytes
        self._bytes_written_comp += len(comp)
        return True

    # ------------------------------------------------------------------ batch / exists are inherited

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [self.get(k, t) for k, t in zip(keys, target_locations or [None] * len(keys))]

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for k, v in zip(keys, values):
            if not self.set(k, v):
                return False
        return True

    # ------------------------------------------------------------------ stats

    def get_stats(self) -> dict:
        ratio_w = (self._bytes_written_orig / self._bytes_written_comp) if self._bytes_written_comp else 0.0
        ratio_r = (self._bytes_read_orig / self._bytes_read_comp) if self._bytes_read_comp else 0.0
        return {
            "bytes_written_orig": self._bytes_written_orig,
            "bytes_written_comp": self._bytes_written_comp,
            "write_ratio": ratio_w,
            "bytes_read_orig": self._bytes_read_orig,
            "bytes_read_comp": self._bytes_read_comp,
            "read_ratio": ratio_r,
            "zstd_level": self.zstd_level,
            "layout": self.force_layout,
        }


def _expected_payload_size(layout_kind: int, orig_bytes: int, target_location: torch.Tensor) -> int:
    """Expected length of the *decompressed* (post-zstd) payload for a given layout."""
    if layout_kind == _LK_SEM_SPLIT:
        n_values = orig_bytes // 2
        sign_bytes = (n_values + 7) // 8
        return sign_bytes + 2 * n_values  # sign_packed + exp + mant
    return orig_bytes
