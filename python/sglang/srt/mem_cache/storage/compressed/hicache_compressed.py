# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Compressed file backend for HiCache (KV cache prefix offload).

This module is the orchestrator. Three pluggable pieces::

    layouts.py — lossless byte-stream reshaping (sem_split_channel, etc.).
    codecs.py  — compression engine (zstd/blosc2/lz4/isal_gzip/nvcomp/...).
    policy.py  — per-layer routing (via YAML or inline dict).

For a HiCache page that contains multiple layers (the typical SGLang case
where ``host_pool.get_data_page`` returns the full ``(2, L, P, H, D)``
slab for one page slot), we *can* compress each (layer, K|V) sub-slab with
its own codec + layout + level. The policy module picks one bundle per
slab; the file format records the per-slab choice so reads are self-
describing.

If no policy is configured, the whole page is treated as a single slab and
compressed with the global codec/layout defaults — that's the simple v1/v2
path.

File format (v3)::

    bytes 0..7    magic = b"SGLKVCMP"
    byte  8       version = 3
    byte  9       flags (reserved, 0)
    bytes 10..11  reserved
    bytes 12..15  num_slabs (uint32 LE)
    bytes 16..23  orig_bytes_total (uint64 LE; sum of all slab orig sizes)
    bytes 24..31  reserved

    For each slab (24 bytes):
        byte 0     layout_kind
        byte 1     codec_kind
        byte 2     dtype_kind
        byte 3     reserved
        bytes 4..7 channels (uint32 LE)
        bytes 8..15 slab_orig_bytes (uint64 LE)
        bytes 16..23 slab_comp_bytes (uint64 LE)

    Concatenated codec payloads in slab order.

Switches (``HiCacheStorageConfig.extra_config``):

    Top-level defaults (used as the "default profile")
    --------------------------------------------------
    codec                   "zstd"|"zstd_chunked"|"blosc2"|"lz4"|"isal_gzip"|"nvcomp"
    compression_level       codec-dependent (1..22 for zstd; 0..16 for lz4; 0..3 for isal)
    layout                  "auto"|"raw"|"channel_major"|"byte_hi_lo"
                              |"byte_hi_lo_channel"|"sem_split"|"sem_split_channel"
                              |"zipnn_channel_delta"|"bit_plane"
    channels                int override for (head_num * head_dim)
    strict_dtype            bool (default True)

    Codec-specific
    --------------
    zstd_threads, chunk_size_kb, chunk_compress_threads,
    blosc2_inner_codec, blosc2_nthreads, blosc2_blocksize,
    lz4_use_hc, nvcomp_algorithm, nvcomp_device_id, nvcomp_chunk_size

    Parallelism
    -----------
    batch_threads           Python ThreadPool for batch_get/batch_set
    slab_threads            ThreadPool for *intra-page* slab compress/decompress
                            (default 1 = serial, follow batch_threads)

    Per-layer policy
    ----------------
    profiles_yaml           Path to a YAML file (see policy.py docstring)
    profiles                Inline dict with the same schema as the YAML

A page is split into multiple slabs only if a policy is set AND the registered
host pool exposes layer_num + page_size + head_num + head_dim.
"""

from __future__ import annotations

import logging
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage.compressed import codecs, layouts
from sglang.srt.mem_cache.storage.compressed.policy import (
    KV_K,
    KV_V,
    LayeredPolicy,
    Profile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File format (v3)
# ---------------------------------------------------------------------------

_MAGIC = b"SGLKVCMP"
_VERSION = 3
_PAGE_HDR_FMT = "<8sBB2sIQ8s"  # 8 + 2 + 2 + 4 + 8 + 8 = 32
_PAGE_HDR_LEN = struct.calcsize(_PAGE_HDR_FMT)
assert _PAGE_HDR_LEN == 32

_SLAB_HDR_FMT = "<BBBBIQQ"  # 1+1+1+1 + 4 + 8 + 8 = 24
_SLAB_HDR_LEN = struct.calcsize(_SLAB_HDR_FMT)
assert _SLAB_HDR_LEN == 24

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

_DTKIND_TO_TORCH = {v: k for k, v in _TORCH_TO_DTKIND.items()}


def _pack_page_hdr(num_slabs: int, orig_total: int) -> bytes:
    return struct.pack(
        _PAGE_HDR_FMT,
        _MAGIC,
        _VERSION,
        0,  # flags
        b"\x00\x00",  # reserved
        num_slabs,
        orig_total,
        b"\x00" * 8,
    )


def _unpack_page_hdr(buf: bytes) -> Tuple[int, int]:
    magic, version, _flags, _r1, num_slabs, orig_total, _r2 = struct.unpack(
        _PAGE_HDR_FMT, buf[:_PAGE_HDR_LEN]
    )
    if magic != _MAGIC:
        raise ValueError(f"bad magic in compressed page: {magic!r}")
    if version != _VERSION:
        raise ValueError(
            f"unsupported compressed-page version {version}; expected {_VERSION}. "
            "Delete old cache files and re-create them."
        )
    return num_slabs, orig_total


def _pack_slab_hdr(
    layout_kind: int, codec_kind: int, dtype_kind: int, channels: int,
    slab_orig: int, slab_comp: int,
) -> bytes:
    return struct.pack(
        _SLAB_HDR_FMT, layout_kind, codec_kind, dtype_kind, 0, channels, slab_orig, slab_comp
    )


def _unpack_slab_hdr(buf: bytes) -> Tuple[int, int, int, int, int, int]:
    lk, ck, dt, _r, ch, sorig, scomp = struct.unpack(_SLAB_HDR_FMT, buf[:_SLAB_HDR_LEN])
    return lk, ck, dt, ch, sorig, scomp


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class CompressedHiCacheFile(HiCacheFile):
    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache_compressed",
    ) -> None:
        super().__init__(storage_config, file_path=file_path)
        extra = dict(storage_config.extra_config or {})

        # ---- default profile (used when no policy / for single-slab) ----
        default_codec = str(extra.get("codec", "zstd"))
        default_layout = str(extra.get("layout", "auto"))
        if default_layout not in ("auto", *layouts.NAME_TO_LK.keys()):
            raise ValueError(
                f"compressed_file: unknown layout {default_layout!r}; "
                f"expected 'auto' or one of {list(layouts.NAME_TO_LK)}"
            )
        default_profile = Profile(
            codec_name=default_codec,
            layout_name=default_layout,
            compression_level=int(extra.get("compression_level", 1)),
            extra={
                k: v
                for k, v in extra.items()
                if k
                not in {
                    "codec",
                    "layout",
                    "compression_level",
                    "channels",
                    "strict_dtype",
                    "batch_threads",
                    "slab_threads",
                    "profiles_yaml",
                    "profiles",
                }
            },
        )
        # Eagerly build the default codec so config errors fail fast.
        default_profile.get_codec()

        # ---- policy ----
        self.policy: Optional[LayeredPolicy] = None
        yaml_path = extra.get("profiles_yaml")
        inline_profiles = extra.get("profiles")
        if yaml_path:
            self.policy = LayeredPolicy.from_yaml(yaml_path)
            # Default in YAML wins over CLI default; if YAML has no default,
            # the constructor inside from_yaml falls back to zstd/auto/1.
        elif inline_profiles:
            self.policy = LayeredPolicy.from_dict(inline_profiles)
        else:
            # No policy: build a trivial one whose default == our default_profile.
            self.policy = LayeredPolicy(default=default_profile)

        # ---- channels / pool info ----
        self._channels_override: Optional[int] = (
            int(extra["channels"]) if extra.get("channels") else None
        )
        self._channels: Optional[int] = self._channels_override
        self._layer_num: Optional[int] = None
        self._page_size: Optional[int] = None
        self._slab_elem_count: Optional[int] = None  # P * H * D

        # ---- parallelism ----
        self.batch_threads = max(1, int(extra.get("batch_threads", 1)))
        self.slab_threads = max(1, int(extra.get("slab_threads", 1)))
        self.strict_dtype = bool(extra.get("strict_dtype", True))
        self._batch_pool: Optional[ThreadPoolExecutor] = None
        self._slab_pool: Optional[ThreadPoolExecutor] = None
        if self.batch_threads > 1:
            self._batch_pool = ThreadPoolExecutor(
                max_workers=self.batch_threads, thread_name_prefix="kvcomp-batch"
            )
        if self.slab_threads > 1:
            self._slab_pool = ThreadPoolExecutor(
                max_workers=self.slab_threads, thread_name_prefix="kvcomp-slab"
            )

        # ---- stats ----
        self._bytes_written_orig = 0
        self._bytes_written_comp = 0
        self._bytes_read_orig = 0
        self._bytes_read_comp = 0

        logger.info(
            "CompressedHiCacheFile: dir=%s default_codec=%s default_layout=%s "
            "policy_rules=%d batch_threads=%d slab_threads=%d strict_dtype=%s "
            "channels=%s",
            self.file_path,
            default_codec,
            default_layout,
            len(self.policy.rules),
            self.batch_threads,
            self.slab_threads,
            self.strict_dtype,
            self._channels,
        )
        if self.policy.rules:
            logger.info("CompressedHiCacheFile policy:\n%s", self.policy.describe())

    # ------------------------------------------------------------------ pool

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        super().register_mem_pool_host(mem_pool_host)
        self._capture_pool(mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name: str) -> None:
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        if host_pool_name in ("kv", "KV"):
            self._capture_pool(host_pool)

    def _capture_pool(self, pool: HostKVCache) -> None:
        head_num = getattr(pool, "head_num", None)
        head_dim = getattr(pool, "head_dim", None)
        layer_num = getattr(pool, "layer_num", None)
        page_size = getattr(pool, "page_size", None)
        if (
            isinstance(head_num, int)
            and isinstance(head_dim, int)
            and head_num * head_dim > 1
        ):
            if self._channels_override is None:
                self._channels = head_num * head_dim
        if isinstance(layer_num, int) and isinstance(page_size, int) and layer_num > 0:
            self._layer_num = layer_num
            self._page_size = page_size
            self._slab_elem_count = page_size * (head_num or 1) * (head_dim or 1)
        layout = getattr(pool, "layout", None)
        logger.info(
            "CompressedHiCacheFile: pool=%s layer_num=%s page_size=%s "
            "head_num=%s head_dim=%s pool_layout=%s -> channels=%s slab_elem=%s",
            type(pool).__name__,
            layer_num,
            page_size,
            head_num,
            head_dim,
            layout,
            self._channels,
            self._slab_elem_count,
        )

    # ------------------------------------------------------------------ slab partitioning

    def _use_multi_slab(self, value: torch.Tensor) -> bool:
        """Multi-slab if we have a policy with explicit rules AND pool geometry."""
        if not self.policy or not self.policy.rules:
            return False
        if self._layer_num is None or self._slab_elem_count is None:
            return False
        expected = 2 * self._layer_num * self._slab_elem_count
        if value.numel() != expected:
            logger.debug(
                "multi-slab disabled: numel %d != expected 2*%d*%d=%d",
                value.numel(),
                self._layer_num,
                self._slab_elem_count,
                expected,
            )
            return False
        return True

    def _slab_iter(self, value: torch.Tensor):
        """Yield (kv_kind, layer_idx, sub_tensor) for each slab in the page."""
        L = self._layer_num
        n_elem = self._slab_elem_count
        flat = value.contiguous().view(-1)
        for kv_kind in (KV_K, KV_V):
            for layer_idx in range(L):
                start = (kv_kind * L + layer_idx) * n_elem
                yield kv_kind, layer_idx, flat[start : start + n_elem]

    # ------------------------------------------------------------------ encode one slab

    def _encode_slab(self, slab: torch.Tensor, profile: Profile) -> tuple[int, int, int, int, bytes]:
        """Return (layout_kind, codec_kind, dtype_kind, channels, comp_bytes)."""
        layout_kind = self._resolve_layout(profile.layout_name, slab)
        channels = (
            self._channels
            if layout_kind in layouts._NEEDS_CHANNELS and self._channels
            else 0
        )
        payload = layouts.encode(layout_kind, slab, channels or None)
        comp = profile.get_codec().encode(payload)
        dt_kind = _TORCH_TO_DTKIND.get(slab.dtype, _DT_UNKNOWN)
        return layout_kind, profile.get_codec().kind, dt_kind, channels, comp

    def _resolve_layout(self, layout_name: str, value: torch.Tensor) -> int:
        if layout_name == "auto":
            return layouts.auto_pick(value.dtype, value.numel(), self._channels)
        kind = layouts.NAME_TO_LK[layout_name]
        # If channel-aware layout requested but we don't have channels, fall back.
        if kind in layouts._NEEDS_CHANNELS and (
            not self._channels or value.numel() % self._channels != 0
        ):
            return layouts.auto_pick(value.dtype, value.numel(), self._channels)
        if kind in layouts._NEEDS_BF16_LIKE and value.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            return layouts.LK_RAW
        return kind

    # ------------------------------------------------------------------ decode one slab

    def _decode_slab(
        self,
        slab_hdr: bytes,
        payload_bytes: bytes,
        target_arr_flat: np.ndarray,
        dtype: torch.dtype,
    ) -> bool:
        layout_kind, codec_kind, dt_kind, channels, slab_orig, slab_comp = _unpack_slab_hdr(slab_hdr)
        if self.strict_dtype and dt_kind != _DT_UNKNOWN:
            expected_dtype = _DTKIND_TO_TORCH.get(dt_kind)
            if expected_dtype is not None and expected_dtype != dtype:
                logger.error(
                    "slab dtype mismatch: stored=%s target=%s",
                    expected_dtype,
                    dtype,
                )
                return False
        codec_name = codecs.CK_NAMES.get(codec_kind)
        if codec_name is None:
            logger.error("slab codec_kind %d unknown", codec_kind)
            return False

        # Look up a codec instance with these settings. If the YAML policy
        # included a matching profile we already have a codec; otherwise build
        # a fresh one with default parameters.
        codec = self._find_codec_for_kind(codec_kind)

        numel = slab_orig // (2 if dtype in (torch.bfloat16, torch.float16) else
                              torch.empty(0, dtype=dtype).element_size())
        expected_payload = layouts.expected_payload_size(
            layout_kind, numel, channels or None, dtype
        )
        payload = codec.decode(
            payload_bytes,
            expected_payload,
            target=None,
            thread_pool=self._slab_pool,
        )
        if payload is None:
            return False
        return layouts.decode(
            layout_kind,
            payload,
            target_arr_flat,
            dtype,
            numel,
            channels or None,
        )

    def _find_codec_for_kind(self, codec_kind: int) -> codecs.Codec:
        """Find a codec instance compatible with the given kind, from the
        policy's profiles. Build a fresh default-config codec if none match."""
        for p in self.policy.all_profiles():
            if p.get_codec().kind == codec_kind:
                return p.get_codec()
        # Fallback: build with default kwargs.
        codec_name = codecs.CK_NAMES[codec_kind]
        return codecs.build_codec(codec_name, {"compression_level": 1})

    # ------------------------------------------------------------------ get

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        suffixed = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        try:
            with open(tensor_path, "rb", buffering=0) as f:
                blob = f.read()
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from CompressedHiCacheFile storage.")
            return None

        if len(blob) < _PAGE_HDR_LEN:
            logger.error(f"Compressed page {key} truncated ({len(blob)} bytes)")
            return None

        num_slabs, orig_total = _unpack_page_hdr(blob)
        if target_location.numel() * target_location.element_size() != orig_total:
            logger.error(
                "target_location bytes %d != stored orig_total %d for key %s",
                target_location.numel() * target_location.element_size(),
                orig_total,
                key,
            )
            return None

        idx_off = _PAGE_HDR_LEN
        slab_hdrs = []
        for i in range(num_slabs):
            slab_hdrs.append(blob[idx_off + i * _SLAB_HDR_LEN : idx_off + (i + 1) * _SLAB_HDR_LEN])
        payload_off = idx_off + num_slabs * _SLAB_HDR_LEN

        # Compute payload offsets
        slabs_meta = []
        cum_payload = payload_off
        cum_orig = 0
        for hdr in slab_hdrs:
            lk, ck, dt, ch, so, sc = _unpack_slab_hdr(hdr)
            slabs_meta.append((hdr, cum_payload, sc, cum_orig, so))
            cum_payload += sc
            cum_orig += so

        target_arr_flat = target_location.view(torch.uint8).contiguous().numpy().reshape(-1)

        def _do_one(idx):
            hdr, pay_off, pay_len, orig_off, orig_len = slabs_meta[idx]
            target_slice = target_arr_flat[orig_off : orig_off + orig_len]
            return self._decode_slab(
                hdr,
                blob[pay_off : pay_off + pay_len],
                target_slice,
                target_location.dtype,
            )

        if self._slab_pool is not None and num_slabs > 1:
            results = list(self._slab_pool.map(_do_one, range(num_slabs)))
        else:
            results = [_do_one(i) for i in range(num_slabs)]

        if not all(results):
            logger.error("one or more slab decodes failed for %s", key)
            return None

        self._bytes_read_orig += orig_total
        self._bytes_read_comp += len(blob) - _PAGE_HDR_LEN - num_slabs * _SLAB_HDR_LEN
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

        suffixed = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")

        # Decide single vs multi slab.
        slabs: List[tuple[int, int, torch.Tensor, Profile]] = []
        if self._use_multi_slab(value):
            for kv_kind, layer_idx, slab in self._slab_iter(value):
                profile = self.policy.resolve(kv_kind, layer_idx)
                slabs.append((kv_kind, layer_idx, slab, profile))
        else:
            # Single slab. Use the default (policy's default).
            slabs.append((-1, -1, value, self.policy.default))

        def _enc(idx):
            kv_kind, layer_idx, slab, profile = slabs[idx]
            lk, ck, dt, ch, comp = self._encode_slab(slab, profile)
            slab_orig = slab.numel() * slab.element_size()
            return lk, ck, dt, ch, slab_orig, comp

        if self._slab_pool is not None and len(slabs) > 1:
            encoded = list(self._slab_pool.map(_enc, range(len(slabs))))
        else:
            encoded = [_enc(i) for i in range(len(slabs))]

        orig_total = sum(s[4] for s in encoded)
        page_hdr = _pack_page_hdr(len(encoded), orig_total)
        slab_hdr_blob = b"".join(
            _pack_slab_hdr(lk, ck, dt, ch, so, len(comp))
            for lk, ck, dt, ch, so, comp in encoded
        )
        payloads_blob = b"".join(comp for _, _, _, _, _, comp in encoded)

        try:
            tmp_path = tensor_path + ".tmp"
            with open(tmp_path, "wb", buffering=0) as f:
                f.write(page_hdr)
                f.write(slab_hdr_blob)
                f.write(payloads_blob)
            os.replace(tmp_path, tensor_path)
        except Exception as e:
            logger.error(f"Failed to save compressed tensor {key}: {e}")
            return False

        self._bytes_written_orig += orig_total
        self._bytes_written_comp += len(payloads_blob)
        return True

    # ------------------------------------------------------------------ batch

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        targets = target_locations or [None] * len(keys)
        if self._batch_pool is None or len(keys) <= 1:
            return [self.get(k, t) for k, t in zip(keys, targets)]
        futures = [self._batch_pool.submit(self.get, k, t) for k, t in zip(keys, targets)]
        return [f.result() for f in futures]

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self._batch_pool is None or len(keys) <= 1:
            for k, v in zip(keys, values):
                if not self.set(k, v):
                    return False
            return True
        futures = [self._batch_pool.submit(self.set, k, v) for k, v in zip(keys, values)]
        return all(f.result() for f in futures)

    # ------------------------------------------------------------------ stats

    def get_stats(self) -> dict:
        ratio_w = (
            self._bytes_written_orig / self._bytes_written_comp
            if self._bytes_written_comp
            else 0.0
        )
        ratio_r = (
            self._bytes_read_orig / self._bytes_read_comp if self._bytes_read_comp else 0.0
        )
        return {
            "default_codec": self.policy.default.codec_name if self.policy else None,
            "default_layout": self.policy.default.layout_name if self.policy else None,
            "policy_rules": len(self.policy.rules) if self.policy else 0,
            "channels": self._channels,
            "layer_num": self._layer_num,
            "page_size": self._page_size,
            "batch_threads": self.batch_threads,
            "slab_threads": self.slab_threads,
            "bytes_written_orig": self._bytes_written_orig,
            "bytes_written_comp": self._bytes_written_comp,
            "write_ratio": ratio_w,
            "bytes_read_orig": self._bytes_read_orig,
            "bytes_read_comp": self._bytes_read_comp,
            "read_ratio": ratio_r,
        }
