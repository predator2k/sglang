# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Pluggable codecs for CompressedHiCacheFile.

This module isolates the compression engine from the layout transform.
The layout (sem_split / sem_split_channel / raw) is decided by
``hicache_compressed.py``; the codec just deals with byte streams.

Each codec implements two methods::

    encode(payload: bytes) -> bytes
    decode(blob: bytes, expected_size: int,
           target: torch.Tensor | None = None,
           thread_pool: ThreadPoolExecutor | None = None) -> bytes | None

``expected_size`` is the *uncompressed* size; codecs use it for sanity checks
and may need it to allocate output buffers (e.g. nvcomp).

If a codec can natively write into a CUDA target tensor (nvcomp), it may do
so and return ``None`` after copying — the wrapper checks that case.
Otherwise it returns the decompressed bytes.

Available codec kinds (file-format byte 11):

    0  ZSTD                 (stock zstandard, single-frame)
    1  ZSTD_CHUNKED         (multi-frame zstd, intra-page parallelism)
    2  BLOSC2               (chunked, codec-of-choice inside)
    3  LZ4                  (lz4 frame format, fastest CPU decomp)
    4  ISAL_GZIP            (Intel ISA-L gzip, AVX-512 accelerated)
    5  NVCOMP               (GPU-side decompression via nvcomp)
"""

from __future__ import annotations

import logging
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

CK_ZSTD = 0
CK_ZSTD_CHUNKED = 1
CK_BLOSC2 = 2
CK_LZ4 = 3
CK_ISAL_GZIP = 4
CK_NVCOMP = 5

CK_NAMES = {
    CK_ZSTD: "zstd",
    CK_ZSTD_CHUNKED: "zstd_chunked",
    CK_BLOSC2: "blosc2",
    CK_LZ4: "lz4",
    CK_ISAL_GZIP: "isal_gzip",
    CK_NVCOMP: "nvcomp",
}
NAME_TO_CK = {v: k for k, v in CK_NAMES.items()}


# ---------------------------------------------------------------------------
# Codec base
# ---------------------------------------------------------------------------


class Codec:
    """Stateful codec; one instance per backend.

    Subclasses set ``kind`` and ``name``, and implement encode / decode.
    ``available`` short-circuits creation if the optional dep isn't installed.
    """

    kind: int = -1
    name: str = "abstract"
    available: bool = False
    decodes_to_gpu: bool = False

    def encode(self, payload: bytes) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def decode(
        self,
        blob: bytes,
        expected_size: int,
        target=None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ) -> Optional[bytes]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ZSTD — single-frame (stock)
# ---------------------------------------------------------------------------


class ZstdCodec(Codec):
    """Single-frame zstd. Compress threads available; decompress is 1 thread per frame."""

    kind = CK_ZSTD
    name = "zstd"
    available = True

    def __init__(self, level: int = 1, threads: int = 0) -> None:
        import zstandard as zstd

        self._zstd = zstd
        self.level = level
        self.threads = threads
        self._cctx = zstd.ZstdCompressor(level=level, threads=threads)
        self._dctx = zstd.ZstdDecompressor()

    def encode(self, payload: bytes) -> bytes:
        return self._cctx.compress(payload)

    def decode(self, blob, expected_size, target=None, thread_pool=None):
        out = self._dctx.decompress(blob)
        if len(out) != expected_size:
            logger.error(
                "zstd decode size mismatch: %d vs expected %d", len(out), expected_size
            )
            return None
        return out


# ---------------------------------------------------------------------------
# ZSTD_CHUNKED — multi-frame, intra-page parallelism (E)
# ---------------------------------------------------------------------------


class ZstdChunkedCodec(Codec):
    """Split each page into fixed-size chunks; each chunk is its own zstd frame.
    Decoding parallelizes across chunks with a ThreadPoolExecutor.

    Container format (after the outer page header)::

        4 bytes:  num_chunks (uint32 LE)
        for each chunk: 4 bytes uncomp_size + 4 bytes comp_size (uint32 LE each)
        concatenated compressed chunks (in order)
    """

    kind = CK_ZSTD_CHUNKED
    name = "zstd_chunked"
    available = True

    def __init__(
        self,
        level: int = 1,
        chunk_size: int = 64 * 1024,
        compress_threads: int = 1,
    ) -> None:
        import zstandard as zstd

        self._zstd = zstd
        self.level = level
        self.chunk_size = max(1024, chunk_size)
        # Pool used for parallel chunk compression at encode-time; decode-time
        # uses the caller-supplied pool so we share threads across the whole
        # batch_get path.
        self._enc_workers = max(1, compress_threads)
        self._cctx_proto = zstd.ZstdCompressor  # call per-thread to avoid contention
        self._dctx = zstd.ZstdDecompressor()

    def _encode_chunk(self, chunk: bytes) -> bytes:
        return self._cctx_proto(level=self.level).compress(chunk)

    def encode(self, payload: bytes) -> bytes:
        n = len(payload)
        cs = self.chunk_size
        chunks = [payload[i : i + cs] for i in range(0, n, cs)]

        if self._enc_workers > 1 and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=self._enc_workers) as pool:
                comp_chunks = list(pool.map(self._encode_chunk, chunks))
        else:
            comp_chunks = [self._encode_chunk(c) for c in chunks]

        header = struct.pack("<I", len(chunks))
        index = b"".join(
            struct.pack("<II", len(c), len(cc)) for c, cc in zip(chunks, comp_chunks)
        )
        return header + index + b"".join(comp_chunks)

    def decode(self, blob, expected_size, target=None, thread_pool=None):
        if len(blob) < 4:
            return None
        n_chunks = struct.unpack("<I", blob[:4])[0]
        idx_off = 4
        idx_size = 8 * n_chunks
        if len(blob) < idx_off + idx_size:
            logger.error("zstd_chunked truncated index")
            return None

        chunk_specs = []
        cum = idx_off + idx_size
        for i in range(n_chunks):
            uc, cc = struct.unpack("<II", blob[idx_off + 8 * i : idx_off + 8 * (i + 1)])
            chunk_specs.append((cum, cc, uc))
            cum += cc

        def _decode(spec):
            offset, comp_len, uc = spec
            out = self._dctx.decompress(blob[offset : offset + comp_len])
            if len(out) != uc:
                raise ValueError(f"chunked: chunk wrong size {len(out)} vs {uc}")
            return out

        if thread_pool is not None and n_chunks > 1:
            outputs = list(thread_pool.map(_decode, chunk_specs))
        else:
            outputs = [_decode(s) for s in chunk_specs]

        result = b"".join(outputs)
        if len(result) != expected_size:
            logger.error(
                "zstd_chunked total size %d != expected %d", len(result), expected_size
            )
            return None
        return result


# ---------------------------------------------------------------------------
# BLOSC2 — chunked, multi-thread, codec-of-choice inside (B)
# ---------------------------------------------------------------------------


class Blosc2Codec(Codec):
    kind = CK_BLOSC2
    name = "blosc2"

    def __init__(
        self,
        inner: str = "zstd",
        clevel: int = 1,
        nthreads: int = 0,
        blocksize: int = 0,
    ) -> None:
        try:
            import blosc2

            self.available = True
        except ImportError:
            self.available = False
            return
        self._blosc2 = blosc2
        if nthreads == 0:
            nthreads = max(1, os.cpu_count() or 1)
        self._nthreads = nthreads
        self._clevel = clevel
        self._blocksize = blocksize  # 0 = let blosc2 pick
        codec_name = inner.upper()
        if codec_name not in blosc2.Codec.__members__:
            raise ValueError(
                f"blosc2 inner codec {inner!r} not found in {list(blosc2.Codec.__members__)}"
            )
        self._inner = blosc2.Codec[codec_name]

    def encode(self, payload: bytes) -> bytes:
        self._blosc2.set_nthreads(self._nthreads)
        return self._blosc2.compress2(
            payload,
            clevel=self._clevel,
            codec=self._inner,
            blocksize=self._blocksize,
        )

    def decode(self, blob, expected_size, target=None, thread_pool=None):
        self._blosc2.set_nthreads(self._nthreads)
        out = self._blosc2.decompress2(blob)
        if len(out) != expected_size:
            logger.error(
                "blosc2 decode size mismatch: %d vs %d", len(out), expected_size
            )
            return None
        return bytes(out)


# ---------------------------------------------------------------------------
# LZ4 — frame format, single-thread but very fast (C)
# ---------------------------------------------------------------------------


class Lz4Codec(Codec):
    kind = CK_LZ4
    name = "lz4"

    def __init__(self, use_hc: bool = False, level: int = 0) -> None:
        try:
            import lz4.frame as lz4f

            self.available = True
        except ImportError:
            self.available = False
            return
        self._lz4f = lz4f
        self._level = level
        # level mapping: lz4-default = 0, lz4-hc levels 3..12
        self._use_hc = use_hc

    def encode(self, payload: bytes) -> bytes:
        return self._lz4f.compress(payload, compression_level=self._level)

    def decode(self, blob, expected_size, target=None, thread_pool=None):
        out = self._lz4f.decompress(blob)
        if len(out) != expected_size:
            logger.error("lz4 decode size mismatch: %d vs %d", len(out), expected_size)
            return None
        return out


# ---------------------------------------------------------------------------
# ISA-L gzip — Intel AVX-512 accelerated (D)
# ---------------------------------------------------------------------------


class IsalGzipCodec(Codec):
    kind = CK_ISAL_GZIP
    name = "isal_gzip"

    def __init__(self, level: int = 1) -> None:
        try:
            from isal import isal_zlib

            self.available = True
        except ImportError:
            self.available = False
            return
        # ISA-L's level range is 0..3 (mapped). Clamp:
        self._level = max(0, min(3, int(level)))
        self._isal_zlib = isal_zlib

    def encode(self, payload: bytes) -> bytes:
        # gzip-format payload (wbits=31)
        return self._isal_zlib.compress(payload, self._level, wbits=31)

    def decode(self, blob, expected_size, target=None, thread_pool=None):
        out = self._isal_zlib.decompress(blob, wbits=31)
        if len(out) != expected_size:
            logger.error("isal_gzip decode size mismatch: %d vs %d", len(out), expected_size)
            return None
        return out


# ---------------------------------------------------------------------------
# nvcomp — GPU-side compression / decompression (F)
# ---------------------------------------------------------------------------


class NvCompCodec(Codec):
    """nvcomp 5.x codec. Compresses on host (host array → host bytes), decompresses
    on GPU. If the target tensor is on CUDA, we decompress directly into it (
    via nvcomp.Array → torch dlpack import). Otherwise we decompress on GPU and
    copy back to host.
    """

    kind = CK_NVCOMP
    name = "nvcomp"
    decodes_to_gpu = True

    def __init__(
        self,
        algorithm: str = "Zstd",
        device_id: int = 0,
        uncomp_chunk_size: int = 65536,
    ) -> None:
        try:
            from nvidia import nvcomp

            self.available = True
        except ImportError:
            self.available = False
            return
        self._nvcomp = nvcomp
        self._algo = algorithm
        self._device_id = device_id
        self._chunk = uncomp_chunk_size
        # codec used for both encode (host) and decode (host or device backend)
        self._codec = nvcomp.Codec(
            algorithm=algorithm,
            device_id=device_id,
            uncomp_chunk_size=uncomp_chunk_size,
        )

    def encode(self, payload: bytes) -> bytes:
        import numpy as np

        nvc = self._nvcomp
        arr = nvc.as_array(np.frombuffer(payload, dtype=np.uint8))
        encoded = self._codec.encode(arr)
        # encoded is an nvcomp.Array — copy to host bytes
        return bytes(encoded.cpu())

    def decode(self, blob, expected_size, target=None, thread_pool=None):
        import numpy as np
        import torch

        nvc = self._nvcomp
        # Push compressed bytes to a host-side nvcomp.Array; nvcomp moves them to
        # device internally and runs the GPU decode.
        arr = nvc.as_array(np.frombuffer(blob, dtype=np.uint8))
        decoded = self._codec.decode(arr)
        decoded_host = bytes(decoded.cpu())
        if len(decoded_host) != expected_size:
            logger.error(
                "nvcomp decode size mismatch: %d vs %d", len(decoded_host), expected_size
            )
            return None
        return decoded_host


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_codec(name: str, extra: dict) -> Codec:
    """Construct a codec by name, pulling its parameters out of ``extra``."""
    level = int(extra.get("compression_level", 1))
    if name == "zstd":
        return ZstdCodec(level=level, threads=int(extra.get("zstd_threads", 0)))
    if name == "zstd_chunked":
        return ZstdChunkedCodec(
            level=level,
            chunk_size=int(extra.get("chunk_size_kb", 64)) * 1024,
            compress_threads=int(extra.get("chunk_compress_threads", 1)),
        )
    if name == "blosc2":
        c = Blosc2Codec(
            inner=extra.get("blosc2_inner_codec", "zstd"),
            clevel=level,
            nthreads=int(extra.get("blosc2_nthreads", 0)),
            blocksize=int(extra.get("blosc2_blocksize", 0)),
        )
        if not c.available:
            raise ImportError("blosc2 not installed; pip install blosc2")
        return c
    if name == "lz4":
        c = Lz4Codec(use_hc=bool(extra.get("lz4_use_hc", False)), level=level)
        if not c.available:
            raise ImportError("python-lz4 not installed; pip install lz4")
        return c
    if name == "isal_gzip":
        c = IsalGzipCodec(level=level)
        if not c.available:
            raise ImportError("isal not installed; pip install isal")
        return c
    if name == "nvcomp":
        c = NvCompCodec(
            algorithm=extra.get("nvcomp_algorithm", "Zstd"),
            device_id=int(extra.get("nvcomp_device_id", 0)),
            uncomp_chunk_size=int(extra.get("nvcomp_chunk_size", 65536)),
        )
        if not c.available:
            raise ImportError(
                "nvcomp not installed; pip install nvidia-nvcomp-cu12"
            )
        return c
    raise ValueError(
        f"compressed_file: unknown codec {name!r}; "
        f"expected one of {list(NAME_TO_CK.keys())}"
    )
