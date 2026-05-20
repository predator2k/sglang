# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Pluggable layout transforms for CompressedHiCacheFile.

A *layout* is a lossless byte-stream reshaping applied before the codec runs
(and inverted after the codec decodes). Picking the right layout is half of
the compression-ratio story — see the standalone ``kvcache_comp`` ablation
where the choice of layout moves the ratio from 1.27x (raw) to 1.61x
(sem_split_channel) on the same zstd codec.

Each layout implements::

    encode(value: torch.Tensor, channels: int | None) -> bytes
    decode(payload: bytes, target_flat: np.ndarray,
           dtype: torch.dtype, numel: int, channels: int) -> None

Encoded byte stream is the input to the codec. Decoded bytes are written
in-place into ``target_flat`` (a ravel'd uint8 view of the destination
tensor).

Layouts (file-format byte 9):

    0  RAW                        original tensor bytes, no transform
    1  CHANNEL_MAJOR              permutation only (needs channels)
    2  BYTE_HI_LO                 split BF16/FP16 into hi/lo byte streams
    3  BYTE_HI_LO_CHANNEL         BYTE_HI_LO + channel-major
    4  SEM_SPLIT                  sign / 8-bit exp / 7-bit mant streams
    5  SEM_SPLIT_CHANNEL          SEM_SPLIT + channel-major  (recommended)
    6  ZIPNN_CHANNEL_DELTA        SEM_SPLIT_CHANNEL + per-channel exp delta
    7  BIT_PLANE                  16 bit-planes, channel-major within each plane
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

LK_RAW = 0
LK_CHANNEL_MAJOR = 1
LK_BYTE_HI_LO = 2
LK_BYTE_HI_LO_CHANNEL = 3
LK_SEM_SPLIT = 4
LK_SEM_SPLIT_CHANNEL = 5
LK_ZIPNN_CHANNEL_DELTA = 6
LK_BIT_PLANE = 7

LK_NAMES = {
    LK_RAW: "raw",
    LK_CHANNEL_MAJOR: "channel_major",
    LK_BYTE_HI_LO: "byte_hi_lo",
    LK_BYTE_HI_LO_CHANNEL: "byte_hi_lo_channel",
    LK_SEM_SPLIT: "sem_split",
    LK_SEM_SPLIT_CHANNEL: "sem_split_channel",
    LK_ZIPNN_CHANNEL_DELTA: "zipnn_channel_delta",
    LK_BIT_PLANE: "bit_plane",
}
NAME_TO_LK = {v: k for k, v in LK_NAMES.items()}

# Which layouts strictly require BF16/FP16-shaped data.
_NEEDS_BF16_LIKE = {
    LK_BYTE_HI_LO,
    LK_BYTE_HI_LO_CHANNEL,
    LK_SEM_SPLIT,
    LK_SEM_SPLIT_CHANNEL,
    LK_ZIPNN_CHANNEL_DELTA,
    LK_BIT_PLANE,
}

# Which layouts use the channel count.
_NEEDS_CHANNELS = {
    LK_CHANNEL_MAJOR,
    LK_BYTE_HI_LO_CHANNEL,
    LK_SEM_SPLIT_CHANNEL,
    LK_ZIPNN_CHANNEL_DELTA,
    LK_BIT_PLANE,
}


def _to_u16_flat(value: torch.Tensor) -> np.ndarray:
    return value.contiguous().view(torch.uint8).numpy().view(np.uint16).ravel().astype(np.uint16)


def _sem_split_u16(u16: np.ndarray) -> bytes:
    sign = ((u16 >> 15) & 1).astype(np.uint8)
    exp = ((u16 >> 7) & 0xFF).astype(np.uint8)
    mant = (u16 & 0x7F).astype(np.uint8)
    return (
        np.packbits(sign, bitorder="little").tobytes()
        + exp.tobytes()
        + mant.tobytes()
    )


def _sem_unsplit(buf: bytes, n: int) -> np.ndarray:
    n_sign_bytes = (n + 7) // 8
    sign = np.unpackbits(
        np.frombuffer(buf[:n_sign_bytes], dtype=np.uint8), bitorder="little"
    )[:n]
    exp = np.frombuffer(buf[n_sign_bytes : n_sign_bytes + n], dtype=np.uint8)
    mant = np.frombuffer(
        buf[n_sign_bytes + n : n_sign_bytes + 2 * n], dtype=np.uint8
    )
    return (
        (sign.astype(np.uint16) << 15)
        | (exp.astype(np.uint16) << 7)
        | mant.astype(np.uint16)
    )


# ---------------------------------------------------------------------------
# Per-layout encode / decode functions
# ---------------------------------------------------------------------------


def encode(kind: int, value: torch.Tensor, channels: Optional[int]) -> bytes:
    if kind in _NEEDS_BF16_LIKE and value.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"layout {LK_NAMES[kind]} requires BF16/FP16, got {value.dtype}"
        )
    if kind in _NEEDS_CHANNELS and (not channels or value.numel() % channels != 0):
        raise ValueError(
            f"layout {LK_NAMES[kind]} requires channels divisor; got channels={channels}, "
            f"numel={value.numel()}"
        )

    raw_u8 = value.contiguous().view(torch.uint8).numpy()

    if kind == LK_RAW:
        return raw_u8.tobytes()

    if kind == LK_CHANNEL_MAJOR:
        assert channels is not None
        # Flatten u8 then reinterpret as element-major matrix (seq, channels * 2 bytes/element)
        # For BF16: each element = 2 bytes. Permutation is on elements, not bytes.
        if value.dtype in (torch.bfloat16, torch.float16):
            u16 = raw_u8.view(np.uint16).ravel()
            seq = u16.size // channels
            permuted = u16.reshape(seq, channels).T.reshape(-1).copy()
            return permuted.view(np.uint8).tobytes()
        # Non-16-bit dtypes: permute uint8 directly.
        seq = raw_u8.size // (channels * value.element_size())
        elems = raw_u8.reshape(seq, channels, value.element_size())
        permuted = np.ascontiguousarray(elems.transpose(1, 0, 2))
        return permuted.tobytes()

    if kind == LK_BYTE_HI_LO:
        flat = raw_u8.reshape(-1, 2)
        return flat[:, 1].tobytes() + flat[:, 0].tobytes()

    if kind == LK_BYTE_HI_LO_CHANNEL:
        assert channels is not None
        u16 = raw_u8.view(np.uint16).ravel()
        seq = u16.size // channels
        permuted = u16.reshape(seq, channels).T.reshape(-1).copy()
        flat = permuted.view(np.uint8).reshape(-1, 2)
        return flat[:, 1].tobytes() + flat[:, 0].tobytes()

    if kind == LK_SEM_SPLIT:
        u16 = raw_u8.view(np.uint16).ravel()
        return _sem_split_u16(u16)

    if kind == LK_SEM_SPLIT_CHANNEL:
        assert channels is not None
        u16 = raw_u8.view(np.uint16).ravel()
        seq = u16.size // channels
        permuted = u16.reshape(seq, channels).T.reshape(-1).copy()
        return _sem_split_u16(permuted)

    if kind == LK_ZIPNN_CHANNEL_DELTA:
        assert channels is not None
        u16 = raw_u8.view(np.uint16).ravel()
        seq = u16.size // channels
        permuted_u16 = u16.reshape(seq, channels).T.reshape(channels, seq)  # (C, S)
        sign = ((permuted_u16 >> 15) & 1).astype(np.uint8)
        exp = ((permuted_u16 >> 7) & 0xFF).astype(np.int32)
        mant = (permuted_u16 & 0x7F).astype(np.uint8)
        beta = np.round(exp.mean(axis=1)).clip(0, 255).astype(np.uint8)
        delta = (exp - beta[:, None]).clip(-128, 127).astype(np.int8)
        sign_packed = np.packbits(sign.ravel(), bitorder="little")
        return (
            beta.tobytes()
            + sign_packed.tobytes()
            + delta.ravel().tobytes()
            + mant.ravel().tobytes()
        )

    if kind == LK_BIT_PLANE:
        assert channels is not None
        u16 = raw_u8.view(np.uint16).ravel()
        seq = u16.size // channels
        permuted = u16.reshape(seq, channels).T.reshape(-1).copy()  # channel-major
        n = permuted.size
        pad = (-n) % 8
        out = []
        for bit in range(16):
            plane = ((permuted >> bit) & 1).astype(np.uint8)
            if pad:
                plane = np.concatenate([plane, np.zeros(pad, dtype=np.uint8)])
            out.append(np.packbits(plane, bitorder="little").tobytes())
        return b"".join(out)

    raise ValueError(f"unknown layout kind {kind}")


def decode(
    kind: int,
    payload: bytes,
    target_flat: np.ndarray,  # uint8 view, contiguous, ravel'd
    dtype: torch.dtype,
    numel: int,
    channels: Optional[int],
) -> bool:
    """Fill target_flat in-place from payload according to layout kind.
    Returns True on success, False on size/shape mismatch (caller logs)."""
    orig_bytes = numel * (2 if dtype in (torch.bfloat16, torch.float16) else
                          torch.empty(0, dtype=dtype).element_size())

    if kind == LK_RAW:
        if len(payload) != orig_bytes:
            return False
        target_flat[:] = np.frombuffer(payload, dtype=np.uint8)
        return True

    if kind == LK_CHANNEL_MAJOR:
        if not channels or len(payload) != orig_bytes:
            return False
        if dtype in (torch.bfloat16, torch.float16):
            u16_perm = np.frombuffer(payload, dtype=np.uint16)
            seq = u16_perm.size // channels
            u16_orig = u16_perm.reshape(channels, seq).T.reshape(-1).copy()
            target_flat[:] = u16_orig.view(np.uint8)
        else:
            esz = torch.empty(0, dtype=dtype).element_size()
            seq = numel // channels
            elems = np.frombuffer(payload, dtype=np.uint8).reshape(channels, seq, esz)
            orig = np.ascontiguousarray(elems.transpose(1, 0, 2)).reshape(-1)
            target_flat[:] = orig
        return True

    if kind == LK_BYTE_HI_LO:
        if len(payload) != orig_bytes:
            return False
        n = numel
        hi = np.frombuffer(payload[:n], dtype=np.uint8)
        lo = np.frombuffer(payload[n : 2 * n], dtype=np.uint8)
        out = np.empty((n, 2), dtype=np.uint8)
        out[:, 0] = lo
        out[:, 1] = hi
        target_flat[:] = out.ravel()
        return True

    if kind == LK_BYTE_HI_LO_CHANNEL:
        if not channels or len(payload) != orig_bytes:
            return False
        n = numel
        hi = np.frombuffer(payload[:n], dtype=np.uint8)
        lo = np.frombuffer(payload[n : 2 * n], dtype=np.uint8)
        permuted_u8 = np.empty((n, 2), dtype=np.uint8)
        permuted_u8[:, 0] = lo
        permuted_u8[:, 1] = hi
        u16_perm = permuted_u8.view(np.uint16).ravel()
        seq = u16_perm.size // channels
        u16_orig = u16_perm.reshape(channels, seq).T.reshape(-1).copy()
        target_flat[:] = u16_orig.view(np.uint8)
        return True

    if kind == LK_SEM_SPLIT:
        n = numel
        expected = (n + 7) // 8 + 2 * n
        if len(payload) != expected:
            return False
        u16 = _sem_unsplit(payload, n)
        target_flat[:] = u16.view(np.uint8)
        return True

    if kind == LK_SEM_SPLIT_CHANNEL:
        if not channels:
            return False
        n = numel
        expected = (n + 7) // 8 + 2 * n
        if len(payload) != expected:
            return False
        u16_perm = _sem_unsplit(payload, n)
        seq = n // channels
        u16_orig = u16_perm.reshape(channels, seq).T.reshape(-1).copy()
        target_flat[:] = u16_orig.view(np.uint8)
        return True

    if kind == LK_ZIPNN_CHANNEL_DELTA:
        if not channels:
            return False
        n = numel
        # bytes: channels (beta) + n/8 (sign) + n (delta int8) + n (mant)
        expected = channels + (n + 7) // 8 + n + n
        if len(payload) != expected:
            return False
        off = 0
        beta = np.frombuffer(payload[off : off + channels], dtype=np.uint8)
        off += channels
        n_sign_bytes = (n + 7) // 8
        sign = np.unpackbits(
            np.frombuffer(payload[off : off + n_sign_bytes], dtype=np.uint8),
            bitorder="little",
        )[:n]
        off += n_sign_bytes
        delta = np.frombuffer(payload[off : off + n], dtype=np.int8).astype(np.int32)
        off += n
        mant = np.frombuffer(payload[off : off + n], dtype=np.uint8)
        seq = n // channels
        exp = (delta.reshape(channels, seq) + beta[:, None].astype(np.int32)).clip(0, 255).astype(np.uint16)
        sign_2d = sign.reshape(channels, seq).astype(np.uint16)
        mant_2d = mant.reshape(channels, seq).astype(np.uint16)
        u16_perm = (sign_2d << 15) | (exp << 7) | mant_2d
        u16_orig = u16_perm.T.reshape(-1).copy()
        target_flat[:] = u16_orig.view(np.uint8)
        return True

    if kind == LK_BIT_PLANE:
        if not channels:
            return False
        n = numel
        pad = (-n) % 8
        plane_bytes = (n + pad) // 8
        expected = plane_bytes * 16
        if len(payload) != expected:
            return False
        u16 = np.zeros(n, dtype=np.uint16)
        for bit in range(16):
            packed = np.frombuffer(
                payload[bit * plane_bytes : (bit + 1) * plane_bytes], dtype=np.uint8
            )
            plane = np.unpackbits(packed, bitorder="little")[:n]
            u16 |= plane.astype(np.uint16) << bit
        # u16 is in channel-major order; invert
        seq = n // channels
        u16_orig = u16.reshape(channels, seq).T.reshape(-1).copy()
        target_flat[:] = u16_orig.view(np.uint8)
        return True

    return False


def expected_payload_size(kind: int, numel: int, channels: Optional[int], dtype: torch.dtype) -> int:
    """Predicted size of the (uncompressed) layout payload — used for sanity checks."""
    esz = 2 if dtype in (torch.bfloat16, torch.float16) else torch.empty(0, dtype=dtype).element_size()
    if kind == LK_RAW or kind == LK_CHANNEL_MAJOR:
        return numel * esz
    if kind in (LK_BYTE_HI_LO, LK_BYTE_HI_LO_CHANNEL):
        return numel * 2
    if kind in (LK_SEM_SPLIT, LK_SEM_SPLIT_CHANNEL):
        return (numel + 7) // 8 + 2 * numel
    if kind == LK_ZIPNN_CHANNEL_DELTA:
        assert channels
        return channels + (numel + 7) // 8 + 2 * numel
    if kind == LK_BIT_PLANE:
        return ((numel + 7) // 8) * 16
    raise ValueError(f"unknown layout kind {kind}")


def auto_pick(dtype: torch.dtype, numel: int, channels: Optional[int]) -> int:
    """Pick the best layout for the given dtype/shape.

    Order of preference (matches the ablation):
    sem_split_channel > sem_split > raw.
    """
    if dtype in (torch.bfloat16, torch.float16):
        if channels and numel % channels == 0 and numel // channels > 1:
            return LK_SEM_SPLIT_CHANNEL
        return LK_SEM_SPLIT
    return LK_RAW
