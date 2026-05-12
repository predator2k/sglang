"""Dummy KV cache and model placeholders for the Tenstorrent backend.

Tenstorrent inference runs through ttnn / tt_transformers, which manage their
own KV cache on-device. SGLang's scheduler still expects a `KVCache` and a
model object with a `forward` attribute, so we provide zero-allocation stubs.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.mem_cache.memory_pool import KVCache


class _DummyKVCache(KVCache):
    """A KV cache that allocates no GPU memory.

    Satisfies the KVCache interface so that TokenToKVPoolAllocator can be
    constructed, but every buffer access raises — the Tenstorrent backend
    manages its own KV cache through ttnn / tt_transformers.
    """

    def __init__(self, size: int, dtype: torch.dtype, device: str):
        # Bypass KVCache.__init__ to avoid custom_mem_pool / memory_saver
        # initialization that may touch CUDA APIs.
        self.size = size
        self.page_size = 1
        self.dtype = dtype
        self.store_dtype = dtype
        self.device = device
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.mem_usage = 0
        self.cpu_offloading_chunk_size = 8192
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError("_DummyKVCache has no key buffer (TT manages KV cache)")

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError("_DummyKVCache has no value buffer (TT manages KV cache)")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("_DummyKVCache has no kv buffer (TT manages KV cache)")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None:
        raise RuntimeError("_DummyKVCache cannot set kv buffer (TT manages KV cache)")

    def get_kv_size_bytes(self):
        return 0, 0


class _DummyModel:
    """Stand-in so `inspect.signature(model.forward)` and `getattr(model, ...)`
    in ModelRunner.__init__ don't crash. The real forward lives in TTLlamaWrapper.
    """

    @staticmethod
    def forward():
        pass
