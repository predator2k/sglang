"""Tenstorrent ModelRunner stub.

Bookkeeping subclass of `ModelRunner` that skips PyTorch weight loading and
KV cache allocation — the real model forward and KV management live inside
`TTTransformersExecutionBackend` (ttnn / tt_transformers). The scheduler still needs a
ModelRunner-shaped object with `req_to_token_pool`, `token_to_kv_pool`,
`token_to_kv_pool_allocator`, and a `model` attribute, so we satisfy those
contracts with the minimum allocations.
"""

from __future__ import annotations

import logging

from sglang.srt.hardware_backend.tenstorrent.model_runner_stub import (
    _DummyKVCache,
    _DummyModel,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)


class TTModelRunner(ModelRunner):
    """ModelRunner stub for Tenstorrent.

    Mirrors MlxModelRunnerStub field-for-field. The only TT-specific
    addition is the `self.device = "cpu"` override in `__init__` — torch
    has no "tenstorrent" device module, and scheduler.init_overlap looks
    up `torch.get_device_module(self.device)`, which would crash on the
    string "tenstorrent".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CRITICAL: torch has no "tenstorrent" device module. Scheduler's
        # init_overlap calls torch.get_device_module(self.device); "cpu"
        # resolves to torch.cpu, which is safe.
        self.device = "cpu"

    def load_model(self):
        """Set only the metadata that downstream code needs, without
        loading any PyTorch model weights."""
        logger.info(
            "TT stub: skipping PyTorch model weight loading "
            "(inference runs through ttnn / tt_transformers)"
        )

        self.model = _DummyModel()

        self.sliding_window_size = None
        if (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size

        self.dtype = self.model_config.dtype
        self.weight_load_mem_usage = 0

    def initialize(self, pre_model_load_memory: float):
        """Lightweight initialize that skips heavy PyTorch setup.

        Creates minimal req_to_token_pool and token_to_kv_pool_allocator
        with a dummy KV cache (zero GPU memory) so the scheduler works.
        """
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        self.sampler = None
        self.load_model()

        model_num_layers = max(
            self.model_config.num_hidden_layers,
            self.model_config.num_attention_layers,
        )
        self.start_layer = 0
        self.end_layer = model_num_layers
        self.num_effective_layers = model_num_layers

        self.kv_cache_dtype = self.dtype

        self.max_total_num_tokens = self.model_config.context_len
        self.max_running_requests = min(
            self.max_total_num_tokens // 2,
            4096,
        )
        self.is_hybrid_swa = False

        self.req_to_token_pool = ReqToTokenPool(
            size=self.max_running_requests,
            max_context_len=self.model_config.context_len,
            device="cpu",
            enable_memory_saver=False,
        )

        dummy_kv = _DummyKVCache(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device="cpu",
        )
        self.token_to_kv_pool = dummy_kv
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device="cpu",
            kvcache=dummy_kv,
            need_sort=False,
        )

        self.graph_runner = None
        self.graph_mem_usage = 0
        self.attn_backend = None

        logger.info(
            "TT stub: initialized minimal pools "
            "(max_total_num_tokens=%d, max_running_requests=%d, "
            "zero TT KV cache allocation)",
            self.max_total_num_tokens,
            self.max_running_requests,
        )
