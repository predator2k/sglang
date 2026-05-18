# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""TTFunctionalCache: HF Cache subclass that bypasses index_copy_ for q_len=1 decode.

Why: production HF StaticCache.update() calls layer.keys.index_copy_(2, cache_position,
key_states) — which lowers via tt-mlir CacheFillUpdatePattern to ttir.update_cache, then
auto-promotes to ttir.paged_update_cache, which requires the cache tensor to have
exactly one user. In autoregressive decode the cache is both written (by update) AND
read (by attention) — 2 users — so legalization fails. T2.1 in tt-mlir-sglang adds a
guard so multi-user scatter falls through to ttnn.scatter; meanwhile this Python-side
class uses torch.where so the same model graph also works on stock tt-mlir.

Decode (q_len=1) uses torch.where to write into a functionally-updated tensor:
the writes are private (no shared XLA mutation across sub-calls), which lets multi-
token compile (A1) work. Prefill (q_len > 1) falls through to StaticCache.update for
correctness; prefill speed isn't the bottleneck.

The Python list rebind (self._latest_k[layer_idx] = new_k) was probed at K=1/2/4 in
probe_t2_1.log: 0 graph breaks, 0 recompiles, bit-exact. dynamo handles it cleanly.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch
from transformers.cache_utils import StaticCache


class TTFunctionalCache(StaticCache):
    """Functional KV cache for tt-xla.

    Decode q_len=1: torch.where + Python list rebind (no index_copy_).
    Prefill q_len>1: falls through to StaticCache.update (parent uses index_copy_, works).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Latest functional KV per layer. None means "read from self.layers[i].keys"
        # (i.e., the parent-class tensor that prefill wrote into).
        self._latest_k: list[Optional[torch.Tensor]] = [None] * len(self.layers)
        self._latest_v: list[Optional[torch.Tensor]] = [None] * len(self.layers)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefill / multi-token: parent index_copy_ path, proven for q_len > 1.
        if key_states.shape[-2] != 1:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # SGLANG_TT_CACHE_MODE=index_copy: bypass torch.where, use parent's in-place
        # index_copy_ even at q_len=1. Removes ~50ms TPOT overhead on Qwen3-8B.
        # Safe iff dynamo doesn't recompile on the cache mutation (verified post-B.2).
        if os.environ.get("SGLANG_TT_CACHE_MODE") == "index_copy":
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Decode: functional path.
        cp = cache_kwargs["cache_position"]  # [1] tensor
        prior_k = (
            self._latest_k[layer_idx]
            if self._latest_k[layer_idx] is not None
            else self.layers[layer_idx].keys
        )
        prior_v = (
            self._latest_v[layer_idx]
            if self._latest_v[layer_idx] is not None
            else self.layers[layer_idx].values
        )

        max_len = prior_k.shape[2]
        pos = torch.arange(max_len, device=cp.device)
        write_mask = (pos == cp[0]).view(1, 1, max_len, 1)

        # torch.where with broadcast — same pattern as probe_t2_1.log V1
        new_k = torch.where(
            write_mask, key_states.expand_as(prior_k), prior_k
        )
        new_v = torch.where(
            write_mask, value_states.expand_as(prior_v), prior_v
        )

        # Python list rebind (probed safe under torch.compile at K=1/2/4)
        self._latest_k[layer_idx] = new_k
        self._latest_v[layer_idx] = new_v
        return new_k, new_v

    def reset(self) -> None:
        """Clear the functional KV state. Parent tensors stay (prefill rewrites them)."""
        for i in range(len(self.layers)):
            self._latest_k[i] = None
            self._latest_v[i] = None
        # Don't call super().reset() — that zeros parent tensors; prefill will overwrite anyway.
