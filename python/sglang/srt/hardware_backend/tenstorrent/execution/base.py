"""TTExecutionBackend ABC — model-level forward contract (P2a INV-1).

P1 shipped a 5-method per-request ABC (new_request / extend / decode_step / free /
reset_all). P2a redesigns to a single model-level `forward(forward_batch)` to
align with SGLang's per-step batched scheduling.

INVARIANTS (from spec §2.4 — violations trigger §5.2b brainstorming replay):

  INV-1: forward(forward_batch) -> LogitsProcessorOutput is model-level forward
         (NOT per-layer attention). MUST NOT be named or aliased to SGLang's
         AttentionBackend base class.
  INV-2: TTPagedKVAdapter subclasses PagedTokenToKVPoolAllocator with
         device="cpu" + custom KVCache wrapper. alloc_extend/alloc_decode
         are pure CPU torch (no Triton).
  INV-3: page_table dtype = torch.int32; values are block IDs in [0, num_pages),
         computed as (token_index // block_size).
  INV-4: SGLang owns the page table; tt_transformers receives kv_cache + page_table
         as parameters (paged_attention_config path is NOT used).
  INV-5: Multiple backends co-exist via SGLANG_TT_EXECUTION_BACKEND; switch
         requires server restart (mesh re-init).
  INV-6: Tenstorrent model arches live under Tenstorrent* namespace
         (TenstorrentLlamaForCausalLM), separate from SGLang's LlamaForCausalLM.
  INV-7: MeshDeviceCtx supports (1,2) and (1,4) parametric shapes; P2a validates
         (1,2) only on hardware, (1,4) is static-lint.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TTExecutionBackend(abc.ABC):
    """Model-level forward contract.

    A backend owns one or more `tt_transformers` Generators and translates a
    SGLang `ForwardBatch` into `Generator.prefill_forward_text` /
    `decode_forward` calls. KV state lives in the SGLang-owned paged pool;
    the backend does NOT track per-request state across calls.
    """

    @abc.abstractmethod
    def __init__(
        self,
        model_path: str,
        mesh_device: Any,
        *,
        max_seq_len: int,
        max_batch_size: int,
        token_to_kv_pool: Any,  # TTPagedKVAdapter (paged) or None (simple/B=1)
    ) -> None:
        ...

    @abc.abstractmethod
    def forward(self, forward_batch: "ForwardBatch") -> "LogitsProcessorOutput":
        """Run one prefill or decode step.

        Returns `LogitsProcessorOutput` with `next_token_logits` populated on
        host CPU (`torch.Tensor[B, vocab]`). On error, set every req's
        finish_reason via `Req.set_finish_with_abort` AND return zero-logits
        (NOT None) per Q2 (see phase0_signature_evidence.txt) — the sampler
        consumes the row in the same forward step.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release all device state. Called from MeshDeviceCtx teardown."""
