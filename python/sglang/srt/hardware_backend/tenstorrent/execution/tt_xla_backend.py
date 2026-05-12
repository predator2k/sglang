"""tt-xla execution backend — POST-P1 PLACEHOLDER.

tt-xla (PJRT → StableHLO → TT-MLIR → TT-Metal) is Tenstorrent's new
general-purpose PyTorch/JAX frontend; tt-torch is deprecated in its
favor. The real implementation is post-P1 — see spec §11 "P2-coverage".

This module exists in P1 only so the registry has a "tt_xla" entry; any
attempt to construct it raises NotImplementedError with a pointer.
"""

from __future__ import annotations

from sglang.srt.hardware_backend.tenstorrent.execution import (
    register_tt_execution_backend,
)
from sglang.srt.hardware_backend.tenstorrent.execution.base import (
    TTExecutionBackend,
)


@register_tt_execution_backend("tt_xla")
class TTXLAExecutionBackend(TTExecutionBackend):
    def __init__(self, model_path, mesh_device, *, max_seq_len):
        raise NotImplementedError(
            "tt-xla execution backend is post-P1; see spec §11 'P2-coverage'. "
            "P1 ships tt_transformers only (set SGLANG_TT_EXECUTION_BACKEND="
            "tt_transformers or leave unset to use the auto default)."
        )

    def new_request(self, req_id, prompt_tokens):
        raise NotImplementedError

    def extend(self, req_id):
        raise NotImplementedError

    def decode_step(self, req_id, last_token):
        raise NotImplementedError

    def free(self, req_id):
        raise NotImplementedError

    def reset_all(self):
        raise NotImplementedError
