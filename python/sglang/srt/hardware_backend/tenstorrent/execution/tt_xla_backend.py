"""tt-xla execution backend — POST-P1 PLACEHOLDER.

tt-xla (PJRT → StableHLO → TT-MLIR → TT-Metal) is Tenstorrent's new
general-purpose PyTorch/JAX frontend; tt-torch is deprecated in its
favor. The real implementation is post-P1 — see spec §11 "P2-coverage".

This module exists in P1 only so the registry has a "tt_xla" entry; any
attempt to construct it raises NotImplementedError with a pointer.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.hardware_backend.tenstorrent.execution import (
    register_tt_execution_backend,
)
from sglang.srt.hardware_backend.tenstorrent.execution.base import (
    TTExecutionBackend,
)

_POST_P1_MSG = (
    "tt-xla execution backend is post-P1; see spec §11 'P2-coverage'."
)


@register_tt_execution_backend("tt_xla")
class TTXLAExecutionBackend(TTExecutionBackend):
    def __init__(
        self, model_path: str, mesh_device: Any, *, max_seq_len: int
    ) -> None:
        raise NotImplementedError(
            _POST_P1_MSG
            + " P1 ships tt_transformers only (set SGLANG_TT_EXECUTION_BACKEND="
            "tt_transformers or leave unset to use the auto default)."
        )

    def new_request(self, req_id: str, prompt_tokens: list[int]) -> None:
        raise NotImplementedError(_POST_P1_MSG)

    def extend(self, req_id: str):
        raise NotImplementedError(_POST_P1_MSG)

    def decode_step(self, req_id: str, last_token: int):
        raise NotImplementedError(_POST_P1_MSG)

    def free(self, req_id: str) -> None:
        raise NotImplementedError(_POST_P1_MSG)

    def reset_all(self) -> None:
        raise NotImplementedError(_POST_P1_MSG)
