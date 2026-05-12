"""TTExecutionBackend ABC — the 5-method contract for SGLang ↔ TT integration.

The worker (Phase G) imports only this ABC plus the registry factory.
Concrete backends live in sibling modules.
"""

from __future__ import annotations

import abc
from typing import Any


class TTExecutionBackend(abc.ABC):
    """Black-box per-request lifecycle interface.

    Implementations own all device interaction; the worker treats this object
    as opaque except for these 5 methods. Return types are host torch tensors
    (the backend is responsible for any device→host transfer + dtype massage).
    """

    @abc.abstractmethod
    def __init__(self, model_path: str, mesh_device: Any, *, max_seq_len: int) -> None:
        ...

    @abc.abstractmethod
    def new_request(self, req_id: str, prompt_tokens) -> None:
        """Allocate per-request KV / position state."""

    @abc.abstractmethod
    def extend(self, req_id: str):
        """Run prefill on the request's prompt; return last-token logits."""

    @abc.abstractmethod
    def decode_step(self, req_id: str, last_token: int):
        """Advance by one token; return next-token logits."""

    @abc.abstractmethod
    def free(self, req_id: str) -> None:
        """Release per-request state."""

    @abc.abstractmethod
    def reset_all(self) -> None:
        """Drop all per-request state (e.g. on scheduler shutdown)."""
