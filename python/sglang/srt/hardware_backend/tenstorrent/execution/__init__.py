"""Registry + factory for TT execution backends.

Mirror of SGLang's attention_registry.py pattern. P2a ships two real
backend keys: tt_transformers_single (simple B=1, ported from P1) and a
placeholder for tt_transformers_paged (Phase 4). tt_xla is a registered
placeholder so the post-P1 add-a-backend path is a swap rather than a
Phase-F rewrite.

IMPORTANT: the registry dict MUST be declared before the backend modules
are imported — their @register_tt_execution_backend(...) decorators fire
at import time and would NameError otherwise.
"""

from __future__ import annotations

from typing import Callable

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tenstorrent.execution.base import (
    TTExecutionBackend,
)

# Step 1: declare the registry FIRST.
TT_EXECUTION_BACKENDS: dict[str, type[TTExecutionBackend]] = {}


def register_tt_execution_backend(
    name: str,
) -> Callable[[type[TTExecutionBackend]], type[TTExecutionBackend]]:
    """Class decorator that registers a TTExecutionBackend subclass under `name`."""

    def _wrap(cls: type[TTExecutionBackend]) -> type[TTExecutionBackend]:
        TT_EXECUTION_BACKENDS[name] = cls
        return cls

    return _wrap


# Step 2: NOW import the backend modules — their decorators populate the
# dict. E402 is suppressed here intentionally: the import MUST follow the
# registry declaration above, otherwise the decorator NameErrors. Don't
# silence E402 elsewhere — this is the one legitimate place for it.
from sglang.srt.hardware_backend.tenstorrent.execution import (  # noqa: E402, F401
    tt_transformers_backend,
    tt_xla_backend,
)


def resolve_execution_backend_name(requested: str | None = None) -> str:
    """Resolve to P2a default: tt_transformers_paged (plugin path).

    P2a.0 default was tt_transformers_single; P2a.1 flips to paged.
    """
    name = (requested or envs.SGLANG_TT_EXECUTION_BACKEND.get() or "auto").lower()
    if name == "auto":
        return "tt_transformers_paged"  # was tt_transformers_single in P2a.0
    return name


def get_tt_execution_backend(name: str | None = None) -> type[TTExecutionBackend]:
    resolved = resolve_execution_backend_name(name)
    if resolved not in TT_EXECUTION_BACKENDS:
        available = ", ".join(sorted(TT_EXECUTION_BACKENDS)) or "<none>"
        raise ValueError(
            f"Unknown TT execution backend {resolved!r} (available: {available}). "
            f"Set SGLANG_TT_EXECUTION_BACKEND or pass an explicit name."
        )
    return TT_EXECUTION_BACKENDS[resolved]
