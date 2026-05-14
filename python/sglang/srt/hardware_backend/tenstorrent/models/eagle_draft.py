# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""EAGLE draft model loader with co-mesh placement (P3a.2 T2.1, INV-9).

Per P3a.0 T0.3 evidence (`_fixtures/p3a_eagle_mesh_evidence.txt`), the chosen
mesh layout is **Layout A — shared single (1, 2) MeshDevice**. Both main and
draft load on the same mesh; concurrent kernel queueing handles draft+target
interleaving. Layout B (split (1, 1) meshes per model) is INFEASIBLE on the
current tt-metal commit due to MeshDevice lifecycle limits.

Usage (typical flow):
    main = TenstorrentQwenForCausalLM(config=main_config)
    draft = load_eagle_draft(main, draft_path="/models/Qwen3-1.7B")
    # main.mesh_device is draft.mesh_device  → True
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_eagle_draft(main_model: Any, draft_path: Optional[str] = None):
    """Load an EAGLE draft model that cohosts on main_model's mesh.

    Args:
        main_model: a TenstorrentLlamaForCausalLM / TenstorrentQwenForCausalLM
            instance whose `mesh_device` will be shared with the draft.
        draft_path: HuggingFace model path / hub id. If None, reads from the
            `SGLANG_TT_SPEC_DRAFT_PATH` environment variable.

    Returns:
        A draft model instance, ready for SGLang's spec runner.

    Raises:
        ValueError: when `draft_path` is unresolvable.
        RuntimeError: when SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT requests an
            unsupported layout (Layout B was empirically INFEASIBLE per T0.3).
    """
    draft_path = draft_path or os.environ.get("SGLANG_TT_SPEC_DRAFT_PATH")
    if not draft_path:
        raise ValueError(
            "SGLANG_TT_SPEC_DRAFT_PATH not set and no draft_path argument; "
            "set --speculative-draft-model-path or the env var to a HF dir"
        )

    layout = os.environ.get("SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT", "shared")
    logger.info(
        f"[EAGLE-draft] loading from {draft_path}; mesh_layout={layout}; "
        f"main mesh num_devices={main_model.mesh_device.get_num_devices()}"
    )

    if layout == "split":
        # Layout B was empirically INFEASIBLE per P3a.0 T0.3 (ETH core 30-25
        # timeout when opening a second MeshDevice in the same process).
        raise RuntimeError(
            "SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT=split is not supported on this "
            "tt-metal commit (T0.3 verified Layout A only). Use 'shared'."
        )
    if layout != "shared":
        raise ValueError(
            f"Unknown SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT={layout!r}; expected 'shared'"
        )

    # Resolve the draft model class by HF architecture. registry.py maps the
    # arch string (e.g. "Qwen3ForCausalLM") to our Tenstorrent wrapper.
    from transformers import AutoConfig
    from sglang.srt.hardware_backend.tenstorrent.models.registry import (
        get_tt_model_class_for_arch,
    )

    draft_config = AutoConfig.from_pretrained(draft_path, trust_remote_code=True)
    archs = getattr(draft_config, "architectures", []) or []
    if not archs:
        raise ValueError(
            f"draft_config has no architectures field; cannot resolve TT class "
            f"for {draft_path}"
        )
    draft_cls = get_tt_model_class_for_arch(archs[0])
    if draft_cls is None:
        raise ValueError(
            f"No Tenstorrent model class registered for arch {archs[0]!r}; "
            f"check registry.py"
        )

    # Set HF_MODEL env so tt_transformers ModelArgs picks up the draft path
    # (overrides main's HF_MODEL temporarily — caller should restore it after
    # if they need the main path back for subsequent state lookups).
    prev_hf_model = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = draft_path
    try:
        # Pass main_model.mesh_device explicitly. The constructor's
        # mesh_device kwarg (added in P3a.2 T2.1) bypasses
        # BaseMetalDeviceRunner.set_device() so a second MeshDevice isn't
        # opened — which would fail per T0.3 Layout B evidence.
        draft = draft_cls(
            config=draft_config,
            quant_config=None,
            tt_model=None,
            mesh_device=main_model.mesh_device,
        )
    finally:
        # Restore the main model's HF_MODEL so downstream lookups (e.g.,
        # tokenizer reload) still resolve to the main path.
        if prev_hf_model is not None:
            os.environ["HF_MODEL"] = prev_hf_model

    # Sanity check: confirm cohost succeeded.
    if draft.mesh_device is not main_model.mesh_device:
        raise RuntimeError(
            f"EAGLE draft mesh ({id(draft.mesh_device)}) is not the same as "
            f"main mesh ({id(main_model.mesh_device)}); cohost failed"
        )
    logger.info(
        f"[EAGLE-draft] loaded {draft_cls.__name__} on shared mesh "
        f"id={id(draft.mesh_device)} (same as main)"
    )
    return draft
