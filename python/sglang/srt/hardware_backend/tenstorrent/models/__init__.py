# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
#
# Adapted from tt-inference-server/tt-sglang-plugin (Apache-2.0, © 2026 Tenstorrent USA, Inc.)
# with rename per spec INV-6 (Tenstorrent* namespace).
"""Tenstorrent model wrappers for SGLang (P2a plugin-absorbed)."""

from .registry import register_tt_models
from .tt_llm import (
    TenstorrentGptOssForCausalLM,
    TenstorrentLlamaForCausalLM,
    TenstorrentMistralForCausalLM,
    TenstorrentQwenForCausalLM,
)

__all__ = [
    "TenstorrentLlamaForCausalLM",
    "TenstorrentQwenForCausalLM",
    "TenstorrentMistralForCausalLM",
    "TenstorrentGptOssForCausalLM",
    "register_tt_models",
]

# Side-effect: register with SGLang on import
register_tt_models()
