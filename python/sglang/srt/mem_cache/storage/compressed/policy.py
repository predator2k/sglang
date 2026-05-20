# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Per-layer compression policy for CompressedHiCacheFile.

The HiCache page tensor for layer_first layout contains *all* layers and both
K and V values: shape ``(2, L, P, H, D)`` where L = num_layers, P = page_size,
H = head_num, D = head_dim. When this is flattened we get 2*L equal-sized
"slabs", in the order::

    [K_layer0, K_layer1, ..., K_layerL-1, V_layer0, V_layer1, ..., V_layerL-1]

Because compressibility varies enormously across layers (layer-0 V is 4-5x
on raw zstd, other layers are 1.3x — see kvcache_comp REPORT §4), it is
attractive to pick a different codec / layout / compression level per slab.

This module defines:

  - ``Profile`` — one (codec_name, codec_kwargs, layout_name, compression_level)
                  bundle, instantiated lazily.
  - ``LayeredPolicy`` — resolves (kv_kind, layer_idx) → Profile, given a
                        default and a list of rules.

YAML schema::

    default:
      codec: zstd
      compression_level: 1
      layout: sem_split_channel
      # any codec-specific kwargs go here too

    rules:
      - layers: [0]                 # explicit list
        K: { codec: zstd, layout: sem_split_channel, compression_level: 1 }
        V: { codec: zstd, layout: sem_split_channel, compression_level: 9 }
      - layers: "1-15"              # inclusive range
        codec: blosc2
        blosc2_inner_codec: zstd
        compression_level: 1
      - layers: "16-31"
        codec: lz4
        layout: byte_hi_lo_channel

Rule resolution: later-listed rules win when multiple match. Within a rule,
if both ``K`` and ``V`` sub-blocks are present they override the top-level
codec/layout/params for that direction only.

Each ``Profile`` keeps its own codec instance — different layers can use
different codec engines without interference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from sglang.srt.mem_cache.storage.compressed import codecs as codecs_mod

logger = logging.getLogger(__name__)

KV_K = 0
KV_V = 1
KV_NAMES = {KV_K: "K", KV_V: "V"}


# ---------------------------------------------------------------------------
# Profile = one (codec, layout, params) bundle
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    codec_name: str = "zstd"
    layout_name: str = "auto"
    compression_level: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)
    # Built lazily on first use
    _codec: Optional[codecs_mod.Codec] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any], base: "Profile") -> "Profile":
        """Layer rule dict overlaid on a base profile."""
        return cls(
            codec_name=d.get("codec", base.codec_name),
            layout_name=d.get("layout", base.layout_name),
            compression_level=int(d.get("compression_level", base.compression_level)),
            extra={**base.extra, **{k: v for k, v in d.items() if k not in {"codec", "layout", "compression_level", "K", "V", "layers"}}},
        )

    def get_codec(self) -> codecs_mod.Codec:
        if self._codec is None:
            self._codec = codecs_mod.build_codec(
                self.codec_name,
                {
                    "compression_level": self.compression_level,
                    **self.extra,
                },
            )
        return self._codec

    def describe(self) -> str:
        return (
            f"codec={self.codec_name}@L{self.compression_level} "
            f"layout={self.layout_name}"
            + (f" extra={self.extra}" if self.extra else "")
        )


# ---------------------------------------------------------------------------
# LayeredPolicy
# ---------------------------------------------------------------------------


def _parse_layer_spec(spec: Union[str, List[int], int]) -> List[int]:
    """Parse 'layers' value into a concrete list of indices.

    Supported forms:
      0                 -> [0]
      [0, 1, 7]         -> [0, 1, 7]
      "1-15"            -> [1, 2, ..., 15]
      "all"             -> []  (meaning "every layer", handled by caller)
    """
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(x) for x in spec]
    if isinstance(spec, str):
        if spec.strip().lower() == "all":
            return []  # sentinel for "match all"
        if "-" in spec:
            lo, hi = spec.split("-")
            return list(range(int(lo), int(hi) + 1))
        return [int(spec)]
    raise ValueError(f"unrecognized layer spec: {spec!r}")


@dataclass
class LayeredPolicy:
    """Holds a default profile plus a list of (layer_set, K_profile, V_profile)
    rules; resolves to a concrete Profile per (kv_kind, layer_idx)."""

    default: Profile
    # rules: list of dicts with the parsed/typed fields
    rules: List[Dict[str, Any]] = field(default_factory=list)
    # Resolved cache: (kv_kind, layer_idx) -> Profile
    _resolved: Dict[tuple, Profile] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "LayeredPolicy":
        import yaml

        with open(path, "r") as f:
            doc = yaml.safe_load(f)
        return cls.from_dict(doc)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "LayeredPolicy":
        default_d = doc.get("default", {})
        default = Profile(
            codec_name=default_d.get("codec", "zstd"),
            layout_name=default_d.get("layout", "auto"),
            compression_level=int(default_d.get("compression_level", 1)),
            extra={
                k: v
                for k, v in default_d.items()
                if k not in {"codec", "layout", "compression_level"}
            },
        )

        rules = []
        for raw in doc.get("rules", []):
            layers = _parse_layer_spec(raw.get("layers", "all"))
            rule = {"layers": layers, "K": None, "V": None}
            if "K" in raw:
                rule["K"] = Profile.from_dict(raw["K"], default)
            if "V" in raw:
                rule["V"] = Profile.from_dict(raw["V"], default)
            if rule["K"] is None and rule["V"] is None:
                # Top-level KV-agnostic
                shared = Profile.from_dict(raw, default)
                rule["K"] = shared
                rule["V"] = shared
            rules.append(rule)
        return cls(default=default, rules=rules)

    def resolve(self, kv_kind: int, layer_idx: int) -> Profile:
        key = (kv_kind, layer_idx)
        if key in self._resolved:
            return self._resolved[key]

        chosen = self.default
        # later rules win — walk the list, last match takes effect
        for rule in self.rules:
            layer_set = rule["layers"]
            if layer_set and layer_idx not in layer_set:
                continue  # explicit set, no match
            kv_profile = rule["K"] if kv_kind == KV_K else rule["V"]
            if kv_profile is not None:
                chosen = kv_profile
        self._resolved[key] = chosen
        return chosen

    def all_profiles(self) -> List[Profile]:
        """For initialization / preflight: return every distinct profile referenced."""
        seen = {id(self.default): self.default}
        for rule in self.rules:
            for p in (rule.get("K"), rule.get("V")):
                if p is not None and id(p) not in seen:
                    seen[id(p)] = p
        return list(seen.values())

    def describe(self) -> str:
        lines = [f"default: {self.default.describe()}"]
        for i, rule in enumerate(self.rules):
            ls = rule["layers"]
            tag = "all" if not ls else (
                f"{ls[0]}-{ls[-1]}" if len(ls) > 4 and ls == list(range(ls[0], ls[-1] + 1)) else str(ls)
            )
            if rule["K"] is rule["V"] and rule["K"] is not None:
                lines.append(f"  rule[{i}] layers={tag}: {rule['K'].describe()}")
            else:
                if rule["K"]:
                    lines.append(f"  rule[{i}] layers={tag} K: {rule['K'].describe()}")
                if rule["V"]:
                    lines.append(f"  rule[{i}] layers={tag} V: {rule['V'].describe()}")
        return "\n".join(lines)
