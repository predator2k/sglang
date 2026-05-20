# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Per-layer compression policy for CompressedHiCacheFile.

The HiCache page tensor for layer_first layout contains *all* layers and both
K and V values: shape ``(2, L, P, H, D)`` where L = num_layers, P = page_size,
H = head_num, D = head_dim. Flattened, that's 2 * L equal-sized slabs in the
order ``[K_layer0..K_layerL-1, V_layer0..V_layerL-1]``.

Compressibility varies significantly per slab (see the kvcache_comp ablation —
layer-0 V is 4-5x on raw zstd, mid layers are 1.3x). A *policy* picks the
right (codec, layout, level, params) bundle per slab.

YAML schema (v3)
================

::

    # 1) Named profile library. Each profile = a full codec+layout+params bundle.
    #    `extends:` inherits and overrides; cycles are detected.
    profiles:
      balanced:
        codec: zstd
        compression_level: 1
        layout: sem_split_channel

      fast:
        codec: blosc2
        blosc2_inner_codec: lz4
        blosc2_nthreads: 0
        layout: sem_split_channel

      aggressive:
        extends: balanced
        compression_level: 9          # everything else inherited

      archive:
        extends: balanced
        compression_level: 19
        layout: bit_plane

    # 2) Default (string profile name | dict | shorthand string)
    default: balanced
    # OR equivalently:
    # default:
    #   codec: zstd
    #   compression_level: 1
    #   layout: sem_split_channel
    # OR (shorthand):
    # default: "zstd:1@sem_split_channel"

    # 3) Routing rules. Each rule = match dict + profile assignment(s).
    rules:
      - match: { layers: [0], kv: V }
        profile: aggressive

      - match: { layers: "1-15" }
        profile: fast

      - match: { layers: "16-31" }
        K: aggressive                  # K/V split inside one match
        V: archive

    # 4) Optional runtime knobs (not part of any single page's bundle)
    runtime:
      batch_threads: 4
      slab_threads: 4
      strict_dtype: true

Shorthand
---------
A profile-string of the form ``codec[:level][:inner]@layout`` is parsed
into an anonymous profile, e.g.::

    "zstd:9@sem_split_channel"  →  codec=zstd, level=9, layout=sem_split_channel
    "blosc2:lz4:3@sem_split_channel" → blosc2 with lz4 inner, level 3
    "lz4@byte_hi_lo_channel"     →  lz4, default level, byte_hi_lo_channel

Backwards compatibility
-----------------------
The v2 schema (``default: {dict}``, ``rules: [{layers, K, V, codec, ...}]``)
is still parsed correctly — the rule's inline codec/layout/level fields are
treated as an anonymous profile.

Match keys
----------
``match`` is a small dict. Supported keys:

  layers : int | list[int] | "lo-hi" | "all"
  kv     : "K" | "V" | ["K", "V"]   (default = both)

(Tier / radix-depth keys are reserved for future versions.)
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from sglang.srt.mem_cache.storage.compressed import codecs as codecs_mod

logger = logging.getLogger(__name__)

KV_K = 0
KV_V = 1
KV_NAMES = {KV_K: "K", KV_V: "V"}
_KV_PARSE = {"K": KV_K, "V": KV_V}


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    """One concrete (codec, layout, params) bundle.

    A separate codec instance is lazily built **per thread** that calls
    :meth:`get_codec` — most underlying codec libraries (zstandard, lz4,
    isal_zlib, …) are not thread-safe for concurrent ``encode``/``decode``
    on the same instance, but they are cheap enough to clone per thread.
    """

    codec_name: str = "zstd"
    layout_name: str = "auto"
    compression_level: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)
    _label: Optional[str] = None  # name in the profile library, if any
    _tls: threading.local = field(default_factory=threading.local, repr=False, compare=False)
    # ``_codec_kind`` is captured the first time we build a codec, so that
    # callers can ask for the kind without touching a thread-local instance.
    _codec_kind: Optional[int] = field(default=None, repr=False, compare=False)

    def get_codec(self) -> codecs_mod.Codec:
        codec = getattr(self._tls, "codec", None)
        if codec is None:
            codec = codecs_mod.build_codec(
                self.codec_name,
                {"compression_level": self.compression_level, **self.extra},
            )
            self._tls.codec = codec
            if self._codec_kind is None:
                self._codec_kind = codec.kind
        return codec

    def codec_kind(self) -> int:
        """Codec ``kind`` (file-format byte) without committing to a TLS slot."""
        if self._codec_kind is None:
            # Trigger a build to learn the kind, then discard.
            self.get_codec()
        return self._codec_kind  # type: ignore[return-value]

    def describe(self) -> str:
        tag = f"[{self._label}] " if self._label else ""
        extra = f" extra={self.extra}" if self.extra else ""
        return (
            f"{tag}codec={self.codec_name}@L{self.compression_level} "
            f"layout={self.layout_name}{extra}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SHORTHAND_RE = re.compile(
    r"^(?P<codec>[A-Za-z0-9_]+)"
    r"(?::(?P<arg1>[A-Za-z0-9_]+))?"
    r"(?::(?P<arg2>[A-Za-z0-9_]+))?"
    r"(?:@(?P<layout>[A-Za-z0-9_]+))?$"
)


def parse_shorthand(s: str) -> Dict[str, Any]:
    """Turn ``"zstd:9@sem_split_channel"`` or ``"blosc2:lz4:3@sem_split_channel"``
    into an inline profile dict."""
    m = _SHORTHAND_RE.match(s.strip())
    if not m:
        raise ValueError(f"profile shorthand does not parse: {s!r}")
    codec = m.group("codec")
    layout = m.group("layout") or "auto"
    arg1 = m.group("arg1")
    arg2 = m.group("arg2")

    if codec == "blosc2" and arg1 and arg2:
        # blosc2:<inner>:<level>
        return {
            "codec": "blosc2",
            "blosc2_inner_codec": arg1,
            "compression_level": int(arg2),
            "layout": layout,
        }
    # codec:<level>
    level = int(arg1) if arg1 is not None else 1
    return {"codec": codec, "compression_level": level, "layout": layout}


def _parse_layer_spec(spec: Union[str, List[int], int]) -> Optional[List[int]]:
    """None means 'match all layers'."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(x) for x in spec]
    if isinstance(spec, str):
        if spec.strip().lower() in {"all", "*"}:
            return None
        if "-" in spec:
            lo, hi = spec.split("-", 1)
            return list(range(int(lo), int(hi) + 1))
        return [int(spec)]
    raise ValueError(f"unrecognized layer spec: {spec!r}")


def _parse_kv_spec(spec: Optional[Any]) -> Optional[List[int]]:
    """None means 'both K and V'."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec.lower() in {"all", "*"}:
            return None
        return [_KV_PARSE[spec.upper()]]
    if isinstance(spec, list):
        return [_KV_PARSE[s.upper()] for s in spec]
    raise ValueError(f"unrecognized kv spec: {spec!r}")


_RESERVED_PROFILE_KEYS = {"codec", "layout", "compression_level", "extends"}


def _resolve_profile_dict(
    raw: Union[str, Dict[str, Any]],
    library: Dict[str, "Profile"],
    fallback: "Profile",
    label: Optional[str] = None,
    _seen: Optional[set] = None,
) -> "Profile":
    """Build a Profile from one of:
      - str: lookup in library, or parse as shorthand, or "" → fallback
      - dict: possibly with ``extends:`` referring to library member
    """
    if raw is None:
        return fallback

    if isinstance(raw, str):
        # Library reference?
        if raw in library:
            return library[raw]
        # Shorthand?
        if "@" in raw or ":" in raw:
            raw = parse_shorthand(raw)
        else:
            raise ValueError(
                f"profile reference {raw!r} is neither a defined profile "
                f"({sorted(library)}) nor a shorthand string"
            )

    if not isinstance(raw, dict):
        raise ValueError(f"cannot resolve profile from {raw!r}")

    _seen = set(_seen or ())
    base = fallback
    ext = raw.get("extends")
    if ext:
        if ext in _seen:
            raise ValueError(f"circular extends in profile chain: {ext}")
        _seen.add(ext)
        base = _resolve_profile_dict(ext, library, fallback, label=None, _seen=_seen)

    codec = raw.get("codec", base.codec_name)
    layout = raw.get("layout", base.layout_name)
    level = int(raw.get("compression_level", base.compression_level))
    # extra = (base extras) overlaid with (this profile's non-reserved keys)
    extra = {**base.extra, **{k: v for k, v in raw.items() if k not in _RESERVED_PROFILE_KEYS}}
    return Profile(
        codec_name=codec,
        layout_name=layout,
        compression_level=level,
        extra=extra,
        _label=label,
    )


# ---------------------------------------------------------------------------
# LayeredPolicy
# ---------------------------------------------------------------------------


@dataclass
class _Rule:
    layers: Optional[List[int]]      # None = all
    kvs: Optional[List[int]]         # None = both
    K_profile: Optional[Profile]
    V_profile: Optional[Profile]

    def describe(self) -> str:
        ls = "all" if self.layers is None else (
            f"{self.layers[0]}-{self.layers[-1]}"
            if len(self.layers) > 4 and self.layers == list(range(self.layers[0], self.layers[-1] + 1))
            else str(self.layers)
        )
        parts = [f"layers={ls}"]
        if self.kvs is not None:
            parts.append("kv=" + "/".join(KV_NAMES[k] for k in self.kvs))
        if self.K_profile is self.V_profile and self.K_profile is not None:
            parts.append(self.K_profile.describe())
        else:
            if self.K_profile:
                parts.append("K:" + self.K_profile.describe())
            if self.V_profile:
                parts.append("V:" + self.V_profile.describe())
        return " ".join(parts)


@dataclass
class LayeredPolicy:
    default: Profile
    rules: List[_Rule] = field(default_factory=list)
    library: Dict[str, Profile] = field(default_factory=dict)
    _resolved: Dict[tuple, Profile] = field(default_factory=dict)

    # ----- builders -----

    @classmethod
    def from_yaml(cls, path: str) -> "LayeredPolicy":
        import yaml

        with open(path, "r") as f:
            doc = yaml.safe_load(f)
        return cls.from_dict(doc)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "LayeredPolicy":
        doc = doc or {}

        # ---- step 1: build profile library (with topological extends resolution) ----
        raw_profiles: Dict[str, Any] = dict(doc.get("profiles") or {})
        fallback = Profile()  # zstd L1 auto, used when nothing else is reachable

        # First pass: build library, allowing forward references via repeated resolution.
        library: Dict[str, Profile] = {}
        # Try to resolve all profiles. Multiple passes since extends might point forward.
        # Bound the number of passes by the number of profiles.
        for _ in range(max(1, len(raw_profiles) + 2)):
            unresolved_before = [n for n in raw_profiles if n not in library]
            for name in unresolved_before:
                raw = raw_profiles[name]
                ext = raw.get("extends") if isinstance(raw, dict) else None
                if ext and ext not in library and ext in raw_profiles:
                    continue  # wait for the parent
                try:
                    library[name] = _resolve_profile_dict(
                        raw, library, fallback, label=name
                    )
                except ValueError as e:
                    raise ValueError(f"profile {name!r}: {e}") from e
            if all(n in library for n in raw_profiles):
                break
        else:
            missing = [n for n in raw_profiles if n not in library]
            raise ValueError(
                f"could not resolve profile library (likely circular extends): {missing}"
            )

        # ---- step 2: default profile ----
        default_raw = doc.get("default", "balanced" if "balanced" in library else None)
        if default_raw is None and library:
            # No default declared, no "balanced" — pick the first profile.
            default_raw = next(iter(library))
        if default_raw is None:
            default_profile = fallback
        else:
            default_profile = _resolve_profile_dict(
                default_raw, library, fallback, label="<default>"
            )

        # ---- step 3: rules ----
        rules: List[_Rule] = []
        for raw_rule in doc.get("rules") or []:
            # Normalise: allow legacy v2 form where match keys live at the top of the rule.
            match = dict(raw_rule.get("match") or {})
            if "layers" in raw_rule and "match" not in raw_rule:
                match["layers"] = raw_rule["layers"]
            if "kv" in raw_rule and "kv" not in match:
                match["kv"] = raw_rule["kv"]
            layers = _parse_layer_spec(match.get("layers", "all"))
            kvs = _parse_kv_spec(match.get("kv"))

            # Profile assignment forms:
            #   profile: <name|dict|shorthand>   → both K and V
            #   K: <...> / V: <...>              → split assignment
            #   inline codec/layout/compression_level keys (v2 form) → both K and V
            profile_raw = raw_rule.get("profile")
            K_raw = raw_rule.get("K")
            V_raw = raw_rule.get("V")
            if profile_raw is None and K_raw is None and V_raw is None:
                # v2-style inline (codec/layout/level on the rule itself, ignoring match keys)
                inline = {
                    k: v
                    for k, v in raw_rule.items()
                    if k not in {"match", "layers", "kv", "K", "V", "profile"}
                }
                if inline:
                    profile_raw = inline
            K_prof = (
                _resolve_profile_dict(K_raw, library, default_profile)
                if K_raw is not None
                else (
                    _resolve_profile_dict(profile_raw, library, default_profile)
                    if profile_raw is not None
                    else None
                )
            )
            V_prof = (
                _resolve_profile_dict(V_raw, library, default_profile)
                if V_raw is not None
                else (
                    _resolve_profile_dict(profile_raw, library, default_profile)
                    if profile_raw is not None
                    else None
                )
            )
            rules.append(_Rule(layers=layers, kvs=kvs, K_profile=K_prof, V_profile=V_prof))

        return cls(default=default_profile, rules=rules, library=library)

    # ----- resolve -----

    def resolve(self, kv_kind: int, layer_idx: int) -> Profile:
        key = (kv_kind, layer_idx)
        if key in self._resolved:
            return self._resolved[key]

        chosen = self.default
        for rule in self.rules:
            if rule.layers is not None and layer_idx not in rule.layers:
                continue
            if rule.kvs is not None and kv_kind not in rule.kvs:
                continue
            profile = rule.K_profile if kv_kind == KV_K else rule.V_profile
            if profile is not None:
                chosen = profile
        self._resolved[key] = chosen
        return chosen

    # ----- introspection -----

    def all_profiles(self) -> List[Profile]:
        seen: Dict[int, Profile] = {id(self.default): self.default}
        for p in self.library.values():
            if id(p) not in seen:
                seen[id(p)] = p
        for rule in self.rules:
            for p in (rule.K_profile, rule.V_profile):
                if p is not None and id(p) not in seen:
                    seen[id(p)] = p
        return list(seen.values())

    def describe(self) -> str:
        lines = []
        if self.library:
            lines.append("profiles:")
            for name, p in self.library.items():
                lines.append(f"  {name}: {p.describe()}")
        lines.append(f"default: {self.default.describe()}")
        if self.rules:
            lines.append("rules:")
            for i, r in enumerate(self.rules):
                lines.append(f"  [{i}] {r.describe()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime knobs (extracted from a config dict so the caller can pass them
# straight into ``HiCacheStorageConfig.extra_config``).
# ---------------------------------------------------------------------------


def extract_runtime(doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(doc.get("runtime") or {})
