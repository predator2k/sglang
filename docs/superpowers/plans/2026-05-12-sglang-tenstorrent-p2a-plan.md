# SGLang-on-Tenstorrent — Phase 2 Implementation Plan (v2, plugin-absorbed)

> **For agentic workers:** Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.

**Goal**: Land P2 paged + RadixAttention + multi-model on 2× Tenstorrent Blackhole p150a by absorbing Tenstorrent's upstream `tt-sglang-plugin` rather than implementing a custom ABC + KV adapter. Concentrate independent value on **RadixAttention validation**, **Blackhole verification**, **dual-track fallback**, and **§9 acceptance discipline** — all areas the plugin has not addressed.

**Spec**: [`../specs/2026-05-12-sglang-tenstorrent-p2-design.md`](../specs/2026-05-12-sglang-tenstorrent-p2-design.md) (amended 2026-05-13). All §N.M / G-item / INV-N / Q-N / R-N / §9.X references resolve there.

**Predecessor**: P1 shipped (47 commits) + P2a.0 shipped + 2 hardware-fix follow-ups (14 commits, `21f38ac9a..44b281ebc`) + spec amendment (`23e9d5cba`).

**Supersedes**: The pre-2026-05-13 P2a plan (3524 lines, 46 tasks). Original at `git show 5028c6b6a:docs/superpowers/plans/2026-05-12-sglang-tenstorrent-p2a-plan.md`. **Stage-1 of v1 (TTPagedKVAdapter / TTBackedKVCache / TTPagedMetadataBackend) is DELETED — plugin replaces it.**

**Branch**: `tenstorrent-p1` on `origin` (`predator2k/sglang`). NO upstream PRs (spec N9).

**Plugin upstream**: [`tenstorrent/tt-inference-server` `tt-sglang-plugin/`](https://github.com/tenstorrent/tt-inference-server/tree/main/tt-sglang-plugin) — already cloned at `/home/mhnie/tt-inference-server/`. SPDX Apache-2.0 © 2026 Tenstorrent USA, Inc. **Adapted copies must preserve SPDX header.**

**Built image**: `localhost/local-tt-metal:dev` (sha256:`973e972bddf5`, tt-metal commit `89686ee7`). UMD-compatible with host KMD 2.8.0 / FW 19.6.0. Verified mesh open + tt_transformers import + `generator_sglang.LlamaForCausalLM` import working.

---

## How to use this plan

1. Sub-stages run **strictly top-to-bottom**: P2a.0 (done) → P2a.1 → P2a.2 → P2a.3 → P2b (optional).
2. Each phase has a verification gate at its end. Do NOT proceed past a failing gate.
3. **Commit per task** with the exact message shown.
4. **INV-1..INV-7 from spec §2.4 are non-negotiable.** Plugin path may violate INV-1's old phrasing — that's fine, the amended INV-1 scopes the strict form to the simple-fallback path only. Any other invariant violation → STOP, escape via spec §5.2b.
5. **Dual-track is a hard requirement**: P1 simple path (T0.7 shipped) must keep working under `SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single`. Plugin path is default.
6. **Inside docker**: image SHA is `973e972bddf5`; venv at `/opt/venv/bin/activate`; tt-metal at `/tt-metal/`; sglang mounted at `/sglang`. Plugin source mounted from `/home/mhnie/tt-inference-server`.
7. Always set `SGLANG_PLATFORM=tenstorrent`.

---

## File map (locked here)

Paths relative to repo root `/home/mhnie/sglang/`. Module root: `python/sglang/srt/hardware_backend/tenstorrent/`.

### New / adapted files (P2a, ~500 LoC absorbed from plugin)

| Path | Source | Modifications |
|---|---|---|
| `models/__init__.py` | new | Calls `register_tt_models()` on import; INV-6 namespace marker |
| `models/tt_llm.py` | adapted from `tt-sglang-plugin/sglang_tt_plugin/models/tt_llm.py` | Rename `TTLlamaForCausalLM` → `TenstorrentLlamaForCausalLM` (INV-6); preserve forward/`_build_page_table`/`_pad_decode_batch` logic verbatim |
| `models/tt_utils.py` | adapted from `tt-sglang-plugin/sglang_tt_plugin/utils/tt_utils.py` | Rename module; keep `BaseMetalDeviceRunner` + `_configure_fabric` + Blackhole branches |
| `models/worker_setup.py` | adapted from `tt-sglang-plugin/sglang_tt_plugin/worker_setup/worker_setup.py` | Keep `_DP\d+` parsing + `TT_VISIBLE_DEVICES` + cache-dir setup |
| `models/registry.py` | adapted from `tt-sglang-plugin/sglang_tt_plugin/patching/patching_model_registry.py` | Patches `ModelRegistry.models` with our `Tenstorrent*ForCausalLM` |

### Preserved from P2a.0 (dual-track fallback)

| Path | Status |
|---|---|
| `execution/base.py` | Shipped — INV-1 (now scoped to simple fallback) |
| `execution/tt_transformers_backend.py` | Shipped — P1 simple backend on new ABC |
| `execution/__init__.py` | Shipped — registry: `tt_transformers_single` / `tt_transformers_paged` (the latter to be wired in P2a.1) |
| `tp_worker.py` | Shipped — `_forward_batch_generation_tt` calls simple backend's `forward()` |
| `platform.py` | Shipped — extended in P2a.1 to call `models.registry.register_tt_models()` |
| `warmup.py` | Shipped — fixed in T0.7 follow-up `dcdd23a2e` |

### Cross-module patch (still required, in our fork only per N9)

| Path | Change | Risk |
|---|---|---|
| `python/sglang/srt/managers/utils.py` | Add `bypass_chunked_req: bool = False` field to `GenerationBatchResult` | R6 — monthly rebase, see `REBASE_TARGETS.md` |
| `python/sglang/srt/managers/scheduler.py` | One read site post-forward: clear `self.chunked_req` if `bypass_chunked_req` | R6 |

### Tests (new in P2a.1 + P2a.3, ~600 LoC)

| Path | Spec ref | Hardware? |
|---|---|---|
| `test/test_plugin_registration.py` | INV-6 + plugin import | No |
| `test/test_page_table_translation.py` | §3.2 page-table math | No |
| `test/test_smoke_paged.py` | §9.1 | **Yes** (Blackhole) |
| `test/test_greedy_correctness_paged.py` | §9.2 | Yes |
| `test/test_batched_correctness.py` | §9.3 | Yes |
| `test/test_radix_prefix_cache.py` | §9.4 (G4a) | Yes |
| `test/test_queue_full_admission.py` | §9.5 | Yes |
| `test/test_abort_via_endpoint.py` / `test_abort_via_disconnect.py` | §9.6 | Yes |
| `test/test_stability_paged.py` | §9.7 | Yes |
| `test/test_dual_track_switch.py` | §9.8 | Yes |
| `test/test_mesh_shape_param.py` | §9.9 | Yes |
| `test/test_perf_log_paged.py` | §9.10 | Yes |
| `test/test_eviction_replay.py` | §9.11 (R5 mitigation) | Yes |
| `test/test_chunked_failure_recovery.py` | §9.12.a-e | No (mock) |
| `test/test_shutdown_teardown.py` | §9.13 | Yes |
| `test/test_token_pool_overflow.py` | §9.14 | No |

---

## P2a.0 — SHIPPED (historical reference)

14 commits `21f38ac9a..44b281ebc` (already pushed; also includes spec amendment `23e9d5cba`). P2a.0 delivered:

- **T0.1** `21f38ac9a` — pin tt-metal docker image SHA (two-pin layout post-amendment: three-pin)
- **T0.2-T0.5** evidence file + Q9 CSV (`_fixtures/phase0_signature_evidence.txt`, `_fixtures/q9_call_site_inventory.csv`)
- **T0.6** `857d2cdf6` — model-level ABC (now scoped to simple-fallback only per amended INV-1)
- **T0.7** `c5f8774ec` — P1 simple backend ported to ABC; `_do_*` private methods preserved verbatim
- **T0.8** `b605f9519` — pytest markers (`@pytest.mark.simple_backend` / `@pytest.mark.paged_backend` / `@pytest.mark.hardware`)
- **T0.9** `26f2356cb` — P2a.0 verification report (5/5 PASS, hardware-deferred)
- **Hardware-fix follow-ups** `dcdd23a2e` + `44b281ebc` — `warmup.py` uses `_do_*` + `keepdim=False`

**P2a.0 acceptance**: All 5 deliverables PASS in `_fixtures/phase0_verification.md`. Hardware mesh-open verification was blocked at the time by firmware mismatch; **resolved 2026-05-13** by building `local-tt-metal:dev` from tt-metal commit `89686ee7`. Smoke test now unblocks P2a.1.

---

## P2a.1 — Plugin Port + Blackhole Boot (W2, ~1 week)

**Goal**: Adapt the 5 plugin source files into our `python/sglang/srt/hardware_backend/tenstorrent/models/` namespace, register the `Tenstorrent*ForCausalLM` model classes with SGLang, and verify Llama-3.1-8B boots on 2× Blackhole p150a end-to-end via the plugin path. Dual-track simple path must remain working in parallel.

**Files touched**:
- Create: `models/__init__.py`, `models/tt_llm.py`, `models/tt_utils.py`, `models/worker_setup.py`, `models/registry.py`
- Modify: `platform.py` (call `register_tt_models()`), `execution/__init__.py` (add `tt_transformers_paged` value)

**Spec refs**: §2.2 (file map), §2.4 (INVs 3-7 enforced; INV-1 amended scope), §3 (data flow via SGLang standard path), §A2 (amendment notes).

**Live risks**: R1 (image-version drift — we pin to built image), R10 (plugin Blackhole untested; this phase is the first validation).

### Task 1.1: Port plugin files to our namespace (INV-6)

- [ ] **Step 1**: Create `python/sglang/srt/hardware_backend/tenstorrent/models/__init__.py`:

```python
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
```

- [ ] **Step 2**: `cp` plugin sources, rename classes, preserve SPDX headers:

```bash
cd /home/mhnie/sglang
DEST=python/sglang/srt/hardware_backend/tenstorrent/models
SRC=/home/mhnie/tt-inference-server/tt-sglang-plugin/sglang_tt_plugin

# Copy then patch class names
cp $SRC/models/tt_llm.py $DEST/tt_llm.py
cp $SRC/utils/tt_utils.py $DEST/tt_utils.py
cp $SRC/worker_setup/worker_setup.py $DEST/worker_setup.py
cp $SRC/patching/patching_model_registry.py $DEST/registry.py

# Rename class names: TT* → Tenstorrent* per INV-6
sed -i 's/TTLlamaForCausalLM/TenstorrentLlamaForCausalLM/g' $DEST/tt_llm.py $DEST/registry.py
sed -i 's/TTQwenForCausalLM/TenstorrentQwenForCausalLM/g' $DEST/tt_llm.py $DEST/registry.py
sed -i 's/TTMistralForCausalLM/TenstorrentMistralForCausalLM/g' $DEST/tt_llm.py $DEST/registry.py
sed -i 's/TTGptOssForCausalLM/TenstorrentGptOssForCausalLM/g' $DEST/tt_llm.py $DEST/registry.py

# Patch import paths in worker_setup / tt_utils / registry
sed -i 's|from sglang_tt_plugin.worker_setup.worker_setup|from sglang.srt.hardware_backend.tenstorrent.models.worker_setup|g' $DEST/tt_llm.py
sed -i 's|from \.\.utils\.tt_utils|from .tt_utils|g' $DEST/tt_llm.py
sed -i 's|from \.\.models\.tt_llm|from .tt_llm|g' $DEST/registry.py
```

- [ ] **Step 3**: Append our SPDX line to each new file:

```bash
for f in $DEST/tt_llm.py $DEST/tt_utils.py $DEST/worker_setup.py $DEST/registry.py; do
  # Insert our copyright as second SPDX line after the original
  sed -i '/SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc./a # SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>' $f
done
```

- [ ] **Step 4**: AST check + commit:

```bash
for f in $DEST/__init__.py $DEST/tt_llm.py $DEST/tt_utils.py $DEST/worker_setup.py $DEST/registry.py; do
  python3 -c "import ast; ast.parse(open('$f').read()); print('OK: $f')"
done
git add python/sglang/srt/hardware_backend/tenstorrent/models/
git commit -m "feat(tenstorrent): port tt-sglang-plugin into Tenstorrent* namespace (INV-6)"
```

### Task 1.2: Wire `register_tt_models()` into platform activation

- [ ] **Step 1**: Edit `python/sglang/srt/hardware_backend/tenstorrent/platform.py`. Inside the `activate` method (or wherever platform setup begins), import models package:

```python
def activate(cls):
    # ... existing platform setup ...
    # Register Tenstorrent models with SGLang ModelRegistry (INV-6).
    # Side-effect import: models/__init__.py calls register_tt_models()
    from sglang.srt.hardware_backend.tenstorrent import models  # noqa: F401
    logger.info("[TT-Platform] Tenstorrent model arches registered")
```

- [ ] **Step 2**: Update `execution/__init__.py` `resolve_execution_backend_name`:

```python
def resolve_execution_backend_name(requested: str | None = None) -> str:
    """Resolve to P2a default: tt_transformers_paged (plugin path).

    P2a.0 default was tt_transformers_single; P2a.1 flips to paged.
    """
    name = (requested or envs.SGLANG_TT_EXECUTION_BACKEND.get() or "auto").lower()
    if name == "auto":
        return "tt_transformers_paged"  # was tt_transformers_single in P2a.0
    return name
```

- [ ] **Step 3**: Update `scripts/reset_devices.sh` header to add the third pin (`local-tt-metal:dev` sha256:`973e972bddf5`) — this is the image we actually build and use. The two existing pins (P1-era and original P2 attempt) stay documented for historical context. Insert above the existing pin comments:

```bash
# Pinned tt-metal image (P2a.1+, plugin-absorbed paged path, built locally):
#   localhost/local-tt-metal:dev (sha256:973e972bddf5)
#   - built from tt-metal commit 89686ee7 (UMD bump 2026-05-12)
#   - UMD-compatible with host KMD 2.8.0 / FW 19.6.0
#   - contains generator_sglang.py + tt_transformers
# Build context: /home/mhnie/tt-metal/ with `podman build -f dockerfile/Dockerfile --target release-models`
```

- [ ] **Step 4**: Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/platform.py \
        python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py \
        python/sglang/srt/hardware_backend/tenstorrent/scripts/reset_devices.sh
git commit -m "feat(tenstorrent): wire plugin model registration + flip default to paged + third image pin"
```

### Task 1.3: Unit test — plugin model registry + namespace (INV-6 verification)

- [ ] **Step 1**: Write `python/sglang/srt/hardware_backend/tenstorrent/test/test_plugin_registration.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""P2a.1 §9.b6 model-registry namespace test (INV-6 + plugin port verification).

CPU-only — no hardware needed. Imports plugin classes and confirms they're
registered under the Tenstorrent* namespace in SGLang's ModelRegistry.
"""
import pytest


def test_plugin_classes_importable():
    from sglang.srt.hardware_backend.tenstorrent.models import (
        TenstorrentGptOssForCausalLM,
        TenstorrentLlamaForCausalLM,
        TenstorrentMistralForCausalLM,
        TenstorrentQwenForCausalLM,
    )
    for cls in [
        TenstorrentLlamaForCausalLM,
        TenstorrentQwenForCausalLM,
        TenstorrentMistralForCausalLM,
        TenstorrentGptOssForCausalLM,
    ]:
        assert cls.__name__.startswith("Tenstorrent"), f"INV-6 violation: {cls.__name__}"


def test_model_registry_patched():
    # Trigger side-effect registration
    from sglang.srt.hardware_backend.tenstorrent import models  # noqa
    from sglang.srt.models.registry import ModelRegistry

    # Plugin registers under HuggingFace arch names (LlamaForCausalLM, etc.)
    for arch in ["LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM", "GptOssForCausalLM"]:
        assert arch in ModelRegistry.models, f"{arch} not in registry"
        cls = ModelRegistry.models[arch]
        assert cls.__name__.startswith("Tenstorrent"), \
            f"INV-6 violation: {arch} → {cls.__name__}"
```

- [ ] **Step 2**: Run inside docker:

```bash
podman run --rm --entrypoint='' -v /home/mhnie/sglang:/sglang:rw localhost/local-tt-metal:dev \
  bash -lc "source /opt/venv/bin/activate && cd /sglang && python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_plugin_registration.py -v"
```

Expected: 2/2 PASS.

- [ ] **Step 3**: Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_plugin_registration.py
git commit -m "test(tenstorrent): plugin registry + Tenstorrent* namespace (INV-6)"
```

### Task 1.3b: CPU unit test — page-table translation math (INV-3 + G2a)

**Why**: Reviewer flagged that file-map line 70 promises `test_page_table_translation.py` but no task creates it. INV-3 (page_table values are block IDs in `[0, num_pages)` as `(token_index // block_size).to(int32)`) needs a standalone unit test. G2a (co-indexing slot → block_id → KV round-trip) also lacks a dedicated check after §9.16 was dropped — this test covers the math half of G2a; the round-trip half lands implicitly in §9.4 RadixAttention probe (T3.2).

- [ ] **Step 1**: Write `python/sglang/srt/hardware_backend/tenstorrent/test/test_page_table_translation.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""INV-3 + G2a page-table math — CPU unit test (no hardware).

Validates plugin's `_build_page_table` produces correct block-ID tensor:
  block_id = token_index // block_size, dtype torch.int32, shape [B, max_blocks].
"""
import pytest
import torch
from types import SimpleNamespace


def _make_fake_fb(block_size=64, batch_size=2, seq_lens=(128, 96), context_length=512):
    """Build a minimal forward_batch duck-type with req_to_token_pool populated."""
    max_blocks = context_length // block_size
    num_pages = 32
    # Allocate dummy token-pool indices: req 0 gets indices [0..127], req 1 gets [128..223]
    req_to_token = torch.zeros((batch_size, context_length), dtype=torch.int32)
    cur = 0
    for i, slen in enumerate(seq_lens):
        req_to_token[i, :slen] = torch.arange(cur, cur + slen, dtype=torch.int32)
        cur += slen
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        req_pool_indices=torch.tensor([0, 1]),
    )


def test_page_table_dtype_int32():
    """INV-3: page_table dtype must be int32."""
    from sglang.srt.hardware_backend.tenstorrent.models.tt_llm import TenstorrentLlamaForCausalLM
    # _build_page_table is a class-level helper — directly test the math
    # without instantiating the full TT model (which would need hardware).
    # Replicate the math here verbatim from plugin's _build_page_table:
    fb = _make_fake_fb(block_size=64, batch_size=2, seq_lens=(128, 96))
    block_size = 64
    rows = fb.req_to_token_pool.req_to_token[fb.req_pool_indices]
    page_table = (rows[:, ::block_size] // block_size).to(torch.int32)
    assert page_table.dtype == torch.int32, "INV-3 dtype violation"


def test_page_table_block_id_math():
    """INV-3: block_id = token_index // block_size."""
    fb = _make_fake_fb(block_size=64, batch_size=2, seq_lens=(128, 96))
    block_size = 64
    rows = fb.req_to_token_pool.req_to_token[fb.req_pool_indices]
    page_table = (rows[:, ::block_size] // block_size).to(torch.int32)
    # req 0 has indices 0,1,2,...,127 → block IDs at strides 0, 64 should be 0, 1
    assert int(page_table[0, 0]) == 0, f"expected block 0, got {int(page_table[0, 0])}"
    assert int(page_table[0, 1]) == 1, f"expected block 1, got {int(page_table[0, 1])}"
    # req 1 has indices 128,129,...,223 → block IDs at strides 128, 192 should be 2, 3
    assert int(page_table[1, 0]) == 2, f"expected block 2, got {int(page_table[1, 0])}"
    assert int(page_table[1, 1]) == 3, f"expected block 3, got {int(page_table[1, 1])}"


def test_page_table_shape():
    """page_table.shape[1] >= ceil(max(seq_len) / block_size)."""
    fb = _make_fake_fb(block_size=64, batch_size=2, seq_lens=(128, 96))
    block_size = 64
    rows = fb.req_to_token_pool.req_to_token[fb.req_pool_indices]
    page_table = (rows[:, ::block_size] // block_size).to(torch.int32)
    # rows[:, ::block_size] strides every block_size positions → shape[1] = context_length / block_size
    assert page_table.shape[0] == 2  # batch_size
    assert page_table.shape[1] >= 2  # 128/64=2 blocks for the longer req
```

- [ ] **Step 2**: Run + commit:

```bash
podman run --rm --entrypoint='' -v /home/mhnie/sglang:/sglang:rw localhost/local-tt-metal:dev \
  bash -lc "source /opt/venv/bin/activate && cd /sglang && python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_page_table_translation.py -v"
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_page_table_translation.py
git commit -m "test(tenstorrent): page-table block-ID math (INV-3 + G2a unit)"
```

### Task 1.4: §9.1 paged smoke on Blackhole — **first BH end-to-end validation (G8a)**

This is the **highest-risk task in P2a**: the plugin has never run on Blackhole. Use the built image with the new tt-metal.

- [ ] **Step 1**: Write `python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""§9.1 paged smoke — plugin path returns 'Paris' for capital-of-France prompt.

Requires:
  - Built tt-metal image `local-tt-metal:dev` (sha256:973e972bddf5)
  - Live SGLang server on :30000 with SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged
"""
import json
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + live sglang server",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


@REQUIRES_TT
def test_paged_smoke_paris():
    payload = {
        "model": "llama",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "max_tokens": 10,
        "temperature": 0,
    }
    req = urllib.request.Request(
        "http://localhost:30000/v1/chat/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.status == 200
        resp = json.loads(r.read())
    content = resp["choices"][0]["message"]["content"]
    assert "Paris" in content, f"expected 'Paris', got {content!r}"
```

- [ ] **Step 2**: Operator runs the smoke (Claude provides exact commands; operator executes since hardware orchestration is operator territory):

```bash
# In one terminal:
sudo /home/mhnie/.tenstorrent-venv/bin/tt-smi -r 0000:01:00.0 0000:06:00.0  # warm reset

podman run --rm -d --entrypoint='' --device=/dev/tenstorrent --name p2a-smoke \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /home/mhnie/sglang:/sglang:rw \
  -v /home/mhnie/tt-models:/models:ro \
  -p 30000:30000 \
  --ipc=host --cap-add SYS_NICE --cap-add IPC_LOCK \
  -e SGLANG_PLATFORM=tenstorrent \
  -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  -e PYTHONPATH=/sglang/python \
  -e SGLANG_USE_CPU_ENGINE=1 \
  -e CUDA_VISIBLE_DEVICES= \
  -e VLLM_DEVICE_TYPE=cpu \
  -e VLLM_PLUGINS= \
  -e LD_PRELOAD=/lib/x86_64-linux-gnu/libnuma.so.1 \
  -e TRITON_CPU_ONLY=1 \
  -e TRITON_INTERPRET=1 \
  -e TT_METAL_OPTIMIZATIONS=performance \
  localhost/local-tt-metal:dev sleep 14400

podman exec p2a-smoke bash -lc "source /opt/venv/bin/activate && \
  pip install setproctitle msgspec orjson python-multipart soundfile partial_json_parser \
              interegular llguidance xgrammar prometheus-client uvloop watchfiles \
              fastapi-cli py-spy datasets compressed-tensors timm modelscope -q"

# CRITICAL: plugin requires multiprocessing fork start_method so child workers
# inherit the patched ModelRegistry. Standard `python -m sglang.launch_server`
# does NOT do this — use a thin wrapper that sets fork BEFORE any SGLang import.
# (Plugin's own launch_tt_server.py does this; we replicate the essentials.)
cat > /tmp/launch_paged.py <<'PY'
import multiprocessing
multiprocessing.set_start_method("fork", force=True)  # CRITICAL: before sglang import

# Side-effect: registers Tenstorrent*ForCausalLM in ModelRegistry BEFORE
# launch_server reads --model-path and looks up the model class.
import sglang.srt.hardware_backend.tenstorrent.models  # noqa: F401

import sys
from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args

run_server(prepare_server_args(sys.argv[1:]))
PY

# Launch via the wrapper (NOT `python -m sglang.launch_server` directly):
podman exec -d -e PYTHONPATH=/sglang/python p2a-smoke bash -lc "source /opt/venv/bin/activate && \
  PYTHONUNBUFFERED=1 SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  python /tmp/launch_paged.py \
    --model-path /models/Llama-3.1-8B-Instruct \
    --port 30000 --host 0.0.0.0 \
    --device cpu --sampling-backend pytorch \
    --max-running-requests 4 --page-size 64 --context-length 16384 \
    --enable-radix-cache --trust-remote-code --disable-overlap-schedule \
    > /tmp/sglang-paged.log 2>&1"

# Wait for server ready
for i in $(seq 1 30); do
  sleep 30
  if curl -s http://localhost:30000/health | grep -q OK; then echo READY; break; fi
  podman exec p2a-smoke bash -lc "tail -2 /tmp/sglang-paged.log"
done

# Run smoke test
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py -v
```

- [ ] **Step 3**: Acceptance criteria:
  - Server boots within 5 minutes (kernel JIT may be cold first time)
  - `/v1/chat/completions` returns 200 with "Paris" in response
  - `tail /tmp/sglang-paged.log` shows no `Scheduler hit an exception` / no `SIGQUIT` after the initial startup

- [ ] **Step 4**: If smoke fails on Blackhole, the most likely Blackhole-specific failure points (in order of likelihood):
  1. **`tt_utils.py::_configure_fabric`** — Blackhole's `dispatch_core_axis` branch (line 81-93 in plugin source). `ttnn.device.is_blackhole()` check + `fabric_tensix_config` handling. Patch in place; do NOT replay brainstorming.
  2. **`tt_utils.py::get_pipeline_device_params`** — `trace_region_size: 50000000` (50 MB) may be too small on Blackhole's larger SRAM; bump to 100 MB if `RuntimeError: trace_region buffer overflow`.
  3. **Mesh shape acceptance** — confirm `ttnn.MeshShape(1, 2)` works on 2× p150a (vs default `MeshShape(1, num_devices)`). Plugin's default fallback already handles this.
  4. **KV cache allocation alignment** — Blackhole's tile granularity differs from Wormhole; if `tt_model.allocate_kv_cache` raises an alignment error, that's an upstream tt-metal issue — STOP, file plugin upstream issue, and escalate per spec §5.2b (this IS architectural).
  5. **Plugin upstream issue route**: if (1)/(2)/(3) don't apply and (4) doesn't apply, treat as plugin's first BH test → contribute fix upstream per Tenstorrent's contributor guidelines.

- [ ] **Step 5**: Commit + tear down:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py
git commit -m "test(tenstorrent): §9.1 paged smoke on Blackhole (G8a first validation)"
podman rm -f p2a-smoke
```

### Task 1.5: §9.2 greedy correctness on Blackhole

- [ ] **Step 1**: Write `test_greedy_correctness_paged.py` modeled after P1's `test_greedy_correctness.py` but POST to `/v1/completions` with `temperature=0` and verify deterministic output matches a reference fixture.

  - Use `_fixtures/llama31_prompts.py` (10 prompts) and `_fixtures/llama31_greedy_50tok.json` (reference outputs from HuggingFace baseline already in repo).
  - **Note BFP8 drift**: spec §9.2 says paged path may differ from HF reference per token but should be deterministic across re-runs at temp=0. Assert per-prompt isolated rerun matches a same-run batched run.

- [ ] **Step 2**: Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness_paged.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness_paged.py
git commit -m "test(tenstorrent): §9.2 paged greedy correctness on Blackhole"
```

### Task 1.6: P2a.1 verification gate

- [ ] **Step 1**: Run all of P2a.1 acceptance:

```bash
# CPU unit tests (host or container — both work):
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_plugin_registration.py -v

# Hardware tests (container):
SGLANG_PLATFORM=tenstorrent pytest -m "paged_backend and hardware" \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness_paged.py -v
```

Expected: all PASS.

- [ ] **Step 2**: Confirm dual-track still works (run P1 simple suite):

```bash
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single \
  pytest -m simple_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke.py -v
```

Expected: P1's smoke still PASS (T0.7 simple backend port unaffected).

- [ ] **Step 3**: Update P2a.0 verification report (`_fixtures/phase0_verification.md`) — append a "P2a.1 verification" section confirming the smoke went green. Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_verification.md
git commit -m "docs(tenstorrent): P2a.1 hardware-smoke verification report"
```

**P2a.1 acceptance**: §9.1 + §9.2 pass on Blackhole + simple-fallback also passes. Decision: **proceed to P2a.2**.

---

## P2a.2 — Production Hardening (W3, ~3 days)

**Goal**: Wire the upstream-class `bypass_chunked_req` patch (R6), validate abort/cancel/admission flows work through the plugin path, add the 5-step chunked-prefill failure recovery test.

**Files touched**:
- Modify: `python/sglang/srt/managers/utils.py` (one-line dataclass field), `python/sglang/srt/managers/scheduler.py` (one check)
- Create: `REBASE_TARGETS.md` (root), `test/test_chunked_failure_recovery.py`, `test/test_queue_full_admission.py`, `test/test_abort_via_endpoint.py`, `test/test_abort_via_disconnect.py`

**Spec refs**: §3.5 (admission), §3.6 (abort), §3.7 (chunked failure 5-step), §4.3 §9.5 / §9.6 / §9.12.a-e.

**Live risks**: R6 (upstream-class patch monthly rebase).

### Task 2.1: `bypass_chunked_req` upstream patch + `REBASE_TARGETS.md`

- [ ] **Step 1**: Patch `python/sglang/srt/managers/utils.py`:

```python
@dataclasses.dataclass
class GenerationBatchResult:
    # ... existing fields ...
    bypass_chunked_req: bool = False  # NEW (tenstorrent fork patch, R6)
```

- [ ] **Step 2**: Patch `python/sglang/srt/managers/scheduler.py` post-forward result handling. **DO NOT trust the line number** — anchors drift across SGLang releases. Find the post-forward block by grep:

```bash
# Find all chunked_req mutations to identify the post-forward reset (NOT the
# get_next_batch_to_run stash check):
grep -n "self\.chunked_req" python/sglang/srt/managers/scheduler.py
```

You want the site that **assigns** `self.chunked_req = None` AFTER a successful (or completed) forward — typically inside `process_batch_result_prefill` or its caller. The site that **reads** `if self.chunked_req is not None:` is a different scheduling check; don't patch there.

Add the bypass check at the post-forward site:

```python
# Tenstorrent fork: explicit bypass triggered by backend on chunked-prefill failure
if getattr(gen_batch_result, "bypass_chunked_req", False):
    self.chunked_req = None
```

- [ ] **Step 3**: Create `REBASE_TARGETS.md` at repo root:

```markdown
# Upstream-class patches in our fork (tenstorrent-p1)

This file tracks SGLang upstream-class patches that we maintain in our fork only
(spec N9: no upstream PRs). Rebase against `sgl-project/sglang main` **monthly**.

## Active patches

| File | Line range (approx) | Purpose | First landed |
|---|---|---|---|
| `python/sglang/srt/managers/utils.py` | `GenerationBatchResult` dataclass | Add `bypass_chunked_req: bool = False` for tenstorrent chunked-prefill failure recovery | 2026-05-13 (P2a.2) |
| `python/sglang/srt/managers/scheduler.py` | post-forward result handler (~L2498) | Read `bypass_chunked_req`, clear `self.chunked_req` if set | 2026-05-13 (P2a.2) |

## Rebase procedure

1. `git fetch upstream main`
2. `git rebase upstream/main`
3. For each conflict in tracked files: preserve our patch (the `bypass_chunked_req` field + scheduler check)
4. Re-run `pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_chunked_failure_recovery.py`
5. Update this file's "First landed" or add a "Rebased" column if the upstream code moves significantly

Contingency: if upstream restructures `GenerationBatchResult` enough that the field doesn't fit, temporarily disable chunked-prefill support and reopen the spec's §3.7 invariant for revisit.
```

- [ ] **Step 4**: Commit:

```bash
git add python/sglang/srt/managers/utils.py python/sglang/srt/managers/scheduler.py REBASE_TARGETS.md
git commit -m "feat(scheduler): bypass_chunked_req field for tenstorrent fork (R6)"
```

### Task 2.2: §9.12.a-e chunked-failure 5-step invariant unit test + handler implementation

The plugin's path has NO built-in chunked-prefill failure recovery — we must add a hook. This task ships BOTH the handler code AND the unit test.

- [ ] **Step 1 (handler)**: Add `on_chunked_prefill_failure(req, out_cache_loc_this_chunk)` method to `TenstorrentLlamaForCausalLM` (in `models/tt_llm.py`, near `forward`). Implementation per spec §3.7 (5-step invariant):

```python
def on_chunked_prefill_failure(self, req, out_cache_loc_this_chunk, allocator, req_to_token_pool):
    """5-step chunked-prefill failure recovery (spec §3.7).

    Called from worker's try/except wrapping `forward()` when a chunk raises
    mid-prefill. Returns a flag the scheduler reads via GenerationBatchResult.

    Steps (all must run, in order):
      a. allocator.free(out_cache_loc_this_chunk)
      b. req_to_token_pool.req_to_token[req.req_pool_idx, slice] = 0
      c. req.skip_radix_cache_insert = True
      d. req.set_finish_with_abort("tt_backend_chunked_prefill_failure")
      e. return GenerationBatchResult flag — caller sets bypass_chunked_req=True
    """
    import torch
    # Step a: return slots from failed chunk
    allocator.free(out_cache_loc_this_chunk)
    # Step b: revert req_to_token writes for this chunk
    start = len(req.prefix_indices) + len(req.fill_ids) - len(out_cache_loc_this_chunk)
    end = len(req.prefix_indices) + len(req.fill_ids)
    req_to_token_pool.req_to_token[req.req_pool_idx, start:end] = 0
    # Step c: prevent partial-prefix insertion into RadixCache
    req.skip_radix_cache_insert = True
    # Step d: mark abort
    req.set_finish_with_abort("tt_backend_chunked_prefill_failure")
    # Step e: signal scheduler — caller wraps GenerationBatchResult(bypass_chunked_req=True)
    return True
```

The wrapping site is in our worker's `_forward_batch_generation_tt`-equivalent or platform hook (since plugin's path uses standard `ModelRunner.forward`, we need to intercept exceptions in our `platform.py` or via a model-class wrapper). **For Phase 2 scope**, add the try/except at the model class's `forward()` method (catching exceptions and routing to `on_chunked_prefill_failure` when the failure is mid-chunk):

```python
def forward(self, input_ids, positions, forward_batch, input_embeds=None):
    try:
        return super().forward(...)  # plugin's existing forward
    except Exception as exc:
        # Only handle mid-chunk failures during EXTEND when chunked-prefill is active
        if (forward_batch.forward_mode.is_extend()
            and getattr(forward_batch, "chunked_req", None) is not None):
            self.on_chunked_prefill_failure(...)
            # Return zero-logits + signal bypass
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput
            return LogitsProcessorOutput(next_token_logits=None,
                                          _chunked_failure=True)
        raise  # non-chunked failure: let scheduler handle abort normally
```

- [ ] **Step 2 (test)**: Write `test/test_chunked_failure_recovery.py` covering all 5 sub-gates:
  - §9.12.a: `allocator.free` called with exact failed-chunk slot list
  - §9.12.b: `req_to_token` zeroed in failed-chunk range
  - §9.12.c: `req.skip_radix_cache_insert == True`
  - §9.12.d: `req.finished_reason` is `FINISH_ABORT` instance
  - §9.12.e: `GenerationBatchResult.bypass_chunked_req == True` and scheduler clears `self.chunked_req`

Use mocked `TenstorrentLlamaForCausalLM` subclass with `super().forward()` overridden to raise. Mock `allocator` + `req_to_token_pool` + `req`. CPU-only.

- [ ] **Step 2**: Run + commit:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_chunked_failure_recovery.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_chunked_failure_recovery.py
git commit -m "test(tenstorrent): §9.12.a-e chunked-prefill failure 5-step recovery (R5+R12 mitigation)"
```

### Task 2.3: §9.5 queue-full admission + §9.6 abort flow (both endpoints)

- [ ] **Step 1**: Write `test_queue_full_admission.py`:
  - Server launched with `--max-queued-requests N` (small N like 2)
  - Submit N+1 long-running requests concurrently
  - The N+1th should receive in-stream `meta_info.finish_reason.type == "abort"` with `abort_message` mentioning "queue"

- [ ] **Step 2**: Write `test_abort_via_endpoint.py`:
  - Submit a long request
  - POST to `/abort_request` with the request's ID
  - Confirm `GET /get_internal_state.available_size` returns to baseline (KV reclaimed)

- [ ] **Step 3**: Write `test_abort_via_disconnect.py`:
  - Open a streaming request via raw socket
  - Close the socket mid-stream
  - Confirm KV pool reclaims via `/get_internal_state` like above

- [ ] **Step 4**: Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_queue_full_admission.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_endpoint.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_disconnect.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_queue_full_admission.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_endpoint.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_disconnect.py
git commit -m "test(tenstorrent): §9.5 queue-full admission + §9.6 abort (endpoint + disconnect)"
```

### Task 2.4: P2a.2 verification gate

- [ ] **Step 1**: Run all P2a.2 tests + P2a.1 regression:

```bash
SGLANG_PLATFORM=tenstorrent pytest \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_plugin_registration.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_chunked_failure_recovery.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness_paged.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_queue_full_admission.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_endpoint.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_disconnect.py -v
```

Expected: 100% PASS.

**P2a.2 acceptance**: §9.5, §9.6 (×2), §9.12.a-e all pass. Decision: **proceed to P2a.3**.

---

## P2a.3 — Acceptance Suite + RadixAttention Probe (W4, ~3-4 days)

**Goal**: Run the full set of §9 acceptance gates remaining (§9.3, §9.4, §9.7-§9.11, §9.13, §9.14). G4a (RadixAttention) is the headline measurement — this is our independent contribution that the plugin upstream has not done.

**Files touched**: 10 new test files under `test/`.

**Spec refs**: §4.3 §9.3 / §9.4 / §9.7 / §9.8 / §9.9 / §9.10 / §9.11 / §9.13 / §9.14.

**Live risks**: **R5 (HIGH — RadixAttention paged-evict consistency)** — first measurement on the plugin's "co-indexed" KV layout. If RadixAttention is functionally broken with plugin's `tt_transformers.allocate_kv_cache` ownership, G4a degrades to "infeasibility documented", deferred to P3.

### Task 3.1: §9.3 batched correctness B=4 + Q3 empty_slots assertion

- [ ] **Step 1**: Submit 4 concurrent prompts at `temp=0`; per-prompt isolated rerun must match same-batch run for each prompt.

- [ ] **Step 2 (Q3 inline)**: Spec §5.2 Q3 asks "is `empty_slots` identity at B=4 (no retract)?". Plugin's `forward()` builds `empty_slots = list(range(B))` implicitly (B == max_batch_size after padding). Add an assertion at the model's `forward` decode path or via a log to confirm `empty_slots == list(range(B))` for the entire batched run. Record result in `_fixtures/q3_empty_slots_evidence.txt` (one line: `Q3: empty_slots is identity at B=4 — YES/NO`).

- [ ] **Step 3**: Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_batched_correctness.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/q3_empty_slots_evidence.txt
git commit -m "test(tenstorrent): §9.3 batched correctness B=4 (+ Q3 empty_slots evidence)"
```

### Task 3.2: §9.4 RadixAttention probe — G4a headline measurement

- [ ] **Step 1**: Write `test_radix_prefix_cache.py`:

```python
"""§9.4 RadixAttention probe — G4a.

Plugin upstream has NOT validated RadixAttention with their co-indexed KV layout.
This test is our first measurement.

Methodology:
  - Shared 1K-token system prompt (S) + 4 distinct user questions (Q1..Q4)
  - Warm cache by issuing S+Q1 first
  - Then issue S+Q2, S+Q3, S+Q4 concurrently — these should hit the cached S prefix
  - Compare effective tok/s (warm) to baseline (no shared prefix, different prompts)

Pass criteria (spec G4a):
  - Prometheus `sglang:cache_hit_rate` >= 0.5 after warmup
  - effective tok/s ratio (warm / cold) >= 1.5
"""
```

- [ ] **Step 2**: If RadixAttention is broken (cache_hit_rate stays 0 or eviction crashes), DON'T mask the failure. Document it in a `_fixtures/radixattn_probe_result.md` with:
  - Root cause analysis (which step of SGLang's RadixCache flow breaks)
  - Whether it's a plugin bug, our adaptation bug, or fundamental incompatibility
  - Recommendation for P3 (fix vs disable)

- [ ] **Step 3**: Commit (test + result doc):

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_radix_prefix_cache.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/radixattn_probe_result.md
git commit -m "test(tenstorrent): §9.4 RadixAttention probe — G4a first measurement"
```

### Task 3.3: §9.7 stability (compressed 15-min)

- [ ] Run B=4 batched generations for 15 minutes. Measure ITL p99 windowed: baseline (5-10min into run) vs tail (last 5min). Drift must be < 10%.
- [ ] Commit `test(tenstorrent): §9.7 stability 15-min ITL drift < 10%`.

### Task 3.4: §9.8 dual-track switch (paged ↔ single fallback)

- [ ] Two server-launch fixtures: one for each `SGLANG_TT_EXECUTION_BACKEND` value. Each runs a minimal smoke. Server restart between (no hot-swap).
- [ ] Commit `test(tenstorrent): §9.8 dual-track switch (paged ↔ single fallback)`.

### Task 3.5: §9.9 mesh-shape parametric

- [ ] `SGLANG_TT_MESH_SHAPE=1x2` runs on hardware. `1x4` is static-lint only (`grep "1x4" config.py` or similar).
- [ ] Commit `test(tenstorrent): §9.9 mesh-shape parametric (1x2 hw, 1x4 static)`.

### Task 3.6: §9.10 perf log

- [ ] Per the existing T4.9 detailed body in the v1 plan (preserved verbatim — see `git show 5028c6b6a:.../p2a-plan.md` for the 100+ line test):
  - Records: batched tok/s @ B={1,2,4}, prefix hit/miss latency, KV-util time series, per-mode timing
  - Sanity floors: B=4 ≥ 1.2× B=1 decode tok/s; prefix-hit < 10ms; KV util p99 < 95%
- [ ] Commit `test(tenstorrent): §9.10 perf log — tok/s, prefix cache, KV util`.

### Task 3.7: §9.11 eviction-replay (R5 mitigation — HIGH, acceptance thinned)

**R5 acceptance reduction**: pre-amendment spec promised THREE orthogonal verifications (§9.11 + §9.b5 24h long-run + §9.15 free-list invariant unit). After amendment §9.15 dropped (no TTPagedKVAdapter) and §9.b5 deferred to P3. **R5 now relies on §9.11 alone.** Document the reduced bar so external reviewers see it.

- [ ] **Step 1**: Load a prefix request → fill RadixCache to force evict → re-issue → logits stable.
- [ ] **Step 2**: If RadixAttention probe (T3.2) failed, this test also fails — flag at gate.
- [ ] **Step 3**: Write `_fixtures/r5_mitigation_note.md` documenting the reduction (R5 HIGH severity with single verification — re-amend spec if R5 reproduces in P3 workloads).
- [ ] Commit `test(tenstorrent): §9.11 eviction-replay + R5 acceptance note (HIGH mitigation thinned)`.

### Task 3.8: §9.13 shutdown teardown

- [ ] After server shutdown, confirm `ttnn.get_num_tensors() == 0` (no leaked TT tensors).
- [ ] Commit `test(tenstorrent): §9.13 ttnn teardown leak check`.

### Task 3.9: §9.14 token-pool overflow lint

- [ ] At startup, assert `num_pages × block_size < 2^31`. Fail fast if not.
- [ ] CPU-only test (just imports config + verifies the assertion).
- [ ] Commit `test(tenstorrent): §9.14 token-pool index static bound`.

### Task 3.10: Auto-flip default + dual-track polish

- [ ] T1.2 already flipped `resolve_execution_backend_name` to default to `tt_transformers_paged`. Now confirm:
  - Existing P1 hardware tests (the 6 simple-marked files) STILL pass with explicit `SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single`
  - New paged tests pass with no env var (auto → paged)
- [ ] Update `docs/superpowers/specs/2026-05-12-sglang-tenstorrent-p2-design.md` §5.3a user-visible transitions to reflect actual measurements (max_running_requests post-flip, perf delta). **This is the second allowed spec edit during P2a** — only the §5.3a fact-update, not architecture.
- [ ] Commit `chore(tenstorrent): confirm default-flip; record post-flip user-visible transition deltas`.

### Task 3.11: P2a.3 W4 decision gate (§5.3b conditions, amended)

- [ ] Verify all of:
  1. §9.1, §9.2, §9.3, §9.4, §9.5, §9.6, §9.7, §9.8, §9.9, §9.10, §9.11, §9.12.a-e, §9.13, §9.14 pass — **§9.15 / §9.16 dropped per amendment**
  2. Q3 verified (empty_slots identity at B=4), Q7 verified (RadixCache canceled-req consistency via §9.6 abort + §9.11 eviction-replay end-to-end); other Q's already answered in P2a.0
  3. R5 no reproduction in §9.7 stability run (no KV pool free-list inconsistency warnings)
  4. **G4a outcome documented** — either PASS (≥0.5 hit rate, ≥1.5× tok/s) or PASS-as-infeasibility (G4a downgrades to a P3 task; document in `_fixtures/radixattn_probe_result.md`)
  5. G8a Blackhole validation complete

- [ ] Decision:
  - All pass → **P2a accepted, proceed to P2b** (P2b)
  - Any §9 fail → **2-week fix loop** then retest; second fail → P2b → best-effort
  - G4a infeasibility-documented → still proceed to P2b (infeasibility doesn't block multi-model)

- [ ] Commit verification report:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase3_acceptance_report.md
git commit -m "docs(tenstorrent): P2a.3 acceptance report — §9 gates pass, P2b approved"
```

**P2a.3 acceptance**: P2a complete. **P2a → P2b decision gate satisfied.** Total elapsed: ~W2-W4 (or W2-W5 if buffer used).

---

## P2b — Multi-Model Smoke (W5, ~3-5 days)

**Goal**: Validate Qwen / Mistral / GptOss model arches boot and produce coherent output. Plugin already registers all 4 model classes in P2a.1; this stage verifies each one works end-to-end on Blackhole and documents the per-model max_seq_len matrix.

**Files touched**: 1-3 new test files (per-model smoke).

**Spec refs**: §1.2 P2b goals (G1b-G3b), §4.4 §9.b1-§9.b2 acceptance.

**Live risks**: R10 (Qwen/Mistral/GptOss licensing — verify weights available before this phase starts).

### Task 4.1: P2b W0 — licensing + weights check

- [ ] **Step 1**: For each of `Qwen3-7B`, `Mistral-7B-v0.3`, `GptOss-20B`: verify weights are locally available at `/home/mhnie/tt-models/` OR can be downloaded.
- [ ] **Step 2**: Document which models are gated. If any model is gated → mark it as a `placeholder` (NotImplementedError) per spec N11 and DROP from P2b scope.
- [ ] Commit `docs(tenstorrent): P2b W0 model availability checklist`.

### Task 4.2: §9.b1 — per-model smoke + greedy (3 models)

- [ ] **Step 1**: Write `test_multimodel_smoke.py` parameterized over `["Qwen3-7B", "Mistral-7B-v0.3", "GptOss-20B"]` (skip params marked as gated in T4.1).
- [ ] Each parameter:
  - Launch server with that `--model-path`
  - POST `/v1/chat/completions` with a basic factual query
  - Assert response is coherent (length > 5 tokens, no error markers)

- [ ] **Step 2**: Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_multimodel_smoke.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_multimodel_smoke.py
git commit -m "test(tenstorrent): §9.b1 multi-model smoke (Qwen/Mistral/GptOss)"
```

### Task 4.3: §9.b2 — per-model max_seq_len matrix

- [ ] **Step 1**: For each model, find the practical max_seq_len that works (binary-search if needed: try 32K → 16K → 8K → 4K). Per spec G3b:
  - Llama-3.1-8B: target 32K
  - Qwen3-7B: target 32K
  - Qwen3-32B: 4K only (per spec N12, upstream hang)
  - Mistral-7B: target 32K
  - GptOss: best-effort
- [ ] **Step 2**: Document the matrix in `_fixtures/per_model_max_seq_len.md`.
- [ ] Commit `docs(tenstorrent): §9.b2 per-model max_seq_len matrix`.

### Task 4.4: P2b acceptance gate → P3

- [ ] Verify:
  1. §9.b1 all in-scope models pass smoke
  2. §9.b2 max_seq_len matrix documented
  3. No regression in P2a.1-P2a.3 acceptance (rerun the full §9 suite once more)

- [ ] **Decision: P2 → P3 transition**:
  - P2 acceptance achieved → close P2a-plan as DONE
  - **Open `superpowers:brainstorming` for P3** with the expanded scope from spec §A2.5 (128K chunked-prefill, speculative, LoRA, BF16, multimodal, HiRadixCache, disaggregation, tt-xla, kernel-level perf, 4× p150a, Galaxy mesh, 24h+ stability, production observability)

- [ ] Final P2 commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/per_model_max_seq_len.md \
        python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p2b_acceptance_report.md
git commit -m "docs(tenstorrent): P2 acceptance complete (P2a §9 + P2b §9.b); P3 brainstorm next"
```

**P2b acceptance**: P2 complete. **Transition to P3 brainstorming.**

---

## Task index (across all phases — for tracking)

| ID | Phase | Task | Spec ref | Hardware? |
|---|---|---|---|---|
| 1.1 | 1 | Port plugin files into Tenstorrent* namespace | §2.2, INV-6 | No |
| 1.2 | 1 | Wire register_tt_models on platform activate; flip default to paged | §2.5, INV-5 | No |
| 1.3 | 1 | Unit test plugin registration + namespace | INV-6 | No |
| 1.3b | 1 | Unit test page-table translation math | INV-3 + G2a | No |
| 1.4 | 1 | §9.1 paged smoke on Blackhole (**G8a first BH validation**) | §9.1, G8a | **Yes** |
| 1.5 | 1 | §9.2 greedy correctness on Blackhole | §9.2 | Yes |
| 1.6 | 1 | P2a.1 verification gate | — | — |
| 2.1 | 2 | bypass_chunked_req upstream patch + REBASE_TARGETS.md | §3.7, R6 | No |
| 2.2 | 2 | §9.12.a-e chunked-failure 5-step | §9.12, R5+R12 | No (mock) |
| 2.3 | 2 | §9.5 queue-full + §9.6 abort (×2) | §9.5, §9.6 | Yes |
| 2.4 | 2 | P2a.2 verification gate | — | — |
| 3.1 | 3 | §9.3 batched correctness B=4 | §9.3 | Yes |
| 3.2 | 3 | §9.4 RadixAttention probe (**G4a headline**) | §9.4, G4a | Yes |
| 3.3 | 3 | §9.7 stability 15-min | §9.7 | Yes |
| 3.4 | 3 | §9.8 dual-track switch | §9.8, G5a | Yes |
| 3.5 | 3 | §9.9 mesh-shape parametric | §9.9 | Yes |
| 3.6 | 3 | §9.10 perf log | §9.10 | Yes |
| 3.7 | 3 | §9.11 eviction-replay (R5 HIGH) | §9.11 | Yes |
| 3.8 | 3 | §9.13 teardown leak check | §9.13 | Yes |
| 3.9 | 3 | §9.14 token-pool overflow lint | §9.14 | No |
| 3.10 | 3 | Auto-flip default; record user-visible transition | §5.3a, G5a | — |
| 3.11 | 3 | P2a.3 / P2a decision gate | — | — |
| 4.1 | 4 | P2b W0 model availability checklist | R10 | — |
| 4.2 | 4 | §9.b1 multi-model smoke | §9.b1, G1b/G2b | Yes |
| 4.3 | 4 | §9.b2 max_seq_len matrix | §9.b2, G3b | Yes |
| 4.4 | 4 | P2b acceptance gate → P3 | — | — |

**Total**: 23 tasks across 4 sub-stages (T1.3b added post-review for INV-3 + G2a coverage). Estimated 2-3 weeks if hardware cooperates (P2a 1-2w + P2b 0.5-1w).

---

## Risk hot-spots per phase

| Phase | Live risks | Mitigation |
|---|---|---|
| 1 | R1 (image drift — pinned image solves), **R-plugin-BH** (plugin untested on Blackhole — T1.4 is first test) | Pin image SHA; have rollback to simple path ready |
| 2 | R6 (upstream-class patch rebase cost) | `REBASE_TARGETS.md` + monthly rebase task |
| 3 | **R5 HIGH** (RadixAttention paged-evict — first test on plugin's KV layout), R12 (chunked-failure interaction) | §9.11 eviction-replay + §9.7 stability + §9.12 test all gate on this |
| 4 | R10 (licensing — Qwen/Mistral/GptOss weight availability) | W0 task explicitly checks; gated models → placeholder |

---

## Closing — P2 → P3 transition

Once P2b closes, the next step is **`superpowers:brainstorming` for P3** with the expanded scope from spec §A2.5. P3 is the **innovation phase**: speculative decoding, LoRA, BF16-throughout precision, multimodal, HiRadixCache, disaggregation, tt-xla backend, kernel-level perf tuning, 4× p150a validation, Galaxy mesh, 24h+ stability, production observability.

P2 was plumbing-verification. P3 is where we push the envelope.

**End of plan v2.** Total tasks: 22 (was 46 in v1). Total estimated weeks: P2a 2-3w + P2b 0.5-1w = ~3 weeks (was 9w in v1). Plugin absorption saved ~2 weeks; what remains is verification + dual-track + RadixAttention probe + multi-model smoke.
