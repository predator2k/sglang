# SGLang-on-Tenstorrent — Phase 3a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land 24h+ stability + production observability + speculative decoding (NGRAM + EAGLE + EAGLE-adaptive num_steps) on top of P2's plugin-absorbed paged path. Concentrate on production-ready posture before adding feature breadth (P3b+).

**Architecture:** P3a sits on top of P2's plugin path (P2 ACCEPTED 2026-05-13). Wires Tenstorrent inference into SGLang's *native* speculative_algorithm runtime (no custom verification loop — INV-8). EAGLE draft model cohosts on the same 2× Blackhole mesh as the main model (INV-9). New SGLang-fork patches needed for CUDA-hardcoded spec workers (R-P3-3 HIGH). Grafana dashboard JSON exports SGLang Prometheus metrics. 24h synthetic bimodal workload validates throughput drift < 5%.

**Tech Stack:** Python 3.10, SGLang `tenstorrent-p1` branch, `ttnn` / `tt-metal` pinned to commit `89686ee7` (image `localhost/local-tt-metal:dev` sha256:`973e972bddf5`), `tt_transformers` (read-only), Tenstorrent `tt-sglang-plugin` (absorbed into our namespace in P2), Prometheus + Grafana for observability.

**Spec:** [`../specs/2026-05-13-sglang-tenstorrent-p3-design.md`](../specs/2026-05-13-sglang-tenstorrent-p3-design.md). All §N.M / G-item / INV-N / Q-N / R-PN / §10.X references resolve there.

**Predecessor:** P2 shipped (34 commits since amendment, `e5bb937a9..1d98f85b3`). P2a + P2b ACCEPTED. Latest commit before P3a start: `1d98f85b3 docs(tenstorrent): P2 ACCEPTED`. Plan v2 supersedes was at `0600c8c03..a6efdb8d2` (P2a plan).

**Branch:** `tenstorrent-p1` on `origin` (`predator2k/sglang`). NO upstream PRs (spec N9).

**Built image (P2-era, still current):** `localhost/local-tt-metal:dev` — built locally from tt-metal commit `89686ee7`. Includes 3 tt-metal patches applied via `scripts/tt_metal_patches/` at container startup. UMD-compatible with host KMD 2.8.0 / FW 19.6.0.

---

## How to use this plan

1. Sub-stages run **strictly top-to-bottom**: P3a.0 → P3a.1 → P3a.2 → P3a.3 → P3a.4.
2. Each sub-stage has a verification gate. Do NOT proceed past a failing gate.
3. **Commit per task** with the exact message shown.
4. **INV-1..INV-9 from spec §2.4 are non-negotiable.** Any discovery violating an invariant triggers §5.2b brainstorming-skill replay — do NOT mutate architecture inline.
5. **Q1-Q5 from spec §5.2 are surfaced as explicit deliverables** in P3a.0. Each must produce evidence in `_fixtures/p3a_*_evidence.txt`.
6. **Dual-track preservation**: simple path (`SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single`, T0.7 shipped in P2) MUST keep passing P1 hardware tests. P3a only extends the paged path.
7. **Inside docker** (same image as P2): image SHA `973e972bddf5`; venv at `/opt/venv/bin/activate`; tt-metal at `/tt-metal/`; sglang mounted at `/sglang`. P2's `/tmp/launch_paged.py` wrapper pattern reused.
8. Always set `SGLANG_PLATFORM=tenstorrent` + the 7 P2-era env vars (`SGLANG_USE_CPU_ENGINE=1` etc. — see plan v2 §1.4 commands).

---

## File map (locked here)

Paths relative to repo root `/home/mhnie/sglang/`. Module root: `python/sglang/srt/hardware_backend/tenstorrent/`.

### New files (P3a, ~1500 LoC total)

| Path | Source | Responsibility | Size |
|---|---|---|---|
| `models/spec_decode.py` | new | `SpecDecodeAdapter` — verify-mode forward routing. Wraps plugin's `prefill_forward` / `decode_forward`. Handles `forward_batch.spec_info` presence (draft-tree shape). | ~400 |
| `models/eagle_draft.py` | new | EAGLE draft model loading + co-mesh placement (INV-9 Q1). Cohosts smaller draft model alongside main on 2× Blackhole. | ~150 |
| `scripts/grafana_p3a_dashboard.json` | new | Grafana dashboard JSON. 8 panels: queue depth / batched decode tok/s / RadixCache hit rate / ITL p50+p99 / KV pool util / per-model breakdown / spec accept_rate / spec accept_length. | ~700 (data) |
| `scripts/stability_24h_bench.py` | new | 24h workload driver. Bimodal prompt distribution. Per-6h-window throughput collector. Writes `_fixtures/stability_24h_<date>.json`. | ~250 |
| `scripts/stability_6h_pregate.py` | new | Compressed 6h pre-gate driver. Same bench logic, shorter run. Same drift check. | ~80 |

### Modified files (existing P2 code)

| Path | Change | Size |
|---|---|---|
| `models/tt_llm.py` | Add `_resolve_speculative_path()` + `forward()` spec_info branch + `_verify_forward()`. Existing forward path unchanged when `spec_info is None`. | +120 |
| `platform.py` | Add 3 new env-var reads: `SGLANG_TT_SPEC_DRAFT_PATH`, `SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT`, `SGLANG_TT_SPEC_NGRAM_TABLE_PATH`. | +30 |
| `models/registry.py` | (no changes — Tenstorrent* classes already registered in P2) | 0 |

### SGLang upstream-class patches (in our fork only, per N9; HIGH-priority new R-P3-3 patches)

| Path | Change | First landed |
|---|---|---|
| `python/sglang/srt/speculative/ngram_worker.py` | Replace `device = f"cuda:{gpu_id}"` (line 48) + `.cuda()` (line 229) + state-allocation (line 122-143) with device-agnostic paths. On TT plugin path, fall back to `device="cpu"` (host-side n-gram table). | P3a.0 T0.2 |
| `python/sglang/srt/speculative/eagle_info.py` | Replace 6+ `device="cuda"` literals with `device=current_platform.get_device_name()` (or accept `device="cpu"` argument from worker). | P3a.0 T0.2 |
| `python/sglang/srt/speculative/spec_utils.py:587` | Same CUDA hardcoding fix. | P3a.0 T0.2 |

Tracked in `REBASE_TARGETS.md` with monthly rebase cadence per R6.

### Test files (new in P3a, ~500 LoC total)

All under `python/sglang/srt/hardware_backend/tenstorrent/test/`.

| File | Gate | Hardware? |
|---|---|---|
| `test_spec_decode_routing.py` | INV-8 spec_info routing | No (mocked) |
| `test_speculative_ngram_smoke.py` | §10.3 | Yes |
| `test_speculative_ngram_correctness.py` | §10.3b bit-exact | Yes |
| `test_speculative_ngram_perf.py` | §10.4 (≥1.3×) | Yes |
| `test_speculative_eagle_smoke.py` | §10.5 | Yes |
| `test_speculative_eagle_correctness.py` | §10.5b bit-exact | Yes |
| `test_speculative_eagle_perf.py` | §10.6 (calibrated τ) | Yes |
| `test_speculative_adaptive.py` | §10.7 (EAGLE+adaptive vs static) | Yes |
| `test_prometheus_metrics_p3a.py` | §10.8 (6 metric names) | Yes |
| `test_grafana_dashboard.py` | §10.2 (JSON schema) | No |
| `test_stability_6h_pregate.py` | §10.1a (drift < 5% @ 6h) | Yes (long) |
| `test_stability_24h_paged.py` | §10.1 (drift < 5% @ 24h × 4 windows) | Yes (very long) |

---

## P3a.0 — Foundation: spec-runner audit + EAGLE setup (W1, ~1 week)

**Goal:** Resolve R-P3-3 (HIGH — SGLang spec workers CUDA-hardcoded) + Q1 (EAGLE draft model + mesh layout) + Q2/Q3/Q5. Land SGLang fork patches so spec runner is device-agnostic. Decide EAGLE draft model + mesh layout. Produce evidence files for all 5 open questions.

**Files touched:**
- Modify: `python/sglang/srt/speculative/ngram_worker.py`, `eagle_info.py`, `spec_utils.py` (R-P3-3 patches)
- Create: `_fixtures/p3a_open_questions_evidence.txt`, `_fixtures/p3a_eagle_mesh_evidence.txt`
- Update: `REBASE_TARGETS.md`

**Spec refs:** §2.3, §5.2 (Q1-Q5), §5.1 R-P3-1 / R-P3-3.

**Live risks:** R-P3-1 HIGH (EAGLE cohost feasibility unknown), R-P3-3 HIGH (3 SGLang files need patching).

### Task 0.1: Audit SGLang spec runner for CUDA hardcoding + Q2 evidence

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_open_questions_evidence.txt`

- [ ] **Step 1: Grep SGLang spec runtime for CUDA-specific code**

```bash
cd /home/mhnie/sglang
grep -rnE 'cuda|\.cuda\(\)|device\s*=\s*["'\''"]cuda' python/sglang/srt/speculative/ | head -40
```

Expected: at least 20 hits across `ngram_worker.py`, `eagle_info.py`, `spec_utils.py`, `spec_v2_*.py`.

- [ ] **Step 2: Categorize hits**

For each unique file, classify hits:
- **MUST patch** (state allocation, device argument)
- **MUST audit** (dispatched-via-tensor-device — works if tensor is on CPU)
- **SAFE** (string in error messages only)

- [ ] **Step 3: Write evidence file**

```
P3a Phase 0 — Open Questions Evidence (locked at <commit-sha>)
Image: localhost/local-tt-metal:dev (sha256:973e972bddf5)
Captured: <date>

================================================================
Q2: Does SGLang speculative runner have CUDA-specific code paths?
================================================================
Q2 answer: YES — non-trivial. 3 files need device-agnostic patches:
  - speculative/ngram_worker.py:48,122-143,229 — device="cuda:{gpu_id}" + .cuda()
  - speculative/eagle_info.py:<lines> — 6+ device="cuda" literals
  - speculative/spec_utils.py:587 — same pattern

Q2 verdict: R-P3-3 HIGH confirmed. Phase 0 T0.2 patches required.

Q3: Does plugin's forward() need new code path for verify-batch?
================================================================
Q3 answer: YES — forward_batch.spec_info is non-None for verify calls.
The input_ids tensor has draft-tree shape (variable per-step). Existing
plugin forward routes only EXTEND or DECODE; verify needs new branch.

Q5: Memory budget for cohosting 8B + 1B + 4 KV pools?
================================================================
Q5 answer (calculated, not measured): BFP8 weights 8B ≈ 8 GB, 1B ≈ 1 GB.
KV pool 4 (B=4 x 16K context) ≈ 40 GB on TT device. Total ≈ 49 GB.
2× Blackhole p150a have ~64 GB device memory each → 128 GB total.
Should fit; T0.3 verifies empirically.
```

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_open_questions_evidence.txt
git commit -m "docs(tenstorrent): P3a Phase 0 — Q2/Q3/Q5 audit evidence"
```

### Task 0.2: Patch SGLang spec workers for device-awareness (R-P3-3)

**Files:**
- Modify: `python/sglang/srt/speculative/ngram_worker.py`
- Modify: `python/sglang/srt/speculative/eagle_info.py`
- Modify: `python/sglang/srt/speculative/spec_utils.py`
- Modify: `REBASE_TARGETS.md`

- [ ] **Step 1: Patch ngram_worker.py — device dispatch**

Replace at line 48:

```python
# Before:
self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

# After:
import torch
from sglang.srt.platforms.utils import current_platform
if current_platform.device_name == "tenstorrent":
    self.device = "cpu"  # NGRAM table is host-side; TT inference dispatches via tensors
elif gpu_id >= 0:
    self.device = f"cuda:{gpu_id}"
else:
    self.device = "cuda"
```

Replace state allocations at lines 122-143 — wrap all `torch.empty(..., device=self.device)` to use `self.device`.

Replace at line 229:

```python
# Before:
batched_token_ids = torch.tensor(token_ids).cuda()

# After:
batched_token_ids = torch.tensor(token_ids, device=self.device)
```

- [ ] **Step 2: Patch eagle_info.py — device argument propagation**

Search:

```bash
grep -n 'device="cuda"\|device='\''cuda'\''' python/sglang/srt/speculative/eagle_info.py
```

For each hit, replace `device="cuda"` with `device=self.device` (assuming a `self.device` attribute is plumbed through `EagleInfo.__init__`). If `EagleInfo` doesn't have one, add it as `__init__(self, *args, device="cuda", **kwargs)` and propagate from worker.

- [ ] **Step 3: Patch spec_utils.py:587**

Similar — replace the single `device="cuda"` literal with a passed-in argument.

- [ ] **Step 4: AST check + scheduler import check**

```bash
for f in python/sglang/srt/speculative/ngram_worker.py \
         python/sglang/srt/speculative/eagle_info.py \
         python/sglang/srt/speculative/spec_utils.py; do
  python3 -c "import ast; ast.parse(open('$f').read()); print('AST OK: $f')"
done
```

- [ ] **Step 5: Update REBASE_TARGETS.md**

Append:

```markdown
## SGLang speculative-worker patches (P3a.0 T0.2, R-P3-3 HIGH)

| File | Lines | Purpose | First landed |
|---|---|---|---|
| `python/sglang/srt/speculative/ngram_worker.py` | 48, 122-143, 229 | device-agnostic NGRAM worker | 2026-05-13 (P3a.0 T0.2) |
| `python/sglang/srt/speculative/eagle_info.py` | 6+ device="cuda" literals | device dispatch from worker | 2026-05-13 (P3a.0 T0.2) |
| `python/sglang/srt/speculative/spec_utils.py` | 587 | same pattern | 2026-05-13 (P3a.0 T0.2) |
```

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/speculative/ngram_worker.py \
        python/sglang/srt/speculative/eagle_info.py \
        python/sglang/srt/speculative/spec_utils.py \
        REBASE_TARGETS.md
git commit -m "fix(sglang): device-agnostic NGRAM/EAGLE workers for non-CUDA platforms (R-P3-3)"
```

### Task 0.3: EAGLE draft model selection + mesh layout (Q1 / INV-9 / R-P3-1)

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_eagle_mesh_evidence.txt`

- [ ] **Step 1: Check whether Llama-3.2-1B is locally available**

```bash
ls /home/mhnie/tt-models/ | grep -i -E 'Llama-3.2|llama-3.2|1B'
```

If present → Llama-3.1-8B (main) + Llama-3.2-1B (draft) is the candidate pair. Otherwise document the gap as a P3a.0 blocker (we need to download weights or pick another pair).

- [ ] **Step 2: Mesh layout experiment (operator-led — Claude provides commands)**

This is the highest-risk part of P3a (R-P3-1 HIGH). Spin up a container with the main model, then attempt to instantiate a SECOND `TenstorrentLlamaForCausalLM` for the draft. Two layout candidates:

**Layout A — shared full (1,2) mesh, interleaved kernel scheduling:**

```python
# Inside Python in container:
from sglang.srt.hardware_backend.tenstorrent.models.tt_llm import TenstorrentLlamaForCausalLM
# Build main model normally
main = TenstorrentLlamaForCausalLM(config=llama_8b_config, ...)
# Try to build draft on the SAME mesh
draft = TenstorrentLlamaForCausalLM(config=llama_1b_config, mesh_device=main.mesh_device, ...)
# Does instantiation succeed? Does first prefill on each work?
```

**Layout B — split mesh, main on device 0, draft on device 1:**

```python
import ttnn
mesh_main = ttnn.open_mesh_device(ttnn.MeshShape(1,1), device_ids=[0])
mesh_draft = ttnn.open_mesh_device(ttnn.MeshShape(1,1), device_ids=[1])
# Build each on its own
```

- [ ] **Step 3: Record evidence**

`_fixtures/p3a_eagle_mesh_evidence.txt`:

```
P3a Phase 0 Q1 — EAGLE draft cohost experiment
Captured: <date>
Hardware: 2× Tenstorrent Blackhole p150a (BDFs 0000:01:00.0 + 0000:06:00.0)

Layout A — shared (1,2) mesh:
  Attempted: <date+time>
  Result: <PASS / FAIL with stack>
  Memory after both loaded: <X GB free vs 64 GB total>

Layout B — split mesh:
  Attempted: <date+time>
  Result: <PASS / FAIL with stack>

DECISION: Layout <A or B> chosen for P3a.2 EAGLE work.

If both fail:
  R-P3-1 INFEASIBLE. EAGLE downgraded to P3b/P3d (defer until 4× p150a or
  Galaxy mesh available). G4/§10.5/§10.6/§10.10 become DEFERRED gates.
  P3a still ships with G1-G3 + G5(adaptive on EAGLE-disabled config).
```

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_eagle_mesh_evidence.txt
git commit -m "docs(tenstorrent): P3a Phase 0 — EAGLE cohost layout decision (Q1)"
```

### Task 0.4: P3a.0 verification gate

- [ ] **Step 1: Verify all evidence files exist**

```bash
ls python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_*_evidence.txt
```

Expected: 2 files (`p3a_open_questions_evidence.txt`, `p3a_eagle_mesh_evidence.txt`).

- [ ] **Step 2: Verify SGLang patches AST-clean**

```bash
python3 -c "import ast
for f in ['python/sglang/srt/speculative/ngram_worker.py',
          'python/sglang/srt/speculative/eagle_info.py',
          'python/sglang/srt/speculative/spec_utils.py']:
    ast.parse(open(f).read())
print('all AST OK')"
```

- [ ] **Step 3: Verify REBASE_TARGETS.md has new section**

```bash
grep -A2 "P3a.0 T0.2" REBASE_TARGETS.md
```

- [ ] **Step 4: Decision gate**

If Layout A or B PASSed → proceed to P3a.1 with chosen layout recorded.
If both FAILed → R-P3-1 INFEASIBLE. Update spec §1.1 G4/§10.5/§10.6/§10.10 to DEFERRED, then proceed to P3a.1 with NGRAM-only path.

**P3a.0 acceptance:** All 5 open questions answered in evidence. R-P3-3 patches landed. EAGLE feasibility known (PASS or documented infeasibility).

---

## P3a.1 — NGRAM speculative integration (W2-W3, ~2 weeks)

**Goal:** Wire SGLang's NGRAM speculative_algorithm into the plugin path. Add `SpecDecodeAdapter` + plugin's `forward()` verify-batch branch. Hit §10.3 (smoke), §10.3b (bit-exact), §10.4 (≥1.3× perf).

**Files touched:**
- Create: `models/spec_decode.py`
- Modify: `models/tt_llm.py`, `platform.py`
- Create: 4 test files

**Spec refs:** §2.5, §3.2, §4.2 §10.3 / §10.3b / §10.4. INV-8 (use native config).

**Live risks:** R-P3-4 (perf threshold), Q3 verify-batch shape (answered in P3a.0).

### Task 1.1: SpecDecodeAdapter skeleton (verify-batch routing)

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/models/spec_decode.py`

- [ ] **Step 1: Write the failing test**

`python/sglang/srt/hardware_backend/tenstorrent/test/test_spec_decode_routing.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""INV-8 spec_info routing test.

CPU-only — verifies SpecDecodeAdapter correctly dispatches verify-batches
to _verify_forward, and standard batches to existing prefill/decode.
"""
import pytest
import torch
from types import SimpleNamespace


def test_spec_decode_adapter_routes_verify():
    from sglang.srt.hardware_backend.tenstorrent.models.spec_decode import SpecDecodeAdapter
    adapter = SpecDecodeAdapter(model=SimpleNamespace())
    fake_batch = SimpleNamespace(
        spec_info=SimpleNamespace(draft_tokens=torch.tensor([1, 2, 3])),
        forward_mode=SimpleNamespace(is_extend=lambda: False, is_decode=lambda: False),
        input_ids=torch.tensor([10, 20, 30]),
        positions=torch.tensor([0, 1, 2]),
    )
    # Adapter should call _verify_forward; mock it
    adapter._verify_forward = lambda fb: "verify_path"
    adapter._standard_forward = lambda fb: "standard_path"
    assert adapter.forward(fake_batch) == "verify_path"


def test_spec_decode_adapter_routes_standard():
    from sglang.srt.hardware_backend.tenstorrent.models.spec_decode import SpecDecodeAdapter
    adapter = SpecDecodeAdapter(model=SimpleNamespace())
    fake_batch = SimpleNamespace(
        spec_info=None,
        forward_mode=SimpleNamespace(is_extend=lambda: True, is_decode=lambda: False),
        input_ids=torch.tensor([10, 20, 30]),
        positions=torch.tensor([0, 1, 2]),
    )
    adapter._verify_forward = lambda fb: "verify_path"
    adapter._standard_forward = lambda fb: "standard_path"
    assert adapter.forward(fake_batch) == "standard_path"
```

- [ ] **Step 2: Run test — expect FAIL (module doesn't exist)**

```bash
podman run --rm --entrypoint='' -v /home/mhnie/sglang:/sglang:rw localhost/local-tt-metal:dev \
  bash -lc "source /opt/venv/bin/activate && cd /sglang && python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_spec_decode_routing.py -v"
```

Expected: 2 ERROR / FAIL with `ModuleNotFoundError: spec_decode`.

- [ ] **Step 3: Create spec_decode.py**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""SpecDecodeAdapter — verify-batch routing for plugin path.

INV-8: P3a uses SGLang's native speculative_algorithm runtime. This adapter
is a THIN wrapper — it routes forward_batch.spec_info presence to
_verify_forward (draft-tree shape), or to _standard_forward (existing
prefill/decode path).

Does NOT implement a verification loop. SGLang's spec scheduler owns that.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class SpecDecodeAdapter:
    """Wraps a TenstorrentLlamaForCausalLM with verify-batch routing.

    Usage (called from TenstorrentLlamaForCausalLM.forward):
        if getattr(forward_batch, "spec_info", None) is not None:
            return self._spec_adapter.forward(forward_batch)
        # else: existing standard forward path
    """

    def __init__(self, model):
        self._model = model

    def forward(self, forward_batch: "ForwardBatch") -> "LogitsProcessorOutput":
        spec_info = getattr(forward_batch, "spec_info", None)
        if spec_info is not None:
            return self._verify_forward(forward_batch)
        return self._standard_forward(forward_batch)

    def _verify_forward(self, forward_batch: "ForwardBatch") -> "LogitsProcessorOutput":
        """Verify draft-proposed tokens.

        forward_batch.spec_info carries draft tree; forward_batch.input_ids is
        the flattened draft-token sequence; forward_batch.positions is per-
        draft-token position. We call plugin's prefill_forward in "small
        prefill" mode (it returns logits per draft position).
        """
        return self._model._call_prefill_for_verify(forward_batch)

    def _standard_forward(self, forward_batch: "ForwardBatch") -> "LogitsProcessorOutput":
        """Delegate to model's existing forward path."""
        return self._model._call_standard_forward(forward_batch)
```

- [ ] **Step 4: Run test — expect PASS**

```bash
podman run --rm --entrypoint='' -v /home/mhnie/sglang:/sglang:rw localhost/local-tt-metal:dev \
  bash -lc "source /opt/venv/bin/activate && cd /sglang && python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_spec_decode_routing.py -v"
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/models/spec_decode.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/test_spec_decode_routing.py
git commit -m "feat(tenstorrent): SpecDecodeAdapter — INV-8 verify-batch routing"
```

### Task 1.2: tt_llm.py forward() spec_info branch + _call_* methods

**Files:**
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/models/tt_llm.py`

- [ ] **Step 1: Read existing tt_llm.py `forward` to find insertion point**

```bash
grep -n "def forward" python/sglang/srt/hardware_backend/tenstorrent/models/tt_llm.py | head -5
```

Locate the `TTModels.forward` method. We need to add `_spec_adapter` field in `__init__` + a spec_info early-exit in `forward`.

- [ ] **Step 2: Add spec_adapter to TTModels.__init__**

In `TTModels.__init__`, near the end:

```python
        # P3a: SpecDecodeAdapter for speculative-decode verify-batch routing
        from sglang.srt.hardware_backend.tenstorrent.models.spec_decode import SpecDecodeAdapter
        self._spec_adapter = SpecDecodeAdapter(self)
```

- [ ] **Step 3: Add spec_info branch to TTModels.forward**

At the top of the existing `forward()` method (after `self._last_chunked_failure = False` reset):

```python
        # P3a: route speculative verify-batch to SpecDecodeAdapter
        if getattr(forward_batch, "spec_info", None) is not None:
            return self._spec_adapter.forward(forward_batch)
```

- [ ] **Step 4: Add _call_prefill_for_verify and _call_standard_forward helpers**

```python
    def _call_prefill_for_verify(self, forward_batch):
        """Called from SpecDecodeAdapter for verify-batch.

        Reuses plugin's prefill_forward path with forward_batch's flattened
        draft-token sequence. Returns LogitsProcessorOutput with logits per
        draft position.
        """
        # Build prefill inputs from spec_info
        # input_ids has shape [total_draft_tokens] (flat across batch)
        page_table = self._build_page_table(forward_batch)
        padded_tokens = self._flatten_to_padded(forward_batch.input_ids, forward_batch)
        # prompt_lens for verify = draft sequence lengths (from spec_info)
        prompt_lens = forward_batch.spec_info.draft_seq_lens if hasattr(
            forward_batch.spec_info, "draft_seq_lens"
        ) else forward_batch.extend_seq_lens

        logits = self.tt_model.prefill_forward(
            tokens=padded_tokens.to(torch.int32),
            page_table=page_table,
            kv_cache=self.kv_caches,
            prompt_lens=prompt_lens,
        )
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))

    def _call_standard_forward(self, forward_batch):
        """Existing prefill/decode path. Used when no spec_info."""
        # This wraps the original forward branches (EXTEND / DECODE).
        # We extract them to a helper for clarity.
        if forward_batch.forward_mode.is_extend():
            return self._do_prefill(forward_batch)
        elif forward_batch.forward_mode.is_decode():
            return self._do_decode(forward_batch)
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")
```

Note: `_do_prefill` / `_do_decode` should be extracted from the existing forward body. Move the EXTEND branch and DECODE branch into separate methods.

- [ ] **Step 5: AST check + run existing P2 tests to ensure no regression**

```bash
python3 -c "import ast; ast.parse(open('python/sglang/srt/hardware_backend/tenstorrent/models/tt_llm.py').read()); print('AST OK')"

podman run --rm --entrypoint='' -v /home/mhnie/sglang:/sglang:rw localhost/local-tt-metal:dev \
  bash -lc "source /opt/venv/bin/activate && cd /sglang && python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_spec_decode_routing.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_plugin_registration.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_page_table_translation.py -v"
```

Expected: ALL still pass (no regression).

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/models/tt_llm.py
git commit -m "feat(tenstorrent): tt_llm.py — spec_info routing via SpecDecodeAdapter (INV-8)"
```

### Task 1.3: §10.3 NGRAM smoke test

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_smoke.py`

- [ ] **Step 1: Write the test**

```python
# SPDX-License-Identifier: Apache-2.0
"""§10.3 NGRAM smoke test.

Requires the server to be running with --speculative-algorithm NGRAM.
"""
import json
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server with NGRAM spec on :30000",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


@REQUIRES_TT
def test_ngram_smoke_paris():
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
    assert "Paris" in content
```

- [ ] **Step 2: Operator runs (instructions)**

Operator relaunches the server with NGRAM spec:

```bash
# Inside container:
podman exec p3a-smoke pkill -9 -f sglang || true
sleep 3
podman exec -d p3a-smoke bash -lc "source /opt/venv/bin/activate && \
  PYTHONUNBUFFERED=1 SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  python /tmp/launch_paged.py \
    --model-path /models/Llama-3.1-8B-Instruct --port 30000 --host 0.0.0.0 \
    --device cpu --sampling-backend pytorch \
    --max-running-requests 4 --page-size 64 --context-length 16384 \
    --trust-remote-code --disable-overlap-schedule \
    --speculative-algorithm NGRAM --speculative-num-steps 5 \
    --enable-metrics \
    > /tmp/sglang-ngram.log 2>&1"

# Wait for ready
for i in $(seq 1 30); do
  sleep 15
  if curl -s http://localhost:30000/health | grep -q OK; then echo READY; break; fi
done

# Run smoke
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_smoke.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_smoke.py
git commit -m "test(tenstorrent): §10.3 NGRAM smoke on Blackhole"
```

### Task 1.4: §10.3b NGRAM bit-exact correctness test

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_correctness.py`

- [ ] **Step 1: Write the test**

```python
# SPDX-License-Identifier: Apache-2.0
"""§10.3b NGRAM bit-exact correctness test.

Speculative MUST be a perf optimization, never a quality regression.
At temperature=0, NGRAM output must token-for-token equal non-speculative.

Methodology:
  1. For each of 10 fixed prompts, request /v1/completions with temp=0
  2. Compare token sequence (decoded text) between two server runs:
     - NGRAM server (currently up)
     - Non-spec baseline server (operator re-launches without --speculative-algorithm)
  
For automation, this test assumes the operator runs it TWICE: once when
NGRAM server is up, once when baseline is up. Outputs are persisted to
_fixtures/ngram_correctness_{ngram,baseline}.json and compared offline.
"""
import json
import os
import pathlib
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]

PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "The first three prime numbers are",
    "Albert Einstein was born in",
    "The chemical symbol for gold is",
    "The speed of light in vacuum is",
    "Photosynthesis is the process by which",
    "The largest planet in our solar system is",
    "World War II ended in",
    "The author of 1984 is",
]


def _complete(prompt: str) -> str:
    payload = {"model": "llama", "prompt": prompt, "max_tokens": 30, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        "http://localhost:30000/v1/completions", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["text"]


@REQUIRES_TT
def test_ngram_outputs_recorded():
    """Records outputs from currently-running server.

    Operator runs this with both NGRAM server and baseline server; the
    fixture file collects both. After both runs, the comparison test runs.
    """
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    fixture_dir.mkdir(exist_ok=True)
    mode = os.environ.get("NGRAM_CORRECTNESS_MODE", "ngram")  # or "baseline"
    out_file = fixture_dir / f"ngram_correctness_{mode}.json"

    results = {}
    for p in PROMPTS:
        results[p] = _complete(p)

    out_file.write_text(json.dumps(results, indent=2))
    assert len(results) == 10


def test_ngram_matches_baseline():
    """Compares two recorded runs. Skipped if either is missing."""
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    ngram_f = fixture_dir / "ngram_correctness_ngram.json"
    baseline_f = fixture_dir / "ngram_correctness_baseline.json"
    if not (ngram_f.exists() and baseline_f.exists()):
        pytest.skip("Both recordings missing; operator must run both")

    ngram = json.loads(ngram_f.read_text())
    baseline = json.loads(baseline_f.read_text())

    mismatches = []
    for prompt, ngram_out in ngram.items():
        baseline_out = baseline.get(prompt)
        if baseline_out != ngram_out:
            mismatches.append((prompt, ngram_out, baseline_out))

    assert not mismatches, f"NGRAM bit-exactness violated for {len(mismatches)} prompts: {mismatches[:3]}"
```

- [ ] **Step 2: Operator runs (instructions)**

```bash
# Run 1: NGRAM mode (server should be running with --speculative-algorithm NGRAM)
SGLANG_PLATFORM=tenstorrent NGRAM_CORRECTNESS_MODE=ngram \
  pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_correctness.py::test_ngram_outputs_recorded -v

# Operator relaunches WITHOUT --speculative-algorithm:
podman exec p3a-smoke pkill -9 -f sglang || true; sleep 3
podman exec -d p3a-smoke bash -lc "source /opt/venv/bin/activate && \
  PYTHONUNBUFFERED=1 SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  python /tmp/launch_paged.py \
    --model-path /models/Llama-3.1-8B-Instruct --port 30000 --host 0.0.0.0 \
    --device cpu --sampling-backend pytorch \
    --max-running-requests 4 --page-size 64 --context-length 16384 \
    --trust-remote-code --disable-overlap-schedule \
    > /tmp/sglang-baseline.log 2>&1"
# wait for ready

# Run 2: Baseline mode
SGLANG_PLATFORM=tenstorrent NGRAM_CORRECTNESS_MODE=baseline \
  pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_correctness.py::test_ngram_outputs_recorded -v

# Run 3: Compare
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_correctness.py::test_ngram_matches_baseline -v
```

Expected: All 3 PASS.

- [ ] **Step 3: Commit (test + 2 fixture files)**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_correctness.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/ngram_correctness_*.json
git commit -m "test(tenstorrent): §10.3b NGRAM bit-exact correctness (10 prompts vs baseline)"
```

### Task 1.5: §10.4 NGRAM perf test (≥1.3× throughput)

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_perf.py`

- [ ] **Step 1: Write the test**

```python
# SPDX-License-Identifier: Apache-2.0
"""§10.4 NGRAM perf test — ≥1.3× decode tok/s vs no-spec baseline.

Operator runs twice (NGRAM server up, baseline server up); test compares.
"""
import json
import os
import pathlib
import statistics
import time
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


def _measure_decode_tok_s(prompt: str, max_tokens: int, n_runs: int = 5) -> float:
    """Return median decode tok/s across n_runs of the same prompt."""
    timings = []
    for _ in range(n_runs):
        payload = {"model": "llama", "prompt": prompt, "max_tokens": max_tokens,
                   "temperature": 0, "stream": False}
        t0 = time.perf_counter()
        req = urllib.request.Request(
            "http://localhost:30000/v1/completions", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        elapsed = time.perf_counter() - t0
        decode_tokens = resp["usage"]["completion_tokens"]
        # subtract ~50ms prefill from elapsed
        decode_s = max(elapsed - 0.05, 1e-6)
        timings.append(decode_tokens / decode_s)
    return statistics.median(timings)


@REQUIRES_TT
def test_ngram_perf_measure():
    """Records decode tok/s; operator runs in both NGRAM + baseline mode."""
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    fixture_dir.mkdir(exist_ok=True)
    mode = os.environ.get("NGRAM_PERF_MODE", "ngram")
    out_file = fixture_dir / f"ngram_perf_{mode}.json"

    prompt = "Explain how a transformer model processes input tokens. " * 5  # ~40 tokens
    tok_s = _measure_decode_tok_s(prompt, max_tokens=200, n_runs=5)
    out_file.write_text(json.dumps({"decode_tok_s_median": tok_s, "n_runs": 5}, indent=2))
    assert tok_s > 0


def test_ngram_perf_speedup():
    """Asserts ≥1.3× speedup vs baseline. Skipped if both recordings missing."""
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    n = fixture_dir / "ngram_perf_ngram.json"
    b = fixture_dir / "ngram_perf_baseline.json"
    if not (n.exists() and b.exists()):
        pytest.skip("Both recordings missing")
    ngram_tok_s = json.loads(n.read_text())["decode_tok_s_median"]
    baseline_tok_s = json.loads(b.read_text())["decode_tok_s_median"]
    speedup = ngram_tok_s / baseline_tok_s
    print(f"NGRAM tok/s: {ngram_tok_s:.2f}, baseline tok/s: {baseline_tok_s:.2f}, speedup: {speedup:.2f}×")
    assert speedup >= 1.3, f"NGRAM speedup {speedup:.2f}× < 1.3× threshold"
```

- [ ] **Step 2: Operator runs both modes (as in T1.4)** then runs comparison test.

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_perf.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/ngram_perf_*.json
git commit -m "test(tenstorrent): §10.4 NGRAM perf ≥1.3× speedup vs no-spec baseline"
```

### Task 1.6: P3a.1 verification gate

- [ ] **Step 1: Run all P3a.1 tests**

```bash
podman exec p3a-smoke bash -lc "source /opt/venv/bin/activate && cd /sglang && \
  SGLANG_PLATFORM=tenstorrent python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_spec_decode_routing.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_smoke.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_correctness.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_perf.py \
    -v 2>&1 | tail -30"
```

Expected: all pass (some may skip if operator hasn't dual-recorded yet).

- [ ] **Step 2: P2 regression check**

```bash
podman exec p3a-smoke bash -lc "source /opt/venv/bin/activate && cd /sglang && \
  SGLANG_PLATFORM=tenstorrent python -m pytest -m paged_backend \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness_paged.py \
    -v 2>&1 | tail -10"
```

Expected: still PASS (no regression).

**P3a.1 acceptance:** §10.3, §10.3b, §10.4 all PASS. Proceed to P3a.2.

---

## P3a.2 — EAGLE speculative integration (W4-W5, ~2 weeks)

**Goal:** Wire EAGLE through plugin path with draft model cohosting on 2× Blackhole mesh (per P3a.0 Q1 decision). Hit §10.5 (smoke), §10.5b (bit-exact), §10.6 (calibrated τ), §10.10 (mesh evidence).

**Files touched:**
- Create: `models/eagle_draft.py`
- Modify: `models/tt_llm.py` (extend SpecDecodeAdapter hookups)
- Create: 3 test files

**Spec refs:** §1.1 G4, §2.4 INV-9, §4.2 §10.5/§10.5b/§10.6/§10.10. R-P3-1 HIGH.

**Live risks:** R-P3-1 (resolved or deferred in P3a.0).

### Task 2.1: eagle_draft.py — draft model load + cohost

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/models/eagle_draft.py`

If P3a.0 T0.3 documented Layout A or B PASSing, implement that layout here. If INFEASIBLE, skip P3a.2 entirely (per spec §5.3b decision-gate fallback) and document deferral.

- [ ] **Step 1: Create eagle_draft.py**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""EAGLE draft model loader with co-mesh placement (INV-9).

Per P3a.0 Q1 evidence (`_fixtures/p3a_eagle_mesh_evidence.txt`),
the chosen mesh layout is documented there.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def load_eagle_draft(main_model, draft_path: str | None = None):
    """Load an EAGLE draft model that cohosts on main_model's mesh.

    Args:
        main_model: TenstorrentLlamaForCausalLM (or similar) instance
        draft_path: path to draft model weights. If None, reads from
                    SGLANG_TT_SPEC_DRAFT_PATH env var.

    Returns:
        A draft model instance, ready for SGLang spec runner.
    """
    draft_path = draft_path or os.environ.get("SGLANG_TT_SPEC_DRAFT_PATH")
    if not draft_path:
        raise ValueError("SGLANG_TT_SPEC_DRAFT_PATH not set and no draft_path argument")

    layout = os.environ.get("SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT", "shared")
    logger.info(f"[EAGLE-draft] loading from {draft_path}, mesh_layout={layout}")

    # Import lazily so non-EAGLE paths don't drag this in
    from sglang.srt.hardware_backend.tenstorrent.models.tt_llm import TenstorrentLlamaForCausalLM
    from transformers import AutoConfig
    draft_config = AutoConfig.from_pretrained(draft_path)

    if layout == "shared":
        # Cohost on the same (1, 2) mesh as main
        return TenstorrentLlamaForCausalLM(
            config=draft_config,
            mesh_device=main_model.mesh_device,  # SHARE mesh
            quant_config=None,
            tt_model=None,  # build fresh tt_model for draft
        )
    elif layout == "split":
        # Open a new (1, 1) mesh on device 1 (main on device 0)
        import ttnn
        draft_mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), device_ids=[1])
        return TenstorrentLlamaForCausalLM(
            config=draft_config,
            mesh_device=draft_mesh,
            quant_config=None,
            tt_model=None,
        )
    else:
        raise ValueError(f"Unknown SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT={layout}; expected shared|split")
```

- [ ] **Step 2: AST check + commit**

```bash
python3 -c "import ast; ast.parse(open('python/sglang/srt/hardware_backend/tenstorrent/models/eagle_draft.py').read()); print('AST OK')"
git add python/sglang/srt/hardware_backend/tenstorrent/models/eagle_draft.py
git commit -m "feat(tenstorrent): eagle_draft.py — co-mesh EAGLE draft loader (INV-9 Q1)"
```

### Task 2.2: §10.5 EAGLE smoke test

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_smoke.py`

- [ ] **Step 1: Write the test (mirror §10.3 NGRAM smoke pattern)**

```python
# SPDX-License-Identifier: Apache-2.0
"""§10.5 EAGLE smoke test."""
import json
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server with EAGLE spec on :30000",
)
pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


@REQUIRES_TT
def test_eagle_smoke_paris():
    payload = {
        "model": "llama",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "max_tokens": 10, "temperature": 0,
    }
    req = urllib.request.Request(
        "http://localhost:30000/v1/chat/completions", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.status == 200
        resp = json.loads(r.read())
    assert "Paris" in resp["choices"][0]["message"]["content"]
```

- [ ] **Step 2: Operator launches EAGLE server**

```bash
# Operator relaunches server with EAGLE flags:
podman exec p3a-smoke pkill -9 -f sglang || true; sleep 3
podman exec -d p3a-smoke bash -lc "source /opt/venv/bin/activate && \
  PYTHONUNBUFFERED=1 SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_SPEC_DRAFT_PATH=/models/Llama-3.2-1B-Instruct \
  SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT=shared \
  python /tmp/launch_paged.py \
    --model-path /models/Llama-3.1-8B-Instruct --port 30000 --host 0.0.0.0 \
    --device cpu --sampling-backend pytorch \
    --max-running-requests 4 --page-size 64 --context-length 16384 \
    --trust-remote-code --disable-overlap-schedule \
    --speculative-algorithm EAGLE --speculative-draft-model-path /models/Llama-3.2-1B-Instruct \
    --speculative-num-steps 5 --enable-metrics \
    > /tmp/sglang-eagle.log 2>&1"
# Wait for ready (will take longer — 2 models load)
```

- [ ] **Step 3: Run test**

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_smoke.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_smoke.py
git commit -m "test(tenstorrent): §10.5 EAGLE smoke on Blackhole"
```

### Task 2.3: §10.5b EAGLE bit-exact correctness test

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_correctness.py`

- [ ] **Step 1: Write the test (mirror §10.3b NGRAM correctness pattern)**

Same structure as `test_speculative_ngram_correctness.py` but with mode env var `EAGLE_CORRECTNESS_MODE={eagle,baseline}`. Reuse the 10 PROMPTS list.

- [ ] **Step 2: Operator records both modes + runs comparison**

(Same flow as T1.4 for NGRAM, but with EAGLE server + baseline.)

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_correctness.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/eagle_correctness_*.json
git commit -m "test(tenstorrent): §10.5b EAGLE bit-exact correctness vs baseline"
```

### Task 2.4: §10.6 EAGLE perf test (calibrated τ)

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_perf.py`

- [ ] **Step 1: Write the test (two-phase calibrate + measure)**

```python
# SPDX-License-Identifier: Apache-2.0
"""§10.6 EAGLE perf — calibrated τ threshold from 10-prompt subset.

Phase 1: calibrate τ baseline from 10 prompts (recorded once)
Phase 2: measure τ across 100 prompts; assert τ ≥ 0.85× baseline
"""
import json
import os
import pathlib
import urllib.request
import statistics

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server with EAGLE",
)
pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


def _measure_accept_length(prompts: list[str], max_tokens: int) -> float:
    """Query Prometheus /metrics for sglang:spec_accept_length after running prompts."""
    # First, run all prompts:
    for p in prompts:
        urllib.request.urlopen(urllib.request.Request(
            "http://localhost:30000/v1/completions", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"model": "llama", "prompt": p, "max_tokens": max_tokens,
                             "temperature": 0, "stream": False}).encode("utf-8"),
        ), timeout=120)

    # Then scrape /metrics for sglang:spec_accept_length
    metrics_resp = urllib.request.urlopen("http://localhost:30000/metrics", timeout=10).read().decode()
    for line in metrics_resp.split("\n"):
        if line.startswith("sglang:spec_accept_length"):
            # parse the value (last token on the line)
            return float(line.split()[-1])
    raise AssertionError("sglang:spec_accept_length not found in /metrics")


# 10 prompts for calibration:
CALIB = [
    "Explain transformers in three sentences.",
    "What is the chemical symbol for gold?",
    "Why is the sky blue?",
    "Name three classical composers.",
    "What does HTTP stand for?",
    "How many continents are there?",
    "What is the speed of light?",
    "Who painted the Mona Lisa?",
    "What is photosynthesis?",
    "How do bees make honey?",
]


@REQUIRES_TT
def test_eagle_calibrate_tau():
    """Phase 1: record τ baseline from 10 prompts. Run once."""
    tau = _measure_accept_length(CALIB, max_tokens=50)
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    fixture_dir.mkdir(exist_ok=True)
    (fixture_dir / "eagle_tau_calibration.json").write_text(
        json.dumps({"calibration_tau": tau, "n_prompts": len(CALIB)}, indent=2))
    print(f"τ_calibration = {tau:.3f}")
    assert tau >= 1.0, "τ < 1.0 is degenerate"


@REQUIRES_TT
def test_eagle_perf_100_prompts():
    """Phase 2: measure τ across 100 prompts; assert ≥ 0.85× calibration baseline."""
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    calib_f = fixture_dir / "eagle_tau_calibration.json"
    if not calib_f.exists():
        pytest.skip("Run test_eagle_calibrate_tau first")
    calib_tau = json.loads(calib_f.read_text())["calibration_tau"]

    # Generate 100 prompts (10 calib prompts × 10 paraphrases — simple)
    prompts = []
    for base in CALIB:
        for i in range(10):
            prompts.append(f"{base} (variation {i})")
    tau = _measure_accept_length(prompts, max_tokens=50)
    threshold = 0.85 * calib_tau
    print(f"τ_measured = {tau:.3f}, threshold = {threshold:.3f}")
    assert tau >= threshold, f"τ {tau:.3f} < threshold {threshold:.3f} (0.85× of calibration {calib_tau:.3f})"
```

- [ ] **Step 2: Operator runs phase 1 then phase 2 against EAGLE server.**

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_perf.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/eagle_tau_calibration.json
git commit -m "test(tenstorrent): §10.6 EAGLE perf — calibrated τ threshold (R-P3-4)"
```

### Task 2.5: §10.10 mesh layout evidence + P3a.2 verification gate

- [ ] **Step 1: Confirm `_fixtures/p3a_eagle_mesh_evidence.txt` reflects actual implementation (not just P3a.0 plan)**

Update with: actual layout used + measured memory + any surprises encountered during EAGLE bring-up.

- [ ] **Step 2: Run all P3a.2 tests + P3a.1 regression**

```bash
podman exec p3a-smoke bash -lc "source /opt/venv/bin/activate && cd /sglang && \
  SGLANG_PLATFORM=tenstorrent python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_smoke.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_correctness.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_eagle_perf.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_ngram_smoke.py \
    -v 2>&1 | tail -20"
```

- [ ] **Step 3: Commit verification report update**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_eagle_mesh_evidence.txt
git commit -m "docs(tenstorrent): P3a.2 EAGLE acceptance — §10.5/§10.5b/§10.6/§10.10 PASS"
```

**P3a.2 acceptance:** §10.5, §10.5b, §10.6, §10.10 all PASS. Proceed to P3a.3.

---

## P3a.3 — Adaptive + observability (W6, ~1 week)

**Goal:** Enable `speculative_adaptive` flag (EAGLE num_steps tuner). Add Grafana dashboard JSON (8 panels). Confirm SGLang Prometheus metrics names.

**Files touched:**
- Create: `scripts/grafana_p3a_dashboard.json`
- Create: 3 test files

**Spec refs:** §1.1 G5, §4.2 §10.2/§10.7/§10.8.

### Task 3.1: §10.7 EAGLE-adaptive test

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_adaptive.py`

- [ ] **Step 1: Write the test**

```python
# SPDX-License-Identifier: Apache-2.0
"""§10.7 EAGLE + speculative_adaptive comparison.

Note: speculative_adaptive is NOT a standalone algorithm; it's a runtime
tuner of EAGLE's num_steps (server_args.py:580).

Compares:
  - EAGLE with --speculative-num-steps 3 (static, well-tuned per T2.4)
  - EAGLE with --speculative-adaptive (dynamic num_steps)

Asserts: adaptive throughput within 5% of best static.
"""
import json
import os
import pathlib
import statistics
import time
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server",
)
pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


def _bench(prompt: str, max_tokens: int, n_runs: int = 5) -> float:
    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        urllib.request.urlopen(urllib.request.Request(
            "http://localhost:30000/v1/completions", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"model": "llama", "prompt": prompt, "max_tokens": max_tokens,
                             "temperature": 0, "stream": False}).encode("utf-8"),
        ), timeout=120).read()
        elapsed = time.perf_counter() - t0
        timings.append(max_tokens / max(elapsed - 0.05, 1e-6))
    return statistics.median(timings)


@REQUIRES_TT
def test_adaptive_bench_recorder():
    """Records bench result for current server mode.

    Operator runs twice (static EAGLE, adaptive EAGLE) and compares.
    """
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    fixture_dir.mkdir(exist_ok=True)
    mode = os.environ.get("ADAPTIVE_MODE", "static")  # or "adaptive"
    tok_s = _bench("Explain quantum entanglement briefly.", max_tokens=100)
    (fixture_dir / f"adaptive_bench_{mode}.json").write_text(
        json.dumps({"decode_tok_s": tok_s, "mode": mode}, indent=2))
    print(f"[{mode}] tok/s = {tok_s:.2f}")


def test_adaptive_within_5pct_of_static():
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    s = fixture_dir / "adaptive_bench_static.json"
    a = fixture_dir / "adaptive_bench_adaptive.json"
    if not (s.exists() and a.exists()):
        pytest.skip("Both recordings missing")
    static_tok_s = json.loads(s.read_text())["decode_tok_s"]
    adaptive_tok_s = json.loads(a.read_text())["decode_tok_s"]
    ratio = adaptive_tok_s / static_tok_s
    print(f"static={static_tok_s:.2f}, adaptive={adaptive_tok_s:.2f}, ratio={ratio:.3f}")
    assert 0.95 <= ratio <= 1.10, f"adaptive {adaptive_tok_s:.2f} vs static {static_tok_s:.2f} ratio {ratio:.3f}: not within 5%"
```

- [ ] **Step 2: Operator runs in both modes:**

```bash
# Static mode: server up with --speculative-num-steps 3
SGLANG_PLATFORM=tenstorrent ADAPTIVE_MODE=static pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_adaptive.py::test_adaptive_bench_recorder -v

# Adaptive mode: relaunch with --speculative-adaptive added (bool flag)
# Then:
SGLANG_PLATFORM=tenstorrent ADAPTIVE_MODE=adaptive pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_adaptive.py::test_adaptive_bench_recorder -v

# Compare:
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_adaptive.py::test_adaptive_within_5pct_of_static -v
```

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_adaptive.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/adaptive_bench_*.json
git commit -m "test(tenstorrent): §10.7 EAGLE-adaptive vs static num_steps comparison"
```

### Task 3.2: §10.8 Prometheus metrics test

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_prometheus_metrics_p3a.py`

- [ ] **Step 1: Write the test**

```python
# SPDX-License-Identifier: Apache-2.0
"""§10.8 Prometheus metric names exported when --enable-metrics."""
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server with --enable-metrics",
)
pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]

REQUIRED_METRICS = [
    "sglang:spec_accept_length",
    "sglang:spec_accept_rate",
    "sglang:cache_hit_rate",
    "sglang:queue_size",
    "sglang:running_requests",
    "sglang:processed_tokens_per_second",
]


@REQUIRES_TT
def test_metrics_endpoint_returns_required_names():
    resp = urllib.request.urlopen("http://localhost:30000/metrics", timeout=10).read().decode()
    missing = []
    for m in REQUIRED_METRICS:
        if m not in resp:
            missing.append(m)
    assert not missing, f"Missing metrics: {missing}"
```

- [ ] **Step 2: Run test against EAGLE-adaptive server (with --enable-metrics already set)**

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_prometheus_metrics_p3a.py
git commit -m "test(tenstorrent): §10.8 required Prometheus metric names exported"
```

### Task 3.3: Grafana dashboard JSON (§10.2)

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/scripts/grafana_p3a_dashboard.json`

- [ ] **Step 1: Construct dashboard JSON**

A Grafana dashboard JSON has top-level keys: `title`, `panels` (array), `time`, `templating`, etc. Each panel has `targets` (Prometheus PromQL queries).

The 8 required panels:

```json
{
  "title": "SGLang on Tenstorrent — P3a",
  "tags": ["sglang", "tenstorrent"],
  "schemaVersion": 38,
  "version": 1,
  "panels": [
    {"id": 1, "title": "Queue depth", "type": "graph",
     "targets": [{"expr": "sglang:queue_size"}]},
    {"id": 2, "title": "Batched decode tok/s", "type": "graph",
     "targets": [{"expr": "rate(sglang:processed_tokens_per_second[1m])"}]},
    {"id": 3, "title": "RadixCache hit rate", "type": "graph",
     "targets": [{"expr": "sglang:cache_hit_rate"}]},
    {"id": 4, "title": "ITL p50/p99", "type": "graph",
     "targets": [
       {"expr": "histogram_quantile(0.50, rate(sglang:inter_token_latency_seconds_bucket[1m]))"},
       {"expr": "histogram_quantile(0.99, rate(sglang:inter_token_latency_seconds_bucket[1m]))"}
     ]},
    {"id": 5, "title": "KV pool utilization", "type": "graph",
     "targets": [{"expr": "sglang:running_requests / sglang:max_running_requests"}]},
    {"id": 6, "title": "Per-arch breakdown (Llama vs Qwen3)", "type": "graph",
     "targets": [{"expr": "rate(sglang:processed_tokens_per_second[1m]) by (model)"}]},
    {"id": 7, "title": "Speculative accept_rate", "type": "graph",
     "targets": [{"expr": "sglang:spec_accept_rate"}]},
    {"id": 8, "title": "Speculative accept_length τ", "type": "graph",
     "targets": [{"expr": "sglang:spec_accept_length"}]}
  ]
}
```

- [ ] **Step 2: Schema validation test**

`test_grafana_dashboard.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""§10.2 Grafana dashboard JSON schema check."""
import json
import pathlib


def test_dashboard_parses():
    f = pathlib.Path(__file__).parent.parent / "scripts" / "grafana_p3a_dashboard.json"
    data = json.loads(f.read_text())
    assert "panels" in data
    assert len(data["panels"]) >= 8, f"Expected ≥8 panels, got {len(data['panels'])}"
    for p in data["panels"]:
        assert "title" in p
        assert "targets" in p


def test_dashboard_has_required_metric_queries():
    f = pathlib.Path(__file__).parent.parent / "scripts" / "grafana_p3a_dashboard.json"
    data = json.loads(f.read_text())
    queries = []
    for p in data["panels"]:
        for t in p["targets"]:
            queries.append(t["expr"])
    body = "\n".join(queries)
    for required in ["sglang:queue_size", "sglang:cache_hit_rate", "sglang:spec_accept_length"]:
        assert required in body, f"Missing query for {required}"
```

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/scripts/grafana_p3a_dashboard.json \
        python/sglang/srt/hardware_backend/tenstorrent/test/test_grafana_dashboard.py
git commit -m "feat(tenstorrent): §10.2 Grafana dashboard JSON (8 panels) + schema test"
```

### Task 3.4: P3a.3 verification gate

- [ ] **Step 1: Run all P3a.3 tests**

```bash
podman exec p3a-smoke bash -lc "source /opt/venv/bin/activate && cd /sglang && \
  SGLANG_PLATFORM=tenstorrent python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_speculative_adaptive.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_prometheus_metrics_p3a.py \
    python/sglang/srt/hardware_backend/tenstorrent/test/test_grafana_dashboard.py \
    -v 2>&1 | tail -15"
```

**P3a.3 acceptance:** §10.2, §10.7, §10.8 all PASS. Proceed to P3a.4.

---

## P3a.4 — Stability + acceptance (W7-W8, ~2 weeks)

**Goal:** 6h compressed pre-gate (§10.1a) on W7. Full 24h stability run (§10.1) on W8. P2 §9 regression. Acceptance report.

**Files touched:**
- Create: `scripts/stability_24h_bench.py`, `scripts/stability_6h_pregate.py`
- Create: 2 test files

**Spec refs:** §1.1 G1, §4.2 §10.1/§10.1a/§10.9.

### Task 4.1: stability_6h_pregate.py + §10.1a test

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/scripts/stability_6h_pregate.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_stability_6h_pregate.py`

- [ ] **Step 1: Write the workload driver**

```python
# SPDX-License-Identifier: Apache-2.0
# scripts/stability_6h_pregate.py
"""§10.1a compressed 6h pre-gate driver.

Bimodal prompt distribution: 80% [1024, 8192], 20% [10240, 12288].
Concurrent B=4. Per-6h-window throughput collector. Drift gate < 5%.
"""
import argparse
import json
import os
import random
import string
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def gen_prompt(length: int) -> str:
    # Random words ~5 chars each
    words = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(length // 6 + 1)]
    return ' '.join(words)[:length * 4]  # approx 4 chars per token


def gen_request_payload():
    if random.random() < 0.20:
        prompt_len = random.randint(10240, 12288)
    else:
        prompt_len = random.randint(1024, 8192)
    max_tokens = random.randint(200, 1000)
    return {"model": "llama", "prompt": gen_prompt(prompt_len), "max_tokens": max_tokens, "temperature": 0.7}


def send(payload):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(
            "http://localhost:30000/v1/completions", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        ), timeout=300) as r:
            resp = json.loads(r.read())
        return {"ok": True, "elapsed": time.perf_counter() - t0,
                "completion_tokens": resp["usage"]["completion_tokens"],
                "prompt_tokens": resp["usage"]["prompt_tokens"]}
    except Exception as e:
        return {"ok": False, "elapsed": time.perf_counter() - t0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--window-hours", type=float, default=1.5)  # 4 windows in 6h
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    end_time = time.time() + args.duration_hours * 3600
    results = []
    windows = [[] for _ in range(int(args.duration_hours / args.window_hours))]
    window_idx = 0
    window_start = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        in_flight = []
        while time.time() < end_time:
            while len(in_flight) < args.concurrency:
                in_flight.append(ex.submit(send, gen_request_payload()))
            done, in_flight_set = set(), set()
            for f in in_flight:
                if f.done():
                    done.add(f)
                else:
                    in_flight_set.add(f)
            in_flight = list(in_flight_set)
            for f in done:
                r = f.result()
                r["window"] = window_idx
                results.append(r)
                # Advance window
                if time.time() - window_start > args.window_hours * 3600:
                    window_idx = min(window_idx + 1, len(windows) - 1)
                    window_start = time.time()
            time.sleep(0.1)

    # Aggregate per-window throughput
    per_window = {}
    for r in results:
        if r["ok"]:
            per_window.setdefault(r["window"], []).append(r["completion_tokens"] / r["elapsed"])
    out = {
        "total_requests": len(results),
        "errors": sum(1 for r in results if not r["ok"]),
        "per_window_median_tok_s": {w: float(sum(t) / len(t)) for w, t in per_window.items()},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the test wrapper**

```python
# test_stability_6h_pregate.py
"""§10.1a — kick off 6h pre-gate run, assert drift < 5%."""
import json
import os
import pathlib
import subprocess

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires running server",
)
pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


@REQUIRES_TT
def test_6h_pregate():
    out = pathlib.Path("/tmp/p3a_6h_pregate.json")
    # Run the bench (this takes 6h — be sure!)
    subprocess.check_call([
        "python",
        "python/sglang/srt/hardware_backend/tenstorrent/scripts/stability_6h_pregate.py",
        "--duration-hours", "6", "--concurrency", "4",
        "--window-hours", "1.5",
        "--output", str(out),
    ], timeout=7 * 3600)
    data = json.loads(out.read_text())
    assert data["errors"] == 0
    per_window = data["per_window_median_tok_s"]
    baseline = per_window["0"]
    tail = per_window[max(per_window.keys(), key=int)]
    drift = abs(tail - baseline) / baseline
    assert drift < 0.05, f"drift {drift:.3f} > 5%"
```

- [ ] **Step 3: Commit** (the actual run is operator-driven and takes 6h+ wall time)

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/scripts/stability_6h_pregate.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/test_stability_6h_pregate.py
git commit -m "test(tenstorrent): §10.1a 6h compressed pre-gate driver + drift check"
```

### Task 4.2: stability_24h_bench.py + §10.1 test

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/scripts/stability_24h_bench.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_stability_24h_paged.py`

- [ ] **Step 1: Adapt the 6h script to 24h with 4×6h windows**

(Same as 6h script but with `--duration-hours 24 --window-hours 6`.)

- [ ] **Step 2: Test wrapper that asserts drift across 4×6h windows**

(Same pattern as `test_stability_6h_pregate.py`.)

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/scripts/stability_24h_bench.py \
        python/sglang/srt/hardware_backend/tenstorrent/test/test_stability_24h_paged.py
git commit -m "test(tenstorrent): §10.1 24h stability driver + 4-window drift gate"
```

### Task 4.3: §10.9 P2 §9 regression rerun

- [ ] **Step 1: Run the P2 §9 suite (all P2a + P2b tests)**

```bash
podman exec p3a-smoke bash -lc "source /opt/venv/bin/activate && cd /sglang && \
  SGLANG_PLATFORM=tenstorrent python -m pytest \
    python/sglang/srt/hardware_backend/tenstorrent/test/ \
    -m 'paged_backend and not adaptive' --ignore=stability_* -v 2>&1 | tail -30"
```

Expected: all P2 tests still PASS.

- [ ] **Step 2: Commit acceptance report**

```bash
echo "P2 §9 regression run: <date>, X/X passed" >> python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_verification.md
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_verification.md
git commit -m "docs(tenstorrent): P3a §10.9 P2 regression rerun PASS"
```

### Task 4.4: P3a decision gate

**Files:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_acceptance_report.md`

- [ ] **Step 1: Verify all P3a §10 gates**

Walk §5.3b decision gate conditions:
1. §10.1a + §10.1-§10.10 all PASS (or G4 documented infeasibility)
2. R-P3-1 resolved
3. P2 §9 regression PASS
4. INV-1..INV-9 NOT violated (read evidence file)
5. Q1-Q5 each have explicit answer (evidence file)
6. Decision evidence file committed

- [ ] **Step 2: Write acceptance report**

```markdown
# P3a Acceptance Report — <date>

## §10 gate scorecard

| Gate | Result | Notes |
|---|---|---|
| §10.1a | <PASS/FAIL> | 6h compressed pre-gate drift |
| §10.1  | <PASS/FAIL> | 24h stability drift < 5% |
| §10.2  | PASS | Grafana 8-panel JSON valid |
| §10.3  | PASS | NGRAM smoke |
| §10.3b | PASS | NGRAM bit-exact (10 prompts) |
| §10.4  | <PASS/FAIL> | NGRAM ≥1.3× tok/s |
| §10.5  | <PASS/FAIL/DEFER> | EAGLE smoke |
| §10.5b | <PASS/FAIL/DEFER> | EAGLE bit-exact |
| §10.6  | <PASS/FAIL/DEFER> | EAGLE τ ≥ 0.85× cal |
| §10.7  | <PASS/FAIL> | Adaptive vs static within 5% |
| §10.8  | PASS | 6 Prometheus metric names |
| §10.9  | PASS | P2 §9 regression |
| §10.10 | PASS | Mesh layout evidence |

## Q1-Q5 evidence

(Reference `_fixtures/p3a_open_questions_evidence.txt` + `_fixtures/p3a_eagle_mesh_evidence.txt`)

## §5.3b decision gate

✅ All conditions met (or documented partial). Decision: **P3a ACCEPTED → proceed to P3b**.
```

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_acceptance_report.md
git commit -m "docs(tenstorrent): P3a ACCEPTED → proceed to P3b (features wave)"
```

**P3a acceptance:** All §10 gates PASS or documented partial. **P3a → P3b decision gate satisfied.** Push to origin + transition to P3b brainstorm.

---

## Task index

| ID | Sub-stage | Task | Spec ref | Hardware? |
|---|---|---|---|---|
| 0.1 | P3a.0 | Audit SGLang spec runner + Q2 evidence | §5.2 Q2 | No |
| 0.2 | P3a.0 | Patch CUDA hardcoding (3 files) | §2.3, R-P3-3 | No |
| 0.3 | P3a.0 | EAGLE mesh layout (Q1) | §2.4 INV-9, R-P3-1 | Yes |
| 0.4 | P3a.0 | Verification gate | — | — |
| 1.1 | P3a.1 | SpecDecodeAdapter routing | INV-8 | No |
| 1.2 | P3a.1 | tt_llm.py spec_info branch | — | No |
| 1.3 | P3a.1 | §10.3 NGRAM smoke | §10.3 | Yes |
| 1.4 | P3a.1 | §10.3b NGRAM bit-exact | §10.3b | Yes |
| 1.5 | P3a.1 | §10.4 NGRAM perf ≥1.3× | §10.4 | Yes |
| 1.6 | P3a.1 | Verification gate | — | — |
| 2.1 | P3a.2 | eagle_draft.py loader | INV-9 | Yes |
| 2.2 | P3a.2 | §10.5 EAGLE smoke | §10.5 | Yes |
| 2.3 | P3a.2 | §10.5b EAGLE bit-exact | §10.5b | Yes |
| 2.4 | P3a.2 | §10.6 EAGLE calibrated τ | §10.6 | Yes |
| 2.5 | P3a.2 | §10.10 + verification gate | §10.10 | — |
| 3.1 | P3a.3 | §10.7 adaptive vs static | §10.7 | Yes |
| 3.2 | P3a.3 | §10.8 Prometheus metric names | §10.8 | Yes |
| 3.3 | P3a.3 | Grafana JSON + §10.2 | §10.2 | No |
| 3.4 | P3a.3 | Verification gate | — | — |
| 4.1 | P3a.4 | §10.1a 6h pre-gate driver + test | §10.1a | Yes (long) |
| 4.2 | P3a.4 | §10.1 24h stability driver + test | §10.1 | Yes (very long) |
| 4.3 | P3a.4 | §10.9 P2 regression rerun | §10.9 | Yes |
| 4.4 | P3a.4 | P3a decision gate | §5.3b | — |

**Total: 23 tasks across 5 sub-stages.** Estimated 6-8 weeks per spec §5.3 timeline.

---

## Risk hot-spots per sub-stage

| Sub-stage | Live risks | Mitigation |
|---|---|---|
| P3a.0 | **R-P3-1 HIGH** (EAGLE cohost feasibility), **R-P3-3 HIGH** (3 SGLang patches) | T0.2 explicit budget for 3-file patches; T0.3 produces feasibility evidence; INFEASIBLE fallback documented |
| P3a.1 | R-P3-4 (perf threshold flexible) | NGRAM smoke easy; perf threshold checked against measured baseline |
| P3a.2 | R-P3-1 carry-over (mesh layout chosen in P3a.0) | If P3a.0 INFEASIBLE → skip P3a.2 entirely, document |
| P3a.3 | (low — wiring + JSON) | Prometheus metric names must match SGLang upstream |
| P3a.4 | R-P3-2 (24h drift surprise) | 6h pre-gate (T4.1) before full 24h; iterate if drift > 5% |

---

## Closing — P3a → P3b transition

Once Phase P3a.4 closes, next step is **`superpowers:brainstorming` for P3b** (features wave per spec §A1: LoRA, multimodal, 128K chunked-prefill, alerting, etc.).

**End of plan.** Total tasks: 23 across 5 sub-stages. Estimated 6-8 weeks of work.
