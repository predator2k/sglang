# SGLang-on-Tenstorrent — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an end-to-end single-prompt Llama-3.1-8B BF16 inference server on SGLang, running TP=2 across two Tenstorrent p150a cards via `ttnn.mesh_device` and ETH fabric, as defined by the validated spec.

**Architecture:** Black-box integration. SGLang owns scheduler / HTTP / sampling-call-site; a `tt_transformers` model instance (built via `models.tt_transformers.tt.common.create_tt_model` returning a `Generator`/`LlamaForCausalLM`, verified during Phase 0) owns the model forward and KV cache. New code lives in-tree under `python/sglang/srt/hardware_backend/tenstorrent/`. Pattern mirrors the MLX backend line-by-line: worker overrides `forward_batch_generation`, ModelRunner is a bookkeeping stub, `attn_backend=None`, `sampler=None`, greedy sampling happens in the worker on host torch tensors.

**Tech Stack:** Python 3.10, SGLang main, `ttnn` / `tt-metal` (bundled in dev docker image), `tt_transformers`, PyTorch (host-side for sampling and tensor bridging), `pytest` for unit tests.

**Spec:** [`/home/mhnie/sglang/docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md`](../specs/2026-05-11-sglang-tenstorrent-p1-design.md). All section references (§N.M) below point into that spec — do not re-derive architecture.

**Reference backend:** `/home/mhnie/sglang/python/sglang/srt/hardware_backend/mlx/` — every new TT file has an MLX twin. When in doubt, open the twin and follow its shape.

---

## How to use this plan

1. Work the phases top-to-bottom. **Each phase has a verification command that gates the next phase** — do not move on until verification passes.
2. Phase 0 is verification-only (no code). Phases A → I are implementation.
3. Inside docker: run `source /home/container_app_user/tt-metal/python_env/bin/activate && cd /sglang/python && pip install -e .` first. Use `SGLANG_PLATFORM=tenstorrent` for any TT runs.
4. Commit at end of each Task (granularity ≈ one logical unit; see git messages below). Never amend prior commits.
5. **If verification fails, do not paper over it — stop and diagnose.** Re-read the relevant spec §, then either fix the bug or escalate (decision gates listed per-phase).
6. Spec §12 lists open questions and §10 lists risks — re-anchor on these before each phase. The most live ones are flagged inline under each phase.

---

## Phase 0 — Prerequisites (Verification only, no code)

Spec §2 enumerates 8 prerequisites. Each is a hard go/no-go. **Block all coding until every box in this phase is checked.** Capture command output in a `phase0-evidence.txt` scratch file so verifications are auditable.

**Live risks:** §10 #1 (tt_transformers Llama support), §10 #10 (HF gate), §12 #7 ("torch version mismatch should be hard go/no-go").

### Task 0.1: Confirm Llama-3.1-8B runs on the bundled `tt_transformers` demo

- [ ] **Step 1:** Inside the dev docker image (`ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-55fd115-aa4ae1e`), source the venv and locate the demo script. Run:

```bash
source /home/container_app_user/tt-metal/python_env/bin/activate
HF_MODEL=/models/Llama-3.1-8B-Instruct \
pytest -s -v models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1"
```

**There is no `--model` pytest flag** (confirmed 2026-05-11 — see `phase0-evidence.txt`). The demo selects the model purely from the `HF_MODEL` env var. The directory pointed to must contain HF-format weights (`config.json`, safetensors shards, tokenizer files) — the existing `$HOME/tt-models/Qwen3-1.7B` directory is a layout reference.

Expected: demo completes, generates non-empty output, prints decode toks/s near the `P300_Llama-3.1-8B: 38` target (line 1086 of `simple_text_demo.py`).

- [ ] **Step 2:** If the test fails or hangs at mesh open, run `sudo tt-smi -r 0000:01:00.0 0000:06:00.0` and retry once. Two failures in a row → STOP. Risk #1 fired; renegotiate the spec target (do not silently swap to Qwen3).

### Task 0.2: Capture `create_tt_model` / `Generator` / `LlamaForCausalLM` API signatures

**Earlier plan draft referenced `tt_transformers.models.llama3.Llama_3_1`. That class/module does NOT exist** in the pinned docker image (verified 2026-05-11). The actual public surface is in `models.tt_transformers.tt.{generator, generator_vllm, common}`.

- [ ] **Step 1:** Inside docker shell:

```bash
python -c "
import inspect
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM
from models.tt_transformers.tt.common import create_tt_model
print('Generator:', Generator)
print('  methods:', [m for m in dir(Generator) if not m.startswith('_')])
print('LlamaForCausalLM:', LlamaForCausalLM)
print('  methods:', [m for m in dir(LlamaForCausalLM) if not m.startswith('_')])
print('create_tt_model signature:', inspect.signature(create_tt_model))
"
```

Expected: prints all three. Capture the exact method names that perform prefill, decode, KV-cache management — these drive Phase F.

- [ ] **Step 2:** Append the captured method-name list to `phase0-evidence.txt`. Phase F.1 / F.2 will reference these names when implementing `TTTransformersExecutionBackend.extend` / `decode_step` (the P1 impl of the `TTExecutionBackend` ABC — see Phase F header for the spec amendment). Likely mapping (verify against actual `Generator` API):
  - `backend.extend(req_id)` → `generator.prefill_forward(...)` or similar
  - `backend.decode_step(req_id, tok)` → `generator.decode_forward(...)` or similar
  - `backend.new_request(req_id, prompt)` → allocate per-request state (may need our own bookkeeping if Generator doesn't expose per-req allocate/free)
  - `backend.free(req_id)` → release per-request state

- [ ] **Step 3:** If `Generator` does not expose enough public methods to drive prefill/decode externally (e.g. only a `generate()` that owns the whole loop), **STOP and escalate** — Risk #1 has fired in a different form. The wrapper layer cannot drive prefill/decode externally, and we'd need to upstream hooks into `tt_transformers` or redesign.

### Task 0.3: Llama-3.1-8B-Instruct weights staged

- [ ] **Step 1:** On host (outside docker):

```bash
huggingface-cli login                          # only if not already logged in
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir $HOME/tt-models/Llama-3.1-8B-Instruct
```

Expected: `$HOME/tt-models/Llama-3.1-8B-Instruct/` contains `config.json`, `tokenizer*`, and `*.safetensors`. Path matches the docker `-v $HOME/tt-models:/models` mount, so weights appear at `/models/Llama-3.1-8B-Instruct` inside docker.

### Task 0.4: Docker image pulled and Qwen3 demo reproduces

- [ ] **Step 1:** `docker pull ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-55fd115-aa4ae1e`. Then run the user's known-good Qwen3 demo to baseline-prove the stack. Pass criterion: demo completes; mesh closes cleanly.

### Task 0.5: `pip install -e /sglang/python` does not break the bundled env

- [ ] **Step 1:** Spawn a fresh docker container with the bind-mount described in spec §1.2. Inside:

```bash
source /home/container_app_user/tt-metal/python_env/bin/activate
cd /sglang/python && pip install -e .
python -c "import sglang, ttnn, vllm; print('ok')"
```

Expected: `ok`. If torch / flashinfer / vllm versions conflict, retry with `pip install -e . --no-deps` and reconcile any missing modules **with a pinned torch wheel — record the pin here**. Spec §12 #7 explicitly marks this as a hard go/no-go.

### Task 0.6: ETH fabric, `tt-smi -r`, `SGLANG_PLATFORM` defensiveness

- [ ] **Step 1:** Spec §2.6 already proven by Qwen3 demo (Task 0.4). No-op except for a sanity check: `tt-smi` shows both p150a devices link-up.
- [ ] **Step 2:** Confirm `sudo tt-smi -r 0000:01:00.0 0000:06:00.0` succeeds passwordlessly. Required for crash recovery.
- [ ] **Step 3:** Note: `SGLANG_PLATFORM=tenstorrent` is defensive on this machine (no other plugins active). All runs from Phase A onward use it explicitly anyway.

**Phase 0 gate:** all 6 task groups above checked. If anything fails, **fix or escalate before starting Phase A**.

---

## Phase A — Plugin skeleton

**Goal:** A `TTSRTPlatform` class is registered as a `sglang.srt.platforms` entry point and is the platform SGLang selects when `SGLANG_PLATFORM=tenstorrent` is set.

**Files touched:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/__init__.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/platform.py`
- Modify: `python/pyproject.toml` (add entry point)

**Key API refs:**
- `sglang.srt.platforms.interface:SRTPlatform` — base class. `class SRTPlatform(DeviceMixin)` at `python/sglang/srt/platforms/interface.py:24`.
- `sglang.srt.platforms.device_mixin:PlatformEnum` — enum, includes `OOT`. Verified via grep at spec §6.1.
- `sglang.srt.plugins.PLATFORM_PLUGINS_GROUP = "sglang.srt.platforms"` (`python/sglang/srt/plugins/__init__.py:28`) — the entry-point group name.
- `sglang.srt.platforms.__init__._resolve_platform()` (`python/sglang/srt/platforms/__init__.py:25-80`) — the discovery flow. `activate_tt_platform()` must return `None` if `ttnn` import fails (per §7.1 row 1).

**Live risks:** §10 #5 (tt-metal API drift — pin the docker image).

### Task A.1: Create the package directory and `__init__.py`

- [ ] **Step 1:** Create `python/sglang/srt/hardware_backend/tenstorrent/__init__.py` containing a single comment header — no imports yet (avoid pulling ttnn at plugin-discovery time):

```python
"""SGLang Tenstorrent hardware backend (P1 black-box integration).

See docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md.
Do not import ttnn at module scope — activate_tt_platform() handles that.
"""
```

- [ ] **Step 2:** Commit: `feat(tenstorrent): scaffold hardware_backend/tenstorrent package`.

### Task A.2: Skeleton `platform.py` with `activate_tt_platform`

- [ ] **Step 1:** Create `python/sglang/srt/hardware_backend/tenstorrent/platform.py`. Implement:
  - Top-level function `activate_tt_platform()` that does `try: import ttnn` / `except ImportError: return None` / on success returns the fully-qualified class string `"sglang.srt.hardware_backend.tenstorrent.platform:TTSRTPlatform"` (matching the contract in `platforms/__init__.py:_load_platform_class`).
  - Class `TTSRTPlatform(SRTPlatform)` with **only the class-level attributes from spec §6.1** filled in: `_enum = PlatformEnum.OOT`, `device_name = "tenstorrent"`, `device_type = "cpu"`, `support_cuda_graph -> False`, `support_piecewise_cuda_graph -> False`, `supports_fp8 -> False`, `get_default_attention_backend -> "tenstorrent"`. Leave `apply_server_args_defaults`, `get_*_kv_pool_cls`, and `get_graph_runner_cls` as `raise NotImplementedError(...)` stubs — they'll be implemented in Phase E and §6.1.
  - `MeshDeviceCtx` class: **stub for now** (just `class MeshDeviceCtx: pass`). Real implementation lands in Phase E.

- [ ] **Step 2:** Write a tiny smoke import test inline:

```bash
SGLANG_PLATFORM=tenstorrent python -c "from sglang.srt.hardware_backend.tenstorrent.platform import activate_tt_platform, TTSRTPlatform; print(activate_tt_platform()); print(TTSRTPlatform)"
```

Expected (inside docker): `sglang.srt.hardware_backend.tenstorrent.platform:TTSRTPlatform` and the class repr. Outside docker (no ttnn): `None` and the class repr.

- [ ] **Step 3:** Commit: `feat(tenstorrent): TTSRTPlatform skeleton + activate_tt_platform`.

### Task A.3: Register the entry point in `pyproject.toml`

- [ ] **Step 1:** Edit `python/pyproject.toml`. Add (under a new section if missing):

```toml
[project.entry-points."sglang.srt.platforms"]
tenstorrent = "sglang.srt.hardware_backend.tenstorrent.platform:activate_tt_platform"
```

If the file already has a `[project.entry-points."sglang.srt.platforms"]` table, append the `tenstorrent = ...` line under it; do not create a duplicate table.

- [ ] **Step 2:** Reinstall in editable mode so entry points get re-registered. **Use `--no-deps`** (verified mandatory in Phase 0.5 — tt-metal docker bundles torch 2.7.1+cpu, sglang pins torch==2.11.0, they cannot coexist; `--no-deps` works because tt-metal's bundled torch already satisfies our runtime needs):

```bash
pip install -e /sglang/python --no-deps
```

Note: the `sglang-grpc` Rust crate may fail to build (no `protoc` in image) — this is non-fatal for Phase A-H; only matters if we ever enable gRPC entrypoints.

- [ ] **Step 3:** Verification command (phase gate):

```bash
SGLANG_PLATFORM=tenstorrent python -c \
  "from sglang.srt.platforms import current_platform; print(type(current_platform).__name__)"
```

**Pass:** prints `TTSRTPlatform`. If it prints `SRTPlatform`, the entry point isn't registered — re-check `pip install -e` succeeded and the toml table name is exactly `"sglang.srt.platforms"`.

- [ ] **Step 4:** Commit: `feat(tenstorrent): register sglang.srt.platforms entry-point`.

**Decision gate at end of Phase A:** if `current_platform` fails to resolve to `TTSRTPlatform`, **do not proceed**. Likely root causes: stale install (`pip install -e .` again), wrong entry-point group name (must be `sglang.srt.platforms` not `sglang.platform_plugins`), or `activate_tt_platform()` returning `None` (ttnn import failing inside docker).

---

## Phase B — CUDA-leakage guards

**Goal:** Spec §4.2 lists ~17 SGLang core files that contain unconditional `torch.cuda.*` calls or NCCL paths. Each needs a guard so the server can be imported and a no-op forward path can run with `SGLANG_PLATFORM=tenstorrent`. **Total scope: ~150 LoC across ~17 files**, each individual edit small (5–15 lines).

**Files touched:** all 17 files listed in spec §4.2 (the table starting at "File | Change") PLUS a Phase 0.5-discovered import chain that triggers torch.cuda symbols missing in tt-metal's bundled torch 2.7.1+cpu:
- `python/sglang/srt/layers/quantization/auto_round.py`
- `python/sglang/srt/layers/quantization/fp8_kernel.py`
- `python/sglang/srt/layers/quantization/deep_gemm_wrapper.py`
- `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`

The chain originates at `ModelRunner → model_config → layers.quantization.*`. Phase 0.5 evidence confirms `_cuda_beginAllocateCurrentThreadToPool` (a torch 2.10/2.11 symbol) is referenced at import-time and fails on torch 2.7.1+cpu. Phase B must either gate the entire chain behind `is_cuda()` import-guards, or restructure these modules so the heavy CUDA imports happen lazily / behind feature flags.

The two anchor identifiers are `is_cuda()` (existing helper, grep to confirm import path) and platform method calls (`current_platform.support_*`).

**Live risks:** §10 #7 (SGLang upstream changes break TT integration during rebase — keep diffs minimal); §10 #5 (tt-metal API drift — Phase 0.5 already showed pyproject.toml's `torch==2.11.0` pin is incompatible with the docker's torch 2.7.1+cpu).

### Task B.1: Inventory the exact CUDA call sites

- [ ] **Step 1:** Run:

```bash
grep -rln "torch\.cuda\." python/sglang/srt/managers python/sglang/srt/utils python/sglang/srt/model_executor \
  | sort -u
```

Compare against spec §4.2's file list. **Spec §4.2 explicitly says the implementer should run this grep before estimating.** If the set is larger than §4.2, document the delta in this plan and proceed; if smaller, also note it (the spec may be stale).

- [ ] **Step 2:** For each file, record the line numbers of unguarded `torch.cuda.*` calls. Aim for one Edit per file, not per call.

### Task B.2: Add `is_cuda()` / platform-method guards

- [ ] **Step 1:** For each file in spec §4.2, wrap the offending block in `if is_cuda():` or substitute the equivalent platform-method check. Pattern (matches MLX precedent — grep `grep -rn "is_cuda()" python/sglang/srt/managers/scheduler.py` for examples):

```python
# Before
torch.cuda.empty_cache()

# After
from sglang.srt.utils import is_cuda  # if not already imported
if is_cuda():
    torch.cuda.empty_cache()
```

For `torch.cuda.current_device()` call sites in `schedule_batch.py:342` and similar, substitute with `current_platform.current_device()` or skip entirely on non-CUDA paths — match whichever pattern the file already uses for non-CUDA cases.

- [ ] **Step 2:** For top-level imports that touch CUDA (e.g. `utils/nvtx_pytorch_hooks.py`, `utils/profile_utils.py`), wrap in `try/except ImportError`. Refer to spec §4.2 row-by-row — do not invent new patterns.

- [ ] **Step 3:** After each file edit, commit individually: `chore(platform): guard torch.cuda.* in <file>`. Frequent small commits make bisection trivial if Phase B verification fails.

### Task B.3: Phase B verification — no-op Python import sweep

- [ ] **Step 1:** Derive the import list **from B.1's grep output**, not from a hardcoded list. Spec §4.2's file list was found during plan review to include dead entries (e.g. `scheduler_pp_mixin`, `mm_utils` with no `torch.cuda.*`) AND miss some live ones (e.g. `utils/common.py`, `utils/numa_utils.py`). **Trust B.1's grep, not the spec list.** Generate the import sweep:

```bash
# From B.1 output: one `import X.Y.Z` line per file that needed guarding
SGLANG_PLATFORM=tenstorrent python -c "
$(grep -rln 'torch\.cuda\.' python/sglang/srt/{managers,utils,model_executor} \
    | sed 's|^python/|import |; s|/|.|g; s|\.py$||')
print('imports ok')
"
```

(The `sed` converts `python/sglang/srt/managers/scheduler.py` → `import sglang.srt.managers.scheduler`.)

**Pass:** `imports ok`. **Fail:** import-time `AttributeError` / `RuntimeError` from a CUDA call — fix the offending file and retry. Do not proceed to Phase C until this is clean.

**Decision gate at end of Phase B:** if more than 5 additional files beyond §4.2 needed guarding, escalate — the original scope estimate is materially off and you should flag that the upstream codebase has drifted. Update spec §4.2 in a follow-up commit so future readers see the corrected list.

---

## Phase C — Dispatch wiring

**Goal:** A `use_tt()` helper exists in `tensor_bridge.py` and the four scheduler sites that currently branch on `use_mlx()` also branch on `use_tt()`. When `TTSRTPlatform` is active, the scheduler routes worker construction and overlap-disable logic through the TT branch.

**Files touched:**
- Modify: `python/sglang/srt/utils/tensor_bridge.py` (add `use_tt()`)
- Modify: `python/sglang/srt/managers/scheduler.py` (4 dispatch sites)

**Key API refs:**
- `use_mlx()` at `python/sglang/srt/utils/tensor_bridge.py:38` (verified). Returns `bool(envs.SGLANG_USE_MLX.get()) and _MLX_AVAILABLE`.
- **`use_tt()` must NOT introduce a `SGLANG_USE_TT` env var** — see spec §4.2 row 2 and §12 #8. It must check platform-plugin activation: `isinstance(current_platform, TTSRTPlatform)`.
- Scheduler dispatch sites (re-verify line numbers against current HEAD before editing): `scheduler.py:384–385` (overlap disable), `:653`, `:1312`, `:1511` (worker construction / mid-step dispatch). Spec §4.2 explicitly says re-verify.

**Live risks:** §10 #12 (FutureMap `-1` sentinels if overlap not disabled — `disable_overlap_schedule = True` is set in Phase E but here we also need the boolean to flow through `enable_overlap = not disable_overlap_schedule and not use_tt()`).

### Task C.1: Add `use_tt()` to `tensor_bridge.py`

- [ ] **Step 1:** Edit `python/sglang/srt/utils/tensor_bridge.py`. Just below the existing `use_mlx()` function (line ~38), add:

```python
def use_tt() -> bool:
    """Return True when SGLang's current platform is the Tenstorrent backend.

    Unlike use_mlx() (env-var gated), TT activation is driven by SGLang's
    platform-plugin discovery — see python/sglang/srt/platforms/__init__.py.
    """
    # Lazy import to avoid circular dependency at module-import time.
    try:
        from sglang.srt.platforms import current_platform
        # Avoid importing TTSRTPlatform directly here to keep this module
        # free of ttnn-touching code on non-TT hosts.
        return type(current_platform).__name__ == "TTSRTPlatform"
    except Exception:
        return False
```

Add `"use_tt"` to the module's `__all__` list (currently includes `"use_mlx"` at line 224).

- [ ] **Step 2:** Quick unit-level check:

```bash
SGLANG_PLATFORM=tenstorrent python -c "from sglang.srt.utils.tensor_bridge import use_tt; print(use_tt())"
```

**Pass (inside docker):** `True`. **Pass (outside docker):** `False`.

- [ ] **Step 3:** Commit: `feat(tenstorrent): add use_tt() dispatch helper`.

### Task C.2: Wire `use_tt()` into scheduler dispatch sites

- [ ] **Step 1:** `grep -n "use_mlx" python/sglang/srt/managers/scheduler.py` to re-confirm line numbers (per §4.2 the spec captured 384–385, 653, 1312, 1511 on 2026-05-11). For each site, add an `or use_tt()` (or analogous branch) following the same pattern as MLX. Example for line 384–385:

```python
# Before
self.enable_overlap = not server_args.disable_overlap_schedule and not use_mlx()
self.enable_overlap_mlx = not server_args.disable_overlap_schedule and use_mlx()

# After
self.enable_overlap = (
    not server_args.disable_overlap_schedule
    and not use_mlx()
    and not use_tt()
)
self.enable_overlap_mlx = not server_args.disable_overlap_schedule and use_mlx()
```

For worker-construction sites (lines 653 / 1312 / 1511), add an `elif use_tt():` branch that imports and instantiates `TTTpModelWorker` (which doesn't exist yet — leave the import behind a TYPE_CHECKING gate or a lazy `from ...` inside the branch). **At the end of Phase C, the import is lazy and may raise on call; that's fine — Phase D-G will fill it in.**

- [ ] **Step 2:** Update `scheduler.py`'s top-level import: `from sglang.srt.utils.tensor_bridge import use_mlx` → `from sglang.srt.utils.tensor_bridge import use_mlx, use_tt`.

- [ ] **Step 3:** Phase C verification — no-op branch reachability:

```bash
SGLANG_PLATFORM=tenstorrent python -c "
from sglang.srt.utils.tensor_bridge import use_tt, use_mlx
print('use_tt=', use_tt(), 'use_mlx=', use_mlx())
import sglang.srt.managers.scheduler  # should import without error
"
```

**Pass:** `use_tt= True use_mlx= False` and scheduler module imports clean.

- [ ] **Step 4:** Commit: `feat(tenstorrent): wire use_tt() into scheduler dispatch sites`.

**Decision gate at end of Phase C:** if scheduler.py line numbers differ materially (>20 lines from spec), do a careful re-read of recent commits — the MLX branch shape may have changed. Match whatever the MLX branch does today, do not assume the spec's snippet is authoritative.

---

## Phase D — `TTModelRunner` stub

**Goal:** A `TTModelRunner(ModelRunner)` that overrides `__init__` to force `self.device = "cpu"`, and `initialize()` to skip weight load and build `_DummyKVCache` directly. Scheduler can construct a worker that holds this runner, and `health` checks pass — but no real model loads yet.

**Files touched:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/model_runner.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/model_runner_stub.py`

**Key API refs:**
- Reference: `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py:19` (`_DummyKVCache(KVCache)`), `:60` (`_DummyModel`), `:69` (`MlxModelRunnerStub(ModelRunner)`). **Copy this file as starting point.**
- Reference: `python/sglang/srt/hardware_backend/mlx/model_runner.py:93` (`MlxModelRunner`).
- Spec §3.2 invariant #3: `model_runner.device = "cpu"` after `super().__init__()` (required by `scheduler.init_overlap` via `torch.get_device_module(self.device)`).
- Spec §3.2 invariant #5: `_DummyKVCache` constructed **directly** in `initialize()`; `get_mha_kv_pool_cls()` raises NotImplementedError. Constructor sig stays `(_DummyKVCache(size, dtype, device))` — do NOT match the framework's many-kwarg signature.
- Spec §6.1: `sampler = None`, `attn_backend = None`, `graph_runner = None`.

**Live risks:** §10 #11 (silent-wrong-output if a future code path adds `model_runner.sample()` — leaving `sampler=None` makes the bug fail loudly instead).

### Task D.1: Port `_DummyKVCache` and `_DummyModel`

- [ ] **Step 1:** Open `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py` and read lines 1–80. Copy `_DummyKVCache` (lines ~19–58) and `_DummyModel` (lines ~60–67) verbatim into `python/sglang/srt/hardware_backend/tenstorrent/model_runner_stub.py`. Rename any MLX-specific identifiers; remove any MLX-only imports.

- [ ] **Step 2:** Add module docstring and `from __future__ import annotations` if missing.

- [ ] **Step 3:** Commit: `feat(tenstorrent): port _DummyKVCache and _DummyModel from MLX backend`.

### Task D.2: `TTModelRunner` class

- [ ] **Step 1:** **Open `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py` and read it end-to-end before writing any TT code.** It is the canonical shape; copy it field-by-field. The previous draft of this plan inlined a `def initialize(self, min_per_gpu_memory: float = 0.0)` snippet with the wrong kwarg name and wrong field-assignment location — **do NOT use that snippet**. Real signature from `model_executor/model_runner.py:606`:

```python
def initialize(self, pre_model_load_memory: float):
    ...
```

MLX assigns `self.sampler = None` inside `initialize()` (see `mlx/model_runner_stub.py:116`); `attn_backend` and `graph_runner` are also set to None in the same method (lines 166-168). The dummy KV is wrapped in a `TokenToKVPoolAllocator` (`MlxModelRunnerStub.initialize` lines 157-163). MLX overrides both `load_model()` (skips real PyTorch weight load) and `initialize()` — open the file and read both end-to-end before mirroring.

- [ ] **Step 2:** In `python/sglang/srt/hardware_backend/tenstorrent/model_runner.py`, create:

```python
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.hardware_backend.tenstorrent.model_runner_stub import _DummyKVCache, _DummyModel


class TTModelRunner(ModelRunner):
    """Bookkeeping stub. Real forward happens in TTExecutionBackend.

    Mirrors MlxModelRunnerStub field-by-field — open that file and copy
    its load_model / initialize / __init__ overrides verbatim, swapping
    MLX-specific names for TT. The only TT-specific addition is the
    self.device = "cpu" override in __init__ (see CRITICAL note).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CRITICAL: torch has no "tenstorrent" device module. Scheduler's
        # init_overlap looks up torch.get_device_module(self.device).
        self.device = "cpu"

    # load_model() — copy from MlxModelRunnerStub.load_model verbatim
    # initialize() — copy from MlxModelRunnerStub.initialize verbatim,
    #   keeping the pre_model_load_memory: float kwarg name from the
    #   base class signature
```

Do NOT invent fields the MLX stub doesn't set. Do NOT change kwarg names.

- [ ] **Step 2:** Add a unit test `python/sglang/srt/hardware_backend/tenstorrent/test/test_model_runner_init.py` (CPU-runnable, no TT hardware required) that instantiates `TTModelRunner` with mocked `ServerArgs` and asserts `runner.device == "cpu"`, `runner.attn_backend is None`, `runner.sampler is None`. Use `unittest.mock.MagicMock` for any non-trivial init args.

- [ ] **Step 3:** Run: `pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_model_runner_init.py -v`. Expected: PASS.

- [ ] **Step 4:** Commit: `feat(tenstorrent): TTModelRunner stub with cpu device + dummy KV`.

**Decision gate at end of Phase D:** verification is the unit test passing. If `ModelRunner.__init__` requires hardware probing that fails on a CPU-only host, the test will need to mock more aggressively — check `MlxModelRunnerStub` for the established pattern, do not invent a new one.

---

## Phase E — `MeshDeviceCtx` + warmup + `apply_server_args_defaults`

**Goal:** `TTSRTPlatform.apply_server_args_defaults` populates the server-args constraints from spec §6.1. `MeshDeviceCtx` opens `ttnn.mesh_device([0, 1])`, registers signal/atexit handlers **inside the Scheduler subprocess**, runs a decode-shape JIT warmup, and closes cleanly on SIGTERM. Server can launch with `SGLANG_PLATFORM=tenstorrent` and reach `/health` ready state (still no model — that's Phase F).

**Files touched:**
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/platform.py` (real `apply_server_args_defaults`, real `MeshDeviceCtx`)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/warmup.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/scripts/reset_devices.sh`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_server_args_defaults.py`

**Key API refs:**
- Spec §6.1 full code listing — `apply_server_args_defaults` body must match line-by-line.
- Spec §7.3 — signal/atexit handlers go in the **Scheduler subprocess**, not in the platform-plugin import (registered in `TTTpModelWorker._init_model_runner` or `MeshDeviceCtx.__init__`, whichever runs in the Scheduler subprocess).
- Spec §7.4 — recovery script.
- Spec §12 #2 (open) — does `tt_transformers` JIT once per layer or once total? Affects warmup wall-time, not correctness.

**Live risks:** §10 #4 (mesh leak on crash → reset required), §10 #12 (FutureMap `-1` sentinels — `disable_overlap_schedule = True` here is the hard guard).

### Task E.1: Implement `apply_server_args_defaults`

- [ ] **Step 1:** Replace the stub `apply_server_args_defaults` in `platform.py` with the full body from spec §6.1 — copy it verbatim. Includes the `tp_size` guard that raises if user passes `--tp 2`.

- [ ] **Step 2:** Add unit test `test_server_args_defaults.py`:

```python
import pytest
from unittest.mock import MagicMock
from sglang.srt.hardware_backend.tenstorrent.platform import TTSRTPlatform


def test_defaults_set_p1_constraints():
    plat = TTSRTPlatform()
    sa = MagicMock(tp_size=None)
    plat.apply_server_args_defaults(sa)
    assert sa.max_running_requests == 1
    assert sa.chunked_prefill_size == -1
    assert sa.disable_radix_cache is True
    assert sa.disable_overlap_schedule is True   # §10 risk #12 guard
    assert sa.sampling_backend == "pytorch"
    assert sa.pre_warm_nccl is False
    assert sa.cpu_offload_gb == 0
    assert sa.enable_torch_compile is False
    assert sa.device == "tenstorrent"
    assert sa.tp_size == 1
    assert sa.enable_dp_attention is False


def test_rejects_explicit_tp_2():
    plat = TTSRTPlatform()
    sa = MagicMock(tp_size=2)
    with pytest.raises(ValueError, match="TT backend requires --tp 1"):
        plat.apply_server_args_defaults(sa)
```

- [ ] **Step 3:** Run: `pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_server_args_defaults.py -v`. **Pass:** both tests green.

- [ ] **Step 4:** Commit: `feat(tenstorrent): apply_server_args_defaults per spec §6.1`.

### Task E.2: Real `MeshDeviceCtx` with handlers

- [ ] **Step 1:** In `platform.py`, replace the `MeshDeviceCtx` stub. Constructor: opens `mesh = ttnn.open_mesh_device([0, 1])` with fabric configuration matching `tt_transformers`' demo (mirror whatever the Qwen3 demo uses — capture exact args in Phase 0). Holds `self.mesh`. `close()` calls `ttnn.close_mesh_device(self.mesh)` with a 5 s watchdog (use `threading.Timer` or similar — spec §7.3 row "close_mesh_device hangs > 5 s").
- [ ] **Step 2:** Inside `__init__`, register handlers per spec §7.3:

```python
import atexit, signal
atexit.register(self._safe_close)
signal.signal(signal.SIGTERM, self._on_signal)
signal.signal(signal.SIGINT, self._on_signal)
```

Document with a comment that this **only works when constructed in the Scheduler subprocess**, not at plugin-import time in the parent.

- [ ] **Step 3:** Configure the ttnn **program cache** here too (spec §3.2, §6.1 wording). Match the demo's config.

- [ ] **Step 4:** Add observability log lines per spec §6.2:
  - `mesh_open` INFO with `{bdfs, eth_live_count, mesh_shape}`
  - `mesh_close` INFO with `{shutdown_elapsed_s}`
  - `mesh_close_timeout` WARN if the 5 s watchdog fires

  Use logger `sglang.srt.hardware_backend.tenstorrent`.

- [ ] **Step 5:** Commit: `feat(tenstorrent): MeshDeviceCtx with signal/atexit handlers`.

### Task E.3: Warmup driver

- [ ] **Step 1:** Create `python/sglang/srt/hardware_backend/tenstorrent/warmup.py`:

```python
"""Decode-shape JIT warmup driver. Runs a dummy [1, 1] decode through the
wrapper so tt-metal's program cache compiles before the first real request.

Logs warmup_start / warmup_done per spec §6.2.
"""

import logging
import time

logger = logging.getLogger("sglang.srt.hardware_backend.tenstorrent")


def warm_decode_shape(wrapper, dummy_token: int = 0, prompt_len: int = 32):
    logger.info("warmup_start", extra={"shape": (1, 1), "prompt_len": prompt_len})
    t0 = time.time()
    # 1) prime extend on a short dummy prompt; 2) one decode step.
    wrapper.new_request("__warmup__", prompt_tokens=[dummy_token] * prompt_len)
    wrapper.extend("__warmup__")
    wrapper.decode_step("__warmup__", dummy_token)
    wrapper.free("__warmup__")
    logger.info("warmup_done", extra={"elapsed_s": time.time() - t0})
```

The `wrapper` parameter takes any `TTExecutionBackend` instance — the API surface is defined in Phase F. Renamed from `TTLlamaWrapper` after the spec amendment that introduces the ABC + registry (see Phase F header). The parameter name `wrapper` in the warmup function stays for readability; the actual type is `TTExecutionBackend`. **Order: write warmup.py before Phase F so Phase F can wire its call site, but the actual call only fires once a concrete backend exists.**

- [ ] **Step 2:** Commit: `feat(tenstorrent): decode-shape JIT warmup driver`.

### Task E.4: Recovery script

- [ ] **Step 1:** Create `python/sglang/srt/hardware_backend/tenstorrent/scripts/reset_devices.sh` with the body from spec §7.4:

```bash
#!/bin/bash
set -e
echo "Resetting Tenstorrent devices 01:00.0 and 06:00.0..."
sudo tt-smi -r 0000:01:00.0 0000:06:00.0
echo "Done. Mesh state cleared."
```

`chmod +x` it. Commit: `chore(tenstorrent): add reset_devices.sh recovery script`.

### Task E.5: Phase E verification — server launch with stub worker

- [ ] **Step 1:** **This step depends on Phase F's wrapper existing — defer the live-launch verification to the end of Phase F.** But before then, do a partial check inside docker:

```bash
SGLANG_PLATFORM=tenstorrent python -c "
from sglang.srt.hardware_backend.tenstorrent.platform import MeshDeviceCtx
ctx = MeshDeviceCtx()
print('mesh opened:', ctx.mesh)
ctx.close()
print('mesh closed cleanly')
"
```

In a second terminal, run `tt-smi` while `MeshDeviceCtx` is alive — the two devices should show in-use. After `close()`, they should free.

- [ ] **Step 2:** **Decision gate**: if `open_mesh_device` raises, run `sudo bash python/sglang/srt/hardware_backend/tenstorrent/scripts/reset_devices.sh` and retry once. Two failures → suspect ETH fabric / KMD; check `dmesg` and `tt-smi`.

---

## Phase F — `TTExecutionBackend` interface + `tt_transformers` implementation

**Goal:** A 5-method ABC (`TTExecutionBackend`) is the only seam the worker depends on; `TTTransformersExecutionBackend` (the P1 implementation) wraps `Generator`/`LlamaForCausalLM` built by `create_tt_model`; a `TTXLAExecutionBackend` placeholder is registered but raises `NotImplementedError` (real impl is post-P1 per spec §11). Unit tests on mocked `tt_transformers` pass; the registry test exercises the auto-resolve; an integration smoke produces non-empty logits via the registry factory, not via direct class import.

Spec amendment rationale (added in this revision): tt-torch was deprecated in favor of tt-xla in <12 months, signalling that Tenstorrent's SW stack still churns rapidly. Making the execution path swappable now — with the same registry pattern SGLang already uses for attention/moe/sampling backends — keeps P1 thin while pre-paying the future refactor cost. See spec §3.1 row "Execution-backend surface", §3.2 invariant #7, §11 "P2-coverage".

**Files touched:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py` (registry + factory)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/execution/base.py` (`TTExecutionBackend` ABC)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_backend.py` (`TTTransformersExecutionBackend`, P1's real impl)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/execution/tt_xla_backend.py` (placeholder, raises NotImplementedError)
- Modify: `python/sglang/srt/environ.py` (add `SGLANG_TT_EXECUTION_BACKEND = EnvStr("")`)
- Delete: `python/sglang/srt/hardware_backend/tenstorrent/llama_adapter.py` (the F.1 prototype — its content is split across `execution/base.py` + `execution/tt_transformers_backend.py`)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_wrapper_lifecycle.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_execution_backend_registry.py`

**Key API refs:**
- Captured `create_tt_model` signature + `Generator`/`LlamaForCausalLM` method names from Task 0.2 — paste into the `tt_transformers_backend.py` module docstring.
- Spec §3.2 — the architecture diagram and invariant #7 (worker depends only on the ABC; never imports tt_transformers directly).
- Spec §5.1 — the request path. Backend methods return host torch tensors (post `Generator.read_decode_output` + `process_decode_output_host`).
- Spec §3.2 invariant #6 — only `EXTEND`, `DECODE`, `IDLE` reach the backend.
- SGLang precedent: `python/sglang/srt/layers/attention/attention_registry.py` (`ATTENTION_BACKENDS` dict + `@register_attention_backend` decorator) — copy the *shape* of the registry, not its specific methods.

**Live risks:** §10 #1 (`Generator` may not expose the methods we need — see decision gate at the end), §10 #8 (silent KV bugs in tt_transformers — greedy correctness test in Phase H catches these).

### Task F.1: `TTExecutionBackend` ABC + registry + tt-xla placeholder

This task replaces the deprecated `llama_adapter.py` (originally committed by an earlier F.1 attempt) with the new `execution/` package layout. The earlier prototype's `__init__` logic is moved verbatim into `tt_transformers_backend.py` — only the file location and class name change.

- [ ] **Step 1:** Create `execution/__init__.py`:

```python
"""Registry + factory for TT execution backends.

Mirror of SGLang's attention_registry.py pattern. P1 ships exactly one real
backend (tt_transformers); tt_xla is a registered placeholder so the
post-P1 add-a-backend path is a swap rather than a Phase-F rewrite.

IMPORTANT: the registry dict MUST be declared before the backend modules
are imported — their @register_tt_execution_backend(...) decorators fire
at import time and would NameError otherwise.
"""

from __future__ import annotations

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tenstorrent.execution.base import (
    TTExecutionBackend,
)

# Step 1: declare the registry FIRST.
TT_EXECUTION_BACKENDS: dict[str, type[TTExecutionBackend]] = {}


def register_tt_execution_backend(name: str):
    def _wrap(cls):
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
    """Resolve "auto" / "" / None to the P1 default."""
    name = (requested or envs.SGLANG_TT_EXECUTION_BACKEND.get() or "auto").lower()
    if name == "auto":
        return "tt_transformers"
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
```

Note: the `noqa: E402` is intentional — the import-after-statement is the *correct* shape here, not a code-style issue to suppress lightly. The earlier subagent review flagged this as a CRITICAL trap; the comment on the registry declaration + the noqa make the requirement obvious to future readers.

- [ ] **Step 2:** Create `execution/base.py` — the ABC:

```python
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
```

- [ ] **Step 3:** Create `execution/tt_xla_backend.py` — the placeholder:

```python
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
```

- [ ] **Step 4:** Add to `environ.py`:

```python
SGLANG_TT_EXECUTION_BACKEND = EnvStr("")
```

Add it near `SGLANG_USE_MLX` for locality.

- [ ] **Step 5:** Create `execution/tt_transformers_backend.py` with `class TTTransformersExecutionBackend(TTExecutionBackend):` decorated `@register_tt_execution_backend("tt_transformers")`. The class body is the existing `llama_adapter.py:TTLlamaWrapper` implementation moved verbatim with three deltas:

  1. Rename the class `TTLlamaWrapper` → `TTTransformersExecutionBackend`.
  2. Add `from .base import TTExecutionBackend` and make the class inherit from it (so the registry's type annotation matches).
  3. **HOIST the lazy imports from inside `__init__` up to module scope behind `try/except ImportError`** — this is non-negotiable for F.3 testability:

     ```python
     # In llama_adapter.py these lived inside __init__ (deferred for
     # non-TT-host importability). For F.3 mocking we need them on the
     # module namespace, so hoist them but keep the non-TT-host
     # importability via try/except + None sentinels.
     try:
         import ttnn
         from models.tt_transformers.tt.common import create_tt_model
         from models.tt_transformers.tt.generator import Generator
         from models.tt_transformers.tt.model_config import DecodersPrecision
     except ImportError:
         ttnn = None
         create_tt_model = None
         Generator = None
         DecodersPrecision = None
     ```

     Then in `__init__`, add a guard block **before** any use of the hoisted names. The existing `llama_adapter.py:97` line `tt_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[dtype]` consumes `ttnn` immediately — if you keep that pattern, the dtype-map AttributeErrors on `None.bfloat16` BEFORE the assert runs. Two acceptable shapes (pick one):

     ```python
     # Option A: hard-fail with a clear message
     if create_tt_model is None or ttnn is None:
         raise RuntimeError(
             "tt_transformers / ttnn not importable; install tt-metal or "
             "activate its venv (source /home/container_app_user/tt-metal/"
             "python_env/bin/activate)"
         )
     tt_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[dtype]
     ```

     ```python
     # Option B: defer the dtype lookup behind the assert
     assert ttnn is not None and create_tt_model is not None, (
         "tt_transformers / ttnn not importable; install tt-metal..."
     )
     tt_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[dtype]
     ```

     Option A is preferred for production code (RuntimeError is more visible than AssertionError in `-O` mode); Option B is fine for the F.1 prototype.

  Reason: F.3 uses `@patch("...tt_transformers_backend.create_tt_model")`, which fails if the name isn't a module attribute. The in-`__init__` form in the prototype was for non-TT-host importability — module-scope `try/except` preserves that AND exposes the names for patching. The dtype-map ordering note above is the round-4 review correction.

  The 5 method stubs (currently raising `"lands in Phase F.2"` NotImplementedError) carry over unchanged.

- [ ] **Step 6:** `git rm python/sglang/srt/hardware_backend/tenstorrent/llama_adapter.py` (use `git rm`, NOT plain `rm` — we want git to record the file's removal so future blame on `tt_transformers_backend.py` traces back to commit `152b67c8a`). Run `grep -rn "llama_adapter\|TTLlamaWrapper" python/sglang/` and update any leftover references (Phase A/E artifacts may reference the old name in comments).

- [ ] **Step 7:** Commit: `refactor(tenstorrent): introduce TTExecutionBackend ABC + registry; tt_transformers as first impl`.

### Task F.2: `new_request` / `extend` / `decode_step` / `free` / `reset_all` (on `TTTransformersExecutionBackend`)

All edits in this task target `execution/tt_transformers_backend.py` — the renamed home of the F.1 prototype. The 5 currently-NotImplementedError stubs gain real bodies. Worker / model_runner code never sees this class directly; they go through the registry factory.

- [ ] **Step 1:** Implement the five methods. Surface shape (defined by `TTExecutionBackend`):

```python
def new_request(self, req_id: str, prompt_tokens) -> None:
    """Allocate per-request KV / position state. prompt_tokens: list[int] or torch.LongTensor."""
    # spec §6.2: log req.new {req_id, prompt_len}

def extend(self, req_id: str):
    """Run prefill. Returns torch.Tensor of last-token logits, shape [vocab]."""
    # spec §6.2: log req.extend, emit tt_extend_latency_ms histogram point

def decode_step(self, req_id: str, last_token: int):
    """Advance by one token. Returns torch.Tensor of next-token logits, shape [vocab]."""
    # spec §6.2: log req.decode, emit tt_decode_latency_ms

def free(self, req_id: str) -> None:
    """Release per-request KV state."""
    # spec §6.2: log req.free

def reset_all(self) -> None:
    """Release all per-request state. Called on SIGTERM."""
```

The exact ttnn calls inside each method depend on `Generator`/`LlamaForCausalLM`'s public surface (captured in Phase 0.2). Likely:
- `extend` → `Generator.prefill_forward_single_user_text(tokens, page_table=None, user_id=0, last_token_idx=len-1, kv_cache=None)` → `process_decode_output_host(..., is_tokens=False)`
- `decode_step` → `Generator.decode_forward_text(tokens, start_pos=..., page_table=None, kv_cache=None, enable_trace=True, read_from_device=True, sampling_params=None)`

**If `Generator` uses different method names** than the Phase 0.2 capture, keep the ABC method names stable (the spec-mandated public API) and adapt inside.

- [ ] **Step 2:** Add metrics emission (spec §6.2): `tt_decode_latency_ms`, `tt_extend_latency_ms`, `tt_d2h_latency_ms`, `tt_active_requests`. Use SGLang's existing metrics endpoint registry — do not add new endpoints (spec §6.2 last paragraph).

- [ ] **Step 3:** Commit: `feat(tenstorrent): TTTransformersExecutionBackend lifecycle methods + observability`.

### Task F.3: Unit tests — `test_wrapper_lifecycle.py` + `test_execution_backend_registry.py`

Two test files: one tests the lifecycle methods of `TTTransformersExecutionBackend` directly (mocked tt_transformers), the other tests the registry + factory + env-var resolution path.

- [ ] **Step 1:** Write `test_wrapper_lifecycle.py`. **PREREQUISITE:** F.1 Step 5 must have hoisted `ttnn`, `create_tt_model`, `Generator`, `DecodersPrecision` to module scope in `tt_transformers_backend.py` (behind `try/except ImportError`). Without that, the `@patch(f"{_MODULE}.create_tt_model")` decorator AttributeErrors because the name isn't on the module namespace. Verify with `grep -n "^import\|^from\|^try:" python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_backend.py` before running tests.

```python
import pytest
from unittest.mock import MagicMock, patch
from sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_backend import (
    TTTransformersExecutionBackend,
)

_MODULE = "sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_backend"


@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
def test_lifecycle_roundtrip(_mock_gen, _mock_create):
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        be.new_request("r1", [1, 2, 3])
        assert "r1" in be._req_state
        _ = be.extend("r1")
        for _ in range(3):
            _ = be.decode_step("r1", last_token=42)
        be.free("r1")
        assert "r1" not in be._req_state


@patch(f"{_MODULE}.create_tt_model")
@patch(f"{_MODULE}.Generator")
def test_reset_all_clears_state(_mock_gen, _mock_create):
    with patch("os.path.isdir", return_value=True):
        be = TTTransformersExecutionBackend(
            model_path="/tmp", mesh_device=MagicMock(), max_seq_len=256
        )
        be.new_request("r1", [1])
        be.new_request("r2", [2])
        be.reset_all()
        assert be._req_state == {}


def test_missing_model_path_raises():
    # No need to mock create_tt_model — model_path validation happens first.
    with pytest.raises(FileNotFoundError, match="mount"):
        TTTransformersExecutionBackend(
            model_path="/does/not/exist", mesh_device=MagicMock(), max_seq_len=256
        )
```

- [ ] **Step 2:** Write `test_execution_backend_registry.py`:

```python
import pytest
from unittest.mock import patch

from sglang.srt.hardware_backend.tenstorrent.execution import (
    TT_EXECUTION_BACKENDS,
    get_tt_execution_backend,
    resolve_execution_backend_name,
)
from sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_backend import (
    TTTransformersExecutionBackend,
)
from sglang.srt.hardware_backend.tenstorrent.execution.tt_xla_backend import (
    TTXLAExecutionBackend,
)


def test_registry_has_both_entries():
    assert TT_EXECUTION_BACKENDS["tt_transformers"] is TTTransformersExecutionBackend
    assert TT_EXECUTION_BACKENDS["tt_xla"] is TTXLAExecutionBackend


def test_auto_resolves_to_tt_transformers_in_p1():
    assert resolve_execution_backend_name("auto") == "tt_transformers"
    assert resolve_execution_backend_name("") == "tt_transformers"
    assert resolve_execution_backend_name(None) == "tt_transformers"


def test_explicit_tt_xla_resolves_to_tt_xla():
    assert resolve_execution_backend_name("tt_xla") == "tt_xla"


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown TT execution backend"):
        get_tt_execution_backend("frobnicate")


def test_tt_xla_construction_raises_with_pointer():
    cls = get_tt_execution_backend("tt_xla")
    with pytest.raises(NotImplementedError, match="P2-coverage"):
        cls(model_path="/tmp", mesh_device=None, max_seq_len=128)


def test_env_var_selects_backend(monkeypatch):
    """The worker's resolve path is `resolve_execution_backend_name()` with
    no args — the env var is the only input. Test that explicitly so a
    regression in `envs.SGLANG_TT_EXECUTION_BACKEND.get()` doesn't slip past.

    EnvField.get() reads os.environ on every call (no lru_cache) — verified
    against environ.py:54. So monkeypatch.setenv is sufficient; no cache
    busting needed.
    """
    monkeypatch.setenv("SGLANG_TT_EXECUTION_BACKEND", "tt_xla")
    assert resolve_execution_backend_name() == "tt_xla"


def test_env_var_unset_defaults_to_tt_transformers(monkeypatch):
    monkeypatch.delenv("SGLANG_TT_EXECUTION_BACKEND", raising=False)
    assert resolve_execution_backend_name() == "tt_transformers"
```

These two tests pass no args to `resolve_execution_backend_name()` —
matching the worker's real call shape — and use pytest's `monkeypatch` to
drive the env var. If a future refactor changes `EnvField.get` to be cached,
these tests will start to flake; fix the caching, don't paper over it here.

- [ ] **Step 3:** Run: `pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_wrapper_lifecycle.py python/sglang/srt/hardware_backend/tenstorrent/test/test_execution_backend_registry.py -v`. **Pass:** all green.

- [ ] **Step 4:** Commit: `test(tenstorrent): execution backend ABC + registry + tt_transformers lifecycle`.

### Task F.4: Phase F verification — integration smoke (real hardware)

The smoke goes through the registry factory, not a direct class import, so the integration path matches what the worker (Phase G) will use.

- [ ] **Step 1:** Inside docker:

```bash
SGLANG_PLATFORM=tenstorrent python -c "
from sglang.srt.hardware_backend.tenstorrent.platform import MeshDeviceCtx
from sglang.srt.hardware_backend.tenstorrent.execution import get_tt_execution_backend
ctx = MeshDeviceCtx()
BackendCls = get_tt_execution_backend()  # auto → tt_transformers in P1
be = BackendCls('/models/Llama-3.1-8B-Instruct', ctx.mesh, max_seq_len=4096)
be.new_request('r1', [1, 2, 3, 4, 5])
logits = be.extend('r1')  # backend returns host torch tensor
print('logits shape:', logits.shape, 'dtype:', logits.dtype)
assert logits.numel() > 0
be.free('r1')
ctx.close()
print('integration smoke ok')
"
```

**Pass:** prints non-zero logits shape, `integration smoke ok`. Wall-time including weight load: expect 30–60 s (PCIe Gen3 x1 bottleneck on Device 1 — spec §5.2).

- [ ] **Step 2:** Repeat with `SGLANG_TT_EXECUTION_BACKEND=tt_xla` and confirm it raises `NotImplementedError` containing "P2-coverage" — this is a 5-second test that exercises the env-var → registry → factory path end-to-end, ensuring the post-P1 swap really is a one-knob change:

```bash
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_xla python -c "
from sglang.srt.hardware_backend.tenstorrent.execution import get_tt_execution_backend
BackendCls = get_tt_execution_backend()
try:
    BackendCls(model_path='/tmp', mesh_device=None, max_seq_len=128)
except NotImplementedError as e:
    assert 'P2-coverage' in str(e), e
    print('tt_xla swap path ok')
"
```

**Pass:** prints `tt_xla swap path ok`. No mesh device required — the NotImplementedError fires before any hardware access.

**Decision gate at end of Phase F:** if `Generator`/`LlamaForCausalLM` does not expose the methods we need to drive prefill/decode externally — e.g. the only public surface is a `generate()` that owns the whole loop — **STOP and escalate**. The wrapper layer cannot be salvaged without either upstreaming hooks into `tt_transformers` or rewriting the loop. Spec §10 #1 (high severity) covers this; the answer is renegotiate the spec, not silently swap models. The ABC + registry remain valid — only the `tt_transformers` impl would need rework.

---

## Phase G — `TTTpModelWorker`

**Goal:** A `TTTpModelWorker(TpModelWorker)` that overrides `_init_model_runner`, `get_pad_input_ids_func`, and `forward_batch_generation` per spec §5.1. Calls into the wrapper, samples greedy in the worker, returns a populated `GenerationBatchResult`. IDLE forward-mode short-circuits with empty result (mirrors MLX `tp_worker.py:136-140`).

**Files touched:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_forward_mode_guard.py`

**Key API refs:**
- Reference: `python/sglang/srt/hardware_backend/mlx/tp_worker.py:34` (`MlxTpModelWorker(TpModelWorker)`), `:94` (`forward_batch_generation`), `:126` (`_forward_batch_generation_mlx`), `:136-140` (IDLE return).
- Spec §5.1 — pseudocode for `_forward_batch_generation_tt`.
- Spec §3.2 invariant #2 — greedy in worker, `next_token_logits=None` in result, P1 is greedy-only.
- Spec §3.2 invariant #6 — IDLE returns empty result; MIXED / SPLIT_PREFILL / DLLM_EXTEND raise.
- Spec §5.1 (mid-body comment) — `mwb.reqs[0].rid`, NOT `mwb.req_ids[0]`. Verified by Appendix B subagent review.

**Live risks:** §10 #11 (silent-wrong-output A — never call `model_runner.sample()`), §10 #12 (silent-wrong-output B — IDLE return path must match MLX exactly).

### Task G.1: Skeleton — class + `_init_model_runner`

- [ ] **Step 1:** Open `python/sglang/srt/hardware_backend/mlx/tp_worker.py` — read it end-to-end first; the file is the source of truth for shape.

- [ ] **Step 2:** Create `python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py`. Define `class TTTpModelWorker(TpModelWorker):`. Override `_init_model_runner` to:
  - Instantiate `MeshDeviceCtx` (this is the Scheduler-subprocess location where mesh handlers must register — see spec §7.3).
  - **Resolve the execution backend via the registry**: `BackendCls = get_tt_execution_backend()` (no arg → reads `SGLANG_TT_EXECUTION_BACKEND` → resolves "auto"/empty to "tt_transformers"). Then `self.execution_backend = BackendCls(model_path, ctx.mesh, max_seq_len=max_seq_len)`. **Do NOT import `TTTransformersExecutionBackend` directly** — spec §3.2 invariant #7 forbids it.
  - Run `warmup.warm_decode_shape(self.execution_backend)` from Phase E.3.
  - Instantiate `TTModelRunner(...)` and **assign to `self._model_runner`** — note the underscore: `TpModelWorker._init_model_runner` writes to `self._model_runner` (managers/tp_worker.py:347) and `self.model_runner` is a read-only property over the same field. Re-read `MlxTpModelWorker._init_model_runner` for the canonical shape and exact ctor kwargs.

- [ ] **Step 3:** Override `get_pad_input_ids_func` to return a no-op padder (MLX does the same — copy its return value). P1 doesn't batch, so padding is trivial.

- [ ] **Step 4:** Commit: `feat(tenstorrent): TTTpModelWorker skeleton + _init_model_runner`.

### Task G.2: `forward_batch_generation` + `_forward_batch_generation_tt`

- [ ] **Step 1:** Implement the two methods exactly per spec §5.1 pseudocode. Reproduced here so this plan is self-contained:

```python
# NB: worker code MUST NOT import ttnn (spec §3.2 invariant #7) —
# host torch tensors are produced inside the execution backend.
import torch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput


class TTTpModelWorker(TpModelWorker):
    # ... __init__ and _init_model_runner from Task G.1 ...

    def forward_batch_generation(
        self,
        model_worker_batch,
        forward_batch=None,
        pp_proxy_tensors=None,
        is_verify=False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        # Polarity matches MLX exactly (mlx/tp_worker.py:103-114):
        # "not None ⇒ take our path; else fall back to super()".
        if model_worker_batch is not None:
            return self._forward_batch_generation_tt(model_worker_batch)
        return super().forward_batch_generation(
            model_worker_batch, forward_batch, pp_proxy_tensors,
            is_verify, skip_attn_backend_init,
        )

    def _forward_batch_generation_tt(self, mwb) -> GenerationBatchResult:
        # --- IDLE: mirror MLX tp_worker.py:136-140 ---
        if mwb.forward_mode.is_idle():
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        # --- P1 guard: only EXTEND/DECODE besides IDLE ---
        # CRITICAL: ForwardMode.is_extend() is multi-mode — returns True
        # for EXTEND, MIXED, DRAFT_EXTEND, TARGET_VERIFY, SPLIT_PREFILL,
        # AND DLLM_EXTEND (forward_batch_info.py:111-119). Use explicit
        # equality so MIXED / SPLIT_PREFILL / DLLM_EXTEND raise per
        # invariant #6 instead of silently entering the EXTEND branch.
        if mwb.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise NotImplementedError(
                f"P1 supports EXTEND/DECODE/IDLE only, got {mwb.forward_mode}. "
                f"Chunked prefill / mixed / DLLM must stay disabled via "
                f"apply_server_args_defaults."
            )

        # max_running_requests=1 ⇒ at most one req
        assert mwb.reqs is not None and len(mwb.reqs) == 1
        req_id = mwb.reqs[0].rid

        if mwb.forward_mode == ForwardMode.EXTEND:
            prompt_tokens = mwb.input_ids  # torch.Tensor[L], CPU
            self.execution_backend.new_request(req_id, prompt_tokens)
            logits = self.execution_backend.extend(req_id)  # host torch.Tensor
        else:  # DECODE (other modes already raised above)
            last_tok = int(mwb.input_ids[-1].item())
            logits = self.execution_backend.decode_step(req_id, last_tok)

        # Greedy sampling in the worker; we never call model_runner.sample
        # because model_runner.sampler is None (§3.2 invariant #2). The
        # execution backend already returns host torch tensors (it owns
        # any ttnn.to_torch + dtype massage), so no ttnn import here.
        next_token_ids = torch.argmax(logits.float(), dim=-1, keepdim=True).long()

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
```

Match MLX's exact `GenerationBatchResult` field population — re-read `MlxTpModelWorker._forward_batch_generation_mlx` and compare field-by-field. Any extra optional fields MLX sets, set them too.

- [ ] **Step 2:** Add `free()` hook: when scheduler marks a req finished, the worker must call `self.execution_backend.free(req_id)`. Wire this into whatever cleanup hook `TpModelWorker` exposes (search `MlxTpModelWorker` for the pattern — it has the same problem).

- [ ] **Step 3:** Commit: `feat(tenstorrent): forward_batch_generation with IDLE/EXTEND/DECODE dispatch + greedy sampling`.

### Task G.3: Unit test — `test_forward_mode_guard.py`

- [ ] **Step 1:** Write:

```python
import pytest
from unittest.mock import MagicMock
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.hardware_backend.tenstorrent.tp_worker import TTTpModelWorker


def _mock_mwb(mode: ForwardMode):
    mwb = MagicMock()
    mwb.forward_mode = mode
    mwb.reqs = [MagicMock(rid="r1")]
    return mwb


@pytest.mark.parametrize("bad_mode", [ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL, ForwardMode.DLLM_EXTEND])
def test_unsupported_modes_raise(bad_mode):
    worker = TTTpModelWorker.__new__(TTTpModelWorker)  # skip __init__
    worker.execution_backend = MagicMock()
    with pytest.raises(NotImplementedError, match="P1 supports EXTEND/DECODE/IDLE only"):
        worker._forward_batch_generation_tt(_mock_mwb(bad_mode))


def test_idle_returns_empty_result():
    worker = TTTpModelWorker.__new__(TTTpModelWorker)
    worker.execution_backend = MagicMock()
    res = worker._forward_batch_generation_tt(_mock_mwb(ForwardMode.IDLE))
    assert res.logits_output.next_token_logits is None
    assert res.can_run_cuda_graph is False
    # MLX-parity check: no next_token_ids in IDLE result (MLX leaves it as the dataclass default)
```

Adapt `ForwardMode` import to whatever the actual path is (verify with `grep -rn "class ForwardMode" python/sglang/srt/`).

- [ ] **Step 2:** Run: `pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_forward_mode_guard.py -v`. **Pass:** all green.

- [ ] **Step 3:** Commit: `test(tenstorrent): forward_mode dispatch guard tests`.

### Task G.4: Add `test_platform_activate.py` (CPU, mocked) — final unit test

- [ ] **Step 1:** Per spec §8.6 row 1:

```python
from unittest.mock import patch
import sys


def test_activate_returns_none_when_ttnn_missing():
    # Simulate ttnn import failure
    with patch.dict(sys.modules, {"ttnn": None}):
        from sglang.srt.hardware_backend.tenstorrent.platform import activate_tt_platform
        assert activate_tt_platform() is None
```

If `patch.dict` doesn't reliably break the import, use `unittest.mock.patch("builtins.__import__", side_effect=ImportError)` scoped to the call. Match whatever pattern MLX's analogous test uses.

- [ ] **Step 2:** Run: `pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_platform_activate.py -v`. **Pass:** green.

- [ ] **Step 3:** Commit: `test(tenstorrent): activate_tt_platform returns None without ttnn`.

### Task G.5: Phase G verification — end-to-end forward via worker (real hardware)

- [ ] **Step 1:** Inside docker, launch the SGLang server fully (this is the §8.1 smoke test, partial):

```bash
SGLANG_PLATFORM=tenstorrent python -m sglang.launch_server \
  --model-path /models/Llama-3.1-8B-Instruct \
  --port 30000 \
  --max-running-requests 1 \
  --chunked-prefill-size -1 --disable-radix-cache \
  --sampling-backend pytorch \
  > /tmp/sglang-tt.log 2>&1 &
```

Wait up to 5 minutes for `curl localhost:30000/health` to return 200 (Device 1's PCIe Gen3 x1 means ~15 s for weight load + 5–30 s mesh + 10–60 s first-prefill JIT).

- [ ] **Step 2:** **Stop here for greedy correctness — Phase H is the gate.** This step only checks that the server reaches healthy state. If `/health` never returns 200, tail `/tmp/sglang-tt.log` and diagnose.

**Decision gate at end of Phase G:** server should reach `/health` 200. If a request issued at this point returns malformed tokens (e.g. `-1` sentinels appearing — spec §10 #12), `disable_overlap_schedule` is not flowing through; re-check Phase E.1 and Phase C.2 wiring.

---

## Phase H — Smoke test + greedy correctness (**P1 ACCEPTANCE GATE**)

**Goal:** Spec §8.1 smoke passes (HTTP 200, response contains "Paris", ≤ 5 min). Spec §8.2 greedy correctness passes (mean ≥ 90% top-1 agreement with HF reference across 10 prompts, no single prompt < 70%).

**Files touched:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness.py`
- Update: spec addendum with first observed performance numbers.

**Key API refs:**
- Spec §8.1 (smoke), §8.2 (correctness), §9 (acceptance criteria 1 + 2).
- `@requires_tt` marker — define in `conftest.py` as `pytest.mark.skipif(...)` checking `os.environ.get("SGLANG_PLATFORM") == "tenstorrent"`.

**Live risks:** §10 #1 (model support), §10 #8 (KV bugs masked by P1 black-box approach — this test is the catch), §10 #11/#12 (silent-wrong-output A/B — surface here).

### Task H.1: Smoke test from spec §8.1

- [ ] **Step 1:** Run the §8.1 smoke shell script verbatim (it's reproduced in the spec). Capture output.

- [ ] **Step 2:** **Pass criteria** (spec §8.1):
  - HTTP 200
  - Response body contains "Paris"
  - End-to-end (server start → response) ≤ 5 minutes
  - Server process still alive after the request returns

  **Fail:** read `/tmp/sglang-tt.log` for the first ERROR line. Most common cause at this point: §10 #12 (FutureMap `-1` sentinels in output). Re-check Phase E.1 verifies `disable_overlap_schedule = True`.

### Task H.2: `test_greedy_correctness.py`

- [ ] **Step 1:** Define 10 fixed prompts in the test file (spec §8.2 enumerates the kinds: short English, long English, code, multilingual, chat-formatted). Commit them as test data.

- [ ] **Step 2:** Test structure:

```python
import os
import pytest
import requests

# Reference generations come from HF transformers run on CPU or remote GPU,
# saved to a JSON fixture committed alongside this test.
REFERENCE_FIXTURE = "tests/fixtures/llama31_greedy_50tok.json"


@pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware; @requires_tt",
)
def test_greedy_correctness_against_hf():
    prompts, references = load_fixture(REFERENCE_FIXTURE)
    assert len(prompts) == 10
    per_prompt_agreement = []
    for p, ref in zip(prompts, references):
        resp = requests.post(
            "http://localhost:30000/v1/completions",
            json={"model": "llama", "prompt": p, "max_tokens": 50, "temperature": 0},
        )
        assert resp.status_code == 200
        tt_tokens = resp.json()["choices"][0]["logprobs"]["tokens"][:50]
        agreement = sum(t == r for t, r in zip(tt_tokens, ref)) / 50
        per_prompt_agreement.append(agreement)
        # Spec §8.2: no single prompt < 70%
        assert agreement >= 0.70, f"Prompt agreement {agreement:.2f} < 0.70: {p[:50]}"
    mean = sum(per_prompt_agreement) / len(per_prompt_agreement)
    # Spec §8.2: mean ≥ 90%
    assert mean >= 0.90, f"Mean agreement {mean:.2f} < 0.90"
```

- [ ] **Step 3:** Generate `tests/fixtures/llama31_greedy_50tok.json` once on CPU or a known-good reference machine using `transformers.AutoModelForCausalLM.from_pretrained(...).generate(do_sample=False, max_new_tokens=50)`. Commit the fixture alongside the test.

- [ ] **Step 4:** Run with the SGLang server up (from Phase G.5):

```bash
SGLANG_PLATFORM=tenstorrent pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness.py -v
```

**Pass:** mean ≥ 0.90, no prompt < 0.70. **Fail:** see decision gate below.

- [ ] **Step 5:** Commit: `test(tenstorrent): greedy correctness vs HF reference`.

### Task H.3: lm-eval-harness MMLU subset (optional, §8.3)

- [ ] **Step 1:** Spec §8.3 marks this as optional/recommended. If the implementer wants extra signal beyond §8.2's 10 prompts, run MMLU 5-shot, 100-question subset, and compare TT vs HF reference. **Pass:** TT ≥ HF − 1pp.

- [ ] **Step 2:** Skip if §8.2 already showed convincing agreement.

**Decision gate at end of Phase H — THIS IS THE P1 GATE:**

- If §8.1 fails: diagnose. Common roots — FutureMap sentinels (Phase E.1), bad token decoding (tokenizer path), wrapper bug surfaced by real prompts.
- If §8.2 fails: **do not proceed to stability or perf logging**. Greedy mismatch with HF is the only signal P1 has for KV correctness (§10 #8). Diagnose first. Likely culprits:
  1. Sampling path called via `model_runner.sample` instead of greedy-in-worker (§10 #11) — re-check Phase G.2 code.
  2. `bonus_token` / `accept_token` confusion only matters if speculative is involved — P1 doesn't use it. **N/A here.**
  3. KV state corruption inside the `tt_transformers` Generator — verify by replaying the same prompt through `tt_transformers`' demo directly (same `HF_MODEL`). If demo agrees with HF but our wrapper doesn't, the bug is in the wrapper or the worker.
  4. Tokenizer mismatch — Llama 3.1 uses tiktoken-style; verify SGLang's path produces the same `input_ids` as HF's tokenizer (spec §12 #4).

- If §8.1 + §8.2 pass: **P1 acceptance criteria 1 and 2 are met.** Proceed to Phase I.

---

## Phase I — Stability + performance log

**Goal:** Spec §8.4 stability passes (1 hour, no crashes, ITL p99 stable). Spec §8.5 performance numbers measured and appended to spec addendum.

**Files touched:**
- Update: `docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md` — append "P1 observed performance" addendum.
- Create: `phase-i-perf-log.txt` scratch file with raw numbers.

**Key API refs:**
- Spec §8.4 — stability test definition. ITL p99 over final 5 min vs minutes 5–10 must differ by < 10%; skip the first 5 minutes to exclude cold-start JIT.
- Spec §8.5 — perf log fields: cold first-prefill TTFT, warm prefill latency at lengths 100 / 500 / 1k / 4k, warm decode steady-state tok/s, D2H measured latency.
- `bench_serving.py` at concurrency=1.

**Live risks:** §10 #2 (~10–25 tok/s ceiling — document, don't fight), §10 #3 (D2H cost dominates at batch 1 — record), §10 #5 (tt-metal API drift — confirm docker image is still pinned).

### Task I.1: Stability test

- [ ] **Step 1:** With the server running from Phase G.5, launch:

```bash
python python/sglang/test/bench_serving.py \
  --backend sglang \
  --base-url http://localhost:30000 \
  --num-prompts <enough-for-1h> \
  --max-concurrency 1 \
  > /tmp/stability-run.log 2>&1
```

Adjust prompt count to consume ~1 hour at the observed decode rate.

- [ ] **Step 2:** From the log, compute ITL p99 over two windows: minutes 5–10 (post-warmup baseline) and final 5 minutes. Use `numpy.percentile`.

- [ ] **Step 3:** **Pass (spec §8.4):**
  - No crashes during the hour
  - No mesh OOM
  - `abs(p99_final - p99_baseline) / p99_baseline < 0.10`

  **Fail:** capture which minute the first error occurred, attach to incident log. Stability regressions usually indicate KV state leaks (request cleanup not happening) or program-cache eviction misbehavior — `execution_backend.free` is the first place to audit.

- [ ] **Step 4:** Commit log artifacts (sanitized, no sensitive paths): `test(tenstorrent): 1-hour stability run baseline`.

### Task I.2: Performance log (spec §8.5)

- [ ] **Step 1:** Record (do not gate):

| Metric | Method |
|---|---|
| Cold first-prefill TTFT | Fresh server start → first request 500-tok prompt; record TTFT |
| Warm prefill latency 100 / 500 / 1k / 4k | Issue prompts of each length after warmup; record TTFT |
| Warm decode steady-state tok/s | Concurrency=1 over 5 min after warmup; tokens / elapsed |
| D2H measured latency | Add inline timing around the `ttnn.to_torch(...)` call **inside `TTTransformersExecutionBackend`** (worker no longer touches ttnn — see §3.2 invariant #7); record p50 / p99 as `tt_d2h_latency_ms` |

- [ ] **Step 2:** Append a "P1 observed performance" section to the spec file. Format as a markdown table; include date, docker image hash, git commit. This is acceptance criterion #4 (§9).

- [ ] **Step 3:** Commit: `docs(tenstorrent): record P1 observed performance addendum`.

### Task I.3: Final acceptance check (spec §9)

- [ ] **Step 1:** Run through all seven §9 criteria:
  1. Smoke test (§8.1) passes — Phase H.1
  2. Greedy correctness (§8.2) passes — Phase H.2
  3. Stability test (§8.4) passes — Phase I.1
  4. Performance log (§8.5) committed — Phase I.2
  5. All unit tests (§8.6) pass: `pytest python/sglang/srt/hardware_backend/tenstorrent/test/ -v` clean
  6. Reset script (§7.4) works on a wedged mesh — manually verify by SIGKILLing the server, running the script, confirming next launch succeeds
  7. Spec self-review + user review complete — flag for user

- [ ] **Step 2:** If all seven pass, **P1 is done**. Open a `superpowers:finishing-a-development-branch` flow to decide on merge/PR.

**Decision gate at end of Phase I:** If any of the 7 criteria fail, do not declare P1 done; fix or document blockers. Specifically, do not move on to P2 scoping (spec §11) until P1 results are in.

---

## Cross-cutting reminders

### Dependencies (linear, with verification gates)

```
Phase 0 (verify) → A (skeleton) → B (cuda guards) → C (dispatch) → D (runner stub)
  → E (mesh+warmup+defaults) → F (wrapper) → G (worker) → H (P1 GATE) → I (stability+perf)
```

Each arrow is a "blocks on verification passing" relationship. Do not start a phase until the previous phase's verification command exits clean.

### Risk hot-spots by phase (cross-reference to spec §10)

| Phase | Live risks |
|---|---|
| 0 | #1 (Llama support), #10 (HF gate), §12 #7 (torch pin) |
| A | #5 (tt-metal API drift) |
| B | #7 (upstream churn) |
| C | #12 (overlap → garbage tokens) |
| D | #11 (silent-wrong-output A) |
| E | #4 (mesh leak), #12 (overlap, hard guard) |
| F | #1 (model support — second-line check), #8 (KV bug masking) |
| G | #11, #12 (both silent-wrong-output hazards) |
| H | #1, #8 (caught here if present), #11, #12 |
| I | #2 (tok/s ceiling — document), #3 (D2H cost) |

### Spec-coverage sanity check (self-review)

Run through spec §0–§13 with this plan in hand and confirm each item maps to a Task:

- §1 (hardware/software) — Phase 0
- §2 (prerequisites) — Phase 0
- §3 (architecture) — Phases A, D, E, F, G, H
- §4.1 (new files) — every file listed has a Task (A.1–A.3, D.1–D.2, E.1–E.4, F.1–F.2, G.1–G.4)
- §4.2 (core mods) — Phase B (~17 files) + Phase C (scheduler dispatch)
- §5 (data flow) — Phase G.2 (pseudocode reproduced verbatim)
- §6.1 (server-args defaults) — Phase E.1 (full body copied)
- §6.2 (observability) — Phase E.2 (logs) + Phase F.2 (metrics)
- §6.3 (concurrency) — documented, no code change
- §7.1 (startup errors) — Phase E.2 (mesh open), Phase F.1 (path check)
- §7.2 (runtime errors) — Phase G.2 (error paths)
- §7.3 (shutdown) — Phase E.2 (handlers in Scheduler subprocess)
- §7.4 (recovery script) — Phase E.4
- §8.1–8.6 (testing) — Phases H + I + all unit-test Tasks
- §9 (acceptance) — Phase I.3 final check
- §10 (risks) — referenced inline per phase
- §11 (P2/P3) — out of scope by spec
- §12 (open questions) — flagged inline (esp. §12 #2, #4, #7)
- §13 (refs) — N/A

### Open questions the plan flags

These come from spec §12 and were not pre-answered. The implementer must resolve each as they're encountered:

1. **Exact `tt_transformers` commit pin** (§12 #1) — resolve in Phase 0 (Tasks 0.1, 0.2). Pin the docker image hash; do not track main.
2. **Decode warmup shape coverage** (§12 #2) — does `tt_transformers` JIT once per layer or once total? Resolved by observation in Phase E.3 wall-time.
3. **`ttnn.from_torch` for int32 prompt tokens on Blackhole** (§12 #3) — surfaces in Phase F.1; if `Generator`'s prefill rejects int32, cast to int64 or whatever the library wants.
4. **HF tokenizer locale handling** (§12 #4) — surfaces only if Phase H.2 greedy correctness fails for multilingual prompt; diagnose by comparing tokenized prompt ids to HF's tokenizer output.
5. **`reset_devices.sh` safety while Python holds the device** (§12 #5) — empirically verify by Phase I.3 step 6 (manual reset after SIGKILL). Document the answer in the spec addendum.
6. **SGLang API drift** (§12 #6) — re-verify every load-bearing identifier in spec §6.1 / §5.1 / §6.2 / §7 with `grep` before each phase that touches it (Phase C re-verifies scheduler line numbers; Phase G re-verifies `forward_batch_generation` signature).
7. **Hidden assumptions** (§12 #7) — torch version compat is a hard go/no-go in Phase 0.5; all others (sampling_info population, fill_ids population, ttnn.to_torch returning CPU tensor) surface in Phase F or G — verify on contact, do not assume.
8. **`use_tt()` location and gating semantics** (§12 #8) — resolved in plan: gate on `isinstance(current_platform, TTSRTPlatform)`, no `SGLANG_USE_TT` env var. Phase C.1.

---

**End of plan.** Total tasks: ~32 across 10 phases (0 + A–I), plus the 6 cross-cutting open questions to resolve in-flight. Estimated 1–1.5 months of one engineer with prior ttnn familiarity (spec §0); 2–3 months without.
