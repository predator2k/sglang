# SGLang-on-Tenstorrent — Phase 3b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 14% perf gap to TT vLLM by rebasing tt-metal, then add tt-xla as a second execution backend for broader model coverage.

**Architecture:** Track 1 rebases the tt-metal fork from `89686ee78d` to `e867533`+ and rebuilds the container image so BFP4 works correctly. Track 2 brings up tt-xla (PJRT compiler) as a new backend following Pattern A (ModelRegistry). Track 3 expands model coverage and validates stability.

**Tech Stack:** Python 3.10, SGLang `tenstorrent-p1` branch, tt-metal (C++ + ttnn), tt-xla (PJRT + StableHLO + TT-MLIR), Podman containers, 2× Blackhole P150a.

**Spec:** [`../specs/2026-05-16-sglang-tenstorrent-p3b-design.md`](../specs/2026-05-16-sglang-tenstorrent-p3b-design.md)

---

## File Map

### Track 1 (tt-metal rebase) — files in tt-metal fork at `/home/mhnie/tt-metal-sglang/`

| Path | Action | Responsibility |
|------|--------|----------------|
| `models/tt_transformers/tt/generator_sglang.py` | Cherry-pick | SGLang integration (use_prefetcher param) |
| `models/tt_transformers/tt/model_config.py` | Cherry-pick | Qwen3 MAX_PREFILL_CHUNK_SIZE + precision config |
| `models/tt_transformers/tt/attention.py` | Cherry-pick | `_skip_self_attention` EAGLE tree-mask flag |
| `models/tt_transformers/tt/distributed_norm.py` | Audit/Drop | force_unsharded workaround (likely fixed upstream) |
| `models/tt_transformers/tt/model.py` | Audit | Blackhole grid/LM-head workarounds |
| `models/tt_transformers/tt/decoder.py` | Audit | Batched prefill reshape |
| `models/tt_transformers/tt/mlp.py` | Audit | experimental.minimal_matmul for long prefill |
| `scripts/bootstrap_container.sh` (SGLang repo) | Modify | Add `TT_METAL_IMAGE_TAG` env var |

### Track 2 (tt-xla) — files in SGLang repo

| Path | Action | Responsibility |
|------|--------|----------------|
| `models/tt_xla_model.py` | Create | `TenstorrentXLAGenericCausalLM` — HF model wrapper compiled via torch_xla |
| `models/registry.py` | Modify | Add tt-xla architecture list + conditional registration |
| `tp_worker.py` | Modify | Rename `_paged_mode` → `_uses_model_registry`; include `tt_xla` |
| `execution/tt_xla_backend.py` | Delete | Dead code (tt-xla uses Pattern A, not Pattern B ABC) |
| `test/test_tt_xla_smoke.py` | Create | Basic tt-xla inference test |
| `test/test_tt_xla_serving.py` | Create | End-to-end serving test |

### Track 3 (coverage + tuning)

| Path | Action | Responsibility |
|------|--------|----------------|
| `scripts/benchmark_dual_backend.py` | Create | Head-to-head comparison script |
| `test/_fixtures/model_compatibility_matrix.json` | Create | Model × backend × throughput × correctness |

All paths relative to `python/sglang/srt/hardware_backend/tenstorrent/` unless noted.

---

## Track 1: tt-metal Rebase (Weeks 1-4)

### Task 1.1: Pre-rebase compatibility check

**Files:**
- Read: `/home/mhnie/tt-metal-sglang/` (fork repo)
- Create: `test/_fixtures/v130_rebase_compat_check.json`

- [ ] **Step 1: Check target commit's KMD/FW requirements**

```bash
cd /home/mhnie/tt-metal-sglang
git fetch upstream
# Check the target commit for KMD version requirements
git show e867533:tt_metal/third_party/umd/device/api/umd/cluster.h 2>/dev/null | grep "KMD_VERSION\|MIN_FW" | head -5
# Check host KMD
/home/mhnie/.tenstorrent-venv/bin/tt-smi -s 2>&1 | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'KMD: {d[\"host_sw_vers\"][\"tt_smi\"]}')"
```

Document compatibility in evidence file. If incompatible, budget 1-2 days for host updates before proceeding.

- [ ] **Step 2: Commit evidence**

```bash
cd /home/mhnie/sglang
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v130_rebase_compat_check.json
git commit -m "evidence(tenstorrent): v130 — pre-rebase KMD/FW compatibility check"
```

### Task 1.2: Audit fork patches

**Files:**
- Read: all 22 commits between `89686ee78d` and HEAD on `tenstorrent-p1`
- Create: `test/_fixtures/v131_patch_audit.md`

- [ ] **Step 1: List all patches and categorize**

For each of the 22 commits, check if the upstream `e867533` has an equivalent fix:

```bash
cd /home/mhnie/tt-metal-sglang
# For each patch, check if the file it modifies exists and differs in e867533
for commit in $(git log --oneline 89686ee78d..HEAD --no-merges --format=%H); do
  msg=$(git log --oneline -1 $commit)
  files=$(git diff-tree --no-commit-id --name-only -r $commit | head -5)
  echo "=== $msg ==="
  echo "Files: $files"
  for f in $files; do
    if git show e867533:$f >/dev/null 2>&1; then
      echo "  $f: EXISTS in e867533 (check if our change is upstream)"
    else
      echo "  $f: NOT in e867533 (new file or renamed)"
    fi
  done
  echo
done
```

- [ ] **Step 2: Write audit document**

Classify each commit as DROP / CHERRY-PICK / ADAPT with rationale. Expected:

| Category | Count | Examples |
|----------|-------|---------|
| DROP | ~10 | Blackhole grid clamps, DRAM fallbacks, distributed_norm workaround |
| CHERRY-PICK | ~5 | generator_sglang.py, model_config.py, _skip_self_attention |
| ADAPT | ~3 | Prefetcher MUX mapping (API may have changed), precision config |
| SKIP (P3a experimental) | ~4 | Prefetcher 3-sub-device, ring size experiments |

- [ ] **Step 3: Commit audit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v131_patch_audit.md
git commit -m "evidence(tenstorrent): v131 — fork patch audit for rebase"
git push origin tenstorrent-p1
```

### Task 1.3: Create rebase branch and apply patches

**Files:**
- Modify: `/home/mhnie/tt-metal-sglang/` (new branch `rebase-p3b`)

- [ ] **Step 1: Create clean rebase branch**

```bash
cd /home/mhnie/tt-metal-sglang
git checkout -b rebase-p3b e867533
git submodule update --init --recursive
```

- [ ] **Step 2: Cherry-pick patches classified as CHERRY-PICK**

For each CHERRY-PICK commit from the audit:

```bash
git cherry-pick <commit-sha>
# If conflict: resolve, then git cherry-pick --continue
```

- [ ] **Step 3: Adapt patches classified as ADAPT**

For each ADAPT commit, manually apply the change to the new codebase. The upstream may have changed function signatures or file structure.

- [ ] **Step 4: Verify build**

```bash
# Inside a container with gcc-12+
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTT_METAL_BUILD_TESTS=OFF -DENABLE_TRACY=OFF
ninja -C build -j$(nproc)
# Expected: 1013/1013 targets, _ttnn.so built
ls build/ttnn/_ttnn.so build/tt_metal/libtt_metal.so
```

- [ ] **Step 5: Commit and push rebase branch**

```bash
git push origin rebase-p3b
```

### Task 1.4: Build new container image

**Files:**
- Modify: `scripts/bootstrap_container.sh`

- [ ] **Step 1: Write Dockerfile or build script for p3b image**

The container needs: rebased tt-metal (compiled from source), matching firmware, SGLang dependencies, Python 3.10 venv.

```bash
# Build image from the rebase-p3b branch
cd /home/mhnie/tt-metal-sglang
# Option A: build inside existing metalium base image
podman run --name tt-p3b-build \
  --device /dev/tenstorrent --ipc host --privileged \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $(pwd):/tt-metal-src \
  ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-models-amd64:latest-rc \
  bash -c '
    apt-get update && apt-get install -y gcc-12 g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    cd /tt-metal-src && \
    git submodule update --init --recursive && \
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTT_METAL_BUILD_TESTS=OFF -DENABLE_TRACY=OFF && \
    ninja -C build -j$(nproc) && \
    ln -sf /tt-metal-src/build/ttnn/_ttnn.so /tt-metal-src/ttnn/ttnn/_ttnn.so && \
    ln -sf /tt-metal-src/build/ttnn/_ttnncpp.so /tt-metal-src/ttnn/ttnn/_ttnncpp.so && \
    pip install -e /tt-metal-src
  '
podman commit tt-p3b-build localhost/local-tt-metal:p3b
podman rm tt-p3b-build
```

- [ ] **Step 2: Update bootstrap_container.sh**

Add `TT_METAL_IMAGE_TAG` support:

```bash
# In bootstrap_container.sh, change:
#   IMAGE="localhost/local-tt-metal:dev"
# To:
IMAGE="localhost/local-tt-metal:${TT_METAL_IMAGE_TAG:-dev}"
```

- [ ] **Step 3: Test the new image**

```bash
TT_METAL_IMAGE_TAG=p3b bash scripts/bootstrap_container.sh
podman exec p3a-ngram python -c "import ttnn; print('ttnn OK')"
```

- [ ] **Step 4: Commit bootstrap change**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/scripts/bootstrap_container.sh
git commit -m "feat(tenstorrent): bootstrap_container.sh supports TT_METAL_IMAGE_TAG"
git push origin tenstorrent-p1
```

### Task 1.5: Validate BFP4 + throughput on rebased image

**Files:**
- Create: `test/_fixtures/v132_rebase_validation.json`

- [ ] **Step 1: Launch server with BFP4+LOFI (default performance mode)**

```bash
/home/mhnie/.tenstorrent-venv/bin/tt-smi -r 0,1
podman exec -d -e SGLANG_PLATFORM=tenstorrent -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  p3a-ngram bash -lc 'source /opt/venv/bin/activate; python -m sglang.launch_server \
    --model-path /models/Qwen3-8B --trust-remote-code --host 0.0.0.0 --port 30000 \
    --device tenstorrent --max-running-requests 1 --context-length 2048 \
    --mem-fraction-static 0.5 --attention-backend torch_native --disable-cuda-graph \
    2>&1 | tee /tmp/sglang_p3b_bfp4.log'
```

- [ ] **Step 2: Test output quality**

Test completions, chat, and thinking mode:

```bash
# Completions
curl -s http://localhost:30000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"/models/Qwen3-8B","prompt":"The capital of France is","max_tokens":50,"temperature":0}'

# Chat with thinking
curl -s http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"/models/Qwen3-8B","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":200,"temperature":0}'
```

Verify: no garbage (`!!!`, `I2J`, `I'm so!` patterns), coherent reasoning in thinking chain.

- [ ] **Step 3: Benchmark throughput**

Run 5 iterations, report average tok/s. Target: >= 32 tok/s.

- [ ] **Step 4: Run GSM8K accuracy check**

```bash
sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-examples 50
```

Target: >= 78% accuracy.

- [ ] **Step 5: EAGLE cohost validation**

Launch with EAGLE-3 and verify it boots and produces output (correctness not required, just no crash):

```bash
podman exec -d -e SGLANG_PLATFORM=tenstorrent -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  -e SGLANG_TT_SPEC_DRAFT_PATH=/models/qwen3_8b_eagle3 -e SGLANG_TT_SPEC_DRAFT_BACKEND=cpu \
  p3a-ngram bash -lc 'source /opt/venv/bin/activate; python -m sglang.launch_server \
    --model-path /models/Qwen3-8B --trust-remote-code --host 0.0.0.0 --port 30000 \
    --device tenstorrent --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /models/qwen3_8b_eagle3 \
    --speculative-eagle-topk 1 --speculative-num-steps 1 --speculative-num-draft-tokens 2 \
    --max-running-requests 1 --context-length 2048 --mem-fraction-static 0.5 \
    --attention-backend torch_native --disable-cuda-graph \
    2>&1 | tee /tmp/eagle_p3b.log'
```

- [ ] **Step 6: Document results and commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v132_rebase_validation.json
git commit -m "evidence(tenstorrent): v132 — rebase validation (BFP4, throughput, EAGLE)"
git push origin tenstorrent-p1
```

### Task 1.6: Track 1 gate

- [ ] **Step 1: Verify all Track 1 exit criteria**

| Criterion | Pass? |
|-----------|-------|
| BFP4+LOFI correct output (completions + chat + thinking) | |
| Throughput >= 32 tok/s | |
| GSM8K >= 78% | |
| EAGLE cohost boots | |
| Old image preserved as rollback | |

If any criterion fails, debug using the rollback image for comparison. Do NOT proceed to Track 2 until Track 1 passes.

---

## Track 2: tt-xla Bring-up (Weeks 5-7)

### Task 2.1: tt-xla discovery — install and test

**Files:**
- Create: `test/_fixtures/v133_ttxla_discovery.json`

- [ ] **Step 1: Check tt-xla installation requirements**

```bash
# Clone tt-xla
git clone https://github.com/tenstorrent/tt-xla.git /home/mhnie/tt-xla
cd /home/mhnie/tt-xla
cat README.md | head -50
cat requirements.txt 2>/dev/null || cat pyproject.toml | head -30
```

- [ ] **Step 2: Install tt-xla in the p3b container**

```bash
podman exec p3a-ngram bash -c '
  pip install /path/to/tt-xla  # or pip install git+https://github.com/tenstorrent/tt-xla.git
  python -c "import torch_xla; print(torch_xla.__version__)"
'
```

If version conflicts with ttnn: try a separate venv. Document the outcome.

- [ ] **Step 3: Run tt-xla basic test**

```bash
podman exec p3a-ngram python -c "
import torch
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f'Device: {device}')
x = torch.randn(2, 3).to(device)
y = torch.randn(3, 4).to(device)
z = x @ y
print(f'Matmul result shape: {z.shape}')
print(f'Result: {z.cpu()}')
"
```

- [ ] **Step 4: Run a small model forward pass**

```bash
podman exec p3a-ngram python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
inputs = tokenizer('Hello world', return_tensors='pt').to(device)
with torch.no_grad():
    outputs = model(**inputs)
print(f'Logits shape: {outputs.logits.shape}')
# Decode
next_token = outputs.logits[0, -1].argmax().item()
print(f'Next token: {tokenizer.decode(next_token)}')
"
```

- [ ] **Step 5: Measure first-compilation latency and throughput**

Time the first vs second forward pass. Document compilation overhead.

- [ ] **Step 6: Go/no-go decision**

If steps 3-4 fail: document why, pivot to alternative work (more tt_transformers models, batched throughput, EAGLE improvements). Skip Tasks 2.2-2.5.

If steps 3-4 succeed: proceed. Write time estimate for integration.

- [ ] **Step 7: Commit discovery evidence**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v133_ttxla_discovery.json
git commit -m "evidence(tenstorrent): v133 — tt-xla discovery (go/no-go)"
git push origin tenstorrent-p1
```

### Task 2.2: Rename `_paged_mode` in tp_worker.py

**Files:**
- Modify: `tp_worker.py`

- [ ] **Step 1: Rename flag**

In `tp_worker.py` line 65, change:

```python
# Before:
self._paged_mode = resolve_execution_backend_name() == "tt_transformers_paged"

# After:
_backend = resolve_execution_backend_name()
self._uses_model_registry = _backend in ("tt_transformers_paged", "tt_xla")
```

And in `_init_model_runner` (line 143):

```python
# Before:
if self._paged_mode:

# After:
if self._uses_model_registry:
```

- [ ] **Step 2: Verify no regression**

Launch the existing tt_transformers_paged server and confirm it still works:

```bash
curl -s http://localhost:30000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"/models/Qwen3-8B","prompt":"Hello","max_tokens":10,"temperature":0}'
```

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py
git commit -m "refactor(tenstorrent): rename _paged_mode to _uses_model_registry for tt-xla"
git push origin tenstorrent-p1
```

### Task 2.3: Create TenstorrentXLAGenericCausalLM

**Files:**
- Create: `models/tt_xla_model.py`

- [ ] **Step 1: Write the model wrapper**

```python
# python/sglang/srt/hardware_backend/tenstorrent/models/tt_xla_model.py
import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


class TenstorrentXLAGenericCausalLM(nn.Module):
    """Generic HF CausalLM wrapper compiled via torch_xla for TT devices.

    Follows Pattern A (ModelRegistry): loaded by SGLang's ModelRunner,
    same interface as TTModels in tt_llm.py.
    """

    def __init__(self, config, quant_config=None, **kwargs):
        super().__init__()
        import torch_xla.core.xla_model as xm
        from transformers import AutoModelForCausalLM

        self.config = config
        self.device = xm.xla_device()
        model_path = getattr(config, "_name_or_path", None)
        logger.info(f"[TT-XLA] Loading {model_path} onto {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        logger.info(f"[TT-XLA] Model loaded: {type(self.model).__name__}")

    def forward(self, forward_batch):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        input_ids = forward_batch.input_ids.to(self.device)
        positions = forward_batch.positions.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.unsqueeze(0))

        logits = outputs.logits[:, -1, :].cpu().float()
        return LogitsProcessorOutput(next_token_logits=logits)
```

Note: This is a minimal first implementation. KV cache management, batched decode, and proper prefill/decode mode handling will be added iteratively based on what torch_xla supports.

- [ ] **Step 2: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/models/tt_xla_model.py
git commit -m "feat(tenstorrent): TenstorrentXLAGenericCausalLM — minimal tt-xla model wrapper"
git push origin tenstorrent-p1
```

### Task 2.4: Register tt-xla models in registry

**Files:**
- Modify: `models/registry.py`

- [ ] **Step 1: Add conditional tt-xla registration**

Add to `register_tt_models()` in `registry.py`:

```python
def register_tt_models():
    """Register TT-Metal models with SGLang's model registry."""
    logger.info("[TT-Plugin] register_tt_models() called")
    try:
        from sglang.srt.hardware_backend.tenstorrent.execution import resolve_execution_backend_name

        backend = resolve_execution_backend_name()

        if backend == "tt_xla":
            # tt-xla: register generic wrapper for supported architectures
            from .tt_xla_model import TenstorrentXLAGenericCausalLM

            TT_XLA_ARCHITECTURES = [
                "LlamaForCausalLM",
                "Qwen2ForCausalLM",
                "Qwen3ForCausalLM",
                "MistralForCausalLM",
                "PhiForCausalLM",
                "Phi3ForCausalLM",
                "GemmaForCausalLM",
                "Gemma2ForCausalLM",
            ]
            xla_registry = {arch: TenstorrentXLAGenericCausalLM for arch in TT_XLA_ARCHITECTURES}
            ModelRegistry.models.update(xla_registry)
            logger.info(f"[TT-Plugin] ✓ Registered tt-xla for {len(xla_registry)} architectures")
        else:
            # tt_transformers: register model-specific classes
            TT_MODEL_REGISTRY = _build_tt_model_registry()
            ModelRegistry.models.update(TT_MODEL_REGISTRY)
            logger.info(f"[TT-Plugin] ✓ Registered {len(TT_MODEL_REGISTRY)} TT models")

        _patch_model_registry_lazy_eagle3()

    except Exception as e:
        logger.error(f"[TT-Plugin] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
```

- [ ] **Step 2: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/models/registry.py
git commit -m "feat(tenstorrent): conditional tt-xla model registration in registry"
git push origin tenstorrent-p1
```

### Task 2.5: Remove dead tt_xla_backend.py stub

**Files:**
- Delete: `execution/tt_xla_backend.py`
- Modify: `execution/__init__.py`

- [ ] **Step 1: Remove the stub file**

```bash
rm python/sglang/srt/hardware_backend/tenstorrent/execution/tt_xla_backend.py
```

- [ ] **Step 2: Update execution/__init__.py**

Remove the import of `tt_xla_backend`:

```python
# Before:
from sglang.srt.hardware_backend.tenstorrent.execution import (
    tt_transformers_backend,
    tt_xla_backend,
)

# After:
from sglang.srt.hardware_backend.tenstorrent.execution import (
    tt_transformers_backend,
)
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(tenstorrent): remove dead tt_xla_backend.py (tt-xla uses Pattern A)"
git push origin tenstorrent-p1
```

### Task 2.6: First model serving via tt-xla

**Files:**
- Create: `test/test_tt_xla_smoke.py`
- Create: `test/_fixtures/v134_ttxla_first_model.json`

- [ ] **Step 1: Launch server with tt-xla backend**

```bash
/home/mhnie/.tenstorrent-venv/bin/tt-smi -r 0,1
podman exec -d -e SGLANG_PLATFORM=tenstorrent -e SGLANG_TT_EXECUTION_BACKEND=tt_xla \
  p3a-ngram bash -lc 'source /opt/venv/bin/activate; python -m sglang.launch_server \
    --model-path /models/Phi-4 --trust-remote-code --host 0.0.0.0 --port 30000 \
    --device tenstorrent --max-running-requests 1 --context-length 2048 \
    --mem-fraction-static 0.5 --attention-backend torch_native --disable-cuda-graph \
    2>&1 | tee /tmp/sglang_ttxla.log'
```

If Phi-4 doesn't fit or isn't available, use Gemma-3-4B as fallback.

- [ ] **Step 2: Test output quality**

```bash
curl -s http://localhost:30000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"Phi-4","prompt":"The capital of France is","max_tokens":30,"temperature":0}'
```

- [ ] **Step 3: Benchmark throughput**

Target: >= 5 tok/s (minimum viable).

- [ ] **Step 4: Write smoke test**

```python
# test/test_tt_xla_smoke.py
"""Smoke test for tt-xla backend — verifies model loads and produces output."""
import json
import urllib.request

def test_tt_xla_completions():
    url = "http://localhost:30000/v1/completions"
    data = json.dumps({"model": "test", "prompt": "Hello", "max_tokens": 10, "temperature": 0}).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    text = resp["choices"][0]["text"]
    assert len(text) > 0, "Empty output"
    assert "!!!" not in text, "Garbage output detected"
    print(f"PASS: {text[:50]}")

if __name__ == "__main__":
    test_tt_xla_completions()
```

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_tt_xla_smoke.py
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v134_ttxla_first_model.json
git commit -m "feat(tenstorrent): tt-xla first model serving + smoke test"
git push origin tenstorrent-p1
```

### Task 2.7: Track 2 gate

- [ ] **Step 1: Verify Track 2 exit criteria**

| Criterion | Pass? |
|-----------|-------|
| 1 non-tt_transformers model serving via tt-xla | |
| Throughput >= 5 tok/s | |
| Output correct (no garbage) | |
| Smoke test passes | |

---

## Track 3: Model Coverage + Tuning (Weeks 8-10)

### Task 3.1: Cross-backend comparison (Qwen3-8B)

**Files:**
- Create: `scripts/benchmark_dual_backend.py`
- Create: `test/_fixtures/v135_cross_backend.json`

- [ ] **Step 1: Write benchmark script**

```python
# scripts/benchmark_dual_backend.py
"""Compare tt_transformers vs tt-xla on the same model."""
import json
import time
import urllib.request
import sys

def bench(url, model, prompt="The capital of Japan is", max_tokens=100, n=5):
    h = {"Content-Type": "application/json"}
    d = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}).encode()
    # Warmup
    for _ in range(2):
        req = urllib.request.Request(url, d, h)
        with urllib.request.urlopen(req, timeout=300) as r: json.loads(r.read())
    # Measure
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        req = urllib.request.Request(url, d, h)
        with urllib.request.urlopen(req, timeout=300) as r: resp = json.loads(r.read())
        elapsed = time.perf_counter() - t0
        ntok = resp["usage"]["completion_tokens"]
        times.append(ntok / elapsed)
    avg = sum(times) / len(times)
    return {"avg_tok_s": round(avg, 1), "tpot_ms": round(1000 / avg, 1), "runs": [round(t, 1) for t in times]}

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000/v1/completions"
    model = sys.argv[2] if len(sys.argv) > 2 else "/models/Qwen3-8B"
    result = bench(url, model)
    print(json.dumps(result, indent=2))
```

- [ ] **Step 2: Run on tt_transformers backend**

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged  # launch server
python scripts/benchmark_dual_backend.py http://localhost:30000/v1/completions /models/Qwen3-8B > /tmp/bench_tt_transformers.json
```

- [ ] **Step 3: Run on tt-xla backend**

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_xla  # launch server
python scripts/benchmark_dual_backend.py http://localhost:30000/v1/completions /models/Qwen3-8B > /tmp/bench_tt_xla.json
```

- [ ] **Step 4: Document comparison and commit**

```bash
git add scripts/benchmark_dual_backend.py test/_fixtures/v135_cross_backend.json
git commit -m "evidence(tenstorrent): v135 — cross-backend comparison (tt_transformers vs tt-xla)"
git push origin tenstorrent-p1
```

### Task 3.2: Model compatibility matrix

**Files:**
- Create: `test/_fixtures/model_compatibility_matrix.json`

- [ ] **Step 1: Test each model on each backend**

For each model × backend combination, record: status (works/fails/untested), throughput, correctness.

- [ ] **Step 2: Write matrix file**

```json
{
  "date": "2026-XX-XX",
  "device": "2xP150a",
  "models": {
    "Qwen3-8B": {
      "tt_transformers": {"status": "works", "tok_s": 32.6, "correct": true, "precision": "BFP4+LOFI"},
      "tt_xla": {"status": "works", "tok_s": "?", "correct": true}
    },
    "Phi-4": {
      "tt_transformers": {"status": "unsupported"},
      "tt_xla": {"status": "works", "tok_s": "?", "correct": true}
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add test/_fixtures/model_compatibility_matrix.json
git commit -m "evidence(tenstorrent): model compatibility matrix"
git push origin tenstorrent-p1
```

### Task 3.3: 6-hour soak test (tt_transformers)

**Files:**
- Create: `test/_fixtures/v136_soak_test_6h.json`

- [ ] **Step 1: Launch server and run continuous workload**

```bash
# Launch server with tt_transformers_paged + BFP4
# Then run continuous requests for 6 hours, sampling throughput every 30 minutes
python -c "
import time, json, urllib.request

url = 'http://localhost:30000/v1/completions'
h = {'Content-Type': 'application/json'}
d = json.dumps({'model':'/models/Qwen3-8B','prompt':'Explain quantum computing.','max_tokens':100,'temperature':0}).encode()

results = []
start = time.time()
while time.time() - start < 6 * 3600:  # 6 hours
    t0 = time.perf_counter()
    req = urllib.request.Request(url, d, h)
    with urllib.request.urlopen(req, timeout=300) as r: resp = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    n = resp['usage']['completion_tokens']
    tps = n / elapsed
    results.append({'elapsed_h': round((time.time() - start) / 3600, 2), 'tok_s': round(tps, 1)})
    if len(results) % 60 == 0:
        print(f'{results[-1][\"elapsed_h\"]}h: {tps:.1f} tok/s')

# Check drift
first_hour = [r['tok_s'] for r in results if r['elapsed_h'] < 1]
last_hour = [r['tok_s'] for r in results if r['elapsed_h'] > 5]
avg_first = sum(first_hour) / len(first_hour)
avg_last = sum(last_hour) / len(last_hour)
drift = abs(avg_last - avg_first) / avg_first * 100
print(f'Drift: {drift:.1f}% (target: < 5%)')
"
```

- [ ] **Step 2: Verify drift < 5% and commit**

```bash
git add test/_fixtures/v136_soak_test_6h.json
git commit -m "evidence(tenstorrent): v136 — 6-hour soak test (tt_transformers)"
git push origin tenstorrent-p1
```

### Task 3.4: P3b final gate

- [ ] **Step 1: Verify all P3b exit criteria**

| # | Criterion | Pass? |
|---|-----------|-------|
| 1 | tt_transformers Qwen3-8B BFP4+LOFI >= 32 tok/s correct | |
| 2 | tt-xla serving 1+ model not in tt_transformers, >= 5 tok/s | |
| 3 | Both backends selectable via env var | |
| 4 | Model compatibility matrix published | |
| 5 | 6-hour soak test drift < 5% | |
| 6 | EAGLE cohost functional on rebased tt-metal | |

- [ ] **Step 2: Final commit**

```bash
git commit -m "docs(tenstorrent): P3b ACCEPTED — all exit criteria met"
git push origin tenstorrent-p1
```
