# tt-xla Tilize-bottleneck Attack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Tilize bottleneck in tt-xla Qwen3-8B serving (81% of decode time per Tracy → target near tt_transformers_paged's 37ms TPOT vs current 132ms) via three serial attacks: A.1.a (set argumentTypeMap so const-eval fires), A.2.a (fold redundant layout ops, conditional), A.3.a (auto-detect parameter markers).

**Architecture:** Three independent compiler-level changes layered onto an existing tt-xla → tt-mlir → tt-metal stack. Phase 1 repoints tt-xla's `ExternalProject_Add(tt-mlir ...)` at the local fork at `/home/mhnie/tt-mlir-sglang/`. Phase 2 wires a `classifyArgs` helper into `module_builder.cc` so the existing `tt-populate-argument-types` pass populates `ArgumentTypeAttr::Parameter` on weight block args, triggering the already-wired `ConstEvalHoistTransform` pipeline. Phase 3 adds a new `TTNNFoldThroughAgnosticOps` pass that folds non-adjacent `to_layout` ops through layout-agnostic unary ops. Phase 4 ports the same classifier into a tt-mlir pass that runs after `tt-populate-argument-types`, so auto-detection works without a plugin-side map.

**Tech Stack:** C++17 (pjrt-plugin-tt, tt-mlir), Python 3.10 (sglang, torch_xla), MLIR/LLVM (tt-mlir dialects), CMake (build), Tenstorrent Blackhole p150a hardware.

---

## Non-negotiable scope — executor instructions

This plan instructs the executor to fully implement every assigned task. The following are NOT acceptable reasons to stop, defer, or descope:

- "out of scope"
- "too hard" / "too complex"
- "too much code involved"
- "not feasible without further design"
- "takes too long to compile"
- "requires code modification in another repo" (this work spans three: `/home/mhnie/sglang`, `/home/mhnie/tt-mlir-sglang`, `/home/mhnie/tt-xla/pjrt_implementation`)
- "I'll defer this to a follow-up"
- "the build broke, I'll skip ahead to the next task" (build failures are blockers; surface as questions, do not auto-skip)
- "verification would take too long, I'll trust the diff" (verification is mandatory)

Genuine blockers are surfaced as questions, not unilateral descope decisions.

Two phases (Phase 3 and the heuristic fallback within Phase 4) have explicit measurement gates — those are the ONLY authorized descope triggers. Everything else is mandatory.

---

## Spec

Full spec: `/home/mhnie/sglang/docs/superpowers/specs/2026-05-17-tt-xla-tilize-attack-design.md` (HEAD `d3b2d2616`, 17 review rounds).

Read the spec before starting Task 1. It contains the rationale for every mechanism in this plan.

---

## File structure

### Files to create

| File | Purpose | Phase |
|---|---|---|
| `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-source-dir-fork.patch` | Patch to `/home/mhnie/tt-xla/third_party/CMakeLists.txt` repointing ExternalProject_Add at the fork | 1 |
| `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-mlir-sglang-build-script-cleanup.patch` | Patch to `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh` (drop dead flag, export TT_METAL_RUNTIME_ROOT) | 1 |
| `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch` | Patch to `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc` adding `classifyArgs` + two assignments | 2 |
| `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/phase0_bucket_tilize.py` | Phase 0.1 tool: bucket Tracy Tilizes by weight shape | 0 |
| `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/phase0_count_to_layout_triples.py` | Phase 0.3 tool: count `to_layout`-pair triples via Python MLIR bindings | 0 |
| `/home/mhnie/tt-mlir-sglang/lib/Dialect/StableHLO/Transforms/PopulateArgumentTypesAutoDetect.cpp` | Phase 4 auto-detect pass | 4 |
| `/home/mhnie/tt-mlir-sglang/test/ttmlir/Conversion/StableHLOToTTIR/auto_detect_argument_types.mlir` | Phase 4 lit test | 4 |
| `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Transforms/TTNNFoldThroughAgnosticOps.cpp` | Phase 3 fold pass (CONDITIONAL on Phase 0.3 ≥ 50) | 3 |

### Files to modify

| File | Phase | Lines (approximate, confirm at execution) |
|---|---|---|
| `/home/mhnie/tt-xla/third_party/CMakeLists.txt` | 1 | 47–85 (`ExternalProject_Add`) and 35 (`TTMLIR_BUILD_DIR`) |
| `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh` | 1 | drop line 38 `-DTTMLIR_SOURCE_DIR_OVERRIDE` flag; insert `TT_METAL_RUNTIME_ROOT` export before `[3/4]` block |
| `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc` | 2 | new `classifyArgs` helper + assignments at ~832 (after `resultPresharded`) and ~985 (after `options` construction) |
| `/home/mhnie/tt-mlir-sglang/include/ttmlir/Dialect/StableHLO/Transforms/Passes.td` | 4 | new pass registration |
| `/home/mhnie/tt-mlir-sglang/lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp` | 4 | insert new pass after `createTTPopulateArgumentTypes` at line 24, before `createAnalyzeMeshPass` at line 52 |
| `/home/mhnie/tt-mlir-sglang/include/ttmlir/Dialect/TTNN/Transforms/Passes.td` | 3 | new pass registration (CONDITIONAL) |
| `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` | 3 | insert in helper at line ~251 (CONDITIONAL) |
| `/home/mhnie/sglang/docs/platforms/tt_xla_tpot_handoff_2026-05-17.md` | 5 | new TPOT table rows; "SUPERSEDED" note on Open work section |

### Fixtures generated

| File | Phase |
|---|---|
| `/tmp/canonical-lib-shas.txt` | 0.0 |
| `/tmp/shlo_dump/irs/shlo_*.mlir`, `ttnn_*.mlir`, etc. | 0.2, 0.3, 2, 3, 4 |
| `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase1_smoke.json` | 1 |
| `_fixtures/v146_3run_server_q8b_phase2_a1a.json` | 2 |
| `_fixtures/v146_3run_server_q8b_phase3_a2a.json` | 3 (conditional) |
| `_fixtures/v146_3run_server_q8b_phase4_a3a.json` | 4 |

---

## Common execution context

**Containers:**
- `tt-xla-eval` — runs the tt-xla plugin; mounts `/home/mhnie/sglang→/sglang`, `/home/mhnie/tt-xla→/tt-xla`, `/home/mhnie/tt-mlir-sglang→/tt-mlir-sglang`, `/opt/tt-mlir-toolchain`.
- Inside the container, paths drop the `/home/mhnie/` prefix (e.g., host `/home/mhnie/tt-mlir-sglang` is `/tt-mlir-sglang` inside).

**Repos / branches:**
- `/home/mhnie/sglang` — branch `tenstorrent-p1` (the repo we commit plan progress to).
- `/home/mhnie/tt-mlir-sglang` — branch `tenstorrent-p1` (fork of tt-mlir).
- `/home/mhnie/tt-xla` — host clone, detached at canonical `470f0fad`.

**Always include in bench/probe commands:**
- `BYPASS_PREWARM=1` (env var, NOT a CLI flag — handoff doc Pitfall #4).
- `SGLANG_TT_CACHE_MODE=index_copy` (consistency with the shipped baseline).

**Bench scripts** (in `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/`):
- `bench_3run_server_alive.py` — server-alive 3-condition bench across v145 models, supports `--backend tt_xla|tt_transformers_single|tt_transformers_paged`.
- `probe_decode_op_profile.py` — Tracy-compatible probe with `--weight-dtype` / `--decode` flags.
- `_fixtures/tracy_aggregate.py` — aggregates `ops_perf_results_*.csv` by OP CODE.

---

# Phase 0 — Investigations

Phase 0 runs **before any rebuild**. Its outputs gate Phase 3 and inform Phase 2. The only mandatory output is Phase 0.0's SHA snapshot.

### Task 0.0: Snapshot canonical .so SHAs (Phase 1 baseline)

**Files:**
- Create: `/tmp/canonical-lib-shas.txt`

- [ ] **Step 1: Snapshot canonical library SHAs before any edits**

Run:
```bash
docker exec tt-xla-eval bash -c \
  'sha256sum /tt-xla/third_party/tt-mlir/install/lib/*.so' \
  > /tmp/canonical-lib-shas.txt
```

Verify:
```bash
wc -l /tmp/canonical-lib-shas.txt
```

Expected: ≥ 1 line (≥ 1 .so file). If 0 lines, the install/lib directory is empty — STOP, investigate before proceeding.

- [ ] **Step 2: Verify libTTMLIRCompiler.so is present**

Run:
```bash
grep -c libTTMLIRCompiler.so /tmp/canonical-lib-shas.txt
```

Expected: `1`. This is the library that Phase 1's SHA-diff verification specifically expects to change after the fork rebuild.

---

### Task 0.1: Bucket Tracy Tilizes by tensor shape

**Files:**
- Create: `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/phase0_bucket_tilize.py`

**Context:** Phase 0.1's purpose is to classify Tilize ops as "weight tilize" (eliminable by A.1.a/A.3.a via const-eval) vs "activation tilize" (eliminable by A.2.a only). The aggregator at `_fixtures/tracy_aggregate.py` only reads `OP CODE` — no shape columns. We must read the raw `ops_perf_results_*.csv` directly. Tracy emits four padded-logical-dim columns per input: `INPUT_0_W_PAD[LOGICAL]`, `INPUT_0_Z_PAD[LOGICAL]`, `INPUT_0_Y_PAD[LOGICAL]`, `INPUT_0_X_PAD[LOGICAL]`.

- [ ] **Step 1: Locate an existing Tracy CSV fixture**

Run:
```bash
docker exec tt-xla-eval bash -c \
  'find /tmp -name "ops_perf_results_*.csv" 2>/dev/null | head -3'
```

If empty: re-run an existing Tracy probe to generate one. Per the handoff doc:
```bash
docker exec -d tt-xla-eval bash -c 'cd /sglang && python3 -m tracy -v -r -p -o /tmp/tracy_phase0 \
  python/sglang/srt/hardware_backend/tenstorrent/test/probe_decode_op_profile.py \
  --model Qwen/Qwen3-8B --input-len 1024 --decode 5 --skip-warmup 0 --out-tag phase0 \
  > /tmp/phase0_probe.log 2>&1'
```

Wait for probe to finish (check `pgrep -af probe_decode`), then locate the CSV:
```bash
docker exec tt-xla-eval bash -c 'find /tmp/tracy_phase0 -name "ops_perf_results_*.csv"'
```

- [ ] **Step 2: Inspect CSV columns**

Run:
```bash
docker exec tt-xla-eval bash -c 'head -1 /tmp/tracy_phase0/reports/*/ops_perf_results_*.csv | tr "," "\n" | cat -n | head -40'
```

Expected: see numbered list of columns including `OP CODE`, `INPUT_0_W_PAD[LOGICAL]`, `INPUT_0_Z_PAD[LOGICAL]`, `INPUT_0_Y_PAD[LOGICAL]`, `INPUT_0_X_PAD[LOGICAL]`, `INPUT_0_LAYOUT`, `INPUT_0_DATATYPE`, `INPUT_0_MEMORY`.

- [ ] **Step 3: Write the bucket script**

Create `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/phase0_bucket_tilize.py`:

```python
"""Phase 0.1 — bucket Tracy Tilize ops by weight shape.

Usage:
    python3 phase0_bucket_tilize.py <ops_perf_results.csv> <hf_model_dir>

Output: prints (weight_tilize_count, activation_tilize_count, weight_fraction).
"""
import csv
import sys
from pathlib import Path


def load_weight_shapes(hf_model_dir: str) -> set[tuple[int, ...]]:
    """Load all (named_parameters) shapes from an HF model dir."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(hf_model_dir, torch_dtype=torch.float32)
    shapes = set()
    for _name, param in model.named_parameters():
        if param.numel() < 1000:
            continue
        shapes.add(tuple(param.shape))
    return shapes


def pad_to_tile(dim: int, tile: int = 32) -> int:
    return ((dim + tile - 1) // tile) * tile


def shape_matches_weight(input_shape: tuple[int, ...],
                         weight_shapes: set[tuple[int, ...]]) -> bool:
    """Match modulo tile-padding. input_shape is from the four W/Z/Y/X cols."""
    # Drop leading 1s (Tracy emits 4D; weights may be 1D/2D/3D)
    trimmed = tuple(d for d in input_shape if d != 1)
    for w in weight_shapes:
        padded = tuple(pad_to_tile(d) for d in w)
        if trimmed == padded:
            return True
        if trimmed == w:
            return True
    return False


def main(csv_path: str, hf_model_dir: str) -> None:
    weight_shapes = load_weight_shapes(hf_model_dir)
    print(f"Loaded {len(weight_shapes)} unique weight shapes from {hf_model_dir}")

    weight_count = 0
    activation_count = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("OP CODE", "").strip() != "TilizeWithValPadding":
                continue
            try:
                w = int(row["INPUT_0_W_PAD[LOGICAL]"])
                z = int(row["INPUT_0_Z_PAD[LOGICAL]"])
                y = int(row["INPUT_0_Y_PAD[LOGICAL]"])
                x = int(row["INPUT_0_X_PAD[LOGICAL]"])
            except (KeyError, ValueError):
                continue
            input_shape = (w, z, y, x)
            if shape_matches_weight(input_shape, weight_shapes):
                weight_count += 1
            else:
                activation_count += 1

    total = weight_count + activation_count
    if total == 0:
        print("No TilizeWithValPadding ops found in CSV.")
        return
    fraction = weight_count / total
    print(f"weight_tilize:     {weight_count}")
    print(f"activation_tilize: {activation_count}")
    print(f"weight_fraction:   {fraction:.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run the bucket script against the existing CSV**

Run:
```bash
docker exec tt-xla-eval bash -c \
  'cd /sglang && python3 python/sglang/srt/hardware_backend/tenstorrent/test/phase0_bucket_tilize.py \
   /tmp/tracy_phase0/reports/*/ops_perf_results_*.csv /root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/'
```

Output: `weight_tilize`, `activation_tilize`, `weight_fraction`. Save to Phase 5 reporting.

- [ ] **Step 5: Document outcome**

If `weight_fraction ≥ 5%`: proceed with both A.1.a (Phase 2) and A.3.a (Phase 4) as primary attacks.
If `weight_fraction < 5%`: prioritize A.2.a (Phase 3), but still land A.1.a and A.3.a as wired infrastructure.

- [ ] **Step 6: Commit the script**

```bash
cd /home/mhnie/sglang
git add python/sglang/srt/hardware_backend/tenstorrent/test/phase0_bucket_tilize.py
git commit -m "phase0(tt-xla): script to bucket Tracy Tilizes by weight shape

Reads raw ops_perf_results_*.csv, joins TilizeWithValPadding rows
against hf_model.named_parameters() shapes (modulo tile-padding).
Outputs weight_tilize / activation_tilize counts + ratio.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 0.2: Dump StableHLO + discover torch_xla parameter marker

**Files:**
- Read: dump under `/tmp/shlo_dump/irs/`

**Context:** Phase 0.2 confirms two things: (1) the `export_path` round-trip from Python to C++ works, (2) which `mhlo.`/`xla.`/`tf.`/`jax.`/`torch.`/`_xla` -prefixed arg attr torch_xla emits on block args of weight parameters. The answer determines Strategy A/B/C selection in Phase 2.

- [ ] **Step 1: Trigger one compile with export_path set**

In an existing tt-xla model-loading path (or a one-off probe), set:
```python
import torch_xla
torch_xla.set_custom_compile_options({
    "export_path": "/tmp/shlo_dump",
    "enable_const_eval": True,
})
```

Easiest mechanism: re-run the probe from Phase 0.1 step 1 but ALSO set these options. The probe at `probe_decode_op_profile.py` can be edited to set them, or use a wrapper:

```bash
docker exec -d tt-xla-eval bash -c 'cd /sglang && \
  python3 -c "
import torch_xla
torch_xla.set_custom_compile_options({\"export_path\": \"/tmp/shlo_dump\", \"enable_const_eval\": True})
import subprocess
subprocess.run([\"python3\", \"python/sglang/srt/hardware_backend/tenstorrent/test/probe_decode_op_profile.py\",
                \"--model\", \"Qwen/Qwen3-8B\", \"--input-len\", \"1024\", \"--decode\", \"1\",
                \"--skip-warmup\", \"0\", \"--out-tag\", \"phase02\"], check=True)
" > /tmp/phase02.log 2>&1'
```

Wait for completion.

- [ ] **Step 2: Confirm the round-trip succeeded (Phase 0.2 round-trip gate)**

Run:
```bash
docker exec tt-xla-eval bash -c 'ls /tmp/shlo_dump/irs/shlo_*.mlir 2>&1 | head -5'
```

Expected: lists at least one `shlo_<timestamp>.mlir` (or `shlo_<model>_<timestamp>.mlir`) file.

If the directory `/tmp/shlo_dump/irs/` doesn't exist OR is empty: the `export_path` option didn't reach `module_builder.cc`. STOP. Investigate Python→C++ option-key serialization. Per the spec's risk note, `compile_options.cc:78–82` does NOT ABORT for default `TTNNFlatbuffer` backend, so the symptom is "no files."

- [ ] **Step 3: Grep for parameter markers**

Run:
```bash
docker exec tt-xla-eval bash -c 'grep -hoE "mhlo\.[a-z_]+|xla\.[a-z_]+|tf\.[a-z_]+|jax\.[a-z_]+|torch\.[a-z_]+|_xla[a-z_]+" /tmp/shlo_dump/irs/shlo_*.mlir | sort -u | head -50'
```

This lists every prefixed attribute name appearing in the dump. The candidate marker names (best-effort starter list from §9 of spec):
- `mhlo.parameter_replication`
- `mhlo.is_same_data_across_replicas`
- `mhlo.layout_mode`
- `_xla_buffer_placement`
- `tf.aliasing_output`
- `tf.entry_function`
- `jax.arg_info`
- `torch.placeholder`

- [ ] **Step 4: Identify per-arg vs module-level placement**

For each candidate marker hit in step 3, check whether it appears on a `%arg`:
```bash
docker exec tt-xla-eval bash -c 'grep -E "%arg[0-9]+ .* mhlo\..* :" /tmp/shlo_dump/irs/shlo_*.mlir | head -20'
```

If per-arg markers exist → Strategy A is viable in Phase 2.
If only module-level arrays exist (e.g., `mhlo.parameter_replication = [false, false, true, true, true]` on the func) → Strategy B.
If neither → Strategy C only (heuristic).

- [ ] **Step 5: Document outcome**

Save findings to Phase 5 reporting:
- Marker name(s) found (or "none")
- Strategy that Phase 2 will use
- Whether torch_xla emits a flat entry func or wrappers helpers via `func.call`

For the wrapper check:
```bash
docker exec tt-xla-eval bash -c 'grep -c "func.call" /tmp/shlo_dump/irs/shlo_*.mlir'
```

Expected: `0` for flat (Strategy C heuristic visible pre-inliner), `> 0` for wrappered (Strategy C falls through to Phase 4 post-inliner).

---

### Task 0.3: Count `to_layout`-pair triples in TTNN IR

**Files:**
- Create: `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/phase0_count_to_layout_triples.py`
- Read: `/tmp/shlo_dump/irs/ttnn_*.mlir`

**Context:** Phase 0.3 measures Phase 3's ceiling. We need to count occurrences of `ttnn.to_layout → allowlist_op → ttnn.to_layout` chains where the second `to_layout` restores the first's input layout. MLIR is SSA-numbered flat — text-matching won't work. Use Python MLIR bindings to walk the def-use graph.

- [ ] **Step 1: Verify MLIR Python bindings available**

Run:
```bash
docker exec tt-xla-eval bash -c \
  'PYTHONPATH=/opt/tt-mlir-toolchain/python_packages/mlir_core python3 -c "from mlir import ir; print(ir.__file__)"'
```

Expected: prints path to `mlir/ir/__init__.py` or similar. If `ImportError`: bindings missing — `find /opt/tt-mlir-toolchain -name "mlir" -type d 2>/dev/null` to locate them and adjust PYTHONPATH.

- [ ] **Step 2: Write the counting script**

Create `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/phase0_count_to_layout_triples.py`:

```python
"""Phase 0.3 — count ttnn.to_layout-pair triples in TTNN MLIR dump.

Usage:
    PYTHONPATH=/opt/tt-mlir-toolchain/python_packages/mlir_core \\
      python3 phase0_count_to_layout_triples.py <ttnn_dump.mlir>

Triple = to_layout(allowlist_op(to_layout(x, L1)), L0)
where L0 is the original layout of x. Reports candidate-pair count.

Phase 3 gate: ≥ 50 means A.2.a is worth implementing.
"""
import sys
from collections import defaultdict


UNARY_ALLOWLIST = {
    "ttnn.relu", "ttnn.gelu", "ttnn.silu", "ttnn.typecast",
}


def main(mlir_path: str) -> None:
    from mlir import ir

    with open(mlir_path) as f:
        src = f.read()

    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(src)

    triples = 0
    seen_outer = set()

    # Walk all ops; for each ttnn.to_layout (outer), check whether its
    # operand is an allowlist op whose operand is another ttnn.to_layout.
    def walk(op: ir.Operation) -> None:
        nonlocal triples
        name = op.name if hasattr(op, "name") else str(op)
        if name != "ttnn.to_layout":
            return
        if len(op.operands) == 0:
            return
        producer = op.operands[0].owner
        if producer is None:
            return
        if isinstance(producer, ir.Operation):
            producer_op = producer
        elif isinstance(producer, ir.OpView):
            producer_op = producer.operation
        else:
            return
        producer_name = producer_op.name if hasattr(producer_op, "name") else ""
        if producer_name not in UNARY_ALLOWLIST:
            return
        if len(producer_op.operands) == 0:
            return
        inner = producer_op.operands[0].owner
        if inner is None:
            return
        inner_op = inner.operation if isinstance(inner, ir.OpView) else inner
        inner_name = inner_op.name if hasattr(inner_op, "name") else ""
        if inner_name != "ttnn.to_layout":
            return
        triples += 1

    # Iterate all ops in the module body.
    for func_op in module.body:
        for region in func_op.regions:
            for block in region:
                for op in block:
                    if isinstance(op, ir.OpView):
                        walk(op.operation)
                    else:
                        walk(op)

    print(f"ttnn.to_layout → unary_allowlist_op → ttnn.to_layout triples: {triples}")
    print(f"Phase 3 gate (≥ 50): {'PASS' if triples >= 50 else 'SKIP'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
```

- [ ] **Step 3: Run the counting script**

Run:
```bash
docker exec tt-xla-eval bash -c \
  'PYTHONPATH=/opt/tt-mlir-toolchain/python_packages/mlir_core python3 \
   /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/phase0_count_to_layout_triples.py \
   /tmp/shlo_dump/irs/ttnn_*.mlir'
```

Expected output:
```
ttnn.to_layout → unary_allowlist_op → ttnn.to_layout triples: <N>
Phase 3 gate (≥ 50): <PASS|SKIP>
```

- [ ] **Step 4: Record outcome for Phase 3 gate**

If PASS → Phase 3 is in scope. Save count.
If SKIP → Phase 3 is de-scoped per spec. Document count.

- [ ] **Step 5: Commit the script**

```bash
cd /home/mhnie/sglang
git add python/sglang/srt/hardware_backend/tenstorrent/test/phase0_count_to_layout_triples.py
git commit -m "phase0(tt-xla): count to_layout-pair triples via MLIR bindings

Phase 3 gate: ≥ 50 cancellable triples per decode → A.2.a in scope.
Uses MLIR Python bindings with allow_unregistered_dialects=True
since TTNN dialect isn't auto-registered.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# Phase 1 — Repoint tt-xla third_party at the fork

Phase 1's outcome: rebuilt `pjrt-plugin-tt` links against `/home/mhnie/tt-mlir-sglang/` (fork HEAD `2fc1d119e`...), not canonical `f3ddbfb6`. The load-bearing verification is the SHA-diff against `/tmp/canonical-lib-shas.txt` from Task 0.0.

### Task 1.1: Create patches directory

**Files:**
- Create: `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/`

- [ ] **Step 1: Make the directory**

```bash
mkdir -p /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/
ls -la /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/
```

Expected: directory exists, empty.

---

### Task 1.2: Edit tt-xla CMakeLists.txt to use the fork

**Files:**
- Modify: `/home/mhnie/tt-xla/third_party/CMakeLists.txt` (lines 35, 47–85)
- Create: `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-source-dir-fork.patch`

- [ ] **Step 1: Capture pre-edit copy for diff**

```bash
cp /home/mhnie/tt-xla/third_party/CMakeLists.txt /tmp/CMakeLists.txt.before
```

- [ ] **Step 2: Apply the three edits to third_party/CMakeLists.txt**

Use Edit tool to make these three changes:

Change A — `TTMLIR_BUILD_DIR` redirection (line 35):

Find:
```
    set(TTMLIR_BUILD_DIR "${TTMLIR_SOURCE_DIR}/src/tt-mlir/build")
```

Replace with:
```
    # Redirected to fork's build dir per docs/superpowers/specs/2026-05-17-tt-xla-tilize-attack-design.md Phase 1.
    set(TTMLIR_BUILD_DIR "/home/mhnie/tt-mlir-sglang/build")
```

Change B — neuter `ExternalProject_Add(tt-mlir ...)` (lines 47–85):

Find this block (the entire `ExternalProject_Add(tt-mlir ...)` call from line 47 through line 85):
```
    ExternalProject_Add(
        tt-mlir
        PREFIX ${TTPJRT_SOURCE_DIR}/third_party/tt-mlir
        ...
        GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
        GIT_TAG ${TT_MLIR_VERSION}
        GIT_PROGRESS ON
    )
```

Replace by adding `SOURCE_DIR`, removing `GIT_*` lines, and neutering DOWNLOAD/CONFIGURE/BUILD:

```
    ExternalProject_Add(
        tt-mlir
        PREFIX ${TTPJRT_SOURCE_DIR}/third_party/tt-mlir
        SOURCE_DIR /home/mhnie/tt-mlir-sglang

        DOWNLOAD_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""

        # Installing the python dependencies before the build (kept as no-op
        # safety; the build_and_install.sh script handles fork build itself)
        PATCH_COMMAND mkdir -p ${TTMLIR_BUILD_DIR}
        COMMAND TTPJRT_SOURCE_DIR=${TTPJRT_SOURCE_DIR} bash ${TTPJRT_SOURCE_DIR}/venv/install_ttmlir_requirements.sh

        CMAKE_GENERATOR Ninja
        BINARY_DIR ${TTMLIR_BUILD_DIR}

        # CRITICAL: --prefix override is required because the fork's build was
        # configured WITHOUT -DCMAKE_INSTALL_PREFIX=${TTMLIR_INSTALL_PREFIX},
        # so its CMakeCache caches /usr/local. Without --prefix, install lands
        # in /usr/local/lib and tt-xla links the stale canonical artifacts.
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --prefix ${TTMLIR_INSTALL_PREFIX} --component SharedLib
        COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --prefix ${TTMLIR_INSTALL_PREFIX} --component DistributedRuntime

        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=${TTMLIR_BUILD_TYPE}
          -DCMAKE_C_COMPILER=clang
          -DCMAKE_CXX_COMPILER=clang++
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
          -DTT_RUNTIME_ENABLE_TTNN=ON
          -DTTMLIR_ENABLE_STABLEHLO=ON
          -DTTMLIR_ENABLE_RUNTIME=ON
          -DTT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          -DTTMLIR_ENABLE_BINDINGS_PYTHON=${TTMLIR_ENABLE_BINDINGS_PYTHON}
          -DCMAKE_INSTALL_PREFIX=${TTMLIR_INSTALL_PREFIX}
          -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
          -DTT_RUNTIME_ENABLE_PERF_TRACE=${TTMLIR_ENABLE_PERF_TRACE}
          -DTTMLIR_ENABLE_OPMODEL=ON
          -DTTMLIR_ENABLE_EXPLORER=${TTXLA_ENABLE_EXPLORER}
          -DTTMLIR_ENABLE_TESTS=OFF
          -DTTMLIR_ENABLE_TOOLS=${TTXLA_ENABLE_TOOLS}
          -DTTMLIR_ENABLE_DOCS=OFF
          -DTT_USE_SYSTEM_SFPI=${TT_USE_SYSTEM_SFPI}
    )
```

(Note: the `CMAKE_ARGS` block above is kept for reference but inert because `CONFIGURE_COMMAND ""` skips configure. Leaving it documents intent. The original `BUILD_COMMAND env ${WITH_METAL_RUNTIME_ROOT_SET} ${CMAKE_COMMAND} --build <BINARY_DIR>` line is replaced by `BUILD_COMMAND ""`.)

Change C — lines 36–39 (the `TT_METAL_RUNTIME_ROOT` guard) — **leave unchanged**. Line 37's `if(NOT DEFINED ENV{TT_METAL_RUNTIME_ROOT})` guard skips the hardcode when the env var is set (Task 1.3 handles that export).

- [ ] **Step 3: Save patch file**

```bash
cd /home/mhnie/tt-xla
diff -u /tmp/CMakeLists.txt.before third_party/CMakeLists.txt \
  > /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-source-dir-fork.patch
cat /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-source-dir-fork.patch | head -20
```

Expected: unified diff showing the three changes.

---

### Task 1.3: Edit build_and_install.sh

**Files:**
- Modify: `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh`
- Create: `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-mlir-sglang-build-script-cleanup.patch`

- [ ] **Step 1: Capture pre-edit copy**

```bash
cp /home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh /tmp/build_and_install.sh.before
```

- [ ] **Step 2: Remove dead `-DTTMLIR_SOURCE_DIR_OVERRIDE` flag**

Use Edit tool. In `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh`:

Find:
```
    cmake -G Ninja -B "$TTXLA_DIR/build-local" -S "$TTXLA_DIR" \
        -DTTMLIR_SOURCE_DIR_OVERRIDE="$TTMLIR_DIR" \
        -DCMAKE_BUILD_TYPE=Release
```

Replace with:
```
    cmake -G Ninja -B "$TTXLA_DIR/build-local" -S "$TTXLA_DIR" \
        -DCMAKE_BUILD_TYPE=Release
```

- [ ] **Step 3: Insert TT_METAL_RUNTIME_ROOT export before step [3/4]**

Use Edit tool. Find:
```
echo "[3/4] Configuring + building tt-xla against local tt-mlir-sglang..."
```

Replace with:
```
# TT_METAL_RUNTIME_ROOT must point at the fork's tt-metal source tree.
# The fork's own ExternalProject for tt-metal lands it at this path after the
# fork's configure step [1/4] runs. Hardcoded (not via `find`) because on a
# fresh fork checkout the path doesn't exist until configure populates it;
# `find` would return empty.
export TT_METAL_RUNTIME_ROOT="/home/mhnie/tt-mlir-sglang/third_party/tt-metal/src/tt-metal"

echo "[3/4] Configuring + building tt-xla against local tt-mlir-sglang..."
```

- [ ] **Step 4: Save patch file**

```bash
diff -u /tmp/build_and_install.sh.before /home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh \
  > /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-mlir-sglang-build-script-cleanup.patch
cat /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-mlir-sglang-build-script-cleanup.patch
```

Expected: unified diff with the two changes.

---

### Task 1.4: Clean stale build artifacts

**Files:**
- Delete (host side): `/home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir-stamp`, `/home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir`, `/home/mhnie/tt-xla/build-local`, `/home/mhnie/tt-xla/build`

- [ ] **Step 1: Confirm what exists before deleting**

```bash
ls -ld /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir-stamp \
       /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir \
       /home/mhnie/tt-xla/build-local \
       /home/mhnie/tt-xla/build 2>&1
```

Expected: at least the canonical `tt-mlir` source dir exists (ExternalProject's earlier clone). Some others may not.

- [ ] **Step 2: Delete (no trailing slashes — `rm -rf` on a symlink without trailing slash removes only the link)**

```bash
rm -rf /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir-stamp \
       /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir \
       /home/mhnie/tt-xla/build-local \
       /home/mhnie/tt-xla/build
```

Verify:
```bash
ls -ld /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir-stamp \
       /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir \
       /home/mhnie/tt-xla/build-local \
       /home/mhnie/tt-xla/build 2>&1
```

Expected: all four "No such file or directory."

---

### Task 1.5: Run fork + plugin build

**Files:**
- Runs: `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh`

- [ ] **Step 1: Run the build script**

```bash
cd /home/mhnie/tt-mlir-sglang
bash scripts/build_and_install.sh 2>&1 | tee /tmp/phase1_build.log
```

This runs steps `[1-2]/4` (configure + build fork — first time, ~30–60 min for LLVM-aware tt-mlir build), step `[3]/4` (configure + build tt-xla against fork, ~10–15 min), and step `[4]/4` (docker-cp the plugin into `tt-xla-eval`).

- [ ] **Step 2: Verify build success**

```bash
tail -10 /tmp/phase1_build.log
ls -la /home/mhnie/tt-mlir-sglang/build/build.ninja 2>&1
ls -la /home/mhnie/tt-xla/build-local/build.ninja 2>&1
```

Expected: no error lines; both build directories exist; `Done.` appears in the script tail.

If build fails: read the failure, fix the cause, re-run. Do NOT skip to Task 1.6 against a broken build.

---

### Task 1.6: Hard verification — SHA-diff and binary symbol check

**Files:**
- Read: `/tmp/canonical-lib-shas.txt`, `/tmp/post-phase1-lib-shas.txt`

- [ ] **Step 1: Snapshot post-Phase-1 SHAs**

```bash
docker exec tt-xla-eval bash -c \
  'sha256sum /tt-xla/third_party/tt-mlir/install/lib/*.so' \
  > /tmp/post-phase1-lib-shas.txt
```

- [ ] **Step 2: Diff against canonical baseline**

```bash
diff /tmp/canonical-lib-shas.txt /tmp/post-phase1-lib-shas.txt
```

Expected: at least one line differs (typically `libTTMLIRCompiler.so`). If the diff is **empty** (zero changes), the fork's source was NOT linked — STOP. Investigate the chain: did INSTALL_COMMAND fire? Did `--prefix` resolve? Re-run from Task 1.4.

- [ ] **Step 3: Confirm libTTMLIRCompiler.so specifically differs**

```bash
diff <(grep libTTMLIRCompiler.so /tmp/canonical-lib-shas.txt) \
     <(grep libTTMLIRCompiler.so /tmp/post-phase1-lib-shas.txt)
```

Expected: 1c1 diff showing different SHAs. If they match, the fork's StableHLOToTTIR patch (B.2 v3 at `2fc1d119e`) wasn't compiled in.

- [ ] **Step 4: Behavioral cross-check via smoke bench**

Run a Qwen3-8B BFP8 server bench (NO Phase 2 changes yet — `argumentTypeMap` is unset, so const-eval doesn't fire):

```bash
docker exec -d tt-xla-eval bash -c 'cd /sglang && BYPASS_PREWARM=1 SGLANG_TT_CACHE_MODE=index_copy \
  python3 -u python/sglang/srt/hardware_backend/tenstorrent/test/bench_3run_server_alive.py \
  --models Qwen3-8B --backend tt_xla --input-len 1024 --output-len 1024 \
  --out-tag phase1_smoke > /tmp/phase1_smoke.log 2>&1'
```

Wait until done (`pgrep -af bench_3run_server` returns empty).

- [ ] **Step 5: Verify TPOT ≈ 132 ms ± 5%**

```bash
jq '.results[0].run2_warm_new_prompt.tpot_ms_mean' \
   /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase1_smoke.json
```

Expected: 125–139 ms (132 ± 5%). If TPOT regresses substantially, one of the 5 fork patches is causing it — `git bisect` from fork HEAD (`2fc1d119e`) downward.

---

### Task 1.7: Commit Phase 1

- [ ] **Step 1: Stage Phase 1 patches + smoke fixture**

```bash
cd /home/mhnie/sglang
git add python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-source-dir-fork.patch
git add python/sglang/srt/hardware_backend/tenstorrent/patches/tt-mlir-sglang-build-script-cleanup.patch
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase1_smoke.json
```

- [ ] **Step 2: Commit**

```bash
git commit -m "phase1(tt-xla): repoint third_party at tt-mlir-sglang fork

Edit tt-xla/third_party/CMakeLists.txt to use SOURCE_DIR=fork,
neutering CONFIGURE/BUILD commands (the build_and_install.sh script
handles those). Add explicit --prefix \${TTMLIR_INSTALL_PREFIX} to
INSTALL_COMMAND to override the fork's default /usr/local cache.

Edit build_and_install.sh to drop the dead -DTTMLIR_SOURCE_DIR_OVERRIDE
flag and export TT_METAL_RUNTIME_ROOT explicitly.

SHA-diff verification: libTTMLIRCompiler.so changed (B.2 v3 fork
patch landed). Smoke bench: 132 ms TPOT, no regression.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# Phase 2 — A.1.a: set argumentTypeMap in module_builder.cc

Phase 2 wires the existing `tt-populate-argument-types` pass to receive a per-function `Input`/`Parameter`/`Constant` map derived from the incoming HLO. The downstream `ConstEvalHoistTransform` (already in the pipeline at `TTNNPipelines.cpp:309/372/386`) then hoists weight-touching subgraphs.

### Task 2.1: Implement `classifyArgs` helper — header + skeleton

**Files:**
- Modify: `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc`
- Modify: `/home/mhnie/tt-xla/pjrt_implementation/inc/api/module_builder/module_builder.h` (if signatures need exposure)

**Context:** `classifyArgs` returns a `TTArgumentTypeMap` populated using Strategy A → B → C in priority order. Strategy A reads per-arg torch_xla markers (name discovered in Phase 0.2); B reads module-level cluster markers; C runs the heuristic.

- [ ] **Step 1: Add `#include` for ArgumentType and FuncOp**

In `module_builder.cc`, near the existing tt-mlir includes, add:
```cpp
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"  // ArgumentType, ArgumentTypeAttr
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"  // TTArgumentTypeMap
#include "mlir/Dialect/Func/IR/FuncOps.h"  // func::FuncOp
```

- [ ] **Step 2: Declare `classifyArgs` skeleton near the top of the file**

Add (as a free function or static method on ModuleBuilder, matching the file's existing style):
```cpp
namespace {

// Phase 2 (A.1.a): classify each block arg of the unique non-private
// FuncOp into Input / Parameter / Constant using Strategy A → B → C.
// Returns a TTArgumentTypeMap keyed by function symbol name.
//
// Strategy A: read per-arg torch_xla marker (e.g., mhlo.parameter_replication).
// Strategy B: read module-level cluster marker array.
// Strategy C: heuristic over use chains (Rule 1–5 per spec).
//
// If the function already has ttcore.argument_type attrs (set by
// frontend_passes::annotateArgumentAttributes for JAX-origin modules),
// SKIP — don't overwrite existing annotations.
mlir::tt::ttcore::TTArgumentTypeMap
classifyArgs(mlir::ModuleOp module);

}  // namespace
```

---

### Task 2.2: Implement `classifyArgs` — find target function

- [ ] **Step 1: Implement function-finding logic**

Below the declaration in Task 2.1, add:
```cpp
namespace {

// Find the unique non-private FuncOp in the module. Returns nullptr if 0 or >1.
mlir::func::FuncOp findUniqueEntryFunc(mlir::ModuleOp module) {
  mlir::func::FuncOp result;
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    if (func.getVisibility() == mlir::SymbolTable::Visibility::Private) {
      continue;
    }
    if (result) {
      // > 1 non-private func — bail.
      LOG_F(WARNING, "[TT-XLA classifyArgs] module has multiple non-private FuncOps; "
                     "skipping auto-classification");
      return {};
    }
    result = func;
  }
  return result;
}

// Returns true if any block arg of `func` already has ttcore.argument_type set.
bool funcHasExistingArgTypeAttrs(mlir::func::FuncOp func) {
  for (uint32_t i = 0; i < func.getNumArguments(); ++i) {
    if (func.getArgAttrOfType<mlir::tt::ttcore::ArgumentTypeAttr>(
            i, mlir::tt::ttcore::ArgumentTypeAttr::name)) {
      return true;
    }
  }
  return false;
}

}  // namespace
```

---

### Task 2.3: Implement Strategy A (per-arg marker)

- [ ] **Step 1: Write Strategy A**

Add to the anonymous namespace:
```cpp
namespace {

// Strategy A: read per-arg torch_xla marker. The marker name is set at
// build time from Phase 0.2 findings; if Phase 0.2 found e.g.
// "mhlo.parameter_replication", set kPerArgMarkerName below to that string.
//
// If the marker presents as a boolean / true-false attr, true → Parameter.
// If marker is a string attr, "parameter"/"weight" → Parameter.
//
// Returns true iff Strategy A populated `arg_types` for ALL args.
constexpr llvm::StringRef kPerArgMarkerName = "mhlo.parameter_replication";

bool applyStrategyA(mlir::func::FuncOp func,
                    llvm::SmallVector<mlir::tt::ttcore::ArgumentType> &arg_types) {
  bool found_any = false;
  for (uint32_t i = 0; i < func.getNumArguments(); ++i) {
    auto attr = func.getArgAttr(i, kPerArgMarkerName);
    if (!attr) {
      // missing marker on this arg → Strategy A cannot fully classify
      return false;
    }
    found_any = true;
    // Best-effort interpretation: BoolAttr true OR StringAttr matching
    // "parameter"/"weight" → Parameter; everything else → Input.
    bool is_parameter = false;
    if (auto bool_attr = mlir::dyn_cast<mlir::BoolAttr>(attr)) {
      is_parameter = bool_attr.getValue();
    } else if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(attr)) {
      auto v = str_attr.getValue();
      is_parameter = (v == "parameter" || v == "weight");
    }
    arg_types.push_back(is_parameter
                            ? mlir::tt::ttcore::ArgumentType::Parameter
                            : mlir::tt::ttcore::ArgumentType::Input);
  }
  return found_any;
}

}  // namespace
```

Note: the executor should update `kPerArgMarkerName` to whatever Phase 0.2 actually discovered. Default `mhlo.parameter_replication` is a reasonable initial guess.

---

### Task 2.4: Implement Strategy B (module-level cluster marker)

- [ ] **Step 1: Write Strategy B**

```cpp
namespace {

// Strategy B: read a module-level array attr like
// `func.func @main attributes {mhlo.parameter_replication = [false, true, true]}`.
// The array length must equal func.getNumArguments().
constexpr llvm::StringRef kClusterMarkerName = "mhlo.parameter_replication";

bool applyStrategyB(mlir::func::FuncOp func,
                    llvm::SmallVector<mlir::tt::ttcore::ArgumentType> &arg_types) {
  auto attr = func->getAttr(kClusterMarkerName);
  if (!attr) {
    return false;
  }
  auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr);
  if (!array) {
    return false;
  }
  if (array.size() != func.getNumArguments()) {
    LOG_F(WARNING, "[TT-XLA classifyArgs] cluster marker size mismatch: %u args, "
                   "%zu marker entries — falling through to Strategy C",
                   func.getNumArguments(), array.size());
    return false;
  }
  for (auto entry : array) {
    bool is_parameter = false;
    if (auto bool_attr = mlir::dyn_cast<mlir::BoolAttr>(entry)) {
      is_parameter = bool_attr.getValue();
    }
    arg_types.push_back(is_parameter
                            ? mlir::tt::ttcore::ArgumentType::Parameter
                            : mlir::tt::ttcore::ArgumentType::Input);
  }
  return true;
}

}  // namespace
```

---

### Task 2.5: Implement Strategy C — Rule 1 (shape gate)

**Context:** Strategy C is the heuristic, used when neither marker is present. Rule 1: shape gate — every dim ≥ 16, OR rank == 1 with that dim ≥ 64.

- [ ] **Step 1: Write Rule 1 helper**

```cpp
namespace {

// Rule 1 (shape gate): tensor passes if every dim ≥ 16 OR rank==1 with dim ≥ 64.
// Excludes attention_mask shaped [1, 1, seq, seq] which has size-1 dims.
bool ruleOneShapeGate(mlir::BlockArgument arg) {
  auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
  if (!tensor_type) {
    return false;
  }
  auto shape = tensor_type.getShape();
  if (shape.empty()) {
    return false;
  }
  if (shape.size() == 1) {
    return shape[0] >= 64;
  }
  for (int64_t dim : shape) {
    if (dim < 16) {
      return false;
    }
  }
  return true;
}

}  // namespace
```

---

### Task 2.6: Implement Strategy C — Rule 2 (use-pattern allowlist) + helpers

**Context:** Rule 2: every use of the arg is one of {dot_general, convolution, gather, multiply/add with non-block-arg other operand, dynamic_slice/slice} (or transitively through ≤ 3 layout-only ops).

- [ ] **Step 1: Write layout-only-op detector**

```cpp
namespace {

bool isLayoutOnlyOp(mlir::Operation *op) {
  if (!op) return false;
  llvm::StringRef name = op->getName().getStringRef();
  return name == "stablehlo.transpose" || name == "stablehlo.reshape" ||
         name == "stablehlo.broadcast_in_dim" || name == "stablehlo.convert" ||
         name == "stablehlo.slice";
}

}  // namespace
```

- [ ] **Step 2: Write terminal-use detector**

```cpp
namespace {

bool isAllowlistTerminalUse(mlir::Operation *op, mlir::Value operand) {
  if (!op) return false;
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "stablehlo.dot_general") return true;
  if (name == "stablehlo.convolution") return true;
  if (name == "stablehlo.gather") return true;
  if (name == "stablehlo.dynamic_slice") return true;
  if (name == "stablehlo.slice") return true;
  if (name == "stablehlo.multiply" || name == "stablehlo.add") {
    // The OTHER operand must NOT be another block arg.
    for (mlir::Value other : op->getOperands()) {
      if (other == operand) continue;
      if (mlir::isa<mlir::BlockArgument>(other)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

}  // namespace
```

- [ ] **Step 3: Write Rule 2 walker**

```cpp
namespace {

// Walk forward from `start` through up-to-`max_layout_hops` layout-only ops.
// Returns true iff every leaf use is in the allowlist.
bool ruleTwoUsePattern(mlir::BlockArgument arg, int max_layout_hops = 3) {
  llvm::SmallVector<std::pair<mlir::Value, int>> worklist;
  worklist.push_back({arg, 0});
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    auto [v, depth] = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;
    if (v.use_empty()) {
      return false;  // dangling arg — be conservative
    }
    for (mlir::OpOperand &use : v.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (isLayoutOnlyOp(user)) {
        if (depth >= max_layout_hops) {
          return false;  // ran out of hops
        }
        for (mlir::Value result : user->getResults()) {
          worklist.push_back({result, depth + 1});
        }
        continue;
      }
      if (!isAllowlistTerminalUse(user, v)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace
```

---

### Task 2.7: Implement Strategy C — Rule 3, 4 (no-mutate, not-batch-dim)

- [ ] **Step 1: Write Rule 3 (no mutating uses)**

```cpp
namespace {

// Rule 3: arg is never operand of a mutating op (no scatter to itself).
bool ruleThreeNoMutate(mlir::BlockArgument arg) {
  for (mlir::OpOperand &use : arg.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (user->getName().getStringRef() == "stablehlo.scatter") {
      // scatter writes back to its first operand (the destination).
      if (user->getNumOperands() > 0 && user->getOperand(0) == arg) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace
```

- [ ] **Step 2: Write Rule 4 (not dynamic-input position)**

```cpp
namespace {

// Rule 4: arg's first dim is not used as a batch dim by stablehlo.dynamic_slice
// (excludes input_ids / position_ids / attention_mask / cache_pos / kv-cache args).
bool ruleFourNotBatchInput(mlir::BlockArgument arg) {
  for (mlir::OpOperand &use : arg.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (user->getName().getStringRef() == "stablehlo.dynamic_slice") {
      // If this arg is the OPERAND being sliced (op->getOperand(0)), and the
      // slice indices come from another block arg, this is a "batch-dim" use.
      if (user->getNumOperands() > 0 && user->getOperand(0) == arg) {
        for (uint32_t i = 1; i < user->getNumOperands(); ++i) {
          if (mlir::isa<mlir::BlockArgument>(user->getOperand(i))) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

}  // namespace
```

---

### Task 2.8: Implement Strategy C — Rule 5 (softmax-path exclusion)

**Context:** Rule 5: BFS forward from arg up to depth 8 (counting only non-layout-only ops); reject if `stablehlo.exponential` reached AND path includes add/sub/multiply consuming arg-lineage.

- [ ] **Step 1: Write Rule 5 BFS**

```cpp
namespace {

bool isLayoutOnlyForRule5(mlir::Operation *op) {
  return isLayoutOnlyOp(op);  // same allowlist as Rule 2
}

// Rule 5: stays Input if forward dataflow reaches stablehlo.exponential
// via at least one stablehlo.add/sub/multiply consuming arg-lineage.
// Depth counts only non-layout-only ops. Worklist capped at 256 ops.
bool ruleFiveSoftmaxPathExcluded(mlir::BlockArgument arg) {
  struct Item { mlir::Value v; int depth; bool seen_add_mul; };
  llvm::SmallVector<Item> worklist;
  worklist.push_back({arg, 0, false});
  llvm::DenseSet<std::pair<mlir::Value, bool>> visited;
  constexpr int kMaxDepth = 8;
  constexpr int kMaxWorklist = 256;
  int popped = 0;
  while (!worklist.empty()) {
    if (++popped > kMaxWorklist) {
      return false;  // fail closed — reject
    }
    Item item = worklist.pop_back_val();
    auto key = std::make_pair(item.v, item.seen_add_mul);
    if (!visited.insert(key).second) continue;
    for (mlir::OpOperand &use : item.v.getUses()) {
      mlir::Operation *user = use.getOwner();
      llvm::StringRef name = user->getName().getStringRef();
      if (name == "stablehlo.exponential" && item.seen_add_mul) {
        return false;  // softmax path detected — Rule 5 fires
      }
      bool layout_only = isLayoutOnlyForRule5(user);
      int new_depth = layout_only ? item.depth : item.depth + 1;
      if (new_depth > kMaxDepth) continue;
      bool is_add_mul = (name == "stablehlo.add" ||
                        name == "stablehlo.subtract" ||
                        name == "stablehlo.multiply");
      bool new_seen = item.seen_add_mul || is_add_mul;
      for (mlir::Value result : user->getResults()) {
        worklist.push_back({result, new_depth, new_seen});
      }
    }
  }
  return true;  // Rule 5 not triggered → arg passes
}

}  // namespace
```

---

### Task 2.9: Implement Strategy C — combine rules

- [ ] **Step 1: Write the combined classifier**

```cpp
namespace {

mlir::tt::ttcore::ArgumentType classifyOneArgStrategyC(mlir::BlockArgument arg) {
  if (!ruleOneShapeGate(arg)) return mlir::tt::ttcore::ArgumentType::Input;
  if (!ruleTwoUsePattern(arg)) return mlir::tt::ttcore::ArgumentType::Input;
  if (!ruleThreeNoMutate(arg)) return mlir::tt::ttcore::ArgumentType::Input;
  if (!ruleFourNotBatchInput(arg)) return mlir::tt::ttcore::ArgumentType::Input;
  if (!ruleFiveSoftmaxPathExcluded(arg)) return mlir::tt::ttcore::ArgumentType::Input;
  return mlir::tt::ttcore::ArgumentType::Parameter;
}

void applyStrategyC(mlir::func::FuncOp func,
                    llvm::SmallVector<mlir::tt::ttcore::ArgumentType> &arg_types) {
  for (auto arg : func.getArguments()) {
    arg_types.push_back(classifyOneArgStrategyC(arg));
  }
}

}  // namespace
```

---

### Task 2.10: Implement `classifyArgs` body

- [ ] **Step 1: Compose the strategies**

```cpp
namespace {

mlir::tt::ttcore::TTArgumentTypeMap
classifyArgs(mlir::ModuleOp module) {
  mlir::tt::ttcore::TTArgumentTypeMap result;
  mlir::func::FuncOp func = findUniqueEntryFunc(module);
  if (!func) {
    return result;  // empty map — pass becomes no-op (warning logged)
  }

  // Short-circuit: don't clobber JAX-emitted annotations.
  if (funcHasExistingArgTypeAttrs(func)) {
    LOG_F(INFO, "[TT-XLA classifyArgs] func '%s' already has ttcore.argument_type — "
                "skipping (likely JAX-origin)", func.getName().str().c_str());
    return result;
  }

  llvm::SmallVector<mlir::tt::ttcore::ArgumentType> arg_types;

  // Strategy A — per-arg marker
  if (applyStrategyA(func, arg_types)) {
    LOG_F(INFO, "[TT-XLA classifyArgs] strategy=A func='%s' K_inputs=%u K_params=%u",
          func.getName().str().c_str(),
          (uint32_t)std::count(arg_types.begin(), arg_types.end(),
                                mlir::tt::ttcore::ArgumentType::Input),
          (uint32_t)std::count(arg_types.begin(), arg_types.end(),
                                mlir::tt::ttcore::ArgumentType::Parameter));
    result[func.getName()] = arg_types;
    return result;
  }

  // Strategy B — module-level cluster marker
  arg_types.clear();
  if (applyStrategyB(func, arg_types)) {
    LOG_F(INFO, "[TT-XLA classifyArgs] strategy=B func='%s' K_inputs=%u K_params=%u",
          func.getName().str().c_str(),
          (uint32_t)std::count(arg_types.begin(), arg_types.end(),
                                mlir::tt::ttcore::ArgumentType::Input),
          (uint32_t)std::count(arg_types.begin(), arg_types.end(),
                                mlir::tt::ttcore::ArgumentType::Parameter));
    result[func.getName()] = arg_types;
    return result;
  }

  // Strategy C — heuristic
  arg_types.clear();
  applyStrategyC(func, arg_types);
  LOG_F(INFO, "[TT-XLA classifyArgs] strategy=C func='%s' K_inputs=%u K_params=%u",
        func.getName().str().c_str(),
        (uint32_t)std::count(arg_types.begin(), arg_types.end(),
                              mlir::tt::ttcore::ArgumentType::Input),
        (uint32_t)std::count(arg_types.begin(), arg_types.end(),
                              mlir::tt::ttcore::ArgumentType::Parameter));
  result[func.getName()] = arg_types;
  return result;
}

}  // namespace
```

---

### Task 2.11: Wire `classifyArgs` into module_builder.cc

- [ ] **Step 1: Locate insertion point for StableHLO PM**

Read `module_builder.cc` around line 832. Expected: line 832 is `stablehlo_pipeline_options.resultPresharded = result_presharded;`. Line 833 is the `createStableHLOPipeline(...)` call that consumes the options.

- [ ] **Step 2: Insert StableHLO map assignment**

Use Edit. Find:
```cpp
  stablehlo_pipeline_options.resultPresharded = result_presharded;
  createStableHLOPipeline(stablehlo_pipeline_pm, stablehlo_pipeline_options);
```

Replace with:
```cpp
  stablehlo_pipeline_options.resultPresharded = result_presharded;
  // Phase 2 A.1.a: populate argumentTypeMap so tt-populate-argument-types
  // sets ttcore.argument_type on block args; downstream ConstEvalHoistTransform
  // then fires for weight-touching subgraphs. See docs/superpowers/specs/
  // 2026-05-17-tt-xla-tilize-attack-design.md §5 Phase 2.
  stablehlo_pipeline_options.argumentTypeMap = classifyArgs(*mlir_module);
  createStableHLOPipeline(stablehlo_pipeline_pm, stablehlo_pipeline_options);
```

(Use `*mlir_module` if the variable is an `OwningOpRef`; adjust to the actual reference syntax visible in the file.)

- [ ] **Step 3: Locate insertion point for TTNN PM**

Read `module_builder.cc` around line 985. Expected: `TTIRToTTNNCommonPipelineOptions options;` declaration. The subsequent `createTTIRToTTNNPipeline(...)` call consumes `options`.

- [ ] **Step 4: Insert TTNN map assignment**

Use Edit. Find the line after `TTIRToTTNNCommonPipelineOptions options;`:
```cpp
  TTIRToTTNNCommonPipelineOptions options;
```

Insert immediately after (still before `createTTIRToTTNNPipeline`):
```cpp
  TTIRToTTNNCommonPipelineOptions options;
  // Phase 2 A.1.a: same argumentTypeMap on the TTNN PM (the StableHLO PM
  // already saw it above; TTNN PM gets it again because tt-populate-argument-
  // types runs inside both pipelines).
  options.argumentTypeMap = classifyArgs(*mlir_module);
```

---

### Task 2.12: Save patch + build

**Files:**
- Create: `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch`

- [ ] **Step 1: Save patch**

```bash
cd /home/mhnie/tt-xla
git diff pjrt_implementation/src/api/module_builder/module_builder.cc \
  > /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch
wc -l /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch
```

Expected: ≥ 100 lines (helpers + assignments + includes).

- [ ] **Step 2: Rebuild plugin**

```bash
cd /home/mhnie/tt-mlir-sglang
bash scripts/build_and_install.sh 2>&1 | tee /tmp/phase2_build.log
tail -5 /tmp/phase2_build.log
```

Expected: `Done.` Build succeeds against the updated `module_builder.cc`.

If compile fails: fix the C++ errors. Common issues:
- Missing include for `func::FuncOp` → add `mlir/Dialect/Func/IR/FuncOps.h`.
- `TTArgumentTypeMap` not in scope → check namespace `mlir::tt::ttcore`.
- `OwningOpRef<ModuleOp>` deref → use `mlir_module.get()` instead of `*mlir_module` if needed.

Do NOT skip the build to "see if it works in Phase 2 verification."

---

### Task 2.13: Verify Phase 2 — IR dump shows argument_type=parameter

- [ ] **Step 1: Re-run one compile with export_path**

Use the same probe-with-export_path mechanism from Task 0.2 Step 1.

- [ ] **Step 2: Grep shlo_compiler dump for argument_type**

```bash
docker exec tt-xla-eval bash -c \
  'grep -E "ttcore.argument_type|argument_type = #ttcore" /tmp/shlo_dump/irs/shlo_compiler_*.mlir | head -10'
```

Expected: lines like `%arg5: tensor<...> {ttcore.argument_type = #ttcore<argument_type parameter>}` for weight args.

If 0 matches: classifier returned empty map OR pass didn't fire. Check `tt-xla-eval` logs for the INFO line:
```bash
docker exec tt-xla-eval bash -c 'grep -E "TT-XLA classifyArgs" /tmp/phase02.log | tail -5'
```

Expected: at least one line showing `strategy=C K_inputs=<N> K_params=<M>`.

- [ ] **Step 3: Grep ttnn dump for consteval wrapper**

```bash
docker exec tt-xla-eval bash -c \
  'grep -E "consteval_|@consteval" /tmp/shlo_dump/irs/ttnn_*.mlir | head -5'
```

Expected: at least one `func @consteval_<entry_fn_name>` wrapper.

If 0 matches: `ConstEvalHoistTransform` didn't fire. Check the `ttir` dump for `argument_type` first — if present there but no consteval wrapper in ttnn, the issue is downstream pipeline ordering. Per spec, this is a real bug in tt-mlir; file upstream.

- [ ] **Step 4: Confirm BFP8 cast inside consteval wrapper**

```bash
docker exec tt-xla-eval bash -c \
  'grep -B2 -A2 "bfp8\|typecast" /tmp/shlo_dump/irs/ttnn_*.mlir | grep -E "consteval|func @" | head -10'
```

Expected: BFP8 cast ops are inside `consteval_*` wrapper, NOT in the main entry function. Per spec, the line-372 ConstEvalHoist re-invocation handles this; if not, file a tt-mlir bug.

Set `enable_const_eval_on_cpu=False` explicitly (via `set_custom_compile_options`) for this verification to prevent CPU hoisting:
```python
torch_xla.set_custom_compile_options({"enable_const_eval_on_cpu": False})
```

---

### Task 2.14: Bench Phase 2

- [ ] **Step 1: Run server bench with A.1.a active**

```bash
docker exec -d tt-xla-eval bash -c 'cd /sglang && BYPASS_PREWARM=1 SGLANG_TT_CACHE_MODE=index_copy \
  python3 -u python/sglang/srt/hardware_backend/tenstorrent/test/bench_3run_server_alive.py \
  --models Qwen3-8B --backend tt_xla --input-len 1024 --output-len 1024 \
  --out-tag phase2_a1a > /tmp/phase2_bench.log 2>&1'
```

Wait for completion.

- [ ] **Step 2: Read TPOT result**

```bash
jq '.results[0].run2_warm_new_prompt.tpot_ms_mean' \
   /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase2_a1a.json
```

Expected: TPOT < 132 ms (baseline). Magnitude depends on Phase 0.1 weight-tilize share.

If TPOT is unchanged: check that the IR dump from Task 2.13 actually shows `argument_type=parameter`. If yes but no TPOT change, ConstEvalHoist might not be doing what we hoped — examine `consteval_` wrapper contents.

- [ ] **Step 3: Negative control bench**

Run again with `enable_const_eval=False` to confirm baseline restoration:
```python
# In the probe/bench's Python startup:
torch_xla.set_custom_compile_options({"enable_const_eval": False})
```

Expected: TPOT returns to ~132 ms.

---

### Task 2.15: Commit Phase 2

- [ ] **Step 1: Stage Phase 2 patch + fixture**

```bash
cd /home/mhnie/sglang
git add python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase2_a1a.json
```

- [ ] **Step 2: Commit**

```bash
git commit -m "phase2(tt-xla): A.1.a — classifyArgs + argumentTypeMap in module_builder.cc

Adds three-strategy classifier (per-arg marker → cluster marker → 5-rule
heuristic) and assigns argumentTypeMap on both StableHLO and TTNN
pipeline options. tt-populate-argument-types then sets ttcore.
argument_type=parameter on weight block args; ConstEvalHoistTransform
hoists those subgraphs.

Verification: shlo_compiler dump shows argument_type=parameter; ttnn
dump shows consteval_ wrapper; bench TPOT dropped.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# Phase 3 — A.2.a: TTNNFoldThroughAgnosticOps (CONDITIONAL)

**Gate:** Phase 0.3 reported ≥ 50 cancellable triples. If < 50, SKIP Phase 3 entirely and proceed to Phase 4. Document the skip in Phase 5.

### Task 3.1: Check Phase 0.3 gate

- [ ] **Step 1: Read Phase 0.3 outcome**

If the Phase 0.3 count was < 50: skip ahead to Task 4.1. Document the skip in Phase 5.

If ≥ 50: proceed to Task 3.2.

---

### Task 3.2: Add pass registration in Passes.td

**Files:**
- Modify: `/home/mhnie/tt-mlir-sglang/include/ttmlir/Dialect/TTNN/Transforms/Passes.td`

- [ ] **Step 1: Add pass declaration**

In `Passes.td`, locate the existing TTNN pass declarations (e.g., `def TTNNDecomposeLayouts`). After the last existing pass declaration, add:

```tablegen
def TTNNFoldThroughAgnosticOps : Pass<"ttnn-fold-through-agnostic-ops", "::mlir::ModuleOp"> {
  let summary = "Fold to_layout(unary_agnostic_op(to_layout(x, L1)), L0) → unary_agnostic_op(x).";
  let description = [{
    Removes redundant layout conversion pairs separated by a single
    layout-agnostic unary op (relu/gelu/silu/typecast). Complements
    foldConsecutiveToLayoutOp (which only handles directly-adjacent
    to_layout ops) by extending the fold across one intermediate op.

    Phase 3 (A.2.a) per docs/superpowers/specs/2026-05-17-tt-xla-tilize-
    attack-design.md.
  }];
  let constructor = "createTTNNFoldThroughAgnosticOps()";
  let dependentDialects = ["::mlir::tt::ttnn::TTNNDialect"];
}
```

- [ ] **Step 2: Declare the constructor in Passes.h**

In `/home/mhnie/tt-mlir-sglang/include/ttmlir/Dialect/TTNN/Transforms/Passes.h`, near other constructor declarations, add:

```cpp
std::unique_ptr<mlir::Pass> createTTNNFoldThroughAgnosticOps();
```

---

### Task 3.3: Implement the fold pass

**Files:**
- Create: `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Transforms/TTNNFoldThroughAgnosticOps.cpp`

- [ ] **Step 1: Write the pass file**

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Phase 3 (A.2.a) fold: to_layout(unary_agnostic_op(to_layout(x, L1)), L0)
// → unary_agnostic_op(x), where L0 is the original layout of x and the
// intermediate op is unary and layout-agnostic for the dtype involved.
//
// Complements foldConsecutiveToLayoutOp at TTNNOps.cpp:2062 which only
// folds DIRECTLY-adjacent to_layout pairs.

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNFOLDTHROUGHAGNOSTICOPS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Allowlist — unary ops only (binary ops have layout-asymmetry; see spec).
const llvm::StringSet<> &unaryAllowlist() {
  static const llvm::StringSet<> kSet = {
      "ttnn.relu", "ttnn.gelu", "ttnn.silu", "ttnn.typecast",
  };
  return kSet;
}

// Returns the input layout of a to_layout op (extracted from its input type).
TTNNLayoutAttr getInputLayout(ToLayoutOp op) {
  auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
  if (!input_type) return {};
  return mlir::dyn_cast<TTNNLayoutAttr>(input_type.getEncoding());
}

// Returns the output layout of a to_layout op (from its result type).
TTNNLayoutAttr getOutputLayout(ToLayoutOp op) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  if (!out_type) return {};
  return mlir::dyn_cast<TTNNLayoutAttr>(out_type.getEncoding());
}

struct FoldThroughAgnostic : public mlir::OpRewritePattern<ToLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ToLayoutOp outer,
                  mlir::PatternRewriter &rewriter) const override {
    // outer = to_layout(?, L0)
    mlir::Operation *producer = outer.getInput().getDefiningOp();
    if (!producer) return mlir::failure();
    if (!unaryAllowlist().contains(producer->getName().getStringRef())) {
      return mlir::failure();
    }
    if (producer->getNumOperands() != 1) return mlir::failure();

    // inner = to_layout(x, L1)
    auto inner = producer->getOperand(0).getDefiningOp<ToLayoutOp>();
    if (!inner) return mlir::failure();

    // L0 must equal inner.input layout (= original layout of x).
    TTNNLayoutAttr outer_out = getOutputLayout(outer);
    TTNNLayoutAttr inner_in = getInputLayout(inner);
    if (!outer_out || !inner_in || outer_out != inner_in) {
      return mlir::failure();
    }

    // DRAM↔L1 safety guard, mirroring foldConsecutiveToLayoutOp.
    MemoryConfigAttr outer_mc = outer.getMemoryConfigAttr();
    MemoryConfigAttr inner_mc = inner.getMemoryConfigAttr();
    if (outer_mc && inner_mc &&
        outer_mc.getBufferType().getValue() == BufferType::L1 &&
        inner_mc.getBufferType().getValue() == BufferType::DRAM) {
      return mlir::failure();
    }

    // Rewrite: replace `outer` with a clone of `producer` taking `x` (the
    // inner.input) as its operand. The new op's result type matches outer's.
    mlir::Value x = inner.getInput();
    mlir::OperationState state(producer->getLoc(), producer->getName());
    state.addOperands({x});
    state.addTypes({outer.getType()});
    state.addAttributes(producer->getAttrs());
    mlir::Operation *clone = rewriter.create(state);
    rewriter.replaceOp(outer, clone->getResults());
    return mlir::success();
  }
};

class TTNNFoldThroughAgnosticOps
    : public impl::TTNNFoldThroughAgnosticOpsBase<TTNNFoldThroughAgnosticOps> {
public:
  using impl::TTNNFoldThroughAgnosticOpsBase<TTNNFoldThroughAgnosticOps>::
      TTNNFoldThroughAgnosticOpsBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldThroughAgnostic>(&getContext());
    if (failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createTTNNFoldThroughAgnosticOps() {
  return std::make_unique<TTNNFoldThroughAgnosticOps>();
}

}  // namespace mlir::tt::ttnn
```

- [ ] **Step 2: Add to CMakeLists.txt**

Locate `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Transforms/CMakeLists.txt`. Find the `add_mlir_dialect_library(TTMLIRTTNNTransforms ...)` block. Add to the source list:

```
add_mlir_dialect_library(TTMLIRTTNNTransforms
  ...
  TTNNFoldThroughAgnosticOps.cpp
  ...
)
```

---

### Task 3.4: Wire pass into TTNNPipelines.cpp

**Files:**
- Modify: `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`

- [ ] **Step 1: Add to layout-decomposition helper**

Read `TTNNPipelines.cpp` around line 249–251. Expected:
```cpp
void createTTNNPipelineLayoutDecompositionPass(
    OpPassManager &pm, const TTIRToTTNNCommonPipelineOptions &options) {
  pm.addPass(createTTNNDecomposeLayouts());
}
```

Use Edit. Change to:
```cpp
void createTTNNPipelineLayoutDecompositionPass(
    OpPassManager &pm, const TTIRToTTNNCommonPipelineOptions &options) {
  pm.addPass(createTTNNDecomposeLayouts());
  pm.addPass(createTTNNFoldThroughAgnosticOps());
}
```

---

### Task 3.5: Rebuild + verify Phase 3

- [ ] **Step 1: Rebuild**

```bash
cd /home/mhnie/tt-mlir-sglang
bash scripts/build_and_install.sh 2>&1 | tee /tmp/phase3_build.log
tail -5 /tmp/phase3_build.log
```

Expected: `Done.`

If compile fails: read C++ errors. Common issues:
- TableGen-generated symbols missing → check `Passes.td` declaration matches `GEN_PASS_DEF_` macro name.
- `applyPatternsAndFoldGreedily` deprecated in newer MLIR → use `applyPatternsGreedily` if linker complains.

- [ ] **Step 2: Re-run probe with export_path; re-run Phase 0.3 counting script**

```bash
docker exec tt-xla-eval bash -c \
  'PYTHONPATH=/opt/tt-mlir-toolchain/python_packages/mlir_core python3 \
   /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/phase0_count_to_layout_triples.py \
   /tmp/shlo_dump/irs/ttnn_*.mlir'
```

Expected: triple count significantly lower than the Phase 0.3 baseline (typically near 0 if the fold matches; otherwise the pass isn't firing).

- [ ] **Step 3: Bench Phase 3**

```bash
docker exec -d tt-xla-eval bash -c 'cd /sglang && BYPASS_PREWARM=1 SGLANG_TT_CACHE_MODE=index_copy \
  python3 -u python/sglang/srt/hardware_backend/tenstorrent/test/bench_3run_server_alive.py \
  --models Qwen3-8B --backend tt_xla --input-len 1024 --output-len 1024 \
  --out-tag phase3_a2a > /tmp/phase3_bench.log 2>&1'
```

```bash
jq '.results[0].run2_warm_new_prompt.tpot_ms_mean' \
   /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase3_a2a.json
```

Expected: TPOT further drops vs Phase 2. Magnitude depends on the activation-tilize share from Phase 0.1.

---

### Task 3.6: Commit Phase 3

- [ ] **Step 1: Commit fork changes**

```bash
cd /home/mhnie/tt-mlir-sglang
git add include/ttmlir/Dialect/TTNN/Transforms/Passes.td \
        include/ttmlir/Dialect/TTNN/Transforms/Passes.h \
        lib/Dialect/TTNN/Transforms/TTNNFoldThroughAgnosticOps.cpp \
        lib/Dialect/TTNN/Transforms/CMakeLists.txt \
        lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp
git commit -m "phase3(tt-mlir): A.2.a — TTNNFoldThroughAgnosticOps

Fold pattern: to_layout(unary_agnostic_op(to_layout(x, L1)), L0) →
unary_agnostic_op(x). Complements foldConsecutiveToLayoutOp which only
handles directly-adjacent to_layout pairs.

Allowlist restricted to unary ops (relu/gelu/silu/typecast) because
binary ops require both operands at the same layout. Mirrors the
existing DRAM→L1 safety guard from foldConsecutiveToLayoutOp.

Scheduled inside createTTNNPipelineLayoutDecompositionPass immediately
after createTTNNDecomposeLayouts.

Phase 3 of docs/superpowers/specs/2026-05-17-tt-xla-tilize-attack-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Commit sglang fixture**

```bash
cd /home/mhnie/sglang
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase3_a2a.json
git commit -m "phase3(tt-xla): bench fixture for A.2.a TPOT measurement

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# Phase 4 — A.3.a: PopulateArgumentTypesAutoDetect pass

Phase 4 ports the classifier from Phase 2 into a tt-mlir pass scheduled after `tt-populate-argument-types`. Auto-detect handles modules where Phase 2's pjrt-side classifier emitted no entries (multi-func modules, wrappered entry, etc.).

### Task 4.1: Add pass registration

**Files:**
- Modify: `/home/mhnie/tt-mlir-sglang/include/ttmlir/Dialect/StableHLO/Transforms/Passes.td`
- Modify: `/home/mhnie/tt-mlir-sglang/include/ttmlir/Dialect/StableHLO/Transforms/Passes.h`

- [ ] **Step 1: Add to Passes.td**

```tablegen
def PopulateArgumentTypesAutoDetect
    : Pass<"tt-populate-argument-types-auto-detect", "::mlir::ModuleOp"> {
  let summary = "Auto-detect Input/Parameter/Constant on un-annotated block args.";
  let description = [{
    Walks each non-private func::FuncOp and classifies each block arg
    that does NOT already have ttcore.argument_type set. Uses Strategy
    A (per-arg marker) → B (cluster marker) → C (5-rule heuristic).

    Runs AFTER tt-populate-argument-types so an explicit map (from
    pjrt-plugin-tt's classifyArgs) takes precedence; auto-detect only
    fills gaps for un-annotated args.

    Phase 4 (A.3.a) per docs/superpowers/specs/2026-05-17-tt-xla-tilize-
    attack-design.md.
  }];
  let constructor = "createPopulateArgumentTypesAutoDetect()";
  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect"];
}
```

- [ ] **Step 2: Declare constructor in Passes.h**

```cpp
std::unique_ptr<mlir::Pass> createPopulateArgumentTypesAutoDetect();
```

---

### Task 4.2: Implement the auto-detect pass

**Files:**
- Create: `/home/mhnie/tt-mlir-sglang/lib/Dialect/StableHLO/Transforms/PopulateArgumentTypesAutoDetect.cpp`

- [ ] **Step 1: Write the pass file**

Port the Strategy A/B/C helpers from Phase 2 (which lived in pjrt-plugin-tt) into the tt-mlir pass. Keep the rules identical so behavior matches.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Phase 4 (A.3.a) — auto-detect parameter args when tt-populate-argument-types
// receives no explicit map (or only a partial map). Runs after that pass.

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::stablehlo {

#define GEN_PASS_DEF_POPULATEARGUMENTTYPESAUTODETECT
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

// === Rule 1: shape gate ===
bool ruleOneShapeGate(mlir::BlockArgument arg) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
  if (!t || t.getShape().empty()) return false;
  if (t.getShape().size() == 1) return t.getShape()[0] >= 64;
  for (int64_t d : t.getShape()) if (d < 16) return false;
  return true;
}

bool isLayoutOnlyOp(mlir::Operation *op) {
  if (!op) return false;
  auto n = op->getName().getStringRef();
  return n == "stablehlo.transpose" || n == "stablehlo.reshape" ||
         n == "stablehlo.broadcast_in_dim" || n == "stablehlo.convert" ||
         n == "stablehlo.slice";
}

bool isAllowlistTerminalUse(mlir::Operation *op, mlir::Value operand) {
  if (!op) return false;
  auto n = op->getName().getStringRef();
  if (n == "stablehlo.dot_general") return true;
  if (n == "stablehlo.convolution") return true;
  if (n == "stablehlo.gather") return true;
  if (n == "stablehlo.dynamic_slice") return true;
  if (n == "stablehlo.slice") return true;
  if (n == "stablehlo.multiply" || n == "stablehlo.add") {
    for (mlir::Value other : op->getOperands()) {
      if (other == operand) continue;
      if (mlir::isa<mlir::BlockArgument>(other)) return false;
    }
    return true;
  }
  return false;
}

bool ruleTwoUsePattern(mlir::BlockArgument arg) {
  llvm::SmallVector<std::pair<mlir::Value, int>> wl;
  wl.push_back({arg, 0});
  llvm::DenseSet<mlir::Value> seen;
  while (!wl.empty()) {
    auto [v, depth] = wl.pop_back_val();
    if (!seen.insert(v).second) continue;
    if (v.use_empty()) return false;
    for (mlir::OpOperand &u : v.getUses()) {
      mlir::Operation *user = u.getOwner();
      if (isLayoutOnlyOp(user)) {
        if (depth >= 3) return false;
        for (mlir::Value r : user->getResults()) wl.push_back({r, depth + 1});
        continue;
      }
      if (!isAllowlistTerminalUse(user, v)) return false;
    }
  }
  return true;
}

bool ruleThreeNoMutate(mlir::BlockArgument arg) {
  for (mlir::OpOperand &u : arg.getUses()) {
    auto *op = u.getOwner();
    if (op->getName().getStringRef() == "stablehlo.scatter" &&
        op->getNumOperands() > 0 && op->getOperand(0) == arg) {
      return false;
    }
  }
  return true;
}

bool ruleFourNotBatchInput(mlir::BlockArgument arg) {
  for (mlir::OpOperand &u : arg.getUses()) {
    auto *op = u.getOwner();
    if (op->getName().getStringRef() == "stablehlo.dynamic_slice" &&
        op->getNumOperands() > 0 && op->getOperand(0) == arg) {
      for (uint32_t i = 1; i < op->getNumOperands(); ++i) {
        if (mlir::isa<mlir::BlockArgument>(op->getOperand(i))) return false;
      }
    }
  }
  return true;
}

bool ruleFiveSoftmaxPathExcluded(mlir::BlockArgument arg) {
  struct Item { mlir::Value v; int depth; bool seen_add_mul; };
  llvm::SmallVector<Item> wl;
  wl.push_back({arg, 0, false});
  llvm::DenseSet<std::pair<mlir::Value, bool>> seen;
  int popped = 0;
  while (!wl.empty()) {
    if (++popped > 256) return false;
    Item item = wl.pop_back_val();
    if (!seen.insert({item.v, item.seen_add_mul}).second) continue;
    for (mlir::OpOperand &u : item.v.getUses()) {
      mlir::Operation *user = u.getOwner();
      auto n = user->getName().getStringRef();
      if (n == "stablehlo.exponential" && item.seen_add_mul) return false;
      bool lo = isLayoutOnlyOp(user);
      int nd = lo ? item.depth : item.depth + 1;
      if (nd > 8) continue;
      bool am = (n == "stablehlo.add" || n == "stablehlo.subtract" ||
                n == "stablehlo.multiply");
      for (mlir::Value r : user->getResults()) {
        wl.push_back({r, nd, item.seen_add_mul || am});
      }
    }
  }
  return true;
}

mlir::tt::ttcore::ArgumentType classifyOne(mlir::BlockArgument arg) {
  using AT = mlir::tt::ttcore::ArgumentType;
  if (!ruleOneShapeGate(arg)) return AT::Input;
  if (!ruleTwoUsePattern(arg)) return AT::Input;
  if (!ruleThreeNoMutate(arg)) return AT::Input;
  if (!ruleFourNotBatchInput(arg)) return AT::Input;
  if (!ruleFiveSoftmaxPathExcluded(arg)) return AT::Input;
  return AT::Parameter;
}

class PopulateArgumentTypesAutoDetect
    : public impl::PopulateArgumentTypesAutoDetectBase<
          PopulateArgumentTypesAutoDetect> {
public:
  using impl::PopulateArgumentTypesAutoDetectBase<
      PopulateArgumentTypesAutoDetect>::PopulateArgumentTypesAutoDetectBase;

  void runOnOperation() final {
    auto module = getOperation();
    auto &ctx = getContext();
    // Find unique non-private func.
    mlir::func::FuncOp target;
    int non_private = 0;
    for (auto f : module.getOps<mlir::func::FuncOp>()) {
      if (f.getVisibility() == mlir::SymbolTable::Visibility::Private) continue;
      ++non_private;
      target = f;
    }
    if (non_private != 1) {
      emitWarning(module.getLoc())
          << "[auto-detect] expected unique non-private func, found "
          << non_private << " — skipping";
      return;
    }
    for (uint32_t i = 0; i < target.getNumArguments(); ++i) {
      // Skip if already annotated (preserve Phase 2 explicit map and JAX-emitted attrs).
      if (target.getArgAttrOfType<mlir::tt::ttcore::ArgumentTypeAttr>(
              i, mlir::tt::ttcore::ArgumentTypeAttr::name)) {
        continue;
      }
      auto cls = classifyOne(target.getArgument(i));
      target.setArgAttr(i, mlir::tt::ttcore::ArgumentTypeAttr::name,
                        mlir::tt::ttcore::ArgumentTypeAttr::get(&ctx, cls));
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createPopulateArgumentTypesAutoDetect() {
  return std::make_unique<PopulateArgumentTypesAutoDetect>();
}

}  // namespace mlir::tt::stablehlo
```

- [ ] **Step 2: Add to CMakeLists.txt**

In `/home/mhnie/tt-mlir-sglang/lib/Dialect/StableHLO/Transforms/CMakeLists.txt`, find the existing `add_mlir_dialect_library(...)` block and add `PopulateArgumentTypesAutoDetect.cpp` to the sources.

---

### Task 4.3: Write lit test

**Files:**
- Create: `/home/mhnie/tt-mlir-sglang/test/ttmlir/Conversion/StableHLOToTTIR/auto_detect_argument_types.mlir`

- [ ] **Step 1: Write fixture covering 4 cases**

```mlir
// RUN: ttmlir-opt --tt-populate-argument-types-auto-detect %s | FileCheck %s

// Four cases:
//   %arg0 — embedding gather target (Parameter — Rule 2 gather hit)
//   %arg1 — RMSNorm scale (Parameter — Rule 2 multiply+broadcast hit)
//   %arg2 — attention mask added before softmax (Input — Rule 5 fires)
//   %arg3 — matmul weight (Parameter — Rule 2 dot_general hit)

module {
  // CHECK: func.func @main
  // CHECK-SAME: %arg0: tensor<151936x4096xf32> {ttcore.argument_type = #ttcore<argument_type parameter>}
  // CHECK-SAME: %arg1: tensor<4096xf32> {ttcore.argument_type = #ttcore<argument_type parameter>}
  // CHECK-SAME: %arg2: tensor<1x1x1024x1024xf32> {ttcore.argument_type = #ttcore<argument_type input>}
  // CHECK-SAME: %arg3: tensor<4096x4096xf32> {ttcore.argument_type = #ttcore<argument_type parameter>}
  func.func @main(
      %arg0: tensor<151936x4096xf32>,
      %arg1: tensor<4096xf32>,
      %arg2: tensor<1x1x1024x1024xf32>,
      %arg3: tensor<4096x4096xf32>,
      %arg4: tensor<1x1024xi32>,
      %arg5: tensor<1x1024x4096xf32>) -> tensor<1x1024x4096xf32> {

    // Embedding gather using arg0.
    %0 = "stablehlo.gather"(%arg0, %arg4) {
      dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0],
                                            start_index_map = [0], index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 4096>
    } : (tensor<151936x4096xf32>, tensor<1x1024xi32>) -> tensor<1x1024x4096xf32>

    // RMSNorm: arg5 * broadcast(arg1).
    %1 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = array<i64: 2>}
      : (tensor<4096xf32>) -> tensor<1x1024x4096xf32>
    %2 = stablehlo.multiply %arg5, %1 : tensor<1x1024x4096xf32>

    // Attention mask: scores + mask → reduce_max → subtract → exponential.
    // (Mocked attn_scores to exercise the softmax path with mask.)
    %scores = stablehlo.add %2, %2 : tensor<1x1024x4096xf32>
    %scores4d = "stablehlo.reshape"(%scores)
      : (tensor<1x1024x4096xf32>) -> tensor<1x1x1024x4096xf32>
    %mask_slice = "stablehlo.slice"(%arg2) {
      start_indices = array<i64: 0, 0, 0, 0>,
      limit_indices = array<i64: 1, 1, 1024, 4096>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<1x1x1024x1024xf32>) -> tensor<1x1x1024x4096xf32>
    %scored_mask = stablehlo.add %scores4d, %mask_slice : tensor<1x1x1024x4096xf32>
    %reduced = stablehlo.constant dense<1.0> : tensor<1x1x1024x1xf32>
    %bcast = "stablehlo.broadcast_in_dim"(%reduced) {broadcast_dimensions = array<i64: 0, 1, 2, 3>}
      : (tensor<1x1x1024x1xf32>) -> tensor<1x1x1024x4096xf32>
    %normed = stablehlo.subtract %scored_mask, %bcast : tensor<1x1x1024x4096xf32>
    %exp = stablehlo.exponential %normed : tensor<1x1x1024x4096xf32>

    // Matmul with arg3 (weight).
    %3 = "stablehlo.dot_general"(%2, %arg3) {
      dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [2],
                                              rhs_contracting_dimensions = [0]>
    } : (tensor<1x1024x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x1024x4096xf32>

    return %3 : tensor<1x1024x4096xf32>
  }
}
```

- [ ] **Step 2: Run the lit test**

```bash
docker exec tt-xla-eval bash -c \
  '/home/mhnie/tt-mlir-sglang/build/bin/ttmlir-opt \
     --tt-populate-argument-types-auto-detect \
     /tt-mlir-sglang/test/ttmlir/Conversion/StableHLOToTTIR/auto_detect_argument_types.mlir \
   | /opt/tt-mlir-toolchain/bin/FileCheck \
     /tt-mlir-sglang/test/ttmlir/Conversion/StableHLOToTTIR/auto_detect_argument_types.mlir'
```

Expected: zero output (FileCheck silent on success), exit code 0. If a CHECK fails, FileCheck prints the mismatch — fix the classifier or fixture.

---

### Task 4.4: Schedule the pass

**Files:**
- Modify: `/home/mhnie/tt-mlir-sglang/lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp`

- [ ] **Step 1: Insert between tt-populate-argument-types (line 24) and createAnalyzeMeshPass (line 52)**

Read `StableHLOPipelines.cpp:20–52` to confirm structure. Expected: line 24 is the `createTTPopulateArgumentTypes` call; line 52 is the `createAnalyzeMeshPass` call.

Use Edit. Find the line:
```cpp
  pm.addPass(createTTPopulateArgumentTypes(options.argumentTypeMap));
```

Replace with:
```cpp
  pm.addPass(createTTPopulateArgumentTypes(options.argumentTypeMap));
  // Phase 4 A.3.a: auto-detect un-annotated args. Runs AFTER the explicit-map
  // pass so the explicit map wins where set; auto-detect fills gaps.
  pm.addPass(stablehlo::createPopulateArgumentTypesAutoDetect());
```

---

### Task 4.5: Rebuild + verify Phase 4

- [ ] **Step 1: Rebuild**

```bash
cd /home/mhnie/tt-mlir-sglang
bash scripts/build_and_install.sh 2>&1 | tee /tmp/phase4_build.log
tail -5 /tmp/phase4_build.log
```

- [ ] **Step 2: Lit test still passes after wiring**

Re-run the lit test from Task 4.3 Step 2.

- [ ] **Step 3: Bench Phase 4 — independent of Phase 2's map**

Temporarily disable Phase 2 by setting an env var inside the bench-launched server: add to the bench command:
```
TT_DISABLE_PJRT_ARG_TYPE_MAP=1
```

This requires a tiny additional patch to `module_builder.cc`: wrap the `classifyArgs` call in `if (!std::getenv("TT_DISABLE_PJRT_ARG_TYPE_MAP")) { ... }`. Apply that patch (small one-line change), rebuild, then run:

```bash
docker exec -d tt-xla-eval bash -c 'cd /sglang && BYPASS_PREWARM=1 SGLANG_TT_CACHE_MODE=index_copy \
  TT_DISABLE_PJRT_ARG_TYPE_MAP=1 \
  python3 -u python/sglang/srt/hardware_backend/tenstorrent/test/bench_3run_server_alive.py \
  --models Qwen3-8B --backend tt_xla --input-len 1024 --output-len 1024 \
  --out-tag phase4_a3a > /tmp/phase4_bench.log 2>&1'
```

```bash
jq '.results[0].run2_warm_new_prompt.tpot_ms_mean' \
   /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase4_a3a.json
```

Expected: TPOT approximately matches Phase 2's TPOT (auto-detect classified the same args as Phase 2's explicit map).

- [ ] **Step 4: Coverage check**

Dump IR with `export_path` (Phase 0.2 mechanism) under `TT_DISABLE_PJRT_ARG_TYPE_MAP=1`. Count parameter-marked args:

```bash
docker exec tt-xla-eval bash -c \
  'grep -cE "argument_type = #ttcore<argument_type parameter>" /tmp/shlo_dump/irs/shlo_compiler_*.mlir'
```

Compare against expected weight count:
```bash
docker exec tt-xla-eval bash -c \
  'python3 -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(\"/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/$(ls /root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/ | head -1)/\")
print(sum(1 for _, p in m.named_parameters() if p.numel() > 1000))
"'
```

Expected: parameter-marked count ≥ 95% of expected weight count. If coverage misses a category (embedding, rotary, RMSNorm), expand the Rule 2 allowlist.

---

### Task 4.6: Commit Phase 4

- [ ] **Step 1: Commit fork**

```bash
cd /home/mhnie/tt-mlir-sglang
git add include/ttmlir/Dialect/StableHLO/Transforms/Passes.td \
        include/ttmlir/Dialect/StableHLO/Transforms/Passes.h \
        lib/Dialect/StableHLO/Transforms/PopulateArgumentTypesAutoDetect.cpp \
        lib/Dialect/StableHLO/Transforms/CMakeLists.txt \
        lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp \
        test/ttmlir/Conversion/StableHLOToTTIR/auto_detect_argument_types.mlir
git commit -m "phase4(tt-mlir): A.3.a — PopulateArgumentTypesAutoDetect

Auto-detect pass that classifies un-annotated block args with the
same Strategy A→B→C as Phase 2's pjrt-side classifier. Skips args
that already carry ttcore.argument_type (preserves Phase 2's
explicit map AND JAX-emitted markers from annotateArgumentAttributes).

Scheduled in StableHLOPipelines.cpp AFTER createTTPopulateArgumentTypes
(line 24) and BEFORE createAnalyzeMeshPass (line 52).

Lit test covers four cases: embedding-gather (Parameter),
RMSNorm-multiply (Parameter), attention-mask add+softmax (Input via
Rule 5), dot_general matmul (Parameter).

Phase 4 of docs/superpowers/specs/2026-05-17-tt-xla-tilize-attack-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Commit sglang fixture**

```bash
cd /home/mhnie/sglang
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q8b_phase4_a3a.json
git commit -m "phase4(tt-xla): bench fixture for A.3.a TPOT measurement

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# Phase 5 — Final reporting

### Task 5.1: Update handoff doc TPOT table

**Files:**
- Modify: `/home/mhnie/sglang/docs/platforms/tt_xla_tpot_handoff_2026-05-17.md`

- [ ] **Step 1: Add rows to the TPOT table at lines 11–22**

Read existing format (columns: `Config | TPOT warm | tok/s | vs baseline`). Add one row per landed phase:

| Config | TPOT warm | tok/s | vs baseline |
|---|---|---|---|
| Phase 1 — fork repointed, no Phase 2 | ~132 ms | ~7.6 | 1.00× |
| **Phase 2 — A.1.a (argumentTypeMap)** | **<TPOT> ms** | <tok/s> | <ratio>× |
| Phase 3 — + A.2.a (TTNNFoldThroughAgnosticOps) | <TPOT> ms | <tok/s> | <ratio>× |
| Phase 4 — + A.3.a (auto-detect, replaces Phase 2 explicit map) | <TPOT> ms | <tok/s> | <ratio>× |

Fill in measured values from each phase's fixture file via `jq '.results[0].run2_warm_new_prompt.tpot_ms_mean'`.

- [ ] **Step 2: Add SUPERSEDED note to Open work section**

Find the "Open work — concrete next steps" section at lines 155–185. Add at the top of that section:

```markdown
> **⚠️ SUPERSEDED.** This section describes the OLD framing of A.1/A.2/A.3 as Python-side pre-tilize work, in-MLIR Tilize/Untilize fold patterns, and a StableHLOToTTIR-only A.3. Through 17 review rounds it became clear those framings were not implementable (or already done). The actual landed implementation is described in `docs/superpowers/specs/2026-05-17-tt-xla-tilize-attack-design.md` (3 phase pairs: A.1.a explicit `argumentTypeMap` + heuristic classifier; A.2.a `to_layout`-pair fold; A.3.a auto-detect pass).
```

---

### Task 5.2: Update memory entries

**Files:**
- Modify: `/home/mhnie/.claude/projects/-home-mhnie-sglang/memory/tenstorrent-tt-xla-tilize-bottleneck.md`
- Modify: `/home/mhnie/.claude/projects/-home-mhnie-sglang/memory/tenstorrent-tt-mlir-sglang-fork.md`
- Modify: `/home/mhnie/.claude/projects/-home-mhnie-sglang/memory/tenstorrent-tt-xla-tpot-workstream-a.md`

- [ ] **Step 1: Update `tenstorrent-tt-xla-tilize-bottleneck.md`**

Append a results section:
```markdown
## Outcome (2026-XX-XX after Phase 1-4 landing)

- Phase 2 (A.1.a explicit map): TPOT dropped from 132 ms → <X> ms.
- Phase 3 (A.2.a fold): <landed | skipped per Phase 0.3 gate>.
- Phase 4 (A.3.a auto-detect): equivalent to Phase 2 standalone; <TPOT> ms.
- Residual bottleneck: <observed remaining Tracy share, if any>.
- Phase 0.1 found weight tilizes were <X%> of total Tilize op count.
```

- [ ] **Step 2: Update `tenstorrent-tt-mlir-sglang-fork.md`**

Bump commit list: append the new commits from Phases 3 and 4 (Phase 3 conditional).

- [ ] **Step 3: Correct `tenstorrent-tt-xla-tpot-workstream-a.md`**

The note "B.2 v3 patch built into canonical wheel" reflected a separate manual operation, NOT anything the build script automated. Correct that line.

---

### Task 5.3: Optional new memory for surprising findings

- [ ] **Step 1: Decide whether Phase 0 / phase verification surfaced anything worth a new memory**

Examples of surprises worth saving:
- Tracy column names different than spec assumed
- BFP8 cast lived outside consteval wrapper (real upstream bug)
- torch_xla emitter version emits a different marker than expected

If so: write a new memory file under `/home/mhnie/.claude/projects/-home-mhnie-sglang/memory/` and add a one-line entry to `MEMORY.md`.

If not: skip.

---

### Task 5.4: Final commit

- [ ] **Step 1: Commit handoff updates + memory updates**

```bash
cd /home/mhnie/sglang
git add docs/platforms/tt_xla_tpot_handoff_2026-05-17.md
git commit -m "phase5(tt-xla): handoff doc — final TPOT table after A.1.a/A.2.a/A.3.a

Adds rows for Phase 1 smoke, Phase 2 (A.1.a), Phase 3 (A.2.a — landed
or skipped per gate), Phase 4 (A.3.a). Annotates the OLD Open-work
section as SUPERSEDED with pointer to the design spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Memory updates: those live outside the repo and don't need a git commit.

---

# Self-review reminder for executor

After completing every task: run `git status` to ensure no stray files. After completing every phase: re-run the verification step explicitly stated for that phase. Do NOT report a phase as complete without producing the listed verification artifact (SHA-diff result, IR dump grep, fixture file, etc.).

The "non-negotiable scope" section at the top of this plan applies: if a task seems too hard, the answer is "ask a question", not "skip it".
