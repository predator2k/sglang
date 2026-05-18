# tt-xla Tilize-bottleneck attack — design

**Date:** 2026-05-17 (revised after subagent review)
**Status:** spec — pending user review
**Predecessor:** `docs/platforms/tt_xla_tpot_handoff_2026-05-17.md` (Priority A: A.1/A.2/A.3)

---

## 0. Executor instructions — no early stopping

The implementation plan derived from this spec MUST instruct the executor to fully implement every assigned phase. The following are NOT acceptable reasons to stop, defer, or descope:

- "out of scope"
- "too hard" / "too complex"
- "too much code involved"
- "not feasible without further design"
- "takes too long to compile"
- "requires code modification in another repo" (this work spans three: `/home/mhnie/sglang`, `/home/mhnie/tt-mlir-sglang`, `/home/mhnie/tt-xla/pjrt_implementation`)
- "I'll defer this to a follow-up"
- "the build broke, I'll skip ahead to the next phase" (a build failure is a blocker; surface as a question, do not auto-skip)
- "the verification step would take too long, I'll trust the diff" (every phase's verification is mandatory)

Genuine blockers are surfaced as questions, not unilateral descope decisions.

Phase 0 contains explicit measurement gates that **may** legitimately de-scope Phase 3 — but only on the specific quantitative criterion stated in the Phase 0 table (≥ 50 cancellable triples). All other phases are mandatory. (Earlier draft framing referenced "Phase 4 fallback to A.3.b" — that's not a fallback; Strategy C IS the heuristic that powers Phase 4 when Phase 0.2 finds no per-arg marker.)

## 1. Goal

Close the Tilize bottleneck in tt-xla Qwen3-8B serving. Tracy on the shipped BFP8 + `index_copy_` config (132.1 ms TPOT) shows TilizeWithValPadding = 81 % of decode time, matmul = 1.8 %. Layout conversion dominates. Target: reduce TPOT toward the tt_transformers_paged reference (37.4 ms TPOT) via three independent attacks landed serially.

Success metric per phase: Tracy Tilize-op share and server-bench TPOT before/after. Aggregate goal: "closer to 37 ms"; no fixed numeric target per phase.

## 2. Scope

In scope:
- **A.1.a** — set the `argumentTypeMap` pipeline option in pjrt-plugin-tt's `module_builder.cc` so the existing `tt-populate-argument-types` pass populates `ttcore.argument_type` on weight block args, allowing the already-wired `ConstEvalHoistTransform` to fire on tt-xla compiles. Const-eval is on by default (`compile_options.h:83 enable_const_eval = true`); only the per-arg type map is missing.
- **A.2.a** — new post-`TTNNDecomposeLayouts` fold pass for redundant Tilize/Untilize kernel sequences (only if Phase 0.3 measures ≥ 50 cancellable pairs per decode).
- **A.3.a** — auto-detect torch_xla parameter markers in either `StableHLOToTTIRPass` or `AnalyzeMesh` (whichever fits — see Phase 4 prereq), set `ArgumentTypeAttr::Parameter`, so const-eval fires without a user-supplied `argumentTypeMap`.

Out of scope:
- Multi-batch / bs=N optimizations (already landed in WS-A).
- B.* server-vs-probe gap (separate from Tilize attack).
- New model coverage.
- Upstream sgl-project/sglang PRs (per memory `feedback_no_upstream_pr`).

## 3. Repo layout this work touches

| Step | Repo | Files |
|---|---|---|
| Phase 1 wiring | `/home/mhnie/tt-xla` (host clone, detached at `470f0fad`) | Edit `third_party/CMakeLists.txt:47–85` (the `ExternalProject_Add(tt-mlir …)` block): replace `GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git` + `GIT_TAG ${TT_MLIR_VERSION}` with `SOURCE_DIR /home/mhnie/tt-mlir-sglang` + `DOWNLOAD_COMMAND ""` so CMake never clones from canonical. tt-mlir is NOT a git submodule and `USE_CUSTOM_TT_MLIR_VERSION` only controls which SHA gets assigned to `TT_MLIR_VERSION` — it does NOT skip the `ExternalProject_Add` block (lines 28–95 always run when `TOOLCHAIN!="ON"`). |
| A.1.a | `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc` | Build a `TTArgumentTypeMap` and assign it to both `stablehlo_pipeline_options.argumentTypeMap` (StableHLO PM, around line 828–855) and `ttnn_pipeline_options.argumentTypeMap` (TTNN PM, around line 980). Pipeline-option fields already exist in `TTNNPipelines.h:319–334` and `StableHLOPipelines.h:47–61`. Plug-in source lives at `/home/mhnie/tt-xla/pjrt_implementation/{inc,src}/`. |
| A.2.a | `/home/mhnie/tt-mlir-sglang` (branch `tenstorrent-p1`) | new `lib/Dialect/TTNN/Transforms/TTNNFoldRedundantLayoutKernels.cpp` + `Passes.td` registration + `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` insertion (see Phase 3 for exact insertion point) |
| A.3.a | `/home/mhnie/tt-mlir-sglang` | `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPass.cpp` (or a small new pass scheduled in `StableHLOPipelines.cpp` before `createTTPopulateArgumentTypes`). Note: `AnalyzeMesh.cpp` *reads* `ArgumentTypeAttr` and emits errors (`AnalyzeMesh.cpp:120–132`) when an arg isn't already annotated — it doesn't populate, so extending it would change its semantics, not "tweak a config". Don't go there. |
| Build/install | `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh` (committed at `95d656c8f`) | Rebuilds tt-mlir under the live tt-xla tree and reinstalls the pjrt plugin into `tt-xla-eval`. **Run from `tt-mlir-sglang`, NOT from `tt-xla`.** |
| Verification | `/home/mhnie/sglang` | bench/probe scripts already in `python/sglang/srt/hardware_backend/tenstorrent/test/` |

## 4. Key findings from spec-exploration

**4.1 The doc's "A.1 = pre-tilize from Python" is not implementable.** `ttnn` is a C++ device-side API not exposed on torch_xla. The spiritual successor is A.1.a: leverage the same compile-time const-eval pipeline that A.3 targets, but trigger it explicitly via the existing `argumentTypeMap` pipeline option rather than via auto-detection.

**4.2 The doc's "A.2 = add a Tilize(Untilize) fold pattern" is already implemented at the `to_layout` level.** `lib/Dialect/TTNN/IR/TTNNOps.cpp:2025–2114` has `foldIdentityToLayoutOp` and `foldConsecutiveToLayoutOp`. The Tilize bottleneck is downstream of these folds — after `TTNNDecomposeLayouts` expands `to_layout` into specific Tilize/Untilize/Typecast kernel calls. A.2.a addresses that post-decomp layer.

**4.3 A.3's const-eval infrastructure is fully wired.** `lib/Transforms/ConstEvalHoist.cpp:112` reads `ttcore::getConstsAndParams()` which filters block args by `ArgumentTypeAttr ∈ {Parameter, Constant}` (defined in `include/ttmlir/Dialect/TTCore/IR/Utils.h:70`). The CLI pass `tt-populate-argument-types` (defined in `lib/Dialect/TTCore/Utils/PopulateArgumentTypes.cpp:118`) accepts an `argument-types` option (key from `include/ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h:18 OptionNames::argumentTypes`) shaped as `function=input,parameter,parameter,constant`. The TTNN side closes the loop via `TTNNPrepareConstEvalCaching` + `TTNNConstEvalInputsToSystemMemory`. Both `TTNNPipelines.h:319` and `StableHLOPipelines.h:47` already declare an `argumentTypeMap` field that pjrt-plugin-tt can assign.

The gate is whether anyone sets `ArgumentTypeAttr` on the lowered func's block args. Today nothing does for tt-xla-compiled functions (`grep argumentTypeMap` in `/home/mhnie/tt-xla/pjrt_implementation/` returns zero hits).

**4.4 IR dumping is a built-in compile_options field, not a research item.** `compile_options.h:121` defines `std::optional<std::string> export_path`. When set, `ModuleBuilder::printModule` (defined at `module_builder.cc:1215`) emits stage IR files. Stages dumped: `vhlo` (line 432), `shlo` (452), `shlo_frontend` (465), `shlo_compiler` (843), `shlo_set_mesh_attr` (852), `ttir` (880), `ttnn` (1117), `ttnn_runtime` (1336). **Two important path details** (per `module_builder.cc:1230–1238`):
- Files land in `<export_path>/irs/`, NOT directly in `<export_path>/` (the `printModule` impl appends `/irs` and creates the directory).
- Filename pattern is `<stage>[_<model_name>]_<timestamp>.mlir`. The `_<model_name>` segment is included only when `export_model_name` was also passed via `set_custom_compile_options`; otherwise files are just `<stage>_<timestamp>.mlir`.

All references in the spec to dump files therefore use globbing in the `/irs/` subdir: `ls <export_path>/irs/shlo_*.mlir` or `find <export_path>/irs -name "shlo_*"`. Trigger via `torch_xla.set_custom_compile_options({"export_path": "/tmp/dump"})`. Use for Phase 0.2 (block-arg attrs in `shlo`-stage file) and Phase 0.3 (`to_layout`-pair count in `ttnn`-stage file).

**4.5 The TTNN pipeline already runs `ConstEvalHoist` three times, sandwiching weight-dtype-conversion.** `TTNNPipelines.cpp:309, 372, 386` invoke `createConstEvalHoistTransform()`; `TTNNWeightDtypeConversion` runs at line 339 between the first two. The comment around line 365–370 explicitly states the second invocation exists to "pick up any const-evalable ops created in weight dtype conversion." This means the BFP8-cast-outside-consteval risk is already mitigated by the pipeline design; if the dump shows otherwise, it's a real bug, not an expected limitation.

## 5. Phase plan

### Phase 0 — Investigations (no rebuilds, ~1.5 hours total)

Phase 0 is a gate, not a formality. The Phase 0.3 quantitative threshold is the ONLY automatic de-scope trigger in this plan.

| Step | Action | Mechanism | Output | Gate |
|---|---|---|---|---|
| 0.1 | Bucket Tracy Tilizes by tensor shape (weight tilize vs activation tilize) | The aggregator at `_fixtures/tracy_aggregate.py` only reads `OP CODE`; it does NOT have shape columns. Use the RAW `ops_perf_results_*.csv` directly. **Tracy's actual shape encoding**: it emits four padded-logical-dim columns per input (`INPUT_0_W_PAD[LOGICAL]`, `INPUT_0_Z_PAD[LOGICAL]`, `INPUT_0_Y_PAD[LOGICAL]`, `INPUT_0_X_PAD[LOGICAL]`) plus `INPUT_0_LAYOUT`, `INPUT_0_DATATYPE`, `INPUT_0_MEMORY`. Shape is reassembled from the four W/Z/Y/X columns. Confirm column names by inspecting `head -1 ops_perf_results_*.csv | tr ',' '\n'` from the existing run dir. Write a small one-shot Python script that filters rows where `OP CODE == "TilizeWithValPadding"`, assembles the shape from the four pad columns, and joins against known weight shapes derived from `hf_model.config` (e.g., `[vocab, hidden]`, `[hidden, 3*hidden]`, etc., padded to tile boundaries). Match modulo tile-padding. | % split | If weights ≪ 5 %, document in Phase 5 and proceed (no descope — A.1.a/A.3 still land, just with lower expected gain) |
| 0.2 | Dump tt-xla StableHLO + TTIR for one decode step. Inspect block-arg attrs | Set `torch_xla.set_custom_compile_options({"export_path": "/tmp/shlo_dump", "enable_const_eval": True})` before warm-up. Run probe with `--decode 1`. **First confirm the round-trip works**: after one decode, `ls /tmp/shlo_dump/irs/shlo_*.mlir` must list at least one file (note: `irs/` is the actual subdir created by `printModule` at `module_builder.cc:1230`; filenames are `shlo_<timestamp>.mlir` if model_name unset, `shlo_<model>_<timestamp>.mlir` if set — the glob covers both). If the directory itself doesn't exist OR is empty, the Python→C++ option-key serialization didn't reach `module_builder.cc`. **Caveat:** `compile_options.cc:78–82` ABORTs the plugin when `export_path` is MISSING and the backend is NOT `TTNNFlatbuffer`. The default tt-xla backend IS `TTNNFlatbuffer`, so a missing/empty `export_path` won't trigger ABORT in practice — the symptom of a failed round-trip is just "no files written", not a crash. Then grep the matched `shlo_*.mlir` file for `mhlo.`/`xla.`/`tf.`/`jax.`/`torch.`/`_xla` -prefixed arg attrs. | Confirmed torch_xla marker name(s) on block args, OR "no per-arg marker" | "No per-arg marker" → Strategy C (heuristic) handles via Phase 4 auto-detect. NOT a descope. |
| 0.3 | Count `to_layout`-pair patterns in TTNN IR | Open the `ttnn`-stage MLIR file from 0.2 (glob `<export_path>/irs/ttnn_*.mlir`). Count `ttnn.to_layout → <allowlist_op from Phase 3> → ttnn.to_layout` chains where the final to_layout's output layout matches the first's input layout. **MLIR isn't a nested-paren language** — string parsing won't work because ops are SSA-numbered flat statements (`%2 = "ttnn.to_layout"(%1) ...`). Implementation:
- Option (a) `ttmlir-opt --print-op-stats`: the binary lives at `/tt-xla/third_party/tt-mlir/src/tt-mlir/build/tools/ttmlir-opt` BEFORE Phase 1 (canonical clone); after Phase 1 the fork's build at `/home/mhnie/tt-mlir-sglang/build/bin/ttmlir-opt` is the right one. Phase 0 runs pre-Phase-1, so use the canonical-clone path.
- Option (b) Python MLIR bindings (`mlir.ir` module): bindings at `/opt/tt-mlir-toolchain/python_packages/mlir_core/` in `tt-xla-eval`. Activate with `PYTHONPATH=/opt/tt-mlir-toolchain/python_packages/mlir_core python3 -c "from mlir import ir"`. **Critical**: TTNN dialect is not auto-registered — set `ctx.allow_unregistered_dialects = True` before parsing the TTNN-stage MLIR file, otherwise the parser fails with `MLIRError: dialect not registered`. Walk ops via `op.walk(...)` and inspect operand SSA values for the `to_layout` chain pattern. **Note:** `ttnn.tilize`/`ttnn.untilize` are NOT MLIR ops (review 6 finding) — only `ttnn.to_layout` survives in IR. Don't grep for the runtime kernel names. | Cancellable-triple count | If < 50 per decode step, de-scope A.2.a (Phase 3 → skipped). Document count in Phase 5. |

If 0.2 also produces a clean per-arg parameter marker, capture an arg attr example for the Phase 4 implementation.

### Phase 1 — Repoint tt-xla third_party at the fork (~1 hour)

`USE_CUSTOM_TT_MLIR_VERSION` does NOT skip cloning. `third_party/CMakeLists.txt:7–9` only controls which SHA gets assigned to `TT_MLIR_VERSION`. The actual cloning happens unconditionally:
- Toolchain build path (`if (TOOLCHAIN STREQUAL "ON")` at lines 13–27) clones via `execute_process`.
- Normal path (`else()` at lines 28–95) clones via `ExternalProject_Add` at lines 47–85, which has `GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git` + `GIT_TAG ${TT_MLIR_VERSION}` hard-coded.

**Reality check up front: the fork is NOT currently the live build despite appearances.** `build_and_install.sh:38` passes `-DTTMLIR_SOURCE_DIR_OVERRIDE="$TTMLIR_DIR"` but no CMakeLists.txt in `/home/mhnie/tt-xla/` actually reads this variable (`grep -rn TTMLIR_SOURCE_DIR_OVERRIDE /home/mhnie/tt-xla/` returns only the stale `build-local/CMakeCache.txt`). The current live `pjrt-plugin-tt` reports `tt-mlir-commit=f3ddbfb6` (canonical). Memory `tenstorrent-tt-xla-tpot-workstream-a`'s claim "B.2 v3 patch built into canonical wheel" reflects a separate manual operation, not anything this script automated. After Phase 1 succeeds, update that memory entry.

**Build-flow choice.** The script already builds the fork independently in steps [1-2]/4 at `/home/mhnie/tt-mlir-sglang/build`. tt-xla's `ExternalProject_Add(tt-mlir …)` is currently structured to ALSO configure + build tt-mlir, with a DIFFERENT set of CMake args than the script uses. The two would collide on the same `CMakeCache.txt` if pointed at the same `BUILD_DIR`. The cleanest fix is to **neuter `ExternalProject_Add` to a passive reference** — let the script's pre-built fork supply the artifacts, and have tt-xla's CMakeLists treat the ExternalProject only as a target-dependency stub that points at pre-built outputs.

**Goal-first framing.** Each spec revision has tried to engineer the exact CMake mechanics and each has been caught with build-system surprises. This Phase 1 describes the GOAL, the CONSTRAINTS, and the VERIFICATION; the implementation plan and the executor iterate on the exact CMake invocations until verification passes. The verification step is load-bearing — do NOT skip it.

**Goal.** When the rebuilt `pjrt-plugin-tt` is loaded inside `tt-xla-eval`, its `tt-mlir-commit` SHA matches the fork HEAD (`2fc1d119e` at time of writing), not the canonical pin `f3ddbfb6`.

**Constraints / known surprises (each one bit a prior reviewer):**

1. tt-mlir is NOT a git submodule. `/home/mhnie/tt-xla/third_party/CMakeLists.txt:47–85` brings it in via `ExternalProject_Add` with `GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git` + `GIT_TAG ${TT_MLIR_VERSION}` hardcoded. Default `USE_CUSTOM_TT_MLIR_VERSION` only controls which SHA gets assigned to the var — it does NOT skip cloning.

2. `build_and_install.sh:38` passes `-DTTMLIR_SOURCE_DIR_OVERRIDE` but no CMakeLists.txt reads this var (grep confirms). Dead flag. Live plugin reports `tt-mlir-commit=f3ddbfb6` — fork has never been live.

3. `third_party/CMakeLists.txt:35` declares `TTMLIR_BUILD_DIR = ${TTMLIR_SOURCE_DIR}/src/tt-mlir/build`. Lines 36–39 hardcode `TT_METAL_RUNTIME_ROOT` to a path inside the canonical clone (`${TTPJRT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`).

4. Fork's `build/` dir doesn't exist on a fresh checkout. `cmake --install <BINARY_DIR>` on an empty dir fails because `cmake_install.cmake` doesn't exist there yet. The build script's step `[1-2]/4` (`cmake -B $TTMLIR_DIR/build -S $TTMLIR_DIR` then `ninja -C $TTMLIR_DIR/build`) creates it. ORDER MATTERS: fork must be configured + built BEFORE tt-xla configure consumes it.

5. Fork's tt-metal lives at `/home/mhnie/tt-mlir-sglang/third_party/tt-metal/src/tt-metal/` (per fork's own ExternalProject_Add for tt-metal). That path EXISTS only after fork configure has run once.

6. `pjrt_plugin_tt/__init__.py:83–95` validates `TT_METAL_RUNTIME_ROOT` only via `Path(user_override).exists()`. Any existing directory passes. No additional file-presence checks.

7. `pip3 show pjrt-plugin-tt` doesn't emit a separate `tt-mlir-commit=` field — that's part of the `Version:` string (e.g., `0.1.260428+dev.470f0fad8`). Use `pip3 show pjrt-plugin-tt | grep Version` AND grep the package's own `.so` for the commit triple: `strings $(python3 -c 'import pjrt_plugin_tt, os; print(os.path.dirname(pjrt_plugin_tt.__file__))')/pjrt_plugin_tt.so | grep "tt-mlir-commit="`.

8. `rm -rf` on a symlink without a trailing slash removes only the link, not the target (POSIX). The toolchain path's `ln -sfn` followed by Phase 1's `rm -rf` cleanup is safe IF no trailing slash is used. Verified.

**Concrete steps (executor refines until verification passes):**

1. Edit `/home/mhnie/tt-xla/third_party/CMakeLists.txt`:
   - ExternalProject_Add block at lines 47–85:
     - Replace `GIT_REPOSITORY ... + GIT_TAG ... + GIT_PROGRESS ON` with `SOURCE_DIR /home/mhnie/tt-mlir-sglang` + `DOWNLOAD_COMMAND ""`.
     - Set `CONFIGURE_COMMAND ""` and `BUILD_COMMAND ""` (build_and_install.sh handles those for the fork in steps [1-2]/4).
     - LEAVE `INSTALL_COMMAND` in place — but precondition the install on the fork's `build/cmake_install.cmake` existing (it will, after script step [2/4] succeeds). The install copies fork's build outputs to `${TTMLIR_INSTALL_PREFIX}` which is then globbed by `file(GLOB TTMLIR_LIBRARIES "${TTMLIR_LIB_DIR}/*.so")` at line 101. If install fails because of stale CMakeCache mismatch, ALSO set `INSTALL_COMMAND ""` and adjust line 101 to glob from `/home/mhnie/tt-mlir-sglang/build/lib/*.so` directly.
     - Update `TTMLIR_BUILD_DIR` at line 35 to `/home/mhnie/tt-mlir-sglang/build`.
   - **Leave lines 36–39 unchanged.** Line 37's `if(NOT DEFINED ENV{TT_METAL_RUNTIME_ROOT})` guards the hardcoded assignment on line 38. Since step 3 below exports `TT_METAL_RUNTIME_ROOT` in the shell environment before the cmake invocation, the guard skips line 38 entirely — no edit needed. (Earlier draft suggested clearing line 38; that was a no-op instruction because of the guard.)
   - If the toolchain path at lines 13–27 is exercised, replace its `git clone …` with `ln -sfn /home/mhnie/tt-mlir-sglang ${PROJECT_SOURCE_DIR}/tt-mlir/src/tt-mlir`.

2. Clean stale stamps and build dirs from prior canonical build: `rm -rf /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir-stamp /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir /home/mhnie/tt-xla/build-local /home/mhnie/tt-xla/build` (no trailing slashes).

3. Edit `build_and_install.sh`:
   - Remove the dead `-DTTMLIR_SOURCE_DIR_OVERRIDE="$TTMLIR_DIR"` flag from line 38.
   - Insert before step `[3/4]`: `export TT_METAL_RUNTIME_ROOT="/home/mhnie/tt-mlir-sglang/third_party/tt-metal/src/tt-metal"`. This path will exist after script step [1-2]/4 completes (fork's own ExternalProject pulls tt-metal there during fork configure+build).
   - Hardcode rather than `find`, because the find returns empty on fresh checkouts. The fork's CMakeLists.txt is what populates the path.

4. Create the patches directory if missing (`mkdir -p /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/`), then save the CMakeLists.txt edit and the build-script edit as checked-in patches under `tt-xla-source-dir-fork.patch` and `tt-mlir-sglang-build-script-cleanup.patch` so they can be re-applied after container restart.

5. Run `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh` (lives in tt-mlir-sglang). It builds fork first (steps [1-2]/4), THEN configures+builds tt-xla which consumes the pre-built fork (step [3]/4), THEN installs plugin (step [4]/4). Ordering matters.

6. **Hard verification — the load-bearing step.** Two prior verification approaches (strings-grep on .so, pip3-show metadata) were unsound — `pip3 show` reads from `setup.py`'s wheel-build-time metadata that doesn't refresh on rebuild; the strings-grep returned empty because the commit triple isn't in the .so binary. Use a **binary differential SHA** check instead:

   **Pre-Phase-1 snapshot (run BEFORE step 5):**
   ```
   docker exec tt-xla-eval bash -c \
     'sha256sum /tt-xla/third_party/tt-mlir/install/lib/*.so' \
     > /tmp/canonical-lib-shas.txt
   ```
   Save this as the canonical baseline.

   **Post-Phase-1 check (run AFTER step 5):**
   ```
   docker exec tt-xla-eval bash -c \
     'sha256sum /tt-xla/third_party/tt-mlir/install/lib/*.so' \
     > /tmp/post-phase1-lib-shas.txt
   diff /tmp/canonical-lib-shas.txt /tmp/post-phase1-lib-shas.txt
   ```
   At least one `.so` MUST differ — typically `libTTMLIRStableHLOToTTIR.so` (B.2 v3 patches it). If the diff is empty, the fork's source did NOT replace the canonical's binary — the rebuild path didn't take. **STOP. Investigate. Do NOT proceed to Phase 2.**

   **Path note:** the `stat -c "%Y"` + path comparison previously suggested won't help here because (a) `cmake --install` does NOT preserve mtimes — installed file gets a fresh mtime regardless of source, and (b) host vs container paths differ (`/home/mhnie/tt-mlir-sglang` on host, `/tt-mlir-sglang` in container). The container-side SHA check above sidesteps both issues.

   **Behavioural cross-check (additional):** run the smoke bench at step 7 and confirm TPOT is within ±5% of the pre-Phase-1 baseline (~132 ms). Both a regression or a surprise improvement could indicate the fork built but with a different compile path — investigate either way.

7. Smoke test: Qwen3-8B BFP8 server bench with `BYPASS_PREWARM=1 SGLANG_TT_CACHE_MODE=index_copy` (per Pitfall #4 — required for the rebuilt plugin). Expect ≈ 132 ms TPOT, no regression.

This re-activates **all 5 fork patches** that are ahead of canonical `f3ddbfb6` (B.2 v3 `2fc1d119e`, B.2 v2 `e62086947`, link tweak `4899ff291`, distributed-disable `8eb2e5b46`, build script `95d656c8f`). B.2 v3 is documented as working in memory `tenstorrent-tt-mlir-sglang-fork`; the other 4 are build-system tweaks expected to be neutral. If smoke regresses TPOT or build breaks: bisect by reverting fork commits individually.

### Phase 2 — A.1.a (set argumentTypeMap in module_builder.cc)

The central design choice in Phase 2 is **the argument classifier**: how to decide which block args are `Input` vs `Parameter` for the lowered HLO function. Phase 0.2's outcome determines which of three strategies to use, in priority order:

**Strategy A (used if Phase 0.2 finds per-arg marker).** Read the per-arg torch_xla marker name found in Phase 0.2 (e.g., `mhlo.parameter_replication`, `jax.arg_info`, or whatever shows up). For each block arg, if the marker says "parameter" → `ArgumentType::Parameter`; else `Input`.

**Strategy B (used if Phase 0.2 finds a module-level cluster marker).** Some HLO emitters mark parameters at module level via an array attr like `mhlo.parameter_replication = [false, false, true, true, true]`. Parse the array, map 1:1 onto block-arg indices.

**Strategy C (used if Phase 0.2 finds no marker).** Use a **classifier identical to A.3.b** (see Phase 4 for the rule set), applied to the incoming MLIR module before pipeline construction. The classifier walks each `func::FuncOp`, examines each block arg's use chain, and decides `Parameter` vs `Input`. **Phase 2 must implement this classifier even if Strategy A/B suffice**, because Phase 4 will reuse it; isolate it into a shared C++ helper to avoid code duplication.

Implementation (in `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc`):

1. Implement `classifyArgs(mlir::ModuleOp module) -> TTArgumentTypeMap` (or call a helper in tt-mlir if more natural). Internally, it tries Strategy A → B → C in order; returns the map keyed by function symbol name.
   - **Short-circuit: if a function's block args already carry `ttcore.argument_type` attrs (set by `frontend_passes::annotateArgumentAttributes` at `module_builder.cc:463` for JAX-origin modules using `tt.mark` custom_calls), do NOT add a map entry for that function. Letting the explicit map overwrite would clobber JAX-emitted annotations with a per-arg warning.** Check via `func.getArgAttrOfType<mlir::tt::ttcore::ArgumentTypeAttr>(idx, mlir::tt::ttcore::ArgumentTypeAttr::name)` — if non-null for any arg, skip the function.
2. Assign `stablehlo_pipeline_options.argumentTypeMap = classifyArgs(mlir_module);` AFTER line 832 (which already sets `stablehlo_pipeline_options.resultPresharded`) and BEFORE the call to `createStableHLOPipeline(...)` that consumes the options at line 833. Insertion line is approximately 833 (push existing line 833 down by one).
3. Assign `options.argumentTypeMap = classifyArgs(mlir_module);` immediately AFTER the `options` variable (`TTIRToTTNNCommonPipelineOptions`) is constructed at `module_builder.cc:985` (insertion line approximately 986).
4. Existing `compile_options.enable_const_eval` (default `true` at `inc/api/compile_options.h:83`) handles the rest: `tt-populate-argument-types` populates `ttcore.argument_type` on block args, `ConstEvalHoistTransform` fires (three invocations at `TTNNPipelines.cpp:309, 372, 386`), `TTNNPrepareConstEvalCaching` + `TTNNConstEvalInputsToSystemMemory` close the loop.
5. Add an INFO log line `"[TT-XLA] argumentTypeMap: strategy=<A|B|C>, function=<name>, K_inputs=%d, K_params=%d"` so plumbing failures are obvious in `tt-xla-eval` logs.

No new compile_options field required — A.1.a is unconditional once `argumentTypeMap` is set. To disable for negative-control benches, set `enable_const_eval=False` via `set_custom_compile_options`.

Verification:
- Re-dump IR with `export_path` (mechanism from Phase 0.2). Find the `shlo_compiler_*.mlir` file via glob (`ls /tmp/dump/irs/shlo_compiler_*.mlir` — note `/irs/` subdir). Confirm it shows `ttcore.argument_type = #ttcore<argument_type parameter>` on weight block args.
- Confirm the `ttnn_*.mlir` file (glob `<export_path>/irs/ttnn_*.mlir`) contains a `consteval_<fn>` wrapper function (per `TTNNPrepareConstEvalCaching.cpp:41`).
- Confirm **BFP8 cast lives inside the consteval wrapper.** Pipeline order in `TTNNPipelines.cpp` is: `ConstEvalHoist` at line 309 → `CPUHoistConstEvalTransform` at line 315 → `TTNNWeightDtypeConversion` at line 339 → `ConstEvalHoist` again at line 372 → `ConstEvalHoist` again at line 386. The line-372 re-invocation (with explicit comment about catching const-eval-able ops created by weight-dtype-conversion) means BFP8 casts on hoisted weights should themselves be hoisted. **If the dump shows BFP8 cast OUTSIDE the consteval wrapper, file as a real bug** — the pipeline order is supposed to handle this and a failure here is a tt-mlir bug worth reporting upstream (not a workaround). **Note about `enable_const_eval_on_cpu`**: defaults to `true` (`compile_options.h:90`). When ON, the line-314 `CPUHoistConstEvalTransform` lifts const-eval subgraphs to CPU execution, where they run FP32 not BFP8 (CPU doesn't have BFP8 ops). For the BFP8 memory win we want, weights must NOT be CPU-hoisted. Either set `enable_const_eval_on_cpu=False` via `set_custom_compile_options`, OR verify in the dump that the relevant subgraphs stayed on device.
- Tracy probe: weight tilizes should drop. To attribute Tilize ops to weights vs activations, diff Tilize **count** in the aggregated Tracy CSV before/after, and correlate with shapes from the `ttnn.mlir` dump (weight matmul shapes are known from `hf_model.config`). Expected: weight-tilize fraction of total Tilize count drops to near zero.
- Server bench: TPOT should drop. Magnitude depends on Phase 0.1 weight-tilize share.
- Negative control: bench with `enable_const_eval=False` via `torch_xla.set_custom_compile_options({"enable_const_eval": False})`, confirm 132 ms baseline restored.

Commits: tt-xla side (`module_builder.cc` change) lives in `/home/mhnie/tt-xla/pjrt_implementation/` — since `/home/mhnie/tt-xla/` is a host clone of canonical tt-xla, the diff lives as a local patch; sync to `/home/mhnie/tt-mlir-sglang/` if a fork branch is later set up for it. For now, save the diff under `python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch` in sglang and `git apply` from the container. Sglang commit captures the patch + bench fixture.

### Phase 3 — A.2.a (extended ToLayoutOp fold) — conditional on Phase 0.3

**Reframed (review 6 finding).** `ttnn.tilize` / `ttnn.untilize` / `ttnn.tilize_with_val_padding` are NOT MLIR ops. The TTNN dialect only has `ttnn.to_layout` (per `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`); the runtime kernels `TilizeWithValPadding` etc. are dispatched by tt-metal when each `to_layout` op executes. So the MLIR pass must match `to_layout` ops, not named tilize/untilize ops. `TTNNDecomposeLayouts.cpp` confirms: after decompose, the IR still contains `to_layout` ops — they're just simpler (layout-only, separated from dtype/memory-config conversions). Each surviving `to_layout` lowers to one runtime kernel dispatch.

If Phase 0.3 reports ≥ 50 candidate fold targets per decode:

Implementation:
- New file `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Transforms/TTNNFoldThroughAgnosticOps.cpp`.
- Pattern matches `ttnn.to_layout(allowlist_op(ttnn.to_layout(x, L1)), L0)` and rewrites to `allowlist_op(x)` when L0 == original layout of x and the allowlist op is genuinely layout-agnostic for the dtype involved. The existing `foldConsecutiveToLayoutOp` at `TTNNOps.cpp:2062` already handles the case where the two `to_layout` ops are directly adjacent; this new pattern handles the non-adjacent case with an allowlist op between them.
- **Allowlist restricted to unary ops only**: `ttnn.relu`, `ttnn.gelu`, `ttnn.silu`, `ttnn.typecast` (when output dtype is layout-preserving). Binary ops like `ttnn.add`/`multiply`/`subtract` (extending `TTNN_ElementwiseBinaryOp`) require BOTH operands at the same layout (encoded in the tensor type's `TTNNLayoutAttr`). The single-input fold `to_layout(add(to_layout(x, L1), y), L0) → add(x, ?)` leaves the second operand at L1 while the first is at L0, breaking the type contract. Binary-op folding is open work — requires either (a) recursively folding both operand chains symmetrically, or (b) inserting a compensating `to_layout` on the second operand (which may not be a net win). For revision 1, restrict to unary; document binary as a follow-up.
- Mirror the DRAM↔L1 safety guard from `foldConsecutiveToLayoutOp` (`TTNNOps.cpp:2078–2089`): don't fold when source op stages to DRAM and the eventual consumer expects L1 (or vice versa).
- Register pass via `Passes.td` (in `include/ttmlir/Dialect/TTNN/Transforms/Passes.td`).
- Wire into `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` inside the helper `createTTNNPipelineLayoutDecompositionPass` (defined at line 249): add `pm.addPass(createTTNNFoldThroughAgnosticOps());` immediately after the existing `pm.addPass(createTTNNDecomposeLayouts());` at line 251. The helper is invoked from the top-level pipeline at line 407 (inside `devicePm`), so placement inside the helper ensures the new fold runs in every context where layout decomposition runs.

Verification:
- Re-dump IR via `export_path`. Glob for the ttnn-stage file (`<export_path>/irs/ttnn_*.mlir`). Re-run Phase 0.3 counting script. Confirm `to_layout`-allowlist_op-`to_layout` triple count drops.
- Tracy probe: total Tilize+Untilize+Typecast kernel count should drop. **The drop is NOT necessarily count × 2** — one MLIR `ttnn.to_layout` may lower to one OR several runtime kernels depending on dtype/memory-config (TilizeWithValPadding + Typecast etc.). Treat the Tracy delta as directional, not numeric.
- Server bench: TPOT should drop. Numeric expectation depends on the activation-tilize share from Phase 0.1.

Risks:
- Allowlist too aggressive → wrong layout reaches a kernel that secretly assumes one layout. Mitigation: bench accuracy test (greedy-only, 10 prompts, byte-exact vs Phase 1 baseline) is a release gate.
- Pass placement wrong → other passes re-insert redundant layout kernels after this fold runs. Mitigation: dump IR after each pipeline pass with `--mlir-print-ir-after-all` (set via env var or extra pass option in pjrt-plugin-tt) and verify the fold's output survives.
- The existing `foldConsecutiveToLayoutOp` may catch cases the new pattern thought it needed to handle — verify by inspecting IR before/after the existing fold runs in pipeline.

### Phase 4 — A.3.a (auto-detect parameter markers in tt-mlir)

Phase 4 takes the classifier logic implemented in Phase 2's `classifyArgs` helper and moves/duplicates it into a tt-mlir pass that runs without a pjrt-side `argumentTypeMap`. The host moves from `pjrt_implementation/module_builder.cc` (per-compile, before pipeline) to `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPass.cpp` (or a new pass in `StableHLOPipelines.cpp` scheduled before `tt-populate-argument-types`).

Why not `AnalyzeMesh.cpp`: `AnalyzeMesh.cpp:120–132` emits errors when a block arg lacks `ArgumentTypeAttr`. It is a *consumer* of the annotation, not a producer. Extending it would change its semantics and contract.

Implementation:
- Add a new small pass `populateArgumentTypesAutoDetect` in `lib/Dialect/StableHLO/Transforms/` (or fold into `StableHLOToTTIRPass.cpp` if simpler).
- For each `func::FuncOp`: classify block args using the same A→B→C strategy as Phase 2 (Strategy A: per-arg marker; B: module-level cluster marker; C: heuristic). Strategy A→B priority: scan all block args once; if ANY block arg carries a per-arg marker, use Strategy A for the entire function and ignore module-level cluster markers (do not mix); otherwise look for module-level cluster marker (Strategy B); otherwise fall to heuristic (Strategy C). Mixing two markers in one function: emit a warning and fall through to Strategy C (graceful degrade, not abort).
- **Target function scope.** Like Phase 2, restrict the auto-detect pass to the **unique non-private `func::FuncOp`** of the module (every torch_xla compile emits one such function; the inliner may collapse helpers into it but the entry function itself is stable). If multiple non-private funcs are present, log a warning and process none — downstream pipeline runs with no auto-detect entries for that module.
- For each classified `Parameter` arg, set the function's arg attr at `ttcore.argument_type = #ttcore<argument_type parameter>`. **Skip args that already carry `ArgumentTypeAttr`** (don't overwrite Phase 2's explicit map).
- Schedule the new pass in `StableHLOPipelines.cpp` AFTER the existing `createTTPopulateArgumentTypes` invocation (`StableHLOPipelines.cpp:24`), and BEFORE the `createAnalyzeMeshPass` invocation (`StableHLOPipelines.cpp:52`). Order in the pipeline: `tt-populate-argument-types → populateArgumentTypesAutoDetect → analyze-mesh`. **Reason for after-populate:** `tt-populate-argument-types` at `PopulateArgumentTypes.cpp:286–313` unconditionally OVERWRITES `ArgumentTypeAttr` whenever the map provides a value for that function (it doesn't merge; it emits a warning and replaces). If auto-detect ran first, every entry it set would be clobbered by Phase 2's explicit map. Running auto-detect AFTER (and short-circuiting on existing attr) means: explicit map wins for mapped functions; auto-detect fills in unmapped functions or args. **Reason for before-analyze-mesh:** `AnalyzeMesh.cpp:120–132` is gated on `automaticArgAnalysis` (line 353); today tt-xla doesn't enable it, but if a future compile path does, `AnalyzeMesh` errors on un-annotated args — auto-detect must populate before it runs.

A.3.b heuristic — concrete rules (used in Strategy C above, both for Phase 2 and Phase 4 — implement once, share):

For each block arg of every `func::FuncOp` in the module, classify as `Parameter` iff ALL of:
1. **Shape gate**: tensor rank ≥ 1 AND **every** dim ≥ 16 OR rank == 1 with that dim ≥ 64. (Tightened from "at least one dim ≥ 64" — excludes attention_mask shaped `[1, 1, seq, seq]` which has size-1 dims.)
2. **Use-pattern allowlist** — every use of the arg is one of the following (or transitively reaches one of the following through layout-only ops `stablehlo.transpose`, `stablehlo.reshape`, `stablehlo.broadcast_in_dim`, `stablehlo.convert` to depth ≤ 3):
   - Operand of `stablehlo.dot_general` (matmul: q/k/v/o projections, gate/up/down projections).
   - Operand of `stablehlo.convolution`.
   - Operand of `stablehlo.gather` (embedding tables — Qwen3-8B's largest weight, vocab×hidden ≈ 152k×4096).
   - Operand of `stablehlo.multiply` or `stablehlo.add` when the other operand is NOT another block arg, AND the arg's use chain through this op does NOT reach a `stablehlo.exponential`, `stablehlo.reduce(..., max)`, or `stablehlo.reduce(..., add)` followed by a divide (the softmax fingerprint — excludes attention_mask, which is added to scores before softmax).
   - Operand of `stablehlo.dynamic_slice` or `stablehlo.slice` (rotary cos/sin tables sliced by current position).
3. **No mutating uses**: arg is never operand of any op that produces a result aliasing the arg (no `stablehlo.scatter` writing back to the arg, no in-place updates).
4. **Not in dynamic-input position**: arg's first dim is not used as a batch dim by any `stablehlo.dynamic_slice(arg, [batch_index, ...])`. (Excludes input_ids, position_ids, attention_mask if it dodged Rule 1, cache_pos, and kv-cache args.)
5. **Softmax-path exclusion** (belt-and-suspenders for attention_mask): the arg is rejected (stays `Input`) iff there exists a forward use chain from the arg to a `stablehlo.exponential` op such that the chain contains at least one `stablehlo.add`, `stablehlo.subtract`, or `stablehlo.multiply` consuming a value derived from the arg as an operand. The classifier walks consumers (BFS) from the arg. **"Depth" = number of `stablehlo.*` ops on the path counting ONLY non-layout-only ops** — `stablehlo.transpose`, `stablehlo.reshape`, `stablehlo.broadcast_in_dim`, `stablehlo.convert`, `stablehlo.slice` are NOT counted toward depth, but they are traversed. Depth-bound = **8** (chosen to safely cover the HF Qwen3 attention path `arg → broadcast → add(scores, mask) → reduce(max) → subtract → exponential` which has 4 non-layout ops; 8 gives ≥2× headroom for HF-emitter variations). If the BFS reaches `stablehlo.exponential` AND the path-to-exponential includes at least one of the add/sub/multiply ops (with arg-lineage on one operand and any value on the other), the arg fails Rule 5. Worklist size is capped at 256 ops per arg to avoid pathological compile times; if cap is hit, fail closed (reject the arg = stays Input).

Implement as an analysis with worklist over uses; for Rule 2 use 3 hops through layout-only ops; for Rule 5 use depth 8 as defined above with a 256-op cap. False positives are critical (mark a runtime-varying arg as Parameter → silent corruption). False negatives are tolerable (arg stays Input → no const-eval gain, no regression). The rules above are biased toward false negatives.

**Where the classifier runs.** Phase 2's classifier runs in `module_builder.cc` BEFORE pipeline construction, so it sees the module as torch_xla emitted it (pre-inliner). `tt-populate-argument-types` runs AFTER `StableHLOPipelines.cpp:20` inliner. If the inliner changes arg counts for any function in Phase 2's map, the pass aborts via `PopulateArgumentTypes.cpp:240–248 signalPassFailure`. Mitigation: the classifier emits map entries only for the **unique non-private `func::FuncOp`** in the module (torch_xla emits one such function per compile; tt-xla calls compile separately for prefill vs decode, so each module has exactly one). If the module contains zero or multiple non-private funcs, log a warning and emit no map entries (downstream Phase 4 auto-detect handles those modules).

**Strategy C visibility caveat.** Strategy C (heuristic) running in `module_builder.cc` sees PRE-inliner IR. If torch_xla emits the entry function as a thin wrapper that calls helpers (`func.call @inner(...)` rather than inlined matmul/gather/etc.), the use-chain walk from each block arg terminates at `func.call` without reaching `stablehlo.dot_general` — Strategy C returns "all Input" for that function. In that case Phase 2 emits no useful map; Phase 4's auto-detect (which runs POST-inliner in `StableHLOPipelines.cpp` after line 20) sees the inlined ops and handles classification. Phase 0.2 should also note whether torch_xla emits a flat or wrappered entry; this informs whether Strategy C is even viable in Phase 2.

**Behaviour on Strategy A/B/C mixed input.** If a function carries BOTH per-arg markers AND a module-level cluster marker (rare; some torch_xla emitter versions only emit one), the classifier emits a warning to `module_builder.cc` log and falls through to Strategy C (heuristic). It does NOT abort compile. Aborting on a build-artifact mismatch would turn graceful degradation into a hard outage.

**Mandatory mini-test for the heuristic before Phase 2 ships:** the classifier must be unit-tested against a small synthetic MLIR module containing: (a) an embedding-gather pattern, (b) an RMSNorm-multiply pattern, (c) an attention add-mask-then-softmax pattern, (d) a dot_general matmul. Expected: a,b,d → Parameter; c (the mask) → Input.

- Test path: `/home/mhnie/tt-mlir-sglang/test/ttmlir/Conversion/StableHLOToTTIR/auto_detect_argument_types.mlir` (matches existing fixtures' depth/style under `test/`).
- Format: lit test with `// RUN: ttmlir-opt --populate-argument-types-auto-detect %s | FileCheck %s`, with `// CHECK:` directives asserting `ttcore.argument_type = #ttcore<argument_type parameter>` on the four target args and `ttcore.argument_type = #ttcore<argument_type input>` (or absence) on the mask. Expected fixture size: ≈ 80–120 lines based on existing test sizes.
- Write this BEFORE wiring the auto-detect pass into the live pipeline. If the heuristic can't pass this, fix it before measuring TPOT.

Verification:
- Run server bench WITHOUT the Phase 2 `argumentTypeMap` set. To do so, plumb an env var `TT_DISABLE_PJRT_ARG_TYPE_MAP=1` in Phase 2 implementation that, when set, skips assigning `argumentTypeMap`. Verify const-eval still fires via `shlo_compiler.mlir` dump (weight args still carry `ttcore.argument_type = parameter`).
- **Coverage check**: parse the ttnn-stage dump (glob `<export_path>/irs/ttnn_*.mlir`) after auto-detect, count block args marked `parameter` and total block args expected to be weights (from `len([p for p in hf_model.named_parameters() if p[1].numel() > 1000])`). Coverage should be ≥ 95% — the embedding table and rotary tables in particular must be marked. If coverage misses a large-tensor category, expand the Rule 2 allowlist.
- Run server bench WITH Phase 2 `argumentTypeMap` also set. Behaviour should be identical (idempotent — the explicit map from Phase 2 overrides, but the values match).
- Negative control: a synthetic test function with only dynamic inputs (e.g., 2 args, both used as `stablehlo.dot_general` LHS via batch dim 0). Confirm neither is marked.

### Phase 5 — Final reporting (~30 min)

- Update `docs/platforms/tt_xla_tpot_handoff_2026-05-17.md` with a new TPOT table. Mirror the existing format at lines 11–22 (columns: Config | TPOT warm | tok/s | vs baseline). Add one row per landed phase.
- Strike or annotate the handoff doc's "Open work — concrete next steps" section (lines 155–185) which still describes the OLD framing of A.1 as "pre-tilize from Python", A.2 as "add a Tilize(Untilize) fold pattern", and A.3 as a StableHLOToTTIR-only change. Add a note at the top of that section: "SUPERSEDED — see `docs/superpowers/specs/2026-05-17-tt-xla-tilize-attack-design.md` for the actual landed implementation."
- Note any patches that needed re-applying after container restart.
- Always update memory `tenstorrent-tt-xla-tilize-bottleneck.md` with which attacks landed, measured TPOT, and residual bottleneck.
- Always update memory `tenstorrent-tt-mlir-sglang-fork.md` if any new commits were added.
- Update memory `tenstorrent-tt-xla-tpot-workstream-a.md` to correct the prior claim that the script "built B.2 v3 into the canonical wheel" — Phase 1's reality check found the script's `-DTTMLIR_SOURCE_DIR_OVERRIDE` was dead code; the working build was a separate manual operation.
- Write a NEW memory only if Phase 0 surfaced a finding worth saving for future sessions (e.g., an unexpected pipeline ordering, a marker name worth remembering, a layout-agnostic-op behavior that's not what the type system claimed).

## 6. Verification cadence

Per phase: Tracy probe → server bench → commit. ~30 min per loop. Isolates which change moved which metric and keeps fixtures aligned with commits.

**All bench commands must include `BYPASS_PREWARM=1` as an environment variable** (NOT a CLI flag — it's read at server startup; handoff doc Pitfall #4 — required on the rebuilt plugin). Tracy probe also requires it. Cache mode: keep `SGLANG_TT_CACHE_MODE=index_copy` for consistency with the shipped baseline.

Tracy probe: `probe_decode_op_profile.py` with **`--decode 5`** (handoff doc Pitfall #2 — each step adds 30 MB to the trace; >5 risks container OOM).

Server bench: `bench_3run_server_alive.py --backend tt_xla --models Qwen3-8B --input-len 1024 --output-len 1024 --out-tag <phase>`.

Fixtures saved alongside commit: `_fixtures/v146_3run_server_q8b_<phase>.json`.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Phase 1 CMake edit doesn't take (build still pulls canonical tt-mlir) | Hard verification in Phase 1 (commit SHA from `pip3 show pjrt-plugin-tt` + symbol grep in libTTMLIR*.so). If still canonical, the `SOURCE_DIR` override didn't apply — re-check whether `${TTPJRT_SOURCE_DIR}/third_party/tt-mlir` was pre-populated correctly or whether ExternalProject still cached the old clone. Re-bisect from a clean `build/` dir. |
| Phase 1 smoke-test regresses TPOT or breaks build (any of 5 fork patches culpable, not just B.2 v3) | Bisect by reverting fork commits individually starting from HEAD. If B.2 v3 is the offender, `git revert 2fc1d119e` and document. |
| A.1.a `argumentTypeMap` doesn't reach the pass (e.g., field name typo) | Phase 2 verification dumps IR and greps for `ttcore.argument_type`; the INFO log line in `module_builder.cc` reports strategy + counts. |
| `enable_const_eval` defaults change in a future tt-mlir bump | Explicitly set `compile_options.enable_const_eval = true` in `module_builder.cc` if the existing default is removed; surface as a build break, not a silent regression. |
| Phase 0.2 finds no per-arg marker → Phase 2 must implement Strategy C heuristic too | Acknowledged in Phase 2 — the classifier is implemented unconditionally as a shared helper, with Strategy C as the fallback path. Same helper is reused in Phase 4. |
| A.3.b heuristic misses large-but-non-matmul weights (embeddings, RMSNorm, rotary tables) | Rule 2 allowlist in Phase 4 explicitly covers `stablehlo.gather` (embeddings), `stablehlo.multiply`/`add` with non-block-arg counterpart (RMSNorm/LayerNorm), and `stablehlo.dynamic_slice`/`slice` (rotary). Coverage check in Phase 4 verification asserts ≥ 95% of expected-weight args are marked. |
| Plugin rebuild breaks (CMake or link error) after a phase commit | Each phase commit captures a working state; revert the offending change in tt-mlir-sglang or tt-xla patch, re-bisect. |
| Tracy OOM on a new pattern | Stick with `--decode 5`. Bench is the source of truth; Tracy is a structural-correctness check, not a numeric one. |
| Phase 3 pass-placement wrong: other passes re-insert redundant layout kernels after the fold | Dump IR after every pipeline pass with `--mlir-print-ir-after-all` (enable via pjrt-plugin-tt CLI option or env var); reorder fold pass placement until output survives. |
| `set_custom_compile_options` Python-bool serialization mismatch | Not a real risk: `compile_options.cc:parseBoolOption` (lines 95–96) calls `::tolower` before matching, so `"True"` → `"true"` → accepted. Documented here in case a future tt-xla version drops the tolower call. |
| Phase 4 auto-detect runs BEFORE `tt-populate-argument-types` → explicit map overwrites auto-detected attrs every compile (with noisy per-arg warnings) | Spec now schedules auto-detect AFTER `tt-populate-argument-types` AND short-circuits on existing attr. Verified via reading `PopulateArgumentTypes.cpp:286–313` (unconditional overwrite + warning). |
| Heuristic Rule 2 bullet 4 false-positives on attention_mask (mask + scores → mask marked Parameter → silent corruption) | Spec now adds Rule 5 (softmax-path exclusion: if dataflow reaches `stablehlo.exponential` via additive/multiplicative op, arg stays Input) AND tightens Rule 1 to require all dims ≥ 16 (mask has size-1 dims). Mandatory mini-test verifies before live use. |
| ExternalProject stale stamps from prior canonical build cause Phase 1 rebuild to silently use old checkout | Phase 1 step 2: explicit `rm -rf` of stamps + source dir + build dirs before rebuild. |
| Phase 2 strategy A and B both produce values; mixing is undefined | Spec now mandates: A short-circuits on first per-arg marker hit; B only consulted when ZERO block args carry per-arg markers in that function; classifier asserts on mixed input. |
| Phase 2 silently clobbers JAX-emitted `ttcore.argument_type` attrs (set by `annotateArgumentAttributes` at `module_builder.cc:463`) | Phase 2 classifier short-circuits per-function when any block arg already carries `ArgumentTypeAttr`. Spec: §5 Phase 2 step 1 includes the check. |
| Strategy C heuristic invisibility: torch_xla emits wrappered entry func (call → inner), so pre-inliner IR has no inlined matmul/gather/etc. visible | Phase 0.2 reports whether the entry func is flat or wrappered. If wrappered, Phase 2 emits only Strategy A/B; Strategy C work falls through to Phase 4 (post-inliner). Not a descope; just a phase-split clarification. |
| `enable_const_eval_on_cpu=true` lifts weight subgraphs to CPU (FP32) — defeats BFP8 memory win | Phase 2 verification: explicitly set `enable_const_eval_on_cpu=False` via `set_custom_compile_options` OR confirm in dump that weight subgraphs stayed on device. |
| Phase 3 fold pattern doesn't safely handle binary ops | Phase 3 allowlist restricted to unary ops only for first revision. Binary asymmetric folding tracked as follow-up. |
| Phase 1 `TT_METAL_RUNTIME_ROOT` points at deleted canonical path | Phase 1 step 1 removes the hardcode at lines 36–39 and requires the script to export the env var explicitly to the fork's tt-metal install location. |
| Phase 1 ExternalProject_Add re-builds the fork with different CMake args than the script does | Phase 1 step 1 sets `CONFIGURE_COMMAND ""`, `BUILD_COMMAND ""` to neuter ExternalProject_Add to a passive reference; the script's pre-build supplies all artifacts. |
| Classifier emits map entries for inlined/internal funcs whose arg count changes after inliner → `tt-populate-argument-types` aborts | Classifier emits map entries only for the top-level entry function. Document this constraint; warn if multiple top-level funcs are observed. |

## 8. Estimated wall-clock

- Phase 0: ~1.5 hours (0.1 ≈ 30 min, 0.2 ≈ 45 min including `export_path` wire-up + Python-bool round-trip, 0.3 ≈ 15 min)
- Phase 1: ~3 hours (CMakeLists.txt edit + ExternalProject_Add neutering + TT_METAL_RUNTIME_ROOT redirect + build-script edit + stamp/dir cleanup + initial fork build from scratch + plugin rebuild + verify SHA-diff + smoke bench). Bumped from 2h: fork's `build/` dir doesn't exist today, so first run includes a full tt-mlir build (~30–60 min) before any tt-xla rebuild.
- Phase 2: ~4 hours (≈ 1.5h C++ classifier with all three strategies + 1h mini-test scaffolding and execution + 30 min rebuild + 1h bench/verify/commit). The mini-test (mandatory before live use) is what bumped this.
- Phase 3: ~5 hours (only if Phase 0.3 ≥ 50 cancellable pairs). Bumped from 3h after review 6 reframed the pass to match `to_layout` patterns rather than non-existent named tilize/untilize ops; new logic must coexist with the existing `foldConsecutiveToLayoutOp` without duplicating its work.
- Phase 4: ~2 hours (most logic is reused from Phase 2 classifier; new work is the schedule-in-StableHLOPipelines + short-circuit-on-existing-attr + verification under `TT_DISABLE_PJRT_ARG_TYPE_MAP=1`)
- Phase 5: ~30 min

Total: ~11 hours minimum (Phase 0 + 1 + 2 + 5, no Phase 3/4), ~16 hours maximum (all phases).

## 9. Open items to confirm in writing-plans

- Naming for the new fold pass — confirm against existing TTNN pass-naming conventions (`TTNNFoldRedundantLayoutKernels` is a placeholder).
- Phase 3 layout-agnostic-op allowlist: define the `(dtype × layout)` compatibility predicate concretely. tile layout supports bf16/bfp8/fp32; row-major has dtype restrictions on Blackhole. The fold pass must table-drive this.
- Phase 2 shared classifier: decide whether the C++ helper lives in pjrt-plugin-tt (closer to caller, no tt-mlir rebuild needed when tweaking) or in tt-mlir as a library function (reused directly by Phase 4). The plan picks one location.
- Phase 0.2 candidate markers to grep for in StableHLO dumps (best-effort list, not exhaustive): `mhlo.parameter_replication`, `mhlo.is_same_data_across_replicas`, `mhlo.layout_mode`, `_xla_buffer_placement`, `tf.aliasing_output`, `tf.entry_function`, `jax.arg_info`, `torch.placeholder`. Treat as a starter list — the executor must additionally grep for ANY `mhlo.`/`xla.`/`tf.`/`jax.`/`torch.`/`_xla` -prefixed arg attr in the dumped IR. Record all hits; do not report "no marker found" without searching the broader prefix space.
