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

Phase 0 contains explicit measurement gates that **may** legitimately de-scope Phase 3 — but only on the specific quantitative criterion stated in the Phase 0 table (≥ 50 cancellable pairs). Phase 4's fallback to A.3.b is also explicitly authorized. Any other descope requires the user's approval.

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

**4.4 IR dumping is a built-in compile_options field, not a research item.** `compile_options.h:121` defines `std::optional<std::string> export_path`. When set, `ModuleBuilder::printModule` emits stage IR files: `vhlo`, `shlo`, `shlo_frontend`, `shlo_compiler`, `ttir`, `ttnn` (calls at `module_builder.cc:262, 270, 276, 315, 351, 357, 432, 452, 465, 843`). Trigger via `torch_xla.set_custom_compile_options({"export_path": "/tmp/dump"})`. Use for Phase 0.2 (block-arg attrs in `shlo`) and Phase 0.3 (kernel-pair count in `ttnn`).

**4.5 The TTNN pipeline already runs `ConstEvalHoist` three times, sandwiching weight-dtype-conversion.** `TTNNPipelines.cpp:309, 372, 386` invoke `createConstEvalHoistTransform()`; `TTNNWeightDtypeConversion` runs at line 339 between the first two. The comment around line 365–370 explicitly states the second invocation exists to "pick up any const-evalable ops created in weight dtype conversion." This means the BFP8-cast-outside-consteval risk is already mitigated by the pipeline design; if the dump shows otherwise, it's a real bug, not an expected limitation.

## 5. Phase plan

### Phase 0 — Investigations (no rebuilds, ~1.5 hours total)

Phase 0 is a gate, not a formality. The Phase 0.3 quantitative threshold is the ONLY automatic de-scope trigger in this plan.

| Step | Action | Mechanism | Output | Gate |
|---|---|---|---|---|
| 0.1 | Bucket existing Tracy CSV Tilizes by sibling-op context (weight tilize vs activation tilize) | Re-run `tracy_aggregate.py` on existing fixture, augment with column joining each Tilize to its consumer op's tensor shape; weights have known shapes from `hf_model.named_parameters()` | % split | If weights ≪ 5 %, document in Phase 5 and proceed (no descope — A.1.a/A.3 still land, just with lower expected gain) |
| 0.2 | Dump tt-xla StableHLO + TTIR for one decode step. Inspect block-arg attrs | Set `torch_xla.set_custom_compile_options({"export_path": "/tmp/shlo_dump", "enable_const_eval": True})` before warm-up. Run probe with `--decode 1`. Files appear under `/tmp/shlo_dump/{vhlo,shlo,shlo_frontend,shlo_compiler,ttir,ttnn}.mlir`. Grep `shlo.mlir` for `mhlo.`/`tf.`/`jax.`/`torch.`-prefixed arg attrs | Confirmed torch_xla marker name(s) on block args, OR "no per-arg marker" | "No per-arg marker" → activate A.3.b fallback in Phase 4 (heuristic). NOT a descope. |
| 0.3 | Count redundant layout kernel pairs in TTNN IR | Open `/tmp/shlo_dump/ttnn.mlir` from 0.2. Grep `ttnn.tilize`, `ttnn.untilize`, `ttnn.tilize_with_val_padding`. Count `untilize → ≤2 layout-agnostic ops → tilize` triples (and the reverse) | Cancellable-pair count | If < 50 per decode step, de-scope A.2.a (Phase 3 → skipped). Document count in Phase 5. |

If 0.2 also produces a clean per-arg parameter marker, capture an arg attr example for the Phase 4 implementation.

### Phase 1 — Repoint tt-xla third_party at the fork (~1 hour)

`USE_CUSTOM_TT_MLIR_VERSION` does NOT skip cloning. `third_party/CMakeLists.txt:7–9` only controls which SHA gets assigned to `TT_MLIR_VERSION`. The actual cloning happens unconditionally:
- Toolchain build path (`if (TOOLCHAIN STREQUAL "ON")` at lines 13–27) clones via `execute_process`.
- Normal path (`else()` at lines 28–95) clones via `ExternalProject_Add` at lines 47–85, which has `GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git` + `GIT_TAG ${TT_MLIR_VERSION}` hard-coded.

**Reality check up front: the fork is NOT currently the live build despite appearances.** `build_and_install.sh:38` passes `-DTTMLIR_SOURCE_DIR_OVERRIDE="$TTMLIR_DIR"` but no CMakeLists.txt in `/home/mhnie/tt-xla/` actually reads this variable (`grep -rn TTMLIR_SOURCE_DIR_OVERRIDE /home/mhnie/tt-xla/` returns only the stale `build-local/CMakeCache.txt`). The current live `pjrt-plugin-tt` reports `tt-mlir-commit=f3ddbfb6` (canonical). Memory `tenstorrent-tt-xla-tpot-workstream-a`'s claim "B.2 v3 patch built into canonical wheel" reflects a separate manual operation, not anything this script automated. After Phase 1 succeeds, update that memory entry.

Concrete mechanism (must edit CMakeLists.txt, clean stamps, and fix the build script):

1. Edit `/home/mhnie/tt-xla/third_party/CMakeLists.txt`. In the `ExternalProject_Add(tt-mlir …)` block at lines 47–85:
   - Remove `GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git` (line 82) and `GIT_TAG ${TT_MLIR_VERSION}` (line 83).
   - Add `SOURCE_DIR /home/mhnie/tt-mlir-sglang` and `DOWNLOAD_COMMAND ""`.
   - Redirect `TTMLIR_BUILD_DIR` (currently `${TTMLIR_SOURCE_DIR}/src/tt-mlir/build` at line 35) to `/home/mhnie/tt-mlir-sglang/build` so the fork's existing build tree is reused; OR keep it as-is and accept that two separate build dirs are maintained. Pick redirection — fewer rebuilds.
   - If the toolchain path at lines 13–27 is exercised in our build, replace its `git clone …` with `ln -sfn /home/mhnie/tt-mlir-sglang ${PROJECT_SOURCE_DIR}/tt-mlir/src/tt-mlir`.
2. Clean stale stamps from the prior canonical build BEFORE rebuilding: `rm -rf /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir-stamp /home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir /home/mhnie/tt-xla/build-local /home/mhnie/tt-xla/build`. ExternalProject's stamp tracking sees `SOURCE_DIR` differing from prior stamps and may behave unpredictably otherwise.
3. Remove the dead `-DTTMLIR_SOURCE_DIR_OVERRIDE="$TTMLIR_DIR"` flag from `build_and_install.sh:38` (it has no effect; leaving it is misleading evidence).
4. Save the CMakeLists.txt edit and the build-script edit as checked-in patches under `/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-source-dir-fork.patch` and `.../patches/tt-mlir-sglang-build-script-cleanup.patch` so they can be re-applied after container restart.
5. Rebuild via `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh` (lives in tt-mlir-sglang, NOT tt-xla, committed at `95d656c8f`).
6. **Hard verification that the live build is the fork, not canonical:**
   - Run `pip3 show pjrt-plugin-tt | grep tt-mlir-commit=` in the `tt-xla-eval` container. Must show `2fc1d119e` (fork HEAD), NOT `f3ddbfb6` (canonical pin). If still `f3ddbfb6`, the edit didn't take — STOP and investigate (do NOT proceed to Phase 2 against a canonical-tt-mlir-backed plugin).
   - Additionally grep one of the fork-specific commit identifiers in the built library: `strings $(python3 -c 'import pjrt_plugin_tt, os; print(os.path.dirname(pjrt_plugin_tt.__file__))')/lib/libTTMLIR*.so | grep -E "B\.2.*v3|tenstorrent-p1"` — should match at least one fork-specific symbol/string.
7. Smoke test: Qwen3-8B BFP8 server bench with `BYPASS_PREWARM=1 SGLANG_TT_CACHE_MODE=index_copy` (per Pitfall #4 — required for the rebuilt plugin). Expect ≈ 132 ms TPOT, no regression.

This re-activates **all 5 fork patches** that are ahead of canonical `f3ddbfb6` (B.2 v3 `2fc1d119e`, B.2 v2 `e62086947`, link tweak `4899ff291`, distributed-disable `8eb2e5b46`, build script `95d656c8f`). B.2 v3 is documented as working in memory `tenstorrent-tt-mlir-sglang-fork`; the other 4 are build-system tweaks expected to be neutral. If smoke regresses TPOT or build breaks: bisect by reverting fork commits individually.

### Phase 2 — A.1.a (set argumentTypeMap in module_builder.cc)

The central design choice in Phase 2 is **the argument classifier**: how to decide which block args are `Input` vs `Parameter` for the lowered HLO function. Phase 0.2's outcome determines which of three strategies to use, in priority order:

**Strategy A (used if Phase 0.2 finds per-arg marker).** Read the per-arg torch_xla marker name found in Phase 0.2 (e.g., `mhlo.parameter_replication`, `jax.arg_info`, or whatever shows up). For each block arg, if the marker says "parameter" → `ArgumentType::Parameter`; else `Input`.

**Strategy B (used if Phase 0.2 finds a module-level cluster marker).** Some HLO emitters mark parameters at module level via an array attr like `mhlo.parameter_replication = [false, false, true, true, true]`. Parse the array, map 1:1 onto block-arg indices.

**Strategy C (used if Phase 0.2 finds no marker).** Use a **classifier identical to A.3.b** (see Phase 4 for the rule set), applied to the incoming MLIR module before pipeline construction. The classifier walks each `func::FuncOp`, examines each block arg's use chain, and decides `Parameter` vs `Input`. **Phase 2 must implement this classifier even if Strategy A/B suffice**, because Phase 4 will reuse it; isolate it into a shared C++ helper to avoid code duplication.

Implementation (in `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc`):

1. Implement `classifyArgs(mlir::ModuleOp module) -> TTArgumentTypeMap` (or call a helper in tt-mlir if more natural). Internally, it tries Strategy A → B → C in order; returns the map keyed by function symbol name.
2. Assign `stablehlo_pipeline_options.argumentTypeMap = classifyArgs(mlir_module);` at `module_builder.cc:831` (immediately after `stablehlo_pipeline_options` is constructed; reviewer verified 831 is the right insertion point).
3. Assign `options.argumentTypeMap = classifyArgs(mlir_module);` at `module_builder.cc:985` (the variable in that block is named `options`, of type `TTIRToTTNNCommonPipelineOptions`; reviewer verified 985 is the right insertion point).
4. Existing `compile_options.enable_const_eval` (default `true` at `inc/api/compile_options.h:83`) handles the rest: `tt-populate-argument-types` populates `ttcore.argument_type` on block args, `ConstEvalHoistTransform` fires (three invocations at `TTNNPipelines.cpp:309, 372, 386`), `TTNNPrepareConstEvalCaching` + `TTNNConstEvalInputsToSystemMemory` close the loop.
5. Add an INFO log line `"[TT-XLA] argumentTypeMap: strategy=<A|B|C>, function=<name>, K_inputs=%d, K_params=%d"` so plumbing failures are obvious in `tt-xla-eval` logs.

No new compile_options field required — A.1.a is unconditional once `argumentTypeMap` is set. To disable for negative-control benches, set `enable_const_eval=False` via `set_custom_compile_options`.

Verification:
- Re-dump IR with `export_path` (mechanism from Phase 0.2). Confirm `shlo_compiler.mlir` shows `ttcore.argument_type = #ttcore<argument_type parameter>` on weight block args.
- Confirm `ttnn.mlir` contains a `consteval_<fn>` wrapper function (per `TTNNPrepareConstEvalCaching.cpp:41`).
- Confirm **BFP8 cast lives inside the consteval wrapper.** Pipeline order in `TTNNPipelines.cpp` is: `ConstEvalHoist` at line 309 → `TTNNWeightDtypeConversion` at line 339 → `ConstEvalHoist` again at line 372 → `ConstEvalHoist` again at line 386. The line-372 re-invocation (with explicit comment about catching const-eval-able ops created by weight-dtype-conversion) means BFP8 casts on hoisted weights should themselves be hoisted. **If the dump shows BFP8 cast OUTSIDE the consteval wrapper, file as a real bug** — the pipeline order is supposed to handle this and a failure here is a tt-mlir bug worth reporting upstream (not a workaround).
- Tracy probe: weight tilizes should drop. To attribute Tilize ops to weights vs activations, diff Tilize **count** in the aggregated Tracy CSV before/after, and correlate with shapes from the `ttnn.mlir` dump (weight matmul shapes are known from `hf_model.config`). Expected: weight-tilize fraction of total Tilize count drops to near zero.
- Server bench: TPOT should drop. Magnitude depends on Phase 0.1 weight-tilize share.
- Negative control: bench with `enable_const_eval=False` via `torch_xla.set_custom_compile_options({"enable_const_eval": False})`, confirm 132 ms baseline restored.

Commits: tt-xla side (`module_builder.cc` change) lives in `/home/mhnie/tt-xla/pjrt_implementation/` — since `/home/mhnie/tt-xla/` is a host clone of canonical tt-xla, the diff lives as a local patch; sync to `/home/mhnie/tt-mlir-sglang/` if a fork branch is later set up for it. For now, save the diff under `python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch` in sglang and `git apply` from the container. Sglang commit captures the patch + bench fixture.

### Phase 3 — A.2.a (post-decomp fold pass) — conditional on Phase 0.3

If Phase 0.3 reports ≥ 50 cancellable pairs per decode:

Implementation:
- New file `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Transforms/TTNNFoldRedundantLayoutKernels.cpp`.
- Pattern matches `tilize → layout-agnostic-op* → untilize` and the symmetric `untilize → … → tilize` where the intervening ops can equivalently run on the source layout. Layout-agnostic = same-shape elementwise ops AND ops that ttnn supports natively in both tile and row-major.
- Allowlist (start conservative): `ttnn.add`, `ttnn.multiply`, `ttnn.subtract`, `ttnn.relu`, `ttnn.gelu`, `ttnn.silu`, `ttnn.typecast` (when output dtype is layout-preserving). Expand only after measuring stable.
- Mirror the DRAM↔L1 safety guard from `foldConsecutiveToLayoutOp` (`TTNNOps.cpp:2078–2089`): don't fold when source op stages to DRAM and the eventual consumer expects L1 (or vice versa).
- Register pass via `Passes.td` (in `include/ttmlir/Dialect/TTNN/Transforms/Passes.td`).
- Wire into `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` inside the helper `createTTNNPipelineLayoutDecompositionPass` (defined at line 249): add `pm.addPass(createTTNNFoldRedundantLayoutKernels());` immediately after the existing `pm.addPass(createTTNNDecomposeLayouts());` at line 251. The helper is invoked from the top-level pipeline at line 407 (inside `devicePm`), so placement inside the helper ensures the new fold runs in every context where layout decomposition runs. Verify by reading any other pass added inside the helper after the decompose call (today: none — decompose is the last op there).

Verification:
- Re-dump `ttnn.mlir` via `export_path`. Re-run Phase 0.3 counting script. Confirm cancellable-pair count drops.
- Tracy probe: total Tilize+Untilize op count should drop by approximately the cancelled-pair count × 2.
- Server bench: TPOT should drop. Numeric expectation depends on the activation-tilize share from Phase 0.1.

Risks:
- Allowlist too aggressive → wrong layout reaches a kernel that secretly assumes one layout. Mitigation: bench accuracy test (greedy-only, 10 prompts, byte-exact vs Phase 1 baseline) is a release gate.
- Pass placement wrong → other passes re-insert redundant layout kernels after this fold runs. Mitigation: dump IR after each pipeline pass with `--mlir-print-ir-after-all` (set via env var or extra pass option in pjrt-plugin-tt) and verify the fold's output survives.

### Phase 4 — A.3.a (auto-detect parameter markers in tt-mlir)

Phase 4 takes the classifier logic implemented in Phase 2's `classifyArgs` helper and moves/duplicates it into a tt-mlir pass that runs without a pjrt-side `argumentTypeMap`. The host moves from `pjrt_implementation/module_builder.cc` (per-compile, before pipeline) to `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPass.cpp` (or a new pass in `StableHLOPipelines.cpp` scheduled before `tt-populate-argument-types`).

Why not `AnalyzeMesh.cpp`: `AnalyzeMesh.cpp:120–132` emits errors when a block arg lacks `ArgumentTypeAttr`. It is a *consumer* of the annotation, not a producer. Extending it would change its semantics and contract.

Implementation:
- Add a new small pass `populateArgumentTypesAutoDetect` in `lib/Dialect/StableHLO/Transforms/` (or fold into `StableHLOToTTIRPass.cpp` if simpler).
- For each `func::FuncOp`: classify block args using the same A→B→C strategy as Phase 2 (Strategy A: per-arg marker; B: module-level cluster marker; C: heuristic). Strategy A→B priority: scan all block args once; if ANY block arg carries a per-arg marker, use Strategy A for the entire function and ignore module-level cluster markers (do not mix); otherwise look for module-level cluster marker (Strategy B); otherwise fall to heuristic (Strategy C). Mixing two markers in one function is undefined and the classifier must assert.
- For each classified `Parameter` arg, set the function's arg attr at `ttcore.argument_type = #ttcore<argument_type parameter>`. **Skip args that already carry `ArgumentTypeAttr`** (don't overwrite Phase 2's explicit map).
- Schedule the new pass in `StableHLOPipelines.cpp` AFTER the existing `createTTPopulateArgumentTypes` invocation. Order in the pipeline: `tt-populate-argument-types → populateArgumentTypesAutoDetect`. **Reason:** `tt-populate-argument-types` at `PopulateArgumentTypes.cpp:286–313` unconditionally OVERWRITES `ArgumentTypeAttr` whenever the map provides a value for that function (it doesn't merge; it emits a warning and replaces). If auto-detect ran first, every entry it set would be clobbered by Phase 2's explicit map. Running auto-detect AFTER (and short-circuiting on existing attr) means: explicit map wins for mapped functions; auto-detect fills in unmapped functions or args.

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
5. **Softmax-path exclusion** (belt-and-suspenders for attention_mask): the arg's forward dataflow (chase consumers up to depth 6 through any non-side-effecting op) must NOT pass through `stablehlo.exponential` while still flowing as an additive/multiplicative operand. If it does, the arg is a softmax input (mask or scaling factor that varies per request) and stays `Input`.

Implement as an analysis with worklist over uses; depth-bound at 3 hops through layout-only ops for Rule 2, depth 6 for Rule 5. False positives are critical (mark a runtime-varying arg as Parameter → silent corruption). False negatives are tolerable (arg stays Input → no const-eval gain, no regression). The rules above are biased toward false negatives.

**Mandatory mini-test for the heuristic before Phase 2 ships:** the classifier must be unit-tested against a small synthetic MLIR module containing: (a) an embedding-gather pattern, (b) an RMSNorm-multiply pattern, (c) an attention add-mask-then-softmax pattern, (d) a dot_general matmul. Expected: a,b,d → Parameter; c (the mask) → Input. Write this as a `lit` test or a pjrt-plugin-tt unit test before wiring into the live build. If the heuristic can't pass this, fix it before measuring TPOT.

Verification:
- Run server bench WITHOUT the Phase 2 `argumentTypeMap` set. To do so, plumb an env var `TT_DISABLE_PJRT_ARG_TYPE_MAP=1` in Phase 2 implementation that, when set, skips assigning `argumentTypeMap`. Verify const-eval still fires via `shlo_compiler.mlir` dump (weight args still carry `ttcore.argument_type = parameter`).
- **Coverage check**: parse the `ttnn.mlir` dump after auto-detect, count block args marked `parameter` and total block args expected to be weights (from `len([p for p in hf_model.named_parameters() if p[1].numel() > 1000])`). Coverage should be ≥ 95% — the embedding table and rotary tables in particular must be marked. If coverage misses a large-tensor category, expand the Rule 2 allowlist.
- Run server bench WITH Phase 2 `argumentTypeMap` also set. Behaviour should be identical (idempotent — the explicit map from Phase 2 overrides, but the values match).
- Negative control: a synthetic test function with only dynamic inputs (e.g., 2 args, both used as `stablehlo.dot_general` LHS via batch dim 0). Confirm neither is marked.

### Phase 5 — Final reporting (~30 min)

- Update `docs/platforms/tt_xla_tpot_handoff_2026-05-17.md` with a new TPOT table (post-Phase-1, post-A.1.a, post-A.2.a if landed, post-A.3.a if landed). Note any patches that needed re-applying after container restart.
- Always update memory `tenstorrent-tt-xla-tilize-bottleneck.md` with which attacks landed, measured TPOT, and residual bottleneck.
- Always update memory `tenstorrent-tt-mlir-sglang-fork.md` if any new commits were added.
- Write a NEW memory only if Phase 0 surfaced a finding worth saving for future sessions (e.g., an unexpected pipeline ordering, a marker name worth remembering, a layout-agnostic-op behavior that's not what the type system claimed).

## 6. Verification cadence

Per phase: Tracy probe → server bench → commit. ~30 min per loop. Isolates which change moved which metric and keeps fixtures aligned with commits.

**All bench commands must include `BYPASS_PREWARM=1`** (handoff doc Pitfall #4 — required on the rebuilt plugin). Tracy probe also requires it. Cache mode: keep `SGLANG_TT_CACHE_MODE=index_copy` for consistency with the shipped baseline.

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
| `set_custom_compile_options` Python-bool serialization mismatch (`True` vs `"true"`) | Phase 0.2 sub-step: verify the round-trip by inspecting the parsed value via the existing INFO log line in `compile_options.cc:44–49`; if `parseBoolOption` rejects `"True"`, set `enable_const_eval` as a string `"true"` explicitly from Python. |
| Phase 4 auto-detect runs BEFORE `tt-populate-argument-types` → explicit map overwrites auto-detected attrs every compile (with noisy per-arg warnings) | Spec now schedules auto-detect AFTER `tt-populate-argument-types` AND short-circuits on existing attr. Verified via reading `PopulateArgumentTypes.cpp:286–313` (unconditional overwrite + warning). |
| Heuristic Rule 2 bullet 4 false-positives on attention_mask (mask + scores → mask marked Parameter → silent corruption) | Spec now adds Rule 5 (softmax-path exclusion: if dataflow reaches `stablehlo.exponential` via additive/multiplicative op, arg stays Input) AND tightens Rule 1 to require all dims ≥ 16 (mask has size-1 dims). Mandatory mini-test verifies before live use. |
| ExternalProject stale stamps from prior canonical build cause Phase 1 rebuild to silently use old checkout | Phase 1 step 2: explicit `rm -rf` of stamps + source dir + build dirs before rebuild. |
| Phase 2 strategy A and B both produce values; mixing is undefined | Spec now mandates: A short-circuits on first per-arg marker hit; B only consulted when ZERO block args carry per-arg markers in that function; classifier asserts on mixed input. |
| Empty `argumentTypeMap` produces a per-compile warning from `tt-populate-argument-types` | Cosmetic. Document in Phase 5 final reporting that the warning is benign and expected on functions where auto-detect alone handles classification. |

## 8. Estimated wall-clock

- Phase 0: ~1.5 hours (0.1 ≈ 30 min, 0.2 ≈ 45 min including `export_path` wire-up + Python-bool round-trip, 0.3 ≈ 15 min)
- Phase 1: ~1.5 hours (CMakeLists.txt edit + build-script edit + stamp/dir cleanup + rebuild + verify-fork-loaded + smoke). Bumped from 1h to account for the cleanup step and the dead-flag removal.
- Phase 2: ~4 hours (≈ 1.5h C++ classifier with all three strategies + 1h mini-test scaffolding and execution + 30 min rebuild + 1h bench/verify/commit). The mini-test (mandatory before live use) is what bumped this.
- Phase 3: ~3 hours (only if Phase 0.3 ≥ 50 cancellable pairs)
- Phase 4: ~2 hours (most logic is reused from Phase 2 classifier; new work is the schedule-in-StableHLOPipelines + short-circuit-on-existing-attr + verification under `TT_DISABLE_PJRT_ARG_TYPE_MAP=1`)
- Phase 5: ~30 min

Total: ~9.5 hours minimum (Phase 0 + 1 + 2 + 5, no Phase 3/4), ~12.5 hours maximum (all phases).

## 9. Open items to confirm in writing-plans

- Naming for the new fold pass — confirm against existing TTNN pass-naming conventions (`TTNNFoldRedundantLayoutKernels` is a placeholder).
- Phase 3 layout-agnostic-op allowlist: define the `(dtype × layout)` compatibility predicate concretely. tile layout supports bf16/bfp8/fp32; row-major has dtype restrictions on Blackhole. The fold pass must table-drive this.
- Phase 2 shared classifier: decide whether the C++ helper lives in pjrt-plugin-tt (closer to caller, no tt-mlir rebuild needed when tweaking) or in tt-mlir as a library function (reused directly by Phase 4). The plan picks one location.
- Phase 4 mini-test placement: lit test under `tt-mlir-sglang/test/Conversion/StableHLOToTTIR/` vs pjrt-plugin-tt unit test under `tt-xla/pjrt_implementation/tests/`. Plan picks one.
- Phase 0.2 candidate markers to grep for in StableHLO dumps (prior to settling on Strategy A): `mhlo.parameter_replication`, `xla_hlo.parameter_buffer_assignments`, `tf.aliasing_output`, `tf.entry_function`, `jax.arg_info`, `torch.placeholder`, `mhlo.layout_mode`. None are guaranteed; Phase 0.2's job is to find what's actually present.
