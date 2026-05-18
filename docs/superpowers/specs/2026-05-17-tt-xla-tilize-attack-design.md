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

Genuine blockers (e.g., a build that won't link due to an upstream API rename) are surfaced as questions, not unilateral descope decisions.

Phase 0 contains explicit measurement gates that **may** legitimately de-scope Phase 3 / Phase 4 — but only on the specific quantitative criteria stated in the Phase 0 table. Any other descope requires the user's approval.

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
| Phase 1 wiring | `/home/mhnie/tt-xla` (host clone, detached at `470f0fad`) | `third_party/CMakeLists.txt:5–28`; set `USE_CUSTOM_TT_MLIR_VERSION=ON` and pre-populate `third_party/tt-mlir/src/tt-mlir/` via symlink to `/home/mhnie/tt-mlir-sglang/`. tt-mlir is NOT a git submodule — it's brought in via `ExternalProject_Add` with a clone-then-checkout step (skipped when `USE_CUSTOM_TT_MLIR_VERSION=ON`). |
| A.1.a | `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc` | Build a `TTArgumentTypeMap` and assign it to both `stablehlo_pipeline_options.argumentTypeMap` (StableHLO PM, around line 828–855) and `ttnn_pipeline_options.argumentTypeMap` (TTNN PM, around line 980). Pipeline-option fields already exist in `TTNNPipelines.h:319–334` and `StableHLOPipelines.h:47–61`. Plug-in source lives at `/home/mhnie/tt-xla/pjrt_implementation/{inc,src}/`. |
| A.2.a | `/home/mhnie/tt-mlir-sglang` (branch `tenstorrent-p1`) | new `lib/Dialect/TTNN/Transforms/TTNNFoldRedundantLayoutKernels.cpp` + `Passes.td` registration + `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` insertion (see Phase 3 for exact insertion point) |
| A.3.a | `/home/mhnie/tt-mlir-sglang` | either `lib/Dialect/StableHLO/Transforms/AnalyzeMesh.cpp` (preferred — already reads `ArgumentTypeAttr` at lines 128, 135–137, 158–160, 227 and has an `automaticArgAnalysis` path at line 353) OR `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPass.cpp`. Phase 4 prereq decides which. |
| Build/install | `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh` (committed at `95d656c8f`) | Rebuilds tt-mlir under the live tt-xla tree and reinstalls the pjrt plugin into `tt-xla-eval`. **Run from `tt-mlir-sglang`, NOT from `tt-xla`.** |
| Verification | `/home/mhnie/sglang` | bench/probe scripts already in `python/sglang/srt/hardware_backend/tenstorrent/test/` |

## 4. Key findings from spec-exploration

**4.1 The doc's "A.1 = pre-tilize from Python" is not implementable.** `ttnn` is a C++ device-side API not exposed on torch_xla. The spiritual successor is A.1.a: leverage the same compile-time const-eval pipeline that A.3 targets, but trigger it explicitly via the existing `argumentTypeMap` pipeline option rather than via auto-detection.

**4.2 The doc's "A.2 = add a Tilize(Untilize) fold pattern" is already implemented at the `to_layout` level.** `lib/Dialect/TTNN/IR/TTNNOps.cpp:2025–2114` has `foldIdentityToLayoutOp` and `foldConsecutiveToLayoutOp`. The Tilize bottleneck is downstream of these folds — after `TTNNDecomposeLayouts` expands `to_layout` into specific Tilize/Untilize/Typecast kernel calls. A.2.a addresses that post-decomp layer.

**4.3 A.3's const-eval infrastructure is fully wired.** `lib/Transforms/ConstEvalHoist.cpp:112` reads `ttcore::getConstsAndParams()` which filters block args by `ArgumentTypeAttr ∈ {Parameter, Constant}` (defined in `include/ttmlir/Dialect/TTCore/IR/Utils.h:70`). The CLI pass `tt-populate-argument-types` (defined in `lib/Dialect/TTCore/Utils/PopulateArgumentTypes.cpp:118`) accepts an `argument-types` option (key from `include/ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h:18 OptionNames::argumentTypes`) shaped as `function=input,parameter,parameter,constant`. The TTNN side closes the loop via `TTNNPrepareConstEvalCaching` + `TTNNConstEvalInputsToSystemMemory`. Both `TTNNPipelines.h:319` and `StableHLOPipelines.h:47` already declare an `argumentTypeMap` field that pjrt-plugin-tt can assign.

The gate is whether anyone sets `ArgumentTypeAttr` on the lowered func's block args. Today nothing does for tt-xla-compiled functions (`grep argumentTypeMap` in `/home/mhnie/tt-xla/pjrt_implementation/` returns zero hits).

**4.4 IR dumping is a built-in compile_options field, not a research item.** `compile_options.h:121` defines `std::optional<std::string> export_path`. When set, `ModuleBuilder::printModule` emits stage IR files: `vhlo`, `shlo`, `shlo_frontend`, `shlo_compiler`, `ttir`, `ttnn` (calls at `module_builder.cc:262, 270, 276, 315, 351, 357, 432, 452, 465, 843`). Trigger via `torch_xla.set_custom_compile_options({"export_path": "/tmp/dump"})`. Use for Phase 0.2 (block-arg attrs in `shlo`) and Phase 0.3 (kernel-pair count in `ttnn`).

## 5. Phase plan

### Phase 0 — Investigations (no rebuilds, ~1.5 hours total)

Phase 0 is a gate, not a formality. The Phase 0.3 quantitative threshold is the ONLY automatic de-scope trigger in this plan.

| Step | Action | Mechanism | Output | Gate |
|---|---|---|---|---|
| 0.1 | Bucket existing Tracy CSV Tilizes by sibling-op context (weight tilize vs activation tilize) | Re-run `tracy_aggregate.py` on existing fixture, augment with column joining each Tilize to its consumer op's tensor shape; weights have known shapes from `hf_model.named_parameters()` | % split | If weights ≪ 5 %, document in Phase 5 and proceed (no descope — A.1.a/A.3 still land, just with lower expected gain) |
| 0.2 | Dump tt-xla StableHLO + TTIR for one decode step. Inspect block-arg attrs | Set `torch_xla.set_custom_compile_options({"export_path": "/tmp/shlo_dump", "enable_const_eval": True})` before warm-up. Run probe with `--decode 1`. Files appear under `/tmp/shlo_dump/{vhlo,shlo,shlo_frontend,shlo_compiler,ttir,ttnn}.mlir`. Grep `shlo.mlir` for `mhlo.`/`tf.`/`jax.`/`torch.`-prefixed arg attrs | Confirmed torch_xla marker name(s) on block args, OR "no per-arg marker" | "No per-arg marker" → activate A.3.b fallback in Phase 4 (heuristic). NOT a descope. |
| 0.3 | Count redundant layout kernel pairs in TTNN IR | Open `/tmp/shlo_dump/ttnn.mlir` from 0.2. Grep `ttnn.tilize`, `ttnn.untilize`, `ttnn.tilize_with_val_padding`. Count `untilize → ≤2 layout-agnostic ops → tilize` triples (and the reverse) | Cancellable-pair count | If < 50 per decode step, de-scope A.2.a (Phase 3 → skipped). Document count in Phase 5. |

If 0.2 also produces a clean per-arg parameter marker, capture an arg attr example for the Phase 4 implementation.

### Phase 1 — Repoint tt-xla third_party at the fork (~45 min)

`/home/mhnie/tt-xla/third_party/CMakeLists.txt:5` defines `option(USE_CUSTOM_TT_MLIR_VERSION ... OFF)`. When ON, the clone-and-checkout block at lines 7–28 is skipped, and the build uses whatever already exists at `/home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir/`.

Concrete mechanism:
1. From `/home/mhnie/tt-xla`, run `cmake -B build -DUSE_CUSTOM_TT_MLIR_VERSION=ON -DTT_MLIR_VERSION=<fork-HEAD-sha>` (or wire into the existing CMake invocation in `scripts/build_and_install.sh`).
2. Replace `/home/mhnie/tt-xla/third_party/tt-mlir/src/tt-mlir/` with a symlink to `/home/mhnie/tt-mlir-sglang/`. Back up the old directory under `.../tt-mlir/src/tt-mlir.canonical-bak/` for fast revert.
3. Rebuild via `/home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh`. Note: this script lives in tt-mlir-sglang, NOT in tt-xla.
4. Smoke test: Qwen3-8B BFP8 server bench with `BYPASS_PREWARM=1 SGLANG_TT_CACHE_MODE=index_copy` (per Pitfall #4 of the handoff doc — required for the rebuilt plugin). Expect ≈ 132 ms TPOT, no regression.

This re-activates **all 5 fork patches** that are ahead of canonical `f3ddbfb6` (B.2 v3 `2fc1d119e`, B.2 v2 `e62086947`, link tweak `4899ff291`, distributed-disable `8eb2e5b46`, build script `95d656c8f`). Per memory `tenstorrent-tt-xla-tpot-workstream-a` and `tenstorrent-tt-mlir-sglang-fork`, B.2 v3 has been built into the canonical wheel and known to work; the other 4 are build-system tweaks expected to be neutral or strictly better. If smoke regresses TPOT or build breaks: bisect by reverting fork commits individually.

### Phase 2 — A.1.a (set argumentTypeMap in module_builder.cc)

Implementation (in `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc`):

1. Build a `TTArgumentTypeMap` from the HLO function signature. For tt-xla → torch_xla, the convention is: the first K args are dynamic inputs (input_ids, attention_mask, position_ids, cache_pos, kv caches), the remainder are static parameters (model weights). K depends on the wrapped HF model's forward signature. Use the per-arg marker found in Phase 0.2 if available; otherwise derive K from `compile_options` metadata or from arg-attr inspection on the incoming module.
2. Build the map keyed by the forward function's symbol name (e.g., `"main" -> SmallVector<ArgumentType>{Input, Input, …, Parameter, Parameter, …}`).
3. Assign `stablehlo_pipeline_options.argumentTypeMap = ttArgumentTypeMap;` around line 828–855 (where `stablehlo_pipeline_options` is constructed before invoking the StableHLO pipeline).
4. Assign `ttnn_pipeline_options.argumentTypeMap = ttArgumentTypeMap;` around line 980 (TTNN PM construction).
5. Existing `compile_options.enable_const_eval` (default `true`) handles the rest: `tt-populate-argument-types` populates `ttcore.argument_type` on block args, `ConstEvalHoistTransform` fires, `TTNNPrepareConstEvalCaching` + `TTNNConstEvalInputsToSystemMemory` close the loop.
6. Add an INFO log line `"[TT-XLA] argumentTypeMap set: K_inputs=%d, K_params=%d"` so plumbing failures are obvious in `tt-xla-eval` logs.

No new env var required — A.1.a is unconditional. If desired, expose a `disable_const_eval=true` escape hatch (just set `compile_options.enable_const_eval = false` from Python), but default behavior is "weights get const-evaled."

Verification:
- Re-dump IR with `export_path` (mechanism from Phase 0.2). Confirm `shlo_compiler.mlir` shows `ttcore.argument_type = #ttcore<argument_type parameter>` on weight block args.
- Confirm `ttnn.mlir` contains a `consteval_<fn>` wrapper function (per `TTNNPrepareConstEvalCaching.cpp:30–34`).
- Confirm **BFP8 cast lives inside the consteval wrapper** (not in the main forward). If cast is outside, the weight-dtype-conversion pass runs after const-eval-hoist and BFP8 weights aren't pre-quantized. Check pipeline order in `TTNNPipelines.cpp` — if `targetDtype` consumption (around line 338 per review) runs after `ConstEvalHoistTransform`, file as a follow-up but still ship A.1.a.
- Tracy probe: weight tilizes should drop. To attribute Tilize ops to weights vs activations, diff Tilize **count** in the aggregated Tracy CSV before/after, and correlate with shapes from the `ttnn.mlir` dump (weight matmul shapes are known from `hf_model.config`).
- Server bench: TPOT should drop. Magnitude depends on Phase 0.1 weight-tilize share.
- Negative control: bench with `enable_const_eval=False`, confirm 132 ms baseline restored.

Commits: tt-xla side (`module_builder.cc` change) lives in `/home/mhnie/tt-xla/pjrt_implementation/` — since `/home/mhnie/tt-xla/` is a host clone of canonical tt-xla, the diff lives as a local patch; sync to `/home/mhnie/tt-mlir-sglang/` if a fork branch is later set up for it. For now, save the diff under `python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch` in sglang and `git apply` from the container. Sglang commit captures the patch + bench fixture.

### Phase 3 — A.2.a (post-decomp fold pass) — conditional on Phase 0.3

If Phase 0.3 reports ≥ 50 cancellable pairs per decode:

Implementation:
- New file `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Transforms/TTNNFoldRedundantLayoutKernels.cpp`.
- Pattern matches `tilize → layout-agnostic-op* → untilize` and the symmetric `untilize → … → tilize` where the intervening ops can equivalently run on the source layout. Layout-agnostic = same-shape elementwise ops AND ops that ttnn supports natively in both tile and row-major.
- Allowlist (start conservative): `ttnn.add`, `ttnn.multiply`, `ttnn.subtract`, `ttnn.relu`, `ttnn.gelu`, `ttnn.silu`, `ttnn.typecast` (when output dtype is layout-preserving). Expand only after measuring stable.
- Mirror the DRAM↔L1 safety guard from `foldConsecutiveToLayoutOp` (`TTNNOps.cpp:2078–2089`): don't fold when source op stages to DRAM and the eventual consumer expects L1 (or vice versa).
- Register pass via `Passes.td` (in `include/ttmlir/Dialect/TTNN/Transforms/Passes.td`).
- Wire into `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` immediately after `TTNNDecomposeLayouts` (line ~251). Verify placement does NOT precede any pass between lines 251 and ~307 that re-inserts redundant layout kernels (read the passes in that range; reorder if needed).

Verification:
- Re-dump `ttnn.mlir` via `export_path`. Re-run Phase 0.3 counting script. Confirm cancellable-pair count drops.
- Tracy probe: total Tilize+Untilize op count should drop by approximately the cancelled-pair count × 2.
- Server bench: TPOT should drop. Numeric expectation depends on the activation-tilize share from Phase 0.1.

Risks:
- Allowlist too aggressive → wrong layout reaches a kernel that secretly assumes one layout. Mitigation: bench accuracy test (greedy-only, 10 prompts, byte-exact vs Phase 1 baseline) is a release gate.
- Pass placement wrong → other passes re-insert redundant layout kernels after this fold runs. Mitigation: dump IR after each pipeline pass with `--mlir-print-ir-after-all` (set via env var or extra pass option in pjrt-plugin-tt) and verify the fold's output survives.

### Phase 4 — A.3.a (auto-detect parameter markers)

**Prerequisite step (10 min, no rebuild): determine where the marker logic belongs.** Read `lib/Dialect/StableHLO/Transforms/AnalyzeMesh.cpp:128–174` and `:353–401`. If `AnalyzeMesh` already runs in the StableHLO pipeline AND `automaticArgAnalysis` can be triggered on standard tt-xla compiles, extend `AnalyzeMesh.cpp` to also mark `Parameter` (today it appears to focus on `BatchParallelism`). Otherwise put the logic in `StableHLOToTTIRPass.cpp` as originally planned.

Implementation (primary, A.3.a):
- Walk every `func::FuncOp` in the module. For each block arg, examine its arg-attr dict.
- If the arg carries the torch_xla parameter marker(s) found in Phase 0.2 (e.g., a specific `mhlo.parameter_replication` or `jax.arg_info` attr — record the exact name from Phase 0.2 here), set `ttcore.argument_type = #ttcore<argument_type parameter>`.
- If marker is per-arg, apply 1:1. If marker is module-level with an arg-index list, parse the list.

Fallback (A.3.b) — activated when Phase 0.2 finds no per-arg marker:
- Heuristic: for each block arg of every `func::FuncOp`, classify as `Parameter` iff ALL of:
  1. Tensor rank ≥ 2.
  2. Every use is as the static (non-batch) operand of `stablehlo.dot_general`, `stablehlo.convolution`, OR as an operand of an `stablehlo.transpose`/`stablehlo.reshape` whose sole use is one of the above.
  3. Argument is not in the "dynamic input" position (HLO convention: dynamic inputs typically have batch-like dim 0, indexed by `stablehlo.dynamic_slice` or similar — exclude args with such uses).
- Implement as a small analysis pass that walks each arg's use chain to fixed depth (3 hops). Mark only on positive confirmation.
- Treat false negatives as acceptable (worst case: arg stays as `Input` → no const-eval gain, no regression). Treat false positives as critical (would silently corrupt: a runtime-varying arg gets const-evaled). The "every use must be matmul/conv operand or layout-only transform thereof" rule guards against false positives.

Verification:
- Run server bench WITHOUT any user-supplied `argumentTypeMap` (i.e., revert the Phase 2 A.1.a flag plumbing temporarily, or set `enable_argument_type_map=false` if Phase 2 exposed such a knob). Const-eval should now fire automatically. Confirm via `shlo_compiler.mlir` dump that weight args carry `ttcore.argument_type = parameter`.
- Run server bench WITH A.1.a plumbing also on. Behaviour should be identical (idempotent — A.1.a's map wins via overwrite, but the values match).
- Negative control: run on a non-weight-carrying function (e.g., a tt-xla unit test compile of a function with all dynamic inputs). Confirm no args get falsely marked.

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
| Phase 1 smoke-test regresses TPOT or breaks build (any of 5 fork patches culpable, not just B.2 v3) | Bisect by reverting fork commits individually starting from HEAD. If B.2 v3 is the offender, `git revert 2fc1d119e` and document. |
| A.1.a `argumentTypeMap` doesn't reach the pass (e.g., field name typo) | Phase 2 verification dumps IR and greps for `ttcore.argument_type`; the INFO log line in `module_builder.cc` reports map construction. |
| `enable_const_eval` defaults change in a future tt-mlir bump | Explicitly set `compile_options.enable_const_eval = true` in `module_builder.cc` if the existing default is removed; surface as a build break, not a silent regression. |
| BFP8 weight-dtype-conversion pass runs AFTER `ConstEvalHoistTransform` → hoisted weights NOT quantized to BFP8 → memory stays high | Phase 2 verification explicitly inspects `ttnn.mlir` for whether `cast`/`typecast` to BFP8 lives inside the `consteval_<fn>` wrapper. If outside, file a follow-up to reorder passes in `TTNNPipelines.cpp` and ship A.1.a anyway (still a Tilize win, just not the BFP8 memory win). |
| A.3.a marker not found (Phase 0.2 produces no per-arg marker) | A.3.b heuristic activates (fleshed out in Phase 4). If A.3.b's accuracy gate fails (false positives detected), ship A.1.a only and document A.3 as "needs torch_xla upstream change to emit a marker." |
| Plugin rebuild breaks (CMake or link error) after a phase commit | Each phase commit captures a working state; revert the offending change in tt-mlir-sglang or tt-xla patch, re-bisect. |
| Tracy OOM on a new pattern | Stick with `--decode 5`. Bench is the source of truth; Tracy is a structural-correctness check, not a numeric one. |
| Phase 4 prereq surprise: `AnalyzeMesh` already marks `Parameter` for HF weights | If so, A.3 is mostly done — just verify it fires for tt-xla compiles. May reduce Phase 4 to a config tweak (e.g., enable `automaticArgAnalysis`). Not a descope — a simplification. |
| Phase 3 pass-placement wrong: other passes re-insert redundant layout kernels after the fold | Dump IR after every pipeline pass with `--mlir-print-ir-after-all` (enable via pjrt-plugin-tt CLI option or env var); reorder fold pass placement until output survives. |

## 8. Estimated wall-clock

- Phase 0: ~1.5 hours (0.1 ≈ 30 min, 0.2 ≈ 45 min including `export_path` wire-up, 0.3 ≈ 15 min)
- Phase 1: ~45 min (CMake config + symlink + rebuild + smoke)
- Phase 2: ~2.5 hours (≈ 1h C++ in `module_builder.cc` + 30 min rebuild + 1h bench/verify/commit)
- Phase 3: ~3 hours (only if Phase 0.3 justifies — ≥ 50 cancellable pairs)
- Phase 4: ~3 hours (≈ 1h prereq + C++ + 30 min rebuild + 1h bench/verify/commit; A.3.b path adds 30–60 min if needed)
- Phase 5: ~30 min

Total: ~6.5 hours minimum (Phase 0 + 1 + 2 + 5), ~10.5 hours maximum (all phases).

## 9. Open items to confirm in writing-plans

- Exact line numbers in `module_builder.cc` for the two `argumentTypeMap` assignments — review cited "around 828–855" and "around 980"; the implementation plan should pin them down by reading the current file.
- Naming for the new fold pass — confirm against existing TTNN pass-naming conventions (`TTNNFoldRedundantLayoutKernels` is a placeholder).
- Whether the symlink at `tt-mlir/src/tt-mlir/` should symlink the whole repo or just specific subdirs (e.g., `lib/`, `include/`) — depends on how `ExternalProject_Add`'s `BUILD_BYPRODUCTS` paths resolve. Decide in Phase 1 setup.
- Patch storage strategy for the tt-xla `module_builder.cc` change — drop under `python/sglang/srt/hardware_backend/tenstorrent/patches/tt-xla-argument-type-map.patch` and apply at container build, or maintain a `tt-xla-sglang` fork. The implementation plan picks one.
