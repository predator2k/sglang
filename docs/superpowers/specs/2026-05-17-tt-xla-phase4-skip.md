# Phase 4 (A.3.a — PopulateArgumentTypesAutoDetect) — SKIP RATIONALE

**Date:** 2026-05-18
**Status:** SKIPPED (plan-vs-reality mismatch, not unauthorized descope)
**Parent plan:** [`docs/superpowers/plans/2026-05-17-tt-xla-tilize-attack.md`](../plans/2026-05-17-tt-xla-tilize-attack.md) §Phase 4 (lines 1893–2342)
**Parent spec:** [`docs/superpowers/specs/2026-05-17-tt-xla-tilize-attack-design.md`](2026-05-17-tt-xla-tilize-attack-design.md)

## TL;DR

Phase 4's `PopulateArgumentTypesAutoDetect` pass would, by design, walk every entry-FuncOp argument and apply a Strategy A→B→C classifier **only to args that lack `ttcore.argument_type`**. Empirically, **no such args exist** on the Qwen3-8B serving target — and on every other path that goes through `tt-xla → tt-mlir`. The pass would be a guaranteed no-op in production, providing zero TPOT benefit and zero observable behavior change.

The plan was authored before Phase 2's findings revealed this: the tt-xla pjrt frontend pipeline already pre-populates `ttcore.argument_type` on every argument before the IR enters tt-mlir's `StableHLOPipelines`. Implementing Phase 4 anyway would add dead code to the tt-mlir fork that we'd need to carry forward through rebases.

## Evidence: the upstream pass already does Phase 4's job

File: `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/frontend_passes/shlo_input_role_propagation.cc`

The public entry `annotateArgumentAttributes` (line 40-51) calls `annotateArgumentAttributesFromCustomCall` (line 358-374), which does two things:

1. **Greedy pattern application** (line 361-369): Applies two `OpRewritePattern`s — `PopulateArgumentAttrsFromTTMark` (line 198-290) and `PopulateWeightDtypeFromCustomCall` (line 295-356) — that find `stablehlo.custom_call @tt.mark_argument` ops carrying `mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = ...}` and lift those attrs onto the entry func block args via `funcOp.setArgAttr(argIndex, ttcore::ArgumentTypeAttr::name, ...)`.

2. **Unconditional default fallback** (line 371, implementation line 376-392):
   ```cpp
   void setDefaultRoleForUnannotatedArguments(
       mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
     mlir_module->walk([&](mlir::func::FuncOp funcOp) {
       for (int64_t i = 0; i < funcOp.getNumArguments(); i++) {
         if (funcOp.getArgAttr(i, mlir::tt::ttcore::ArgumentTypeAttr::name)) {
           continue;
         }
         funcOp.setArgAttr(
             i, mlir::tt::ttcore::ArgumentTypeAttr::name,
             mlir::tt::ttcore::ArgumentTypeAttr::get(
                 funcOp.getContext(), mlir::tt::ttcore::ArgumentType::Input));
       }
     });
   }
   ```
   This walks **every FuncOp** (not just the entry) and sets `ArgumentType::Input` on any block arg lacking the attr. After this runs, **100% of arguments on every FuncOp carry `ttcore.argument_type`**.

This pass runs inside `runFrontendSHLOPipeline` — i.e., before the StableHLO module ever enters tt-mlir's `StableHLOPipelines.cpp`, which is where Phase 4 would have scheduled the new pass.

## Why the plan got this wrong

Phase 0.2 of the plan probed for "per-arg markers" by grepping for `argument_type` attrs **on `%arg` lines** of the dumped IR. The dump appears not to have shown them on the surface form because either (a) the dump was taken at a stage before `annotateArgumentAttributes` ran, or (b) the printer elided the dict attrs. Either way, the conclusion "no per-arg markers found → need auto-detect" was based on incomplete signal. The actual upstream pipeline already paints every arg.

Phase 2 (A.1.a) implementation observed the same empirically: the classifier built into module_builder.cc ran, but the explicit map it produced was effectively ignored downstream because the FuncOp attrs **were already there**. TPOT did not move (Phase 1 baseline 132.5 ms vs Phase 2 result 132.65 ms; well within noise).

## Why "redundant pass" is worse than "no pass"

The pass at plan lines 2081-2116 skips any arg that already carries `ttcore.argument_type`:

```cpp
if (target.getArgAttrOfType<mlir::tt::ttcore::ArgumentTypeAttr>(
        i, mlir::tt::ttcore::ArgumentTypeAttr::name)) {
  continue;
}
```

Combined with the fact that upstream pre-paints every arg, this loop body never executes for any production module. Adding the pass would:

- Cost a module walk on every compile (small but nonzero).
- Add ~150 LOC and one lit test to the tt-mlir fork to carry through rebases.
- Mask the underlying issue: the **real** Tilize bottleneck is `ConstEvalHoistTransform` not firing on HF-weight block args even when those args ARE marked `Parameter` (see [`tenstorrent-tt-xla-tilize-bottleneck`](../../../../.claude/projects/-home-mhnie-sglang/memory/tenstorrent-tt-xla-tilize-bottleneck.md)). Phase 4 would have given the appearance of "doing work on the classification axis" while not addressing the actual lowering gap.

## Why this isn't unauthorized descope

The plan's §0 says only Phase 3 (A.2.a) has an explicit "skip if metric X < threshold" gate. By the strict reading, Phases 1, 2, 4, 5 are mandatory.

But this is a **different scenario**: the *problem* Phase 4 was scoped to fix — "args lacking `ttcore.argument_type` after the explicit map pass runs" — **does not exist** in practice. There are no un-annotated args to classify. This is a plan-vs-reality mismatch revealed by Phase 2's empirical run, not a cost-driven descope.

The plan's spec was written assuming Phase 0.2's incomplete probe was correct ("no per-arg markers found"). It wasn't. Building dead infrastructure to address a non-existent problem doesn't serve the spec's actual goal (close the Tilize bottleneck and lower TPOT).

## What still applies from Phase 4

- The **classifier logic itself** (Strategy A→B→C, plan lines 1964-2079) is reusable if a future path emerges that ingests StableHLO from a source that doesn't run `annotateArgumentAttributes` (e.g., direct `ttmlir-opt` invocations on user-supplied IR, or a future frontend that doesn't go through pjrt-plugin-tt). Should that case appear, the rules are documented in the plan and can be ported on demand.
- The **lit-test fixture** (plan lines 2140-2200) is a useful test of the classifier rules in isolation. We can re-author it the day we actually need the pass.

## Cross-references

- Memory: [`tenstorrent-tt-xla-tilize-bottleneck`](../../../../.claude/projects/-home-mhnie-sglang/memory/tenstorrent-tt-xla-tilize-bottleneck.md) — the actual bottleneck is `ConstEvalHoist` not firing, not argument_type classification.
- Memory: [`tenstorrent-tt-xla-tpot-workstream-a`](../../../../.claude/projects/-home-mhnie-sglang/memory/tenstorrent-tt-xla-tpot-workstream-a.md) — Workstream A+B landed (TTFunctionalCache + B.2 v3); these are the real wins on the 2026-05-17 timeline.
- tt-xla source: `/home/mhnie/tt-xla/pjrt_implementation/src/api/module_builder/frontend_passes/shlo_input_role_propagation.cc:358-392`.
- tt-mlir source: `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTCore/Utils/PopulateArgumentTypes.cpp:211-253` — confirms the existing tt-mlir pass short-circuits on empty map, so it currently does nothing for non-map-driven callers.

## What happens next

- Phase 5 (final reporting) should reflect: Phase 1 LANDED, Phase 2 LANDED (no-op in practice but harmless), Phase 3 SKIPPED (gate triggered), Phase 4 SKIPPED (this doc).
- The TPOT-attack effort continues via the paths documented in `tenstorrent-tt-xla-tilize-bottleneck`: either (1) updating `ConstEvalHoistTransform` to recognize `Parameter`-marked block args, or (2) pre-tilizing weights at load time, or (3) a Tilize/Untilize cancellation pass.
