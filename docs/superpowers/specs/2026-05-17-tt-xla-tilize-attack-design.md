# tt-xla Tilize-bottleneck attack — design

**Date:** 2026-05-17
**Status:** spec — pending user review
**Predecessor:** `docs/platforms/tt_xla_tpot_handoff_2026-05-17.md` (Priority A: A.1/A.2/A.3)

---

## 1. Goal

Close the Tilize bottleneck in tt-xla Qwen3-8B serving. Tracy on the shipped BFP8 + `index_copy_` config (132.1 ms TPOT) shows TilizeWithValPadding = 81 % of decode time, matmul = 1.8 %. Layout conversion dominates. Target: reduce TPOT toward the tt_transformers_paged reference (37.4 ms TPOT), via three independent attacks landed serially.

Success metric: each phase reports Tracy Tilize-op share and server-bench TPOT before/after. Aggregate goal is "closer to 37 ms"; no fixed numeric target per phase.

## 2. Scope

In scope:
- A.1.a — explicit `experimental_const_eval_weights` flag wired through pjrt-plugin-tt → tt-mlir's existing `populate-argument-types` pass → existing `ConstEvalHoistTransform`.
- A.2.a — new post-`TTNNDecomposeLayouts` fold pass for redundant Tilize/Untilize kernel sequences (only if Phase 0.3 investigation shows ≥ ~50 cancellable pairs per decode).
- A.3.a — auto-detect torch_xla parameter markers in `StableHLOToTTIRPass`, set `ArgumentTypeAttr::Parameter` on block args, so const-eval fires without user opt-in.

Out of scope:
- Multi-batch / bs=N optimizations (already landed in WS-A).
- B.* server-vs-probe gap (separate from Tilize attack).
- New model coverage.
- Upstream sgl-project/sglang PRs (per memory `feedback_no_upstream_pr`).

## 3. Repo layout this work touches

| Step | Repo | Files |
|---|---|---|
| Wiring | `/home/mhnie/tt-xla` (host clone) | `third_party/tt-mlir/src/tt-mlir/` → repointed at fork (one-time) |
| A.1.a | pjrt-plugin-tt (live in `tt-xla-eval` container) | option parser + pipeline pass-options wiring |
| A.2.a | `/home/mhnie/tt-mlir-sglang` | new `lib/Dialect/TTNN/Transforms/TTNNFoldRedundantLayoutKernels.cpp` + `Passes.td` registration + `TTNNPipelines.cpp` insertion |
| A.3.a | `/home/mhnie/tt-mlir-sglang` | `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPass.cpp` (and `StableHLOToTTIRPatterns.cpp` if needed) |
| Verification | `/home/mhnie/sglang` | bench/probe scripts already in `python/sglang/srt/hardware_backend/tenstorrent/test/` |

## 4. Key findings from spec-exploration

**4.1 The doc's "A.1 = pre-tilize from Python" is not implementable.** `ttnn` is a C++ device-side API not exposed on torch_xla. Real A.1 = "set a flag that triggers existing const-eval-hoist". Same destination as A.3, different trigger.

**4.2 The doc's "A.2 = add a Tilize(Untilize) fold pattern" is already implemented at the `to_layout` level.** `lib/Dialect/TTNN/IR/TTNNOps.cpp:2025–2114` has `foldIdentityToLayoutOp` and `foldConsecutiveToLayoutOp`. The Tilize bottleneck is downstream of these folds — after `TTNNDecomposeLayouts` expands `to_layout` into specific kernel calls. A.2.a addresses that post-decomp layer.

**4.3 A.3's const-eval infrastructure is fully wired.** `lib/Transforms/ConstEvalHoist.cpp:112` reads `ttcore::getConstsAndParams()` which filters block args by `ArgumentTypeAttr ∈ {Parameter, Constant}` (defined in `include/ttmlir/Dialect/TTCore/IR/Utils.h:70`). There is even a CLI pass-option `populate-argument-types` accepting `function=input,param,param,...` mappings. The TTNN side closes the loop via `TTNNPrepareConstEvalCaching` + `TTNNConstEvalInputsToSystemMemory`.

The gate for the whole const-eval machinery is whether anyone sets `ArgumentTypeAttr` on the lowered func's block args. Today nothing does for tt-xla-compiled functions.

## 5. Phase plan

### Phase 0 — Investigations (no rebuilds, ~1 hour total)

Phase 0 is a gate, not a formality. Each investigation can re-weight or de-scope the corresponding phase.

| Step | Action | Output | Gates |
|---|---|---|---|
| 0.1 | Bucket existing Tracy CSV Tilizes by sibling-op context (weight tilize vs activation tilize) | % split | If weights ≪ 5 %, de-emphasize A.1/A.3 and prioritize A.2 |
| 0.2 | Dump tt-xla StableHLO for one decode step. Inspect block-arg attrs | Confirmed torch_xla marker name (`mhlo.parameter_replication`, `tf.aliasing_output`, `mhlo.is_donated`, …) or "none found" | "None found" → fall back to A.3.b heuristic or skip A.3 |
| 0.3 | Dump TTNN IR after `TTNNDecomposeLayouts` on one decode step. Count redundant layout kernel pairs | Cancellable-pair count | If < 50, de-scope A.2.a |

### Phase 1 — Repoint third_party (~45 min)

Switch `/tt-xla/third_party/tt-mlir/src/tt-mlir/` to consume `/home/mhnie/tt-mlir-sglang/` directly (symlink the cloned source dir at the fork, or change git submodule URL).

This re-activates the fork's 6 patches in the live build, notably B.2 v3 (`2fc1d119e`). Per memory `tenstorrent-tt-xla-tpot-workstream-a`, B.2 v3 was built into the canonical wheel previously and worked, so this is expected to be neutral.

Rebuild plugin once via `scripts/build_and_install.sh` (committed at `95d656c8f`). Smoke test: Qwen3-8B BFP8 server bench, expect ≈ 132 ms TPOT (no regression).

### Phase 2 — A.1.a (explicit flag)

Implementation:
- In pjrt-plugin-tt option parser, accept new key `experimental_const_eval_weights` (bool).
- When set, generate a `populate-argument-types` mapping for each compiled func. Mapping is derived from XLA parameter conventions (e.g., the well-known split between dynamic inputs and static parameters in HLO's parameter list).
- Invoke the existing `populate-argument-types` pass with the mapping before the compilation pipeline reaches `ConstEvalHoist`.

Verification:
- Tracy probe with `SGLANG_TT_CONST_EVAL_WEIGHTS=1` → Tilize op count should drop noticeably (target: matmul-weight tilizes eliminated).
- Server bench → TPOT should drop. Magnitude depends on Phase 0.1 split.
- Negative control: bench without the env var, confirm pre-existing 132 ms baseline.

Commits: sglang (env-var wiring + bench script) and tt-mlir-sglang separately.

### Phase 3 — A.2.a (post-decomp fold pass) — conditional on Phase 0.3

If Phase 0.3 justifies (≥ ~50 cancellable pairs per decode):

Implementation:
- New file `lib/Dialect/TTNN/Transforms/TTNNFoldRedundantLayoutKernels.cpp`.
- Pattern matches `tilize → layout-agnostic-op* → untilize` and the symmetric `untilize → … → tilize` where the intervening ops can equivalently run on the source layout. Layout-agnostic = same-shape elementwise ops, or ops that ttnn supports natively in both tile and row-major.
- Register pass via `Passes.td`, schedule in `TTNNPipelines.cpp` after `TTNNDecomposeLayouts`.

Verification: same Tracy + bench cadence. Confirm cancellable-pair count drops in Phase 0.3 re-measurement.

Risks:
- Layout-agnostic-op classification may be too aggressive (some ops claim layout-agnosticism but assume tile or row-major in their kernel). Start with a conservative allowlist (`add`, `multiply`, `relu`-class), expand only if measured-stable.
- Cancellation may change memory residency (DRAM ↔ L1) and pressure other allocators. Mirror the existing `foldConsecutiveToLayoutOp` DRAM→L1 guard at lines 2078–2089.

### Phase 4 — A.3.a (auto-detect)

Prerequisite: Phase 0.2 produced a real torch_xla marker name.

Implementation:
- In `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPass.cpp`, when finalizing func.func conversion, walk arg attrs. For each block arg with the identified marker (e.g., a specific `mhlo.*` attr), set `ArgumentTypeAttr::Parameter`.
- If marker is per-arg, map 1:1. If marker is module-level with arg-index list, parse the list.

Verification:
- Run server bench WITHOUT the A.1.a opt-in flag. Const-eval should now fire automatically — confirm via Tracy that weight tilizes are gone in the steady-state decode.
- Run server bench WITH the A.1.a opt-in flag. Behaviour should be identical (idempotent — both routes set the same attribute).

Fallback if Phase 0.2 says "no marker found":
- A.3.b — heuristic: any block arg whose only uses are static operands to matmul/conv (no dynamic reshape, no indexing into it) → `Parameter`. Add sanity checks: arg must be 2D+, must not be in the dynamic-input position by HLO convention.
- If A.3.b is too lossy → skip A.3, document A.1.a as the only path.

### Phase 5 — Final reporting

- Update `docs/platforms/tt_xla_tpot_handoff_2026-05-17.md` with new TPOT table (post-A.1.a, post-A.2.a, post-A.3.a rows).
- Update memory `tenstorrent-tt-xla-tilize-bottleneck.md` with landed attacks and residual bottleneck (if any).
- Write new memory only if Phase 0 surfaced a surprising finding worth saving for future sessions.

## 6. Verification cadence (per user decision)

Per step: Tracy probe → server bench → commit. ~30 min per loop. Isolates which change moved which metric and keeps fixtures aligned with commits.

Tracy probe uses `probe_decode_op_profile.py` with `--decode 5–20` cap (handoff doc §"Pitfalls #1,2"). Server bench uses `bench_3run_server_alive.py --backend tt_xla --models Qwen3-8B --input-len 1024 --output-len 1024 --out-tag <phase>`.

Fixtures saved alongside commit: `_fixtures/v146_3run_server_q8b_<phase>.json`.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Phase 1 smoke-test regresses TPOT | Revert B.2 v3 in fork via `git revert 2fc1d119e`; document. |
| A.1.a flag plumbing fails (option doesn't reach pass) | Add log line in pjrt-plugin-tt option handler; verify in `tt-xla-eval` logs. |
| A.3 marker not found (Phase 0.2 fails) | Fall back to A.3.b heuristic. If lossy → A.1.a is the only path, document and ship. |
| Plugin rebuild breaks (CMake or link error) | Each phase commit captures a working state; revert in tt-mlir-sglang and re-bisect. |
| Tracy infeasible on a new pattern (OOM or 20-min cold compile) | Stick with `--decode 5` cap. Bench is the source of truth. |
| BFP8 + const-eval interaction unexpected (e.g., cast happens before hoist, weights end up tilized but not BFP8) | Inspect IR after const-eval-hoist; if cast lives outside the hoisted subgraph, file as a follow-up and ship A.1.a anyway (still a Tilize win). |

## 8. Estimated wall-clock

- Phase 0: 1 hour (3 investigations, no rebuild)
- Phase 1: 45 min (repoint + rebuild + smoke)
- Phase 2: ~2 hours (C++ + rebuild + bench + commit)
- Phase 3: ~3 hours (only if Phase 0.3 justifies)
- Phase 4: ~3 hours (only if Phase 0.2 produces a marker)
- Phase 5: 30 min

Total: ~6 hours minimum (Phase 0 + 1 + 2 + 5), ~10 hours maximum (all phases).

## 9. Open questions deferred to writing-plans

- pjrt-plugin-tt source location and exact option-parser file (need to confirm in the live container — `/tt-xla` clone has Python tests but plugin source may live elsewhere).
- Naming for `TTNNFoldRedundantLayoutKernels` pass — confirm against existing TTNN naming conventions when writing the file.
- Whether Phase 1 should symlink or change git submodule URL — needs `git status` on `/tt-xla` to see how third_party is currently configured.
