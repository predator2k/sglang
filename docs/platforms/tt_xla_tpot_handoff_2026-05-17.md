# tt-xla TPOT optimization — handoff

**Date:** 2026-05-17 (session ran into 2026-05-18 03:00 UTC)
**Branch:** `tenstorrent-p1`
**Last commit:** `33496f196 perf(tt-xla): BFP8 weights + index_copy cache + Qwen3-8B 1.23× speedup`

---

## What this session achieved

**Server-path Qwen3-8B (1024 in / 1024 out, bs=1) TPOT progression:**

| Config | TPOT warm | tok/s | vs baseline |
|---|---|---|---|
| tt-xla BF16 + torch.where cache (original) | 162.6 ms | 6.15 | 1.00× |
| tt-xla BFP8 weights + torch.where | 140.2 ms | 7.13 | 1.16× |
| **tt-xla BFP8 weights + index_copy_ (shipped default)** | **132.1 ms** | **7.57** | **1.23×** |
| Phase 1 — fork rebuilt against canonical pin, no A.1.a/A.2.a/A.3.a | 132.52 ms | 7.55 | 1.23× (= baseline) |
| Phase 2 — A.1.a (`classifyArgs` + explicit `argumentTypeMap`) | 132.65 ms | 7.54 | 1.23× (no movement; within noise) |
| Phase 3 — A.2.a (to_layout-pair fold) | _SKIPPED_ | — | gate: 4 cancellable triples « 50 |
| Phase 4 — A.3.a (auto-detect pass) | _SKIPPED_ | — | upstream `annotateArgumentAttributesFromCustomCall` already paints every arg |
| tt_transformers_paged (production reference) | 37.4 ms | 26.7 | 4.35× |

**Tilize-attack outcome (Phase 0–4):** the classification-axis fixes (A.1.a, A.3.a) did not move TPOT because the upstream tt-xla pjrt frontend pass `annotateArgumentAttributesFromCustomCall` already lifts `tt.mark_argument` custom_calls into per-arg `ttcore.argument_type` attrs and falls back to `Input` for any arg missing the attr. `ConstEvalHoistTransform` already fires (88 wrappers observed) but the hoisted wrappers do not bake into compile-time tile-layout tensors — the residual Tilize bottleneck lives **downstream** of classification, not at it. Phase 0.1 also showed that **only 14.07% of Tilizes are on weight shapes** — the remaining ~86% are activation Tilizes, which no const-eval-hoist style fix can address. See `docs/superpowers/specs/2026-05-17-tt-xla-phase4-skip.md` and the new memory `tenstorrent-tt-xla-argument-type-already-set-upstream.md`.

**Probe-path (direct torch.compile, no sglang server) with same BFP8:** 91.1 ms TPOT. The 41 ms gap from server is host-side overhead, unexplained.

**Key finding from Tracy device profile:** TilizeWithValPadding is **81 %** of tt-xla decode time. Matmul is **1.8 %**. Compute is cheap; layout conversion dominates. This is where the remaining 2.5–3× gap lives.

---

## What's shipped

### Code changes (`tenstorrent-p1` HEAD `33496f196`)

| File | Change |
|---|---|
| `python/sglang/srt/hardware_backend/tenstorrent/models/tt_xla_model.py` | Sets `experimental_weight_dtype=bfp_bf8` via `torch_xla.set_custom_compile_options()` before `torch.compile`. Override with env var. |
| `python/sglang/srt/hardware_backend/tenstorrent/models/tt_functional_cache.py` | `SGLANG_TT_CACHE_MODE=index_copy` opt-in to bypass `torch.where` workaround and use parent `StaticCache` in-place path at q_len=1. |
| `python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_backend.py` | Replaced `LLAMA_DIR` env wiring with `HF_MODEL=model_path` so the (newer) tt-metal `model_config.py` accepts our local model dirs. |

### Knobs added

| Env var | Default | Purpose |
|---|---|---|
| `SGLANG_TT_WEIGHT_DTYPE` | `bfp_bf8` | `""` disables, `bfp_bf4` for max compression (greedy diverges on Qwen3-8B at token 1) |
| `SGLANG_TT_CACHE_MODE` | unset (uses torch.where) | `index_copy` to use parent `StaticCache.update` even at q_len=1 |
| `BYPASS_PREWARM` | unset | bench-only — skip the broken pre-warm hook on the rebuilt plugin |
| `BENCH_HEALTH_TIMEOUT_S` | 900 (in bench) | Extend server-ready wait. tt_transformers_paged needs ≥ 1200 for full warmup. |
| `TT_PROFILE` | unset | bench-only — enable TT_METAL_DEVICE_PROFILER on the launched server |

### New bench / probe scripts

- `python/sglang/srt/hardware_backend/tenstorrent/test/bench_3run_server_alive.py` — server-alive 3-condition bench (cold JIT / warm JIT new prompt / warm JIT + KV hit) across v145 models, backend-selectable (`--backend tt_xla|tt_transformers_single|tt_transformers_paged`)
- `python/sglang/srt/hardware_backend/tenstorrent/test/probe_decode_op_profile.py` — Tracy-compatible probe with `--weight-dtype` for BFP8/BFP4 sweep
- `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/tracy_aggregate.py` — aggregates `ops_perf_results_*.csv` by OP CODE using OP→OP LATENCY (DEVICE KERNEL DURATION column is unreliable on this build)

### Fixtures saved

`python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/`:
- `v146_3run_server_qwen3_8b_baseline.json` — tt-xla BF16 baseline
- `v146_3run_server_q8b_bfp8_server.json` — tt-xla BFP8 + torch.where
- `v146_3run_server_q8b_bfp8_ic.json` — tt-xla BFP8 + index_copy_ (shipped config)
- `v146_3run_server_qwen3_8b_paged2.json` — tt_transformers_paged reference

---

## What's where (containers + repos)

### Containers

| Container | Image | Use | Mounts |
|---|---|---|---|
| `tt-xla-eval` | `ghcr.io/tenstorrent/tt-xla-slim:latest` | tt-xla bench, probe, build | `/home/mhnie/sglang→/sglang`, `/home/mhnie/tt-xla→/tt-xla`, `/home/mhnie/tt-mlir-sglang→/tt-mlir-sglang`, `/opt/tt-mlir-toolchain` |
| `p3a-ngram` | `localhost/local-tt-metal:dev` | tt_transformers_paged bench | `/home/mhnie/tt-models→/models`, `/home/mhnie/sglang→/sglang` |

Both contain Qwen3-8B (`tt-xla-eval` pulls from HF cache, `p3a-ngram` uses local `/models/Qwen3-8B`).

### Plugin state (in `tt-xla-eval`)

- `pjrt-plugin-tt 0.1.260428+dev.470f0fad8` — my locally rebuilt editable install (matches canonical commit triple but build flags differ from the original `1.1.0` wheel; rebuild does not produce a pre-warm-clean plugin, so we set `BYPASS_PREWARM=1` for everything).
- Currently BOTH B.2 patches reverted in the cloned source (`/tt-xla/third_party/tt-mlir/src/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`). The fork at `/home/mhnie/tt-mlir-sglang/` still has the B.2 v3 patch committed; we just don't apply it in the live build because it was never needed for the BFP8 work.

### Repos

| Path | Branch | Notes |
|---|---|---|
| `/home/mhnie/sglang/` | `tenstorrent-p1` | Main repo. `33496f196` is the latest commit. Push only to local `origin` per memory. |
| `/home/mhnie/tt-xla/` | (detached at `470f0fad`) | Cloned-source tt-mlir (under `third_party/tt-mlir/src/tt-mlir`) is canonical `f3ddbfb6`. |
| `/home/mhnie/tt-mlir-sglang/` | `tenstorrent-p1` | Local tt-mlir fork. B.2 v3 at `2fc1d119e`. Not applied in current build. |
| `/home/mhnie/tt-metal-sglang/` | `tenstorrent-p1` (base `89686ee78d`) | 14 fork commits including Blackhole grid clamps. Required for Qwen3-8B in `p3a-ngram`. We `git apply`d a generated patch to the container's `/tt-metal/` during this session — patch may need re-applying if `p3a-ngram` is recreated. |

---

## Reproduction

### tt-xla BFP8 production (the shipped config)

```bash
docker exec -d tt-xla-eval bash -c 'cd /sglang && BYPASS_PREWARM=1 SGLANG_TT_CACHE_MODE=index_copy \
  python3 -u python/sglang/srt/hardware_backend/tenstorrent/test/bench_3run_server_alive.py \
  --models Qwen3-8B --backend tt_xla --input-len 1024 --output-len 1024 \
  --out-tag <tag> > /tmp/<tag>.log 2>&1'
```

Output: `_fixtures/v146_3run_server_<tag>.json` with run1_cold / run2_warm_new_prompt / run3_warm_kv_hit.

### tt_transformers reference

```bash
# In p3a-ngram, NOT tt-xla-eval. /models mount + fork patches required.
docker exec -d p3a-ngram bash -c 'cd /sglang && BENCH_HEALTH_TIMEOUT_S=1200 BYPASS_PREWARM=1 \
  python3 -u python/sglang/srt/hardware_backend/tenstorrent/test/bench_3run_server_alive.py \
  --models Qwen3-8B --backend tt_transformers_paged --input-len 1024 --output-len 1024 \
  --out-tag <tag> > /tmp/<tag>.log 2>&1'
```

### Per-op Tracy profile (probe path only)

```bash
docker exec -d tt-xla-eval bash -c 'cd /sglang && python3 -m tracy -v -r -p -o /tmp/tracy_out \
  python/sglang/srt/hardware_backend/tenstorrent/test/probe_decode_op_profile.py \
  --model Qwen/Qwen3-8B --input-len 1024 --decode 5 --skip-warmup 0 --out-tag <tag> \
  > /tmp/<tag>.log 2>&1'

# Then aggregate:
docker exec tt-xla-eval python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/tracy_aggregate.py \
  /tmp/tracy_out/reports/*/ops_perf_results_*.csv
```

---

## Pitfalls / gotchas

1. **Tracy disables JIT kernel cache.** Probe path: 5–20× slower with profiler. Server path: cold compile exceeds 15 min for Qwen3-8B — even 900 s health timeout isn't enough. Don't try to profile the server path via Tracy.

2. **Tracy can OOM the container** on long captures. Each decode step adds ~30 MB to the trace. Cap probe at `--decode 5–20` and output 5–30 tokens per server run.

3. **DEVICE KERNEL DURATION column in `ops_perf_results.csv` is unreliable on this build.** Most values are ~3.9 × 10¹² ns (uninitialized cycle counter). Use `OP TO OP LATENCY [ns]` instead (already done in `tracy_aggregate.py`).

4. **Pre-warm hook is broken on the locally rebuilt plugin.** Always set `BYPASS_PREWARM=1` in benches against `tt-xla-eval`. Root cause unknown — likely a CMake-config / link-time difference between the original `pjrt-plugin-tt 1.1.0` wheel build and my editable rebuild. The canonical wheel itself works but I no longer have it installed (the rebuild replaced it).

5. **`p3a-ngram` /tt-metal needs the fork patches.** During this session I `git apply`d `/home/mhnie/tt-metal-sglang` patches into the container's `/tt-metal/`. If the container is recreated the patches need re-applying:

   ```bash
   cd /home/mhnie/tt-metal-sglang && git diff 89686ee78d..HEAD -- ':!tt_metal/impl/device/device.cpp' > /tmp/p.patch
   docker cp /tmp/p.patch p3a-ngram:/tmp/p.patch
   docker exec p3a-ngram bash -c 'cd /tt-metal && git apply /tmp/p.patch'
   ```

6. **Context-length off-by-1 in sglang.** `--context-length N` allows up to `N-1` input tokens. For input 1024 + output 5, use `--context-length 1040` (the bench's auto-derive `input_len + output_len` doesn't leave enough slack for short outputs; it's fine for 1024+1024).

7. **BFP4 silently breaks greedy.** Token 1: 356 (BFP4) vs 220 (BF16/BFP8). Don't enable as default. The 5 % extra speed isn't worth it.

8. **Server-path bench needs `start_new_session=True` on the server Popen.** Without it, `os.killpg(getpgid(proc.pid), SIGTERM)` kills the bench script itself between models. Fixed in current bench, but worth knowing.

9. **PJRT overhead is ~4 ms, not ~38 ms.** The original `tt_xla_tpot_optimization.md` estimate was wrong. K=N multi-token compile can't help; bs=N batching wins via core utilization, not amortization. See `tenstorrent-tt-xla-pjrt-overhead-myth` memory.

---

## Open work — concrete next steps

> **SUPERSEDED (2026-05-18).** This section describes the OLD framing of A.1/A.2/A.3 as Python-side pre-tilize work, in-MLIR Tilize/Untilize fold patterns, and a StableHLOToTTIR-only A.3. Through 17 review rounds it became clear those framings were either not implementable or already done upstream. The actual landed implementation is described in `docs/superpowers/specs/2026-05-17-tt-xla-tilize-attack-design.md` (3 phase pairs: A.1.a explicit `argumentTypeMap` + heuristic classifier; A.2.a `to_layout`-pair fold; A.3.a auto-detect pass). Phase 2 (A.1.a) landed with **no TPOT movement** because the upstream pjrt frontend already populates `ttcore.argument_type` on every arg; Phases 3 and 4 were SKIPPED. The real residual bottleneck is **downstream of classification** (hoisted const-eval wrappers don't bake into compile-time tile-layout tensors) and is dominated by **activation Tilizes (~86%)**, not weight Tilizes (14.07%). The three Priority-A bullets below are kept for historical context only.

### Priority A — close the gap to tt_transformers (~3.5×)

Tilize = 81 % of decode time. **Three direct attack paths**, ranked:

**A.1 — Pre-tilize weights at load time (lowest risk, side-steps the compiler).**

Where: `python/sglang/srt/hardware_backend/tenstorrent/models/tt_xla_model.py`, after `self.hf_model.to(self.device)`.

What: walk `self.hf_model.named_parameters()` and, for tensors that are 2D matmul weights, call something like `ttnn.tilize_with_val_padding(param)` once and replace the parameter in-place. The compiled graph then sees a tile-layout weight and skips the per-step Tilize.

Risk: the compiled graph may still insert a Tilize check if it doesn't know the layout is already tile. Need to verify with Tracy that the Tilize count drops. If it doesn't, the fix is at the IR layer (A.3).

**A.2 — Eliminate redundant Tilize/Untilize pairs in tt-mlir.**

Where: `/home/mhnie/tt-mlir-sglang/lib/Dialect/TTNN/Transforms/`.

What: add a canonicalization that folds `Tilize(Untilize(x)) → x` and `Untilize(Tilize(x)) → x`. Many of the 1080 Tilizes likely have a paired Untilize. Use `ttmlir-opt --mlir-print-ir-after-all --ttir-to-ttnn-runtime-pipeline` on a single-decode StableHLO file to count cancellable pairs first.

Risk: layout type mismatches between paired ops may make the cancellation unsafe.

**A.3 — Mark HF weight block args as const-eval-eligible in StableHLOToTTIR.**

Where: `/home/mhnie/tt-mlir-sglang/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`.

What: when lowering `func.func`, look at `tf.aliasing_output` or `mhlo.is_donated` or similar attributes on block args. For args coming from torch_xla parameters, add a `tt.const_eval = true` attribute. Then update `ConstEvalHoistTransform` to honour the attribute.

Risk: largest scope, biggest win. Need to verify torch_xla emits some marker we can detect. If not, may need to track via input naming.

### Priority B — close the 41 ms server-vs-probe gap

At BFP8 the probe gets 91 ms but the server gets 132 ms. `SGLANG_TT_CACHE_MODE=index_copy` only saved 8 ms, so it's not the cache. Tracy on server-path is infeasible.

**B.1 — Python-level wrapper timing.** Add `time.perf_counter()` around major sections in `tt_xla_model.py::forward()` to localise. Specifically:
- Around `self.compiled_model(...)` (the actual compute)
- Around input-prep (input_ids stacking, attention mask, etc.)
- Around output prep (logits → sample → output token)
- Around the sglang scheduler hand-off

The 41 ms is one of these.

**B.2 — Diff `bench_3run_server_alive.py` request flow vs `probe_decode_op_profile.py` decode loop.** Read both side-by-side and identify the structural differences. Likely candidates: extra device-host sync points in the server, extra tensor reshapes, sampling-side overhead.

### Priority C — reproduce / validate Workstream A's pre-warm on rebuilt plugin

The pre-warm hook works on the **canonical** `pjrt-plugin-tt 1.1.0` wheel but fails with Error 13 on my **rebuilt** plugin. CMake config likely differs. Two fixes:

- Match the canonical CMake config exactly (find the build command for the 1.1.0 release wheel and replicate)
- Or accept the rebuild's quirk and document `BYPASS_PREWARM=1` as required for benches against this container

### Priority D — long tail

- B.1 fix in tt-mlir (`aten::index_put` dim mismatch — long-standing crash on sglang's built-in warmup)
- Real lm-eval-harness accuracy run on BFP8 (verified token 1 only this session)
- Multi-request stress (10 sequential requests) to confirm no recompile cliffs surface in production

---

## Useful one-liners

```bash
# Show all Qwen3-8B fixtures' run2 TPOT
for f in /home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v146_3run_server_q*.json; do
  jq -r '"\(.tag): \(.results[0].run2_warm_new_prompt.tpot_ms_mean // "fail")ms"' "$f"
done

# Verify rebuilt plugin commits match canonical
docker exec tt-xla-eval bash -c 'pip3 show pjrt-plugin-tt | grep commit'
# Expect: 470f0fad88fb9c2a... tt-mlir-commit=f3ddbfb6b0eab2... tt-metal-commit=90c914ef258b5cc92...

# Force-kill all sglang procs in a container
docker exec <name> bash -c 'pkill -9 -f sglang.launch_server; pkill -9 -f bench_3run; sleep 2; pgrep -af "sglang|bench_3run"'
```

---

## Memory entries written this session

- `tenstorrent-tt-xla-qwen3-8b-baseline.md` — Qwen3-8B TPOT baselines + reproduction
- `tenstorrent-tt-xla-tilize-bottleneck.md` — Tracy finding + the three fix paths
- (updated) `tenstorrent-tt-xla-tpot-workstream-a.md` — added shipped status

Index at `~/.claude/projects/-home-mhnie-sglang/memory/MEMORY.md`.
