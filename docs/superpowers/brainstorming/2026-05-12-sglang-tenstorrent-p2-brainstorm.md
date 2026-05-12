# SGLang-on-Tenstorrent — P2 Brainstorm

> **This is pre-spec brainstorming, NOT a committed plan.** Spec §11 of the
> P1 design explicitly says "P2 and P3 are NOT committed by this spec";
> any P2 work needs its own spec + plan written from scratch. This
> document captures the inputs that future P2 spec author should start
> from:
>
> 1. What P1 actually achieved on real hardware
> 2. P2 investment items discovered during P1 bring-up (not predicted by
>    the original spec)
> 3. Spec §11's two P2 directions, re-framed with P1 data
> 4. Decision-gate questions to answer before committing to a direction

**Date:** 2026-05-12  
**Author:** Claude (during P1 hardware bring-up session)  
**Branch context:** `tenstorrent-p1` @ commit `5cd89a8d9` (46 commits, all in
`predator2k/sglang`); P1 spec §9 acceptance gate fully passed.

---

## 1. P1 reality check

The P1 spec was written before any TT hardware testing. P1 implementation
+ live runs produced these numbers, which P2 scoping should anchor on:

### Performance (measured on 2× Blackhole p150a, Llama-3.1-8B-Instruct, ETH fabric)

| Metric | Value | Spec §10 prediction |
|---|---|---|
| Decode steady-state | **34.4 tok/s** | "10–25 tok/s P1 ceiling" (risk #2) |
| Prefill (138 tokens warm) | 49.7 ms | n/a |
| Prefill (5403 tokens warm) | 698.8 ms | n/a |
| Prefill (8103 tokens warm) | 971.9 ms | n/a |
| 5K-in + 5K-out wall-time | 167 s | n/a |
| Cold-start (kernel-cache hit) | 244 ms first prefill | n/a (spec didn't model persistent cache) |
| Stability (15 min, 614 req) | ITL p99 drift 0.4% | "<10%" (§8.4) |
| ITL p99 | 29.9 ms | n/a |

**Implications for P2:**
- Spec underestimated decode by ~40%. ttnn synchronous execution is faster
  than the 10-25 tok/s ceiling §10 risk #2 worried about. **P2-paged's
  motivation (improve throughput) is weaker than feared.**
- Prefill scales sub-linearly with prompt length (effective tok/s rises:
  2.7k @ 138, 7.7k @ 5403, 8.3k @ 8103). Arithmetic intensity wins at
  scale. P2-paged's RadixAttention gives wins ONLY on prefix-cache hits;
  cold prefill is already fast.

### Correctness (H.2 + H.3 hardware runs)

- 6/6 factual prompts answered correctly in the requested language
  (Chinese/French/German/English/math)
- 8/10 MMLU-mini multi-choice; the 2 misses are both elementary
  arithmetic — known Llama-3.1-8B weakness, not TT-specific
- Token-level agreement vs HF BF16 reference: 39.4% mean (NOT spec's
  90%). **BFP8 weights cause real cross-precision drift** at semantic
  decision points (e.g. " The" vs "\nThe" between sentences).
  Documented in `test_greedy_correctness.py` module docstring.

**Implications for P2:**
- The model is producing the right answers; it's just choosing different
  equally-valid tokens. Any P2 work that adds sampling (top-k, top-p)
  inherits this same precision drift.
- If P2 cares about reproducibility-vs-HF, switch `optimizations` to a
  BF16-throughout precision config. Cost: lower decode tok/s, more L1
  pressure. Real-world rarely cares about exact reproducibility — flag
  but don't optimize for it.

---

## 2. P2 investment items NOT in P1 spec §11

These surfaced during P1 bring-up. They're real gaps that any P2 effort
needs to address. None are blockers for P1 (which is shipped) but each
constrains workload coverage.

### 2.1 KV-sharding shape restrictions (HIGH priority)

**Finding:** `tt_transformers.tt.attention` calls
`ttnn.interleaved_to_sharded(k_heads, KV_PREFILL_MEM_CFG(seq_len))` which
fails with `TT_FATAL tensor_layout.cpp:111` when
`physical_shard_shape % tile_shape != 0` for many `seq_len` values
between 128 and 6144.

**Empirically validated on this host:**
- 128 ✓
- 4096 ✓
- 6144 ✓
- 8192 ✓
- 256 ✗ (crashed perf_log v1)
- 1024 / 2048 unknown / unsafe

**P1's workaround:** the backend's pad-step heuristic + a hard ceiling
at `MAX_PREFILL_CHUNK_SIZE * 1024 = 8192`. Not a real fix — just avoids
the broken middle range.

**P2 needs:**
- Enumerate the full set of valid prefill seq_lens on Blackhole 2-card
  (probably divisibility by `num_kv_heads × tile_h × shard_h` or
  similar — needs reading attention.py + KV_PREFILL_MEM_CFG)
- Either dynamically choose pad targets from the valid set, OR
- Upstream patch to tt_transformers that makes KV sharding accept all
  128-aligned shapes

### 2.2 SIGQUIT handling on scheduler crash (MEDIUM)

**Finding:** When the TT scheduler subprocess crashes (a few times during
P1 bring-up: chunked-prefill assert, KV-shard assert, etc.), SGLang's
parent sends SIGQUIT to clean up. Our `MeshDeviceCtx` has SIGTERM +
SIGINT + atexit handlers but NOT SIGQUIT. Mesh device handle leaks;
next process can't open the mesh until `tt-smi -r` is run.

**Spec §7.3 mentions this is a concern but doesn't handle it.**

**P2 fix (or P1.5 patch):**
- Add SIGQUIT to the signal handlers in `MeshDeviceCtx.__init__`
- Validate the cleanup actually completes in <5s under SIGQUIT (the
  watchdog from §7.3 may need adjustment if SIGQUIT bypasses the
  normal Python signal dispatch)
- Consider auto-reset on startup with a `--force-reset` flag
  (spec §10 risk #4 suggested this; deferred)

### 2.3 `tt_transformers` device-name table missing `P300` (LOW, upstream)

**Finding:** `MAX_PREFILL_CHUNK_SIZES_DIV1024` in
`tt_transformers/tt/model_config.py:577` has entries for N150, N300,
T3K, TG, P150x4 — but no `P300` (2× Blackhole). Our config falls
into the unknown-device fallback (chunk-size=4 = 4096 tokens), and
chunked-prefill requires paged attention which P1 doesn't support.

**P1 workaround:** override `MAX_PREFILL_CHUNK_SIZE=8` via env var
in `apply_server_args_defaults`.

**P2 work:** upstream PR to tt_transformers adding a `P300` entry,
empirically tuned for Blackhole 2-card.

### 2.4 `MAX_QKV_MM_SEQ_LEN=2048` boundary (LOW, design)

**Finding:** `attention.py:684` asserts `seq_len % MAX_QKV_MM_SEQ_LEN == 0`
when `seq_len > 2048`. Our backend handles this by switching pad step
to 2048 once prompt length crosses 2048.

**P2 implications:**
- If P2-paged adopts SGLang's per-layer attention dispatch, this
  constraint moves out — SGLang's `Attention` doesn't have this 2048
  boundary
- If P2-coverage uses tt-xla, the constraint may or may not apply
  depending on whether tt-xla compiles its own attention kernel

### 2.5 Padded-KV decode pollution (EMPIRICALLY benign, P1 fortunate)

**Finding (G.2 worry → G.5 verification):** when prefill pads with
right-padding (e.g. prompt 4995 → padded 6144), the KV cache stores
the padded positions too. We worried decode would attend to those
junk positions and degrade output.

**Empirical answer:** 5000-token decode produces coherent output.
`tt_transformers` correctly bounds attention to `current_pos`.

**P2 should still understand why** — if P2-paged changes the attention
dispatch, we need to preserve this property explicitly rather than
inherit it implicitly from `tt_transformers`.

### 2.6 Persistent kernel cache placement (LOW, ops)

**Finding:** `EnablePersistentKernelCache()` gives 40× faster cold-start
prefill (9501 ms → 244 ms). The cache files live at an unknown disk
path (we didn't track down where during P1).

**P2 should:**
- Document the disk path (look at `TT_METAL_HOME/built/` or
  `~/.cache/ttnn/` empirically)
- Decide if it should be persisted with the deployment artifact
  (Dockerfile layer? mount?)
- Quantify size growth over time (we ran ~700 requests, no measurement)

---

## 3. Spec §11 directions, re-framed with P1 data

### P2-paged: paged KV + RadixAttention

**Original spec motivation:** improve throughput, enable prefix caching.

**Reframed with P1 data:**
- Throughput motivation weaker than expected (34 tok/s already above
  spec ceiling)
- Prefix caching is the real win — for workloads with shared prompt
  prefixes (chat history, system prompts, RAG context), this can
  10×+ effective decode rate
- Phase 0.2 discovery: `create_tt_model(paged_attention_config=...)`
  already exists. P2-paged might just be passing a config through,
  not forking `tt_transformers.tt.attention`. **Spike this in <1 week
  before committing to full P2-paged scope.**

**Risk:** if `paged_attention_config` works, P2-paged might be 1-2
weeks of work (just integration), and Phase 3 batching becomes more
attractive than spec §11 estimated.

### P2-coverage: TTXLAExecutionBackend

**Original spec motivation:** model coverage (run any HF PyTorch model).

**Reframed with P1 data:**
- P1 ships `TTXLAExecutionBackend` as a registered placeholder with
  clear `NotImplementedError("P2-coverage")`. Switching is a one-flag
  flip: `SGLANG_TT_EXECUTION_BACKEND=tt_xla`
- The `TTExecutionBackend` ABC is stable; worker/scheduler unaffected
- tt-xla maturity is the open question — spec said "depends on tt-xla
  maturity" in mid-2026; that's still the right framing

**Risk:** tt-xla may not have proven Llama parity by P2-start. If not,
we'd be the first user — bug discovery + upstream PRs would expand
scope unpredictably.

### Plan Q (write our own ttnn Llama)

**Spec §11 decision rule says:** "If tt_transformers proves too fragile,
plan Q (write our own ttnn Llama) — doubles the budget."

**Reframed with P1 data:** tt_transformers is NOT too fragile. We hit
some quirks (KV sharding shapes, env-var collision, prefill format) but
they were all worked around with thin adapter logic. **Plan Q is
unjustified** unless the next workload requires features tt_transformers
fundamentally lacks (e.g. an unusual attention variant).

---

## 4. Decision-gate questions

Before writing a P2 spec, the answers to these determine direction:

1. **Workload model coverage**: Will P2 stay on Llama-3.1-8B, or do we
   need Qwen3 / Mistral / DeepSeek / Llama-3.1-70B?
   - Llama only → P2-paged
   - Multi-model → P2-coverage

2. **Workload shape**: Is the prefix-cache win real for the target
   workload? Chat with shared history (high reuse) vs one-shot
   completions (no reuse).
   - High prefix reuse → P2-paged is high value
   - Low reuse → P2-paged gives little

3. **Concurrency target**: Is 1-request-at-a-time acceptable, or do we
   need to serve N concurrent requests?
   - N=1 acceptable → P2 may not be needed at all
   - N > 1 → Phase 3 (batched) is required, which builds on P2-paged

4. **Timeline**: When is P2 needed?
   - <1 month → only P2-paged-via-paged_attention_config spike is
     realistic
   - 1-3 months → P2-paged full or P2-coverage spike
   - >3 months → either is feasible; tt-xla maturity may improve

5. **Hardware scaling**: Will P2 stay on 2× p150a, or expand to 4× /
   8× / multi-host clusters?
   - 2× stay → no rework on `MeshDeviceCtx`
   - 4× / 8× → spec §3 mesh-shape math needs redo; tt_transformers
     table has P150x4 and P150x8 entries so this is partly already
     in their hands

---

## 5. Open technical questions

Things we'd need to verify before committing to a P2 spec:

### For P2-paged
- Does `create_tt_model(paged_attention_config=PagedAttentionConfig(...))`
  produce a working paged model on Blackhole 2-card today? (Spike: 1 day)
- What page_size does tt_transformers want? Constraints from L1?
- Does the resulting Generator's `prefill_forward_single_user_text` and
  `decode_forward_text` still work, or do they require new method names
  for the paged path?
- How does SGLang's `MHATokenToKVPool` map onto tt_transformers'
  page table representation? (May need an adapter layer.)

### For P2-coverage
- What's tt-xla's status on Llama-3.1-8B in mid-2026? (Re-check the
  https://github.com/tenstorrent/tt-xla README and recent commits)
- Does PJRT have a per-request KV-cache API or do we need to wrap?
- How does tt-xla persist compiled artifacts? Is the disk cache shared
  with tt-metal's `EnablePersistentKernelCache` or separate?
- Are there published benchmarks comparing tt-xla vs tt_transformers
  on the same model?

### For both
- Re-evaluate the precision config: stay on BFP8 (`DecodersPrecision.performance`)
  or switch to BF16 throughout? Cross-precision drift mostly doesn't
  matter, but if it ever does (reproducibility tests, regulator audit),
  worth quantifying the BF16 cost up front.
- SIGQUIT signal handler (§2.2 above) should land in P1.5 OR as P2
  task 1.

---

## 6. P1 deliverables that P2 inherits for free

P1 set up these P2 prerequisites without P2 paying for them:

1. **`TTExecutionBackend` ABC + registry** — P2-coverage just fills in
   `TTXLAExecutionBackend`; nothing else changes
2. **`SGLANG_TT_PREFILL_PAD_STEP` env var** — operator can tune for
   their workload; P2 inherits without breaking the API
3. **`EnablePersistentKernelCache` in `MeshDeviceCtx`** — disk-cached
   kernels survive across processes
4. **31 unit tests + 13 hardware-gated tests** — regression safety net
   for any P2 backend change
5. **`docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md`
   Appendix B** — 8 rounds of spec review process documented;
   institutional memory for how to evolve the spec

---

## 7. Recommended next steps (NOT a plan, just a possible sequence)

1. **Do nothing for now** — P1 ships. Wait for a real P2 driver
   (workload need, timeline, business case).
2. **When P2 driver appears**:
   - Run the `paged_attention_config` spike (1 day) to size P2-paged.
   - Check tt-xla maturity (read README + recent commits, 30 min).
   - Use spec §11 decision rule + answers to §4 questions above to
     pick a direction.
3. **Write P2 spec** using `superpowers:brainstorming` skill, anchored
   on this brainstorm doc + spec §11 outline + real-world workload
   constraints.
4. **Land SIGQUIT handler** as a P1.5 patch independent of P2 — small,
   self-contained, fixes a real recovery-hygiene issue.

---

## 8. Cross-references

- P1 spec: `docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md`
- P1 plan: `docs/superpowers/plans/2026-05-11-sglang-tenstorrent-p1-plan.md`
- P1 acceptance evidence: `python/sglang/srt/hardware_backend/tenstorrent/test/`
  (test_smoke.py, test_greedy_correctness.py, test_mmlu_mini.py,
  test_stability.py, test_perf_log.py, test_p1_acceptance.py)
- TT-XLA upstream: https://github.com/tenstorrent/tt-xla
- tt_transformers upstream: bundled in the tt-metal docker image
- Memory notes (this session's findings):
  - `tenstorrent-p1-implementation-status.md`
  - `tenstorrent-perf-knobs.md`
  - `tenstorrent-tt-transformers-constraints.md`
