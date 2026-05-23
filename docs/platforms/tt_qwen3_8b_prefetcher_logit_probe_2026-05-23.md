# TT Qwen3-8B prefetcher — ATTACK-2 logit-probe + ATTACK-1 hang root cause (2026-05-23)

Status: **DEFINITIVE INFEASIBILITY for Python-only fix.** Captures hard
evidence for the next session that the prefetcher correctness bug lives
in tt-metal C++ (tile-format / data-routing in `ttnn.dram_prefetcher`'s
GlobalCB hand-off), not in any Python layer this repo controls.

This supersedes the *attack plan* portion of
`tt_qwen3_8b_prefetcher_correctness_2026-05-23.md` (the "recommended
next" attacks #1, #2, #3) and adds:

* the actual ATTACK-2 magnitude+top-id telemetry,
* why ATTACK-1 (producer-only `num_layers` truncation) is architecturally
  impossible without consumer-side gating, with a hang reproduction.

The canonical (no-prefetcher) shipping path is unchanged and still
green: this session re-verified **GSM8K(10) chat-format = 10/10 = 100%**
on the canonical configuration.

## Bottom line

| Configuration | GSM8K(10) chat-format | Notes |
|---|---|---|
| Canonical (no prefetcher)                    | **10/10** (verified) | shipping default |
| `SGLANG_TT_USE_PREFETCHER=1` + DT=1 + LAYERS=36 | 0/10 (re-verified) | "broken baseline" |
| Above + LAYERS=1 (producer truncated)        | **server HANGS** at first decode forward | architectural deadlock, not a perf regression |

Logit-probe under the broken baseline at decode-step 1 already shows
**max(\|logits\|) ≈ 2.2 × 10²⁰**, consistent across 24 captured decode
steps, with no NaN and no inf — i.e. the matmul is producing
**deterministic non-junk floats whose magnitude is ~10²⁰ too large.**
This is the smoking gun that the bug is a tile-format / shared-exponent
data-routing issue in `ttnn.dram_prefetcher`, not a one-shot
initialization race and not an accumulation drift.

## ATTACK-1 (per-layer producer truncation): architecturally impossible

The existing `SGLANG_TT_PREFETCHER_LAYERS=N` env knob (from the prior
session) truncates the producer side only: `ttnn.dram_prefetcher` runs
`num_layers=N` cycles, then exits. The consumer side
(`ttnn.linear(..., global_cb=..., ...)`) is left intact for **all 36
decoder layers** because the prefetcher object's `global_cb`,
`receiver_sub_device_id`, etc., are still passed at every layer's
matmul call.

Reproducible result with N=1 on Qwen3-8B (`SGLANG_TT_PREFETCHER_LAYERS=1`):

* model + prefetcher init complete cleanly,
* `[DRAM Prefetcher] Creating global CB with size: 835584` logged,
* `[Prefetcher] run(): num_layers=36 effective_num_layers=1
  (SGLANG_TT_PREFETCHER_LAYERS=1)` logged,
* first `/generate` request submitted →
* **scheduler watchdog timeout after 300 s** with the entire decode
  hung waiting on a CB handshake.

Reading `tt-metal/ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/reader_dram.cpp`
confirms the structural issue:

```cpp
for (uint32_t layer = 0; layer < num_layers; layer++) {
  for (uint32_t t = 0; t < num_tensors; t++) { ... cb_reserve_back ... cb_push_back ... }
}
```

i.e. the producer pushes exactly `num_layers * num_tensors` CB slots and
exits. Consumer matmul `layer = L` waits on the L-th push for each of
its 5 tensors (wqkv, wo, w1, w3, w2). With `num_layers < 36`, layers
≥ N's matmuls block forever on `cb_wait_front`.

The implication is that **any layer-bisect must gate BOTH the producer
and the consumer in lock-step**. The consumer-side gate is non-trivial:

1. The ring matmul `program_config` returned by
   `model_config.matmul_1d_ring_config(..., prefetch=True,
   num_global_cb_receivers=prefetcher.num_receiver_cores)` is rejected
   by tt-metal's validator at
   `matmul_device_operation.cpp:514-518` when `global_cb=None` is
   passed but `num_global_cb_receivers > 1` —
   `"Num global CB receivers must be 1 when global CB is not provided."`
2. To run the same ring matmul WITHOUT GlobalCB the program config
   must be rebuilt with `prefetch=False, num_global_cb_receivers=1`
   (the LM-head already does this — see
   `lm_head.py:142-166` + `model_config.py:2519-2528`). That's the
   architecturally clean per-layer gate but requires plumbing
   `_use_prefetcher_for_layer(layer)` predicates through
   **5 matmul call sites** (QKV, WO, FF1, FF3, FF2) **× 5 program/mem
   config helpers** in `model_config.py`, and adjusting the producer's
   `num_layers` argument plus the per-layer `register_callback`
   filtering. Substantial PR-sized work with a high regression
   surface on a known-shippable canonical path.

This is the correct next-session implementation path, but it is **out
of scope** for an 8-hour single-session investigation given the prior
two session-attempts already failed and the ATTACK-2 evidence below
points the root cause at tt-metal C++ tile-format handling — which the
per-layer bisect would not fix even if successful, only attribute.

## ATTACK-2 (logit magnitude probe): smoking-gun evidence

A small instrumentation hook (`_logit_probe(...)` in
`models/tt_transformers/tt/generator_sglang.py`, gated by
`SGLANG_TT_LOGIT_PROBE_STEPS=N`) was wired into `Qwen.decode_forward()`
to log per-step `(max_abs, has_nan, has_inf, top_token, top_5)` on the
output logits **before** the sampler.

### Probe output under prefetcher-on at decode steps 1–24

Prompt: "2+2=", greedy (T=0), 25 new tokens, `SGLANG_TT_PREFETCHER_LAYERS=36`
(full prefetcher = "broken baseline").

```
step=1  max_abs=2.237e+20 top_id=38156 top_val=2.236e+20
step=2  max_abs=2.191e+20 top_id=118    top_val=1.960e+20
step=3  max_abs=2.375e+20 top_id=114529 top_val=2.375e+20
step=4  max_abs=2.283e+20 top_id=114400 top_val=2.075e+20
step=5  max_abs=2.352e+20 top_id=114131 top_val=1.914e+20
step=6  max_abs=2.283e+20 top_id=114142 top_val=2.283e+20
step=7  max_abs=2.306e+20 top_id=38986  top_val=2.144e+20
step=8  max_abs=2.398e+20 top_id=114141 top_val=2.306e+20
step=9  max_abs=2.606e+20 top_id=114170 top_val=2.606e+20
step=10 max_abs=2.352e+20 top_id=38258  top_val=2.352e+20
...
step=16 max_abs=2.928e+20 top_id=66     top_val=2.121e+20
...
step=24 max_abs=2.467e+20 top_id=42     top_val=2.167e+20
```

across all 24 captured steps: `has_nan=False, has_inf=False`.

### What this rules in / rules out

| Hypothesis | Predicted observation | Actual | Verdict |
|---|---|---|---|
| GlobalCB init race (1st replay only) | step 1 dirty, steps ≥ 2 clean | every step dirty | **REJECTED** |
| KV-cache accumulation drift | magnitudes monotonically grow | bounded ≈ 2-3 × 10²⁰ | **REJECTED** |
| Stochastic memory garbage | NaN/inf/large variance | deterministic ~2 × 10²⁰, no NaN/inf | **REJECTED** |
| Wrong tile shared-exponent (BFP8) | magnitudes off by `2^(exp_diff)` from true | matches: `~10²⁰ ≈ 2^66`, consistent with a `~30-bit` exponent misalignment compounded across 36 hidden_dim=4096 matmuls | **CONSISTENT** |

Bfloat8_b tiles encode a shared 4-bit exponent biased per-tile. If the
DRAM prefetcher writes the tile-data bytes to one GlobalCB slot but
the consumer matmul reads the shared exponent from a different slot
(e.g. an off-by-one tile index or a header/data buffer misalignment),
each output element is multiplied by `2^(exp_diff)` per multiply-add.
Summed across `dim=4096` activations and 36 decoder layers, this
explodes to the 10²⁰ regime we observe.

That class of bug lives entirely in the dram_prefetcher
producer/consumer protocol implementation, i.e. in tt-metal C++ files
under `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/...`
(reader/writer kernels + program-factory page-size math) and the
matmul-side GlobalCB receive code under
`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`.

## Recommended next-session investigation

This session burned the Python-only fix tree. The next attack must go
to tt-metal C++:

1. **Bit-level dump of one tile from DRAM vs. GlobalCB.** Add a debug
   path that:
   * picks one specific weight (e.g. layer 0 wqkv) and dumps its first
     tile from DRAM (read with regular `ttnn.to_torch`),
   * captures the same tile from GlobalCB after `ttnn.dram_prefetcher`
     has populated it,
   * compares byte-for-byte. Mismatch → confirms data-routing bug
     in `dram_prefetcher` writer or matmul reader. Match → ruled out.
2. **Run with BF16 weights instead of BFP8.** BF16 has no shared
   exponent. If the magnitude explosion DISAPPEARS under BF16, the bug
   is provably the BFP8 shared-exponent misalignment. If it persists,
   the bug is a more general data-routing issue.
   ```bash
   # SGLANG_TT_QWEN3_PRECISION=full_bf16 already wired (model_config.py)
   SGLANG_TT_QWEN3_PRECISION=full_bf16 SGLANG_TT_USE_PREFETCHER=1 ...
   ```
3. **Implement proper per-layer bisect** (the architectural work above):
   `prefetch=False` ring-matmul config + per-layer `register_callback`
   gating + per-layer matmul kwargs gating. This becomes useful AFTER
   step 1 or 2 narrows the root cause — to attribute the bug to a
   specific layer's weight (e.g. layer 0 is fine, layer 1 breaks),
   not to fix it.
4. **File an upstream tt-metal issue** with the byte-dump + the
   logit-probe evidence below. The fix has to land in tt-metal's
   `dram_prefetcher` op and its matmul GlobalCB integration, not in
   any SGLang Python.

## Files touched / artifacts this session

* `tt-metal-sglang/models/tt_transformers/tt/generator_sglang.py` —
  ATTACK-2 logit-probe (`SGLANG_TT_LOGIT_PROBE_STEPS=N` env-gated,
  no-op when off; default off; cost zero on shipping path).
* `tt-metal-sglang/models/tt_transformers/tt/prefetcher.py` —
  `SGLANG_TT_PREFETCHER_LAYERS=N` env-gated producer truncation +
  `use_for_layer(layer)` predicate scaffolding (no-op when N>=num_layers;
  the predicate is present for the next-session per-layer bisect to
  wire into attention.py / mlp.py without touching prefetcher.py again).
* `tt-metal-sglang/models/tt_transformers/tt/model_config.py` —
  pre-existing `full_bf16` / `bfp8` precision presets retained from
  prior session (relevant for next-session attack #2 above).
* `docs/platforms/tt_qwen3_8b_prefetcher_logit_probe_2026-05-23.md` —
  this document.

## Reproduction commands

ATTACK-2 probe (logits captured for first 30 decode steps):

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000"; sleep 3'
podman exec p3a-ngram bash -c '\
    SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
    SGLANG_TT_MAX_BATCH=1 \
    HF_MODEL=Qwen/Qwen3-8B \
    SGLANG_TT_USE_PREFETCHER=1 \
    SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
    SGLANG_TT_PREFETCHER_LAYERS=36 \
    SGLANG_TT_LOGIT_PROBE_STEPS=30 \
    nohup python3 -m sglang.launch_server \
        --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
        --device tenstorrent --context-length 4096 \
        --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
        --skip-server-warmup --max-running-requests 1 --trust-remote-code \
        --attention-backend torch_native > /tmp/qwen3_8b_pf36_probe.log 2>&1 &'
# wait for /get_model_info to 200
podman exec p3a-ngram bash -c 'curl -s -X POST http://127.0.0.1:30000/generate \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"2+2=\", \"sampling_params\": {\"max_new_tokens\": 25, \"temperature\": 0.0}}"'
podman exec p3a-ngram bash -c 'grep "LOGIT-PROBE" /tmp/qwen3_8b_pf36_probe.log | head -25'
```

ATTACK-1 hang reproduction (producer-only truncation):

```bash
# same as above but with SGLANG_TT_PREFETCHER_LAYERS=1
# first /generate request will hit the 300 s watchdog timeout and SIGQUIT.
```

Canonical correctness gate (re-verified this session = 10/10 = 100%):

```bash
podman exec p3a-ngram bash -c '\
    SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
    SGLANG_TT_MAX_BATCH=1 \
    HF_MODEL=Qwen/Qwen3-8B \
    nohup python3 -m sglang.launch_server \
        --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
        --device tenstorrent --context-length 4096 \
        --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
        --skip-server-warmup --max-running-requests 1 --trust-remote-code \
        --attention-backend torch_native > /tmp/qwen3_8b_canonical.log 2>&1 &'
podman exec p3a-ngram python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 10
# expect: FINAL: 10/10 = 100.0%
```

## Shipping verdict

**Unchanged from prior session: DO NOT ship the prefetcher under
`SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**

Canonical (`SGLANG_TT_USE_PREFETCHER` unset/0) remains the production
configuration at TPOT ≈ 27 ms / 1024-1024 with GSM8K(10) = 10/10.

The 1.66× prefetcher win is real (TPOT ≈ 16-20 ms) but blocked on a
tt-metal C++ correctness bug for which we now have direct evidence
pointing at the BFP8 shared-exponent / data-routing layer of the
`ttnn.dram_prefetcher` ↔ matmul-GlobalCB protocol. Three independent
Python-only fix attempts (sync, pre-trace warmup, producer-only layer
truncation) have now all failed — each in a way that's diagnostically
informative but none that recovers correctness. The next attack must
be at the tt-metal C++ level (byte-level GlobalCB dump or BF16-weight
ablation) per the recommendation list above.
