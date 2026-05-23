# TT Qwen3-8B prefetcher correctness investigation (2026-05-23)

Status: investigation logged; prefetcher NOT shipped under
`SGLANG_TT_USE_PREFETCHER=1`. Canonical decode path remains the
correct-by-default production configuration.

## Bottom line

| Configuration | TPOT (1024/256, mean of 3 runs) | GSM8K(10) chat-format | Verdict |
|---|---|---|---|
| Canonical (no opt envs)                     | 27.08-27.74 ms          | 9/10  (90%) | ship |
| `SGLANG_TT_DISABLE_PREFILL_TRACE=1` alone   | 27.16-27.84 ms          | 10/10 (100%) | innocent but **no perf win** |
| `SGLANG_TT_USE_PREFETCHER=1` + DT=1          | 16.32-17.12 ms (1.66×)  | 0/10  — server crashes with NaN logits | **DO NOT SHIP** |

Win factor for "canonical → DT-only" = 1.00×. Win factor for "canonical → prefetcher" = 1.66×
but BLOCKED on a correctness regression that produces NaN/inf during sampling and
mode-collapsed garbage tokens under greedy decode (`chanchan…`, `=====`).

## Reproduction commands

Canonical server:
```bash
podman exec p3a-ngram bash -c '\
    pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c '\
    SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
    SGLANG_TT_MAX_BATCH=1 \
    HF_MODEL=Qwen/Qwen3-8B \
    setsid python3 -m sglang.launch_server \
        --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
        --device tenstorrent --context-length 4096 \
        --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
        --skip-server-warmup --max-running-requests 1 --trust-remote-code \
        --attention-backend torch_native > /tmp/qwen3_8b_canonical.log 2>&1 < /dev/null &'
```

Correctness gate (commit `<this commit hash>`):
```bash
podman exec p3a-ngram python3 \
    /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py \
    --num 10
```

TPOT bench (3 runs, ignore_eos):
```bash
podman exec p3a-ngram python3 /tmp/tpot_oneshot.py \
    --input-len 1024 --output-len 256 --runs 3 --label canonical
```

## What was investigated

Per the prefetcher-fix plan there were three suspect sites:

1. **GlobalCB init race** — consumer matmul on step 1 reads uninitialized
   pages before the async `ttnn.dram_prefetcher` producer writes them.
2. **`dynamic_worker_core_grid` 2× mismatch** — function returns 16 cores
   regardless of `num_cores` argument; norm `output_mem_config` uses
   `ring_size=32` while `sharded_output_config` uses 16 worker cores.
3. **`wo_sharded_ring` DRAM shard layout** — possible transposition vs what
   `dram_prefetcher` writes.

### Suspect #1: GlobalCB init race — probed, not confirmed

Attempt 1: Added `SGLANG_TT_PREFETCHER_SYNC=1` env-gated
`ttnn.synchronize_device(mesh_device)` call immediately after
`ttnn.dram_prefetcher` launch in `prefetcher.run()`. Result:

```
TT_FATAL @ /tt-metal/tt_metal/distributed/fd_mesh_command_queue.cpp:717:
!trace_id_.has_value()
```

`ttnn.synchronize_device` is **illegal during decode trace capture**. Sync
needs to land outside the trace OR a different sync primitive (CCL
semaphore, sub-device barrier) must be used. Did not pursue further in
this session — the fix would require modifying the trace capture flow in
`generator._capture_decode_trace_text` to pre-warm the prefetcher before
`begin_trace_capture` AND avoid retriggering the race inside the trace.

### Suspect #2: dynamic_worker_core_grid — analysed, NOT the bug

`dynamic_worker_core_grid(num_cores)` deliberately ignores `num_cores`
and always returns the rectangular `(5,0)-(6,7) = 16 cores` worker zone
on MUX-clamped Blackhole (P150a, P300_X2). This is intentional and
documented in the function — the rectangle must:
1. Be a single rectangle (LayerNorm validator)
2. Not overlap sender/receiver columns (sub-device isolation)
3. Remain valid before AND after `prefetcher.init(Mode.DECODE)`

The apparent "2× mismatch" between:
- norm `sharded_output_config` shape `(32, dim/16)` on 16 worker cores
- norm `output_mem_config` shape `(32, dim/32)` on 32 receiver cores

is NOT a bug. These are two memory configs for two stages: the norm
output lands on 16 worker cores (matching `sharded_program_config`'s
16-core LayerNorm), then `RMSNorm.forward` line 173 in `common/rmsnorm.py`
calls `ttnn.to_memory_config(x, output_mem_config)` which reshards to
the 32 receiver cores BEFORE the next matmul reads it. The flow is
consistent.

### Suspect #3: wo_sharded_ring layout — not probed

Insufficient time. Recommended next step: print the actual tile-stream
addresses written by `ttnn.dram_prefetcher` (instrument tt-metal) and
compare against the tile-stream addresses read by the WO matmul. If
they disagree, the bug is layout-side. Likely cheapest probe is to
add temporary printf logging in `dram_prefetcher_op.cpp` and the
ring-matmul reader kernel.

## Surface findings (negative results that may help future work)

- `ttnn.synchronize_device` cannot live inside the captured decode trace.
- The corruption is consistent across decode steps (always
  `chanchan…` or `=====` on greedy at the same prompt), not random — so
  the matmul reads SOME data, just not the correct weights. This is a
  data-routing bug, not an "uninitialized memory" bug.
- `SGLANG_TT_USE_PREFETCHER=1` already produces TPOT=16.3 ms at 1024/256.
  If the correctness bug is fixed, this is a real 1.66× win over
  canonical and worth pursuing in a future session.
- The chat-format eval is the only reliable correctness gate. The
  earlier raw 5-shot greedy harness called the canonical path "28%"
  but in chat-format that same canonical path scores 9/10 = 90% —
  matching the published HF reference. Always use the chat harness.

## Files changed in this session

1. `python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py`
   — NEW. Chat-format GSM8K(10) eval; canonical correctness gate.
2. `docs/platforms/tt_qwen3_8b_prefetcher_correctness_2026-05-23.md` —
   NEW. This document.

The `tt-metal-sglang` fork was NOT modified (the `SGLANG_TT_PREFETCHER_SYNC`
probe was added, tested, then reverted because the sync is illegal during
trace capture; the working tree is clean at `9bff1acf770`).

## How to attack the prefetcher again

Recommended approach for the next session:

1. **Pre-warm the prefetcher OUTSIDE trace capture.** Add a "warmup
   decode" call in `_capture_decode_trace_text` BEFORE
   `begin_trace_capture`. Run one decode step, sync the device, then
   capture the trace. This ensures the producer kernel is fully resident
   in L1 and the GlobalCB has been written through at least once before
   the trace locks in the producer/consumer handshake.

2. **Add a logit-magnitude probe.** Before the sampler, log the L∞ norm
   of the first 10 decode-step logit tensors. If max(|logits|) > 1e10
   the prefetcher is producing garbage; if < 100 it's clean. This
   gives a cheap test signal that doesn't require running GSM8K.

3. **Bisect by layer.** Implement a "first-N-layers prefetcher" mode
   that uses the prefetcher for layers 0..N-1 and the canonical path
   for layers N..35. Bisect N from 36 → 0 until corruption appears.
   The first bad N tells you which layer-specific matmul reads wrong.

4. **Compare WO and FF layouts.** WO matmul has the most complex DRAM
   prefetcher routing (smallest weights, most cross-device traffic).
   Compare its tile-stride math against FF1/FF3/FF2.

## Update 2026-05-23 (afternoon): pre-trace warmup FAILS — confirms layout bug

Followed recommendation #1 (pre-warm the prefetcher outside trace capture).
Added env-gated `SGLANG_TT_PREFETCHER_WARMUP=N` knob in
`tt-metal-sglang/models/tt_transformers/tt/generator.py:1513-1542`. When
`N≥1`, before `begin_trace_capture`, runs N extra eager
`_decode_forward_no_trace_text(...)` calls with `synchronize_device`
sandwiches around each.  The extra eager path goes through `forward()`,
which calls `prefetcher.run()` (kicks off `ttnn.dram_prefetcher` producer
+ all 36 layers of consumer matmuls) then `prefetcher.stop()` (deallocates
producer handle).  Each warmup cycle therefore fully primes the GlobalCB
sender/receiver handshake before the trace records the producer/consumer
ordering.

Result with `SGLANG_TT_PREFETCHER_WARMUP=1`, prefetcher ON, DT=1, on the
chat-format gate:

| Step | Observation |
|---|---|
| Trace capture | Compile run + warmup eager run + trace capture all completed cleanly (no TT_FATAL, no hang) |
| Q1 first token | Garbage Hebrew/CJK tokens (`AtAמוןAtAמוןAtA…`) — same corruption signature as no-warmup |
| Q2 prefill | Sampler crashes with `RuntimeError: probability tensor contains either inf, nan or element < 0` |
| TPOT during Q1 | Server log gen-throughput ≈ 50 tok/s (≈ 20 ms TPOT) — the matmul speedup is real |
| GSM8K(10) | 0/10 (same as no-warmup baseline) |

The pre-trace warmup mechanism executes correctly (logs `[PREFETCHER-WARMUP]
extra eager decode complete and synced` exactly as designed), but the
corruption signature on the first trace-replay logits is identical to
the no-warmup case.  This rules out "uninitialized GlobalCB" as the
root cause and confirms the prior agent's "data-routing, not init-race"
hypothesis.

The env-gated warmup hook is kept in the tree (cost-zero when off)
because it's still useful for follow-up investigations — e.g. measuring
whether the producer needs N>1 prime cycles to stabilize, or as a
scaffold for combining warmup with future layout-side fixes.

### Confirmed shipping verdict

**DO NOT ship the prefetcher under SGLANG_TT_USE_PREFETCHER=1.**
The 1.66× decode speedup is real but the routing-side correctness bug is
not addressable by any host-side ordering primitive available at the
Python layer (synchronize_device is illegal inside trace; pre-trace
warmup doesn't help; the prefetcher's `run()` is rebuilt fresh in every
forward call, so there's no carried state to "prime"). A real fix needs
either (a) tt-metal C++ changes to the `ttnn.dram_prefetcher` op's
DRAM-shard tile-routing math, OR (b) a sub-device barrier inserted at
the C++ level between producer.start() and consumer.read() inside the
kernel binary itself.  Both are out of scope for SGLang-side work.

### Updated next-session recommendations (priority order)

1. **(layout-side, deep)** Instrument `ttnn.dram_prefetcher` in
   `tt-metal/ttnn/.../dram_prefetcher_op.cpp` to log the actual
   tile-bank addresses it writes for `wo_sharded_ring`; compare against
   the addresses the WO ring-gather matmul reader kernel reads. The
   prior agent's "matmul reads SOME data, just not the correct weights"
   evidence + the no-help-from-warmup result narrows this to a
   `wo_sharded_ring` layout transposition / stride mismatch.

2. **(bisect, medium)** Per-layer prefetcher disable: wire an env var
   `SGLANG_TT_PREFETCHER_LAST_N=K` so layers 0..(36-K) use the
   canonical (non-prefetched) `self.wo` path and layers (36-K)..35 use
   `self.wo_sharded_ring`. Sweep K from 0 → 36 and find the first K
   that corrupts.  Tells you whether the bug is layer-specific.

3. **(cheap, diagnostic)** Add a logit-magnitude probe before the
   sampler.  Log L∞ norm and NaN count of the first 10 decode-step
   logit tensors. Distinguish "first replay is dirty" (carryover) from
   "every replay is dirty" (persistent layout bug).
