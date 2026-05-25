# TT Qwen3-8B prefetcher — U17 Phase-0: PRE-RS L1 probe shows stomper acts UPSTREAM of RS reader — GlobalSemaphore handshake plan invalidated — 2026-05-25

Status: **EVIDENCE_ADVANCE — A device-side PRE-RS probe
(`SGLANG_TT_U17_PROBE_RS_PRE=1`), inserted at the very start of
`line_reduce_scatter_minimal_async_reader.cpp::kernel_main()`
BEFORE any normal RS work fires, reads producer's first tile L1
bytes via `noc_async_read(input_tensor_addrgen.get_noc_addr(0),
...)` and `noc_async_read_barrier()`. Under zero-weight injection +
prefetcher + Qwen3-8B decode, the probe consistently observes the
W2 matmul output L1 address `0xa6700` as **NONZERO with
activation-like bytes** (e.g. `w0=0xd328_5f35 w1=0xcd92_9dbb
w2=0x1d87_9c92 w3=0x9d62_2223`; 528/1152 probes at `0xa6700`
returned NONZERO; the other 624 saw DRAM intermediate-tensor
addresses, all zero). Because the probe fires BEFORE the RS reader's
compute kernel starts, this proves the stomp happens
**before the RS reader runs**, not during it. This RULES OUT the
original U16 Phase-5c hypothesis that a device-side GlobalSemaphore
handshake between W2 matmul PACK and RS reader's `noc_async_read`
will close the race: the L1 region is already stomped before the RS
reader would wait on any such semaphore — so the wait would be a
no-op for correctness.**

Continuation of
`tt_qwen3_8b_prefetcher_U16_dispatch_barrier_pack_fence_RULED_OUT_2026-05-25.md`.

tt-metal-sglang HEAD: **`2f48d0e5cec`** (this U17 Phase-0 commit;
env-gated probe; default-off; canonical bytewise-equal).
sglang HEAD: pending this doc commit.

## TL;DR

U17 was dispatched to implement the GlobalSemaphore PACK→reader
handshake (the "STANDING — TOP PRIORITY" attack from U16). Per
plan, Phase-0 was a quick probe to confirm the bug locus before
the multi-day C++ implementation began.

The Phase-0 result invalidated the implementation plan. The bug
is NOT a race during RS reader execution — by the time the RS
reader's `kernel_main()` starts running, the producer's L1
already contains the wrong (stomped) bytes. A semaphore the RS
reader waits on cannot help if the corruption is already present
before the reader runs.

## What U17 Phase-0 added

| File | Change |
|---|---|
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp` | `#ifdef SGLANG_TT_U17_PROBE_RS_PRE` block at top of `kernel_main()` after addrgen setup. Issues `cb_reserve_back(cb_input_id, 1)` to claim a scratch L1 slot (never `push_back`s — slot is overwritten by main loop), zero-fills it, then `noc_async_read` of `page_size` bytes from `get_noc_addr(0, input_tensor_addrgen)`, `noc_async_read_barrier`, and `DPRINT`s the first 16 bytes. Per-launch static budget of 16 probes. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp` | `std::getenv("SGLANG_TT_U17_PROBE_RS_PRE")`; if `"1"`, set `reader_compute_defines["SGLANG_TT_U17_PROBE_RS_PRE"] = "1"`. |

Both env-gated. With env unset, no behavior change (the kernel
emits no extra ops; the program-factory emits no extra defines).
Canonical Qwen3-8B verified: `"What is 2+2? What is "` at 9.0s
e2e_latency.

## Hardware evidence

Repro:

```bash
# Inside p3a-ngram, after rebuild_tt_metal_kernels.sh ttnn:
PYTHONUNBUFFERED=1 \
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_MAX_BATCH=1 \
HF_MODEL=/models/Qwen3-8B \
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
TT_METAL_DPRINT_CORES=all \
python3 -u -m sglang.launch_server ... &

# After server up:
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":3,"temperature":0.0}}'
# Result: garbage " Dat抄袭/Resources" (U11 zero-weight signature).

# Then:
grep "U17_PRE_RS" /tmp/u17_phase0.log | wc -l
# 63360 probes (across many cores + many decode steps).

grep "0xa6700.*NONZERO" /tmp/u17_phase0.log | wc -l
# 528

grep "0xa6700.* zero\]" /tmp/u17_phase0.log | wc -l
# 624 (these correspond to RS calls where the input has just been
#       allocated and is genuinely zero).
```

Sample NONZERO probe output (worker core `0:1-2`, RISC `NC`,
device 0):

```
[U17_PRE_RS in_addr=0xa6700 noc=0x1c20000a6700
            w0=0xb6d6be41 w1=0x842545c0
            w2=0x33133bdc w3=0xa9908bd2 NONZERO]
```

Interpreting `0xb6d6be41` as packed BF16:
- High 16: `0xb6d6` ≈ -6.4e-3
- Low 16: `0xbe41` ≈ -0.188

These are typical post-quantization activation magnitudes for
Qwen3-8B's residual stream. Pattern varies across decode steps
but is internally consistent within a single step — exactly the
fingerprint of a stale L1 buffer being reused for a different
tensor.

## Distinct byte patterns observed

The top 10 most-common 16-byte sequences at L1 `0xa6700` across
the run:

```
24x w0=0xd3285f35 w1=0xcd929dbb w2=0x1d879c92 w3=0x9d622223
24x w0=0xd2e75ef5 w1=0xcd529d6e w2=0x1d3b9c4f w3=0x9d2721e8
24x w0=0xd28e5e96 w1=0xcd149ce1 w2=0x1d019c18 w3=0x9cfb21cb
16x w0=0xd3435f5d w1=0xcda89dde w2=0x1d9a9ca5 w3=0x9d7e2237
...
```

The 24-count rows match the 24-worker-core count for the
reduce_scatter dispatch (8 cores × 2 directions × ... yielding
`num_workers_per_link * num_links` ≈ 24). I.e. all worker cores
observed the SAME bytes on the SAME decode step — confirming
they're reading from the SAME L1 region (no per-core data race;
this is a global stomp).

## What this rules out / refines

### Rules out: U16 Phase-5c GlobalSemaphore handshake (as originally specified)

The original plan: matmul writer kernel does
`noc_semaphore_inc(sema_on_worker, 1)` after PACK + writes-flushed;
RS reader does `noc_semaphore_wait_min(sema_local, expected_count)`
before first `noc_async_read`. The implicit assumption was that the
L1 region at `0xa6700` was correct (= zero under zero-weights) at
some point and only became NONZERO due to an in-flight PACK racing
with the reader's NoC read.

Phase-0 evidence falsifies the assumption: L1 `0xa6700` is NONZERO
at RS reader entry, BEFORE the reader fires any `noc_async_read`.
A semaphore the reader waits on would only delay the (subsequent)
noc_async_read — it cannot reach back in time and change L1's
state before kernel_main() began.

### Standing — refined hypotheses for U18

| ID | Hypothesis | Test to falsify |
|---|---|---|
| **P1** | W2 matmul never actually writes ZERO to L1 `0xa6700` under the gathered/prefetcher path. `mm_out_cb` is locally-allocated via `set_globally_allocated_address(*out_buffer)` — the gathered code path may take a branch that skips PACK→L1 entirely for certain tile-shape/grid combinations, leaving L1 untouched. | Run U13 PACK probe on the W2 matmul ELF specifically (per-ELF tag). U13 has previously fired with `zero` results — but ALL 528 NONZERO bytes at `0xa6700` were on the W2→RS path, which the existing U13 didn't disambiguate from WQKV→AR. Re-run U13 with a per-tensor-shape ELF filter and confirm W2's PACK actually executes. |
| **P2** | W2 output is at a DIFFERENT L1 address than where RS reads. The RS reader's `input_tensor_address=0xa6700` is wrong / stale from a prior allocation. The host-side L1 buffer registry tracks `0xa6700` as w2_out, but the actual W2 matmul output buffer is at a different address (e.g., per-iteration variation). | Per-decode-step Python probe: print BOTH `w2_out.buffer_address()` AND `rs_input.buffer_address()` for every iteration, NOT just the first. If they differ across iterations or differ from each other within an iteration, the RS reader is reading the wrong L1. |
| **P3** | W2 matmul writes ZERO, but some intervening kernel (an op between W2 and RS — e.g., a layout conversion, deallocation/realloc, another CCL op, an idle prefetcher data movement) stomps L1 `0xa6700` with prior-iteration activations between W2 completion and RS reader start. | Trace replay's dispatched program order: dump `MeshWorkload`'s program sequence around W2 → RS for layer 0. Identify any program that touches `0xa6700` core's L1 between W2's kernel exit and RS reader's kernel entry. |

P3 is the closest cousin to the original race hypothesis, but at
a different scope: the racer would be a separate KERNEL, not the
RS reader's own NoC read. The fix would be different too — either
re-order the dispatch, eliminate the intervening kernel, or use a
semaphore wait inside that intervening kernel (not inside the RS
reader).

## What this means for the multi-day U17 implementation plan

The C++ GlobalSemaphore handshake implementation (matmul-writer
`noc_semaphore_inc` + RS-reader `noc_semaphore_wait_min` + op-attrs
hash + Python wiring) was scoped at 2-3 sessions in U16. Phase-0
shows that even if perfectly implemented, it will not fix the
garbage — the wait would unblock against an already-stomped L1.

**Recommendation: do NOT proceed with U17 Phases 1-9.** Instead,
dispatch U18 to systematically falsify P1/P2/P3 in that order, with
the same per-phase budget discipline that worked for U12-U16.

## Hypothesis ledger (post-U17 Phase-0)

| ID | Suspect | Pre-U17 | Post-U17 |
|---|---|---|---|
| S1-S10, Lead 2-3, U3, U4-A, U4-B | various | RULED OUT | unchanged |
| U1 cross-sub-device WAIT_STREAM gap | LOCATED, U16 Phase-5a global BARRIER tested | RULED OUT | unchanged |
| U5 / U16 Phase-2 receiver-subdev routing or Synchronize | RULED OUT (persistent-kernel deadlock) | unchanged |
| U16 Phase-5b matmul-PACK kernel-exit tensix_sync | RULED OUT | unchanged |
| **U16 Phase-5c device-side GlobalSemaphore PACK→reader handshake** | STANDING — TOP PRIORITY | **RULED OUT (U17 Phase-0)** — Phase-0 evidence shows the stomp acts BEFORE RS reader runs; a semaphore wait inside the reader cannot reach back in time. |
| **P1 — W2 gathered path skips PACK→L1 for some ELF/shape combo** | (new) | **STANDING — high priority** |
| **P2 — W2 output's actual L1 ≠ `0xa6700`; RS reader reads stale L1** | (new) | **STANDING — high priority** |
| **P3 — intervening kernel stomps L1 `0xa6700` between W2 and RS** | (new) | **STANDING — secondary** |

## Working state at session end

- tt-metal-sglang HEAD: **`2f48d0e5cec`** (U17 Phase-0 PRE-RS
  probe commit; env-gated; default-off; canonical bytewise-equal)
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/*
  fork only per project policy)
- sglang HEAD: pending this doc commit
- Container `/tt-metal/`: synced (probe `.cpp` files + rebuilt
  `_ttnn.so` / `_ttnncpp.so`)
- `/root/.cache/tt-metal-cache/*`: cleared at session end
- `stash@{0,1,2}`: untouched
- TT devices: healthy (`tt-smi -r 0,1` performed once during
  this session to clear a stuck dispatcher)
- Server: stopped at session end (canonical sanity-check passed:
  `"What is 2+2? What is "` at 9.0s e2e)
- Probe DPRINT scope: `TT_METAL_DPRINT_CORES=all` (this captures
  all worker cores' probe output; required because the RS reader
  workers are NOT at logical `(0,0)` where the default is)

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal C++ files,
      env-gated)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Container is NOT bind-mounted — used
      `rebuild_tt_metal_kernels.sh` (not host-side rebuild)
- [x] Canonical re-verified post-rebuild: coherent output
- [x] All new probes env-gated (`SGLANG_TT_U17_PROBE_RS_PRE`)
- [x] tensix_sync() not required for U17 probe (NoC reads with
      `noc_async_read_barrier` provide the necessary ordering)

## Shipping verdict (unchanged from U16)

Canonical Qwen3-8B (no prefetcher) remains the production
shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10)
chat ≈ 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**
  Trace-replay race persists despite U5, U16 Phase 2, U16 Phase
  5a, U16 Phase 5b, and U17 Phase-0 evidence advance.
- **DO NOT ship `SGLANG_TT_U17_PROBE_RS_PRE=1`.** Diagnostic
  probe only; adds DPRINT overhead per RS dispatch.

## U18 dispatch recommendation

Run the three falsification tests in P1 → P2 → P3 order, each
with the same per-phase scope discipline that worked for U12-U16:

1. **P1 test (~1 session)**: Re-enable
   `SGLANG_TT_PREFETCHER_PACK_PROBE` (U13 part 1) under zero
   weights + prefetcher + W2-specific ELF tag filter. Confirm
   whether the W2 matmul's PACK kernel actually fires AND writes
   zero to `mm_out_cb`.
   - If PACK does NOT fire on the W2 ELF → P1 is the bug. Fix
     by understanding what gathered-path branch causes the skip
     (could be a tile-shape / grid combo not handled).
   - If PACK fires and writes zero → P1 is RULED OUT, proceed to P2.

2. **P2 test (~1 session)**: Sglang Python probe in `mlp.py`:
   for the first 16 decode iterations, print `w2_out.buffer_address()`
   AND `_rs_input.buffer_address()` (the latter computed inside
   `tt_all_reduce` and passed via a closure or env-gated re-read).
   If they ever disagree, or if `w2_out.buffer_address()` is not
   `0xa6700`, P2 is the bug. Fix by understanding the L1 allocator
   policy under prefetcher + trace-replay.
   - If both consistently equal `0xa6700` → P2 is RULED OUT,
     proceed to P3.

3. **P3 test (~1-2 sessions)**: Dump `MeshWorkload`'s program
   sequence for layer 0 W2 → RS span. Identify any program with
   any cores overlapping the W2-output core grid between W2's
   kernel-exit stream-counter increment and RS reader's
   kernel-start stream-counter consume. Implement a probe in
   that program (or right before/after it) reading L1 `0xa6700`
   on the affected cores. If the bytes transition from zero to
   the observed activation-like values, P3 is the bug. Fix
   options:
   - re-order dispatch (move the intervening program before W2)
   - eliminate the intervening program (likely a layout op)
   - add a per-program semaphore wait

## Commits this session

- (tt-metal-sglang) **`2f48d0e5cec`** —
  `prefetcher: U17 Phase-0 PRE-RS L1 probe — stomper is upstream of RS reader (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit
