# TT Qwen3-8B prefetcher — U18: P1 + P2 RULED OUT, P3 CONFIRMED; prefetcher writer_l1 RULED OUT as the stomper — 2026-05-25

Status: **EVIDENCE_ADVANCE — Three independent probes under FULL
prefetcher + trace replay + zero weights confirm that L1 0xa6700 is
ZERO when the W2 matmul kernel EXITS (U14 CONSUMER_PROBE, 6912/6912
zero) and ZERO when the W2 PACK writes it (U18 PACK_PROBE, 34560/34560
zero, 6912/6912 at L1 0xa6700 specifically zero), yet NONZERO ~46% of
the time when the RS reader runs (U17 PRE_RS, 528 NONZERO + 624 zero
at L1 0xa6700). Combined with the Python `w2_out.buffer_address()`
trace showing stable 0xa6700 across both compile-run and trace-capture
decode iters, this RULES OUT P1 (PACK doesn't fire / writes wrong)
and P2 (stale address) and CONFIRMS P3 (an intervening device-side
write between W2 program exit and RS reader's first NoC read stomps
L1 0xa6700 on the receiver cores). A follow-on probe in the
prefetcher writer_l1 kernel (8192 entries logged) shows fifo_start =
0xac700 and wr_ptr advances forward within [0xac700, 0xac700 +
0xCC000); NONE of the prefetcher mcast writes target 0xa6700, so the
prefetcher writer RULED OUT as the stomper. KERNEL_CONFIG region is
allocated at MEM_MAP_END (~0x4000) with default 69KB size, ending
~0x15000, well below 0xa6700 — dispatcher kernel-binary writes also
RULED OUT. Next-layer-W1 / cross-sub-device dispatch race was already
RULED OUT by U16 Phase 5a (cached-path BARRIER + bug persists).
Remaining standing suspect: a kernel inside the SAME RS program
(writer or reduction kernel) that touches receiver-cores' L1 on
0xa6700, OR a hidden L1 buffer reuse not visible via Python
`.buffer_address()`.**

Continuation of `tt_qwen3_8b_prefetcher_U17_pre_rs_probe_stomper_upstream_2026-05-25.md`
(U17 narrowed the bug to "stomper acts BEFORE RS reader runs"; the
GlobalSemaphore PACK→reader handshake plan was invalidated).

tt-metal-sglang HEAD: **`8157e192efd`** (U18 probes landed;
default-off; canonical bytewise-equal).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (post-U18)

| Step | Question | Probe / Source | Result |
|---|---|---|---|
| 1 | Does W2 PACK fire under TRACE replay (not just eager)? | U18_PACK_PROBE under FULL prefetcher + trace | YES — 34560 PACK_OUT entries across all 4 gathered ELFs |
| 2 | Does W2 PACK write zero to L1 0xa6700 under trace? | U18_PACK_PROBE, filter `l1=0xa6700` | YES — 6912/6912 entries at 0xa6700 are ZERO across ELFs 0x2d96e + 0x2df6a |
| 3 | Is w2_out.buffer_address() stable across decode iters? | U18_ADDR_PROBE in mlp.py | YES — both iter 7 (compile-run) AND iter 8 (trace-capture) print `0xa6700` |
| 4 | Is L1 0xa6700 zero at matmul kernel EXIT under trace? | U14_END (CONSUMER_PROBE) budget bumped to 2048 | YES — 6912/6912 entries at 0xa6700 are ZERO |
| 5 | Is L1 0xa6700 zero at RS reader ENTRY under trace? | U17_PRE_RS budget bumped to 2048 | NO — 528/1152 NONZERO at 0xa6700 (~46% stomped, racy) |
| 6 | Does the prefetcher writer_l1 mcast to 0xa6700? | U18_PREFETCHER_WRITE_PROBE in writer_l1 | NO — fifo_start=0xac700; 8192 logged wr_ptr values are all ≥0xac700 |

**Punchline:** W2 PACK writes 0 → matmul kernel exits with L1 = 0 →
something writes nonzero to L1 0xa6700 → RS reader reads nonzero ~46%
of the time. The stomper is NOT the prefetcher writer_l1, NOT the
dispatcher's KERNEL_CONFIG region (off-region), and NOT the next-layer
W1 (U16 Phase 5a redo with global BARRIER did not fix the bug).

## Phase 1 — P1 falsification (W2 PACK)

### Probe

Added `SGLANG_TT_U18_PACK_PROBE` — large-budget (2048) version of
U13's PACK probe. Wired in:

  - `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`:
    new `#ifdef SGLANG_TT_U18_PACK_PROBE` block right after the U13
    PACK probe block, same `PACK((...))` reader pattern, but with
    `u18_out_budget = 2048` and `u18_out_total` running counter for
    the total PACK fire count per binary.
  - `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`:
    env-gated propagation of `SGLANG_TT_U18_PACK_PROBE` define to
    the kernel.

### Run

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged
SGLANG_TT_MAX_BATCH=1
HF_MODEL=/models/Qwen3-8B
SGLANG_TT_USE_PREFETCHER=1
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1
SGLANG_TT_U18_PACK_PROBE=1
SGLANG_TT_U17_PROBE_RS_PRE=1
TT_METAL_DPRINT_CORES=all
# NOTE: no SGLANG_TT_DISABLE_PREFILL_TRACE; trace fully active.
```

### Result

| ELF | total entries | NONZERO | zero |
|---|---:|---:|---:|
| 0x2d96e | 6912 | 0 | 6912 |
| 0x2db6e | 13824 | 0 | 13824 |
| 0x2de6b | 6912 | 0 | 6912 |
| 0x2df6a | 6912 | 0 | 6912 |
| **Total** | **34560** | **0** | **34560** |

L1 addresses written by PACK per ELF:

| ELF | Top L1 addrs (counts) |
|---|---|
| 0x2d96e | `0xa6700 (4608)` / `0xa4700 (2304)` |
| 0x2db6e | `0xa5700 (2496)` / `0xa2700 (2496)` / `0xa3700 (2304)` / `0xa0700 (2304)` / `0xa1700 (2112)` |
| 0x2de6b | `0xaaf00 (2304)` / `0xa8f00 (2304)` / `0xa6f00 (2304)` |
| 0x2df6a | `0xaa700 (2304)` / `0xa8700 (2304)` / `0xa6700 (2304)` |

The two W2-related ELFs (`0x2d96e` and `0x2df6a`) write **6912 zero
entries to L1 0xa6700**. PACK definitively fires under trace replay
AND writes zero to mm_out_cb at L1 0xa6700.

**P1 RULED OUT.**

## Phase 2 — P2 falsification (W2 address stability)

### Probe

Added `SGLANG_TT_U18_ADDR_PROBE` env gate in
`models/tt_transformers/tt/mlp.py` — prints `w2_out.buffer_address()`
on EVERY layer-0 forward (compile-run + trace-capture). Trace replay
itself doesn't run Python, but trace-capture inside
`_capture_decode_trace_text` runs the same Python path once, so we
get one decode-shape print per trace capture.

### Run

Same env as Phase 1 but with `SGLANG_TT_U18_ADDR_PROBE=1` and
`SGLANG_TT_U17_PROBE_RS_PRE=1`; `max_new_tokens=8` to ensure decode
fires.

### Result

```
[U18_W2_ADDR] iter=1 layer=0 w2_out.addr=0x47b75640 shape=(1, 1, 128, 4096) shard_grid=None        # prefill
[U18_W2_ADDR] iter=2 layer=0 w2_out.addr=0x47ba5640 shape=(1, 1, 128, 4096) shard_grid=None        # prefill
[U18_W2_ADDR] iter=3 layer=0 w2_out.addr=0x47d67e40 shape=(1, 2, 512, 4096) shard_grid=None        # prefill
[U18_W2_ADDR] iter=4 layer=0 w2_out.addr=0x47e67e40 shape=(1, 2, 512, 4096) shard_grid=None        # prefill
[U18_W2_ADDR] iter=5 layer=0 w2_out.addr=0x480b3e40 shape=(1, 4, 512, 4096) shard_grid=None        # prefill
[U18_W2_ADDR] iter=6 layer=0 w2_out.addr=0x4845be40 shape=(1, 8, 512, 4096) shard_grid=None        # prefill
[U18_W2_ADDR] iter=7 layer=0 w2_out.addr=0xa6700 shape=(1, 1, 32, 4096) shard_grid={...32 cores}   # decode compile-run
[U18_W2_ADDR] iter=8 layer=0 w2_out.addr=0xa6700 shape=(1, 1, 32, 4096) shard_grid={...32 cores}   # decode trace-capture
```

Both decode-shape iterations (compile-run + trace-capture) print
`0xa6700`. Trace replay subsequently uses the commands captured at
iter 8, so the device-side `input_tensor_address` in RS reader is the
same `0xa6700` we measured.

**P2 RULED OUT.**

## Phase 3 — P3 confirmation (intervening kernel)

### Probe

Bumped existing U14 CONSUMER_PROBE budget from 16 to 2048 + added
`u14_end_total` running counter. Bumped U17 PRE_RS budget from 16
to 2048 + added `u17_pre_total` running counter.

### Run

Same env as Phase 1 but enabled all of:
`SGLANG_TT_U18_PACK_PROBE=1`,
`SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1`,
`SGLANG_TT_U17_PROBE_RS_PRE=1`,
`SGLANG_TT_U18_ADDR_PROBE=1`.

### Result

| Probe | Site in pipeline | Entries at L1 0xa6700 | NONZERO | zero |
|---|---|---:|---:|---:|
| U18_PACK | inside W2 matmul kernel, immediately after `pack_tile_block` | 6912 | 0 | 6912 |
| U14_END | inside W2 matmul kernel, just before kernel exit | 6912 | 0 | 6912 |
| U17_PRE_RS | inside RS reader kernel, very first thing kernel_main does | 1152 | **528** | 624 |

The W2 matmul kernel ALWAYS exits with L1 0xa6700 = ZERO under trace
replay. The RS reader's first read at L1 0xa6700 is NONZERO ~46% of
the time (racy).

NoC-encoding analysis on the U17 NONZERO entries: all 528 NONZERO
reads target the SAME NoC address `0x1c20000a6700` = receiver core
(x=2, y=7) at L1 0xa6700. The 624 zero entries also target the SAME
core's same L1. Per-worker-core analysis: each of the 16 worker
cores observed both NONZERO and zero reads (~33 NONZERO + ~39 zero
per worker-core) across the 72 RS dispatches captured.

Thus the stomp is a per-decode-step race against an unknown writer
on the receiver core; sometimes the RS reader wins (zero), sometimes
the stomp wins (nonzero). The bug is racy at this scale.

**P3 CONFIRMED.**

## Phase 3 follow-up — find the stomper

### Suspect set

1. Prefetcher writer_l1 mcasting to receiver-cores' L1 (writes to
   GlobalCB at 0xac700+)
2. Dispatcher kernel-binary writes to KERNEL_CONFIG region
3. Next layer's W1/W3 matmul (pipelined cross-sub-device race)
4. One of the RS program's other kernels (writer / reduction)

### Falsification — prefetcher writer_l1

Added `SGLANG_TT_U18_PREFETCHER_WRITE_PROBE` — DPRINT of
`fifo_wr_ptr` + `fifo_start` for every `remote_cb_push_back_and_write_pages`
in the prefetcher writer_l1 kernel. Propagated via env in the
prefetcher program factory.

Result (8192 log entries):

```
fifo_start_addr = 0xac700 (uniform)
fifo_wr_ptr values: 0xac700, 0x100d00, 0x102800, 0x104300, ...
                    (all in range [0xac700, 0xac700 + 0xCC000))
```

**No prefetcher write targets 0xa6700.** Prefetcher writer_l1
mcast RULED OUT as the stomper.

### Falsification — dispatcher KERNEL_CONFIG region

Blackhole tensix `KERNEL_CONFIG` = `MEM_MAP_END` (~0x4000) with
default 69KB size, ending ~0x15000. L1 0xa6700 = ~680KB is well
above this region. Dispatcher kernel-binary writes can't reach
0xa6700. **RULED OUT.**

### Falsification — next layer W1 (cross-sub-device race)

Already RULED OUT by U16 Phase 5a (cached-path BARRIER forces global
sync between every program in trace replay; bug persists with this
override). If the stomper were next-layer-W1, the global BARRIER
would serialize it after layer N's RS, but it does not fix the bug.

### Standing — RS program's other kernels

Not yet probed. The RS program includes:
 - `line_reduce_scatter_minimal_async_reader.cpp` (the reader we
   probed)
 - `line_reduce_scatter_minimal_async_writer.cpp` (writes to output
   addr; doesn't appear to write back to input)
 - `line_reduction.cpp` (compute kernel; writes to local `output_cb`)

The writer writes to `output_addrgen` (the RS output tensor, NOT the
input). The reduction kernel writes to a local CB on worker cores.
Neither obviously writes to the receiver cores' L1 0xa6700. But the
RS program may also run kernels we haven't enumerated (e.g.,
intermediate-CB management on the receiver side).

## What U18 lands (all env-gated; canonical bytewise-equal)

| File | Change |
|---|---|
| `models/tt_transformers/tt/mlp.py` | `SGLANG_TT_U18_ADDR_PROBE` per-iteration `w2_out.buffer_address()` print, layer-0 only. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | `SGLANG_TT_U18_PACK_PROBE` block (2048-budget post-pack L1 read on gathered path); bumped `U14_CONSUMER_PROBE` budget from 16 to 2048 + running total counter. |
| `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | env-gated `SGLANG_TT_U18_PACK_PROBE` propagation. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp` | bumped U17 PRE_RS budget from 16 to 2048 + running total counter. |
| `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp` | `SGLANG_TT_U18_PREFETCHER_WRITE_PROBE` prints `fifo_wr_ptr / fifo_start` before each `remote_cb_push_back_and_write`. |
| `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp` | env-gated `SGLANG_TT_U18_PREFETCHER_WRITE_PROBE` propagation. |

## Hypothesis ledger (post-U18)

| ID | Suspect | Pre-U18 | Post-U18 | Evidence |
|---|---|---|---|---|
| S1-S10, Lead 2-3, U3, U4-A, U4-B | various | RULED OUT | unchanged | — |
| U1 cross-sub-device WAIT_STREAM gap | RULED OUT | unchanged | — |
| U5 / U16 Phase 2 receiver-subdev routing or Synchronize | RULED OUT (persistent-kernel deadlock) | unchanged | — |
| U16 Phase 5a cached-path BARRIER | RULED OUT | unchanged | — |
| U16 Phase 5b matmul-PACK kernel-exit tensix_sync | RULED OUT | unchanged | — |
| U16 Phase 5c device-side GlobalSemaphore PACK→reader handshake | RULED OUT (U17) | unchanged | — |
| **U17 P1 — W2 gathered path skips PACK→L1 for some ELF/shape combo** | STANDING (high priority) | **RULED OUT** | U18 Phase 1: 34560 PACK_OUT entries all zero; 6912 specifically at L1 0xa6700 |
| **U17 P2 — W2 output's actual L1 ≠ 0xa6700; RS reads stale L1** | STANDING (high priority) | **RULED OUT** | U18 Phase 2: compile-run + trace-capture both print 0xa6700 |
| **U17 P3 — intervening kernel stomps L1 0xa6700 between W2 exit and RS reader** | STANDING (secondary) | **CONFIRMED** | U18 Phase 3: U14_END=zero, U17_PRE_RS=46% NONZERO at same L1 |
| **U18-followup: prefetcher writer_l1 mcast is the stomper** | (new) | **RULED OUT** | 8192 logged wr_ptr values; none target 0xa6700 |
| **U18-followup: dispatcher KERNEL_CONFIG writes hit 0xa6700** | (new) | **RULED OUT** | KERNEL_CONFIG ends ~0x15000, off-region from 0xa6700 (~680KB) |
| **U18-followup: next-layer W1/W3 pipelined dispatch stomp** | (new — variant of U16 5a) | **RULED OUT** | U16 5a cached-path global BARRIER already tested, bug persists |
| **U18 STANDING — RS program's own writer / reduction / hidden kernel writes back to receiver L1 0xa6700** | (new) | **STANDING — TOP PRIORITY** | — |
| **U18 STANDING — hidden L1 buffer reuse not visible via Python `.buffer_address()`** | (new) | **STANDING — TOP PRIORITY** | — |

## Reproducer (working)

```bash
# Phase-1 PACK probe (P1 falsification):
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U18_PACK_PROBE=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u18p1_pack_trace.log \
python3 -u -m sglang.launch_server ... &
curl -s -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"2+2=","sampling_params":{"max_new_tokens":3,"temperature":0.0}}'
grep "U18_PACK" /tmp/u18p1_pack_trace.log | grep -c NONZERO       # 0
grep "U18_PACK" /tmp/u18p1_pack_trace.log | grep -c " zero]"      # 34560
grep "U18_PACK.*l1=0xa6700" /tmp/u18p1_pack_trace.log | grep -c " zero]"   # 6912

# Phase-2 ADDR probe (P2 falsification):
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U18_ADDR_PROBE=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
python3 -u -m sglang.launch_server ... &
curl ...
grep "U18_W2_ADDR" /tmp/u18p2_server.log
# iter 7 + iter 8 both: w2_out.addr=0xa6700

# Phase-3 combined (P3 confirmation):
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U18_PACK_PROBE=1 \
SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
SGLANG_TT_U18_ADDR_PROBE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u18p3.log \
python3 -u -m sglang.launch_server ... &
curl ...
grep "U14_END_OUT.*l1=0xa6700" /tmp/u18p3.log | grep -c " zero]"  # 6912
grep "U17_PRE_RS in_addr=0xa6700" /tmp/u18p3.log | grep -c NONZERO  # 528
grep "U17_PRE_RS in_addr=0xa6700" /tmp/u18p3.log | grep -c " zero]"  # 624

# Phase-3 follow-up — prefetcher writer probe:
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U18_PREFETCHER_WRITE_PROBE=1 \
TT_METAL_DPRINT_CORES=all \
python3 -u -m sglang.launch_server ... &
curl ...
grep "U18_PREF_WR" /tmp/u18p3_pref_write.log | sed -E 's/.*wr_ptr=(0x[a-f0-9]+).*/\1/' | sort -u
# All values ≥ 0xac700; none at 0xa6700.

# Canonical (no probes, no prefetcher) re-verify:
HF_MODEL=/models/Qwen3-8B \
python3 -u -m sglang.launch_server ... &
curl -s -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Output: " What is 2+2? What is 2+2? What"
# e2e_latency: 9.08s
```

## U19 dispatch recommendation

The remaining suspect set is small. Two tractable next attacks:

### U19 Phase A — probe RS program's writer + reduction kernels

Add per-kernel probes (analogous to U17 PRE_RS) in:
 - `line_reduce_scatter_minimal_async_writer.cpp` at kernel_main
   entry: dump first 16 bytes at the RS reader's `input_tensor_address`
   via a NoC read.
 - `line_reduction.cpp` at kernel_main entry: same.

If either fires the NONZERO bytes BEFORE the reader probe, that
kernel is the stomper. Time budget: 1 session.

### U19 Phase B — host-side L1 polling

Add a tiny new kernel that runs on the worker_sub_device (no
persistent kernels there), takes a target NoC address + L1 offset as
runtime args, and polls L1 at that address every N cycles, printing
the value if it changes. Dispatch this kernel BETWEEN W2 and RS in
the Python pipeline. The transitions zero→nonzero and nonzero→zero
will pinpoint the exact write event. Time budget: 1-2 sessions.

### Speculative fix to try alongside

Force W2 matmul output to a DIFFERENT L1 address (e.g., use an
explicit `set_globally_allocated_address(...)` with a higher L1
offset like 0x180000, well above GlobalCB). If the bug moves to
that new address, the stomp is L1-allocator-dependent rather than
addr-specific. If the bug disappears, 0xa6700 is special (likely
overlaps with some firmware/sub-device state region we haven't
mapped).

## Working state at session end

- tt-metal-sglang HEAD: **`8157e192efd`** (U18 commit; env-gated;
  default-off; canonical bytewise-equal).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/*
  fork only per project policy).
- sglang HEAD: pending this doc commit.
- Container `/tt-metal/`: synced (PACK + CONSUMER + ADDR +
  PREFETCHER_WRITE probes applied; rebuilt _ttnn.so / _ttnncpp.so).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `"What is 2+2? What is 2+2? What"` at
  9.08s e2e_latency (bug absent; coherent output; no regression).

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal files, all env-gated)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Container is NOT bind-mounted — used `rebuild_tt_metal_kernels.sh`
- [x] Canonical re-verified: coherent output
- [x] All new probes env-gated (`SGLANG_TT_U18_*`)
- [x] `tensix_sync()` before all L1 reads in PACK probes

## Shipping verdict (unchanged from U17)

Canonical Qwen3-8B (no prefetcher) remains production shipping config
at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≈ 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**
  Trace-replay race persists across U5, U16 Phase 2 / 5a / 5b / 5c,
  U17 Phase-0, and U18 Phase 1-3.
- **DO NOT ship any `SGLANG_TT_U18_*` probe.** Diagnostic only;
  adds DPRINT overhead.

## Commits this session

- (tt-metal-sglang) **`8157e192efd`** —
  `prefetcher: U18 — P1/P2 RULED OUT, P3 CONFIRMED via PACK+CONSUMER+ADDR+PREF_WRITE probes (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.
