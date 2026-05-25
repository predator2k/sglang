# TT Qwen3-8B prefetcher — U19: FILL W2 ZERO workaround CONFIRMS L1 0xa6700 stomp is the bug; RS writer + mux + L1-cache ruled out as stomper — 2026-05-24

Status: **EVIDENCE_ADVANCE — Five independent U19 attacks land
new ruling-out + ruling-in evidence.  Critical positive finding:
inserting `ttnn.fill(w2_out, 0)` in Python between W2 matmul and
`tt_all_reduce` (env `SGLANG_TT_U19_FILL_W2_ZERO=1`) eliminates the
L1 0xa6700 NONZERO race entirely — U17 PRE_RS probe goes from
528/1152 NONZERO to **0/1920 NONZERO** at L1 0xa6700, and zero-weight
decode output collapses to the deterministic U11 constant-token
signature (token 38649 looped) instead of the racy varying garbage.
This PROVES the bug is exactly the L1 0xa6700 stomp between W2 PACK
exit and RS reader entry, AND that re-writing zero into L1 0xa6700
via a captured trace op overrides the stomp.  ttnn.clone /
ttnn.copy(w2_out, w2_out) (no-op data-preserving) does NOT fix it —
the FILL fix is data-destructive (overwrites W2's value with zero),
which is why it doesn't ship for real weights.  Three of the
remaining stomper candidates from U18 are now ruled out as well:
RS WRITER kernel (intermediate=DRAM, output=0xa4700/0xa8700/0xaa700,
never 0xa6700), RS MUX kernel (zeros only 0x1b220–0x1b310), and
RISC-V L1 data-cache invalidate fence (invalidate before every
noc_async_read does not change the 46% NONZERO rate).**

Continuation of `tt_qwen3_8b_prefetcher_U18_P1P2_RULED_OUT_P3_CONFIRMED_2026-05-25.md`.

tt-metal-sglang HEAD: **`fa35e4144a3`** (U19 commit; all probes
+ workarounds env-gated; default-off; canonical bytewise-equal).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (post-U19)

| Step | Question | Probe / Workaround | Result |
|---|---|---|---:|
| 1 | Does the RS writer write to L1 0xa6700? | SGLANG_TT_U19_WRITER_PROBE (intermediate+output addr dump) | NO — inter=0x47b35640 (DRAM); out=0xa4700/0xa8700/0xaa700; never 0xa6700 |
| 2 | Does the RS mux clear L1 0xa6700? | SGLANG_TT_U19_MUX_PROBE | NO — mux only zeros 0x1b220 / 0x1b2b0 / 0x1b2d0 / 0x1b2f0 (all <<0xa6700) |
| 3 | Is a RISC-V L1 data-cache invalidate the fix? | SGLANG_TT_U19_INVALIDATE_CACHE (fence before every noc_async_read) | NO — 528/1152 NONZERO ratio at 0xa6700 UNCHANGED |
| 4 | Are NoC writes to producer L1 0xa6700 coherent and persistent? | SGLANG_TT_U19_FORCE_PRODUCER_ZERO + readback | YES — after writeback, readback always zero (1152/1152 zero, 0 NONZERO) |
| 5 | Does `ttnn.fill(w2_out, 0)` in Python between W2 and RS fix the stomp? | SGLANG_TT_U19_FILL_W2_ZERO | YES — U17 PRE_RS NONZERO=0/1920 (vs baseline 528/624); zero-weight decode collapses to deterministic U11 (token 38649 looped) |
| 6 | Does `ttnn.copy(w2_out, w2_out)` (no-op data-preserving) fix it? | SGLANG_TT_U19_COPY_W2 | NO — real-weight output still garbage; the FILL fix is data-destructive, not just a dispatch barrier |

**Punchline:** L1 0xa6700 on receiver core (2,7) is GENUINELY
NONZERO between W2 PACK exit and RS reader entry.  Writing zero
back to that L1 via either (a) Python `ttnn.fill` (overrides the
stomp) or (b) reader-kernel NoC writeback (verifiable via readback)
both succeed, confirming L1 is writable and coherent — the bytes
truly arrive from some intervening write between W2 PACK and the
RS reader's noc_async_read.  The stomper has been narrowed to a
small remaining candidate set.

## Phase 1 — FORCE_ZERO in reader (workaround test 1)

### Probe

Added `SGLANG_TT_U19_FORCE_ZERO` in `line_reduce_scatter_minimal_async_reader.cpp`.
After `noc_async_read_barrier()` in both "is_first_device_in_direction"
and "not first" branches, overwrites the freshly-loaded local CB
bytes with ZERO (both input and intermediate L1 regions).  Compute
kernel then sums zero into the reduction.

### Result (zero weights + prefetcher + FORCE_ZERO)

```
{"text":" DatIessen的带领下 нуж.hours县公安局nposnposnpos瘦身创客 нуж нуж нуж",...}
```

Garbage starts with " Dat" — same U11 zero-weight signature class
as canonical zero-weight runs.  But still non-deterministic
varying signature (different from baseline U18).

**Interpretation:** zeroing the reader's local L1 makes the reducer
sum zero, but the W2 contribution to other layers' residuals is
still garbage downstream.  Phase-1 alone is inconclusive — too
indirect to discriminate.

## Phase 2 — RS writer kernel addr dump

### Probe

Added `SGLANG_TT_U19_WRITER_PROBE` in
`line_reduce_scatter_minimal_async_writer.cpp::kernel_main`,
budgeted to 2048 entries.  Logs `intermediate_address` and
`output_address` at entry.  If either equals 0xa6700, the writer
is a candidate stomper.

### Result

| Unique (intermediate, output) pair | Mode |
|---|---|
| `inter=0x47b35640 (DRAM), out=0xa4700` | decode |
| `inter=0x47b35640 (DRAM), out=0xa8700` | decode |
| `inter=0x47b35640 (DRAM), out=0xaa700` | decode |
| `inter=0x47b95640 (DRAM), out=0x47b5be40 (DRAM)` | prefill |
| (many more DRAM/DRAM pairs in prefill) | prefill |

NONE of the writer's intermediate or output addresses equal
0xa6700 in decode mode.  **RS writer RULED OUT as the stomper.**

## Phase 3 — RS mux kernel zero-region dump

### Probe

Added `SGLANG_TT_U19_MUX_PROBE` in `tt_fabric_mux.cpp::kernel_main`.
At kernel entry the mux clears 4 memory regions per
`FabricMuxConfig::get_memory_regions_to_clear()` —
termination_signal, connection_handshake, flow_control,
buffer_index.  The probe DPRINTs `(address, size)` for each.

### Result

```
[U19_MUX_ENTRY num_regions=4]
[U19_MUX_ZERO addr=0x1b220 size=0x10 i=0]
[U19_MUX_ZERO addr=0x1b2b0 size=0x20 i=1]
[U19_MUX_ZERO addr=0x1b2d0 size=0x20 i=2]
[U19_MUX_ZERO addr=0x1b2f0 size=0x20 i=3]
```

All four regions are at L1 addresses 0x1b220–0x1b310, well below
0xa6700 (≈ 680KB).  Mux kernel runs on cores `0:3-0`, `0:9-0`,
`1:0-0`, `1:6-0` (logical), NOT (2,7).  **RS mux RULED OUT as
the stomper.**

## Phase 4 — L1 cache invalidate fence

### Probe

Added `SGLANG_TT_U19_INVALIDATE_CACHE` in
`line_reduce_scatter_minimal_async_reader.cpp`.  On Blackhole,
`invalidate_l1_cache()` compiles to a RISC-V `fence` instruction
which both invalidates the small L1 data cache (if enabled) and
orders pending memory operations.  Inserts a fence before every
`noc_async_read` of producer L1 in the reader's main loops AND in
the U17 PRE_RS probe.

### Result

```
U17 PRE_RS at 0xa6700: 528 NONZERO / 624 zero  (unchanged from U18)
```

**RULED OUT as the fix.**  The cache theory doesn't fit on
Blackhole anyway — L1 data cache is disabled by default
(`TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS` env unset; firmware-init
calls `set_l1_data_cache<false>()`), and Blackhole's L1 data
cache is "write-through" per the BH bring-up guide — writes go to
both cache and L1 SRAM, so cross-core NoC reads always see the
post-write value.

## Phase 5 — `ttnn.fill(w2_out, 0)` Python workaround **(KEY RESULT)**

### Workaround

Inserted in `mlp.py` between `w2_out = ttnn.linear(...)` and
`w2_out_reduced = tt_all_reduce(w2_out, ...)`:

```python
import os as _u19f_os
if (mode == Mode.DECODE
        and _u19f_os.environ.get("SGLANG_TT_U19_FILL_W2_ZERO", "0") == "1"):
    try:
        ttnn.fill(w2_out, 0)
    except Exception as _u19f_e:
        print(...)
```

This dispatches a `ttnn.fill` (eltwise unary) op on the SAME
producer cores as W2 (cores where w2_out is sharded), captured
into the prefill / decode trace.

### Result A — zero weights + prefetcher + FILL + U17 probe

```
NONZERO at 0xa6700:  0
zero    at 0xa6700:  1920
```

Down from baseline 528/624 NONZERO/zero to **0/1920**.  The
stomp at L1 0xa6700 is eliminated.

### Result B — zero weights + prefetcher + FILL (output check)

```
{"text":" Dat來自,...\n\n此基础上extrextrextrextrextrextrextrextrextrextrextr",
 "output_ids":[21631, 114107, 38225, 114155, 38649, 38649, 38649, ...],...}
```

Tokens collapse to `38649 38649 38649 ...` — the canonical
"zero output → uniform logits → constant token" signature.  This
is exactly what a correct zero-weight model produces (every linear
output zero, every softmax uniform, greedy picks one constant
token).  WITHOUT FILL, output is racy varying garbage (different
tokens per run, e.g. ` " Dat'一个好的IDX掐 ..."` / `" DatIessen的带领下..."` /
`" Und替换аль"`).

### Result C — real weights + FILL (sanity check)

```
{"text":" What''(\"\"0K3&.*/'a", ...}
```

Garbage as expected — FILL destroys W2's contribution.  Confirms
the FILL fix works by re-writing zero, not by inserting a generic
barrier.

**Phase 5 PROVES the bug is the L1 0xa6700 stomp** AND **shows
the bug is fixable by re-writing the correct value into L1 0xa6700
AFTER the stomp fires.**  Under zero weights "correct value" =
zero, so FILL is correct; under real weights "correct value" =
W2's matmul output, which FILL doesn't preserve.

## Phase 6 — `ttnn.copy(w2_out, w2_out)` (data-preserving no-op)

### Workaround

```python
if (mode == Mode.DECODE
        and _u19f_os.environ.get("SGLANG_TT_U19_COPY_W2", "0") == "1"):
    try:
        ttnn.copy(w2_out, w2_out)
    except Exception as _u19cp_e:
        print(...)
```

`ttnn.copy(a, b)` copies `a` → `b`.  Calling with `a == b` reads
the L1 region then writes the same bytes back — a "no-op" that
preserves data but still dispatches a kernel between W2 and RS.

### Result — real weights + COPY

```
{"text":" What十条 sunlight成长为 sunlight sunlight sunlight... ",...}
```

Still garbage.  **COPY does NOT fix the bug.**  This rules out
the "any dispatch barrier suffices" theory — the FILL fix is
specifically about data-destructively overriding the stomp at
L1 0xa6700 with the correct (zero) value.

### Implication

To fix the bug under REAL weights, we need either:
- (a) **Identify and prevent the stomper** (the right answer), OR
- (b) **Re-write W2's correct value into L1 0xa6700 AFTER the
  stomp fires** (requires a captured op that re-runs W2's PACK or
  copies from a backup L1 region — non-trivial).

(b) is hard because W2's correct value lives only at L1 0xa6700;
copying from another buffer would mean writing W2's output twice.

## Phase 7 — NoC writeback coherence + readback (confirming
mechanics)

### Probe

Added `SGLANG_TT_U19_FORCE_PRODUCER_ZERO` in the U17 PRE_RS probe
block.  After the probe reads producer L1 0xa6700 (possibly
NONZERO), it writes 0 back to producer L1 0xa6700 via
`noc_async_write` + `noc_async_write_barrier`, then immediately
re-reads the same address via `noc_async_read` + barrier and
DPRINTs the result.

### Result

```
RBACK NONZERO: 0
RBACK zero:    6336   (across all probe NoC addresses)
RBACK at 0x1c20000a6700 (decode receiver (2,7)): 1152 zero / 0 NONZERO
```

**Every readback shows zero**, including the receiver-core (2,7)
L1 0xa6700.  This confirms:
1. Cross-core NoC writes from the RS reader CAN modify producer
   L1 0xa6700 coherently.
2. The bytes STAY zero after the writeback — no continuous
   stomper.  The stomper fires ONCE between W2 PACK and reader's
   first read (or its U17 probe), not repeatedly.

This means the stomper is a SINGLE write event per RS dispatch,
not a continuous background process.  Combined with Phase 5,
the stomper fires BEFORE the FILL op's dispatch (otherwise the
FILL wouldn't fix it).

## Hypothesis ledger (post-U19)

| ID | Suspect | Pre-U19 | Post-U19 |
|---|---|---|---|
| S1-S10, Lead 2-3, U3, U4-A/B, U1, U5/U16-2/5a/5b/5c, U17 P1/P2 | various | RULED OUT | unchanged |
| **U17 P3 — intervening kernel stomps L1 0xa6700 between W2 exit and RS reader** | CONFIRMED | unchanged | confirmed |
| **U18 — prefetcher writer_l1 mcasts to 0xa6700** | RULED OUT | unchanged | — |
| **U18 — KERNEL_CONFIG region writes hit 0xa6700** | RULED OUT | unchanged | — |
| **U18 — next-layer W1/W3 pipelined dispatch stomp** | RULED OUT | unchanged | — |
| **U19 — RS writer kernel writes to 0xa6700** | (new) | **RULED OUT** | inter=DRAM; out=0xa4700/0xa8700/0xaa700; never 0xa6700 |
| **U19 — RS mux kernel zeros L1 0xa6700** | (new) | **RULED OUT** | zeros only 0x1b220–0x1b310 |
| **U19 — L1 data-cache stale-read race fix via invalidate_l1_cache()** | (new) | **RULED OUT** | 528/1152 NONZERO unchanged |
| **U19 — bug is exactly L1 0xa6700 byte stomp (overridable)** | (new) | **CONFIRMED** | ttnn.fill(w2_out, 0) eliminates NONZERO race (0/1920); zero-weight output collapses to deterministic U11 |
| **U19 — fix is "any dispatch barrier between W2 and RS"** | (new) | **RULED OUT** | ttnn.copy(w2_out, w2_out) (data-preserving) does NOT fix |
| **U19 — stomper is a continuous overwriter** | (new) | **RULED OUT** | NoC writeback + readback always returns zero (no re-stomp) |
| **U19 STANDING — what writes garbage to receiver-core (2,7) L1 0xa6700 between W2 PACK exit (zero) and RS reader entry (46% NONZERO)?** | (new) | **STANDING — TOP PRIORITY** | — |

## Remaining stomper candidate set

After U18 + U19, the stomper is one of:

1. **A kernel inside the W2 program's dataflow set that we haven't
   probed**:
   - `reader_bmm_tile_layout_in0_sender_padding.cpp` (in0 sender;
     does mcast)
   - `reader_bmm_tile_layout_in0_ring_all_gather.cpp` (in0 ring AG)
   - `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` (in1
     sender/writer)
   - `reader_bmm_tile_layout_in1_ring_all_gather.cpp` (in1 ring AG)
   - These dataflow kernels run in PARALLEL with the matmul compute
     kernel; any of them doing an unintended write to L1 0xa6700
     could stomp.
2. **A DMA / hardware path that writes to L1 0xa6700** as a side
   effect (e.g., a prefetcher reader_dram completion notification
   landing at a wrong address, an EDM fabric ack getting routed to
   L1 0xa6700, a stream-counter incorrectly mapped to L1 0xa6700,
   etc.).
3. **An L1 buffer reuse not visible via Python `.buffer_address()`
   or `get_buffers()`**:  some internal CB / sub-device scratch
   that shares the same L1 region as W2's mm_out_cb during the
   gathered path.

## Reproducer

```bash
# Phase 5 — FILL workaround (eliminates the L1 0xa6700 NONZERO race):
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U19_FILL_W2_ZERO=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u19_fillw2_probe.log \
python3 -u -m sglang.launch_server ... &
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"2+2=","sampling_params":{"max_new_tokens":5,"temperature":0.0}}'
grep -c "U17_PRE_RS in_addr=0xa6700.*NONZERO" /tmp/u19_fillw2_probe.log   # 0
grep -c "U17_PRE_RS in_addr=0xa6700.* zero\]" /tmp/u19_fillw2_probe.log   # 1920

# Phase 5 — FILL + zero weights output (collapses to U11):
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U19_FILL_W2_ZERO=1 \
python3 -u -m sglang.launch_server ... &
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Output ends in: "...extrextrextrextrextrextrextrextrextr"  (token 38649 loop)

# Phase 7 — Writeback + readback coherence check:
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U19_FORCE_PRODUCER_ZERO=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u19_fpz_rback.log \
python3 -u -m sglang.launch_server ... &
curl ...
grep -c "U19_FORCED_ZERO_RBACK.*probe_noc=0x1c20000a6700.*NONZERO" /tmp/u19_fpz_rback.log  # 0
grep -c "U19_FORCED_ZERO_RBACK.*probe_noc=0x1c20000a6700.* zero\]"  /tmp/u19_fpz_rback.log  # 1152

# Canonical (no prefetcher, no probes) re-verify:
SGLANG_TT_USE_PREFETCHER=0 \
python3 -u -m sglang.launch_server ... &
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Output: " What is 2+2? What is 2+2? What"  e2e_latency 9.06s
```

## U20 dispatch recommendation

The stomper is now in a small candidate set.  Most-tractable next attacks:

### U20 Phase A — probe all W2 dataflow kernels at entry

Add per-kernel-entry probes (like U19 WRITER_PROBE) to each W2
dataflow kernel (in0 sender, in0 receiver, in1 sender_writer, etc.)
dumping any L1 address that could be a write target.  Specifically,
check if any of them resolve a NoC address with L1 offset = 0xa6700.

### U20 Phase B — host-side L1 polling kernel between W2 and RS

Build a tiny custom kernel (single-tile, single-core, takes a NoC
target as RT arg) that polls L1 0xa6700 on receiver (2,7) every
N cycles, printing the value if it changes.  Dispatch this kernel
between W2 and RS in the Python pipeline.  The zero→NONZERO
transition timestamp pinpoints the exact stomp moment.

### U20 Phase C — try a DATA-PRESERVING workaround

If we can't identify the stomper quickly, ship a workaround for
real weights:
- Allocate a SECOND L1 buffer for w2_out (e.g., via
  `ttnn.to_memory_config` to a fresh L1 region).
- Use the SECOND buffer as the RS input.  If the stomp is tied
  specifically to L1 0xa6700, the second buffer would be safe.
- Risk: if the stomper follows the L1 allocator (allocator picks
  the slot it stomps), this doesn't help.  The address-move test
  (U18 recommendation) covers this.

### U20 Phase D — examine prefetcher reader_dram

The prefetcher reader_dram fetches weights from DRAM and writes
to the prefetcher's L1 region.  U18 verified writer_l1 mcast
doesn't target 0xa6700, but the READER (separate kernel) might
have a different write pattern.  Add a probe to reader_dram.

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal Python +
      C++ files, all env-gated)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Container is NOT bind-mounted — used `rebuild_tt_metal_kernels.sh`
- [x] All new probes/workarounds env-gated (`SGLANG_TT_U19_*`)
- [x] Canonical re-verified: coherent output, 9.06s e2e_latency

## Shipping verdict (unchanged from U18)

Canonical Qwen3-8B (no prefetcher) remains the production
shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≈ 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**
  Trace-replay race persists; U19 confirms the bug mechanism but
  the available workaround (`ttnn.fill(w2_out, 0)`) destroys
  real-weight output.
- **DO NOT ship any `SGLANG_TT_U19_*` probe / workaround.**
  Diagnostic only.

## Commits this session

- (tt-metal-sglang) **`fa35e4144a3`** —
  `prefetcher: U19 — confirm L1 0xa6700 stomp is the bug via ttnn.fill(w2_out,0) workaround (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`fa35e4144a3`** (U19 commit; env-gated;
  default-off; canonical bytewise-equal).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/*
  fork only per project policy).
- sglang HEAD: pending this doc commit.
- Container `/tt-metal/`: synced (reader + writer + program factory +
  mux + mlp.py with U19 probes/workarounds applied; rebuilt
  _ttnn.so / _ttnncpp.so).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at
  9.06s e2e_latency (bug absent; coherent output; no regression).
