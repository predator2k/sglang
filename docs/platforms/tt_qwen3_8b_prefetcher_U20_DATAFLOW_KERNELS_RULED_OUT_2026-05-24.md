# TT Qwen3-8B prefetcher — U20: W2 gathered matmul DATAFLOW kernels (in0_ring_ag + in1_ring_ag) RULED OUT as the L1 0xa6700 stomper — 2026-05-24

Status: **EVIDENCE_ADVANCE — Per-kernel entry+exit L1 0xa6700
probes wired into BOTH W2 dataflow kernels under
`SGLANG_TT_U20_DATAFLOW_PROBE=1`.  Run under full prefetcher +
zero weights + Qwen3-8B decode, producing 230,400 probe entries
(2048-budget × 4 probe-sites × ~28 cores).  Address probes show
the in0 ring-all-gather kernel's cross-core NoC writes target
`cb_in2_wr=0x1b200` + `shard_offset` (with `shard_size=0x2000`,
`ring_size=32` -> max addr `0x5b200`), and the in1 ring-all-gather
kernel reads weights from DRAM into local `cb_in1` at L1
addresses `0xac700`-`0x172100`.  NEITHER kernel ever writes to
`0xa6700`.  On the receiver core (2,7), only 3 zero->NONZERO L1
transitions per kernel out of 900 program runs (~0.3%) — and BOTH
kernels see IDENTICAL transition counts and per-core distributions,
which is the signature of an EXTERNAL concurrent writer that both
kernels observe at the same time, not the kernels themselves.
Combined with U18's compute-kernel results (PACK writes ZERO to
L1 0xa6700; U14 CONSUMER_PROBE shows L1=zero at kernel exit), all
THREE kernels of the W2 gathered matmul program (compute + in0_ag
+ in1_ag) are now ruled out as direct writers of NONZERO at L1
0xa6700 on receiver core (2,7).  The stomper must originate
OUTSIDE the W2 program: between W2 program exit and RS reader
entry (firmware dispatch machinery, prefetcher persistent kernel
sub-device crossing, EDM fabric kernel completion writes,
kernel-binary writeback), or from a concurrent persistent kernel
on a non-W2 sub-device that targets L1 0xa6700 on (2,7).**

Continuation of `tt_qwen3_8b_prefetcher_U19_FILL_W2_ZERO_WORKAROUND_CONFIRMS_STOMP_2026-05-24.md`.

tt-metal-sglang HEAD: **`a75bc37b75b`** (U20 commit; env-gated;
default-off; canonical bytewise-equal).  Branch `tenstorrent-p1`;
NOT pushed (predator2k/* fork only per project policy).
sglang HEAD: pending this doc commit.

## Phase 1 — W2 gathered program kernel enumeration

`process_gather_in0_program_and_create_override_variables` in
`matmul_multicore_reuse_mcast_1d_program_factory.cpp` (line 1967)
creates exactly THREE kernels per W2 dispatch:

| # | File | RISC | Role |
|---|---|---|---|
| 1 | `dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp` | NCRISC (RISCV_1) | in0 ring all-gather sender/receiver — local cb_in0 read + cross-core NoC write to next core's cb_in2 |
| 2 | `dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | BRISC (RISCV_0) | in1 DRAM read (weights) into local cb_in1 |
| 3 | `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | TRISC (3 cores) | UNPACK + MATH + PACK; PACK writes mm_out_cb to L1 0xa6700 |

All three kernels run in parallel on the SAME tile's RISCs.  U18
PACK_PROBE + U14 CONSUMER_PROBE have already verified kernel #3
exits with L1 0xa6700 = ZERO (6912/6912 entries).  U20 covers
kernels #1 and #2.

## Phase 2 — U20 dataflow entry/exit probes

### What was added (env-gated; default off)

| File | Change |
|---|---|
| `reader_bmm_tile_layout_in0_ring_all_gather.cpp` | `SGLANG_TT_U20_DATAFLOW_PROBE` block at kernel_main entry (after IDLE_CORE early-return) AND at kernel exit (after `noc.async_atomic_barrier()`); dumps L1 0xa6700 first 16 bytes.  Also dumps `cb_in0.get_read_ptr()`, `cb_in2.get_write_ptr()`, `shard_size_bytes`, `ring_size`, `next_x/next_y` in an ADDR probe. |
| `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Same pattern.  Entry probe after IDLE/HOP early-return.  Exit probe after `noc.async_write_barrier()`.  ADDR probe dumps `cb_in1.get_write_ptr()` and `in1_single_tile_size_bytes`. |
| `matmul_multicore_reuse_mcast_1d_program_factory.cpp` | `mm_in0_kernel_defines` map created; both `mm_in0_kernel_defines` and `mm_in1_kernel_defines` receive `SGLANG_TT_U20_DATAFLOW_PROBE=1` when env is set.  In0 kernel's CreateKernel now wires `.defines = mm_in0_kernel_defines`. |

Probes use `noc_async_read_barrier()` + `noc_async_write_barrier()`
to drain in-flight ops before reading local L1 0xa6700.  Per-RISC
static budget = 2048; per-RISC static counter `tot=`.

### Run

```bash
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U20_DATAFLOW_PROBE=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u20_dataflow_probe.log \
TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
python3 -u -m sglang.launch_server \
  --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
  --device tenstorrent --context-length 4096 \
  --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
  --skip-server-warmup --max-running-requests 1 --trust-remote-code \
  --attention-backend torch_native &
curl -s -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"2+2=","sampling_params":{"max_new_tokens":5,"temperature":0.0}}'
```

### Result — aggregate counts

| Probe | Total | NONZERO | zero |
|---|---:|---:|---:|
| U20_ENTRY in0_ring_ag | 57600 | 18386 | 39214 |
| U20_EXIT  in0_ring_ag | 57600 | 18288 | 39312 |
| U20_ENTRY in1_ring_ag | 57600 | 18386 | 39214 |
| U20_EXIT  in1_ring_ag | 57600 | 14328 | 43272 |
| U17_PRE_RS @ 0xa6700  |  1920 |   880 |  1040 |

### Per-core ENTRY -> EXIT transitions (top 30 by zero->NZ)

Pairing each (core, kernel)'s consecutive ENTRY,EXIT events:

```
kernel             core         zero->NZ  NZ->zero  zero->zero  NZ->NZ  total_pairs
------------------------------------------------------------------------------------
in1_ring_ag        0:8-6            33        94       547        226      900
in0_ring_ag        0:8-6            33         0       547        320      900
in1_ring_ag        0:1-7            27       147       602        124      900
in0_ring_ag        0:1-7            27        55       602        216      900
in1_ring_ag        1:8-6            18        94       547        241      900
in0_ring_ag        1:8-6            18         0       547        335      900
in1_ring_ag        0:0-6             3        63       605        229      900
in0_ring_ag        0:0-6             3         0       605        292      900
in1_ring_ag        0:2-5             3        63       605        229      900
in0_ring_ag        0:2-5             3         0       605        292      900
in1_ring_ag        0:2-7             3        63       605        229      900
in0_ring_ag        0:2-7             3         0       605        292      900
in1_ring_ag        0:3-7             3        63       605        229      900
in0_ring_ag        0:3-7             3         0       605        292      900
...

Totals per kernel:
in1_ring_ag: zero->NZ=120  NZ->zero=4178  zero->zero=39094  NZ->NZ=14208  total_pairs=57600
in0_ring_ag: zero->NZ=120  NZ->zero=218   zero->zero=39094  NZ->NZ=18168  total_pairs=57600
```

### Receiver core (2,7) specifically

| Probe | NONZERO | zero |
|---|---:|---:|
| U20_ENTRY in0_ring_ag | 292 | 608 |
| U20_EXIT  in0_ring_ag | 295 | 605 |
| U20_ENTRY in1_ring_ag | 292 | 608 |
| U20_EXIT  in1_ring_ag | 232 | 668 |

zero->NONZERO transitions at (2,7): **3 per kernel out of 900
program runs (~0.3%)**.  Compare to U17 PRE_RS rate of ~46%
NONZERO at the SAME L1 address on the SAME core.

## Phase 3 — U20 ADDR probe: where do in0/in1 actually write?

### in0 ring-all-gather NoC write addresses

`async_write` target = `cb_in2.get_write_ptr() + shard_size_bytes
* (shard_cnt - hop_core_offset)` on the NEXT core in the ring.

Observed (all cores, all matmul programs, first 256 dispatches):

```
cb_in2_wr=0x1b200 shard_size_bytes=0x2000 ring_size=32
cb_in2_wr=0x1b200 shard_size_bytes=0x1980 ring_size=32
```

Max write address = `0x1b200 + 31 * 0x2000 = 0x59200` (well below
`0xa6700`).  in0 sender NoC writes RULED OUT as the stomper at L1
0xa6700.

### in1 ring-all-gather local L1 write addresses

`cb_in1.get_write_ptr()` observed values (all cores, all matmul
programs):

```
0xac700  0xb2d00  0xb3300  0xb9300  0xb9f00  0xc0b00  0xc5f00
0xc7700  0xd1900  0xd2b00  0xd8500  0xdf100  0xdf700  0xec900
0xf2900  0xfa100  0xff500  0x105b00 0x107900 0x10af00 0x10c100
0x111b00 0x112700 0x118700 0x118d00 0x11f300 0x125f00 0x12bf00
0x12cb00 0x136d00 0x138b00 0x13d900 0x145700 0x14b100 0x158900
0x165500 0x166100 0x16bb00 0x170300 0x172100
```

NONE equal `0xa6700`.  Closest is `0xac700` (offset +0x6000).
in1 reader local DRAM-staging writes RULED OUT.

### Aside — local_in0 == 0xa6700 occurs on some programs

```
local_in0=0x9c700  cb_in2_wr=0x1b200 shard_size_bytes=0x2000
local_in0=0x9e700  cb_in2_wr=0x1b200 shard_size_bytes=0x2000
local_in0=0xa0700  cb_in2_wr=0x1b200 shard_size_bytes=0x2000
local_in0=0xa2700  cb_in2_wr=0x1b200 shard_size_bytes=0x2000
local_in0=0xa6700  cb_in2_wr=0x1b200 shard_size_bytes=0x2000   <-- shared L1 slot
local_in0=0xa7700  cb_in2_wr=0x1b200 shard_size_bytes=0x2000
local_in0=0xa9700  cb_in2_wr=0x1b200 shard_size_bytes=0x2000
local_in0=0xaa700  cb_in2_wr=0x1b200 shard_size_bytes=0x2000
... (and *_d80 variants for ring_size=32 shard_size=0x1980)
```

`local_in0=0xa6700` on some cores in some programs means a
downstream matmul's cb_in0 (input) is allocated at the SAME L1
slot as W2's mm_out_cb (output) on those cores.  This is normal
allocator slot reuse, but it's a red flag if a downstream matmul
fires its in0 reader on (2,7) while W2's RS hasn't yet consumed
0xa6700.  HOWEVER: the in0 reader of those programs READS from
local cb_in0 (it doesn't WRITE to it), so reading from
`local_in0=0xa6700` doesn't stomp it; only the prior PACK that
filled that cb_in0 could have written there.

## Phase 4 — Decision

The "120 zero->NZ across all cores" total is dominated by 3 cores
(`0:8-6`, `0:1-7`, `1:8-6` -> 78 transitions total).  Receiver
core (2,7) contributes only 3 transitions per kernel.

**Both kernels observe IDENTICAL zero->NZ counts (120 each) AND
identical per-core distributions.**  In0 and in1 ring all-gather
run on DIFFERENT RISCs of the same tile, doing DIFFERENT work
(in0 = cross-core NoC writes; in1 = DRAM reads).  If either kernel
were the writer of these zero->NZ transitions, the OTHER kernel
would not see them at the same rate.

The only explanation consistent with this data: an EXTERNAL
concurrent writer fires during the W2 program's execution and is
observed by BOTH dataflow kernels via local L1 reads at their
EXIT.

The "120 / 57600 = 0.2%" rate is also far below the U17 PRE_RS
NONZERO rate of "880 / 1920 = 46%" — meaning the bulk of the L1
0xa6700 stomp does NOT occur during the W2 program run.  It
occurs AFTER the W2 program exits but BEFORE the RS reader runs.

## Hypothesis ledger (post-U20)

| ID | Suspect | Pre-U20 | Post-U20 |
|---|---|---|---|
| S1-S10, Lead 2-3, U3, U4-A/B, U1, U5/U16-2/5a/5b/5c, U17 P1/P2 | various | RULED OUT | unchanged |
| **U17 P3 — intervening kernel stomps L1 0xa6700 between W2 exit and RS reader** | CONFIRMED | unchanged | confirmed |
| **U18 — prefetcher writer_l1 mcasts to 0xa6700** | RULED OUT | unchanged | — |
| **U18 — KERNEL_CONFIG region writes hit 0xa6700** | RULED OUT | unchanged | — |
| **U18 — next-layer W1/W3 pipelined dispatch stomp** | RULED OUT | unchanged | — |
| **U19 — RS writer kernel writes to 0xa6700** | RULED OUT | unchanged | — |
| **U19 — RS mux kernel zeros L1 0xa6700** | RULED OUT | unchanged | — |
| **U19 — L1 data-cache stale-read race fix via invalidate_l1_cache()** | RULED OUT | unchanged | — |
| **U19 — bug is L1 0xa6700 byte stomp** | CONFIRMED | unchanged | — |
| **U19 — fix is "any dispatch barrier between W2 and RS"** | RULED OUT | unchanged | — |
| **U19 — stomper is a continuous overwriter** | RULED OUT | unchanged | — |
| **U20 — W2 in0 ring all-gather NoC writes target 0xa6700** | (new) | **RULED OUT** | cb_in2_wr=0x1b200; max write 0x59200; never 0xa6700 |
| **U20 — W2 in1 ring all-gather local writes target 0xa6700** | (new) | **RULED OUT** | cb_in1_wr in [0xac700, 0x172100]; never 0xa6700 |
| **U20 — W2 dataflow kernel(s) themselves stomp L1 0xa6700 during run** | (new) | **RULED OUT** | only 0.2% zero->NZ; identical between in0/in1 -> external concurrent writer; on receiver (2,7) only 0.3% during dataflow run vs 46% at RS reader entry |
| **U20 STANDING — stomper fires AFTER W2 program (all 3 kernels) exits and BEFORE RS reader runs** | (new) | **STANDING — TOP PRIORITY** | — |

## Remaining stomper candidate set (very narrow now)

Across U17 + U18 + U19 + U20, all kernels INSIDE the W2 program
are ruled out as writers of NONZERO to L1 0xa6700 on receiver
core (2,7).  The remaining suspects are STRICTLY external to the
W2 program:

1. **Firmware dispatch machinery between programs** — when the
   dispatcher finishes program N (W2) and starts program N+1
   (RS), it writes program N+1's kernel binaries to L1
   KERNEL_CONFIG.  U18 ruled this out for the KERNEL_CONFIG
   region itself (<0x15000), but the dispatcher may also write
   semaphore reset values, runtime-arg blocks, sub-device launch
   message regions, or trace-replay buffer descriptors that could
   land at 0xa6700 if mis-sized.

2. **EDM (Ethernet Data Movement) fabric kernels** completing
   in-flight cross-device transfers.  In 2-device TP=1 mesh, EDM
   may forward fabric ack messages to tensix L1 for completion
   semaphores.  If an ack lands at 0xa6700 on (2,7), that's the
   stomp.

3. **Prefetcher persistent kernel sub-device** — the prefetcher
   runs continuously on a separate sub-device, reading weights
   from DRAM and writing to GlobalCB.  U18 ruled out
   `writer_l1.cpp` (writes target GlobalCB at 0xac700+), but the
   PREFETCHER may also do bookkeeping writes via
   `update_remote_cb_config_in_l1` (writes
   `remote_cb_interface.config_ptr + offsetof(fifo_rd_ptr)`).
   If that config_ptr resolves to receiver (2,7) L1 0xa6700, this
   would be the stomp.

4. **Sub-device launch message region** — tt-metal allocates a
   launch-message ring buffer per sub-device.  Cross-sub-device
   dispatch (worker_sub_device for matmul -> receiver_sub_device
   for the prefetcher consumer, then back to worker for RS) may
   write launch messages to receiver L1 at addresses derived from
   sub-device base offsets.  Worth a probe.

5. **Trace replay buffer descriptors** — under trace replay, the
   trace replay engine writes per-program descriptors to L1 on
   the dispatcher cores AND on worker cores.  If a worker-core
   descriptor lands at 0xa6700, this is the stomp.

## What U20 lands (all env-gated; canonical bytewise-equal)

| File | Change |
|---|---|
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp` | `SGLANG_TT_U20_DATAFLOW_PROBE` entry+exit L1 0xa6700 probe, ADDR probe (cb_in0/cb_in2/shard_size/ring_size/next_x/y).  2048-budget; per-RISC static counter. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Same probes (entry/exit/ADDR).  Includes `cb_in1.get_write_ptr()` in ADDR probe. |
| `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | New `mm_in0_kernel_defines` map; env-gated `SGLANG_TT_U20_DATAFLOW_PROBE` propagation to both `mm_in0_kernel_defines` and `mm_in1_kernel_defines`; in0 kernel's `CreateKernel` now passes `.defines = mm_in0_kernel_defines`. |

## Reproducer

```bash
# U20 ENTRY/EXIT/ADDR probes:
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U20_DATAFLOW_PROBE=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u20_dataflow_probe.log \
TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
python3 -u -m sglang.launch_server ... &
curl -s -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"2+2=","sampling_params":{"max_new_tokens":5,"temperature":0.0}}'

# Aggregate counts:
grep -c "U20_ENTRY in0_ring_ag" /tmp/u20_dataflow_probe.log   # 57600
grep -c "U20_EXIT  in0_ring_ag" /tmp/u20_dataflow_probe.log   # 57600
grep "U20_ADDR in0_ring_ag" /tmp/u20_dataflow_probe.log | \
  grep -oP "cb_in2_wr=0x[a-f0-9]+" | sort -u   # cb_in2_wr=0x1b200

# Canonical (no prefetcher, no probes) re-verify:
SGLANG_TT_USE_PREFETCHER=0 \
python3 -u -m sglang.launch_server ... &
curl ...   # " What is 2+2? What is 2+2? What" at 9.08s e2e_latency
```

## U21 dispatch recommendation

The stomp is now provably external to the W2 program.  Most
tractable next attacks:

### U21 Phase A — sub-device launch-message region probe

tt-metal maintains a launch-message ring buffer per sub-device,
stored at a fixed L1 offset.  When the dispatcher dispatches a
new program to a sub-device, it writes the launch message to this
buffer on every core in the sub-device's grid.  If the buffer's
L1 base address overlaps 0xa6700 on (2,7), this is the stomp.

Probe: dump `SUB_DEVICE_LAUNCH_MSG_BUF` L1 address per sub-device
at device init; if any equals 0xa6700, this is the stomper.

### U21 Phase B — prefetcher's `update_remote_cb_config_in_l1` target

`update_remote_cb_config_in_l1` writes to
`remote_cb_interface.config_ptr + offsetof(fifo_rd_ptr)`.  Add a
probe in `writer_l1.cpp` AT EXIT dumping this address.  If any
config_ptr lands at receiver (2,7) L1 0xa6700, this is the
stomper.

### U21 Phase C — EDM fabric kernel L1 ack region

EDM kernels run on ethernet cores but write completion semaphores
to tensix L1.  Find the EDM kernel + probe its L1 write addresses
for ack messages.  Worth ~1 session.

### U21 Phase D — host-side L1 poller kernel between W2 and RS

Per U19 Phase B recommendation: write a tiny custom kernel that
takes a NoC target + L1 offset as RT arg and polls L1 every N
cycles, printing on change.  Dispatch BETWEEN W2 and RS in the
Python pipeline (capture into trace).  The zero->NONZERO
transition timestamp pinpoints the exact stomp event.  Worth 1-2
sessions.

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal C++ files,
      all env-gated; mlp.py untouched)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Container is NOT bind-mounted — used `rebuild_tt_metal_kernels.sh`
      AND `podman cp` for host->container source sync
- [x] All new probes env-gated (`SGLANG_TT_U20_DATAFLOW_PROBE`)
- [x] `noc_async_read_barrier()` + `noc_async_write_barrier()`
      before all L1 reads in probes
- [x] Canonical re-verified post-rebuild: coherent output,
      9.08s e2e_latency

## Shipping verdict (unchanged from U19)

Canonical Qwen3-8B (no prefetcher) remains the production
shipping config at TPOT ~= 27 ms / 1024-1024 and GSM8K(10) chat
~= 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**
  Trace-replay race persists; U20 confirms the W2 program itself
  doesn't stomp, but the external stomp is unfixed.
- **DO NOT ship any `SGLANG_TT_U20_*` probe.**  Diagnostic only.

## Commits this session

- (tt-metal-sglang) **`a75bc37b75b`** —
  `prefetcher: U20 — dataflow kernel entry+exit L1 0xa6700 probes; W2 dataflow kernels RULED OUT as stomper (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`a75bc37b75b`** (U20 commit; env-gated;
  default-off; canonical bytewise-equal).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/*
  fork only per project policy).
- sglang HEAD: pending this doc commit.
- Container `/tt-metal/`: synced (in0/in1 ring AG kernels +
  program factory with U20 probes applied; rebuilt
  `_ttnn.so` / `_ttnncpp.so` and verified `SGLANG_TT_U20_DATAFLOW_PROBE`
  string present).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at
  9.08s e2e_latency (bug absent; coherent output; no regression).
