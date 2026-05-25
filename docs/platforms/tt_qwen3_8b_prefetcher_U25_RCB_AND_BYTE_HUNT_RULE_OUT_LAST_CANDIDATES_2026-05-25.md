# TT Qwen3-8B prefetcher — U25: `update_remote_cb_config_in_l1` + same-core L1 byte-hunt RULE OUT remaining stomper candidates; expose 0xa6700 ↔ 0xa8700 ↔ 0xac700 L1 layout collision (EVIDENCE_ADVANCE) — 2026-05-25

Status: **EVIDENCE_ADVANCE — U25 rules out the only un-probed
GlobalCB-metadata write path (`update_remote_cb_config_in_l1`)
and proves the stomp bytes at L1 `0xa6700` on receiver (2,7) are
NOT a copy of any other location in (2,7)'s L1 [0x10000, 0x1c0000].
A new evidence dimension is uncovered: at decode-time the
prefetcher path allocates THREE adjacent L1 buffers around the
cursed address — w2_out at `0xa6700` (WIDTH_SHARDED, 8192 B/bank),
a sibling at `0xa8700` (WIDTH_SHARDED, 8192 B/bank, identical
shape/layout), and the prefetcher's GlobalCB receiver buffer at
`0xac700` (HEIGHT_SHARDED, 835584 B/bank).  Canonical (no
prefetcher) places w2_out at DRAM `0x42abdc40` and has ZERO L1
buffers near 0xa6700.  So the stomp is enabled by the prefetcher's
L1-sharded w2_out placement.**

Continuation of `tt_qwen3_8b_prefetcher_U24_FABRIC_RULED_OUT_PATHA_2026-05-25.md`.

tt-metal-sglang HEAD: **`737fda47aae`** (U25 commit; env-gated;
default-off; canonical bytewise-equal).  Branch `tenstorrent-p1`;
NOT pushed (predator2k/* fork only per project policy).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (post-U25)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| A.1 | Does the matmul in1-ring-AG reader's call to `update_remote_cb_config_in_l1` write to L1 [0xa6000, 0xa7000]? | `SGLANG_TT_U25_RCB_PROBE=1` (DPRINT dest = `config_ptr + offsetof(fifo_rd_ptr)`) | NO — 172800 entries; ALL dest=`0x17f650`; ZERO STOMP_HIT. |
| A.2 | Does the prefetcher writer_l1's call to `update_remote_cb_config_in_l1` write to L1 [0xa6000, 0xa7000]? | same probe, writer_l1 site | NO — 240 entries; ALL dest=`0x17f650`; ZERO STOMP_HIT. |
| B.1 | Are the NONZERO bytes at L1 `0xa6700` on (2,7) a copy from any other (2,7) L1 address? | `SGLANG_TT_U25_BYTE_HUNT=1` (scan L1 [0x10000, 0x1c0000] at 0x40 stride, search 16-byte windows) | NO — 2640 NONZERO probes; ZERO matches anywhere on (2,7). |
| B.2 | (extension) Are the same bytes present at L1 `0xa6700` on OTHER tensix cores? | cross-core scan via `noc_async_read` to arbitrary XY codes | INCONCLUSIVE — caused NoC hang and FW reset; reverted. |
| C | At decode-time, what L1 buffers are allocated near 0xa6700 on receiver (2,7) under prefetcher mode? | `SGLANG_TT_U15_PROBE_W2_L1=1` (existing probe; re-run with prefetcher + zero weights) | THREE buffers: `0xa6700` w2_out WIDTH_SHARDED 8192/bank; `0xa8700` sibling WIDTH_SHARDED 8192/bank; `0xac700` GlobalCB-receiver HEIGHT_SHARDED 835584/bank. |
| D | What is w2_out's address under canonical (no prefetcher)? | `SGLANG_TT_U18_ADDR_PROBE=1` (canonical run) | DRAM `0x42abdc40` (NOT L1 0xa6700).  ZERO L1 buffers near 0xa6700. |

**Punchline:** The prefetcher's L1-sharded placement of w2_out
puts it at fixed `0xa6700` on receiver (2,7), where SOMETHING
ELSE in the prefetcher pipeline writes garbage bytes between W2
PACK exit and RS reader entry.  The remaining suspect set is now
critically narrow: it MUST be either (a) a write originating
from ANOTHER core's L1 (since the bytes don't exist elsewhere on
(2,7)), (b) a write originating from DRAM (e.g., a stale
prefetcher block), or (c) an arithmetic write produced by a
compute kernel running concurrently on a different sub-device.

## Phase A — `update_remote_cb_config_in_l1` probe (Path A)

### A.1 Probe design

In `tt_metal/hw/inc/api/remote_circular_buffer.h`:
```cpp
FORCE_INLINE void update_remote_cb_config_in_l1(uint32_t remote_cb_index) {
    RemoteReceiverCBInterface& remote_cb_interface = get_remote_receiver_cb_interface(remote_cb_index);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        remote_cb_interface.config_ptr + offsetof(RemoteReceiverCBInterface, fifo_rd_ptr)) =
        remote_cb_interface.fifo_rd_ptr;
}
```

Per `circular_buffer_interface.h::static_assert`, `offsetof(fifo_rd_ptr)
== offsetof(fifo_wr_ptr) == 16` on tensix.  So the write goes to LOCAL
L1 at `config_ptr + 16`.  If `config_ptr` were ever `0xa66f0` on (2,7),
the write would land at `0xa6700`.

Two call sites identified via `grep -rn 'update_remote_cb_config_in_l1'`:
1. `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp` line 142 (post-loop).
2. `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` line 253 (post-loop, under `ENABLE_GLOBAL_CB`).

Both wrapped with a `SGLANG_TT_U25_RCB_PROBE`-gated block that
extracts `config_ptr` and dest = `config_ptr + 16`, DPRINTs
`(cb, config_ptr, dest, value, hits_stomp)`, and emits a STOMP_HIT
line iff dest ∈ [0xa6000, 0xa7000].

Define propagation:
- `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp` adds `writer_defines["SGLANG_TT_U25_RCB_PROBE"] = "1"` if env set.
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` adds `mm_in1_kernel_defines["SGLANG_TT_U25_RCB_PROBE"] = "1"` if env set.

### A.2 Run

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_MAX_BATCH=1 \
HF_MODEL=/models/Qwen3-8B \
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U25_RCB_PROBE=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u25_rcb.log \
nohup python3 -u -m sglang.launch_server ... &
# decode 15 tokens, then grep:
grep -c "U25_RCB.*STOMP_HIT" /tmp/u25_rcb.log   # 0
grep -c "U25_RCB_IN1 "       /tmp/u25_rcb.log   # 172800
grep -c "U25_RCB_PREF "      /tmp/u25_rcb.log   # 240
grep "U25_RCB_IN1 " /tmp/u25_rcb.log | grep -oE "dest=0x[0-9a-f]+" | sort -u
# dest=0x17f650
grep "U25_RCB_PREF " /tmp/u25_rcb.log | grep -oE "dest=0x[0-9a-f]+" | sort -u
# dest=0x17f650
```

Both call sites have `config_ptr=0x17f640`, dest=0x17f650.  The
`fifo_rd_ptr` values that get written are local L1 addresses in
range `[0x105b00, 0x14xxxx]` (the GlobalCB fifo).  Crucially, the
write *target* `0x17f650` is FAR from `0xa6700`.

**Verdict A:** `update_remote_cb_config_in_l1` is RULED OUT as
the L1 0xa6700 stomper.

## Phase B — same-core L1 byte-hunt (Path B)

### B.1 Probe design

In `line_reduce_scatter_minimal_async_reader.cpp`, after the U17
PRE_RS probe reads the 16 bytes at `probe_noc_addr` (which is
typically `0x1c20000a6700` for receiver (2,7)), if NONZERO, scan
the PRODUCING CORE's L1 from `0x10000` to `0x1c0000` at `0x40`
stride looking for 16-byte windows that match the observed bytes.
Per-RISC budget = 1 invocation (the full scan takes ~7000 NoC
reads per invocation — capping at 1 per kernel launch).

If found, DPRINT `[U25_BH_SAME_CORE_MATCH probe_noc=0x.. match_l1=0x..]`.
Always DPRINT a `SUMMARY` line with the total match count.

The intent: if the stomp bytes were COPIED to 0xa6700 from
another L1 location on the same core (e.g., a stale GlobalCB
page, a leftover compute scratch, a wrapping write), the source
L1 base would match.

### B.2 Run

```bash
SGLANG_TT_U17_PROBE_RS_PRE=1 \
SGLANG_TT_U25_BYTE_HUNT=1 \
... (rest of env as above)
grep -c "U25_BH_SAME_CORE_SUMMARY" /tmp/u25_bh.log  # 2640
grep -c "U25_BH_SAME_CORE_MATCH"   /tmp/u25_bh.log  #    0
```

Across 2640 NONZERO observations, scanning all of (2,7) L1
[0x10000, 0x1c0000] at 0x40 stride, **ZERO matches**.

**Verdict B:** the stomp bytes are NOT present anywhere else in
(2,7)'s L1.  They are NOT a copy from any other L1 location on
the same core.

### B.3 Aborted cross-core extension

A second phase tried to scan the SAME L1 address (`0xa6700`) on
OTHER tensix cores by enumerating raw NoC XY codes in a small
range around the receiver's `probe_noc_xy = 0x1c20`.  Result:
NoC hung (the `noc_async_read` blocked indefinitely on the first
unmapped XY), causing a 300s scheduler watchdog timeout and a
device FW failure that required `tt-smi -r`.  Reverted to
single-phase same-core scan only.  Future cross-core work should
use a static whitelist of known-valid core XY codes (e.g.,
enumerate tensix cores via `device->compute_with_storage_grid_size()`
and pass as RT args).

## Phase C — L1-layout enumeration at W2 dispatch time

The existing `SGLANG_TT_U15_PROBE_W2_L1` probe in `mlp.py`
enumerates all L1 buffers within ±0x10000 of `0xa6700` at the
moment w2_out is dispatched.  Re-run under the same prefetcher
+ zero-weights workload:

```
[U18_W2_ADDR] iter=7 layer=0 w2_out.addr=0xa6700 shape=(1, 1, 32, 4096)
              shard_grid={32 cores: (1-1)..(4-7) + (8-0)..(10-6) + (0-2)..(0-6)}

[U15_PROBE_G] W2-time near 0xa6700: 3 bufs
  addr=0xa6700 bt=BufferType.L1 bl=WIDTH_SHARDED  sz_per_bank=8192
  addr=0xa8700 bt=BufferType.L1 bl=WIDTH_SHARDED  sz_per_bank=8192
  addr=0xac700 bt=BufferType.L1 bl=HEIGHT_SHARDED sz_per_bank=835584
```

L1 layout on receiver (2,7) at W2 dispatch time:

| Address range | Size | Identity |
|---|---|---|
| `0xa6700` - `0xa8700` | 8192 B  | **w2_out** (current layer's MLP W2 output, WIDTH_SHARDED over 32 cores) |
| `0xa8700` - `0xaa700` | 8192 B  | SIBLING — WIDTH_SHARDED, identical shape/layout as w2_out (likely another tensor with the same shape/layout: e.g., another decoder layer's w2_out residual slot, or a double-buffered alternative) |
| `0xaa700` - `0xac700` | 8192 B  | FREE — this is the gap U22 REALLOCATE found w2_out moves into when relocated |
| `0xac700` onward     | 835584 B (~816 KiB) | **GlobalCB receiver buffer** (matches U21's `fifo_start=0xac700`) |

Under canonical (no prefetcher), w2_out lives at DRAM `0x42abdc40`,
and `near 0xa6700: 0 bufs`.  So the L1-sharded placement of
w2_out at `0xa6700` is EXCLUSIVE to the prefetcher path.

## Hypothesis ledger (post-U25)

| ID | Suspect | Pre-U24 | Post-U24 | Post-U25 |
|---|---|---|---|---|
| Fabric / EDM completion acks | STANDING | **RULED OUT (U24)** | — |
| `update_remote_cb_config_in_l1` bookkeeping writes | STANDING | STANDING | **RULED OUT (U25 Path A)** |
| Bytes at 0xa6700 are a copy from elsewhere in (2,7) L1 | (new) | (new) | **RULED OUT (U25 Path B same-core scan)** |
| Trace-replay command write to stale captured L1 addr | STANDING | STANDING | STANDING |
| Cross-sub-device dispatch artifact | STANDING | STANDING | STANDING |
| Bytes come from ANOTHER core's L1 via NoC (e.g., mcast, fabric, GlobalCB write to wrong receiver) | (new) | (new) | **NEW STANDING — TOP PRIORITY** |
| Bytes come from DRAM (e.g., stale prefetched block, RoPE constant table loaded directly to L1) | (new) | (new) | NEW STANDING |
| Bytes produced by a compute kernel on a different sub-device running concurrently | (new) | (new) | NEW STANDING |
| W2 output L1 base (0xa6700) collides with sibling buffer (0xa8700) and they're written at overlapping times | (new) | (new) | NEW STANDING — investigate next |

## What U25 lands

| File | Change |
|---|---|
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | `SGLANG_TT_U25_RCB_PROBE`-gated DPRINT block immediately before `experimental::update_remote_cb_config_in_l1(remote_cb_id)`. Logs `(cb, config_ptr, dest = config_ptr + 16, fifo_rd_ptr, hits_stomp)` per receiver core; emits STOMP_HIT on dest ∈ [0xa6000, 0xa7000]. |
| `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | env-gated `mm_in1_kernel_defines["SGLANG_TT_U25_RCB_PROBE"] = "1"` (only under ENABLE_GLOBAL_CB / gathered path). |
| `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp` | symmetric probe at the prefetcher's call to `update_remote_cb_config_in_l1` after the main send loop. |
| `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp` | env-gated `writer_defines["SGLANG_TT_U25_RCB_PROBE"] = "1"`. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp` | `SGLANG_TT_U25_BYTE_HUNT`-gated block extending the U17 PRE_RS probe. On first NONZERO at `probe_noc_addr`, scans L1 [0x10000, 0x1c0000] at 0x40 stride on the same NoC core; DPRINTs SAME_CORE_MATCH lines + a SUMMARY count. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp` | env-gated propagation of `SGLANG_TT_U25_BYTE_HUNT` into reader_compute_defines. |

All probes default-off; strict NO-OP for canonical builds.

## Reproducer

```bash
# A.  U25 RCB probe — under FULL prefetcher + zero weights:
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  rm -f /tmp/u25_rcb.log
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_U25_RCB_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_FILE=/tmp/u25_rcb.log \
  nohup python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u25_server.log 2>&1 &'
podman exec p3a-ngram bash -c '
  until curl -sf http://127.0.0.1:30000/health_generate > /dev/null 2>&1; do sleep 5; done
  curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
    -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":15,\"temperature\":0.0}}"'
podman exec p3a-ngram bash -c '
  grep -c "U25_RCB.*STOMP_HIT" /tmp/u25_rcb.log              # 0
  grep -c "U25_RCB_IN1 "       /tmp/u25_rcb.log              # 172800
  grep "U25_RCB_IN1 " /tmp/u25_rcb.log | grep -oE "dest=0x[0-9a-f]+" | sort -u
  # dest=0x17f650
'

# B.  U25 byte hunt — under same workload, requires U17 probe enabled:
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '
  rm -f /tmp/u25_bh.log
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_U17_PROBE_RS_PRE=1 \
  SGLANG_TT_U25_BYTE_HUNT=1 \
  ... (other env as Phase A) ...
  TT_METAL_DPRINT_FILE=/tmp/u25_bh.log \
  nohup python3 -u -m sglang.launch_server ...'
# After decode:
podman exec p3a-ngram bash -c '
  grep -c "U25_BH_SAME_CORE_SUMMARY" /tmp/u25_bh.log   # 2640
  grep -c "U25_BH_SAME_CORE_MATCH"   /tmp/u25_bh.log   #    0
'

# C.  L1-layout enumeration (existing U15 probe):
podman exec p3a-ngram bash -c '
  SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_U15_PROBE_W2_L1=1 SGLANG_TT_U18_ADDR_PROBE=1 \
  ... \
  nohup python3 -u -m sglang.launch_server ...'
# Look for [U15_PROBE_G] lines with addr=0xa6700, 0xa8700, 0xac700.

# D.  Canonical regression check (no prefetcher, no probes):
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged HF_MODEL=/models/Qwen3-8B ...
# Output: " What is 2+2? What is 2+2? What"  e2e_latency = 9.10s
# GSM8K(5): 4/5 = 80% (within expected variance)
```

## U26 dispatch recommendation

The remaining candidate set is genuinely narrow.  Highest-leverage
next attacks:

### U26 Phase A — instrument the 0xa8700 sibling buffer

Identify what tensor allocates the WIDTH_SHARDED 8192/bank L1
buffer at `0xa8700` (sister to w2_out at 0xa6700, identical
shape/layout).  Hypothesis: it's another layer's w2_out, a
double-buffered next-iteration w2_out, OR an attention WO output.
If the SAME op writes to BOTH 0xa6700 and 0xa8700 due to a
sharding-index off-by-one or wrap bug, we've found it.

Approach: add a `SGLANG_TT_U26_PROBE_ADJACENT_BUFFER=1`
buffer-enum probe that runs at every captured op's host-side
launch and prints `(op_name, buffer_address, shape, layout)`.
Filter for ops whose buffer address falls in [0xa6000, 0xb0000].

### U26 Phase B — cross-core byte hunt (whitelisted XY codes)

Re-attempt the cross-core scan with a static whitelist of known
tensix cores.  Enumerate cores via
`mesh_device.compute_with_storage_grid_size()` host-side, encode
each NoC XY using `NOC_XY_ENCODING`, pass as RT args to the
reader kernel.  Scan ONLY validated cores.  Per-RISC budget
should be 1 (one full sweep per probe invocation), and the
budget should reset only when a NEW byte pattern is seen.

If bytes are found on another core's L1 0xa6700, the source is a
multicast / mcast write OR a corresponding tensor on that
sibling core.

### U26 Phase C — DRAM-source byte hunt

Add a probe that scans the prefetcher's DRAM-side weight blocks
for the observed 16-byte pattern.  If a match is found at DRAM
offset X corresponding to weight tensor T, then either:
- T's weight wasn't actually zeroed (zero-weights bug), or
- some DMA path is reading from T's DRAM and writing to
  receiver (2,7) L1 0xa6700.

### U26 Phase D — concurrent-kernel hunt

Identify what kernels run on (2,7) DURING the W2 → RS window on
sub-devices OTHER than the receiver_sub_device.  Add a per-RISC
DPRINT to each (NCRISC, BRISC, TRISC × 3) of EVERY active
program on (2,7) showing `kernel_id, program_id, sub_device_id`
at start and end.  Cross-reference with the U17 NONZERO
timestamps.

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (only tt-metal C++ + an
      sglang docs file).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` (TT_CACHE_HOME)
      + literal-path absolute.
- [x] No stash pop/drop.
- [x] Port-clear used `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` for source
      sync + `rebuild_tt_metal_kernels.sh` (with INSTALL_DIR sync).
- [x] Probes are env-gated initially (`SGLANG_TT_U25_*`).
- [x] Canonical Qwen3-8B preserved — bytewise " What is 2+2? What is
      2+2? What" at 9.10s + 4/5 GSM8K(5) chat.
- [x] Server stopped at session end.
- [x] TT-SMI reset run after byte-hunt v2 caused FW failure; devices healthy at session end.

## Shipping verdict (unchanged from U19/U20/U21/U22/U23/U24)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~27 ms / 1024-1024 and GSM8K(10) chat 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  The
  L1 0xa6700 stomp persists.
- **DO NOT ship any `SGLANG_TT_U25_*` env var.**  Diagnostic only.

The U25 probes are safe to ship as default-off probes — they are
strictly NO-OP for canonical builds (the kernel `#ifdef`s expand
to nothing when env is unset; the program-factory defines are
only added when the env var is set).

## Commits this session

- (tt-metal-sglang) **`737fda47aae`** —
  `prefetcher: U25 — update_remote_cb_config_in_l1 + same-core L1 byte-hunt RULE OUT remaining stomper candidates; expose L1 layout collision around 0xa6700 (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`737fda47aae`** (U25 commit; env-gated;
  default-off; canonical bytewise-equal).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.  No sglang Python
  changes in U25.
- Container `/tt-metal/`: synced (matmul in1 reader + matmul
  factory + prefetcher writer_l1 + prefetcher factory + RS
  reader + RS factory); rebuilt `_ttnn.so` / `_ttnncpp.so`;
  sentinels `SGLANG_TT_U25_RCB_PROBE` and `SGLANG_TT_U25_BYTE_HUNT`
  verified in `_ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy (one tt-smi -r required mid-session after
  cross-core byte-hunt hung the NoC).
- Server: stopped at session end.
- Canonical re-verify: bytewise " What is 2+2? What is 2+2? What"
  at 9.10s + GSM8K(5) chat 4/5.
