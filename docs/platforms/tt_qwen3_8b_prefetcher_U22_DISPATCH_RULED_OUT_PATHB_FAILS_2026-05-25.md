# TT Qwen3-8B prefetcher — U22: dispatcher process_write_* RULES OUT 0xa6700 as target; all 5 Path-B sharding-preserving relocation primitives BREAK under real weights — 2026-05-25

Status: **EVIDENCE_ADVANCE — Two decisive rule-outs this
session.  Path A (dispatcher write-target trace under
`SGLANG_TT_U22_DISP_TRACE=1`) instrumented
`process_write_linear`, `process_write_packed`, and
`process_write_packed_large` in `cq_dispatch.cpp`.  Sniff
output shows dispatcher writes target ONLY KERNEL_CONFIG
(0x9e30..0xdcb0 via write_packed_large) and GO-msg slot
0x17ffc0 (via write_linear); NONE target L1 0xa6700.
Dispatcher direct writes RULED OUT.  Path B (5 sharding-
preserving relocation primitives) all fail under real
weights: `to_memory_config`(same cfg) errors with
is_allocated; `ttnn.reallocate` succeeds at moving w2_out
to 0xaa700 (PRE_RS at 0xaa700 = 0/8640 NONZERO confirms)
but produces racy/garbage output due to a discovered
per-core chunk-size bug in `move_sharded_program_factory`;
`ttnn.assign(input, memory_config=...)`, `ttnn.copy(snap,
w2_out)`, and `BARRIER_CLONE` all fail or produce
real-weight garbage.**

Continuation of `tt_qwen3_8b_prefetcher_U21_STOMP_AT_FIXED_ADDR_2026-05-24.md`.

tt-metal-sglang HEAD: **`d1448de43b9`** (U22 commit; env-gated;
default-off; canonical bytewise-equal at 9.05s e2e_latency).
Branch `tenstorrent-p1`; NOT pushed (predator2k/* fork only).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (post-U22)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| A | Does any `process_write_*` in `cq_dispatch.cpp` target L1 0xa6700? | `SGLANG_TT_U22_DISP_TRACE=1` env-gated sniff probes in `process_write_linear` / `process_write_packed` / `process_write_packed_large` | NO — wl: 0x17ffc0 only; wpl: 0x9e30..0xdcb0; wp: never fires.  Dispatcher RULED OUT. |
| B.RESHARD | Does `ttnn.to_memory_config(w2_out, w2_out.memory_config())` preserve sharding AND fix the bug? | `SGLANG_TT_U22_RESHARD_W2=1` | NO — TT_FATAL `self.is_allocated()` (downstream tensor invalidated by returned same-cfg no-op + our dealloc) |
| B.REALLOCATE | Does `ttnn.reallocate` move w2_out off 0xa6700 AND fix the bug? | `SGLANG_TT_U22_REALLOCATE_W2=1` | PARTIAL — moves to 0xaa700 (PRE_RS at 0xaa700 = 0/8640 NONZERO; RS reader correctly consumes from 0xaa700) but real-weight output is garbage; root cause: `move_sharded` per-core chunk-size bug |
| B.ASSIGN | Does `ttnn.assign(w2_out, memory_config=...)` fix the bug? | `SGLANG_TT_U22_ASSIGN_W2=1` | NO — TT_FATAL `num_intersections == num_cores` (sub-device mismatch) |
| B.SNAPSHOT_RESTORE | Does `clone(w2_out) → copy(snap, w2_out)` (save-and-restore) fix the bug? | `SGLANG_TT_U22_SNAPSHOT_RESTORE_W2=1` | NO — `ttnn.copy` step hits same sub-device error |
| B.BARRIER_CLONE | Does `clone(w2_out)` (then discard) fix the bug? | `SGLANG_TT_U22_BARRIER_CLONE_W2=1` | NO — under zero weights "appears" clean (deterministic 39075 collapse, PRE_RS at 0xa6700 = 0/5760 NONZERO) but under real weights output is garbage low-id tokens (clone's NoC read perturbs subsequent RS reader semantics) |

**Punchline:** All Python-level sharding-preserving relocation
primitives in the ttnn surface are exhausted as workarounds.
The dispatcher's direct L1 writes (via `process_write_*` paths
in cq_dispatch.cpp) do not target 0xa6700.  The remaining
stomper candidate set is **fabric / ethernet / EDM completion
acks** (Hypothesis 1 below), **a non-dispatcher kernel** (e.g.,
a prefetcher op's READER, not WRITER), or **a UMD host-side
write between trace replays** (very unlikely).

## Phase 1 — Path B: sharding-preserving relocation primitives

### Setup

Added 5 mutually-exclusive env-gated variants to `mlp.py`,
each acting on `w2_out` between `w2_out = ttnn.linear(...)`
and `w2_out_reduced = tt_all_reduce(w2_out, ...)`.  Variants:

| Env var | Operation |
|---|---|
| `SGLANG_TT_U22_RESHARD_W2=1` | `w2_out = ttnn.to_memory_config(w2_out, w2_out.memory_config())` |
| `SGLANG_TT_U22_REALLOCATE_W2=1` | `w2_out = ttnn.reallocate(w2_out, w2_out.memory_config())` |
| `SGLANG_TT_U22_ASSIGN_W2=1` | `w2_out = ttnn.assign(w2_out, memory_config=w2_out.memory_config())` |
| `SGLANG_TT_U22_SNAPSHOT_RESTORE_W2=1` | `snap = ttnn.clone(w2_out); ttnn.copy(snap, w2_out); ttnn.deallocate(snap)` |
| `SGLANG_TT_U22_BARRIER_CLONE_W2=1` | `snap = ttnn.clone(w2_out); ttnn.deallocate(snap)` (discard) |

Diagnostic addr print also added (`SGLANG_TT_U22_PRINT_ADDR=1`).

### Result B.REALLOCATE — the most informative case

```bash
SGLANG_TT_USE_PREFETCHER=1
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1
SGLANG_TT_U22_REALLOCATE_W2=1
SGLANG_TT_U22_PRINT_ADDR=1
SGLANG_TT_U17_PROBE_RS_PRE=1
```

```
[U22_REALLOCATE_W2] old=0xa6700 new=0xaa700
[U22_REALLOCATE_W2] old=0xa4700 new=0xaa700      # per-core differing src
... (72 prints for 2-token decode)

U17 PRE_RS at 0xa6700: NONZERO=0 zero=0          # no longer probed!
U17 PRE_RS at 0xaa700: NONZERO=0 zero=8640       # RS reader at NEW addr
```

`ttnn.reallocate` (via `ttnn::move` → `move_sharded`) correctly
moves w2_out from L1 0xa6700 to 0xaa700.  The RS reader's NoC
read target is updated; it now consumes from 0xaa700 (clean).
The L1 0xa6700 slot is freed.

Zero-weight output: `21631, 114137, 114107, 114179, 114179,
114179, 46, 153, ...` — racy garbage (114179 repeats but
46 / 153 are dramatically off).  Compare to:
- U19 FILL: deterministic 38649 constant collapse (correct).
- U21 CLONE: deterministic 39228 constant collapse (mostly
  correct).
- U22 REALLOCATE: racy non-deterministic garbage (broken).

Real-weight output: `What;7�wN.?:..$C$+` — fully garbage.

### Root cause of REALLOCATE failure — move_sharded per-core chunk-size bug

`ttnn/cpp/ttnn/operations/data_movement/move/device/move_sharded_program_factory.cpp`:

```cpp
const uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
```

`move_chunk_size_bytes` is computed from BANK-AVERAGED addresses
(input = bank-averaged 0xa6700, output = bank-averaged 0xaa700,
chunk = 0x4000).  But w2_out's per-core L1 base is 0xa6700 on
SOME cores and 0xa4700 on OTHERS (U20 confirmed:
`local_in0=0xa6700, local_in0=0xa4700, ...`).  The kernel
`reader_unary_local_l1_copy_backwards.cpp` does:

```cpp
uint32_t src_cb_addr = src_cb_base_addr + total_size_bytes;
uint32_t dst_cb_addr = dst_cb_base_addr + total_size_bytes;
// for each chunk:
src_cb_addr -= chunk_size_bytes;
dst_cb_addr -= chunk_size_bytes;
noc_async_read(get_noc_addr(src_cb_addr), dst_cb_addr, chunk_size_bytes);
```

On cores where dst=0xaa700 and chunk=0x4000, the backward copy
reads from `dst - chunk = 0xa6700` (correct for cores where
src=0xa6700).  But on cores where src=0xa4700 (delta to dst is
0x6000), the kernel STILL uses chunk=0x4000 (it's a compile-time
arg, not per-core).  So those cores read from
`0xaa700 - 0x4000 = 0xa6700` — but their REAL data is at 0xa4700.
They copy STALE / WRONG L1 data on those cores.

**This is a tt-metal `move_sharded` defect for multi-shard-base-addr
buffers.**  Not fixable from sglang Python.  Filed mentally as a
candidate tt-metal upstream issue.

### Result B.BARRIER_CLONE — the seductive false positive

```
[U22_BARRIER_CLONE_W2] w2_addr=0xa6700           # unchanged
U17 PRE_RS at 0xa6700: NONZERO=0 zero=5760       # appears clean!
```

Zero-weight output: deterministic 39075 collapse.  Looks like
a fix.

But under REAL weights (no PREFETCHER_ZERO_WEIGHTS):
`What!2&.a=!26%!-3S` — garbage tokens 0,17,5,13,64,28,... .

The "clean" U17 PRE_RS reading under zero weights is an artifact:
W2's correct output IS zero, so the L1 stays at zero either
because the stomp didn't fire OR because the stomp DID fire and
the value happens to coincide with zero.  Re-checking with a
NEW U17 PRE_RS run under REAL weights + BARRIER_CLONE shows
NONZERO=5760 zero=0 at 0xa6700 — the L1 holds nonzero bytes
(which is correct, since W2's real output is nonzero), but the
MODEL OUTPUT is still garbage.  Conclusion: the clone's NoC
read of 0xa6700 perturbs subsequent RS reader semantics in a
way that breaks real-weight correctness.

## Phase 2 — Path A: dispatcher write trace

### Probe

Added env-gated `SGLANG_TT_U22_DISP_TRACE` define in
`dispatch.cpp` (propagated to kernel via `defines["SGLANG_TT_U22_DISP_TRACE"]`)
and `cq_dispatch.cpp` (DPRINT in process_write_linear,
process_write_packed, process_write_packed_large when dst_addr
in [0xa6000, 0xa7000]).  Verified via log_info that the define
is set and via `strings _ttnn.so | grep U22_DISP` that the
kernel binary contains the probe text.

### Pre-run pivot — wide sniff (every write)

Initial narrow-filter probe produced ZERO entries.  Widened to
"first 32 writes of each kind" to verify the probe FIRES and
to discover the actual address range:

| Probe | Address range observed | Targets 0xa6700? |
|---|---|---|
| `process_write_linear` (sniff first 32) | 0x17ffc0 (single GO-msg slot, 16 B each, varying noc 0x81..0x8c) | NO |
| `process_write_packed_large` (sniff first 32) | 0x9e30, 0x9f40, 0xa2e0, 0xa590, 0xaa40, 0xacb0, 0xb4d0, 0xb5e0, 0xb980, 0xbc30, 0xc0e0, 0xc350, 0xcb50, 0xcc60, 0xcd90, 0xcee0, 0xd380, 0xd5b0, 0xdba0, 0xdcb0 (all KERNEL_CONFIG slots) | NO |
| `process_write_packed` (sniff first 32) | never fired (no small packed writes for this workload) | N/A |

The address `0xa6700` is NEVER in the observed set.  Dispatcher
direct writes RULED OUT.

### Aside — the narrow-filter probe didn't fire because dispatcher writes don't target 0xa6700

The narrow-filter probe (only print when dst_addr in
[0xa6000, 0xa7000]) produced 0 entries because no dispatcher
write targets that range.  Verified by widening to "first 32 of
any addr" — the dispatcher writes target adjacent ranges
(0xa2e0, 0xa590, 0xaa40, etc.) but skip 0xa6700.

This is consistent with KERNEL_CONFIG allocation: each program's
runtime args / kernel binaries get distinct slots, and 0xa6700
happens to be in a gap (or alignment skip) between consecutive
program configs.  Or it's in a region the allocator uses for
USER buffers (above KERNEL_CONFIG), and the dispatcher doesn't
write user-region addresses — that's the operator kernels' job.

## Hypothesis ledger (post-U22)

| ID | Suspect | Pre-U22 | Post-U22 |
|---|---|---|---|
| (all previous) | … | per U21 | unchanged |
| **U22 A — `process_write_linear` targets 0xa6700** | (new) | **RULED OUT** | only writes to 0x17ffc0 (GO-msg) |
| **U22 A — `process_write_packed_large` targets 0xa6700** | (new) | **RULED OUT** | writes to KERNEL_CONFIG range 0x9e30..0xdcb0 only |
| **U22 A — `process_write_packed` (small) targets 0xa6700** | (new) | **RULED OUT** | never fires for this workload |
| **U22 B — `ttnn.to_memory_config` reshard preserves sharding AND fixes stomp** | (new) | **RULED OUT** | TT_FATAL is_allocated (no-op return + dealloc) |
| **U22 B — `ttnn.reallocate` (ttnn::move) preserves sharding AND fixes stomp** | (new) | **PARTIAL → RULED OUT** | moves buffer correctly but `move_sharded` per-core chunk-size bug corrupts data on cores with base 0xa4700 |
| **U22 B — `ttnn.assign(input, memory_config)` fixes stomp** | (new) | **RULED OUT** | sub-device intersection error |
| **U22 B — `ttnn.clone + ttnn.copy(snap, w2_out)` snapshot-and-restore** | (new) | **RULED OUT** | copy step fails with sub-device intersection error |
| **U22 B — `ttnn.clone` as a no-op dispatch barrier (discard snapshot)** | (new) | **RULED OUT** | under zero weights "appears" clean (illusion); under real weights output is garbage |
| **U22 STANDING — what writes garbage to receiver-core (2,7) L1 0xa6700 between W2 PACK exit and RS reader entry?** | (new) | **STANDING — TOP PRIORITY** | candidate set now: fabric/EDM completion acks → tensix L1; some non-dispatcher kernel running on (2,7); UMD host-side write between trace replays (unlikely) |

## Remaining stomper candidate set (very narrow now)

After U17 + U18 + U19 + U20 + U21 + U22, ruled out:
- All W2 program kernels (compute + in0_ag + in1_ag)
- All prefetcher kernels (writer_l1 already; reader_dram TBD)
- The RS writer kernel
- The RS mux kernel
- Launch_msg / KERNEL_CONFIG region (firmware turf)
- Dispatcher's process_write_linear / process_write_packed
  / process_write_packed_large (this session)
- All ttnn-level sharding-preserving relocation primitives
  (5 variants, all fail)

The remaining possibilities:
1. **Fabric / EDM completion acks** routed from ethernet cores
   to tensix L1 0xa6700 on (2,7).
2. **A persistent kernel on a different sub-device** (e.g.,
   the dram_prefetcher's reader_dram, or a fabric router
   kernel) that writes to (2,7) L1.
3. **The prefetcher's `update_remote_cb_config_in_l1`**
   bookkeeping writes (per U21 recommendation Phase B; not
   probed in U22).

## What U22 lands (all env-gated; canonical bytewise-equal)

| File | Change |
|---|---|
| `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` | `SGLANG_TT_U22_DISP_TRACE` probes in process_write_linear, process_write_packed, process_write_packed_large (dst_addr in [0xa6000, 0xa7000]). |
| `tt_metal/impl/dispatch/kernel_config/dispatch.cpp` | env-gated `defines["SGLANG_TT_U22_DISP_TRACE"]` propagation + `log_info` trace. |
| `models/tt_transformers/tt/mlp.py` | 5 env-gated Path-B variants: `SGLANG_TT_U22_RESHARD_W2`, `_REALLOCATE_W2`, `_ASSIGN_W2`, `_SNAPSHOT_RESTORE_W2`, `_BARRIER_CLONE_W2`. |
| `python/sglang/srt/hardware_backend/tenstorrent/scripts/u22_run_one.sh` | helper launcher used for all U22 runs (env-pass-through, cache clear, /generate, kill). |

## Reproducer

```bash
# Path B — REALLOCATE under zero weights (shows buffer correctly
# moves to 0xaa700 but model output is racy garbage):
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U22_REALLOCATE_W2=1 \
SGLANG_TT_U22_PRINT_ADDR=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u22_realloc.log \
TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
python3 -u -m sglang.launch_server ... &
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'

grep "U22_REALLOCATE_W2" /tmp/u22_realloc.log | head -5
# [U22_REALLOCATE_W2] old=0xa6700 new=0xaa700
# [U22_REALLOCATE_W2] old=0xa4700 new=0xaa700
grep -c "U17_PRE_RS in_addr=0xaa700.*NONZERO" /tmp/u22_realloc.log   # 0
grep -c "U17_PRE_RS in_addr=0xaa700.* zero\]"  /tmp/u22_realloc.log   # 8640

# Path A — dispatcher write trace under zero weights:
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U22_DISP_TRACE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u22_disp.log \
python3 -u -m sglang.launch_server ... &
curl ...

grep -c "U22_DISP wl" /tmp/u22_disp.log   # 0 (no linear writes to 0xa6700)
grep -c "U22_DISP wpl" /tmp/u22_disp.log  # 0 (no packed_large writes to 0xa6700)

# Canonical re-verify (no prefetcher, no probes):
SGLANG_TT_USE_PREFETCHER=0 \
python3 -u -m sglang.launch_server ... &
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Output: " What is 2+2? What is 2+2? What"  e2e_latency 9.05s
```

## U23 dispatch recommendation

The stomp at L1 0xa6700 is in a very narrow remaining candidate
set.  Highest-leverage next attacks:

### U23 Phase A — fabric / EDM completion ack probe

Instrument ethernet core kernels (fabric routers / EDM
forwarders) to dump every tensix L1 write target.  Specifically:
- `tt_metal/fabric/...` ethernet kernel sources
- look for `noc_async_write` / `cq_noc_async_write_with_state`
  targeting tensix L1 (NoC XY mapping (2,7) virtual coord =
  `0x1c20` per U19/U21 prior data)
- gate on `dst_addr in [0xa6000, 0xa7000]`

### U23 Phase B — host-side L1 poller kernel

Per U21/U22 Phase C recommendation: write a tiny custom kernel
that takes (noc XY, L1 offset, poll_iter) as RT args and polls
L1 every N cycles, DPRINTing on change.  Dispatch on a clean
sub-device that has only the poller core.  Inserted between W2
and RS in the Python pipeline (captured into trace).  The
zero→NONZERO transition timestamp pinpoints the stomp event.

### U23 Phase C — fix the `move_sharded` per-core chunk-size bug upstream

If the underlying tt-metal `move_sharded` bug were fixed (use
per-core src/dst delta instead of bank-averaged delta), then
U22's REALLOCATE workaround WOULD give a real-weight workaround
that preserves sharding.  This is a small, well-scoped tt-metal
fix.  Worth considering as a parallel upstream contribution.

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal C++ + mlp.py,
      all env-gated)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
      (u22_run_one.sh uses `TT_CACHE_HOME=/root/.cache/tt-metal-cache`
      with `[ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ]`)
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`)
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Container is NOT bind-mounted — used `podman cp` for source
      sync + `rebuild_tt_metal_kernels.sh` for relinking
- [x] All new probes / variants env-gated (`SGLANG_TT_U22_*`)
- [x] Canonical re-verified at session end: 9.05s e2e_latency,
      `" What is 2+2? What is 2+2? What"` coherent output

## Shipping verdict (unchanged from U19/U20/U21)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat ~ 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**
  The L1 0xa6700 stomp persists; no real-weight-preserving
  workaround found.
- **DO NOT ship any `SGLANG_TT_U22_*` env var.**  Diagnostic only.

## Commits this session

- (tt-metal-sglang) **`d1448de43b9`** —
  `prefetcher: U22 — Path A (dispatcher write trace) RULES OUT dispatcher writes to L1 0xa6700; Path B sharding-preserving relocation primitives all break under real weights (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`d1448de43b9`** (U22 commit; env-gated;
  default-off).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.
- Container `/tt-metal/`: synced (cq_dispatch.cpp + dispatch.cpp
  + mlp.py with U22 probes/variants applied; rebuilt
  `_ttnn.so` / `_ttnncpp.so`; verified `SGLANG_TT_U22_DISP_TRACE`
  string in `libtt_metal.so`).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at
  9.05s e2e_latency (bug absent; coherent output; no regression).
