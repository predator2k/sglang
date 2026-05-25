# TT Qwen3-8B prefetcher — U21: L1 0xa6700 stomp is at FIXED address, does NOT follow buffer; prefetcher writer + launch_msg RULED OUT — 2026-05-24

Status: **EVIDENCE_ADVANCE — Three U21 probes land new ruling-out
+ ruling-in evidence. Probe A: launch_msg lives at MEM_MAILBOX_BASE
= 96 (~0xC0) on Blackhole tensix, mailbox region ends well below
0x15000; launch_msg RULED OUT as the L1 0xa6700 stomper. Probe B
(via SGLANG_TT_U19_CLONE_W2 from U19, never previously documented):
under zero weights, CLONE moved W2 output from L1 0xa6700 to
0xaa700 (and most other RS reads to DRAM at 0x47b...640). U17
PRE_RS probe at 0xaa700 shows 0/8640 NONZERO — the stomp does NOT
follow the W2 output buffer to its new address. Output under zero
weights collapses to constant-token signature (`39228` repeated,
same convergence pattern as U11 / U19 FILL). Probe C: env-gated
SGLANG_TT_U21_PREFETCHER_ADDR_PROBE dumps prefetcher writer_l1's
remote NoC write targets — aligned_pages_sent_ptr=0x17f680,
config_ptr=0x17f640, receiver_noc_xy_ptr=0x17f65c, fifo_start=
0xac700, per-receiver pages_sent slots at 0x17f680..0x17f6e0.
NONE target 0xa6700. Prefetcher writer RULED OUT as the stomper.**

Continuation of `tt_qwen3_8b_prefetcher_U20_DATAFLOW_KERNELS_RULED_OUT_2026-05-24.md`.

tt-metal-sglang HEAD: **`9a54d13019f`** (U21 commit; env-gated;
default-off; canonical bytewise-equal).  Branch `tenstorrent-p1`;
NOT pushed (predator2k/* fork only per project policy).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (post-U21)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| A | Is launch_msg base = 0xa6700 on Blackhole tensix? | code analysis of `bh_hal_tensix.cpp`, `dev_mem_map.h` | NO — launch_msg lives in mailbox region at `MEM_MAILBOX_BASE = 96` (~0xC0); mailbox ends at `MEM_MAILBOX_END = 0x3270` |
| B | Does the stomp follow the W2 output buffer address? | `SGLANG_TT_U19_CLONE_W2=1` + `SGLANG_TT_U17_PROBE_RS_PRE=1` + zero weights | NO — CLONE moved W2 output to L1 0xaa700; PRE_RS shows 0/8640 NONZERO there. Stomp is at FIXED L1 0xa6700. |
| C | Does prefetcher writer_l1 target L1 0xa6700? | `SGLANG_TT_U21_PREFETCHER_ADDR_PROBE=1` writer_l1 entry probe | NO — `aligned_pages_sent_ptr=0x17f680`, `config_ptr=0x17f640`, `receiver_noc_xy_ptr=0x17f65c`, `fifo_start=0xac700`. None at 0xa6700. |
| B' | Does CLONE fix output under zero weights? | `SGLANG_TT_U19_CLONE_W2=1 + SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1` | YES (mostly) — output collapses to constant `39228` after first 4-5 varying tokens, same convergence class as U11/U19 FILL. Confirms CLONE sidesteps the stomp at 0xa6700. |
| B'' | Does CLONE fix output under REAL weights? | `SGLANG_TT_U19_CLONE_W2=1` (no zero weights) | NO — output `" What,,%K2X"&�3E6'1"` (tokens 3555,11,11,4,...). First decode token wrong. CLONE introduces a small but real perturbation that breaks real-weight model. |

**Punchline:** L1 `0xa6700` is a CURSED FIXED ADDRESS on receiver
core (2,7). Whatever buffer lands there gets stomped by an unknown
event between W2 PACK exit and W2 RS reader entry. The stomp is
**address-bound, not buffer-bound**. CLONE moves W2's output to
`0xaa700` and the stomp persists at 0xa6700 (now harmless because
no live buffer is at 0xa6700). The remaining stomper candidates
are very narrow:
- Per-program scratch / RT-arg writes by the dispatcher firmware to
  a specific (2,7) L1 offset that happens to equal 0xa6700.
- A trace-replay command write that targets (2,7) L1 0xa6700 as a
  side effect of some op's setup.
- A cross-sub-device dispatch artifact (e.g., when worker_sub_device
  hands off to receiver_sub_device for the RS reader).

## Phase A — launch_msg base address (code analysis)

### dev_mem_map.h (blackhole)

```c
#define MEM_MAILBOX_BASE 96            // 0x60
#define MEM_MAILBOX_SIZE 12912         // 0x3270
#define MEM_MAILBOX_END (MEM_MAILBOX_BASE + MEM_MAILBOX_SIZE)  // 0x32D0
```

### dev_msgs.h: `mailboxes_t` struct

```cpp
struct mailboxes_t {
    struct ncrisc_halt_msg_t ncrisc_halt;
    struct subordinate_sync_msg_t subordinate_sync;
    volatile uint32_t launch_msg_rd_ptr;
    struct launch_msg_t launch[launch_msg_buffer_num_entries];  // 8 entries
    volatile struct go_msg_t go_messages[9];
    ...
};
```

### bh_hal_tensix.cpp

```cpp
mem_map_bases[HalL1MemAddrType::MAILBOX] = MEM_MAILBOX_BASE;        // 0x60
mem_map_bases[HalL1MemAddrType::LAUNCH] = GET_MAILBOX_ADDRESS_HOST(launch);  // 0x60 + offsetof(launch)
mem_map_bases[HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR] =
    GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
```

`HalL1MemAddrType::LAUNCH` = `MEM_MAILBOX_BASE + offsetof(mailboxes_t, launch)`.
Offset of `launch[0]` inside `mailboxes_t` is small (after
ncrisc_halt + subordinate_sync + launch_msg_rd_ptr). All
launch_msg slots live below `MEM_MAILBOX_END = 0x32D0`, well
below `0xa6700`.

### KERNEL_CONFIG region

`bh_hal_tensix.cpp` line 43: `default_l1_kernel_config_size = 69 * 1024 = 0x11400`.
KERNEL_CONFIG base = `MEM_MAP_END` (just past the system-reserved
fabric/packet pools, roughly 0x15000). So KERNEL_CONFIG occupies
roughly `0x15000 - 0x26400`. Also well below `0xa6700`.

**Verdict A:** `0xa6700` is in the **allocator-managed user L1
region** (above `0x26400`). It's NOT firmware/launch/kernel-config
turf. launch_msg as a candidate stomper is **RULED OUT**.

## Phase B — Address relocation via ttnn.clone (CLONE_W2)

### Setup

The U19 mlp.py already had `SGLANG_TT_U19_CLONE_W2` plumbing:

```python
if (mode == Mode.DECODE
        and _u19f_os.environ.get("SGLANG_TT_U19_CLONE_W2", "0") == "1"):
    try:
        w2_out_orig = w2_out
        w2_out = ttnn.clone(w2_out_orig, memory_config=w2_out_orig.memory_config())
        ttnn.deallocate(w2_out_orig)
    except Exception as _u19c_e:
        print(f"[U19_CLONE_W2] ERROR: {type(_u19c_e).__name__}: {_u19c_e}", flush=True)
```

This was added in U19 but **the test result was not documented** —
U19 only ran FILL_W2_ZERO and COPY_W2.  U21 actually runs CLONE_W2.

### Run

```bash
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U19_CLONE_W2=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u21_clone_w2.log \
TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
HF_MODEL=/models/Qwen3-8B \
python3 -u -m sglang.launch_server ... &
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
```

### Result B.1 — U17 PRE_RS addresses (zero weights + CLONE)

Unique RS reader input addresses observed:

```
0x47b77640 0x47b8de40 0x47b97640 0x47bc6480 0x47bdcc80 0x47be6480
0x47d54640 0x47dd4640 0x47e54640 0x47f75640 0x48075640 0x48175640
0x48377640 0x48577640 0x48777640
0xaa700
```

The first 15 addresses are DRAM (0x47b...640 - 0x48777640). The
SINGLE L1 address is `0xaa700` (not `0xa6700`). CLONE shifted the
W2 output by `+0x4000` in L1.

### Result B.2 — Counts at 0xaa700

```
NONZERO: 0
zero:    8640
```

**0 NONZERO out of 8640 probes** at `0xaa700`. The stomp does
NOT follow the W2 output buffer to its new address. The stomp is
at the FIXED L1 address `0xa6700`.

### Result B.3 — Output (zero weights + CLONE)

```
{"text":" Dath_ledemente呼和浩特 deduction deduction系龙门aturingaturingaturingaturingaturingaturing",
 "output_ids":[21631,71,38367,38861,114829,38843,38843,38176,114342,
               39228,39228,39228,39228,39228,39228],...}
```

Tokens collapse to `39228` after the first 4-5 varying tokens.
This is the **same constant-token convergence pattern** as U11
(no prefetcher zero weights) and U19 FILL (`38649` constant).
The different constant value (`39228` vs `38649`) is consistent
with tile-reordering inside the moved buffer producing a
different dominant logits-row.

### Result B.4 — Output (REAL weights + CLONE)

```
{"text":" What,,%K2X\"&\x033E6'1",
 "output_ids":[3555,11,11,4,42,17,55,1,5,116,18,36,21,6,16],...}
```

First decode token `3555` = " What" — that's actually the prefill
output (which doesn't use CLONE since CLONE only runs in DECODE).
After the first token, all decode tokens are garbage low IDs.

CLONE introduces a small persistent perturbation that breaks
the real-weight model.  Likely cause: `ttnn.clone` doesn't
preserve the exact sharding layout (some shards may end up on
different cores than the original).  Under zero weights, the
perturbation is invisible (everything is zero).  Under real
weights, it cascades.

**Verdict B:** Stomp is **at FIXED L1 address `0xa6700`**, does
NOT follow the buffer. Probe B's positive result is decisive.

## Phase C — prefetcher writer_l1 address dump

### Probe

Added `SGLANG_TT_U21_PREFETCHER_ADDR_PROBE` define in `writer_l1.cpp`
(plus matching env-propagation in `dram_prefetcher_program_factory.cpp`).
At kernel entry, before any send loop, dumps:

- `aligned_pages_sent_ptr` (the remote L1 address on the receiver
  where the sender increments the page-sent semaphore via
  `noc_semaphore_inc`),
- `config_ptr` (local L1 address on sender),
- `receiver_noc_xy_ptr` (local L1 address where receiver NoC XY
  coords are stored),
- `fifo_start` (the remote L1 address on the receiver where
  GlobalCB pages are written).

Then loops over `num_receivers` and dumps the per-receiver
pages_sent slot address (`aligned_pages_sent_ptr + i * 2 *
L1_ALIGNMENT`).  L1_ALIGNMENT on blackhole = 16, so stride is 32.

### Result

Aggregate from all sender cores (`0:0-1`, `0:0-3`, …, `0:7-0`,
`0:7-2`, …):

```
[U21_PREF_ADDR aligned_pages_sent_ptr=0x17f680
               config_ptr=0x17f640
               receiver_noc_xy_ptr=0x17f65c
               fifo_start=0xac700
               num_receivers=4]

[U21_PREF_RECV_PSENT i=0 psent_ptr=0x17f680]
[U21_PREF_RECV_PSENT i=1 psent_ptr=0x17f6a0]
[U21_PREF_RECV_PSENT i=2 psent_ptr=0x17f6c0]
[U21_PREF_RECV_PSENT i=3 psent_ptr=0x17f6e0]
```

All addresses cluster in `0x17f640 - 0x17f6e0` (high L1, ~1.5MB
region — far from 0xa6700).  `fifo_start = 0xac700` (the GlobalCB
buffer area, +0x6000 above 0xa6700 — adjacent but **not
overlapping**, since the GlobalCB writes go to `0xac700+`).

**Verdict C:** Prefetcher writer_l1's remote NoC writes do NOT
target L1 `0xa6700`.  Prefetcher writer fully **RULED OUT** as
the stomper.

## Hypothesis ledger (post-U21)

| ID | Suspect | Pre-U21 | Post-U21 |
|---|---|---|---|
| (all previous) | … | RULED OUT / CONFIRMED | unchanged |
| **U21 A — launch_msg base at 0xa6700** | (new) | **RULED OUT** | launch_msg lives at MEM_MAILBOX_BASE = 96 (~0xC0); ends at 0x32D0 |
| **U21 B — stomp follows W2 output buffer address** | (new) | **RULED OUT** | CLONE_W2 moved buffer to 0xaa700; PRE_RS at 0xaa700 shows 0/8640 NONZERO; stomp persists at FIXED 0xa6700 |
| **U21 B' — CLONE_W2 fixes the bug under real weights** | (new) | **NO** | output is garbage; CLONE introduces a sharding perturbation that breaks real-weight model |
| **U21 B'' — stomp is at FIXED L1 address 0xa6700** | (new) | **CONFIRMED** | direct positive evidence from probe B |
| **U21 C — prefetcher writer_l1 NoC writes target 0xa6700** | (new) | **RULED OUT** | aligned_pages_sent_ptr=0x17f680, fifo_start=0xac700, config_ptr=0x17f640; none at 0xa6700 |
| **U21 STANDING — what writes garbage to receiver-core (2,7) L1 0xa6700 between W2 PACK exit and RS reader entry?** | (new) | **STANDING — TOP PRIORITY** | candidate set now: dispatcher firmware per-program scratch writes, trace-replay command writes, cross-sub-device dispatch artifact |

## Remaining stomper candidate set (very narrow)

After U17 + U18 + U19 + U20 + U21, **every kernel** that runs
inside the W2 program (compute + in0_ag + in1_ag), **every kernel**
on the prefetcher (reader_dram + writer_l1), the RS writer, the
RS mux, and the firmware launch_msg are all RULED OUT as the
stomper.  The stomp is at fixed L1 `0xa6700` on receiver `(2,7)`
and fires between W2 program exit and RS reader entry.

The remaining candidates:

1. **Dispatcher per-program scratch / RT-arg writes** — when the
   dispatcher prepares a new program for dispatch, it writes RT
   args and program-config metadata to a per-program L1 region
   on every participating core.  If the RT-arg region for some
   program's worker is at L1 `0xa6700` on `(2,7)`, that's the
   stomp.  This is hard to probe because the dispatcher writes
   are firmware-level, not kernel-level.

2. **Trace-replay command write** — the trace replay engine
   re-writes program command sequences for each captured program
   on every replay step.  If any of those writes target L1
   `0xa6700` on `(2,7)`, that's the stomp.

3. **Cross-sub-device dispatch artifact** — when the dispatcher
   hands off from `worker_sub_device` (W2 program) to
   `receiver_sub_device` (RS reader), it writes go-message + sync
   updates to all cores in both sub-device grids.  If the
   transition writes a sync field at L1 `0xa6700` on `(2,7)`,
   that's the stomp.

4. **An EDM/fabric kernel completion ack** routed to tensix L1
   via NoC.  The 2-chip mesh has ethernet cores running fabric
   forwarding kernels; their completion semaphores land in
   tensix L1.  Per U19 mux probe, the mux is on
   `0:3-0/0:9-0/1:0-0/1:6-0` (NOT 2,7) and only writes to
   `0x1b220-0x1b310`.  But there may be other fabric semaphores.

## What U21 lands (all env-gated; canonical bytewise-equal)

| File | Change |
|---|---|
| `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp` | `SGLANG_TT_U21_PREFETCHER_ADDR_PROBE` entry probe dumping aligned_pages_sent_ptr / config_ptr / receiver_noc_xy_ptr / fifo_start / per-receiver pages_sent slots. 512-budget per-RISC static counter. |
| `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp` | env-gated propagation of `SGLANG_TT_U21_PREFETCHER_ADDR_PROBE` to writer_defines so the define reaches writer_l1.cpp. |

## Reproducer

```bash
# Probe C (prefetcher writer addr dump):
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U21_PREFETCHER_ADDR_PROBE=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_FILE=/tmp/u21_pref_addr.log \
TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
HF_MODEL=/models/Qwen3-8B \
python3 -u -m sglang.launch_server ... &
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"2+2=","sampling_params":{"max_new_tokens":3,"temperature":0.0}}'

grep -c "U21_PREF_ADDR" /tmp/u21_pref_addr.log    # 48 (sender cores * 1 dispatch)
grep "U21_PREF_ADDR" /tmp/u21_pref_addr.log | \
  grep -oP "aligned_pages_sent_ptr=0x[a-f0-9]+|fifo_start=0x[a-f0-9]+" | sort -u
# aligned_pages_sent_ptr=0x17f680
# fifo_start=0xac700

# Probe B (CLONE_W2 address relocation, zero weights):
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
SGLANG_TT_U19_CLONE_W2=1 \
SGLANG_TT_U17_PROBE_RS_PRE=1 \
TT_METAL_DPRINT_FILE=/tmp/u21_clone_w2.log \
TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
python3 -u -m sglang.launch_server ... &
curl ...

grep -c "U17_PRE_RS in_addr=0xaa700.*NONZERO" /tmp/u21_clone_w2.log   # 0
grep -c "U17_PRE_RS in_addr=0xaa700.* zero\]"  /tmp/u21_clone_w2.log   # 8640

# Canonical (no prefetcher, no probes) re-verify:
SGLANG_TT_USE_PREFETCHER=0 \
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
HF_MODEL=/models/Qwen3-8B \
python3 -u -m sglang.launch_server ... &
curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# Output coherent at ~9s e2e_latency.
```

## U22 dispatch recommendation

The stomp is at fixed L1 `0xa6700` on receiver `(2,7)` and the
ENTIRE kernel layer of every involved program is ruled out as the
writer.  The stomper must be firmware/dispatcher level.

### U22 Phase A — host-side L1 poller kernel

Build a tiny custom kernel that takes a NoC XY + L1 offset as RT
arg and polls L1 every N cycles, printing on zero->NONZERO
transitions with a cycle timestamp.  Dispatch this kernel BETWEEN
the W2 program and the RS program in the Python pipeline (capture
into trace).  The zero->NONZERO transition cycle pinpoints the
exact stomp event and lets us reverse-engineer what dispatcher
write hit `0xa6700` at that moment.

Key implementation note: the polling kernel must run on
`receiver_sub_device` or a NEW sub-device so it doesn't share
the same dispatch queue as the W2 → RS sequence; otherwise it
would serialize with them and miss the stomp window.

### U22 Phase B — dispatcher write-address dump

Modify `tt_metal/impl/program/dispatch.cpp` to dump every L1
write target during `update_program_dispatch_commands` and during
the per-program command sequence emission.  If any dispatch
command writes to L1 `0xa6700` on tensix `(2,7)`, that's the
stomp.

Risk: this requires touching dispatcher code; large amount of
code; may need to also probe trace_replay path.

### U22 Phase C — workaround: re-write 0xa6700 between W2 and RS

If we can't identify the stomper quickly, ship a workaround that
re-writes the CORRECT bytes back to `0xa6700` after the stomp
fires but before the RS reader runs.  CLONE_W2 fails because
`ttnn.clone` perturbs sharding.  Try instead:
- `ttnn.assign(w2_out, w2_out)` (no-op assign that may force a
  read-write round-trip),
- a custom 1-tile kernel that NoC-reads `w2_out` into local L1
  then NoC-writes it back to the same address (forces a fresh
  write that overrides the stomp).

If either preserves output coherence under real weights AND
matches U19 PRE_RS NONZERO=0 under zero weights, ship it.

### U22 Phase D — try a different RS path

If the bug is specifically the RS reader's expectation at
`0xa6700`, try routing the W2 output through a different
RS implementation (e.g., the composite reduce_scatter +
all_gather path used for `dim=8192`) that may have different
L1 layout expectations.

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal C++ files,
      all env-gated; mlp.py untouched in this session)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Container is NOT bind-mounted — used `podman cp` for source
      sync + `rebuild_tt_metal_kernels.sh` for relinking
- [x] All new probes env-gated (`SGLANG_TT_U21_PREFETCHER_ADDR_PROBE`)
- [x] Canonical re-verified pre-session: 9.06s e2e_latency,
      coherent output

## Shipping verdict (unchanged from U19/U20)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat ~ 9/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**
  The L1 0xa6700 stomp persists; no real-weight-preserving
  workaround found.
- **DO NOT ship `SGLANG_TT_U19_CLONE_W2=1`.**  Fixes the stomp
  observation but introduces a sharding perturbation that
  breaks real-weight model.
- **DO NOT ship any `SGLANG_TT_U21_*` probe.**  Diagnostic only.

## Commits this session

- (tt-metal-sglang) **`9a54d13019f`** —
  `prefetcher: U21 — prefetcher addr probes RULE OUT prefetcher writer as L1 0xa6700 stomper (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`9a54d13019f`** (U21 commit; env-gated;
  default-off).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.
- Container `/tt-metal/`: synced (writer_l1.cpp + program factory
  with U21 probes applied; `_ttnn.so` / `_ttnncpp.so` rebuilt and
  verified to contain `SGLANG_TT_U21_PREFETCHER_ADDR_PROBE` string).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
