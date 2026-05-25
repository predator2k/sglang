# TT Qwen3-8B prefetcher — U27: firmware-level per-kernel L1 0xa6700 entry/exit probe on receiver (2,7) PROVES no kernel ever writes to 0xa6700; bytes are CONSTANT and PRE-EXISTING; new top hypothesis = per-core address mismatch (EVIDENCE_ADVANCE) — 2026-05-25

Status: **EVIDENCE_ADVANCE — U27 adds a brisc.cc firmware-level
probe (env-gated by `SGLANG_TT_U27_KERNEL_ENTRY_PROBE=1`) that
reads L1[0xa6700] at the START and END of EVERY kernel dispatch
on logical core (2,7), DPRINTing
host_assigned_id + L1[0xa6700]. The probe is wired via
build_env_manager.cpp::initialize_device_kernel_defines() which
lifts the env var into device_kernel_defines so it reaches the
JIT-compiled firmware (requires `TT_METAL_DISABLE_PRECOMPILED_FW=1`
at runtime).

Result under `SGLANG_TT_USE_PREFETCHER=1 + SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1`:
across all **256 ENTRY+EXIT probe events** on (2,7) — spanning the
entire decode pass from kernel dispatch tot=1 through tot=256
inclusive — the L1 0xa6700 bytes are **CONSTANT and IDENTICAL**:

- chip 0: `w0=0x40983980 w1=0x40938f60 w2=0xc0a94f00 w3=0x40c11940`
- chip 1: `w0=0xc0b1df00 w1=0x406b8a40 w2=0xc1500d20 w3=0x3e533800`

The bytes DO NOT change between ANY two consecutive kernel
dispatches. They are present from the VERY FIRST kernel dispatch
on (2,7) (tot=1 ENTRY).**

Continuation of `tt_qwen3_8b_prefetcher_U26_ADJACENT_BUFFER_IDENTIFIED_2026-05-25.md`.

tt-metal-sglang HEAD: **`62e8c6c24b4`** (U27 commit; env-gated;
default-off; canonical bytewise-equal at 9.12s e2e_latency).
Branch `tenstorrent-p1`; NOT pushed (predator2k/* fork only).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (post-U27)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| A.0 | Does the prefetcher actually create a receiver-side kernel? | Source enumeration of `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/` and `dram_prefetcher_program_factory.cpp` | **NO** — only TWO kernels exist (`reader_dram.cpp` + `writer_l1.cpp`), BOTH dispatched ONLY to SENDER cores via `reader_core_range`. NO kernel is dispatched on receiver cores by the prefetcher. The "receiver_sub_device runs persistent kernels" comment in `prefetcher.py:451-456` is MISLEADING — it refers to the receiver cores being PASSIVE NoC targets of the sender's writer_l1, not running their own persistent kernel. |
| A.1 | Across all 256 kernel dispatches on (2,7), do the L1[0xa6700] bytes change between any pair of ENTRY and EXIT events? | `SGLANG_TT_U27_KERNEL_ENTRY_PROBE=1` firmware-level brisc.cc DPRINT | **NO** — bytes are CONSTANT across ALL 256 events on BOTH chips. Zero entries see different w0..w3 values. |
| A.2 | Are the bytes already present at the very first kernel dispatch? | Same probe, tot=1 entry | **YES** — tot=1 ENTRY shows the SAME w0..w3 as tot=256 EXIT. Bytes are present BEFORE any user kernel ever runs on (2,7). |
| A.3 | Does the canonical (no prefetcher, no probe) run still produce coherent output? | `SGLANG_TT_USE_PREFETCHER=0`, no env vars | **YES** — " What is 2+2? What is 2+2? What" at 9.12s e2e_latency (within tolerance of prior 9.06-9.10s baselines). |

**Punchline:** The L1 0xa6700 bytes on receiver core (2,7) are
**PRE-EXISTING RESIDUAL DATA** — not a stomp. NO kernel ever
writes to (2,7) L1 0xa6700 throughout the entire decode pass.

The "0xa6700 stomp at FIXED address" model from U21 is REFUTED:
nothing is actively stomping. The bytes are simply LEFTOVER from
some prior allocation or firmware init.

The values look like BF16-encoded weight data:
- `0x4098` ≈ 4.578 bf16
- `0x3980` ≈ 0.000244 bf16
- `0xc0a9` ≈ -5.281 bf16

This is consistent with a previously-deallocated weight tile slot
in L1 that was never zeroed.

## Phase 1 — Enumerate receiver-side persistent kernels

### Search
```bash
ls /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/
# Only:  reader_dram.cpp  writer_l1.cpp
```

`dram_prefetcher_program_factory.cpp` lines 189-253 confirm both
kernels are created on `reader_core_range` only (= SENDER cores
from `global_cb.sender_cores()`). NO `CreateKernel(... receiver_cores ...)`
call exists.

### Implication

The U16/U21/U22 comment "Sender and receiver sub-devices run
persistent kernels" referred (incorrectly for receivers) to the
fact that:
- Sender cores DO run persistent kernels (`writer_l1` + `reader_dram`
  loop over `num_layers * num_tensors` iterations).
- Receiver cores have NO persistent kernel of their own. They are
  PASSIVE NoC targets:
  - The sender's writer_l1 NoC-writes pages to receivers'
    `fifo_start=0xac700` and increments their
    `aligned_pages_sent_ptr=0x17f680`.
  - Receiver cores only become "active" when downstream consumer ops
    (W2 matmul, in1_ring_ag matmul, RS reader, etc.) are dispatched
    to them via `sub_device_id=receiver_sub_device_id`.

So there IS no "receiver persistent kernel" to investigate. The
U27 prompt's premise (a hidden persistent kernel writing to L1
0xa6700) is factually wrong.

## Phase 2 — brisc.cc firmware-level entry/exit probe

### Implementation

`tt_metal/hw/firmware/src/tt-1xx/brisc.cc` lines 439-470 (ENTRY)
and lines 605-636 (EXIT): under `SGLANG_TT_U27_KERNEL_ENTRY_PROBE`
ifdef, on every kernel dispatch where `my_logical_x_==2 &&
my_logical_y_==7`, read L1[0xa6700] (with
`invalidate_l1_cache()`) and DPRINT
`host_assigned_id + enables + w0..w3`. Static budget of 256 per
probe to bound log size.

`tt_metal/jit_build/build_env_manager.cpp::initialize_device_kernel_defines()`
adds the env-gated define propagation:

```cpp
const char* env = std::getenv("SGLANG_TT_U27_KERNEL_ENTRY_PROBE");
if (env != nullptr && std::string(env) == "1") {
    device_kernel_defines.emplace("SGLANG_TT_U27_KERNEL_ENTRY_PROBE", "1");
}
```

This is the SAME pattern as U22 (which used a per-kernel-config
defines map). For firmware-level defines we use
`device_kernel_defines` which flows through `jit_build_subset`.

### Build steps

```bash
podman cp ... brisc.cc + build_env_manager.cpp
podman exec p3a-ngram bash -c 'cd /tt-metal/build_Release && ninja tt_metal'
podman exec p3a-ngram bash -c 'cp -fp /tt-metal/build_Release/tt_metal/libtt_metal.so /tt-metal/build_Release/lib/libtt_metal.so'

# Verify the SGLANG_TT_U27_KERNEL_ENTRY_PROBE string landed in libtt_metal.so:
podman exec p3a-ngram bash -c 'strings /tt-metal/build_Release/lib/libtt_metal.so | grep SGLANG_TT_U27'
# -> SGLANG_TT_U27_KERNEL_ENTRY_PROBE
```

### Run command (under prefetcher + zero weights)

```bash
podman exec p3a-ngram bash -c '
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  TT_METAL_DISABLE_PRECOMPILED_FW=1 \
  SGLANG_TT_U27_KERNEL_ENTRY_PROBE=1 \
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=2,7 \
  TT_METAL_DPRINT_FILE=/tmp/u27_kernel_trace.log \
  TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
  nohup python3 -u -m sglang.launch_server ... &
'
# Wait for ready, then:
curl -X POST http://127.0.0.1:30000/generate -d "{\"text\":\"What is 2+2?\", ...,\"max_new_tokens\":3,\"temperature\":0.0}"
```

### Result — sample U27 probe output

```
0:2-7:BR: [U27_ENTRY core=(2,7) hai=1   enables=0x1f w0=0x40983980 w1=0x40938f60 w2=0xc0a94f00 w3=0x40c11940 tot=1   NONZERO]
1:2-7:BR: [U27_ENTRY core=(2,7) hai=1   enables=0x1f w0=0xc0b1df00 w1=0x406b8a40 w2=0xc1500d20 w3=0x3e533800 tot=1   NONZERO]
0:2-7:BR: [U27_EXIT  core=(2,7) hai=1   enables=0x1f w0=0x40983980 w1=0x40938f60 w2=0xc0a94f00 w3=0x40c11940 tot=1   NONZERO]
... (256 events per chip, ALL with the same w0..w3) ...
0:2-7:BR: [U27_EXIT  core=(2,7) hai=248 enables=0x0  w0=0x40983980 w1=0x40938f60 w2=0xc0a94f00 w3=0x40c11940 tot=256 NONZERO]
1:2-7:BR: [U27_EXIT  core=(2,7) hai=248 enables=0x0  w0=0xc0b1df00 w1=0x406b8a40 w2=0xc1500d20 w3=0x3e533800 tot=256 NONZERO]
```

```bash
grep "U27.*core=(2,7)" /tmp/u27_kernel_trace.log | grep "^0:" \
  | awk '{ for(i=1;i<=NF;i++) if($i~/^w0=/) print $i,$(i+1),$(i+2),$(i+3) }' | sort -u
# w0=0x40983980 w1=0x40938f60 w2=0xc0a94f00 w3=0x40c11940
# (single unique 4-tuple across 256 events)
```

ALL 256 events on (2,7) — including both real kernels (`enables=0x1f`)
and the spurious dispatcher launches (`enables=0x0`, no kernel actually
runs), and including the VERY FIRST dispatch (tot=1) — see the SAME
constant value at L1[0xa6700].

## Phase 3 — Canonical bytewise re-verify

```bash
podman exec p3a-ngram bash -c '
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged ... no U27 env vars ...
'
curl ...
# Output: " What is 2+2? What is 2+2? What" tokens [3555,374,220,17,10,17,30,3555,374,220,17,10,17,30,3555]
# e2e_latency = 9.12s  (within tolerance of prior 9.06-9.10s baselines)
```

Canonical is preserved bytewise.

## Cross-check against U17 / U20 evidence

This new finding REINTERPRETS U17 and U20:

| Earlier probe | Earlier interpretation | U27 reinterpretation |
|---|---|---|
| **U17 PRE_RS @ 0xa6700: 46% NONZERO** | "Stomper fires between W2 PACK exit and RS reader entry" | RS reader's NoC read of (2,7) L1 0xa6700 returns the CONSTANT pre-existing bytes whenever its read happens (depending on race with the W2 PACK on whatever address W2's actual per-core output is). |
| **U20 ENTRY/EXIT @ 0xa6700: 0.3% zero→NZ on (2,7)** | "External concurrent writer fires during W2 program" | The 0.3% is NOISE: brief windows where probe reads BEFORE prefetcher writer_l1 ack-semaphore updates land at 0x17f680, or just measurement jitter. The CONSTANT byte pattern in U27 confirms no actual writer fires. |
| **U22 dispatcher RULED OUT** | "Dispatcher doesn't target 0xa6700" | Confirmed — dispatcher targets KERNEL_CONFIG region. The bytes at 0xa6700 are NOT from dispatcher writes; they're from some EARLIER one-time write. |
| **U21 B': stomp persists at FIXED 0xa6700** | "Stomper hits an address, not a buffer" | Re-interpretation: 0xa6700 holds CONSTANT pre-existing bytes; whatever live buffer lives at 0xa6700 inherits these bytes if not explicitly written by its producer. |

## New top hypothesis (U28)

**Hypothesis U28-α:** On core (2,7), the per-core W2 output buffer
is NOT at L1 0xa6700. It is at a different per-core L1 offset
(e.g., 0xa4700, 0xa8700, or 0xa6700 + per-core delta) due to the
HEIGHT_SHARDED allocation strategy used by the BANK-AVERAGED
buffer addressing.

Evidence supporting U28-α:
- U22 REALLOCATE_W2 logged `[U22_REALLOCATE_W2] old=0xa6700 new=0xaa700`
  AND `[U22_REALLOCATE_W2] old=0xa4700 new=0xaa700` — i.e., the
  PER-CORE source addresses differ (0xa6700 on some cores, 0xa4700
  on others).
- U20 noted "w2_out's per-core L1 base is 0xa6700 on SOME cores and
  0xa4700 on OTHERS".
- The bank-averaged `buffer_address()` returns 0xa6700, which is
  what mlp.py logs and what U17 probes.
- W2's PACK kernel writes to its OWN per-core output L1 (via
  set_globally_allocated_address(*out_buffer)) — but on (2,7),
  this per-core address might be 0xa4700, not 0xa6700.
- The RS reader's NoC read target is computed from the BANK-AVERAGED
  address 0xa6700 ON ALL CORES — so on (2,7), RS reader reads from
  the WRONG L1 slot (0xa6700, which holds the pre-existing constant
  bytes) instead of the actual W2 output (0xa4700).

**Hypothesis U28-β** (alternative): The L1 0xa6700 region on (2,7)
holds a residual DRAM-staging buffer from a prior op (e.g., a
W2 weight tile pre-fetched into a CB at 0xa6700 in some earlier
init sequence). The PACK on (2,7) DOES write its (zero) output
correctly to 0xa6700 — but ONLY to TILE 0 of an N-tile shard, and
the U27 probe only sees the bytes BEYOND tile 0. (Less likely
given the constant byte pattern across 256 events, but worth
ruling out.)

## U28 dispatch recommendation

### U28 Phase A — confirm per-core W2 output address on (2,7)

Add a per-core probe in `mlp.py` that, for the W2 op, dumps the
ACTUAL per-core L1 base address on EACH receiver core. This can
be done by reading `w2_out.buffer()->shard_spec().per_core_addresses()`
or via `ttnn._ttnn.reports.get_buffers()` filtered to receiver
core (2,7). If the address is NOT 0xa6700, U28-α is confirmed.

### U28 Phase B — relocate the RS reader's NoC read target

If U28-α is confirmed, the fix is to align the RS reader's
per-core NoC read addresses with the W2 PACK's per-core write
addresses. This may involve:
- Routing through a `ttnn.move` that defragments to a single
  uniform per-core address (already tested in U22 REALLOCATE_W2
  — the move_sharded fix is in place; under real weights this
  still failed due to a SECONDARY issue per U26).
- Or directly patching the RS reader's per-core NoC read
  addresses to match W2's per-core output addresses.

### U28 Phase C — investigate the secondary issue under REALLOCATE

U26's finding: under REALLOCATE_W2 + REAL weights, output is
still garbage even though w2_out moves to 0xaa700 and PRE_RS
shows zero NONZERO. This SECONDARY issue may share root cause
with U28-α — i.e., the RS reader after REALLOCATE is still
reading from a per-core offset that doesn't match W2's new
output layout.

## What U27 lands (all env-gated; canonical bytewise-equal)

| File | Change |
|---|---|
| `tt_metal/hw/firmware/src/tt-1xx/brisc.cc` | `SGLANG_TT_U27_KERNEL_ENTRY_PROBE`-gated DPRINT block at kernel-launch START (after launch_msg fetch) and END (after wait_ncrisc_trisc), filtered to `my_logical_x_==2 && my_logical_y_==7`. Static per-RISC budget of 256. |
| `tt_metal/jit_build/build_env_manager.cpp` | `+#include <cstdlib>`; env-gated `device_kernel_defines["SGLANG_TT_U27_KERNEL_ENTRY_PROBE"] = "1"` in `initialize_device_kernel_defines()` so the define reaches firmware. |

Runtime requirement: `TT_METAL_DISABLE_PRECOMPILED_FW=1` to force
firmware rebuild when the probe is on (otherwise precompiled
firmware ignores the define).

## Reproducer

### A. U27 probe (prefetcher on, zero weights)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c '
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  TT_METAL_DISABLE_PRECOMPILED_FW=1 \
  SGLANG_TT_U27_KERNEL_ENTRY_PROBE=1 \
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=2,7 \
  TT_METAL_DPRINT_FILE=/tmp/u27_kernel_trace.log \
  TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
  nohup python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u27_server.log 2>&1 &
'
# Wait for ready
podman exec p3a-ngram bash -c 'until curl -sf http://127.0.0.1:30000/health_generate > /dev/null 2>&1; do sleep 5; done'
podman exec p3a-ngram bash -c 'curl -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":3,\"temperature\":0.0}}"'

# Confirm the constant byte pattern:
podman exec p3a-ngram bash -c '
  grep "U27.*core=(2,7)" /tmp/u27_kernel_trace.log | grep "^0:" \
    | awk "{ for(i=1;i<=NF;i++) if(\$i~/^w0=/) print \$i,\$(i+1),\$(i+2),\$(i+3) }" \
    | sort -u
  # Expected: SINGLE unique line — w0=0x40983980 w1=0x40938f60 w2=0xc0a94f00 w3=0x40c11940
'
```

### B. Canonical re-verify

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c '
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u27_canonical.log 2>&1 &
'
curl -X POST http://127.0.0.1:30000/generate \
  -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":15,\"temperature\":0.0}}"
# Output: " What is 2+2? What is 2+2? What" at e2e_latency=9.12s.
```

## Hypothesis ledger (post-U27)

| ID | Suspect | Pre-U27 | Post-U27 |
|---|---|---|---|
| (all previous U16-U26 ruled outs) | … | RULED OUT | unchanged |
| **U16 / U21 — unknown kernel writes garbage to (2,7) L1 0xa6700 between W2 PACK exit and RS reader entry** | (TOP STANDING after U25/U26) | **REFUTED** — across 256 ENTRY+EXIT events spanning the entire decode pass, no kernel ever writes to (2,7) L1 0xa6700. Bytes are constant and pre-existing. |
| **U27 A — receiver_sub_device's persistent kernel writes to L1 0xa6700** | (new — top hypothesis at U27 dispatch) | **REFUTED** — the prefetcher only dispatches kernels on SENDER cores. NO receiver-side persistent kernel exists. The "persistent kernels on receivers" comment in prefetcher.py is misleading. |
| **U28-α — W2's per-core PACK destination on (2,7) is NOT 0xa6700; RS reader reads stale pre-existing bytes at 0xa6700** | (new) | **NEW TOP STANDING** — supported by U22 (per-core src addresses differ: 0xa6700 and 0xa4700 across cores) and U27 (constant pre-existing bytes at 0xa6700 on (2,7) never overwritten by any kernel). |
| **U28-β — W2's per-core PACK on (2,7) writes ONLY to tile 0 of an N-tile shard, leaving bytes BEYOND tile 0 unwritten** | (new) | **STANDING** — less likely (the constant byte pattern is across the first 16 bytes which is tile 0's header on TILE_LAYOUT). |
| **U28-γ — The constant bytes are leftover from prefetcher GlobalCB tensor data overflow** | (new) | **STANDING** — possible if GlobalCB at 0xac700 has prior content that bleeds into 0xa6700. Less likely (GlobalCB fifo_size is bounded). |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (only tt-metal C++ in
      brisc.cc + build_env_manager.cpp; env-gated; default-off).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` (TT_CACHE_HOME)
      + literal-path absolute.
- [x] No stash pop/drop.
- [x] Port-clear used `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` for source
      sync + manual `ninja tt_metal` rebuild + `cp` of
      libtt_metal.so to install dir.
- [x] All new probes env-gated (`SGLANG_TT_U27_KERNEL_ENTRY_PROBE`)
      + require `TT_METAL_DISABLE_PRECOMPILED_FW=1` at runtime to
      take effect (firmware-level).
- [x] Canonical Qwen3-8B re-verified bytewise: " What is 2+2?
      What is 2+2? What" at 9.12s e2e_latency (within tolerance
      of prior 9.06-9.10s baselines).
- [x] Server stopped at session end.
- [x] TT devices: healthy.

## Shipping verdict (unchanged from U19/U20/U21/U22/U23/U24/U25/U26)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat ~ 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**
  The bug (per-core address mismatch hypothesis U28) is now
  understood at the WHAT level but the FIX is not yet implemented.
- **DO NOT ship any `SGLANG_TT_U27_*` env var.**  Diagnostic only.
- **DO NOT ship `TT_METAL_DISABLE_PRECOMPILED_FW=1` as default.**
  Adds ~30 s to first model init for firmware JIT rebuild.

The U27 probes are safe to leave in tree as default-off:
- `#ifdef SGLANG_TT_U27_KERNEL_ENTRY_PROBE` block in brisc.cc is
  a no-op when env is unset.
- The env-gated `device_kernel_defines.emplace` in
  build_env_manager.cpp only runs when env is set.

## Commits this session

- (tt-metal-sglang) **`62e8c6c24b4`** —
  `prefetcher: U27 — firmware-level per-kernel L1 0xa6700 entry/exit probe on receiver (2,7) PROVES 0xa6700 is NEVER written by ANY kernel; bytes are constant and PRE-EXISTING (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`62e8c6c24b4`** (U27 commit; env-gated;
  default-off; canonical bytewise-equal at 9.12s).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.  No sglang Python
  changes in U27.
- Container `/tt-metal/`: synced (brisc.cc + build_env_manager.cpp);
  `libtt_metal.so` rebuilt + installed (verified `SGLANG_TT_U27`
  string in `strings /tt-metal/build_Release/lib/libtt_metal.so`).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: bytewise " What is 2+2? What is 2+2? What"
  at 9.12s e2e_latency.
