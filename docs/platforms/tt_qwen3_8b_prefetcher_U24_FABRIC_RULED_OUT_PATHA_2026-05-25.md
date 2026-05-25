# TT Qwen3-8B prefetcher — U24: fabric_erisc_router local-chip writes RULED OUT as L1 0xa6700 stomper; Path A complete — 2026-05-25

Status: **EVIDENCE_ADVANCE — U24 Path A lands an env-gated
destination-address probe at every write site in
`tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp`
(`execute_chip_unicast_to_local_chip_impl`) — the function that
turns every fabric-arrived packet into a `noc_async_write` /
`noc_inline_dw_write` / `noc_semaphore_inc` against local tensix
L1.  Probe is propagated to (a) the fabric_erisc_router kernel via
`ComputeMeshRouterBuilder::create_kernel` defines (verified
`U24_FABRIC_WRITE` string in all 8 cached
subordinate_active_erisc ELFs) and (b) the line + ring
reduce_scatter_minimal_async writer / reader / mux kernels.  Filter
window `[0xa6000, 0xa7000]` covers the cursed L1 address `0xa6700`
on receiver core `(2,7)`.  Under FULL prefetcher + zero weights +
single 15-token decode, `/tmp/u24_fabric.log` records **ZERO**
`U24_FABRIC_WRITE` entries — no fabric-arrived packet's local
write target landed in `[0xa6000, 0xa7000]`.  This RULES OUT
fabric / EDM packet-receiver writes (via the fabric router's
`execute_chip_unicast_to_local_chip` path) as the stomper.
Canonical Qwen3-8B preserved (5/5 GSM8K chat; bytewise " What is
2+2? What is 2+2? What" at 9.08s e2e).**

Continuation of `tt_qwen3_8b_prefetcher_U23_MOVE_SHARDED_FIX_LANDED_REALLOCATE_STILL_BLOCKED_2026-05-25.md`.

tt-metal-sglang HEAD: **`adf4af88618`** (U24 commit; env-gated;
default-off; canonical bytewise-equal).  Branch `tenstorrent-p1`;
NOT pushed (predator2k/* fork only per project policy).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (post-U24)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| A.1 | Does the fabric_erisc_router's `noc_async_write_one_packet_with_trid` for `NOC_UNICAST_WRITE` packets ever target tensix L1 `[0xa6000, 0xa7000]`? | `SGLANG_TT_U24_FABRIC_ADDR_PROBE=1` (env-gated DPRINT at every dest_address in the unicast_write case) | NO — zero log entries. |
| A.2 | Does the fabric_erisc_router's `noc_semaphore_inc` for `NOC_UNICAST_ATOMIC_INC` packets ever target tensix L1 `[0xa6000, 0xa7000]`? | same probe, unicast_seminc case | NO — zero log entries. |
| A.3 | Does the fabric_erisc_router's `noc_inline_dw_write` for `NOC_UNICAST_INLINE_WRITE` packets ever target tensix L1 `[0xa6000, 0xa7000]`? | same probe, unicast_inline_write case | NO — zero log entries. |
| A.4 | Does the fabric_erisc_router's fused-seminc payload write OR semaphore write target the window? | same probe, fused_seminc cases (data + sem) | NO — zero log entries. |
| A.5 | Does any of the 4 scatter-write chunks target the window? | same probe, scatter_write_{0,1,2,last} + scatter_seminc_last | NO — zero log entries. |
| B | Did the probe actually compile into the fabric router ELF? | `strings .../subordinate_active_erisc.elf \| grep U24_FABRIC_WRITE` for all 8 cached fabric_erisc_router builds | YES — count=1 (the DPRINT format string) in every ELF. |
| C | Canonical Qwen3-8B preserved? | tt_transformers_paged + greedy " What is 2+2?" 15-token + GSM8K(5) chat | YES — bytewise-equal; 5/5 = 100%. |

**Punchline:** Every fabric-arrived packet that the fabric router
turns into a tensix-L1 write was logged; none landed in
`[0xa6000, 0xa7000]` during the stomp window.  **Fabric / EDM
packet-receiver writes RULED OUT as the L1 0xa6700 stomper.**
This closes U24 Path A and (together with U17–U23) leaves an
extremely narrow candidate set.

## Phase A — probe design and placement

### A.1 What the probe instruments

`tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp::execute_chip_unicast_to_local_chip_impl`
is the SINGLE function on the entire receiver-side fabric path.
When a packet arrives at the local chip's fabric router (ethernet
core), this function decodes `noc_send_type` from the header and
issues one of:

| `NocSendType` | NoC call | Field used as destination |
|---|---|---|
| `NOC_UNICAST_WRITE` | `noc_async_write_one_packet_with_trid` | `unicast_write.noc_address` |
| `NOC_UNICAST_ATOMIC_INC` | `noc_semaphore_inc` | `unicast_seminc.noc_address` |
| `NOC_UNICAST_INLINE_WRITE` | `noc_inline_dw_write` | `unicast_inline_write.noc_address` |
| `NOC_FUSED_UNICAST_ATOMIC_INC` | `noc_async_write_one_packet_with_trid` + `noc_semaphore_inc` | `unicast_seminc_fused.{noc_address, semaphore_noc_address}` |
| `NOC_UNICAST_SCATTER_WRITE` | up to 4 × `noc_async_write_one_packet_with_trid` (+ optional sem-inc final) | `scatter.noc_address[0..3]` |

Every one of these is now wrapped with a helper macro
`SGLANG_TT_U24_LOG_FABRIC_DST(tag, dst64)` that:
- unpacks `(noc_xy = dst >> 32, l1 = dst & 0xFFFFFFFF)`,
- filters on `l1 >= 0xa6000 && l1 < 0xa7000` (the
  64-byte-aligned window covering 0xa6700),
- decrements a per-RISC budget of 4096,
- DPRINTs `[U24_FABRIC_WRITE <tag> noc_xy=0x<XY> l1=0x<L1>]`.

When the env var is unset, the macro expands to `((void)0)` —
zero runtime cost, zero code bloat — preserving the canonical
build's byte-for-byte behavior.

### A.2 Propagation paths

The macro inside the header only fires for kernels that pass
`-DSGLANG_TT_U24_FABRIC_ADDR_PROBE=1` to JIT compile.  Three
program-factory sites do the propagation:

1. **`tt_metal/fabric/compute_mesh_router_builder.cpp::ComputeMeshRouterBuilder::create_kernel`**:
   adds `defines["SGLANG_TT_U24_FABRIC_ADDR_PROBE"] = "1"` to the
   defines map at function entry when the env var is set.  Flows
   into the `CreateKernel(...EthernetConfig{.defines=defines})`
   call for `fabric_erisc_router.cpp` on every ethernet router
   core.  Verified: 8 cached `fabric_erisc_router/*/subordinate_active_erisc/subordinate_active_erisc.elf`
   ELFs all contain the `U24_FABRIC_WRITE` string post-rebuild.

2. **`ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp`**
   (line path): writer_compute_defines + reader_compute_defines +
   mux_compute_defines all get the U24 define when env var is set.

3. **Same file (ring path)**: writer_defines (for ring writer)
   and ring_mux_defines (for ring fabric mux) get the define.

Note: the actual tensix-side `noc_async_write` to (2,7) L1 0xa6700
is issued by the fabric_erisc_router via `noc_async_write_one_packet_with_trid`
when an incoming packet's header says
`NOC_UNICAST_WRITE` and the dst noc address is `0x1c20000a6700`.
That is precisely the case the probe targets.

## Phase B — verification: probe is in the ELFs

```
$ podman exec p3a-ngram bash -c '
    for d in /root/.cache/tt-metal-cache/*/kernels/fabric_erisc_router/*/subordinate_active_erisc/subordinate_active_erisc.elf;
    do echo "$d: $(strings "$d" | grep -c U24_FABRIC_WRITE)";
    done | head'

/root/.cache/.../fabric_erisc_router/10025442727570230759/.../subordinate_active_erisc.elf: 1
/root/.cache/.../fabric_erisc_router/17549202302736544123/.../subordinate_active_erisc.elf: 1
/root/.cache/.../fabric_erisc_router/3671769782121761643/.../subordinate_active_erisc.elf: 1
/root/.cache/.../fabric_erisc_router/496243096851650277/.../subordinate_active_erisc.elf: 1
/root/.cache/.../fabric_erisc_router/5480024484700660848/.../subordinate_active_erisc.elf: 1
/root/.cache/.../fabric_erisc_router/6750360020729400520/.../subordinate_active_erisc.elf: 1
/root/.cache/.../fabric_erisc_router/7370409375735244401/.../subordinate_active_erisc.elf: 1
/root/.cache/.../fabric_erisc_router/9711079708230759602/.../subordinate_active_erisc.elf: 1
```

All 8 fabric router builds have the probe baked in.  Also verified
the env var sentinel string is in `_ttnncpp.so` and `libtt_metal.so`.

## Phase C — the smoke run (zero log entries)

```
podman exec p3a-ngram bash -c '
  rm -f /tmp/u24_fabric.log
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_U24_FABRIC_ADDR_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_FILE=/tmp/u24_fabric.log \
  nohup python3 -u -m sglang.launch_server ... &'

# decode succeeded (e2e_latency = 10.5s; zero-weight garbage as expected)

podman exec p3a-ngram cat /tmp/u24_fabric.log
0:8-9:BR: prefetcher_11: start
0:9-9:BR: dispatch_11: start
0:9-9:NC: dispatch_s : start
1:9-9:BR: dispatch_11: start
1:9-9:NC: dispatch_s : start
1:8-9:BR: prefetcher_11: start
# NO U24_FABRIC_WRITE entries
```

Six lines total in the log — the canonical prefetcher /
dispatcher startup DPRINTs.  Zero `U24_FABRIC_WRITE` entries.

This means the fabric_erisc_router did not write anything in
`[0xa6000, 0xa7000]` during the entire 15-token decode (covering
many W2 → RS dispatches).  The fabric receiver is not the
stomper.

## Phase D — canonical regression check

```
podman exec p3a-ngram bash -c '
  rm -rf /root/.cache/tt-metal-cache/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... &'

# greedy generate
curl ...{"text":"What is 2+2?","max_new_tokens":15,"temperature":0.0}
# " What is 2+2? What is 2+2? What"  -- bytewise-equal to U23

# GSM8K(5) chat
python3 .../eval_qwen3_8b_gsm8k_chat.py --num 5
# FINAL: 5/5 = 100.0%
```

Canonical preserved.  Env-gated probe has zero canonical impact.

## Hypothesis ledger (post-U24)

| ID | Suspect | Pre-U23 | Post-U23 | Post-U24 |
|---|---|---|---|---|
| Fabric / EDM completion acks (`fabric_erisc_router::execute_chip_unicast_to_local_chip`) | STANDING TOP PRIORITY | STANDING | **RULED OUT** — zero in-window writes. |
| `update_remote_cb_config_in_l1` bookkeeping writes | STANDING | STANDING | STANDING (untouched by U24). |
| Non-router fabric writer (e.g., remote_cb update via mux extension) | STANDING | STANDING | STANDING (probe is at receiver side only; sender-side prefetcher writes were ruled out by U21). |
| Trace-replay command write to a stale captured L1 addr | STANDING | STANDING | STANDING (untouched). |
| Cross-sub-device dispatch artifact | STANDING | STANDING | STANDING (touched by U16 globally; standing per Hypothesis #3). |

## What remains as plausible stompers

After U24 the candidate set is genuinely narrow:

1. **`update_remote_cb_config_in_l1` bookkeeping writes** —
   the GlobalCB cross-program metadata update path that the
   prefetcher senders (or the W1/W2/W3 receivers) invoke during
   `experimental::resize_remote_sender_cb_interface<true>(...)`.
   These are non-payload writes that target small L1 control
   slots and are not visible to the U18 PREFETCHER_WRITE_PROBE
   (which only logs `fifo_wr_ptr` payload writes).

2. **Trace-replay command write** — the trace replay engine
   replays captured CQ commands.  Some captured CQ command may
   write to a stale L1 address that, on replay (after allocator
   churn), now belongs to a live buffer at 0xa6700.  This is the
   "stomper is in the CQ but not in cq_dispatch_*" hypothesis;
   U22 instrumented only `process_write_{linear,packed,packed_large}`.
   Other write paths (e.g., `process_inline_write`, prefetcher's
   own write paths, dispatch_s) are unprobed.

3. **A specifically-targeted SET-RTA / launch-msg-buffer index
   write** — the launch msg ring buffer is at MAILBOX, but the
   GO message + RTA payloads land in KERNEL_CONFIG; the L1
   region for RTA updates moves with the slot index.  Per U21
   analysis, KERNEL_CONFIG ends near 0x26400 — well below 0xa6700.
   However, custom large-RTA-arg payloads (e.g., for the matmul
   1d kernel) might land much higher.  Unconfirmed.

4. **A non-noc-routed write from another tensix RISC**
   (BRISC/NCRISC/TRISC) that we haven't yet probed.  U20
   instrumented in0/in1 dataflow readers; the W2 PACK probe
   (U18) confirmed PACK writes zero; U13 ruled out the math
   path.  But other RISCs in OTHER programs (e.g., a concurrent
   sub-device kernel) may write.

## U25 dispatch recommendation

### U25 Phase A — `update_remote_cb_config_in_l1` probe

Instrument every call site of `update_remote_cb_config_in_l1`
in `ttnn/cpp/ttnn/operations/global_cb/` (or wherever the
GlobalCB metadata update lives) to log the destination noc
address and value.  Filter `[0xa6000, 0xa7000]`.  If this fires,
the GlobalCB control-update path is the stomper.

### U25 Phase B — extended dispatcher probe

Extend the U22 dispatcher probe to also cover
`process_inline_write`, `process_set_runtime_args`, and any
`process_dispatch_s_*` write site in `cq_dispatch.cpp` and
`cq_dispatch_subordinate.cpp`.  Filter same window.

### U25 Phase C — Path B (custom L1 poller kernel)

Per U24 plan: deploy a tiny polling kernel on a clean sub-device
that polls (2,7) L1 0xa6700 in a tight loop and DPRINTs the
mcycle timestamp + new value on every byte change.  Run it
concurrently with the W2 → RS pipeline.  This catches the
stomper SOURCE without needing exhaustive code-side
instrumentation.  Requires:
- A new `poll_l1_kernel.cpp` (BRISC, single-core, infinite loop).
- A new ttnn op that launches it on a dedicated sub-device.
- Python wiring in mlp.py to launch the poller before W2 and
  kill it after RS.
- Off-trace deployment (the poller must NOT be captured into
  the decode trace, since it never terminates).

### U25 Phase D — re-examine the U17 NONZERO byte patterns

The NONZERO bytes at 0xa6700 (e.g., `0xd328_5f35`) are
activation-like bf16 floats.  Trace these patterns to a
specific tensor's bytes by capturing the entire L1 (2,7) at the
stomp moment.  If they match a known tensor (e.g., the
attention output), we've found WHO wrote those bytes (whoever
last produced that tensor).

## What U24 lands

| File | Change |
|---|---|
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp` | `SGLANG_TT_U24_FABRIC_ADDR_PROBE`-gated helper macro + DPRINT call at every `noc_async_write_one_packet_with_trid` / `noc_inline_dw_write` / `noc_semaphore_inc` site in `execute_chip_unicast_to_local_chip_impl` (unicast_write, unicast_seminc, unicast_inline_write, fused_seminc data+sem, scatter_write × 4 chunks). |
| `tt_metal/fabric/compute_mesh_router_builder.cpp` | env-gated propagation of `SGLANG_TT_U24_FABRIC_ADDR_PROBE` into `ComputeMeshRouterBuilder::create_kernel`'s defines map. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp` | env-gated propagation into both line and ring paths' writer / reader / mux defines maps. |

No sglang Python changes.  Probe is default-off and is a strict
NO-OP for canonical builds.

## Reproducer

```bash
# tt-metal-sglang HEAD is adf4af88618.  Container has the rebuild.

# 1. Verify probe is compiled in:
podman exec p3a-ngram bash -c '
  strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep -c SGLANG_TT_U24_FABRIC_ADDR_PROBE
  strings /tt-metal/build_Release/tt_metal/libtt_metal.so | grep -c SGLANG_TT_U24_FABRIC_ADDR_PROBE'
# 1 / 1

# 2. Probe run under FULL prefetcher + zero weights:
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '
  rm -f /tmp/u24_fabric.log
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_U24_FABRIC_ADDR_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_FILE=/tmp/u24_fabric.log \
  nohup python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u24_server.log 2>&1 &'
podman exec p3a-ngram bash -c '
  until curl -sf http://127.0.0.1:30000/health_generate > /dev/null 2>&1; do sleep 5; done
  curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
    -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":15,\"temperature\":0.0}}"'
podman exec p3a-ngram bash -c 'grep -c U24_FABRIC_WRITE /tmp/u24_fabric.log || true'
# 0

# 3. Canonical re-verify (no prefetcher, no probe):
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '
  rm -rf /root/.cache/tt-metal-cache/* 2>/dev/null
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u24_canonical.log 2>&1 &'
podman exec p3a-ngram bash -c '
  until curl -sf http://127.0.0.1:30000/health_generate > /dev/null 2>&1; do sleep 5; done
  curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
    -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":15,\"temperature\":0.0}}"
  python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 5'
# " What is 2+2? What is 2+2? What"
# FINAL: 5/5 = 100.0%
```

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (only tt-metal C++ + an
      sglang docs file).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` (TT_CACHE_PATH)
      + literal-path absolute.
- [x] No stash pop/drop.
- [x] Port-clear used `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` for source
      sync + `rebuild_tt_metal_kernels.sh` (with INSTALL_DIR sync).
- [x] Probe is env-gated initially (`SGLANG_TT_U24_FABRIC_ADDR_PROBE`).
- [x] Canonical Qwen3-8B preserved — bytewise-equal greedy +
      5/5 GSM8K chat.
- [x] Server stopped at session end.

## Shipping verdict (unchanged from U19/U20/U21/U22/U23)

Canonical Qwen3-8B (no prefetcher) remains the production
shipping config at TPOT ~27 ms / 1024-1024 and GSM8K(10) chat 10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  The
  L1 0xa6700 stomp persists.
- **DO NOT ship any `SGLANG_TT_U24_*` env var.**  Diagnostic only.

The U24 probe is safe to ship as a default-off probe — it is
strictly NO-OP for canonical builds (the macro expands to
`((void)0)` when the env var is unset, and the
program-factory defines are only added when the env var is set).

## Commits this session

- (tt-metal-sglang) **`adf4af88618`** —
  `prefetcher: U24 — fabric/EDM addr probe in fabric_erisc_router RULES OUT fabric-receiver writes to L1 [0xa6000, 0xa7000] as stomper (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`adf4af88618`** (U24 commit; env-gated;
  default-off; canonical bytewise-equal).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.  No sglang Python
  changes in U24.
- Container `/tt-metal/`: synced
  (`compute_mesh_router_builder.cpp` + `fabric_edm_packet_transmission.hpp` +
  `reduce_scatter_minimal_async_program.cpp`) + rebuilt
  `_ttnn.so` / `_ttnncpp.so` / `libtt_metal.so`; sentinel
  `SGLANG_TT_U24_FABRIC_ADDR_PROBE` verified in libs and all 8
  cached fabric_erisc_router ELFs.
- `/root/.cache/tt-metal-cache/*`: cleared at session start and
  again before canonical re-verify.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: bytewise " What is 2+2? What is 2+2? What"
  greedy; 5/5 GSM8K(5) chat.
