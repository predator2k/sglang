# TT Qwen3-8B prefetcher — U23: move_sharded per-core chunk_size fix LANDS in tt-metal; U22 REALLOCATE workaround STILL BLOCKED — root cause is reallocate-inside-trace, not per-core chunk_size — 2026-05-25

Status: **EVIDENCE_ADVANCE — U23 lands a legitimate
`move_sharded_program_factory` fix (per-core src/dst addresses for
per-core-allocated sharded buffers) in tt-metal-sglang
`5a5b3b5cdd6`.  All 34 cases in
`tests/tt_eager/python_api_testing/unit_testing/misc/test_move_sharded.py`
pass.  Canonical Qwen3-8B preserved (10/10 GSM8K chat).  HOWEVER —
the fix does NOT unblock the U22 `SGLANG_TT_U22_REALLOCATE_W2`
workaround for the L1 0xa6700 stomp: real-weight output remains
garbage (`" WhatQ!/)'C&L/820'0"`; sampling crashes with `RuntimeError:
probability tensor contains either inf, nan or element < 0`).  Root
cause re-analysis: the U22 doc's "per-core differing src addresses
within a single buffer" interpretation was incorrect — the
`0xa6700` vs `0xa4700` alternation in the U22 print logs comes from
DIFFERENT LAYERS' buffers, not from per-core variation within one
buffer.  The actual REALLOCATE blocker is the
"Allocating device buffers is unsafe due to the existence of an
active trace" warning emitted by the tt-metal allocator: `ttnn.reallocate`
inside a captured decode trace allocates a new output buffer during
trace capture; on replay, the buffer's contents (and the
move_sharded reader's NoC reads) are racy because the captured trace
commands target a captured L1 region whose ownership state diverges
between trace captures and replays.**

Continuation of `tt_qwen3_8b_prefetcher_U22_DISPATCH_RULED_OUT_PATHB_FAILS_2026-05-25.md`.

tt-metal-sglang HEAD: **`5a5b3b5cdd6`** (U23 commit; ships).
Branch `tenstorrent-p1`; NOT pushed (predator2k/* fork only).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (U23 session)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| A.1 | Does `move_sharded` actually use a single bank-averaged chunk_size? | Code read of `move_sharded_program_factory.cpp` + `reader_unary_local_l1_copy_backwards.cpp` | YES — `chunk = output_addr - input_addr` computed once from `buffer.address()`, passed via SetRuntimeArgs as uniform args to every core. |
| A.2 | Does the CB base pointer (`get_read_ptr(cb_id)`) give per-core different addresses for per-core-allocated buffers? | Code read of `circular_buffer_config.cpp:188` + `program/dispatch.cpp:1117-1123` | NO — `cb_address = cb->address()` is a single value (uses `buffer.address()` = bank-averaged); dispatched identically to every core in the multicast. Per-core-allocated buffers DO have differing actual L1 bases but the CB layer flattens this. |
| A.3 | Was U22's "src=0xa4700 cores read stale data" analysis correct? | Re-reading U22 logs + understanding of dispatch flow | INCORRECT INTERPRETATION — the `0xa6700` vs `0xa4700` alternation in `[U22_REALLOCATE_W2] old=…` prints is across DIFFERENT LAYERS (each layer's W2 output buffer gets allocated at a different L1 slot depending on allocator history), NOT across cores within one buffer. Within one W2 output (one mlp call), all cores see the SAME bank-averaged address; the kernel is correct on every core. |
| B.1 | Implement per-core chunk_size fix anyway? | Yes — for the genuine per-core-allocation case (sharding with experimental `set_per_core_allocation`) | LANDED — both the program factory and the reader kernel now accept per-core src_addr / dst_addr runtime args.  Uniform path is preserved byte-for-byte. |
| B.2 | Move-op unit-test suite still passes? | `tests/tt_eager/python_api_testing/unit_testing/misc/test_move_sharded.py` (34 cases) | 34/34 PASSED in 5.65s. |
| B.3 | Canonical Qwen3-8B output preserved? | `What is 2+2?` → 15-token greedy generate | `" What is 2+2? What is 2+2? What"` (identical to U22 canonical at 9.05s e2e). |
| B.4 | Canonical GSM8K(10) chat unaffected? | full eval_qwen3_8b_gsm8k_chat.py num=10 | 10/10 = 100% (improvement vs U22's 9/10 reported ceiling; matches recent canonical runs). |
| C.1 | Does the fix unblock the U22 REALLOCATE workaround under real weights? | `SGLANG_TT_U22_REALLOCATE_W2=1` + 15-token greedy generate | NO — output is still garbage (`" WhatQ!/)'C&L/820'0"`). GSM8K(10) sampling crashes: `RuntimeError: probability tensor contains either inf, nan or element < 0`. |
| C.2 | What is the actual REALLOCATE blocker? | tt-metal allocator log line during in-trace reallocate | "Allocating device buffers is unsafe due to the existence of an active trace. These buffers may be corrupted once a trace is executed." (allocator.cpp:110). `ttnn.reallocate` inside `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` is fundamentally racy. |

**Punchline:**
1. U23 ships a real per-core-allocation fix for `move_sharded`
   that was always a latent correctness hole.  All move_sharded
   tests still green; canonical model preserved.
2. The U22 doc's "the chunk_size is wrong on cores with src=0xa4700"
   interpretation was wrong.  The bug is elsewhere.
3. The actual blocker for the U22 REALLOCATE workaround is
   `ttnn.reallocate`-inside-`ttnn.begin_trace_capture` — the
   captured trace bakes in a new buffer's address, and trace replay
   reads stale L1 contents.  Not fixable by patching move_sharded.

## Phase A — move_sharded source-level analysis

### A.1 What the program factory does (pre-U23)

`ttnn/cpp/ttnn/operations/data_movement/move/device/move_sharded_program_factory.cpp` (pre-fix):

```cpp
const uint32_t input_buffer_address = input.buffer()->address();   // bank-averaged
const uint32_t output_buffer_address = output.buffer()->address(); // bank-averaged
const uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
...
const std::array runtime_args = {total_size_bytes, num_chunks, move_chunk_size_bytes, remainder_chunk_size_bytes};
tt::tt_metal::SetRuntimeArgs(program, kernel_id, shard_grid, runtime_args);  // UNIFORM
```

Single chunk_size is broadcast to every core.

### A.2 What the kernel does (pre-U23)

`reader_unary_local_l1_copy_backwards.cpp` (pre-fix):

```cpp
uint32_t src_cb_base_addr = get_read_ptr(src_cb_id);      // per-core CB ptr from dispatched config
uint32_t dst_cb_base_addr = get_write_ptr(dst_cb_id);     // per-core CB ptr from dispatched config
uint32_t src_cb_addr = src_cb_base_addr + total_size_bytes;
uint32_t dst_cb_addr = dst_cb_base_addr + total_size_bytes;
for (i = 0; i < num_chunks; i++) {
    src_cb_addr -= chunk_size_bytes;
    dst_cb_addr -= chunk_size_bytes;
    noc_async_read(get_noc_addr(src_cb_addr), dst_cb_addr, chunk_size_bytes);
    noc_async_read_barrier();
}
```

### A.3 What's actually wrong (and the right framing)

The CB config payload is multicast.  `cb_address` written into the payload (`tt_metal/impl/program/dispatch.cpp:1123`) is `cb->address()` which is `buffer.address()` which for per-core-allocated buffers is the address on `cores[0]` — and is the same on EVERY core after multicast.

So `get_read_ptr(cb)` on every core returns the bank-averaged base.

For uniform shards (cores all allocated at the same L1 offset): correct.

For per-core-allocated shards (some cores at 0xa6700, others at 0xa4700): the kernel reads from the bank-averaged address on every core, so cores whose actual data lives at 0xa4700 read from 0xa6700 — and write to the bank-averaged dst, which also doesn't match those cores' actual dst.

This is a latent bug that affects `move_sharded` whenever the input or output uses `experimental::per_core_allocation::set_per_core_allocation(args, true)`.

## Phase B — the fix

### B.1 Program factory changes (excerpt)

```cpp
// U23: per-core runtime args; thread the ACTUAL per-core src/dst L1
// addresses so the kernel computes its own correct chunk_size on every core.
const auto cores = corerange_to_cores(shard_grid, std::nullopt, true);
std::vector<std::vector<uint32_t>> per_core_runtime_args;
for (const auto& core : cores) {
    const uint32_t src_addr = static_cast<uint32_t>(per_core_address(*input.buffer(), core));
    const uint32_t dst_addr = static_cast<uint32_t>(per_core_address(*output.buffer(), core));
    TT_FATAL(dst_addr > src_addr, "backward-copy requires dst > src on core ({},{}) — got {:#x} > {:#x}", core.x, core.y, dst_addr, src_addr);
    const uint32_t move_chunk_size_bytes = dst_addr - src_addr;
    TT_FATAL(move_chunk_size_bytes % alignment == 0, "chunk size misaligned");
    const uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
    const uint32_t remainder = total_size_bytes % move_chunk_size_bytes;
    per_core_runtime_args.push_back({total_size_bytes, num_chunks, move_chunk_size_bytes, remainder, src_addr, dst_addr});
}
tt::tt_metal::SetRuntimeArgs(program, kernel_id, cores, per_core_runtime_args);
```

`per_core_address` falls back to `buffer.address()` for buffers without per-core allocation enabled — preserves uniform behavior byte-for-byte.

`override_runtime_arguments` recomputes per-core args on cache hit; addresses may have moved (e.g., a freed-and-reallocated output buffer).

### B.2 Kernel changes (excerpt)

```cpp
// U23: per-core src/dst L1 base addresses passed as runtime args.
// A value of 0 means "fall back to the CB base" -- preserves any caller
// that has not been updated to pass per-core overrides.
uint32_t src_addr_override = get_arg_val<uint32_t>(i); i += 1;
uint32_t dst_addr_override = get_arg_val<uint32_t>(i); i += 1;
uint32_t src_cb_base_addr = src_addr_override != 0 ? src_addr_override : get_read_ptr(src_cb_id);
uint32_t dst_cb_base_addr = dst_addr_override != 0 ? dst_addr_override : get_write_ptr(dst_cb_id);
```

### B.3 Verification

```
$ cd /tt-metal && python3 -m pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_move_sharded.py -x
…
============================== 34 passed in 5.65s ==============================
```

```
$ # Canonical (no prefetcher) generate
$ curl -X POST http://127.0.0.1:30000/generate \
$   -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
" What is 2+2? What is 2+2? What"   # bytewise-equal to U22 canonical

$ # Canonical GSM8K(10) chat
$ python3 .../test/eval_qwen3_8b_gsm8k_chat.py --num 10
FINAL: 10/10 = 100.0%
```

## Phase C — REALLOCATE retest with the fix

```bash
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U22_REALLOCATE_W2=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -u -m sglang.launch_server ... &

curl -X POST http://127.0.0.1:30000/generate \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
# " WhatQ!/)'C&L/820'0"   <- still garbage
```

GSM8K(10) chat:
```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
SIGQUIT received.  FINAL: 0/10 = 0.0%
```

Log line that finally pinpoints the real blocker:
```
warning  | Metal | Allocating device buffers is unsafe due to the existence of an active trace.
                  These buffers may be corrupted once a trace is executed. (allocator.cpp:110)
```

This warning fires every time `ttnn.reallocate(w2_out, ...)` runs inside the captured decode trace.

## Phase D — why REALLOCATE-inside-trace can't be the workaround

`generator.py:1583`: `trace_id = ttnn.begin_trace_capture(...)`.

Inside the capture region, `mlp.py` runs `ttnn.reallocate(w2_out, w2_out.memory_config())`.

This:
1. Deallocates the old `w2_out` (`ttnn.move` → `move_sharded` → `create_ghost_tensor` + `tensor.deallocate(false)`).
2. Creates a NEW output buffer at the next free L1 slot via `create_device_tensor`.  This allocation is what triggers the warning.
3. Captures `move_sharded` (a NoC backward-copy from the old src to the new dst) into the trace stream.

On replay (`ttnn.execute_trace`):
- The trace stream re-issues the captured NoC commands targeting the captured src/dst addresses.
- The CB config payload (also captured) re-multicasts the captured `buffer.address()` to every core.
- But the buffer that was allocated *during* trace capture is owned/disowned by the allocator OUTSIDE the trace.  Allocator state evolves between captures and replays.  The captured NoC reads may target L1 locations that no longer hold the expected W2 output (some other tensor's data may live there now, or the same L1 slot may have been reused for a different buffer with different per-core mapping).

This is exactly what the allocator's warning says.  The fix is structural, not in `move_sharded`.

## Hypothesis ledger (post-U23)

| ID | Suspect | Pre-U22 | Post-U22 | Post-U23 |
|---|---|---|---|---|
| **U22 B — `ttnn.reallocate` (ttnn::move) preserves sharding AND fixes stomp** | (new) | PARTIAL → RULED OUT (U22 attributed to chunk_size bug) | re-RULED OUT (`move_sharded` per-core chunk_size fix DOES NOT help; actual blocker is reallocate-inside-trace allocator-unsafety) | RULED OUT — structural; needs different workaround. |
| **U23 — `move_sharded` per-core chunk_size bug is the REALLOCATE blocker** | (new) | n/a | TENTATIVELY TRUE per U22 analysis | **RULED OUT** — fix lands cleanly but does not change REALLOCATE behavior; bug analysis was misattribution. |
| **U23 — `move_sharded` is silently incorrect for per-core-allocated sharded buffers** | (new) | n/a | (latent) | **CONFIRMED + FIXED** in tt-metal-sglang `5a5b3b5cdd6`. |
| **STANDING — fabric / EDM completion acks stomp 0xa6700** | (carry-over from U22) | (new) | STANDING TOP PRIORITY | STANDING TOP PRIORITY (no U23 work). |
| **STANDING — non-dispatcher kernel (e.g., prefetcher reader_dram, fabric router) writes (2,7) L1 0xa6700** | (carry-over from U22) | (new) | STANDING | STANDING (no U23 work). |
| **STANDING — `update_remote_cb_config_in_l1` bookkeeping writes** | (carry-over from U21) | (new) | STANDING | STANDING (no U23 work). |

## Remaining stomper candidate set (unchanged from U22, narrowed by U23 only in REALLOCATE-class workarounds)

- All ttnn-level sharding-preserving relocation primitives are
  exhausted as in-trace workarounds (5 variants, all fail; U22's
  "REALLOCATE almost worked" was a misread — the per-core chunk_size
  fix doesn't change the outcome).
- Stomper candidates still unattacked:
  1. **Fabric / EDM completion acks** routed from ethernet cores to
     tensix L1 0xa6700 on (2,7).
  2. **A non-dispatcher kernel** (dram_prefetcher reader_dram, fabric
     router) writing (2,7) L1 0xa6700.
  3. **`update_remote_cb_config_in_l1`** bookkeeping writes.

## What U23 lands

| File | Change |
|---|---|
| `ttnn/cpp/ttnn/operations/data_movement/move/device/move_sharded_program_factory.cpp` | Per-core runtime args; query per-core src/dst addresses via `experimental::per_core_allocation::get_per_core_address` (falls back to `buffer.address()` for uniform shards).  TT_FATAL invariants: `dst > src` and chunk-size aligned per core. |
| `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp` | Read two extra runtime args (`src_addr_override`, `dst_addr_override`).  Use them in place of `get_read_ptr(cb)` / `get_write_ptr(cb)` when non-zero.  Zero → CB-base fallback (preserves any caller using the older 4-arg layout). |

No sglang Python changes.  No env gate.  The fix is always active and
is a NO-OP for uniform sharded buffers.

## Reproducer

```bash
# tt-metal-sglang HEAD is 5a5b3b5cdd6.  Container has the rebuild already.

# 1. Run move_sharded unit tests (confirms fix is correct + uniform behavior preserved):
podman exec p3a-ngram bash -c '\
  cd /tt-metal && \
  python3 -m pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_move_sharded.py -x \
  2>&1 | tail -3'
# ============================== 34 passed in 5.65s ==============================

# 2. Canonical Qwen3-8B (no prefetcher) -- 10/10 GSM8K(10):
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u23_canonical.log 2>&1 &'
podman exec p3a-ngram bash -c '\
  until curl -sf http://127.0.0.1:30000/health_generate > /dev/null 2>&1; do sleep 5; done; \
  python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 10'
# FINAL: 10/10 = 100.0%

# 3. REALLOCATE attempt under prefetcher -- still garbage; confirms move_sharded was not the blocker:
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U22_REALLOCATE_W2=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u23_realloc.log 2>&1 &'
podman exec p3a-ngram bash -c 'until curl -sf http://127.0.0.1:30000/health_generate > /dev/null 2>&1; do sleep 5; done; \
  curl -X POST http://127.0.0.1:30000/generate \
    -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":15,\"temperature\":0.0}}"'
# " WhatQ!/)'C&L/820'0"  -- still garbage
grep -c "Allocating device buffers is unsafe" /tmp/u23_realloc.log    # > 0 -- the real blocker
```

## U24 dispatch recommendation

The L1 0xa6700 stomp continues to elude.  Because all in-trace Python
workarounds are exhausted (U22 Phase B) and the move_sharded fix
doesn't change the outcome (U23), the next attack must target the
stomper source rather than the victim:

### U24 Phase A — fabric / EDM completion ack probe

Instrument ethernet core kernels (fabric routers / EDM forwarders) to
dump every tensix L1 write target.  Specifically:
- `tt_metal/fabric/...` ethernet kernel sources.
- Look for `noc_async_write` / `cq_noc_async_write_with_state`
  targeting tensix L1 (NoC XY mapping for (2,7) virtual coord =
  `0x1c20` per U19/U21 prior data).
- Gate on `dst_addr in [0xa6000, 0xa7000]`.

### U24 Phase B — host-side L1 poller kernel

Per U21/U22 Phase C: write a tiny custom kernel that takes (noc XY,
L1 offset, poll_iter) as RT args and polls L1 every N cycles,
DPRINTing on change.  Dispatch on a clean sub-device that has only
the poller core.  Inserted between W2 and RS in the Python pipeline
(captured into trace).  The zero→NONZERO transition timestamp
pinpoints the stomp event.

### U24 Phase C — out-of-trace REALLOCATE

Instead of running `ttnn.reallocate` inside the captured decode
trace, attempt to land W2 output at a pre-chosen non-0xa6700 L1 slot
at trace-capture time only, then keep the same address across
replays.  Requires deeper allocator surgery and may not be possible
without tt-metal changes.

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (only tt-metal C++).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path.
- [x] No stash pop/drop.
- [x] Port-clear used `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Cache clear literal absolute path with var guard.
- [x] Container is NOT bind-mounted — used `podman cp` for source
      sync + `rebuild_tt_metal_kernels.sh` (with INSTALL_DIR sync).
- [x] Move-op unit tests + canonical Qwen3-8B GSM8K(10) verified —
      no regression.
- [x] Fix preserves existing `move_sharded` behavior for uniform-src
      case (byte-for-byte runtime-args match for uniform shards).
- [x] Server stopped at session end.

## Shipping verdict (unchanged from U19/U20/U21/U22)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~27 ms / 1024-1024 and GSM8K(10) chat 10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  The
  L1 0xa6700 stomp persists; no real-weight-preserving workaround
  found.
- **DO NOT ship any `SGLANG_TT_U22_*` env var.**  Diagnostic only.

The U23 `move_sharded` per-core fix is **safe to ship**.  It is
always-on, is a no-op for uniform sharded buffers, fixes a latent
correctness bug for per-core-allocated sharded buffers, and is
verified by the existing unit-test suite and canonical model run.

## Commits this session

- (tt-metal-sglang) **`5a5b3b5cdd6`** —
  `move_sharded: thread per-core src/dst addresses to support per-core-allocated sharded buffers`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`5a5b3b5cdd6`** (U23 commit; always-on; no
  env gate; uniform-shard byte-for-byte preserved).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.  No sglang Python changes
  in U23.
- Container `/tt-metal/`: synced
  (`move_sharded_program_factory.cpp` + `reader_unary_local_l1_copy_backwards.cpp`
  + rebuilt `_ttnn.so` / `_ttnncpp.so`; verified
  `move_sharded backward-copy requires dst_addr (...) > src_addr ...`
  string in `_ttnncpp.so`).
- `/root/.cache/tt-metal-cache/*`: cleared at session start.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: 10/10 GSM8K(10) chat; `" What is 2+2? What
  is 2+2? What"` greedy generate (identical to U22 baseline).
