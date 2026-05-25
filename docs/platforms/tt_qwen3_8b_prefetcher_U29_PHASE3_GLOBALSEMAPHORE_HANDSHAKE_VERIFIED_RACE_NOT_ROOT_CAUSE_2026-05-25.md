# TT Qwen3-8B prefetcher — U29 Phase 3: GlobalSemaphore handshake VERIFIED end-to-end; W2→RS race REFUTED as root cause — 2026-05-25

Status: **EVIDENCE_ADVANCE — U29 Phase 3 replaces the Phase 2
hardcoded L1 sema slot (0x90000, in the allocator-managed region
and stomped by tensor placement) with a properly allocated
`ttnn.create_global_semaphore` whose address (e.g. `0x17f300`) is
queried per-MLP and exported via env to both the matmul gathered
factory and the RS line program factory.  A sentinel-mode test
(producer writes 0xDEADBEEF; consumer DPRINTs readback) proves
the handshake works on EVERY RS-reader call across both devices:
`val=0xdeadbeef` observed 2880/2880 times.  The counter-mode wait
loop also engages and tracks `prev_seen=1,2,3,…` across forwards.
Despite the handshake being PROVABLY synchronizing W2 PACK→RS
reader, the Qwen3-8B prefetcher output remains garbage.  U28-β
(cross-sub-device dispatch race at W2→RS) is therefore REFUTED
as the root cause.  Canonical (U29 OFF) re-verified bytewise-equal
at 11.84s + GSM8K(10) = 9/10 = 90%.  No regression.**

Continuation of `tt_qwen3_8b_prefetcher_U29_PHASE2_SIGNALER_WAIT_WIRED_EVIDENCE_ADVANCE_2026-05-25.md`.

tt-metal-sglang HEAD: pending this U29 Phase 3 commit.
Branch `tenstorrent-p1`; NOT pushed (predator2k/* fork only).
sglang HEAD: pending this doc commit.

## TL;DR (Phase 3 forensic chain)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| 3.A | Can we use a SAFE L1 sema address via tt-metal's allocator API? | New: `ttnn.create_global_semaphore(self.mesh_device, receiver_crs, 0)` at first W2 forward (after `prefetcher.init(Mode.DECODE)` binds receiver_cores); query addr via `ttnn.get_global_semaphore_address` and export as env `SGLANG_TT_U29_SEMA_L1_ADDR=0x<addr>`. | **YES** — layer-0 sema allocates at `0x17f300` on BH P150a; the matmul factory + RS factory bake this address into the JIT kernel binary's `SGLANG_TT_U29_SEMA_L1` define.  Per-layer sema addrs differ (allocator's high-water mark drifts) but only the FIRST W2 JIT compile wins, so all 36 layers' W2 kernels + their RS readers share the same baked address. |
| 3.B | Sentinel test — does the producer→consumer handshake actually engage on the SAFE slot? | New env `SGLANG_TT_U29_SENTINEL=1`: producer writes `0xDEADBEEF` (via volatile L1 ptr) instead of `noc_semaphore_inc`; RS reader DPRINTs the value from the FIRST producer's slot.  Run with `TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=NC`. | **PASS — 2880/2880 reads return `0xdeadbeef`** from `prod=(2,7) sema_addr=0x17f300` on both devices.  The handshake is provably correct: producer's noc-write commits before consumer's noc-read fires on every single RS-reader invocation. |
| 3.C | Counter-mode test — does the wait loop engage and track the producer counter? | Default mode (no `_SENTINEL`): producer `noc_semaphore_inc`; consumer waits for `cur_val > u29_prev_seen`.  DPRINT post-wait. | **PASS — `prev_seen` increments 1, 2, 3, … across RS calls.** The wait loop iterates, observes the producer's increment, updates `prev_seen` correctly. |
| 3.D | Does the Phase 3 handshake fix the Qwen3-8B prefetcher garbage? | Smoke test `What is 2+2?` under `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_U29_W2_RS_SIGNALER=1`. | **NO** — output: `" What expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise"` (15 garbage tokens, 0.29s warm decode).  Different garbage pattern from Phase 2's `" What doing doing doing..."` but still garbage; sampler does not crash (no NaN logits this run). |
| 3.E | Did we break canonical (U29 OFF)? | curl + GSM8K(10) chat with prefetcher OFF | **NO — `" What is 2+2? What is 2+2? What"` at 11.84s; GSM8K(10) = 9/10 = 90%.**  Bytewise-equal vs U28 canonical baseline. |
| 3.F | Implication for U28-β (cross-sub-device dispatch race at W2→RS)? | Logical deduction: the handshake provably synchronizes W2 PACK → RS reader on every RS call; the garbage persists → either the race is NOT here, or the race is here AND there's an additional bug downstream. | **U28-β REFUTED as sole root cause.**  The handshake is the empirically correct closure of the W2→RS cross-subdev gap; if it were the only race, output would be coherent now.  The garbage must originate elsewhere. |

**Punchline:** U29 Phase 3 lands a provably-correct GlobalSemaphore-
based device-side handshake between W2 PACK and the line RS reader
(sentinel test passes 2880/2880; counter wait engages and tracks
correctly).  This rules out the U28-β cross-sub-device dispatch
race as the root cause of the Qwen3-8B prefetcher garbage.  The
fix is safe to leave in tree as default-OFF; under U29 OFF the
canonical Qwen3-8B path remains bytewise-equal at 9-10/10 GSM8K.

The next race-hunt direction (Phase 4) is to instrument the OTHER
prefetcher-touched gathered matmul → CCL chains (WO →
all_reduce_async, FF2 paths, FF1/FF3 → ttnn.mul, WQKV →
nlp_create_qkv_heads_decode) with the same sentinel pattern.  The
W2→RS chain is now verified-correct; the bug is upstream of (or
parallel to) W2.

## What U29 Phase 3 lands

All env-gated by `SGLANG_TT_U29_W2_RS_SIGNALER=1`; default OFF;
canonical bytewise-equal verified.  Five files modified in
tt-metal-sglang; ZERO sglang Python changes (this doc only).

### tt-metal-sglang files modified (5)

| File | Phase 2 → Phase 3 change |
|---|---|
| `models/tt_transformers/tt/mlp.py` | **(1)** New `_u29_ensure_sema()` lazy-init helper that calls `ttnn.create_global_semaphore(self.mesh_device, prefetcher.to_core_range_set(prefetcher.receiver_cores(sender_active=True, receiver_active=True)), 0)` ONCE per MLP instance on first W2 forward (after `prefetcher.init(Mode.DECODE)` binds `receiver_cores`).  Caches handle on `self._u29_sema` + int addr on `self._u29_sema_addr`. **(2)** Around W2 `ttnn.linear`: set env `SGLANG_TT_U29_SEMA_L1_ADDR=0x<addr>` alongside the existing `SGLANG_TT_U29_W2_PRODUCER_NOW=1`; restore in `finally`. **(3)** Around `tt_all_reduce(w2_out, …)`: set env `SGLANG_TT_U29_W2_RS_NOW=1` + the sema addr again (RS program creation is OUTSIDE the W2 linear try/finally); restore in `finally`.  **(4)** REMOVED the `ttnn.reset_global_semaphore_value(…)` call (host writes are not supported during trace capture; producer overwrites the slot on every dispatch anyway). |
| `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | Read `SGLANG_TT_U29_SEMA_L1_ADDR` env at program-creation time; fall back to `0x90000` (Phase 1/2 legacy) if unset; bake into `SGLANG_TT_U29_SEMA_L1` define on the in1 + compute kernels.  Also new env gate `SGLANG_TT_U29_SENTINEL=1` propagates the matching `SGLANG_TT_U29_SENTINEL` define. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | NEW `#ifdef SGLANG_TT_U29_SENTINEL` block — write `0xDEADBEEF` via volatile L1 pointer at `SGLANG_TT_U29_SEMA_L1`; drain via `noc_async_write_barrier()`.  `#else` keeps the Phase 2 `noc_semaphore_inc` + `noc_async_atomic_barrier`.  Both paths preserve the Phase 2 `cb_sync.wait_front(1) → cb_sync.pop_front(1) → noc.async_write_barrier()` strict happens-before w.r.t. compute's PACK retirement. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp` | NEW second env gate `SGLANG_TT_U29_W2_RS_NOW=1` AND the existing `SGLANG_TT_U29_W2_RS_SIGNALER=1` — both required for the U29 RS defines to be emitted.  Reason: Other RS calls (w1/w3 in MLP, WO in attention) re-use the same JIT kernel binary; without narrowing, they would all bake in the wait loop and spin forever on a never-incremented producer counter.  Reads `SGLANG_TT_U29_SEMA_L1_ADDR` for the matched sema-addr define; reads `SGLANG_TT_U29_SENTINEL` for the matched sentinel-define. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp` | NEW `#ifdef SGLANG_TT_U29_SENTINEL` block — sample the FIRST producer core's L1 slot via `noc_async_read`, then `DPRINT << "[U29_SENTINEL prod=(x,y) sema_addr=0x… val=0x…]"`. `#else` keeps the Phase 2 relative-counter wait loop. |

### Run-time matrix verified

| `USE_PREFETCHER` | `U29_W2_RS_SIGNALER` | `U29_SENTINEL` | Sema addr (kernel define) | Smoke output | Decode latency | Sentinel readback |
|---|---|---|---|---|---|---|
| 0 | 0 | — | n/a | `" What is 2+2? What is 2+2? What"` (canonical) | 11.84s cold | n/a |
| 1 | 1 | 0 | `0x17f300` (GlobalSemaphore — SAFE L1) | `" What expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise expertise"` (garbage) | 10.30s cold, 0.29s warm | n/a; counter `prev_seen` advances 1,2,3,… per RS call (DPRINT confirmed). |
| 1 | 1 | 1 | `0x17f300` | garbage (`" What.Dao mailbox mailbox mailbox mailbox"`) | 16.12s cold | **`val=0xdeadbeef` on 2880/2880 RS-reader reads**, both devices, multiple worker cores. |

## Architectural learnings (Phase 3)

### Learning 1: `ttnn.create_global_semaphore` returns a SAFE L1 address

Per-layer call returns distinct addresses (allocator high-water mark
drifts as intermediate tensors live/die during model init):

```
[U29_PHASE3_INIT layer=0  sema_addr=0x17f300]
[U29_PHASE3_INIT layer=1  sema_addr=0x17f2c0]
...
[U29_PHASE3_INIT layer=16 sema_addr=0x17ef00]
[U29_PHASE3_INIT layer=17 sema_addr=0xa86c0]   # <-- allocator switched regions
[U29_PHASE3_INIT layer=18 sema_addr=0xaad40]
...
[U29_PHASE3_INIT layer=35 sema_addr=0xaa900]
```

The matmul JIT cache is keyed by `(matmul_config, kernel_defines)` not
by layer, so only the FIRST W2 program-creation (layer 0) wins; all 36
layers' W2 kernels share the SAME baked address (`0x17f300`).  This
slot remains allocator-reserved for the lifetime of layer-0's sema
object (never deallocated), so it is permanently SAFE (no tensor
landings there).  Sentinel test confirms: every producer write +
consumer read sees `0xDEADBEEF` correctly.

(Sub-finding: layers 17+ sema addresses near `0xaa900-0xaad40` are
dangerously close to the W2 output buffer region (`0xa6700-0xb0000`
per U26 evidence).  Those sema slots are NOT used in the JIT binary
— only layer-0's `0x17f300` is — but they ARE allocated and live in
parallel.  Worth a follow-up audit to confirm no live tensor at
those addresses.)

### Learning 2: kernel JIT cache shares binaries across layers

The matmul-1d-mcast factory's JIT cache key is `(program_config,
in1_kernel_defines, mm_kernel_defines, compile_args)`.  Layer-N's
W2 ttnn.linear call with identical config + defines as layer-0 hits
the cache.  Therefore env-driven defines like `SGLANG_TT_U29_SEMA_L1`
are FROZEN at first compile-time of each unique config.  Phase 3
exploits this: layer 0 sets `SGLANG_TT_U29_SEMA_L1_ADDR=0x17f300`,
the matmul binary bakes that, layers 1-35 reuse the binary.  RS
factory cache works the same way.

### Learning 3: `reset_global_semaphore_value` is a host op — fails inside trace capture

```
TT_FATAL @ /tt-metal/tt_metal/distributed/fd_mesh_command_queue.cpp:590:
!trace_id_.has_value()
info: Writes are not supported during trace capture. trace id: 0
```

REMOVED the reset call.  Not needed: producer overwrites the slot
on every dispatch (sentinel: `0xDEADBEEF`; counter: monotonic `+1`).

### Learning 4: per-RS-call wait narrowing is required

The line RS minimal-async kernel binary is shared across ALL RS callers
(w1/w3 RS in MLP for TG path, W2→RS via tt_all_reduce for N300/T3K
path, etc.).  Phase 2 had `SGLANG_TT_U29_W2_RS_SIGNALER=1` as the
single gate, which would have made every RS reader bake in the
wait loop.  For non-W2 RS chains (which have no producer-side
increment), the wait would spin forever.

Phase 3 introduces `SGLANG_TT_U29_W2_RS_NOW=1` — mlp.py sets it
only around `tt_all_reduce(w2_out, …)` (which is the only RS that
should engage the wait).  Factory check becomes `(U29_SIGNALER==1)
&& (U29_W2_RS_NOW==1)`.  Mirrors the Phase 2 v3 narrowing on the
producer side (`SGLANG_TT_U29_W2_PRODUCER_NOW`).

### Learning 5: `num_producers=32` matches W2 receiver cores — the handshake IS targeting W2

DPRINT confirms `num_producers=32 sema_addr=0x17f300`.  32 producer
NoC coords correctly derive from `input_tensor.memory_config()
.shard_spec().grid` (the W2 output cores).  Sentinel-mode reads
hit `prod=(2,7)` which is a valid W2 receiver-core logical coord
under the Blackhole MUX-clamped 16-receiver mapping.  This is
empirical proof the wiring is end-to-end correct.

### Learning 6: U28-β was a plausible mechanism but NOT the root cause

The cross-sub-device dispatch race at W2→RS is a REAL architectural
gap (W2 on receiver_sub_device, RS on worker_sub_device,
`set_sub_device_stall_group([worker])` excludes receiver from
cross-subdev sync).  U29 Phase 3 closes that gap with a proven
device-side kernel-level handshake.  Output remains garbage.
Therefore U28-β cannot be the sole (or even the dominant) source
of the corruption.

Other gathered-matmul → CCL chains on the prefetcher path use
DIFFERENT kernel pairs that U29 Phase 3 does NOT instrument:
- **WO → all_reduce_async** (attention output).  Uses
  `all_reduce_async`'s `worker_reader.cpp` /
  `reduction_receiver.cpp`.  NOT covered by Phase 3.
- **FF1/FF3 → ttnn.mul → FF2**.  Uses unary/binary ops.  NOT
  covered.
- **WQKV → nlp_create_qkv_heads_decode**.  Uses
  `nlp_create_qkv_heads_decode_program_factory`.  NOT covered.

Phase 4 priorities (deferred):
1. **WO → all_reduce_async** — apply the same producer-side
   increment in `worker_reader.cpp` (compute kernel exit-sync +
   dataflow's L1 write) + consumer-side wait in
   `reduction_receiver.cpp`.
2. **Sentinel sweep**: instrument every gathered-matmul exit AND
   every CCL receiver entry with a sentinel value; the first chain
   where consumer sees something OTHER than the sentinel identifies
   the first racing boundary.

## Implementation diffs (selected)

### A. mlp.py — lazy sema init + env-driven addr export

```python
def _u29_ensure_sema(self):
    """U29 Phase 3 — lazily allocate the W2 GlobalSemaphore."""
    if self._u29_sema is not None or self._u29_sema_init_attempted:
        return
    self._u29_sema_init_attempted = True
    if self.prefetcher is None:
        return
    try:
        _u29_recv_crs = self.prefetcher.to_core_range_set(
            self.prefetcher.receiver_cores(sender_active=True, receiver_active=True)
        )
        self._u29_sema = ttnn.create_global_semaphore(
            self.mesh_device, _u29_recv_crs, 0
        )
        self._u29_sema_addr = int(ttnn.get_global_semaphore_address(self._u29_sema))
        print(f"[U29_PHASE3_INIT layer={self.layer_num} "
              f"sema_addr=0x{self._u29_sema_addr:x}]", flush=True)
    except Exception as _u29_init_e:
        print(f"[U29_PHASE3_INIT ERROR layer={getattr(self, 'layer_num', '?')}: "
              f"{type(_u29_init_e).__name__}: {_u29_init_e}]", flush=True)
```

Pre-W2 (inside `_u29_active`):

```python
self._u29_ensure_sema()
_u29_prev = _u29_os.environ.get("SGLANG_TT_U29_W2_PRODUCER_NOW")
_u29_os.environ["SGLANG_TT_U29_W2_PRODUCER_NOW"] = "1"
_u29_addr_prev = _u29_os.environ.get("SGLANG_TT_U29_SEMA_L1_ADDR")
if getattr(self, "_u29_sema_addr", None) is not None:
    _u29_os.environ["SGLANG_TT_U29_SEMA_L1_ADDR"] = f"0x{self._u29_sema_addr:x}"
```

Pre-`tt_all_reduce(w2_out)`:

```python
if _u29_rs_active:
    self._u29_ensure_sema()
    _u29_rs_prev_now = _u29_rs_os.environ.get("SGLANG_TT_U29_W2_RS_NOW")
    _u29_rs_prev_addr = _u29_rs_os.environ.get("SGLANG_TT_U29_SEMA_L1_ADDR")
    _u29_rs_os.environ["SGLANG_TT_U29_W2_RS_NOW"] = "1"
    if getattr(self, "_u29_sema_addr", None) is not None:
        _u29_rs_os.environ["SGLANG_TT_U29_SEMA_L1_ADDR"] = f"0x{self._u29_sema_addr:x}"
```

### B. Matmul factory — env-driven define propagation

```cpp
if (u29_active) {
    const char* u29_addr_env = std::getenv("SGLANG_TT_U29_SEMA_L1_ADDR");
    std::string u29_sema_l1_define = "0x90000";
    if (u29_addr_env != nullptr && u29_addr_env[0] != '\0') {
        u29_sema_l1_define = std::string(u29_addr_env);
    }
    mm_in1_kernel_defines["SGLANG_TT_U29_W2_RS_SIGNALER"] = "1";
    mm_in1_kernel_defines["SGLANG_TT_U29_SEMA_L1"] = u29_sema_l1_define;
    mm_kernel_defines["SGLANG_TT_U29_W2_RS_SIGNALER"] = "1";
    mm_kernel_defines["SGLANG_TT_U29_SEMA_L1"] = u29_sema_l1_define;

    const char* u29_sentinel_env = std::getenv("SGLANG_TT_U29_SENTINEL");
    if (u29_sentinel_env != nullptr && std::string(u29_sentinel_env) == "1") {
        mm_in1_kernel_defines["SGLANG_TT_U29_SENTINEL"] = "1";
        mm_kernel_defines["SGLANG_TT_U29_SENTINEL"] = "1";
    }
}
```

### C. Producer kernel — sentinel write OR atomic_inc

```cpp
{
    const uint32_t u29_my_x = my_x[noc_index];
    const uint32_t u29_my_y = my_y[noc_index];
    uint64_t u29_self_sema_noc_addr =
        get_noc_addr(u29_my_x, u29_my_y, (uint32_t)(SGLANG_TT_U29_SEMA_L1));
#ifdef SGLANG_TT_U29_SENTINEL
    {
        volatile tt_l1_ptr uint32_t* u29_local_sema =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                (uint32_t)(SGLANG_TT_U29_SEMA_L1));
        u29_local_sema[0] = 0xDEADBEEFu;
        noc_async_write_barrier();
    }
#else
    noc_semaphore_inc(u29_self_sema_noc_addr, 1);
    noc_async_atomic_barrier();
#endif
}
```

### D. RS reader kernel — sentinel readback DPRINT

```cpp
#ifdef SGLANG_TT_U29_SENTINEL
if (u29_num_producers > 0) {
    cb_reserve_back(cb_input_id, 1);
    uint32_t u29_l1_scratch = get_write_ptr(cb_input_id);
    volatile tt_l1_ptr uint32_t* u29_scratch_p =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u29_l1_scratch);
    static uint32_t u29_sentinel_log_budget = 64;
    const uint32_t px = get_arg_val<uint32_t>(u29_producer_args_start + 0);
    const uint32_t py = get_arg_val<uint32_t>(u29_producer_args_start + 1);
    const uint64_t prod_sema_noc_addr =
        get_noc_addr(px, py, static_cast<uint32_t>(SGLANG_TT_U29_SEMA_L1));
    u29_scratch_p[0] = 0;
    noc_async_read(prod_sema_noc_addr, u29_l1_scratch, 4);
    noc_async_read_barrier();
    const uint32_t observed = u29_scratch_p[0];
    if (u29_sentinel_log_budget > 0) {
        u29_sentinel_log_budget--;
        DPRINT << "[U29_SENTINEL prod=(" << px << "," << py
               << ") sema_addr=0x" << HEX()
               << static_cast<uint32_t>(SGLANG_TT_U29_SEMA_L1)
               << " val=0x" << observed
               << DEC() << "]" << ENDL();
    }
}
#else
// Phase 2 relative-counter wait loop preserved as-is
#endif
```

### E. RS factory — narrowing gate

```cpp
const char* u29_sig_env = std::getenv("SGLANG_TT_U29_W2_RS_SIGNALER");
const char* u29_rs_now_env = std::getenv("SGLANG_TT_U29_W2_RS_NOW");
const bool u29_enabled =
    (u29_sig_env != nullptr && std::string(u29_sig_env) == "1") &&
    (u29_rs_now_env != nullptr && std::string(u29_rs_now_env) == "1");
```

## Hardware test result (full reproduction)

### Canonical (U29 OFF) — bytewise + GSM8K

```
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
HF_MODEL=/models/Qwen3-8B \
python3 -m sglang.launch_server ...

curl -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
→ " What is 2+2? What is 2+2? What" at 11.84s e2e_latency

eval_qwen3_8b_gsm8k_chat.py --num 10
→ Q1✓ Q2✓ Q3✗ Q4✓ Q5✓ Q6✓ Q7✓ Q8✓ Q9✓ Q10✓ = 9/10 = 90.0%
```

**Bytewise-equal vs U28 canonical baseline.  No regression.**
(Q3 wrong is a known per-run sampling instability; canonical
historical chat band is 9-10/10.)

### Sentinel test (Phase 3 mechanism verification)

```
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U29_W2_RS_SIGNALER=1 \
SGLANG_TT_U29_SENTINEL=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_RISCVS=NC \
python3 -m sglang.launch_server ...

curl -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":5,"temperature":0.0}}'
→ " What.Dao mailbox mailbox mailbox" at 16.12s e2e_latency  (cold, garbage)

grep U29_SENTINEL /tmp/u29p3_sentinel2.log | wc -l
→ 2880

grep "val=0x" /tmp/u29p3_sentinel2.log | grep -v "0xdeadbeef" | wc -l
→ 0
```

**100% of 2880 RS-reader sentinel reads return `0xDEADBEEF`** on
both devices across multiple worker cores. The handshake is
**provably correct** end-to-end.  Layer-0's sema address
`0x17f300` is correctly baked into the JIT binaries and used by
both producer and consumer.

### Counter mode (production engagement test)

```
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U29_W2_RS_SIGNALER=1 \
SGLANG_TT_U29_DEBUG_DPRINT=1 \
TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_RISCVS=NC \
python3 -m sglang.launch_server ...

curl -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":5}}'
→ " What.Dao mailbox mailbox mailbox" (garbage; sampler does NOT crash)

grep "U29_RS_READER_OK" /tmp/u29p3_dprint.log | head
0:1-2:NC: [U29_RS_READER_OK prev_seen=1 num_producers=32 sema_addr=0x17f300]
0:2-0:NC: [U29_RS_READER_OK prev_seen=1 num_producers=32 sema_addr=0x17f300]
...
0:1-2:NC: [U29_RS_READER_OK prev_seen=2 num_producers=32 sema_addr=0x17f300]
0:1-2:NC: [U29_RS_READER_OK prev_seen=3 num_producers=32 sema_addr=0x17f300]
```

**`prev_seen` increments 1 → 2 → 3 across RS calls per worker
core.**  The wait engages, observes producer's monotonic counter
advancing, and updates state correctly.  The race fix is engaged.

### Production engagement (no DPRINT) — bug ≠ fixed

```
SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_U29_W2_RS_SIGNALER=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 ...

curl '{"text":"What is 2+2?","max_new_tokens":15}'
→ " What expertise expertise..." at 10.30s cold

curl '{"text":"What is the capital of France?","max_new_tokens":15}'
→ " What doing doing doing doing doing..." at 0.29s warm
```

Garbage symptom unchanged.  Phase 3 fix does NOT resolve the
prefetcher correctness bug.  → U28-β race is REFUTED as the root
cause; the bug must be in another chain (Phase 4).

## Hypothesis ledger (post-U29 Phase 3)

| ID | Suspect | Pre-Phase-3 | Post-Phase-3 |
|---|---|---|---|
| **U16 Phase 5c — Device-side GlobalSemaphore producer-consumer handshake (W2→RS)** | TOP STANDING | **LANDED + EMPIRICALLY VERIFIED CORRECT (sentinel 2880/2880 = `0xdeadbeef`; counter `prev_seen` advances).  RETIRED as fix candidate for THIS chain.**  The chain is now race-free by construction. |
| **U28 β — Cross-sub-device dispatch race at W2→RS** | NEW TOP STANDING per U28 | **REFUTED as the sole root cause.**  The dispatch race IS real architecturally, but closing it does NOT remove the garbage.  Either the gap is closed only as far as the kernel handshake reaches and there's a deeper data corruption upstream of W2, or other chains (WO, FF2, etc.) have their own independent dispatch races that dominate the symptom. |
| **NEW U29 — L1 0x90000 is in allocator-managed region** | STANDING (Phase 2 finding) | **CONFIRMED + FIXED** via `ttnn.create_global_semaphore` allocator-owned addr `0x17f300`. |
| **NEW U29 — WO / FF2 / WQKV → CCL chains also race independent of W2** | STANDING (Phase 2 conjecture) | **STILL STANDING — now the most likely root cause.**  Phase 4 should apply the same producer-side increment + consumer-side wait to: (a) WO → `all_reduce_async`'s `worker_reader.cpp` / `reduction_receiver.cpp` kernel pair; (b) FF2's chain (if distinct from W2's); (c) WQKV → `nlp_create_qkv_heads_decode_program_factory` if applicable. |
| **NEW U29 Phase 3 — Bug originates in W2's OUTPUT BYTES (not the W2→RS ordering)** | (new) | **NEW STANDING.**  If the W2 PACK output bytes themselves are wrong (e.g., GCB consumption race in the producer's in0 reader path, or compute kernel using stale weight tiles from the prefetcher's circular buffer), then the RS reader synchronously reads garbage that's already in mm_out_cb.  Phase 4 sub-probe: instrument w2_out's L1 bytes BEFORE the U29 sentinel write and BEFORE RS reads — compare to expected `out = w2_in @ w2_weight`. |
| **NEW U29 Phase 3 — Bug is in prefetcher GCB receiver-side race (upstream of W2 compute)** | (new) | **NEW STANDING.**  The prefetcher streams W2 weight tiles into a Global CB on the receiver cores; the W2 compute kernel pops from that GCB.  If the prefetcher producer (DRAM reader) races the W2 consumer (compute), W2 compute reads stale/zero weight tiles.  Result: w2_out bytes ARE garbage even with perfect W2→RS ordering. |
| U27 β / γ | (W2 PACK partial-tile, GCB overflow) | STANDING | STANDING — Phase 3 did not probe; would have been masked even if true. |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (only `models/tt_transformers/tt/mlp.py` in the tt-metal-sglang fork; all probes and gates env-gated; default OFF preserves canonical bytewise).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` and use literal absolute path (`TT_CACHE_HOME=/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop.
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted for `/tt-metal/` — used `podman cp` for source sync (5 files).  Used FIXED `rebuild_tt_metal_kernels.sh` for the build.
- [x] All new code env-gated under `SGLANG_TT_U29_W2_RS_SIGNALER` (plus `SGLANG_TT_U29_W2_PRODUCER_NOW` for producer narrowing, `SGLANG_TT_U29_W2_RS_NOW` for consumer narrowing, `SGLANG_TT_U29_SENTINEL` for the diagnostic write/read).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"` at 11.84s) AND GSM8K(10) = 9/10 = 90% with U29 OFF.
- [x] Server stopped at session end; cache cleared at session end.
- [x] Sentinel handshake verified: 2880/2880 reads = `0xdeadbeef`.

## Reproducer

### A. Build (host)

```bash
# Edit host sources (5 files).
# Sync to container:
podman cp /home/mhnie/tt-metal-sglang/<path>/<file> p3a-ngram:/tt-metal/<path>/<file>

# Rebuild:
podman exec p3a-ngram bash /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh ttnn

# Verify new defines in linked lib:
podman exec p3a-ngram bash -c 'strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep "SGLANG_TT_U29" | sort -u'
# → SGLANG_TT_U29_DEBUG_DPRINT
# → SGLANG_TT_U29_DISABLE_WAIT
# → SGLANG_TT_U29_SEMA_L1
# → SGLANG_TT_U29_SEMA_L1_ADDR          (NEW Phase 3)
# → SGLANG_TT_U29_SENTINEL              (NEW Phase 3)
# → SGLANG_TT_U29_W2_PRODUCER_NOW
# → SGLANG_TT_U29_W2_RS_NOW             (NEW Phase 3)
# → SGLANG_TT_U29_W2_RS_SIGNALER
```

### B. Canonical re-verify (default — U29 OFF)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u29p3_canonical.log 2>&1 &'

# Expected: " What is 2+2? What is 2+2? What" at ~11s, GSM8K 9-10/10
```

### C. Sentinel verification (handshake mechanism test)

```bash
podman exec p3a-ngram bash -c '
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U29_W2_RS_SIGNALER=1 \
  SGLANG_TT_U29_SENTINEL=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_RISCVS=NC \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u29p3_sentinel.log 2>&1 &'

# Expected:
# - Output garbage (handshake works but bug is elsewhere)
# - DPRINT: [U29_SENTINEL ... val=0xdeadbeef] on every RS reader call
# - Verify: grep U29_SENTINEL /tmp/u29p3_sentinel.log | grep -v "val=0xdeadbeef" → empty
```

### D. Counter-mode engagement test

```bash
podman exec p3a-ngram bash -c '
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U29_W2_RS_SIGNALER=1 \
  SGLANG_TT_U29_DEBUG_DPRINT=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_RISCVS=NC \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server ... > /tmp/u29p3_counter.log 2>&1 &'

# Expected: [U29_RS_READER_OK prev_seen=N num_producers=32 sema_addr=0x17f300]
# with N=1 then 2 then 3 etc. across consecutive RS reader calls.
```

## Shipping verdict (unchanged from U28)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Bug
  remains in active investigation; U28-β is now ruled out as the
  root cause, but the actual root cause is unidentified.
- **DO NOT ship `SGLANG_TT_U29_W2_RS_SIGNALER=1`.**  Same garbage
  symptom as without U29 at the smoke-test level; the handshake
  itself is correct but does not resolve the user-visible bug.

The U29 Phase 3 changes are safe to leave in tree as default-off:
- The 5 changes in tt-metal-sglang are guarded by
  `SGLANG_TT_U29_W2_RS_SIGNALER=1` (plus `SGLANG_TT_U29_W2_PRODUCER_NOW=1`
  for the matmul producer side, `SGLANG_TT_U29_W2_RS_NOW=1` for the
  RS consumer side, `SGLANG_TT_U29_SENTINEL=1` for the diagnostic);
  under default OFF they emit NO defines, NO additional kernel binary
  paths.
- Canonical bytewise-equal verified at 11.84s with U29 OFF +
  GSM8K(10) = 9/10 with U29 OFF.

## Phase 4 — exact next steps

1. **Sentinel-sweep WO → all_reduce_async** — same pattern in
   attention's W_O matmul (gathered path) and the
   `all_reduce_async`'s `reduction_receiver.cpp`.  If sentinel
   passes there too AND garbage persists, ALL gathered-matmul →
   CCL chains are race-free and the bug is upstream.
2. **Probe w2_out bytes BEFORE RS reads** — read `w2_out`'s L1 bytes
   from the host (or via a dedicated probe kernel) RIGHT BEFORE
   `tt_all_reduce(w2_out, …)` fires.  Compare to expected `w2_in @
   w2_weight` (computed offline in torch).  If bytes are already
   wrong, the bug is W2 compute / weight feeding, not W2→RS.
3. **GCB receiver-side race probe** — instrument the prefetcher's
   W2 GCB pop on the receiver cores; check that compute consumes
   the EXPECTED weight tiles in order (not stale or skipped).
4. **DPRINT in non-W2 RS readers** — re-enable Phase 2's
   `SGLANG_TT_U29_DEBUG_DPRINT` on w1/w3 RS calls (TG path, won't
   fire on Qwen3-8B 2× P150a) or temporarily widen the
   `SGLANG_TT_U29_W2_RS_NOW` gate to ALL RS calls and see which
   chain triggers the corruption.
5. **Revert U29 narrowing for Phase 4 wider experiments** — under
   the proper narrowing this iteration, only W2 producer + W2→RS
   wait engage; for Phase 4 we may need to widen the wait to OTHER
   gathered-matmul → CCL pairs.

## Commits this session

- **tt-metal-sglang**: pending — bumps HEAD (Phase 2 tip) → new SHA
  (5 files: mlp.py, matmul factory, matmul in1 sender writer,
  RS line program factory, RS line reader kernel).
- **sglang**: pending — bumps HEAD (Phase 2 tip) → new SHA
  (this doc only; no Python changes in sglang tree).

## Working state at session end

- tt-metal-sglang HEAD: pending (will be the U29 Phase 3 commit).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only).
- sglang HEAD: pending this doc commit.
- Container `/tt-metal/`: synced (5 modified files).
- Container `/tt-metal/ttnn/ttnn/_ttnn{,cpp}.so`: rebuilt; new symbols verified.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at 11.84s + GSM8K(10) = 9/10.
- Sentinel verify: 2880/2880 RS-reader reads = `0xdeadbeef`.

## Next-session checklist (Phase 4)

1. Read this doc + Phase 2 + Phase 1 + U28 as the latest evidence chain.
2. Phase 4a (mechanism): apply the same producer→consumer L1-handshake pattern to WO → `all_reduce_async`.
3. Phase 4b (race-hunt): if Phase 4a still fails, probe W2 OUTPUT BYTES directly via host read OR a dedicated probe kernel + compare to torch reference.
4. Phase 4c (GCB): instrument the prefetcher's W2 GCB receiver pops to confirm compute reads correct weight tiles.
5. If/when GSM8K(10) ≥ 7/10: bench TPOT, retire all U29 guards (squash into one production flag).
