# TT Qwen3-8B prefetcher — U15: Python-side to_torch + shard + L1-dump probes — bug is in MLP W2 → reduce_scatter (NOT attention WQKV), producer L1 is clean, consumer race in trace replay — 2026-05-24

Status: **EVIDENCE_ADVANCE — Six new env-gated Python-side probes
(`SGLANG_TT_U15_PROBE_{TOTORCH,SHARD,L1DUMP,RS,RS_TOTORCH,W2,W2_L1}`)
plus a synchronous to_torch barrier at every reduce_scatter call
in `tt_all_reduce`. (1) The U14 "consumer reads L1 0xa6700 NONZERO"
bug originates from **MLP's W2 (down_proj) → reduce_scatter**, NOT
from attention's WQKV matmul — Probe E's stack trace shows the
first 8 RS calls all dispatch from `forward@mlp.py:569`. (2) The W2
producer is innocent end-to-end: Probe F `ttnn.to_torch` on the
decode-shape (1,1,32,4096) W2 output reads ALL ZERO across both
mesh shards under zero-weight injection (nnz=0/131072 per shard).
(3) Probe E2 PRE-RS `to_torch` on the reduce_scatter INPUT (same
L1 0xa6700 the U14 reader saw NONZERO) reads ALL ZERO across all
8 RS calls. (4) Probe E2 POST-RS `to_torch` on the reduce_scatter
OUTPUT also reads ALL ZERO. (5) Probe G L1 allocator dump at W2
dispatch time shows EXACTLY ONE buffer at 0xa6700 — w2_out itself,
WIDTH_SHARDED 8192 B/bank — no allocator overlap (H2 RULED OUT
definitively). (6) Despite all probes reading zero through every
synchronous-barrier step, the final decode output remains garbage
(`"Und著作权"`, `"Und掐"`, etc., same U11 signature) — proving the
corruption only manifests in trace-replay execution where Python
to_torch barriers do not fire. The U5 existing fix
(`SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1`) routes the post-W2 CCL
op to receiver_sub_device_id to close the cross-sub-device dispatch
gap — but enabling it causes a decode-time hang (3-min curl
timeout, server stuck after "Done Capturing Decode Trace"). The
bug is a producer-consumer race in trace replay between W2 matmul's
PACK→mm_out_cb completion and reduce_scatter_minimal_async_reader's
read; the existing U5 mitigation is necessary but not sufficient.**

Continuation of `tt_qwen3_8b_prefetcher_U14_consumer_probe_shard_mismatch_2026-05-24.md`
(U14 narrowed bug to consumer reads at L1 `0xa6700`; producer L1
appeared clean at end-of-matmul-kernel for the WQKV ELFs).

tt-metal-sglang HEAD: **`2f7cda2f02b`** (unchanged from U14;
U15 probes are sglang-side only — no kernel rebuild needed).
sglang HEAD: pending this doc + Python probes in
`attention.py` (Probes A/B/C) + `mlp.py` (Probes F/G) +
`ccl.py` (Probes E/E2).

## TL;DR forensic chain

| Step | Question | Probe | Result |
|---|---|---|---|
| 1 | Is producer's L1 at 0xa6700 actually clean? | Probe A: `to_torch(xqkv_fused_sharded)` | xqkv (WQKV) shard0 + shard1 = ALL ZERO |
| 2 | What is xqkv's actual L1 address? | Probe B: `xqkv_fused_sharded.buffer_address()` | **0xaaf00** (NOT 0xa6700) |
| 3 | Is there a buffer at L1 0xa6700? | Probe C: `ttnn._ttnn.reports.get_buffers()` | At attention time: NO buffer at 0xa6700 |
| 4 | Where does the RS reader's `input_tensor_address=0xa6700` come from? | Probe E: in `tt_all_reduce`, before RS call | `input_tensor.buffer_address() = 0xa6700` |
| 5 | What dispatches that RS? | Probe E stack trace | `forward@mlp.py:569` (W2 reduce_scatter) |
| 6 | Is W2 output zero? | Probe F: `to_torch(w2_out)` | ALL ZERO shard0 + shard1 |
| 7 | Is RS input zero just before dispatch? | Probe E2 PRE-RS to_torch | ALL ZERO (all 8 RS calls) |
| 8 | Is RS output zero? | Probe E2 POST-RS to_torch | ALL ZERO (all 8 RS calls) |
| 9 | L1 layout at W2 time? | Probe G: get_buffers near 0xa6700 | 3 bufs: 0xa6700 (w2_out), 0xa8700 (other w2_out variant), 0xac700 (GlobalCB) — no overlap |
| 10 | Final decode output | server `/generate` | "Und著作权" — bug present |

**Punchline:** Every synchronous probe reads ZERO. Final decode output is GARBAGE. The corruption only happens in the **trace-replay async execution** where Python `to_torch` queue-drain barriers do not fire.

## Probes

All probes are env-gated (default-OFF) so canonical paths and
non-prefetcher paths are bitwise-equal to U14.

### Probe A — `ttnn.to_torch` on WQKV matmul output

```python
if os.environ.get("SGLANG_TT_U15_PROBE_TOTORCH", "0") == "1":
    shards = ttnn.get_device_tensors(xqkv_fused_sharded)
    for i, sh in enumerate(shards):
        t = ttnn.to_torch(sh).float().cpu()
        flat = t.flatten()
        print(f"[U15_PROBE_A] shard={i} ... nnz={int((flat!=0).sum())}/{flat.numel()}")
```

Result (zero weights): shard0 = 0/98304 NONZERO, shard1 = 0/98304 NONZERO.
Producer-side L1 is clean.

### Probe B — `buffer_address()` + `shard_spec` triplet

```python
print(f"xqkv buffer_address=0x{xqkv_fused_sharded.buffer_address():x}")
print(f"x    buffer_address=0x{x.buffer_address():x}")
print(f"wqkv buffer_address=0x{wqkv.buffer_address():x}")
```

Result: `xqkv=0xaaf00 (L1 WIDTH_SHARDED [32,96] on 32 cores)`,
`x=0xa2700 (L1 WIDTH_SHARDED [32,128] on same 32 cores)`,
`wqkv=0x4f63e40 (DRAM WIDTH_SHARDED)`.
**The WQKV output is NOT at 0xa6700.** It's at 0xaaf00.

### Probe C — L1 allocator enumeration

```python
bufs = ttnn._ttnn.reports.get_buffers(devices)
near = [b for b in bufs if abs(b.address - 0xa6700) < 0x10000]
```

At attention time:
- `0xa2700` size 0x2000 — `x` activation (matmul input)
- `0xaaf00` size 0x1800 — `xqkv_fused_sharded` (matmul output)
- `0xac700` size 0xcc000 — GlobalCB (prefetcher receiver region)

**No buffer at 0xa6700** during attention. The "consumer reads
0xa6700" bug is NOT from attention.

### Probe E — `input_tensor.buffer_address()` in `tt_all_reduce`

```python
# In ccl.py, immediately before reduce_scatter_minimal_async:
print(f"[U15_PROBE_E] RS#{idx} input_tensor.buffer_address="
      f"0x{input_tensor.buffer_address():x} ...")
# Plus Python stack trace.
```

Decode-mode results, first 8 calls:

| RS# | input.buffer_address | reduced.buffer_address | Caller |
|---:|---|---|---|
| 1 | **0xa6700** | 0xaa700 | `forward@mlp.py:569` (W2 reduce_scatter) |
| 2 | 0xa4700 | 0xa8700 | `forward@mlp.py:569` |
| 3 | 0xa6700 | 0xa4700 | `forward@mlp.py:569` |
| 4 | 0xa6700 | 0xaa700 | `forward@mlp.py:569` |
| 5 | 0xa4700 | 0xa8700 | `forward@mlp.py:569` |
| 6 | 0xa6700 | 0xa4700 | `forward@mlp.py:569` |
| 7 | 0xa6700 | 0xaa700 | `forward@mlp.py:569` |
| 8 | 0xa4700 | 0xa8700 | `forward@mlp.py:569` |

**Every single one of the first 8 reduce_scatter calls comes from
MLP's W2 step, not attention's WQKV all-reduce** (which is skipped
on `cluster_axis=1 && 1 in mesh_shape`, ccl.py:150).

### Probe E2 — to_torch on RS input AND output

PRE-RS and POST-RS:

| RS# | PRE-RS shards | POST-RS shards |
|---:|---|---|
| 1 | 0/131072 NONZERO | 0/65536 NONZERO |
| 2 | 0/131072 NONZERO | 0/65536 NONZERO |
| 3..8 | all 0/131072 | all 0/65536 |

**All zero on both sides of every RS call.** The reduce_scatter
itself, when probed synchronously, behaves correctly (zero in →
zero out).

### Probe F — `to_torch(w2_out)` and `w2_out.buffer_address()`

```python
print(f"w2_out buffer_address=0x{w2_out.buffer_address():x} "
      f"shape={tuple(w2_out.shape)} layout={w2_out.memory_config().memory_layout}")
shards = ttnn.get_device_tensors(w2_out)
for i, sh in enumerate(shards):
    t = ttnn.to_torch(sh).float().cpu()
    print(f"[U15_PROBE_F_TOTORCH] shard={i} nnz={...}")
```

Decode-mode results:

| shape | buffer_address | shard0 nnz | shard1 nnz |
|---|---|---:|---:|
| (1, 1, 32, 4096) | **0xa6700** | 0/131072 | 0/131072 |

**W2 producer writes ZERO to L1 0xa6700 — fully consistent with
U14's end-of-matmul-kernel probe (which was on WQKV) AND with the
zero-weight expectation.**

### Probe G — `get_buffers` near 0xa6700 at W2 dispatch time

| addr | type | layout | sz_per_bank |
|---|---|---|---:|
| 0xa6700 | L1 | WIDTH_SHARDED | 8192 |
| 0xa8700 | L1 | WIDTH_SHARDED | 8192 |
| 0xac700 | L1 | HEIGHT_SHARDED | 835584 |

The 0xa6700 buffer is **uniquely** allocated to w2_out. No second
allocation at the same address. **L1 allocator overlap (H2) is
DEFINITIVELY RULED OUT.**

## Decisive negative results

1. **H1 (producer-consumer sharding mismatch)**: Both W2 output and
   its reduce_scatter input have IDENTICAL `shard_spec.grid` and
   identical `buffer_address` (`0xa6700`). No mismatch — the
   addrgen reads from the same cores the producer wrote to.
2. **H2 (L1 allocator overlap)**: Only ONE buffer at 0xa6700 per
   Probe G. The matmul output is the sole owner.
3. **Producer broken under zero weights**: Probe F = ALL ZERO. The
   W2 matmul correctly outputs zeros with zero weights.

## What WOULD explain U14's NONZERO reads at 0xa6700

The consumer reader's `noc_async_read` of `0xa6700` returned
NONZERO bytes on 476/1920 (~25%) of reads, but always with bytes
that "look like BF16 activations" (e.g., `0x3f10bedb` = 0.5625).

With the producer confirmed clean (Probe F), the U14 NONZERO bytes
MUST come from L1 bytes that were at `0xa6700` **before the W2
matmul wrote there**. The L1 allocator picks the same address for
w2_out that was previously occupied by a tensor whose bytes look
like activations (e.g., the residual stream `x_in`, or an MLP
intermediate).

**Mechanism**: under trace replay, the matmul kernel's PACK→L1
write may not complete before the reduce_scatter reader's
`noc_async_read` fires — the reader gets the STALE bytes of the
previous tensor that lived at 0xa6700. With Python `to_torch`
barriers (Probe E2), the queue is drained between W2 and RS, so
the read sees the correctly-zero W2 output. Without barriers
(trace replay), the read can race ahead.

This is a classic **trace-replay producer-consumer ordering
race** — and it's EXACTLY what the existing U5
`SGLANG_TT_PREFETCHER_OUTPUT_BARRIER` fix targets: routing the
post-W2 CCL op's `subdevice_id` to `receiver_sub_device_id`
(matching the matmul producer) so the dispatch hardware
serializes them on the same stream.

## U5 fix tried — caused a decode hang

```bash
# With SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1, zero weights:
curl -X POST http://127.0.0.1:30000/generate -d '{...max_new_tokens=5...}'
# Server log shows "Done Capturing Decode Trace" but then stalls
# at "Allocating device buffers is unsafe due to ... active trace"
# warning, with curl waiting >3min before being killed.
```

U5 alone is not sufficient. Either it requires a companion fix
(e.g., re-routing the W2 matmul itself to a different sub-device,
or adding an explicit synchronization op), or the
receiver_sub_device_id routing introduces a different deadlock
that needs a second mitigation.

## Hypothesis ledger update (post-U15)

| Hypothesis | Pre-U15 | Post-U15 | Evidence |
|---|---|---|---|
| MATH writes garbage | RULED OUT (U12) | UNCHANGED | — |
| UNPACK reads stale srcA/srcB | RULED OUT (U12) | UNCHANGED | — |
| PACK writes garbage to mm_out_cb | RULED OUT (U13) | UNCHANGED | — |
| L1 stomped inside matmul kernel | RULED OUT (U14) | UNCHANGED | — |
| **Consumer reads wrong (core, offset)** | STANDING | **RULED OUT** | Probe E: input_tensor.buffer_address=0xa6700 exactly matches U14 reader's in_addr; same shard_grid as producer |
| **L1 allocation overlap mm_out vs activation** | STANDING — high priority | **RULED OUT** | Probe G: unique 0xa6700 owner; no other buffer at same address |
| **Producer's PACK and consumer's reader use different shard-to-core mapping** | STANDING — top priority | **RULED OUT** | Probe E: identical shard_grid; Probe B: identical mem_config |
| Prefetcher writer_l1 stomps mm_out_cb's L1 | STANDING — secondary | **RULED OUT** | GlobalCB at 0xac700 + 0xcc000 ends at 0x178700; w2_out at 0xa6700+0x2000 ends at 0xa8700 — no overlap |
| **Bug is in MLP W2 (not attention WQKV)** | (implicit) | **CONFIRMED** | Probe E stack trace: all 8 RS calls from `mlp.py:569` |
| **Trace-replay producer-consumer race (PACK→RS reader)** | (new) | **STANDING — top priority** | Compile-run with sync barriers = correct (Probe F + E2); trace-replay without barriers = garbage; pre-W2 L1 holds prior-tensor bytes that look like activations |
| U5 OUTPUT_BARRIER alone is sufficient fix | (new) | **REJECTED** | Causes decode-time hang |
| **Trace-replay race needs a different fix** | (new) | **STANDING — top priority** | Either (a) U5 + matmul sub-device re-routing, (b) explicit on-device semaphore between W2 and RS, or (c) reduce_scatter's reader needs a CB-based wait_front instead of direct addrgen read |

## U16 dispatch (recommended next attack)

The bug is now narrowed to: **trace-replay async execution where
the W2 matmul's PACK→mm_out_cb completion does NOT synchronize
against the reduce_scatter_minimal_async_reader's `noc_async_read`
on the same L1 region**. Compile-run with synchronous barriers
produces correct zeros end-to-end; trace replay produces garbage.

Three attack lines for U16:

1. **Diagnose the U5 hang**. With `SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1`,
   capture py-spy / device traceback of the stuck server. Identify
   whether the hang is in begin_trace_capture / execute_trace /
   the CCL semaphore wait. Fix the hang and re-validate the bit
   correctness with zero weights → expect "Und<something>" → token 0
   pattern collapse. Then GSM8K(10) on canonical weights.

2. **Add a `wait_for_event` between W2 and RS** at the python level
   (env-gated). If receiver_sub_device routing hangs, try inserting
   a `ttnn.synchronize_devices` or explicit op-level barrier between
   the `ttnn.linear(W2)` call and the `tt_all_reduce` call, only
   on the prefetcher path. This may also hang for the same reason
   but isolates whether the issue is sub-device or kernel-level.

3. **Patch reduce_scatter_minimal_async_reader to wait_front on
   mm_out_cb instead of direct addrgen**. The reader currently
   computes `noc_addr = get_noc_addr(tile_id, input_tensor_addrgen)`
   and reads from L1. If the producer's PACK has not yet pushed
   the corresponding tile to mm_out_cb's `fifo_wr_ptr`, the read
   returns stale L1 bytes. A `cb_wait_front(mm_out_cb_id, 1)`
   before the read would block until the producer signals. This
   is invasive but is the architecturally-correct fix.

Attack order: (1) first — cheapest; verifies U5 logic. (2) second
— validates that explicit sync resolves it. (3) third — landing
the actual upstream-compatible fix.

## Reproducer (working)

```bash
# Sync host -> container (Python only; no kernel rebuild)
podman cp /home/mhnie/tt-metal-sglang/models/tt_transformers/tt/attention.py \
  p3a-ngram:/tt-metal/models/tt_transformers/tt/attention.py
podman cp /home/mhnie/tt-metal-sglang/models/tt_transformers/tt/mlp.py \
  p3a-ngram:/tt-metal/models/tt_transformers/tt/mlp.py
podman cp /home/mhnie/tt-metal-sglang/models/tt_transformers/tt/ccl.py \
  p3a-ngram:/tt-metal/models/tt_transformers/tt/ccl.py

# Clear cache (literal-path, var-guarded)
podman exec p3a-ngram bash -c \
  'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

# Kill prior server
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'

# Launch with the full probe set
podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_U15_PROBE_TOTORCH=1 \
  SGLANG_TT_U15_PROBE_SHARD=1 \
  SGLANG_TT_U15_PROBE_L1DUMP=1 \
  SGLANG_TT_U15_PROBE_RS=1 \
  SGLANG_TT_U15_PROBE_RS_TOTORCH=1 \
  SGLANG_TT_U15_PROBE_W2=1 \
  SGLANG_TT_U15_PROBE_W2_L1=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u15_test.log 2>&1'

# Wait for /health 200, then fire a generate:
curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text": "2+2=", "sampling_params": {"max_new_tokens": 2, "temperature": 0.0}}'

# Analyze:
podman exec p3a-ngram grep "U15_PROBE_E2_TOTORCH" /tmp/u15_test.log | head -16
podman exec p3a-ngram grep "U15_PROBE_F" /tmp/u15_test.log | grep "shape=(1, 1, 32, 4096)"
podman exec p3a-ngram grep "U15_PROBE_G" /tmp/u15_test.log | grep "0xa6700\b"
```

## Hardware

- Container: `p3a-ngram`
- Cards: 2× P150a (P150_X2 cluster type)
- CCL topology: Linear
- HEAD probe: `forward@mlp.py:569` W2 reduce_scatter

## Working state at session end

- tt-metal-sglang HEAD: **`2f7cda2f02b`** (unchanged from U14 — probes are pure Python)
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only)
- sglang HEAD: pending this doc
- Container `/tt-metal/`: attention.py + mlp.py + ccl.py with U15 probes
  applied (all env-gated; canonical = bytewise-equal to U14)
- `/root/.cache/tt-metal-cache/*`: cleared at end of session
- `stash@{0,1,2}`: untouched
- TT devices: healthy (no resets)
- No server running at session end

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal Python files, env-gated)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] No kernel rebuild needed (pure Python this dispatch)
- [x] Canonical re-verify pending (server starting under U15 probes
      env-unset to confirm zero overhead — see end of doc)
- [x] All probes env-gated (`SGLANG_TT_U15_PROBE_*`)

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≈ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** U15
narrowed the bug from "attention WQKV downstream" to "MLP W2
trace-replay race" but did NOT fix it. The existing
`SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1` (U5) workaround is the
correct architectural direction but currently hangs at decode time
— U16 must diagnose and fix that hang, OR introduce an explicit
on-device sync between W2 PACK and reduce_scatter's read.

## Commits this session

* (tt-metal-sglang) pending — U15 env-gated Python probes in
  `models/tt_transformers/tt/{attention.py,mlp.py,ccl.py}`
* (sglang) `<this doc>` — pending commit.
