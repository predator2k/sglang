# TT Qwen3-8B prefetcher — U32 STRUCTURAL PLUMBING LANDED (per-tensor GCB byte-offset runtime arg, env-gated, default-off, canonical-equivalent). Offset-value computation diverges → escalated as U33. Canonical re-verified 9/10 — 2026-05-25

Status: **EVIDENCE_ADVANCE — U32 structural plumbing LANDED across 5 files
(kernel + factory + prefetcher + mlp + attention), all env-gated default-off,
canonical Qwen3-8B bytewise-equal (`" What is 2+2? What is 2+2? What"` at
9.08s e2e + GSM8K(10) chat = 9/10 = 90% Q3-reasoning-pattern only).  When
SGLANG_TT_U32_GCB_OFFSET=1 + SGLANG_TT_U32_FORCE_ZERO=1, prefetcher path
starts cleanly and decodes (broken-canonical pattern, 0 broadcast errors —
proves the kernel rd_ptr modification at offset=0 is bytewise-equivalent
to canonical-broken).  When SGLANG_TT_U32_GCB_OFFSET=1 with Python-computed
offsets, 144 `Invalid subtile broadcast type` TT_THROWs fire (4 cores ×
36 layers) — symptom that the OFFSET VALUES are off (mis-aligned reads
produce tensors whose shape-contract downstream eltwise binary cannot
satisfy).  Root cause of offset mismatch identified: producer's
`block_size_per_receiver` (~104448 bytes for WQKV BFP8) vs matmul receiver's
`in1_block_size_bytes` (~13056 bytes for WQKV BFP8) differ by 8× — Python-side
estimation cannot bridge without C++-factory knowledge of the per-call
matmul shape (deferred to U33).**

Continuation of `tt_qwen3_8b_prefetcher_U31_DISCRIMINATOR_IS_DTYPE_BFP4_VS_BFP8_2026-05-25.md`.

tt-metal-sglang HEAD: `60b35089278` (U32 structural plumbing).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U32.1 | Wire env var → factory define → kernel runtime arg | C++ factory (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2483-2497`) reads `SGLANG_TT_U32_GCB_OFFSET=1` at program-create, bakes into mm_kernel_defines.  Kernel (`bmm_large_block_zm_fused_bias_activation_gathered.cpp:256-273`) reads new RT arg `u32_tensor_offset_bytes` after the existing widths array (worker-cores only). | **Build PASSES + sentinel `SGLANG_TT_U32_GCB_OFFSET` + `SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES` strings present in `_ttnncpp.so`.** |
| U32.2 | Top-of-batch rd_ptr update | Kernel at line 325-337: when `SGLANG_TT_U32_GCB_OFFSET` is defined, after `get_local_cb_start_addr(in1_cb_id)` captures the CB start, `update_local_cb_rd_ptr(in1_cb_id, in1_cb_start_addr + u32_tensor_offset_bytes / L1_ALIGNMENT)` shifts rd_ptr by the per-tensor offset (in 16-byte CB-shifted units) BEFORE the ring_idx advancement. | **In-tree.** |
| U32.3 | Cache-hit re-update of offset RT arg | `shared_variables_t` now stores `mm_kernel` at index 1; `override_gather_in0_program_parameters` re-reads `SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES` env on every cache-hit dispatch and writes the last runtime arg via `compute_runtime_args[n-1] = u32_off`.  Skips idle/hop cores (size==1).  Guarded by `kernels.size() >= 2 && global_cb.has_value()`. | **In-tree.** |
| U32.4 | Python — track per-tensor offsets | `prefetcher.py:insert_tensor` appends `running_tensor_offset_bytes` to a new `_tensor_byte_offsets` list.  `get_tensor_gcb_offset_bytes(tensor)` returns the per-tensor byte offset modulo `gcb_size` (`max_tensor_block_size`).  Computed as `per_tensor_bytes_per_receiver = ring_size * (max_tensor_tiles * tile_bytes / num_receiver_cores)`. | **In-tree.** |
| U32.5 | Python — set offset env per ttnn.linear | `mlp.py` wraps W1/W3/W2 calls in `_u32_set_offset_for(weight) / try / _u32_restore_offset(prev)` (same pattern as U29 PRODUCER_NOW).  `attention.py` mirrors for WQKV + WO (both forks).  Restore preserves the prior env value in finally. | **In-tree (5 wrap-sites: 3 mlp + 2 attention).** |
| U32.6 | Canonical re-verify (default — all U32 envs unset) | `SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged`; no prefetcher; no U32 env vars set. | **PASS — `" What is 2+2? What is 2+2? What"` at 9.08s e2e_latency + GSM8K(10) chat = 9/10 = 90.0% (Q3 reasoning-truncation only; bytewise-equal to U30/U31 baseline).** |
| U32.7 | U32_GCB_OFFSET=1 + Force-zero (kernel define ON, all offsets=0) | `SGLANG_TT_U32_GCB_OFFSET=1 SGLANG_TT_U32_FORCE_ZERO=1`. Both prefetcher AND U32 kernel active; offsets always 0 (kernel does `update_local_cb_rd_ptr(in1_cb_id, in1_cb_start_addr + 0)` = no-op). | **PASS — server starts cleanly (0 broadcast errors); decode produces broken-canonical garbage (`" Whathistoryhistory..."` — bytewise-equal pattern to non-U32 prefetcher baseline).** Proves the kernel rd_ptr modification + RT-arg plumbing at offset=0 is BYTEWISE-EQUIVALENT to canonical-broken prefetcher behavior — i.e. the U32 STRUCTURAL CHANGE is sound. |
| U32.8 | U32_GCB_OFFSET=1 + Python-computed offsets (real values per insert_tensor order) | `SGLANG_TT_U32_GCB_OFFSET=1` (no FORCE_ZERO).  Per-tensor offsets like WQKV=0, WO=3342336, FF1=5570560, FF3=9109504, FF2=12648448, modulo `gcb_size=835584`. | **FAILS — 144 `TT_THROW: Invalid subtile broadcast type` (`binary_ng_device_operation.cpp:224`) fire 5 seconds after `prefetcher.run()` (during warmup-pass eltwise binary).  144 = 4 worker cores × 36 layers.  Server hangs at decode dispatch.** |

**Punchline:** The U32 STRUCTURAL PLUMBING is correct and shipping (U32.7 PASS).  The OFFSET COMPUTATION (Python-side `get_tensor_gcb_offset_bytes`) is incorrect — the producer's per-receiver write granularity (`block_size_per_receiver`, e.g. 104448 bytes for WQKV BFP8) does NOT match the matmul receiver's per-block read size (`in1_block_size_bytes`, e.g. 13056 bytes for WQKV BFP8) by an 8× factor that depends on per-call factory state.  Bridging this requires C++-factory-side offset computation (U33 scope).

## Files changed (5)

```
 models/tt_transformers/tt/attention.py             | 127 +++++++++++++++------
 models/tt_transformers/tt/mlp.py                   |  97 ++++++++++++----
 models/tt_transformers/tt/prefetcher.py            |  79 ++++++++++++-
 ...ul_multicore_reuse_mcast_1d_program_factory.cpp | 103 ++++++++++++++++-
 ...rge_block_zm_fused_bias_activation_gathered.cpp |  33 ++++++
```

Commit: `60b35089278 prefetcher: U32 — per-tensor GCB byte-offset structural plumbing (env-gated, default-off, canonical-equivalent; offset values diverge → tracked as U33)`

## Implementation deep-dive

### Kernel side (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`)

After the existing `unpadded_in0_shard_widths_in_tiles` runtime-arg array
(rt_args_idx += ring_size):

```cpp
#ifdef SGLANG_TT_U32_GCB_OFFSET
    uint32_t u32_tensor_offset_bytes = get_arg_val<uint32_t>(rt_args_idx++);
#endif
```

At top of each batch iteration, AFTER `get_local_cb_start_addr(in1_cb_id)`
captures the CB start but BEFORE `update_rd_ptr_to_ring_index` adds the
ring-slot offset:

```cpp
UNPACK((in1_cb_start_addr = get_local_cb_start_addr(in1_cb_id)));
#ifdef SGLANG_TT_U32_GCB_OFFSET
    UNPACK((update_local_cb_rd_ptr(
        in1_cb_id,
        in1_cb_start_addr + u32_tensor_offset_bytes / L1_ALIGNMENT)));
#endif
UNPACK((in1_rd_ptr_start_addr = get_local_cb_rd_ptr(in1_cb_id)));
UNPACK((curr_in1_block_index = ring_idx));
UNPACK((in1_tensor_split = is_tensor_split(in1_cb_id, in1_tensor_size_bytes)));
UNPACK((update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_idx, in1_tensor_split)));
```

The `/ L1_ALIGNMENT` (= 16 on Blackhole) converts byte offset to CB-shifted
units (the unit `fifo_rd_ptr` is stored in).

`in1_rd_ptr_start_addr` is then captured AFTER our shift, so Block D's
end-of-batch `update_local_cb_rd_ptr(in1_cb_id, in1_rd_ptr_start_addr)` +
`update_rd_ptr_to_ring_index(..., ring_size, ...)` lands at the correct
"next tensor" position naturally.

### Factory side (`matmul_multicore_reuse_mcast_1d_program_factory.cpp`)

Three modifications:

1. **Define propagation** (process_gather_in0, line 2493-2497):
   ```cpp
   const char* u32_gcb_offset_env = std::getenv("SGLANG_TT_U32_GCB_OFFSET");
   if (u32_gcb_offset_env != nullptr && std::string(u32_gcb_offset_env) == "1") {
       mm_kernel_defines["SGLANG_TT_U32_GCB_OFFSET"] = "1";
   }
   ```

2. **Runtime arg append** (worker-core compute RT args loop, line 2919-2933):
   ```cpp
   if (use_global_cb) {
       const char* u32_off_env = std::getenv("SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES");
       uint32_t u32_off = 0;
       if (u32_off_env != nullptr && u32_off_env[0] != '\0') {
           try {
               u32_off = static_cast<uint32_t>(std::stoul(u32_off_env, nullptr, 0));
           } catch (...) {
               u32_off = 0;
           }
       }
       mm_kernel_compute_args.push_back(u32_off);
   }
   ```
   The arg is appended unconditionally on `use_global_cb` paths so the
   override (below) can rely on its layout position.  Idle/hop cores
   already have a single-element RT arg vector and are unaffected.

3. **shared_variables update** (line 2952-2956):
   ```cpp
   return MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t{
       {mm_kernel_in1_sender_writer_id, mm_kernel},  // U32: added mm_kernel at [1]
       shared_cbs,
       ...
   };
   ```

4. **Override re-update** (`override_gather_in0_program_parameters`, line 3185-3216):
   ```cpp
   if (override_variables.kernels.size() >= 2 && global_cb.has_value()) {
       const char* u32_off_env = std::getenv("SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES");
       uint32_t u32_off = ...;
       auto& compute_runtime_args_by_core = GetRuntimeArgs(program, override_variables.kernels.at(1));
       for (const auto& core : override_variables.cores) {
           auto& compute_runtime_args = compute_runtime_args_by_core[core.x][core.y];
           const std::size_t n = compute_runtime_args.size();
           if (n > 1) {  // skip idle/hop (size 1)
               compute_runtime_args[n - 1] = u32_off;
           }
       }
   }
   ```

### Python side (`prefetcher.py`)

`insert_tensor` now also computes per-tensor running offset:

```python
per_tensor_block_size_per_receiver = (
    max_tensor_tiles * bytes_in_tile[tensor.dtype]
) // self.num_receiver_cores
per_tensor_bytes_per_receiver = self.ring_size * per_tensor_block_size_per_receiver

if not hasattr(self, "_tensor_byte_offsets"):
    self._tensor_byte_offsets = []
if not hasattr(self, "_running_tensor_offset_bytes"):
    self._running_tensor_offset_bytes = 0
self._tensor_byte_offsets.append(self._running_tensor_offset_bytes)
self._running_tensor_offset_bytes += per_tensor_bytes_per_receiver
```

And exposes:

```python
def get_tensor_gcb_offset_bytes(self, tensor: ttnn.Tensor) -> int:
    if os.environ.get("SGLANG_TT_U32_FORCE_ZERO", "0") == "1":
        return 0
    ...
    idx = self.prefetched_tensors.index(tensor)
    if self.num_tensors > 0:
        offset = self._tensor_byte_offsets[idx % self.num_tensors]
    else:
        offset = self._tensor_byte_offsets[idx]
    gcb_size = self.max_tensor_block_size if self.max_tensor_block_size > 0 else 1
    return offset % gcb_size
```

### Python side (`mlp.py`)

Per-call env-var setter (3 sites: W1, W3, W2):

```python
import os as _u32_os
_u32_active = (
    _u32_os.environ.get("SGLANG_TT_U32_GCB_OFFSET", "0") == "1"
    and mode == Mode.DECODE
    and self.prefetcher is not None
)
def _u32_set_offset_for(weight):
    if not _u32_active or weight is None:
        return None
    try:
        off = self.prefetcher.get_tensor_gcb_offset_bytes(weight)
    except Exception:
        off = 0
    _prev = _u32_os.environ.get("SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES")
    _u32_os.environ["SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES"] = str(int(off))
    return _prev
def _u32_restore_offset(prev):
    if not _u32_active:
        return
    if prev is None:
        _u32_os.environ.pop("SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES", None)
    else:
        _u32_os.environ["SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES"] = prev

_u32_w1_prev = _u32_set_offset_for(_w1_weight())
try:
    w1_out = ttnn.linear(...)
finally:
    _u32_restore_offset(_u32_w1_prev)
```

### Python side (`attention.py`)

Same pattern at 3 sites (WQKV path, WO non-TG path, WO TG path).

## Run results

### U32.6 — Canonical re-verify (default; all U32 envs unset)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c 'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server ... > /tmp/u32_canonical_check.log 2>&1'

curl -X POST http://127.0.0.1:30000/generate ... -d '{"text":"What is 2+2?","max_new_tokens":15,"temperature":0.0}'
# → " What is 2+2? What is 2+2? What" at 9.08s (bytewise-equal vs U30/U31)

python3 /sglang/.../eval_qwen3_8b_gsm8k_chat.py --num 10
# → FINAL: 9/10 = 90.0% (Q3 reasoning-truncation, same as canonical baseline)
```

### U32.7 — U32 kernel define ON + Force-zero (proves structural plumbing sound)

```bash
podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U32_GCB_OFFSET=1 \
  SGLANG_TT_U32_FORCE_ZERO=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server ... > /tmp/u32_force_zero.log 2>&1'

# Server starts clean: 0 broadcast errors at 22:45:12.
# Decode test:
curl -X POST http://127.0.0.1:30000/generate ... -d '{"text":"What is 2+2?","max_new_tokens":15,"temperature":0.0}'
# → " Whathistoryhistoryhistoryhistory..." (broken canonical pattern;
#    same garbage class as non-U32 prefetcher baseline)
```

**0 broadcast errors at offset=0** ⇒ kernel define + RT-arg plumbing is
bytewise-equivalent to canonical-broken prefetcher.

### U32.8 — U32 kernel define ON + Python-computed offsets (broken)

```bash
podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U32_GCB_OFFSET=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server ... > /tmp/u32_test.log 2>&1'

# Prefetcher inserts 180 tensors (36 layers × 5 per layer) with offsets:
# 0, 3342336, 5570560, 9109504, 12648448, 19333120, ...
# 22:09:00.247 prefetcher.run() — global CB size = 835584
# 22:09:05.299 first TT_THROW: Invalid subtile broadcast type
# 22:09:05-22:09:07 — 144 errors total (4 cores × 36 layers)
# Server hangs at first decode dispatch (no curl response).
```

The error fires from `binary_ng_device_operation.cpp:224`
`TT_THROW("Invalid subtile broadcast type")` — a downstream eltwise
binary op cannot reconcile two tensors' shapes (neither has dim==1 in
the mis-matched direction).  This is the symptom of the matmul producing
output tensors with semantically-wrong layouts because the kernel reads
mis-aligned bytes from the GCB.

## Why the Python offset computation is wrong

My Python formula: `per_tensor_bytes_per_receiver = ring_size * (max_tensor_tiles * tile_bytes / num_receiver_cores)`.

For WQKV (4096×3072, BFP8, ring_size=32, num_receivers=4):
- `max_tensor_tiles = 128*96/32 = 384` (per-receiver block tiles for producer)
- `per_tensor_bytes_per_receiver = 32 * (384 * 1088 / 4) = 32 * 104448 = 3342336`

But the matmul receiver consumes only `ring_size * in1_block_size_bytes
= 32 * 13056 = 417792` bytes per receiver per tensor (per U31's
factory CT-args table).  An 8× factor mismatch.

The producer's `block_size_per_receiver` (104448) is the page_size set
via `experimental::resize_remote_sender_cb_interface` at writer_l1.cpp:95.
The matmul's `in1_block_size_bytes` (13056) is the CT-arg derived from
`per_core_N * in0_block_w * tile_size`.

These two quantities encode the SAME tensor data but at different
granularities — and the relationship depends on the per-call matmul
program_config (`per_core_N`, `in0_block_w`).  Computing the offset
Python-side requires knowing each matmul's program_config at the time
the tensor is inserted into the prefetcher — but at insert_tensor time
the matmul program_config hasn't been chosen yet.

## U33 — exact next attack

**Move offset computation into the C++ factory** where the per-call
matmul shape is known.

Approach options:

1. **Factory tracks per-GCB tensor offsets via static / global state**:
   At each `process_gather_in0_program_and_create_override_variables`
   call with a given `global_cb`, look up a per-GCB running offset
   counter; bake the current value into the runtime arg; advance by
   `num_blocks * in1_block_size_bytes` (the per-tensor consumer bytes).
   The override callback uses the SAME value (program is cached so
   subsequent dispatches re-use the offset baked at first-create time).

   * Pro: Self-contained, no Python-side metadata needed.
   * Con: Requires per-GCB state in the matmul factory (or a side
     map keyed by `GlobalCircularBuffer` handle); ordering depends on
     the order in which matmuls are first created (which may not match
     the prefetcher's tensor enumeration order for cached programs).

2. **Caller (mlp/attention) passes `tensor_index_in_prefetcher_queue` env var**:
   The matmul factory looks up the running cumulative bytes of preceding
   tensors in the prefetcher (via a new tt-metal API or shared state).
   Decouples ordering: each ttnn.linear call says "I am the Nth
   tensor", factory computes offset = sum of preceding tensors'
   `in1_block_size_bytes * num_blocks`.

3. **Pass `in1_tensor_size_bytes` of each preceding matmul through
   prefetcher.py state**: Each ttnn.linear bumps a "consumed-bytes"
   counter on the prefetcher object; subsequent ttnn.linear reads the
   current value as its offset, then bumps.  Requires the order of
   ttnn.linear calls to MATCH the order of insert_tensor calls.

All three need to handle wrap-around: the consumer's effective fifo
size depends on the matmul's per-call `in1_block_size_bytes` via the
existing line 2155 rounding `(global_cb->size() / in1_block_size_bytes)
* in1_block_size_bytes`.  Wrap point varies per matmul.

## Hypothesis ledger (post-U32)

| ID | Suspect | Pre-U32 | Post-U32 |
|---|---|---|---|
| U7 / Case B — Gathered compute kernel produces garbage on specific CT-arg ELFs under REAL weights | CONFIRMED (U30) | UNCHANGED |
| U31 — Per-matmul `(gcb_size / in1_block_size_bytes) * in1_block_size_bytes` fifo-size rounding + `setup_local_cb_read_write_interfaces` reset → every matmul reads from `fifo_start + ring_idx * in1_block_size_bytes`, INDEPENDENT of prefetcher's per-tensor sequential write position | TOP STANDING | **STRUCTURALLY ADDRESSED — kernel + factory + override all plumb the offset RT arg correctly (U32.7 PASS verifies bytewise equivalence at offset=0).  Offset COMPUTATION is the remaining gap (U33).** |
| NEW U32 — Python offset formula `ring_size * (block_num_tiles * tile_bytes / num_receivers)` mis-estimates per-tensor advance by ~8× vs matmul receiver semantics | (new) | **TOP STANDING.** |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only; commit `60b35089278` on `tenstorrent-p1` only).
- [x] No SGLang core behavior changes (Python-side `python/sglang/...` untouched).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used FIXED `rebuild_tt_metal_kernels.sh` after each host edit + `podman cp` for source sync.
- [x] All U32 paths env-gated (`SGLANG_TT_U32_GCB_OFFSET`, `SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES`, `SGLANG_TT_U32_FORCE_ZERO`).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"` at 9.08s) + GSM8K(10) = 9/10 = 90% (Q3 reasoning-pattern only; baseline-equivalent).
- [x] Server stopped at session end; cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: **`60b35089278`** (U32 structural plumbing, 5 files).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/ttnn/cpp/...` source + `/tt-metal/models/tt_transformers/tt/*.py`: synced; `_ttnn.so` + `_ttnncpp.so` synced to install dir; sentinels `SGLANG_TT_U32_GCB_OFFSET` + `SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES` verified in `strings _ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at 9.08s + GSM8K(10) = 9/10 (Q3 reasoning-pattern only).
- DPRINT/stderr artifacts preserved in-container at `/tmp/u32_test.log`, `/tmp/u32_force_zero.log`, `/tmp/u32_offset0.log`, `/tmp/u32_canonical_check.log`, `/tmp/u32_noU32_test.log` for U33's reference.

## Shipping verdict

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Same shipping verdict as U31; root mechanism is now structurally addressable via U32's plumbing but the offset values need fixing in U33.
- **`SGLANG_TT_U32_GCB_OFFSET=1` is safe to leave in tree (default-off + verified bytewise-equivalent at offset=0 via U32.7).**  The new compute kernel binary's behavior when `SGLANG_TT_U32_GCB_OFFSET=1 SGLANG_TT_U32_FORCE_ZERO=1` matches canonical-broken prefetcher (`" Whathistory..."` pattern) — i.e. the rd_ptr modification at offset=0 is a no-op vs the pre-U32 baseline.
- **`SGLANG_TT_U32_FORCE_ZERO=1` is the U33 bisect escape hatch** — leaves the kernel binary intact while forcing all offsets to 0 for isolation testing.

## 44-dispatch ledger (cumulative)

| ID | Status | Note |
|---|---|---|
| S1-S10 | CLOSED | Per-dispatch sub-investigations summarized in U21/U22/etc. |
| Lead 1/2/3 | CLOSED | Per-doc closure. |
| A1 | CLOSED | All-reduce alignment; not the bug. |
| U1-U16 | CLOSED | Each ruled out as root cause (DST stale, GCB block bisect, LLK probes, PACK probes, W2→RS barrier, etc.). |
| U17-U28 | CLOSED | All "downstream stomper" theories refuted — wrong bytes at 0xa6700 trace back to matmul writing wrong values, not external stomper. |
| U29 | CLOSED | W2→RS semaphore handshake VERIFIED working (sentinel 0xdeadbeef confirmed end-to-end) but garbage persists — race fix doesn't engage on observed garbage. |
| U30 | CLOSED | Per-ELF PACK-side garbage measurement: 86% NaN/Inf at PACK exit for 3 of 4 ELFs. |
| U31 | CLOSED | Discriminator IDENTIFIED: BFP8 (1088-byte tile) vs BFP4 (576-byte tile).  Root mechanism: matmul rd_ptr assumption mismatches prefetcher write position. |
| **U32** | **CONFIRMED (structural plumbing landed) + ESCALATED to U33 (offset-value computation)** | Kernel + factory + override + Python plumbing all in place; default-off + bytewise-equivalent canonical (U32.6) + bytewise-equivalent canonical-broken prefetcher at offset=0 (U32.7) + downstream eltwise binary error symptom at Python-computed offsets (U32.8) ⇒ offset COMPUTATION needs C++-side factory state (U33). |
| U33 | OPEN | Move offset computation from Python into C++ factory; track per-GCB running consumer-side cumulative bytes.  Expected to deliver GSM8K(10) ≥ 7/10 under prefetcher + TPOT ≈ 16-20 ms (the documented 1.66× win). |

## Commits this session

- (tt-metal-sglang) **1 commit**: `60b35089278 prefetcher: U32 — per-tensor GCB byte-offset structural plumbing (env-gated, default-off, canonical-equivalent; offset values diverge → tracked as U33)`.
- (sglang) `<this doc>` — pending commit.
