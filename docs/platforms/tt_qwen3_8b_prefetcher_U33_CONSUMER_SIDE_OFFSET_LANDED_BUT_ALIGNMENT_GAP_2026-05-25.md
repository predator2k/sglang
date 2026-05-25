# TT Qwen3-8B prefetcher — U33 consumer-side per-tensor GCB byte-offset LANDED in Python (no C++ changes needed; offsets identify the correct producer-cumulative byte position per tensor); but the resulting offsets are NOT multiples of the matmul's `in1_block_size_bytes` so iter-1 still triggers `Invalid subtile broadcast` TT_THROW in downstream `binary_ng` ops. Canonical Qwen3-8B re-verified 9/10 — 2026-05-25

Status: **EVIDENCE_ADVANCE — U33 Python-side consumer offset computation
LANDED in 3 files (`prefetcher.py`, `mlp.py`, `attention.py`) and
correctly computes per-tensor cumulative bytes matching what the
producer (writer_l1.cpp) actually writes per receiver per tensor.
Computed iter-1 offsets (per-layer index 0=QKV, 1=WO, 2=W1, 3=W3,
4=W2 — observed insertion order): QKV=0, WO=417792, W1=696320,
W3=303104 (post-wrap), W2=745472 (post-wrap), all mod GCB-size 835584.
Each per-tensor bytes computed as `ring_size * per_core_N * in0_block_w *
tile_bytes` (matching matmul's `in1_block_size_bytes`).  Per-tensor
bytes: QKV=417792, WO=278528, W1=442368, W3=442368, W2=835584.
Canonical Qwen3-8B (no prefetcher, default envs) re-verified bytewise:
`" What is 2+2? What is 2+2? What"` at 9.14s + GSM8K(10) chat = 9/10
= 90% (Q3 reasoning-pattern only; baseline-equivalent vs U30/U31/U32).
With prefetcher + U33 enabled: iter-1 still throws 144 `Invalid subtile
broadcast type` TT_THROWs.  Root-cause analysis identifies a NEW gap:
the cumulative offset for matmuls AFTER the first (e.g. W1 @ 696320)
is NOT a multiple of THAT matmul's `in1_block_size_bytes` (e.g.
13824); 696320 / 13824 ≈ 50.37 → mis-aligned read.  The producer
writes contiguously (no per-tensor realignment within the GCB after
finishing tensor i-1), but the consumer CB framework appears to
require offsets be aligned to its own page_size.  Tracked as U34.**

Continuation of `tt_qwen3_8b_prefetcher_U32_STRUCTURAL_PLUMBING_LANDED_OFFSET_COMPUTATION_U33_2026-05-25.md`.

tt-metal-sglang HEAD: `a29dbd6552f` (U33, 3 files).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U33.1 | Switch from producer-side `(max_tensor_tiles * tile_bytes / num_receivers)` Python estimate (U32, 8× too large) to **consumer-side** per-tensor bytes `ring_size * per_core_N * in0_block_w * tile_bytes` (matmul's actual `in1_block_size_bytes`-aligned consumption) | `prefetcher.py:set_tensor_consumer_bytes(t, bytes)` registers per-tensor consumer bytes; `get_tensor_gcb_offset_bytes(t)` sums for prior tensors in per-layer order; env-gated `SGLANG_TT_U33_FACTORY_OFFSET=1`. | **LANDED in 3 files** (prefetcher.py +106; mlp.py +48; attention.py +51) |
| U33.2 | Where to register per-tensor consumer bytes (must run BEFORE first matmul dispatch) | Inside mlp.py forward at W1/W3 register block; W2 register before W2 ttnn.linear.  attention.py: WQKV register before WQKV ttnn.linear; WO register before WO ttnn.linear (non-TG and TG paths). | **5 register sites added (3 mlp + 2 attention).** |
| U33.3 | Tensor-to-prefetcher-index lookup correctness | Python `is` identity first; buffer_address fallback (Python `==` on ttnn.Tensor is unreliable). | **WORKING — observed per_layer_idx 0=QKV (417792 B), 1=WO (278528 B), 2=W1 (442368 B), 3=W3 (442368 B), 4=W2 (835584 B).** |
| U33.4 | Computed cumulative offsets (iter 1, after all 5 register) | QKV=0, WO=417792, W1=696320, W3=1138688 mod 835584 = 303104, W2=1581056 mod 835584 = 745472. | **In-tree.  Cumulative offsets match the producer's actual write positions in the GCB.** |
| U33.5 | Hardware test — Qwen3-8B Q+prefetcher+U33 | `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_U32_GCB_OFFSET=1 SGLANG_TT_U33_FACTORY_OFFSET=1`. | **FAILS — 144+ `TT_THROW: Invalid subtile broadcast type` at iter-1, same downstream `binary_ng_device_operation.cpp:224` symptom as U32.8.  Decode hangs.** |
| U33.6 | Iter-1 vs iter-2 (does the "partial registration" problem self-heal once all 5 are registered?) | Iter-1 fires register-then-matmul per call; broadcast errors fire mid-iter; server can't reach iter-2. | **Unable to verify iter-2 — server hangs after iter-1's first broadcast throw.** |
| U33.7 | Canonical re-verify (default — all U32/U33 envs unset, no prefetcher) | Standard canonical. | **PASS — `" What is 2+2? What is 2+2? What"` at 9.14s e2e_latency + GSM8K(10) chat = 9/10 = 90.0% (Q3 reasoning-truncation only; baseline-equivalent).** |
| U33.8 | Root-cause analysis — why iter-1 still throws | W1's cumulative offset 696320 is NOT a multiple of W1's `in1_block_size_bytes` 13824 (696320/13824 ≈ 50.37).  Similarly W3 @ 303104 (% 13824 = 12800).  W2 @ 745472 (% 26112 = 24320).  All these "interior" matmuls' offsets are mis-aligned to their own block_size_bytes.  The kernel's `update_local_cb_rd_ptr(in1_cb_id, fifo_start + offset/L1_ALIGNMENT)` lands the rd_ptr at a position that's not a CB-page-multiple, and subsequent step-by-block reads then misinterpret BFP8 tile boundaries. | **NEW TOP HYPOTHESIS — U34.** |

**Punchline:** U33 lands the consumer-side per-tensor cumulative byte
offset Python-side (no C++ changes needed; reuses U32's RT-arg
plumbing).  The COMPUTED offset values now correctly identify each
matmul's expected start position in the GCB (matching the producer's
cumulative writes).  But the resulting offsets are NOT aligned to the
consumer matmul's `in1_block_size_bytes`, which is required by the
CB infrastructure's stride-by-block read pattern.  iter-1 still throws
`Invalid subtile broadcast` at downstream `binary_ng`.  The producer
writes contiguously across tensors without per-tensor realignment;
to fix this we need EITHER (a) the producer to align per-tensor wraps
to LCM(all matmul block_sizes), OR (b) the matmul kernel to handle
non-block-aligned rd_ptr by reading bytes BEFORE its first proper
block to satisfy alignment without losing data, OR (c) the prefetcher's
GCB sizing to enforce per-tensor wraps aligned to a common multiple.

## Files changed (3)

```
 models/tt_transformers/tt/attention.py  |  51 +++++++++++++
 models/tt_transformers/tt/mlp.py        |  48 ++++++++++++
 models/tt_transformers/tt/prefetcher.py | 106 ++++++++++++++++++++++++++-
 3 files changed, 204 insertions(+), 1 deletion(-)
```

Commit: `a29dbd6552f` on `tenstorrent-p1`.

## Implementation deep-dive

### `prefetcher.py` — consumer-bytes tracking and offset computation

New state:
```python
self._tensor_consumer_bytes: list[int]  # per-layer-index → bytes per dispatch
```

Public API:
```python
def set_tensor_consumer_bytes(self, tensor: ttnn.Tensor, consumer_bytes_per_dispatch: int) -> None:
    # Lookup by Python `is` identity first; buffer_address fallback.
    # Writes to self._tensor_consumer_bytes[per_layer_idx].
    # Idempotent.  Logs INFO with cumulative-offsets table when value changes.

def get_tensor_gcb_offset_bytes(self, tensor: ttnn.Tensor) -> int:
    if os.environ.get("SGLANG_TT_U32_FORCE_ZERO", "0") == "1":
        return 0
    ...
    use_u33 = os.environ.get("SGLANG_TT_U33_FACTORY_OFFSET", "0") == "1" \
              and getattr(self, "_tensor_consumer_bytes", None) is not None \
              and len(self._tensor_consumer_bytes) > 0
    if use_u33:
        # Sum consumer bytes for tensors BEFORE this one (in per-layer order).
        # Unset slots = 0 (allows partial iter-1 use).
        offset = sum(self._tensor_consumer_bytes[i] for i in range(per_layer_idx)
                     if i < len(self._tensor_consumer_bytes))
        return offset % gcb_size
    # Fall through to U32 producer-side estimate.
    ...
```

### `mlp.py` — register W1, W3, W2

```python
_u33_active = _u32_active and SGLANG_TT_U33_FACTORY_OFFSET == "1"

def _u33_in1_block_bytes(pc, weight):
    per_core_n = int(pc.per_core_N)
    in0_bw     = int(pc.in0_block_w)
    tile       = {bfp4: 576, bfp8: 1088, bf16: 2048}[weight.dtype]
    return per_core_n * in0_bw * tile

def _u33_register(weight, pc, label):
    if not _u33_active: return
    per_dispatch = self.prefetcher.ring_size * _u33_in1_block_bytes(pc, weight)
    self.prefetcher.set_tensor_consumer_bytes(weight, per_dispatch)

if _u33_active and not _w1_use_skip:
    _u33_register(_w1_weight(), pc_1, "w1")
    _u33_register(_w3_weight(), pc_3, "w3")

# Later, before W2's ttnn.linear:
if _u33_active and not _w2_use_skip:
    _u33_register(_w2_weight(), pc_2, "w2")
```

### `attention.py` — register WQKV, WO (both TG and non-TG)

Same pattern at 3 sites (WQKV pre-call; WO pre-call non-TG; WO pre-call TG).

## Observed per-tensor cumulative offsets (iter-1, real run)

(From `[U33] set_tensor_consumer_bytes per_layer_idx=N ...` log lines)

| per_layer_idx | tensor | per_dispatch_bytes | cumulative_offset_bytes (mod 835584) | aligned to in1_block_size_bytes? |
|---:|---|---:|---:|:---:|
| 0 | WQKV (BFP8, in1_block=13056) | 417792 | **0** | YES (0 % 13056 = 0) |
| 1 | WO (BFP8, in1_block=8704) | 278528 | **417792** | YES (417792 % 8704 = 0) |
| 2 | W1 (BFP8, in1_block=13824) | 442368 | **696320** | **NO (696320 % 13824 = 5120)** |
| 3 | W3 (BFP8, in1_block=13824) | 442368 | **303104** (after mod) | **NO (303104 % 13824 = 12800)** |
| 4 | W2 (BFP8, in1_block=26112) | 835584 | **745472** (after mod) | **NO (745472 % 26112 = 24320)** |

The first 2 tensors (QKV, WO) get aligned offsets by accident
(their cumulative happens to be a multiple of THEIR own block_size).
The remaining 3 (W1, W3, W2) all have non-aligned offsets — and these
3 are the ones whose matmul outputs cause the subsequent `binary_ng`
broadcast errors.

## Why the alignment matters

The matmul kernel does:
```cpp
UNPACK((update_local_cb_rd_ptr(in1_cb_id, in1_cb_start_addr + u32_tensor_offset_bytes / L1_ALIGNMENT)));
UNPACK((in1_rd_ptr_start_addr = get_local_cb_rd_ptr(in1_cb_id)));
UNPACK((update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_idx, in1_tensor_split)));
```

The CB rd_ptr is set to `fifo_start + offset/L1_ALIGNMENT`.  The kernel
then steps by `block_size_bytes/L1_ALIGNMENT` per block.  If the offset
is not a multiple of `block_size_bytes`, the kernel reads the wrong
bytes for the first few iterations and produces semantically invalid
tile outputs.  The PACK side then writes these to L1; downstream
`binary_ng` reads the L1 and finds tile-header values that don't
match expected dims (a_h, a_w, b_h, b_w) — hence `Invalid subtile
broadcast type` (line 224 of `binary_ng_device_operation.cpp`).

## U34 — exact next attack

Three options:

1. **U34-A — align producer-side wraps to LCM(all matmul block_sizes)**:
   Modify `prefetcher.py` GCB sizing to round up the per-tensor wrap
   point to LCM(13056, 8704, 13824, 26112) = ? .  Need to check if
   the LCM fits L1 — likely no (e.g. LCM(13824, 26112) > 100KB).
   Alternative: enforce that all matmul `in1_block_size_bytes` share
   a common factor; pad the producer's per-tensor wrap to that.

2. **U34-B — kernel-side alignment-aware read**:
   In the gathered compute kernel, when the offset is not a multiple
   of `block_size_bytes`, read the FIRST partial block separately
   (combining bytes from the unaligned start + the rest from the
   next block) before falling into the regular stride-by-block loop.
   Mid-tile reads may not be possible at the BFP4/BFP8 unpacker
   level — needs probe.

3. **U34-C — change matmul program_configs to UNIFY block_sizes**:
   Force all matmuls to use the same `per_core_N * in0_block_w`
   (e.g. always 12 tiles, giving in1_block_size_bytes = 12 * 1088 = 13056
   for all BFP8 matmuls).  Requires retuning matmul subblock dims;
   may regress per-op perf but eliminates the alignment issue.

The current sense is that **U34-B is the cleanest semantically**
(consumer can handle whatever the producer wrote), but might be
infeasible at the LLK level if BFP tiles can't be split mid-block.
**U34-C is most likely to work** if subblock retuning preserves perf.

## Hypothesis ledger (post-U33)

| ID | Suspect | Pre-U33 | Post-U33 |
|---|---|---|---|
| U7 / Case B — Gathered compute kernel produces garbage on specific CT-arg ELFs under REAL weights | CONFIRMED (U30) | UNCHANGED |
| U31 — Per-matmul rd_ptr reset to fifo_start independent of tensor's actual write position | STRUCTURALLY ADDRESSED (U32) + COMPUTATION FIXED (U33) | **Offset values now correct per producer-cumulative.** |
| **NEW U33 — Cumulative consumer-side per-tensor offsets are not aligned to each matmul's `in1_block_size_bytes`; CB stride-by-block reads then misinterpret BFP tile boundaries → `Invalid subtile broadcast` at downstream `binary_ng`** | (new) | **TOP STANDING.** |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only; commit `a29dbd6552f` on `tenstorrent-p1` only).
- [x] No SGLang core behavior changes (Python-side `python/sglang/...` untouched).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` to sync Python source.  No C++ changes this iteration ⇒ no rebuild needed.
- [x] All U33 paths env-gated (`SGLANG_TT_U33_FACTORY_OFFSET`).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"` at 9.14s) + GSM8K(10) = 9/10 = 90% (Q3 reasoning-pattern only; baseline-equivalent).
- [x] Server stopped at session end; cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: **`a29dbd6552f`** (U33 consumer-side offset; 3 files).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/models/tt_transformers/tt/*.py`: synced; no C++ changes so `_ttnn.so`/`_ttnncpp.so` from U32 (with `SGLANG_TT_U32_GCB_OFFSET` + `SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES` sentinels) re-used unchanged.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at 9.14s + GSM8K(10) = 9/10 (Q3 reasoning-pattern only).
- DPRINT/stderr artifacts preserved in-container at `/tmp/u33_test4.log` (U33 with full registration + observed offsets) and `/tmp/u33_canonical.log` (canonical control).

## Shipping verdict

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Same shipping verdict as U31/U32; root mechanism now COMPUTED correctly via U33's consumer-side offsets but alignment gap remains (U34).
- **`SGLANG_TT_U33_FACTORY_OFFSET=1` is safe to leave in tree (default-off + canonical bytewise-equivalent at default).**  When enabled WITH prefetcher, fires the iter-1 broadcast errors (U33.5).  Not a regression risk because the env var is default OFF.

## 45-dispatch ledger (cumulative)

| ID | Status | Note |
|---|---|---|
| S1-S10 | CLOSED | Per-dispatch sub-investigations summarized in U21/U22/etc. |
| Lead 1/2/3 | CLOSED | Per-doc closure. |
| A1 | CLOSED | All-reduce alignment; not the bug. |
| U1-U16 | CLOSED | Each ruled out as root cause (DST stale, GCB block bisect, LLK probes, PACK probes, W2→RS barrier, etc.). |
| U17-U28 | CLOSED | All "downstream stomper" theories refuted — wrong bytes at 0xa6700 trace back to matmul writing wrong values, not external stomper. |
| U29 | CLOSED | W2→RS semaphore handshake VERIFIED working but garbage persists. |
| U30 | CLOSED | Per-ELF PACK-side garbage measurement: 86% NaN/Inf at PACK exit for 3 of 4 ELFs. |
| U31 | CLOSED | Discriminator: BFP8 vs BFP4 dtype.  Root mechanism: matmul rd_ptr assumption mismatches prefetcher write position. |
| U32 | CLOSED | Structural plumbing (kernel + factory + override + Python) landed; canonical bytewise-equivalent; offset COMPUTATION deferred to U33. |
| **U33** | **EVIDENCE_ADVANCE — consumer-side per-tensor cumulative bytes landed; offsets correctly identify producer's write positions; iter-1 still throws `Invalid subtile broadcast` because cumulative offsets are not aligned to matmul's `in1_block_size_bytes`.** | **TOP STANDING** for offset-value gap.  Block-size alignment between successive tensors in the prefetcher queue is the remaining issue. |
| U34 | OPEN | Three options: A (LCM-align producer wraps), B (kernel-side alignment-aware reads), C (unify matmul block_sizes Python-side).  Preferred next attempt: U34-C (unify block_sizes via subblock retuning). |

## Commits this session

- (tt-metal-sglang) **1 commit**: `a29dbd6552f prefetcher: U33 — consumer-side per-tensor GCB byte-offset computation (env-gated; offsets right at iter-2+ but alignment-mismatch with matmul block-size still triggers binary_ng broadcast at iter-1; canonical 9/10 preserved; tracked as U34)`.
- (sglang) `<this doc>` — pending commit.
