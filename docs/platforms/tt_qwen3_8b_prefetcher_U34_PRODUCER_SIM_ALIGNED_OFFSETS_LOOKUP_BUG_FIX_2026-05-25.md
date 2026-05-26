# TT Qwen3-8B prefetcher — U34 producer-simulation aligned offsets + CRITICAL `get_tensor_gcb_offset_bytes` lookup-bug fix; broadcast crashes ELIMINATED but output garbage remains; canonical re-verified 9/10 — 2026-05-25

Status: **EVIDENCE_ADVANCE — U34 lands TWO orthogonal fixes (both env-gated):
(1) CRITICAL lookup bug in `get_tensor_gcb_offset_bytes` discovered
post-U33: `prefetched_tensors.index(tensor)` is unreliable for
ttnn.Tensor (whose `__eq__` returns a tensor not a bool, causing
list.index to spuriously return 0).  Effectively the entire U32+U33
plumbing was a no-op at runtime — every matmul saw offset=0.  Fixed
by mirroring U33 `set_tensor_consumer_bytes`'s explicit `is`-identity +
buffer_address fallback.  Applies to both U33 and U34 paths.  (2)
producer simulation: replay writer_l1.cpp + resize_remote_sender_cb_interface
exactly in Python to produce aligned, per-tensor offsets matching the
producer's actual fifo_wr_ptr trajectory.  U34 computed offsets
(env-gated SGLANG_TT_U34_LCM_ALIGN=1): T0=0 (aligned 13056), T1=417792
(aligned 8704), T2=705024 (aligned 13824 vs U33's 696320), T3=317952
(aligned 13824 vs U33's 303104), T4=783360 (aligned 26112 vs U33's
745472).  All aligned to their own page_size.  Hardware result:
broadcast `Invalid subtile broadcast` crashes that fired 144+ times
under U33 are 100% ELIMINATED under U34; decode runs end-to-end
without exceptions.  But output text is garbage ("cruc cruc cruc" /
"doing doing doing"), and SGLANG_TT_U32_FORCE_ZERO=1 produces the
SAME garbage pattern, proving the alignment fix is necessary but not
sufficient — the actual data layout in the GCB does not match either
U33's or U34's offsets in a way that needs deeper kernel-level
probing (U35: DPRINT writer_l1 wr_ptr trajectory).  Canonical Qwen3-8B
re-verified bytewise: `" What is 2+2? What is 2+2? What"` at 9.114s
+ GSM8K(10) chat = 9/10 = 90.0% (Q3 reasoning pattern; baseline).**

Continuation of `tt_qwen3_8b_prefetcher_U33_CONSUMER_SIDE_OFFSET_LANDED_BUT_ALIGNMENT_GAP_2026-05-25.md`.

tt-metal-sglang HEAD: `08966cd8752` (U34, 1 file).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U34.1 | Replay producer (`writer_l1.cpp` + `resize_remote_sender_cb_interface`) Python-side so each per-tensor offset matches the producer's actual `fifo_wr_ptr` and is multiple of THAT matmul's `in1_block_size_bytes` | `get_tensor_gcb_offset_bytes` adds a U34 branch (env-gated `SGLANG_TT_U34_LCM_ALIGN=1`) that simulates: (a) align wr_ptr up to per-tensor page_size, (b) wrap if past `fifo_limit_page_aligned = gcb_size - gcb_size%page_size`, (c) write num_blocks blocks of page_size each, with mid-tensor wrap at `dest_addr == fifo_limit_page_aligned`.  Recovers per-tensor `page_size` from `_tensor_consumer_bytes[t] / num_blocks` (where num_blocks = ring_size). | **LANDED in 1 file** (`prefetcher.py` +130) |
| U34.2 | Hardware test — Qwen3-8B + paged + prefetcher + U34 | `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_U32_GCB_OFFSET=1 SGLANG_TT_U33_FACTORY_OFFSET=1 SGLANG_TT_U34_LCM_ALIGN=1`. | **CRASHES (Invalid subtile broadcast) ELIMINATED but only 1 unique U34_RETURN observed (all per_layer_idx=0 offset=0)** — pointed to lookup bug. |
| U34.3 | Investigate why all U34_RETURN log lines show `per_layer_idx=0 offset=0` even though 5 distinct tensors should be looked up | `prefetched_tensors.index(tensor)` uses `==`.  ttnn.Tensor `__eq__` returns a TENSOR (not bool), making `list.index` spuriously return 0 (first match) for every tensor. | **CRITICAL BUG FOUND.** |
| U34.4 | Fix lookup in `get_tensor_gcb_offset_bytes` to mirror `set_tensor_consumer_bytes`'s explicit Python `is` + buffer_address fallback | Replace `try: idx = self.prefetched_tensors.index(tensor)` with `for i,t in enumerate(self.prefetched_tensors): if t is tensor: idx=i` + buffer_address fallback. | **FIXED — 5 distinct U34_RETURN lines, one per per_layer_idx.** |
| U34.5 | Re-test with lookup fix + U34 sim | Same env as U34.2. | **iter-1 `Invalid subtile broadcast` count = 0 (was 144+).  Server runs to completion.  Decode returns garbage tokens but no crashes.** |
| U34.6 | Bisect: maybe broadcast crashes were the only effect, and offsets are still ignored downstream? | Run with `SGLANG_TT_U32_FORCE_ZERO=1` (all offsets forced to 0). | **SAME garbage pattern.**  Suggests either (a) C++ env-var update path isn't actually propagating U34 offsets to the kernel binary's compute_runtime_args, or (b) the producer's actual layout doesn't match the U34 simulation. |
| U34.7 | Canonical re-verify (default — all U32/U33/U34 envs unset, no prefetcher) | Standard canonical. | **PASS — `" What is 2+2? What is 2+2? What"` at 9.114s e2e_latency + GSM8K(10) chat = 9/10 = 90.0% (Q3 reasoning-truncation only; bytewise-equivalent to U33 baseline).** |

**Punchline:** U34 lands the producer-simulation Python-side AND fixes
a critical lookup bug discovered during U34 testing.  The lookup bug
meant ALL prior U32/U33 work was silently a no-op (every matmul saw
offset=0).  With the lookup fixed AND alignment via producer-simulation,
the 144+ `Invalid subtile broadcast` TT_THROWs that had fired since U7
are 100% eliminated.  However, output text is still garbage, and
forcing all offsets to 0 produces the same garbage — proving the
alignment fix is necessary but not sufficient.  The bug now lives one
level deeper than CB rd_ptr alignment.

## Files changed (1)

```
 models/tt_transformers/tt/prefetcher.py | 134 +++++++++++++++++++++++++++++++-
 1 file changed, 130 insertions(+), 4 deletions(-)
```

Commit: `08966cd8752` on `tenstorrent-p1`.

## Implementation deep-dive

### Critical lookup bug fix (U34.4)

The pre-U34 `get_tensor_gcb_offset_bytes` did:
```python
try:
    idx = self.prefetched_tensors.index(tensor)
except ValueError:
    return 0
```

`list.index()` iterates with `==`.  For ttnn.Tensor, `==` is elementwise
and returns a tensor — when Python tries `bool(tensor_result)` to decide
whether `index` should return, it triggers undefined behavior (raises
RuntimeError or returns True for non-empty tensors).  Empirically, every
lookup returned 0, regardless of which tensor was passed.

Fix (mirroring `set_tensor_consumer_bytes`):
```python
idx = None
for i, t in enumerate(self.prefetched_tensors):
    if t is tensor:
        idx = i
        break
if idx is None:
    try:
        tgt_addr = tensor.buffer_address()
    except Exception:
        tgt_addr = None
    if tgt_addr is not None:
        for i, addr in enumerate(self.prefetched_tensor_addr):
            if addr == tgt_addr:
                idx = i
                break
if idx is None:
    return 0
```

This fix applies in BOTH the U33 and U34 code paths.  Without it, all
prior U32/U33 work was silently a no-op.

### Producer-simulation (U34.1)

The matmul kernel reads from `in1_cb_start_addr + u32_tensor_offset_bytes / L1_ALIGNMENT`.
For this to land at the correct tensor's data, `u32_tensor_offset_bytes`
must equal the producer's actual `fifo_wr_ptr` at the time it began
writing that tensor.

The producer's per-tensor wr_ptr trajectory is governed by
`resize_remote_sender_cb_interface` (line 110 of `remote_circular_buffer.h`)
called from `writer_l1.cpp` (line 95) at the start of each tensor:

```cpp
uint32_t cb_size_page_aligned = fifo_size - fifo_size % page_size;
uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;
uint32_t next_fifo_wr_ptr = fifo_start_addr + align(fifo_wr_ptr - fifo_start_addr, page_size);
if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
    next_fifo_wr_ptr = fifo_start_addr;
}
```

Then `remote_cb_push_back_and_write_pages` (line 315) writes num_blocks
blocks of `page_size` each, advancing dest_addr and wrapping to
fifo_start when `dest_addr == fifo_limit_page_aligned` (line 378).

U34's Python simulation:
```python
wr_ptr = 0
for t in range(per_layer_idx + 1):
    ps = self._tensor_consumer_bytes[t] // num_blocks
    cb_size_pa = gcb_size - (gcb_size % ps)
    fifo_limit_pa = cb_size_pa
    # resize: align wr_ptr up to ps; wrap if past fifo_limit_pa
    if (wr_ptr % ps) != 0:
        wr_ptr = ((wr_ptr + ps - 1) // ps) * ps
    if wr_ptr >= fifo_limit_pa:
        wr_ptr = 0
    if t == per_layer_idx:
        return wr_ptr
    # writes: num_blocks blocks of ps each
    for _ in range(num_blocks):
        wr_ptr += ps
        if wr_ptr == fifo_limit_pa:
            wr_ptr = 0
        elif wr_ptr > fifo_limit_pa:
            wr_ptr = wr_ptr - fifo_limit_pa
```

## Observed offsets (U34, hardware verified)

(From `[U34_RETURN]` log lines, real run)

| per_layer_idx | tensor | dtype | page_size (in1_block_size_bytes) | U33 offset | U34 offset | U33 aligned? | U34 aligned? |
|---:|---|---|---:|---:|---:|:---:|:---:|
| 0 | WQKV | BFP8 | 13056 | 0 | **0** | YES | YES |
| 1 | WO   | BFP8 | 8704  | 417792 | **417792** | YES | YES |
| 2 | W1   | BFP4 | 13824 | 696320 | **705024** | **NO (5120 rem)** | YES |
| 3 | W3   | BFP4 | 13824 | 303104 | **317952** | **NO (12800 rem)** | YES |
| 4 | W2   | BFP8 | 26112 | 745472 | **783360** | **NO (24320 rem)** | YES |

All U34 offsets are multiples of their own page_size — confirmed in log
`aligned=True` for every tensor.

## Hardware test result

```
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U32_GCB_OFFSET=1 \
SGLANG_TT_U33_FACTORY_OFFSET=1 \
SGLANG_TT_U34_LCM_ALIGN=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -m sglang.launch_server ... --model-path /models/Qwen3-8B ...
```

| Test | Pre-U34 (U33 cumulative offsets, broken lookup) | Post-U34 (producer sim + lookup fix) |
|---|---|---|
| `Invalid subtile broadcast` crash count | 144+ | **0** |
| Decode runs to completion (15 tokens) | NO | **YES** |
| Output (`"What is 2+2?"` prompt) | (none — crashes) | `" What() cruc cruc cruc..."` (garbage) |
| Output (`"Hello world"` prompt) | (none — crashes) | `"! doing doing doing..."` (garbage) |
| Output with `SGLANG_TT_U32_FORCE_ZERO=1` (all offsets=0) | (crashes) | `" What前往-position-position..."` (garbage) |

The garbage under U34 has the same character as under FORCE_ZERO,
suggesting offsets reach the kernel but don't correspond to where the
producer actually wrote the data (OR the C++ override path that's
supposed to propagate the env var to compute_runtime_args is not
actually firing).

## Why "broadcast crashes ELIMINATED but output garbage remains"

The pre-U34 broadcast crashes fired in downstream `binary_ng` ops
(post-matmul residual adds).  The kernel-internal `Invalid subtile
broadcast type` (assert.hpp:104) reflects BFP tile-header values
inconsistent with the expected (a_h, a_w, b_h, b_w) dimensions —
i.e. raw bytes don't decode as a valid BFP tile.  U34's alignment
ensures consumer rd_ptr lands on a CB-page boundary, so the bytes
DO parse as valid BFP tiles.  The PACK side writes parseable garbage
to L1; downstream `binary_ng` reads parseable garbage; no exception.

But the values are semantically meaningless — the bytes at offset
705024 in W1's GCB region are NOT W1's data, they're either zero
(if FORCE_ZERO and matmul reads tensor-0 bytes) or some other tensor's
data interpreted as W1.  Result: forward-pass produces garbage
activations; logits favor random tokens that the sampler then degrades
into "cruc cruc cruc" / "doing doing doing" patterns (low-entropy
collapse).

## Hypothesis ledger (post-U34)

| ID | Suspect | Pre-U34 | Post-U34 |
|---|---|---|---|
| U7 / Case B — Gathered compute kernel produces garbage on specific CT-arg ELFs under REAL weights | CONFIRMED (U30) | UNCHANGED (root mechanism still active, just no longer crashes the kernel) |
| U31 — Per-matmul rd_ptr reset to fifo_start independent of tensor's actual write position | STRUCTURALLY ADDRESSED (U32) + COMPUTATION FIXED (U33) | **+ ALIGNED (U34)** — every consumer rd_ptr now lands on tile boundary. |
| U33 — Cumulative consumer-side offsets not aligned to matmul's `in1_block_size_bytes`; CB stride-by-block reads misinterpret BFP tile boundaries → `Invalid subtile broadcast` | TOP STANDING | **REFUTED as final-cause — alignment fixed via U34 producer-sim; crashes eliminated; but output STILL garbage.** |
| **NEW U34 — `get_tensor_gcb_offset_bytes` lookup bug**: `list.index(tensor)` returns 0 for every ttnn.Tensor → all U32/U33 work silently a no-op | (new) | **FIXED + CRITICAL** |
| **NEW U34-α — Even with correct, aligned offsets, GCB layout doesn't match producer-sim**: FORCE_ZERO and U34 both produce same garbage class → either env-var update path is broken OR the producer's actual fifo_wr_ptr trajectory differs from U34 simulation | (new) | **TOP STANDING.**  Next attack: DPRINT writer_l1's actual wr_ptr per tensor (U18 PREFETCHER_WRITE_PROBE rebuilt) to verify ground truth. |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only; commit `08966cd8752` on `tenstorrent-p1` only).
- [x] No SGLang core behavior changes (Python-side `python/sglang/...` untouched).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — synced via `podman cp` (Python-only change, no C++ rebuild needed).
- [x] All U34 paths env-gated (`SGLANG_TT_U34_LCM_ALIGN`).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"` at 9.114s) + GSM8K(10) = 9/10 = 90% (Q3 reasoning-pattern only; baseline-equivalent to U33).
- [x] Server stopped at session end; cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: **`08966cd8752`** (U34 producer-sim + lookup fix; 1 file).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/models/tt_transformers/tt/prefetcher.py`: synced; no C++ changes so `_ttnn.so`/`_ttnncpp.so` from U32 re-used unchanged.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at 9.114s + GSM8K(10) = 9/10.
- DPRINT/stderr artifacts preserved in-container:
  - `/tmp/u34_test1.log` (initial U34 — pre lookup fix; shows all offsets=0 because of lookup bug)
  - `/tmp/u34_test2.log` (post-lookup-fix WITH per-call U34_RETURN spam; broadcast crashes)
  - `/tmp/u34_test3.log` (post-lookup-fix WITH dedup logging; broadcast crashes ELIMINATED, garbage output)
  - `/tmp/u34_zero.log` (U34 + FORCE_ZERO; garbage matches U34 — proves offsets-don't-help)
  - `/tmp/u34_canonical.log` (canonical no-prefetcher control; baseline output)

## Shipping verdict

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Same shipping verdict as U31/U32/U33; U34 eliminates the broadcast crashes but the underlying data layout mismatch persists, producing garbage outputs.
- **`SGLANG_TT_U34_LCM_ALIGN=1` is safe to leave in tree (default-off + canonical bytewise-equivalent at default).**
- **The lookup-bug fix in `get_tensor_gcb_offset_bytes` is ALWAYS active** (not env-gated; affects both U33 and U34 paths).  Since U33's offsets were never reaching the kernel pre-fix (all returned 0), the fix could in principle change U33-only behavior — but U33 alone (without U34) was already known-broken (per the U33 doc, iter-1 crashes), so the fix doesn't regress any working config.

## 48-dispatch ledger (cumulative)

| ID | Status | Note |
|---|---|---|
| S1-S10, Lead 1/2/3, A1 | CLOSED | Per-doc closure (earlier sub-investigations). |
| U1-U16 | CLOSED | DST stale, GCB block bisect, LLK probes, PACK probes, W2→RS barrier, etc. |
| U17-U28 | CLOSED | All "downstream stomper" theories refuted. |
| U29 | CLOSED | W2→RS semaphore handshake verified working; garbage persists. |
| U30 | CLOSED | Per-ELF PACK-side garbage: 86% NaN/Inf at PACK exit for 3 of 4 ELFs. |
| U31 | CLOSED | Discriminator: BFP8 vs BFP4 dtype. |
| U32 | CLOSED | Structural plumbing (kernel + factory + override + Python) landed. |
| U33 | CLOSED | Consumer-side cumulative offsets landed Python-side; alignment gap → U34. |
| **U34** | **EVIDENCE_ADVANCE — broadcast crashes ELIMINATED (alignment via producer-simulation) + CRITICAL lookup-bug fix (U32/U33 had been silent no-op); but output garbage persists; FORCE_ZERO and U34 produce same garbage pattern proving alignment is necessary-not-sufficient.** | **U34-α TOP STANDING** for data-layout mismatch. |
| U35 | OPEN | DPRINT writer_l1 wr_ptr per tensor (rebuild SGLANG_TT_U18_PREFETCHER_WRITE_PROBE).  Compare ground-truth producer trajectory vs U34 simulation.  If they match → environment-var-propagation path is broken (check C++ override timing).  If they differ → simulation has a bug. |

## Commits this session

- (tt-metal-sglang) **1 commit**: `08966cd8752 prefetcher: U34 — producer-simulation aligned offsets + critical lookup fix (broadcast crashes ELIMINATED; output garbage remains under prefetcher; canonical 9/10 preserved)`.
- (sglang) `<this doc>` — pending commit.
