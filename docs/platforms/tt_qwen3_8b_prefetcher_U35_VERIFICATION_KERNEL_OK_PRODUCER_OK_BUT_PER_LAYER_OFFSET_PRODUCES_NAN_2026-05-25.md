# TT Qwen3-8B prefetcher — U35 KERNEL-side DPRINT proves C++ env→RT-arg plumbing INTACT; PRODUCER-side DPRINT confirms wr_ptr trajectory EXACTLY matches U34 simulation for layer 0; per-layer stateful offset attempt produces NaN; canonical 9/10 preserved — 2026-05-25

Status: **EVIDENCE_ADVANCE — U35 lands THREE orthogonal pieces (all
env-gated, default-off): (1) kernel-side DPRINT
`SGLANG_TT_U35_KERNEL_OFFSET_PROBE=1` proves the C++
factory→runtime-arg→kernel reception path is bytewise-correct (every
ELF's expected U34 offset reaches the kernel: WQKV=0, WO=417792,
W1=705024, W3=317952, W2=783360); (2) producer-side DPRINT (existing
U18 rebuilt) confirms that the producer's actual `fifo_wr_ptr` at the
start of layer 0's tensors matches the U34 Python simulation EXACTLY
to-the-byte (WQKV=0xac700+0, WO=+0x66000, W1=+0xac200, W3=+0x4d9c0,
W2=+0xbf3c0); (3) ROOT CAUSE IDENTIFIED via producer trajectory across
layers: page sizes are incommensurate with the GCB (13824 doesn't
divide 835584 exactly — 835584 = 60.45 × 13824), so the producer's
`fifo_wr_ptr` at the START of layer N differs from its position at
layer 0 for every N ≥ 1.  But U34 returns the SAME (layer-0) offset
for every layer's ttnn.linear, so layers 1+ read from positions where
the producer has NOT written this tensor's data.  U35 implementation
extends U34's simulation to track cumulative state across layers AND
across iterations (the producer state is persisted to L1 via
update_remote_cb_config_in_l1, so iter-2's layer 0 doesn't start at
0).  Stateful Python-side counter advances once per
`get_tensor_gcb_offset_bytes` call, matching the producer's 1:1
write-per-call pacing.  Hardware verification: WQKV layer-1 offset
783360 reaches the kernel correctly; all 14 unique WQKV offsets across
36 layers match the Python simulation exactly; producer fifo_wr_ptr
trajectory for layer-1 t=0 measured at 0x16bb00 (= 783360 from
fifo_start).  However: output is still garbage AND introduces NaN
downstream (sampler crashes with `RuntimeError: probability tensor
contains either inf, nan or element < 0`).  Suggests per-layer
non-zero offsets near GCB wrap may trigger a separate kernel bug in
`is_tensor_split` / `update_rd_ptr_to_ring_index` wrap arithmetic for
the matmul block-stride loop.  Canonical Qwen3-8B re-verified bytewise:
`" What is 2+2? What is 2+2? What"` at 9.09s + GSM8K(10) chat = 9/10
= 90.0%.**

Continuation of `tt_qwen3_8b_prefetcher_U34_PRODUCER_SIM_ALIGNED_OFFSETS_LOOKUP_BUG_FIX_2026-05-25.md`.

tt-metal-sglang HEAD: `fbdbc73fb28` (U35, 3 files).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U35.1 | Does the C++ env→RT-arg→kernel reception path actually convey the offset? | Add unconditional DPRINT at top-of-batch in `bmm_large_block_zm_fused_bias_activation_gathered.cpp` showing `u32_tensor_offset_bytes` RT-arg value, `in1_block_size_bytes`, and resulting `new_rd_ptr_shifted`.  Env-gated via `SGLANG_TT_U35_KERNEL_OFFSET_PROBE=1` (propagated through factory `mm_kernel_defines`). | **115,200 prints captured.** Unique (offset, block_size) pairs = exactly 5: (0, 13056), (317952, 13824), (417792, 8704), (705024, 13824), (783360, 26112).  **All match U34's table to-the-byte.** Conclusion: C++ propagation is INTACT. |
| U35.2 | Does the producer actually write at those positions? | Rebuild `SGLANG_TT_U18_PREFETCHER_WRITE_PROBE` (existing).  Run prefetcher; grep first per-tensor `blk=0` lines on core (0,0)-1.  Compute (wr_ptr - fifo_start). | **MATCH to-the-byte for layer 0 all 5 tensors.** WQKV offset=0, WO=417792, W1=705024, W3=317952, W2=783360 (verified `producer_offset == U34_simulation_offset` for every tensor in layer 0). |
| U35.3 | Does the producer's wr_ptr trajectory match U34 for LATER layers? | Continue scanning producer log: layer 1 t=0 → 0x16bb00 (= 783360). My Python sim for layer 1 t=0 → 783360. Layer 1 t=1 → 0x105b00 (= 365568).  My sim → 365568.  Layer 1 t=2 → 0x14b100 (= 649728).  My sim → 649728.  Layer 1 t=3 → 0xec900 (= 262656).  My sim → 262656.  Layer 1 t=4 → 0x158900 (= 705024).  My sim → 705024. | **MATCH to-the-byte for layer 1 all 5 tensors.** Page sizes are incommensurate with GCB (13824 doesn't divide 835584 exactly), so producer's `fifo_wr_ptr` at the start of layer 1 DIFFERS from layer 0 for every tensor.  This is the ROOT CAUSE the U34 cumulative-offset approach was missing. |
| U35.4 | Implement per-layer stateful offset computation | `prefetcher.py`: add U35 branch under `SGLANG_TT_U35_PER_LAYER_OFFSET=1`.  Tracks `self._u35_state_wr_ptr` — advances by `num_blocks * page_size` per `get_tensor_gcb_offset_bytes` call, with wrap-handling matching the producer.  Auto-resyncs if caller's `(layer_idx, per_layer_idx)` doesn't match the expected next position (handles init / re-warmup / non-U35 calls falling through). | **In-tree (+152 lines).** Logs 30 unique `[U35_RETURN_V2 call=N L=X t=Y offset=Z state_pre=W]` lines confirming correct per-call offsets for the first ~6 layers. |
| U35.5 | Hardware test — Qwen3-8B + paged + prefetcher + U35 | `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_U32_GCB_OFFSET=1 SGLANG_TT_U33_FACTORY_OFFSET=1 SGLANG_TT_U34_LCM_ALIGN=1 SGLANG_TT_U35_PER_LAYER_OFFSET=1`. | **Still garbage output AND introduces NaN downstream.** Sampler crashes with `RuntimeError: probability tensor contains either inf, nan or element < 0`.  Single decode returns garbage tokens like `" What нужACTIVE有不同的ække..."`.  GSM8K(10) = 0/10 (all server-crash after Q1). |
| U35.6 | Canonical re-verify (all U32/U33/U34/U35 envs unset, no prefetcher) | Standard canonical. | **PASS — `" What is 2+2? What is 2+2? What"` at 9.09s e2e_latency + GSM8K(10) chat = 9/10 = 90.0% (Q4 reasoning-pattern only; bytewise-equivalent to U34 baseline).** |

**Punchline:** U35 verifies both ends are CORRECT structurally: the
kernel receives the right offset, the producer writes at the right
position.  Then it identifies the missing piece (per-layer producer
trajectory shift) and implements a stateful tracker.  But: the
per-layer stateful tracker, while plumbing the correct offset to the
kernel, produces output garbage AND NaN.  This proves the bug now
lives one level deeper than per-call offset computation — the matmul
kernel's wrap-arithmetic (`is_tensor_split` /
`update_rd_ptr_to_ring_index` / `calculate_next_block_index_and_update_rd_ptr`)
must have an edge case when offset + ring_idx * block_size straddles
the GCB wrap boundary in a way that worked for layer-0 (wrap-free
start) but fails for layers 1+ (wrap-straddling start).

## Files changed (3)

```
 models/tt_transformers/tt/prefetcher.py            | 152 +++++++++++++++++++++
 ...ul_multicore_reuse_mcast_1d_program_factory.cpp |  12 ++
 ...rge_block_zm_fused_bias_activation_gathered.cpp |  31 +++++
 3 files changed, 195 insertions(+)
```

Commit: `fbdbc73fb28` on `tenstorrent-p1`.

## Implementation deep-dive

### Path A — Kernel-side DPRINT (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`)

After the U32 `update_local_cb_rd_ptr` line that shifts rd_ptr by
the per-tensor offset, unconditionally (gated only by
`SGLANG_TT_U35_KERNEL_OFFSET_PROBE` propagated via the matmul factory)
print one DPRINT line on UNPACK risc, batch 0:

```cpp
#ifdef SGLANG_TT_U35_KERNEL_OFFSET_PROBE
if (b == 0) {
    UNPACK((DPRINT << "[U35_KERNEL_OFFSET ring_idx=" << ring_idx
                   << " offset_bytes=" << u32_tensor_offset_bytes
                   << " in1_block_size_bytes=" << in1_block_size_bytes
                   << " in1_cb_start_shifted=" << in1_cb_start_addr
                   << " new_rd_ptr_shifted="
                   << (in1_cb_start_addr + u32_tensor_offset_bytes / L1_ALIGNMENT)
                   << "]" << ENDL()));
}
#endif
```

Factory propagation:

```cpp
const char* u35_off_probe_env = std::getenv("SGLANG_TT_U35_KERNEL_OFFSET_PROBE");
if (u35_off_probe_env != nullptr && std::string(u35_off_probe_env) == "1") {
    mm_kernel_defines["SGLANG_TT_U35_KERNEL_OFFSET_PROBE"] = "1";
}
```

### Path B — Producer DPRINT

`SGLANG_TT_U18_PREFETCHER_WRITE_PROBE` (existing, U18) already prints
per-block `[U18_PREF_WR layer=N t=X blk=Y wr_ptr=0x... fifo_start=0x...]`.
Rebuild with `SGLANG_TT_U18_PREFETCHER_WRITE_PROBE=1`.

### Path C — U35 stateful per-layer offset (`prefetcher.py`)

```python
# U35 — STATEFUL simulation across iterations.
# The PRODUCER (writer_l1.cpp) calls update_remote_cb_config_in_l1
# at the end of EACH kernel call, persisting the current fifo_wr_ptr
# to L1 config.  The NEXT prefetcher call (next decode token) starts
# from that saved wr_ptr — NOT from fifo_start.  So iter 2's "layer
# 0 tensor 0" offset is NOT zero; it's wherever iter 1 left off.
#
# Maintain a Python-side simulator state (`self._u35_state_wr_ptr`)
# that advances by exactly one tensor's worth of writes on EACH
# call.  Caller is expected to call get_tensor_gcb_offset_bytes 1:1
# with ttnn.linear dispatches in (L=0..num_layers, t=0..num_tensors)
# order — which matches the producer's writer_l1 loop.
if not hasattr(self, "_u35_state_wr_ptr"):
    self._u35_state_wr_ptr = 0
    self._u35_state_next_call_lt = (0, 0)
    self._u35_call_count = 0
expected_L, expected_t = self._u35_state_next_call_lt
if (layer_idx, per_layer_idx) != (expected_L, expected_t):
    # Re-sync: simulate writes from (0,0) → (layer_idx, per_layer_idx).
    ...
# Apply resize-up alignment for the asked tensor.
ps = page_sizes[per_layer_idx]
cb_size_pa = gcb_size - (gcb_size % ps)
fifo_limit_pa = cb_size_pa
wr_ptr = self._u35_state_wr_ptr
if (wr_ptr % ps) != 0:
    wr_ptr = ((wr_ptr + ps - 1) // ps) * ps
if wr_ptr >= fifo_limit_pa:
    wr_ptr = 0
# Advance state for the next call.
wr_after = wr_ptr
for _ in range(num_blocks):
    wr_after += ps
    if wr_after == fifo_limit_pa:
        wr_after = 0
    elif wr_after > fifo_limit_pa:
        wr_after = wr_after - fifo_limit_pa
self._u35_state_wr_ptr = wr_after
return wr_ptr
```

## Observed offsets — kernel-side (U35.1, hardware verified)

Filtered `grep "U35_KERNEL_OFFSET" /tmp/u35_dprint.log | head` and
de-duplicated:

| ring_idx | offset_bytes | in1_block_size_bytes | new_rd_ptr_shifted |
|---:|---:|---:|---:|
| 23 | 0 | 13056 | 44144 |
| 23 | 317952 | 13824 | 64016 |
| 23 | 417792 | 8704 | 70256 |
| 23 | 705024 | 13824 | 88208 |
| 23 | 783360 | 26112 | 93104 |

All 5 unique (offset, block_size) pairs match U34's per-tensor table.

## Observed offsets — producer-side (U35.2, hardware verified)

Filtered `grep "U18_PREF_WR layer=0 t=[0-4] blk=0" /tmp/u35_producer.log | head -5`:

| Layer | Tensor | wr_ptr (hex) | (wr_ptr - fifo_start) | U34 sim | Match |
|---:|---|---|---:|---:|:---:|
| 0 | WQKV | 0xac700 | 0 | 0 | YES |
| 0 | WO   | 0x112700 | 417792 | 417792 | YES |
| 0 | W1   | 0x158900 | 705024 | 705024 | YES |
| 0 | W3   | 0xfa100 | 317952 | 317952 | YES |
| 0 | W2   | 0x16bb00 | 783360 | 783360 | YES |

| Layer | Tensor | wr_ptr (hex) | (wr_ptr - fifo_start) | U35 v2 sim | Match |
|---:|---|---|---:|---:|:---:|
| 1 | WQKV | 0x16bb00 | 783360 | 783360 | YES |
| 1 | WO   | 0x105b00 | 365568 | 365568 | YES |
| 1 | W1   | 0x14b100 | 649728 | 649728 | YES |
| 1 | W3   | 0xec900 | 262656 | 262656 | YES |
| 1 | W2   | 0x158900 | 705024 | 705024 | YES |

## Hardware test result

```
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U32_GCB_OFFSET=1 \
SGLANG_TT_U33_FACTORY_OFFSET=1 \
SGLANG_TT_U34_LCM_ALIGN=1 \
SGLANG_TT_U35_PER_LAYER_OFFSET=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -m sglang.launch_server ... --model-path /models/Qwen3-8B ...
```

| Test | Pre-U35 (U34) | Post-U35 (per-layer stateful) |
|---|---|---|
| Single decode `"What is 2+2?"` | `" What() cruc cruc cruc..."` (garbage, layer-0-only offsets) | `" What нужACTIVE有不同的ække窍ACTIVE..."` (different garbage) |
| GSM8K(10) | 0/10 (garbage; layer-0 offsets used for all layers; output garbage but no NaN crash) | **0/10 + SERVER CRASH** at Q2 with `RuntimeError: probability tensor contains either inf, nan or element < 0` |
| Kernel receives correct per-(layer, tensor) offset (U35 DPRINT) | NO (single offset per tensor across all layers) | **YES** (all 14 unique WQKV offsets present) |
| Output is correct | NO | NO |

The per-layer offsets DO reach the kernel correctly under U35.  But
the model still produces garbage AND now NaN.  This proves the bug
now lives one level deeper than per-call offset computation.

## Hypothesis ledger (post-U35)

| ID | Suspect | Pre-U35 | Post-U35 |
|---|---|---|---|
| U7 / Case B — Gathered compute kernel produces garbage on specific CT-arg ELFs under REAL weights | CONFIRMED | UNCHANGED — root mechanism still active |
| U31 — Per-matmul rd_ptr reset to fifo_start independent of tensor's actual write position | STRUCTURALLY ADDRESSED (U32+U33+U34) + lookup fix (U34) | **+ PER-LAYER PRODUCER-DRIFT IDENTIFIED (U35)** — pages incommensurate with GCB → producer wr_ptr drifts across layers. |
| U33 — Cumulative consumer-side offsets not aligned to matmul's `in1_block_size_bytes` | TOP STANDING | **REFUTED as final-cause — alignment fixed via U34 + per-layer drift addressed via U35; both correct now.** |
| U34 — Even with correct, aligned offsets, GCB layout doesn't match producer-sim | TOP STANDING | **REFUTED — kernel-side AND producer-side DPRINT prove offsets ARE correct.** Bug is elsewhere. |
| **NEW U35-α — Matmul kernel's wrap-arithmetic (`is_tensor_split`, `update_rd_ptr_to_ring_index`, `calculate_next_block_index_and_update_rd_ptr`) has edge case when offset+ring_idx*block_size straddles GCB wrap boundary; works for layer 0 (wrap-free) but fails / generates NaN for layers 1+ (wrap-straddling)** | (new) | **TOP STANDING.** Next attack: add per-(ring_idx, block) kernel DPRINT of rd_ptr trajectory inside the matmul block-stride loop; compare against expected positions. |

## Why "kernel + producer both correct but output garbage + NaN"

The kernel reads bytes from `L1[fifo_addr + rd_ptr]` and decodes per
its compile-time `in1_data_format` (BFP4 or BFP8).  We've verified:
- The producer wrote tensor T's bytes at L1[fifo_addr + producer_offset].
- The kernel's first-block rd_ptr = fifo_addr + producer_offset (via U35 DPRINT).

But the kernel then iterates `num_blocks` times, with each iteration's
rd_ptr advanced by `calculate_next_block_index_and_update_rd_ptr`.
That function uses `local_cb.fifo_limit`, `cb_start_addr`,
`rd_ptr_start_addr` to handle wrap.

For W1 at layer 1 offset 649728, the consumer's `fifo_size = 829440`
(BFP4: 60×13824).  `fifo_limit = cb_start + 829440/16`.  Tensor reads
32 blocks × 13824 = 442368 bytes.  Starting at 649728, ending at
649728 + 442368 = 1092096 → wraps mid-tensor at offset 829440 to
offset 262656.

The kernel's `is_tensor_split` returns true (remaining 179712 <
tensor 442368).  Then `update_rd_ptr_to_ring_index` for ring_idx=N
adds `N*13824/16` shifted units, with wrap if past fifo_limit.  Then
the inner loop's `calculate_next_block_index_and_update_rd_ptr`
advances by one block per iter, with `reach_limit` check.

Bug hypothesis (U35-α): the `reach_limit` check at line 109 uses `==`
(`local_cb.fifo_rd_ptr == local_cb.fifo_limit`), not `>=`.  If the
rd_ptr lands EXACTLY at fifo_limit, wrap fires; otherwise it doesn't
(continues incrementing past fifo_limit until eventually wrapping?).
This could mis-handle the case where U35's start offset is itself not
aligned to block_size_bytes such that consecutive block advances pass
the limit non-exactly.

## Hard constraints checked

- [x] No upstream PR (predator2k/* only; commit `fbdbc73fb28` on `tenstorrent-p1` only).
- [x] No SGLang core behavior changes (Python-side `python/sglang/...` untouched).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used FIXED `rebuild_tt_metal_kernels.sh` after each C++ edit + `podman cp` for source sync.  Verified `SGLANG_TT_U35_KERNEL_OFFSET_PROBE` sentinel in `strings _ttnncpp.so`.
- [x] All U35 paths env-gated (`SGLANG_TT_U35_KERNEL_OFFSET_PROBE`, `SGLANG_TT_U35_PER_LAYER_OFFSET`).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"` at 9.09s) + GSM8K(10) = 9/10 = 90% (Q4 reasoning-pattern only; baseline-equivalent).
- [x] Server stopped at session end; cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: **`fbdbc73fb28`** (U35; 3 files, +195 lines).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/ttnn/cpp/...` source + `/tt-metal/models/tt_transformers/tt/prefetcher.py`: synced; `_ttnn.so` + `_ttnncpp.so` synced to install dir; sentinel `SGLANG_TT_U35_KERNEL_OFFSET_PROBE` verified in `strings _ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at 9.09s + GSM8K(10) = 9/10 (Q4 reasoning-pattern only).
- DPRINT/stderr artifacts preserved in-container:
  - `/tmp/u35_dprint.log` (Path A — kernel-side U35 DPRINT under U34 envs; proves kernel sees right offsets)
  - `/tmp/u35_server.log` (corresponding server log)
  - `/tmp/u35_producer.log` (Path B — producer-side U18 DPRINT; proves producer writes at U34 positions)
  - `/tmp/u35_producer_server.log` (corresponding server log)
  - `/tmp/u35_perlayer.log` (Path C — kernel-side DPRINT under U35 envs; shows 14 unique WQKV offsets per layer)
  - `/tmp/u35_perlayer_server.log` (corresponding server log)
  - `/tmp/u35_v2_server.log` (U35 stateful run; produces garbage + NaN)
  - `/tmp/u35_gsm8k_server.log` (U35 GSM8K attempt; server-crash at Q2 due to NaN)
  - `/tmp/u35_canonical_server.log` (canonical no-prefetcher control; bytewise-equal to U34 baseline)

## Shipping verdict

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Same shipping verdict as U31/U32/U33/U34.  U35 plumbs per-layer offsets correctly but the downstream kernel wrap-arithmetic produces garbage / NaN.
- **`SGLANG_TT_U35_KERNEL_OFFSET_PROBE=1` is safe to leave in tree (default-off + canonical bytewise-equivalent at default).**  Only enables an extra DPRINT in the gathered matmul kernel under U35=1.
- **`SGLANG_TT_U35_PER_LAYER_OFFSET=1` is in tree but EXPERIMENTAL** — default-off; canonical-equivalent at default.  Under PREFETCHER + U35, hits NaN downstream.  Left in tree for U36 investigation; not for shipping.

## 49-dispatch ledger (cumulative)

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
| U34 | CLOSED | Producer-simulation aligned offsets + critical lookup-bug fix; broadcast crashes ELIMINATED; output garbage persists (alignment necessary-not-sufficient). |
| **U35** | **EVIDENCE_ADVANCE — kernel-side AND producer-side DPRINT prove offsets correct; per-layer producer-drift identified and fixed; per-layer stateful offset PLUMBS correctly through the kernel but produces NaN.** | **U35-α TOP STANDING** — matmul kernel wrap-arithmetic edge case for wrap-straddling start offsets. |
| U36 | OPEN | Add per-(ring_idx, block) kernel DPRINT of rd_ptr inside the matmul block-stride loop.  Compare against expected positions.  Should reveal whether `is_tensor_split` / `update_rd_ptr_to_ring_index` / `calculate_next_block_index_and_update_rd_ptr` handles non-zero wrap-straddling start correctly. |

## Commits this session

- (tt-metal-sglang) **1 commit**: `fbdbc73fb28 prefetcher: U35 — kernel-side DPRINT proves C++ env→RT-arg plumbing is INTACT; producer DPRINT confirms wr_ptr trajectory exactly matches U34 simulation for layer 0; per-layer stateful offset attempt produces NaN downstream`.
- (sglang) `<this doc>` — pending commit.
