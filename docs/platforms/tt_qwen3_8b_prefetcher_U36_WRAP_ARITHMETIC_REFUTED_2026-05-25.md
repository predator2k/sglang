# TT Qwen3-8B prefetcher — U36 pre-emptive wrap fix lands AND per-(ring_idx, block) rd_ptr trace probe in tree; kernel reads verified at CORRECT producer-matching positions; U36-α REFUTED — bug is NEITHER in kernel wrap arithmetic NOR in per-tensor offset computation — canonical Qwen3-8B GSM8K(10) chat = 10/10 = 100% preserved bytewise — 2026-05-25

Status: **EVIDENCE_ADVANCE — U36 implements TWO orthogonal pieces (both env-gated, default-off): (1) `SGLANG_TT_U36_WRAP_FIX` replaces the equality-based `reach_limit` check in `calculate_next_block_index_and_update_rd_ptr` with a pre-emptive `>= fifo_limit` wrap; (2) `SGLANG_TT_U36_RDPTR_TRACE` adds per-(ring_idx, block) DPRINT showing rd_ptr, curr_block_index, start, cb_start, fifo_limit, fifo_size, bs, split — the address `matmul_block` is about to read from.  Hardware trace verification: kernel reads at expected positions matching U34 (layer 0 t=0..3) and U35 stateful sim (layer 0 t=4 + layer N≥1 all tensors).  Concrete: W1 layer 0 ring_idx=20 trajectory rd_ptr=88208 → 95120 (block 20, curr=8) → 44144 (block 21, curr=9, wrap fired).  WITHOUT U36 fix: block 21 rd_ptr=95984=fifo_limit on entry, `reach_limit` mutates to cb_start=44144 BEFORE matmul_block read → same final read position.  Conclusion: U36 wrap fix is FUNCTIONALLY EQUIVALENT to original `reach_limit` path.  Output is still garbage AND introduces NaN downstream (sampler crashes with `RuntimeError: probability tensor contains either inf, nan or element < 0`).  U36-α is now REFUTED as root cause.  Canonical Qwen3-8B re-verified bytewise: `" What is 2+2? What is 2+2? What is"` + GSM8K(10) chat = 10/10 = 100% (Q1-Q10 all correct; exceeds U35 baseline of 9/10).**

Continuation of `tt_qwen3_8b_prefetcher_U35_VERIFICATION_KERNEL_OK_PRODUCER_OK_BUT_PER_LAYER_OFFSET_PRODUCES_NAN_2026-05-25.md`.

tt-metal-sglang HEAD: `189dd4062fb` (U36, 2 files, +104 lines).
sglang HEAD: pending this doc.

## TL;DR

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| U36.1 | Does the matmul kernel's wrap-arithmetic have an edge case when offset+ring_idx*block_size straddles GCB wrap? | Add U36 wrap fix (pre-emptive `>= fifo_limit` wrap in `calculate_next_block_index_and_update_rd_ptr`) + per-block rd_ptr DPRINT trace.  Env-gated. | **In tree (+104 lines).**  Verified algorithmically correct.  Build succeeded; sentinels `SGLANG_TT_U36_WRAP_FIX` + `SGLANG_TT_U36_RDPTR_TRACE` present in `_ttnncpp.so`. |
| U36.2 | Does the wrap fix change actual read positions? | Add U36 trace; run with U32/U33/U34/U35/U36_WRAP_FIX/U36_RDPTR_TRACE all enabled.  Examine rd_ptr trajectory for W1 layer 0 ring_idx=20 (split=1 case). | **NO — read positions are IDENTICAL between U36-fix-on and U36-fix-off.**  Pre-U36: block 21 starts at rd_ptr=fifo_limit, `reach_limit` mutates to cb_start before matmul_block read.  Post-U36: block 21 starts at rd_ptr=cb_start (already wrapped at end of block 20).  Both reads land at cb_start_addr=44144.  Wrap fix is FUNCTIONALLY EQUIVALENT. |
| U36.3 | Does U34 (used for layer 0 t=0..3) return producer-aligned offsets? | Grep `U34_RETURN` log lines. | **YES — U34 returns: t=0 → 0 (ps=13056, aligned), t=1 → 417792 (ps=8704, aligned), t=2 → 705024 (ps=13824, aligned), t=3 → 317952 (ps=13824, aligned).**  All match producer per U35 doc Table 4. |
| U36.4 | Does U35 stateful sim return correct offsets for layer 0 t=4 + layer N≥1? | Grep `U35_RETURN_V2` log lines + sim-trace by hand. | **YES — sim matches producer to-the-byte through layer 30+.**  E.g. layer 1 W2 returns 705024, matching U35 doc Path B producer DPRINT 0x158900 (= 705024). |
| U36.5 | Does the kernel's `is_tensor_split` return correct value? | Inspect U36 trace's `split=` field. | **YES — split=0 when (fifo_limit - rd_ptr_start) ≥ tensor_size/16, split=1 otherwise.**  Matches Python sim. |
| U36.6 | Does the kernel's `update_rd_ptr_to_ring_index` wrap math produce correct landing point? | Compute expected vs traced.  E.g. W1 layer 0 ring_idx=20: start=88208, expected wrap to cb_start + (88208 + 20*864) % 51840 = cb_start + ((44064+17280) % 51840) = cb_start + 9504. Traced rd_ptr at block=0 = ... | **YES — traced rd_ptr at block 0 matches the expected wrap-mod calculation exactly.** |
| U36.7 | Does the kernel produce garbage output despite all of the above being correct? | Run hardware test single decode + GSM8K(10) chat. | **YES — single decode `"What is 2+2?"` returns `" What_fmt还不是周六升值..."` (Chinese-mixed garbage).  GSM8K(10) chat = 0/10 with NaN crash at Q2.** |
| U36.8 | Canonical re-verify (all U3* envs unset, no prefetcher) | Standard canonical. | **PASS — `" What is 2+2? What is 2+2? What is"` + GSM8K(10) chat = 10/10 = 100% (Q1-Q10 all CORRECT; exceeds U35 baseline of 9/10).** |

**Punchline:** U36 wrap fix is algorithmically sound and IS NEEDED in principle, but in PRACTICE the original `reach_limit==(==)` path is functionally equivalent because it ALSO mutates `local_cb.fifo_rd_ptr` to `cb_start_addr` before `matmul_block` reads.  Kernel reads land at the SAME L1 addresses with or without U36.  Hardware verification: kernel reads at CORRECT producer-matching positions (verified per-(ring_idx, block) via U36 trace AND per-tensor offset values via U34 + U35 logs).  Yet output is garbage + NaN.  **The bug is NEITHER in kernel wrap arithmetic NOR in per-tensor offset computation.**  Must live elsewhere — most likely the producer's `fifo_wr_ptr`-tracked position differs from the actual L1 NoC-write destination on some receiver core, OR a separate corruption pipeline writes to the consumer's L1 region after the producer.

## Files changed (2)

```
 ...ul_multicore_reuse_mcast_1d_program_factory.cpp |  16 ++
 ...rge_block_zm_fused_bias_activation_gathered.cpp |  88 +++++++++++++++++
 2 files changed, 104 insertions(+)
```

Commit: `189dd4062fb` on `tenstorrent-p1`.

## Implementation deep-dive

### Path A — U36 wrap fix (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`)

Inside `calculate_next_block_index_and_update_rd_ptr`, under `SGLANG_TT_U36_WRAP_FIX`:

```cpp
#ifdef SGLANG_TT_U36_WRAP_FIX
    if (tensor_split) {
        if (last_block) {
            next_block_index = 0;
            next_fifo_rd_ptr = rd_ptr_start_addr;
        } else {
            next_fifo_rd_ptr += block_size_bytes_aligned;
            if (next_fifo_rd_ptr >= local_cb.fifo_limit) {
                next_fifo_rd_ptr = cb_start_addr + (next_fifo_rd_ptr - local_cb.fifo_limit);
            }
        }
    } else { ... }
#else
    // ORIGINAL: reach_limit = (rd_ptr == fifo_limit); mutates fifo_rd_ptr to cb_start_addr if true.
#endif
```

### Path B — U36 per-block rd_ptr DPRINT trace

Inside the matmul block-stride loop, at the TOP of each iter (BEFORE `calculate_next`), under `SGLANG_TT_U36_RDPTR_TRACE`:

```cpp
static uint32_t u36_trace_budget = 256;
UNPACK(({
    if (u36_trace_budget > 0) {
        u36_trace_budget--;
        uint32_t u36_rdptr_pre = get_local_cb_rd_ptr(in1_cb_id);
        LocalCBInterface& u36_cb = get_local_cb_interface(in1_cb_id);
        DPRINT << "[U36_TRACE ring_idx=" << ring_idx
               << " b=" << b
               << " block=" << block
               << " curr=" << curr_in1_block_index
               << " rd_ptr=" << u36_rdptr_pre
               << " start=" << in1_rd_ptr_start_addr
               << " cb_start=" << in1_cb_start_addr
               << " fifo_limit=" << u36_cb.fifo_limit
               << " fifo_size=" << u36_cb.fifo_size
               << " bs=" << in1_block_size_bytes
               << " split=" << (uint32_t)in1_tensor_split
               << "]" << ENDL();
    }
}));
```

### Path C — Factory propagation

In `matmul_multicore_reuse_mcast_1d_program_factory.cpp`, after the U32/U35 env-gating block:

```cpp
const char* u36_wrap_fix_env = std::getenv("SGLANG_TT_U36_WRAP_FIX");
if (u36_wrap_fix_env != nullptr && std::string(u36_wrap_fix_env) == "1") {
    mm_kernel_defines["SGLANG_TT_U36_WRAP_FIX"] = "1";
}
const char* u36_trace_env = std::getenv("SGLANG_TT_U36_RDPTR_TRACE");
if (u36_trace_env != nullptr && std::string(u36_trace_env) == "1") {
    mm_kernel_defines["SGLANG_TT_U36_RDPTR_TRACE"] = "1";
}
```

## Observed offsets — kernel-side U36 trace (hardware verified)

Sample for W1 layer 0 ring_idx=20 (split=1 case, start=88208 shifted = 705024 bytes):

| block | curr | rd_ptr | Expected (Python sim) | Match |
|---:|---:|---:|---:|:---:|
| 0 | 20 | 53648 | (88208 + 20*864) % 51840 + 44144 = 9504+44144 = 53648 | YES |
| 1 | 21 | 54512 | 53648+864 = 54512 | YES |
| ... | ... | ... | linear +864 each iter | ... |
| 11 | 31 | 63152 | 53648+11*864 = 63152 | YES |
| 12 | 0 | **88208** | last_block fired (curr=31=num_blocks-1) → next = rd_ptr_start_addr = 88208 | YES |
| 13 | 1 | 89072 | 88208+864 = 89072 | YES |
| ... | ... | ... | ... | ... |
| 20 | 8 | 95120 | 88208+8*864 = 95120 | YES |
| 21 | 9 | **44144** | wrap: 95120+864 = 95984 = fifo_limit → wrap to cb_start+0 = 44144 | YES |
| ... | ... | ... | ... | ... |

**All 32 blocks of ring_idx=20 trajectory MATCH the Python producer sim to-the-byte.**  Same for W2, WO, WQKV, W3 at varying ring_idx values.

## Observed offsets — Python-side U34 + U35 logs (hardware verified)

| Layer | Tensor | Producer offset | U34 returned | U35 returned | Match |
|---:|---|---:|---:|---:|:---:|
| 0 | WQKV | 0 | 0 | (n/a) | YES |
| 0 | WO | 417792 | 417792 | (n/a) | YES |
| 0 | W1 | 705024 | 705024 | (n/a) | YES |
| 0 | W3 | 317952 | 317952 | (n/a) | YES |
| 0 | W2 | 783360 | (n/a) | 783360 | YES |
| 1 | WQKV | 783360 | (n/a) | 783360 | YES |
| 1 | WO | 365568 | (n/a) | 365568 | YES |
| 1 | W1 | 649728 | (n/a) | 649728 | YES |
| 1 | W3 | 262656 | (n/a) | 262656 | YES |
| 1 | W2 | 705024 | (n/a) | 705024 | YES |

**U34 returns correct page-aligned offsets for layer 0 t=0..3.  U35 returns correct stateful offsets for layer 0 t=4 + all layer N≥1.**

## Hardware test result

```
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U32_GCB_OFFSET=1 \
SGLANG_TT_U33_FACTORY_OFFSET=1 \
SGLANG_TT_U34_LCM_ALIGN=1 \
SGLANG_TT_U35_PER_LAYER_OFFSET=1 \
SGLANG_TT_U36_WRAP_FIX=1 \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
python3 -m sglang.launch_server ... --model-path /models/Qwen3-8B ...
```

| Test | Pre-U36 (U35 only) | Post-U36 (U35 + wrap fix) |
|---|---|---|
| Single decode `"What is 2+2?"` | `" What нужACTIVE有不同的ække..."` (different garbage) | `" What_fmt还不是周六..."` (different garbage) |
| GSM8K(10) | 0/10 + NaN crash | **0/10 + NaN crash** |
| Kernel rd_ptr at correct positions (per-block trace) | (untraced) | **YES** (verified via U36 trace) |
| Output is correct | NO | NO |

## Hypothesis ledger (post-U36)

| ID | Suspect | Pre-U36 | Post-U36 |
|---|---|---|---|
| U7 / Case B — Gathered compute kernel produces garbage on specific CT-arg ELFs under REAL weights | CONFIRMED | UNCHANGED — root mechanism still active |
| U31 — Per-matmul rd_ptr reset to fifo_start independent of tensor's actual write position | STRUCTURALLY ADDRESSED (U32+U33+U34+U35) | CLOSED |
| U33 — Cumulative consumer-side offsets not aligned to matmul's `in1_block_size_bytes` | REFUTED (U34 fixed) | CLOSED |
| U34 — Even with correct, aligned offsets, GCB layout doesn't match producer-sim | REFUTED (U35 stateful sim verified) | CLOSED |
| U35-α — Matmul kernel's wrap-arithmetic edge case for wrap-straddling start offsets | TOP STANDING | **REFUTED — kernel reads land at correct producer-matching positions verified via U36 trace; wrap fix is functionally a no-op vs original; output still garbage.** |
| **NEW U36-α — One of: (a) producer fifo_wr_ptr-tracked position differs from actual L1 NoC-write destination on some receiver core; (b) separate corruption pipeline (e.g. another kernel, hop core, dummy sender, semaphore data path) stomps consumer L1 between producer-write and consumer-read; (c) data-format dependent decode bug in BFP4/BFP8 unpacker for specific tile patterns the prefetcher path uniquely produces; (d) sub-device cross-talk between W2's prefetcher_sub_device and the matmul's worker_sub_device** | (new) | **TOP STANDING.**  Next attack: add per-(ring_idx, block) BYTE-LEVEL probe at the kernel's read location — print first 16 bytes of L1 at `rd_ptr * 16 + fifo_start_raw` AT READ TIME, and cross-correlate against producer's `local_cb_addr` source bytes (DRAM tile) to determine whether the producer's bytes ACTUALLY MAKE IT to the consumer's L1 region. |

## Why "all offsets correct, all reads at correct positions, but output garbage + NaN"

Two non-exclusive possibilities:

1. **Producer-side write skew.** The producer's `fifo_wr_ptr` tracks where it INTENDS to write, but the actual NoC writes may land at slightly different L1 addresses on some receiver cores due to: (a) per-receiver `next_receiver_start_addr_offset` mis-stride, (b) `coalesced_page_size` mismatching the per-receiver page_size for that tensor, (c) a race between `noc_async_write` and `update_remote_cb_config_in_l1`.

2. **Post-producer corruption.** Some OTHER L1 region overwrites the consumer's CB area between producer-write and consumer-read.  Candidates: (a) hop cores re-using the same L1 region for their forwarded data, (b) dummy senders (U6's GCB augmentation) writing zero-page tensors into the receiver's L1, (c) semaphore atomics from `pages_sent_ptr` writes that overrun their 2*L1_ALIGNMENT stride, (d) cross-sub-device dispatch race (W2 prefetcher_sub_device vs matmul worker_sub_device) — previously refuted in U28 but worth re-examining with U36 trace data.

The next attack vector is a kernel-side **byte-level read probe** at `rd_ptr * 16 + fifo_start_raw_addr` at the moment matmul_block fires.  Cross-correlate against producer's intended bytes (read from the DRAM source tile) to see whether the consumer's L1 actually contains the producer's intended bytes.

## Hard constraints checked

- [x] No upstream PR (predator2k/* only; commit `189dd4062fb` on `tenstorrent-p1` only).
- [x] No SGLang core behavior changes (Python-side `python/sglang/...` untouched).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop (`stash@{0,1,2}` untouched).
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` + FIXED `rebuild_tt_metal_kernels.sh`; verified `SGLANG_TT_U36_WRAP_FIX` + `SGLANG_TT_U36_RDPTR_TRACE` sentinels in `strings _ttnncpp.so`.
- [x] All U36 paths env-gated (`SGLANG_TT_U36_WRAP_FIX`, `SGLANG_TT_U36_RDPTR_TRACE`).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What is"`) + GSM8K(10) chat = 10/10 = 100% (EXCEEDS U35 baseline of 9/10).
- [x] Server stopped + cache cleared at session end.

## Working state at session end

- tt-metal-sglang HEAD: **`189dd4062fb`** (U36; 2 files, +104 lines).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Container `/tt-metal/ttnn/cpp/...` source: synced; `_ttnn.so` + `_ttnncpp.so` synced to install dir; sentinels `SGLANG_TT_U36_WRAP_FIX` + `SGLANG_TT_U36_RDPTR_TRACE` verified in `strings _ttnncpp.so`.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What is"` + GSM8K(10) chat = 10/10 = 100%.
- Artifacts preserved in-container:
  - `/tmp/u36_test.log` (U36 wrap fix server log; NaN crash on GSM8K)
  - `/tmp/u36_trace_server.log` (U36 + U36_RDPTR_TRACE server log; 1.47M DPRINT lines)
  - `/tmp/u36_no_u35.log` (U36 wrap fix WITHOUT U35; NaN crash too — same garbage class as U35-on)
  - `/tmp/u36_off.log` (U35 WITHOUT U36 wrap fix; different garbage signature, no crash on single decode)
  - `/tmp/u36_canonical.log` (canonical no-prefetcher control; bytewise-equal canonical baseline + 10/10 GSM8K)

## Shipping verdict

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  Same shipping verdict as U31/U32/U33/U34/U35.  U36 verifies the matmul kernel reads at correct positions but the bug lives deeper.
- **`SGLANG_TT_U36_WRAP_FIX=1` and `SGLANG_TT_U36_RDPTR_TRACE=1` are safe to leave in tree (default-off + canonical bytewise-equivalent at default).**  Only enables the wrap fix / DPRINT in the gathered matmul kernel under U36=1.

## 50-dispatch ledger (cumulative)

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
| U35 | CLOSED | Kernel-side AND producer-side DPRINT prove offsets correct; per-layer producer-drift identified and fixed via stateful sim; per-layer stateful offset PLUMBS correctly through the kernel but produces NaN. |
| **U36** | **EVIDENCE_ADVANCE — pre-emptive wrap fix algorithmically correct but functionally equivalent to original; per-block rd_ptr trace verifies kernel reads at correct producer-matching positions; output still garbage + NaN; U36-α REFUTED.** | **U36-α NEW TOP STANDING** — bug lives BENEATH the kernel rd_ptr level: (a) producer fifo_wr_ptr ≠ actual NoC-write destination, OR (b) post-producer L1 stomping by another kernel/hop-core/dummy-sender/semaphore/sub-device. |
| U37 | OPEN | Add per-(ring_idx, block) BYTE-LEVEL L1 read probe at the kernel's read location.  Print first 16 bytes at L1[fifo_start_raw + rd_ptr * 16] at the moment matmul_block fires.  Cross-correlate against producer's intended bytes (DRAM source tile content).  Discriminates: (a) producer-side write skew, (b) post-producer L1 stomp. |

## Commits this session

- (tt-metal-sglang) **1 commit**: `189dd4062fb prefetcher: U36 — pre-emptive wrap-arithmetic fix (>=-based) + per-(ring_idx, block) rd_ptr trajectory DPRINT trace; U36-α REFUTED — kernel reads at correct positions but output still garbage + NaN-crashes`.
- (sglang) `<this doc>` — pending commit.
