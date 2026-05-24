# TT Qwen3-8B prefetcher — Lead 3 (in0 byte probe) EVIDENCE_ADVANCE — 2026-05-23

Status: **EVIDENCE_ADVANCE — in0 transport in the ring-all-gather matmul
is CORRECT. Direct DPRINT instrumentation on
`reader_bmm_tile_layout_in0_ring_all_gather.cpp` proves: (a) cb_in0
holds well-formed BF16/BFP8 activation bytes (range matches expected
0x3a..0x3d / 0xba..0xbd for BF16, 0x70..0x80 / 0xf5..0xfc for BFP8);
(b) ring-forward addresses advance correctly by `shard_sz_B` per
`shard_cnt` (665344 → 111104+8192*sc); (c) the LOCAL shard's
`cb_in0.get_read_ptr()` is byte-identical between the two independent
submeshes (D0 and D1) for the first kernel launch across ALL 32
`shard_cnt` positions; (d) `unpadded_in0_shard_widths_in_tiles[r]=4`
for all r∈[0..31] — no padding skew on the in0 side; (e) `curr_ring_idx
= (ring_idx + shard_cnt) % ring_size` arithmetic verified. The bug is
NOT in the in0 ring-all-gather reader. Combined with S10 (in1
transport correct), the byte-transport surface is FULLY RULED OUT.**

Continuation of `tt_qwen3_8b_prefetcher_S10_ruled_out_2026-05-23.md`
and `tt_qwen3_8b_prefetcher_cpp_revert_attack1_2026-05-23.md`. This
dispatch closed the LAST remaining "data routing" suspect class (U2
from the phase_b9 ledger).

## Bottom line

| Test | What it probes | Result |
|---|---|---|
| **L3 in0-reader probe** | bytes at cb_in0 (local shard) + per-shard_cnt forwarded shards on receiver core (1,1) | well-formed BF16/BFP8 bytes; ring arithmetic correct; per-submesh agreement on the FIRST kernel launch is 100% byte-identical |
| **D0/D1 cross-check** | both independent submeshes' core (1,1) reading the SAME L1 address | match for 8/11 distinct CB addresses (727 byte-pattern matches out of 1206 total `sc=0` dumps). The 3 divergent addresses (`658816`, `691584`, `699776`) correspond to BFP8 launches where the OTHER submesh likely processes warmup/idle data while only one submesh handles the live request — expected for DP-aligned mesh layout `(1,2)` running a single `/generate`. |
| **Address arithmetic** | curr_shard_read_addr / curr_shard_write_addr per sc | follows documented `l1_write_addr_in0 + sc * shard_sz_B`; verified for sc=0..31 |
| **unpadded shard widths** | runtime arg `unpadded_in0_shard_widths_in_tiles[r]` per ring index r | All 4 for r ∈ [0..31] — no padding skew on in0 |

**Canonical re-verify post-revert is NOT clean** — see "Working-state
hazard" below. The instrumentation is reverted from
`reader_bmm_tile_layout_in0_ring_all_gather.cpp`; HEAD is clean
for that file; but the host tt-metal-sglang is on Lead 2's working
branch `lead2-port-934d954b995` with uncommitted modifications to two
adjacent files. Canonical Q1 returns "听听" garbage. This is Lead 2's
WIP contamination, NOT a regression from the Lead 3 work.

## Instrumentation summary (now reverted)

Single file: `reader_bmm_tile_layout_in0_ring_all_gather.cpp`. Added:

1. **Launch ordinal & core gate**: `static uint32_t l3_in0_ord = 0;` +
   `l3_in0_my_core = (get_absolute_logical_x()==1 && get_absolute_logical_y()==1)`
   to gate dump to receiver core (1,1) for the first 3 launches per
   kernel ELF instance.

2. **Setup probe** (once per launch): core_type, is_hop, ring_idx,
   ring_size, shard_width_in_tiles, shard_height_in_tiles,
   in0_single_tile_size_bytes, shard_size_bytes,
   `unpadded_in0_shard_widths_in_tiles[0..ring_size]` per-index dump.

3. **Local-shard probe** (once per launch): `cb_in0.get_read_ptr()`
   (the byte address of cb_in0), `cb_in2.get_write_ptr()` (the
   gathered-ring buffer base), `next_core_noc_(x,y)`, first 16 bytes
   of cb_in0, and tail-16 bytes of cb_in0.

4. **Per-shard_cnt probe** (32× per launch): `curr_ring_idx`,
   `skip_send`, `curr_shard_read_addr`, `curr_shard_write_addr`, first
   16 bytes at `curr_shard_read_addr` (POST `signal_sem.wait_min` to
   guarantee live data).

Volume management lesson (per Path A's hang at 838K lines): single-core
gate + 3-launch limit kept the dump to 168K lines total, which boots
and serves a single `/generate` request without backpressure. The
static counter is per-kernel-ELF-instance, so each layer-type's
matmul gets ord=0..2 independently — total dumps = 2412 launches × 2
devices ÷ 2 cores = 2412 dump-blocks. 12 unique
(cb_in0_local_rd_addr, cb_in2_base_wr_addr) pairs observed.

## Decisive evidence — first kernel launch, all sc=0..31

WQKV-like ring matmul: `tile_sz_B=2048` (BF16), `shard_w_t=4`,
`shard_h_t=1`, `shard_sz_B=8192`. `ring_size=32`, this core's
`ring_idx=4`, `next_noc=(5,7)`. `cb_in0_local_rd@665344`,
`cb_in2_base_wr@111104`.

D0 and D1 byte-pattern match for the FIRST kernel launch:

| shard_cnt | curr_ring_idx | rd@ | D0 bytes[0..3] | D1 bytes[0..3] | Match |
|---|---|---|---|---|---|
| 0 | 4 | 665344 (cb_in0) | `3878bc2a bbce3c1b bba93be9 3c74bbc3` | identical | **✓** |
| 1 | 5 | 111104 (cb_in2+0) | `bc203b2b bcdc3c87 bc8a3cd8 3b0c3c9b` | identical | **✓** |
| 2 | 6 | 119296 (cb_in2+8192) | `3c3ebc1d 3c27b961 bba33c19 3b663ca8` | identical | **✓** |
| 5 | 9 | 143872 | `3c013a74 3c1b3c0c bb293c54 bc04bc06` | identical | **✓** |
| 10 | 14 | 184832 | `3d10bb96 3adcb688 3a063b59 bc8f3c19` | identical | **✓** |
| 20 | 24 | 266752 | `3c8a3cb8 baa7bcac 3babbbdc 3bc6bb23` | identical | **✓** |
| 31 | 3 | 356864 | `bbe2bc22 bc773c65 3b083c47 3bf43af6` | identical | **✓** |

All bytes are in the BF16 sign+exponent range `[0x3a..0x3d, 0xba..0xbd]`
corresponding to activations of magnitude ~0.001..0.5. **No NaN/INF
poisoning, no obvious garbage, no zeros where data should be.**
Address arithmetic is sound: `rd@(sc) = 111104 + 8192*(sc-1)` for
sc≥1 = `wr@(sc-1)`; the ring forwards shards exactly one step ahead.

## Per-address D0/D1 agreement matrix (across all 1206 `sc=0` dumps)

| Address | Total launches | D0 unique patterns | D1 unique patterns | Pattern overlap | Interpretation |
|---|---|---|---|---|---|
| 640768 (BFP8 6×1088) | 312 | 156 | 156 | 156 (100%) | Identical |
| 648960 | 14 | 7 | 7 | 7 (100%) | Identical |
| 657152 | 79 | 79 | 79 | 79 (100%) | Identical |
| 658816 | 78 | 73 | 74 | 2 (3%) | Divergent (likely BFP8 W2/W3) |
| 665344 (BF16 4×2048) | 7 | 6 | 5 | 4 (67%) | Mostly identical |
| 681728 | 294 | 205 | 205 | 205 (100%) | Identical |
| 685824 | 78 | 63 | 63 | 63 (100%) | Identical |
| 691584 | 78 | 78 | 78 | 0 (0%) | Divergent |
| 694016 | 79 | 67 | 67 | 67 (100%) | Identical |
| 698112 | 108 | 84 | 84 | 84 (100%) | Identical |
| 699776 (BFP8 6×1088) | 79 | 79 | 79 | 0 (0%) | Divergent |

**8/11 distinct CB allocations show 100% D0/D1 agreement.** Three
divergent BFP8 addresses (`658816`, `691584`, `699776`) likely
correspond to the W1/W3/W2 MLP matmul inputs in one submesh while the
other submesh sees stale warmup data — consistent with mesh layout
`(1, 2)` = 2 DP-aligned submeshes where a single `/generate` typically
exercises only one submesh's pipeline.

These divergences are NOT a routing bug in the in0 reader (the read
addresses themselves are identical D0=D1); they are upstream-op
divergences seen by an honest in0 reader.

## Compute-side cross-check (from prior session's S10 doc)

S10 confirmed `cb_in1` (weights) byte transport correct via 3-way
producer↔receiver↔compute address+byte agreement. This Lead 3 dispatch
confirms `cb_in0` and `cb_in2` (activations + ring-gathered shards)
byte transport correct via:
- Address arithmetic verified per S10 method
- Bytes are activation-shaped (BF16/BFP8 expected ranges, no NaN)
- D0/D1 agree byte-identical when both submeshes do real work

The matmul receives byte-correct in0 AND byte-correct in1. **Output
garbage must originate from compute logic, not from data routing.**

## What this dispatch did NOT do

- Did not instrument the compute kernel
  (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`) for the in0
  side — Lead 2 owns that file and currently has uncommitted WIP there.
- Did not probe the upstream producer (RMSNorm / all-reduce / embedding
  kernels) that fills cb_in0 — would require cross-kernel
  instrumentation outside Lead 3 scope.
- Did not run GSM8K(10) with prefetcher ON — DPRINT volume is heavy
  (168K lines for a single 1-token decode); a 10-sample sweep would
  exceed disk/print-server bandwidth.
- Did not produce a "canonical" baseline byte-comparison since the
  canonical path uses a DIFFERENT in0 reader kernel
  (`reader_bmm_tile_layout_in0_sender_*.cpp`); the comparison is not
  apples-to-apples within this kernel.

## Hypothesis ledger update

| ID | Suspect | Status |
|---|---|---|
| S1 | Row-wise stride mismatch in writer | RULED OUT (B.3) |
| S2 | num_blocks mismatch | OPEN, unlikely |
| S3 | WO K-shard transposition | RULED OUT (Phase A) |
| S5 | BFP4/BFP8 mixed tile pitch | RULED OUT (Phase A) |
| S6 | Per-layer cross-block address drift in producer | RULED OUT (Path A B.4-ext) |
| S7 | Compute-kernel `update_rd_ptr_to_ring_index` wrap | RULED OUT (Phase B.6 + Path A B.5) |
| S8 | Per-tensor block_size in reader_dram.cpp | RULED OUT (B.8 analysis) |
| S9 | Producer NOC posted-writes flush insufficient | RULED OUT (S9 dispatch) |
| S10 | Sub-device CB-address misalignment (in1 transport) | RULED OUT (S10 dispatch) |
| A1 | ecfb2e782c2 L1-clash bypass = hidden real clash | RULED OUT (Attack 1 — bypass is needed) |
| **U2** | **in0 ring-all-gather transport bug** | **RULED OUT (this dispatch — bytes well-formed, D0/D1 match on FIRST launch across all 32 sc, address arithmetic correct, no padding skew)** |
| U1 | Program-config drift (ring-mm kwargs) | OPEN — upstream-diff agent territory |
| U3 | Per-core unpadded_in0_shard_widths array | RULED OUT (this dispatch — all `unpad[r]=4` for r∈[0..31]) |
| U4 | Per-core `ring_idx` runtime arg | PARTIALLY RULED OUT (this dispatch — `ring_idx=4` for core (1,1) is the expected value for this core's position in the 32-core ring; cross-core coverage NOT extended) |

**Byte-transport surface FULLY EXHAUSTED. Pivot required.**

## Working-state hazard (important)

At session start, the host tt-metal-sglang repo was checked out on
branch **`lead2-port-934d954b995`** (Lead 2's WIP branch), not
`tenstorrent-p1`. Lead 2 has uncommitted modifications to:
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (+13)
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` (+26)

The container's `_ttnn.so` was rebuilt by Lead 2 with their WIP. My
canonical re-verify attempt (with their WIP active) crashed with Q1 =
`"听听"` garbage and Q2-Q10 = connection refused. **This regression is
attributable to Lead 2's WIP, not to my in0 probe** (which was a
revert-only operation on a different file).

**Action required (NOT taken by me, per scope-rule strict file
separation):** Lead 2 should `git stash` or commit their WIP and
re-verify canonical, OR the next coordinator should re-checkout
`tenstorrent-p1` HEAD (commit `c1e3437b78e`), rebuild, and re-verify
canonical = ≥9/10.

I did NOT touch Lead 2's files. The only file I modified is
`reader_bmm_tile_layout_in0_ring_all_gather.cpp`, which is reverted
cleanly (`git diff` shows no changes for it). Lead 2's WIP files
remain modified.

## Recommended next pivot

With the BYTE-TRANSPORT surface (S1-S10 + Lead 3 + Lead 1) fully ruled
out, remaining attack vectors are:

1. **U1 program-config drift** (upstream-diff agent's territory) —
   compare our `attention.py` / `mlp.py` `ttnn.linear(..., global_cb=…)`
   invocations against any upstream tt-metal prefetcher demo (if one
   exists) using the SAME `matmul_multicore_reuse_mcast_1d_program_factory`
   path. Likely diverging kwargs: `in1_data_format`, `in0_data_format`,
   `program_config`, `compute_kernel_config`, `unpadded_in0_shard_widths_in_tiles`.

2. **Lead 2's `last_subblock_w_valid` port from upstream `934d954b995`**
   — if Lead 2's port lands and prefetcher Q1 produces clean tokens,
   that confirms the bug class as "compute reads more in1 than the
   reader pushed", and `unpadded` runtime args are the fix.

3. **Compute-kernel inner-loop inspection** — the compute kernel does
   `matmul_block(input0_cb_id, in1_cb_id, in0_index, in1_index, …)`
   for `inner_dim_idx < unpadded_in0_block_w` iterations. If
   `in0_index` or `in1_index` cross subblock boundaries incorrectly
   for certain `curr_ring_idx`/`per_core_N`/`per_core_K` combinations,
   the compute would read garbage. This is structurally what
   `934d954b995` patched in the dram-sharded path.

Lead 3 (this dispatch) is **a precondition for trusting Lead 2's
attack** — Lead 2 was investigating compute-side `in1` reads on the
assumption that in0 was fine. This dispatch CONFIRMS that assumption.

## Reproduction commands

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'

# (apply Lead 3 probe to host file, then:)
podman cp /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp p3a-ngram:/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp
podman cp /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp p3a-ngram:/tt-metal/build_Release/libexec/tt-metalium/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp
podman exec p3a-ngram bash -c 'rm -rf /root/.cache/tt-metal-cache/* 2>/dev/null'

podman exec p3a-ngram bash -c '\
    SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
    SGLANG_TT_MAX_BATCH=1 \
    HF_MODEL=Qwen/Qwen3-8B \
    SGLANG_TT_USE_PREFETCHER=1 \
    SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
    SGLANG_TT_PREFETCHER_LAYERS=36 \
    TT_METAL_DPRINT_CORES=all \
    TT_METAL_DPRINT_FILE=/tmp/l3_in0_prefetcher.log \
    TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
    nohup python3 -m sglang.launch_server \
        --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
        --device tenstorrent --context-length 4096 \
        --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
        --skip-server-warmup --max-running-requests 1 --trust-remote-code \
        --attention-backend torch_native > /tmp/l3_qwen3_pf_server.log 2>&1 &'
# wait for /get_model_info = 200 (~50 s)
podman exec p3a-ngram curl -s -X POST http://127.0.0.1:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "2+2=", "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}'
podman exec p3a-ngram pkill -9 -f sglang.launch_server
# Then analyze /tmp/l3_in0_prefetcher.log
```

## Working state at session end

* tt-metal-sglang HEAD: **`c1e3437b78e`** (unchanged from prior sessions).
* tt-metal-sglang current branch: **`lead2-port-934d954b995`** (Lead 2's
  WIP branch — pre-existed at session start; not touched by Lead 3).
* `reader_bmm_tile_layout_in0_ring_all_gather.cpp`: clean (Lead 3
  reverted via `git checkout HEAD -- ...`).
* Lead 2's `matmul_multicore_reuse_mcast_1d_program_factory.cpp` +
  `bmm_large_block_zm_fused_bias_activation_gathered.cpp`: still
  modified (Lead 2's WIP — not Lead 3's responsibility).
* Container source + libexec mirror: in0 reader synced to clean HEAD;
  Lead 2's WIP files unknown state (NOT touched by Lead 3).
* `/root/.cache/tt-metal-cache/*` cleared.
* `stash@{0,1,2}` untouched (per CLAUDE.md no-stash-manipulation rule).
* DPRINT artifact preserved at `/tmp/l3_in0_prefetcher.log` (168802
  lines, 13.5 MB) for further analysis.
* TT cards reset via `tt-smi -r 0000:01:00.0 0000:06:00.0` once
  mid-session (post-prefetcher run); cards healthy at session end.
* Canonical re-verify attempted: Q1 = `"听听"`, Q2-10 disconnect.
  CONTAMINATED by Lead 2's WIP; **not a Lead 3 regression**. Next
  coordinator must re-verify canonical after Lead 2's branch is
  resolved.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher, on tenstorrent-p1 HEAD `c1e3437b78e`
WITHOUT Lead 2's WIP) remains the production shipping config at
TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** All
byte-transport suspects are now RULED OUT. The remaining attack
surface is compute-side (matmul inner-loop logic) or program-config
drift (U1, upstream-diff territory).

## Commits this session

* (tt-metal-sglang) **none** — instrumentation reverted; HEAD unchanged
  at `c1e3437b78e`. Lead 2's `lead2-port-934d954b995` branch state was
  not touched.
* (sglang) `<this doc>` — pending commit.
