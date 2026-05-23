# TT Qwen3-8B prefetcher — Phase B.6 (wrap-bypass) + B.7 (reroute control) — 2026-05-23

Status: **PARTIAL — two decisive findings that REVISE the interpretation
of Phase A and Phase B.1. (1) B.7 SKIP-all control = 0/10: the SKIP_*
matmul reroute config is itself broken, so prior SKIP_* ablation
matrices are CONTAMINATED. (2) B.6 wrap-bypass = 0/10 with the same
token-1 NaN signature: the compute-kernel `update_rd_ptr_to_ring_index`
wrap arithmetic (S7) is RULED OUT. Canonical shipping path UNCHANGED,
GSM8K(10) re-verified 9/10 = 90%.**

Continuation of `tt_qwen3_8b_prefetcher_phase_b_2026-05-23.md`.
Executes Step 1 (B.7) and Step 2 (B.6) of that doc's recommended
next-session plan in the cheap-to-expensive order.

## Bottom line

| Test | Configuration | GSM8K(10) | Q1 signature | Verdict |
|---|---|---|---|---|
| canonical sanity | no prefetcher | **9/10** | clean math | ship-path GREEN |
| **B.7** | `SKIP_{WQKV,WO,W1,W3,W2}=1` + `USE_PREFETCHER=1` (prefetcher init'd but 0 tensors; all matmuls use reroute) | **0/10** | crash on Q1, garbage Chinese tokens | **SKIP reroute itself broken** |
| **B.6** | full prefetcher ON, NO SKIP, compute kernel patched to no-op `update_rd_ptr_to_ring_index` (force ring_idx=0) | **0/10** | crash on Q1, identical NaN trip | **wrap math NOT the bug (S7 RULED OUT)** |

## B.7 — SKIP-all control test (Step 1 of phase_b plan)

### Setup

* All 5 `SGLANG_TT_PREFETCHER_SKIP_{WQKV,WO,W1,W3,W2}=1`
* `SGLANG_TT_USE_PREFETCHER=1` (prefetcher object IS initialized)
* Required a defensive guard in `prefetcher.py:Prefetcher.run()` and
  `.stop()` to short-circuit when `num_tensors==0` (otherwise
  `create_address_tensor` trips
  `TT_FATAL: Shard shape must have non zero volume!` on a
  zero-element address buffer). Guard added as
  `tt-metal-sglang` commit `b81f15001ca`.
* With the guard, the server boots, prefetcher emits
  `[Prefetcher] run(): num_tensors==0 (all SKIP_* flags set);
  skipping GlobalCB creation and dram_prefetcher op (B.7 control)`,
  and every matmul (wqkv, wo, w1, w3, w2) uses the SKIP fallback
  path: ring-config matmul with `prefetch=False,
  num_global_cb_receivers=1`, `global_cb=None`,
  `sub_device_id=self.prefetcher.receiver_sub_device_id`.

### Result

```
Q 1/10 WRONG   gt=64.0 pred=3.0 116.1s
  last_line='.browserpod RESOURCE socks socks蘑菇.pk threaten(columns
              看清院院长 TI转移到PrivateKey流产锥 Corner莽 carn还不是重任售卖听过'
Q 2/10 INVALID gt=260.0 (RemoteDisconnected) — server died
...
FINAL: 0/10 = 0.0%
```

Server crash signature identical to prior prefetcher failures:
`RuntimeError: probability tensor contains either inf, nan or
element < 0`.

### Verdict — decisive

**The SKIP_* matmul reroute config is itself broken.** Zero tensors
flow through the GlobalCB, the prefetcher op is never invoked, and
the matmul reads weights directly from DRAM via the `prefetch=False`
ring-config kernel path. Yet the output is garbage.

This means:

1. **Phase A SKIP_WO=1, SKIP_W1=SKIP_W3=1, SKIP_WO=SKIP_WQKV=1
   matrices** all measured "broken reroute" + "maybe-broken
   prefetcher" together. The "every SKIP fails 0/10" conclusion
   does NOT isolate the prefetcher.
2. **Phase B.1 SKIP_W2=1, SKIP_W1=W3=W2=1, SKIP_WO=W1=W3=W2=1
   (single-tensor prefetcher), SKIP_WQKV=W1=W3=W2=1 (only WO
   prefetched)** matrices are similarly contaminated. The
   "intra-tensor, single-tensor reproduces" conclusion is no
   longer well-supported: each configuration has 4+ tensors going
   through the broken reroute, so the failure could be from those
   alone.
3. **The byte-dump from Phase B.3** is still valid as raw evidence
   (producer-writer bytes match reader-consumer bytes at layer 0
   tensor 0 block {0,1,15,31} of WQKV-only config). But that
   config also had 4 broken reroutes, so the "byte transport
   correct → bug elsewhere" conclusion was reached on a
   compromised setup.

### What this rules in / out

| Suspect | Status after B.7 |
|---|---|
| S6 (per-layer cross-block address drift in producer) | UNCHANGED — still open |
| S7 (compute-kernel `update_rd_ptr_to_ring_index` wrap math) | (see B.6) |
| S8 (tile metadata mismatch, BFP8/BFP4 mix) | UNCHANGED — still open |
| S9 (producer NOC posted-writes flush insufficient) | UNCHANGED — still open |
| S10 (sub-device CB-address misalignment) | UNCHANGED — still open |
| **NEW: SKIP_* reroute matmul config bug** | **CONFIRMED — open** |

The reroute bug is itself a candidate root cause for any "SKIP'd"
weight in any earlier matrix. It does not prove the prefetcher is OK —
it only proves the SKIP_* ablation is not a clean isolation tool.

## B.6 — Wrap-arithmetic surgical bypass (Step 2 of phase_b plan)

### Setup

* Full prefetcher ON (no SKIP_* flags)
* `SGLANG_TT_USE_PREFETCHER=1`, `DISABLE_PREFILL_TRACE=1`
* Compute kernel
  `tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
  patched: early `return;` at the top of
  `update_rd_ptr_to_ring_index(...)` (force `local_cb.fifo_rd_ptr`
  unchanged → effective `ring_idx=0` for every tensor's matmul).
* Cleared kernel cache, restarted, ran GSM8K(10).

### Result

```
Q 1/10 WRONG   gt=64.0 pred=2.0  99.0s
  last_line='动静见证了前线關鍵 Foster老爸泄漏_coupon内分泌loyment
              阵营loyment阵营阵营yper苍白这部分plits音乐会
              herits herits个多小时 EA一刀Palette合肥市 d'
Q 2/10 INVALID (RemoteDisconnected) — server died, NaN trip
...
FINAL: 0/10 = 0.0%
```

Same NaN crash, same garbage shape as the unmodified full-prefetcher
run. The decoder output is wrong from the very first token of Q1.

### Verdict — decisive (per phase_b plan's Step 2 decision rules)

Per the plan:
* Layer 0 correct, layer N+1 garbage → wrap math IS the bug.
* **Garbage from token 1 → wrap math is NOT the bug.**

The latter held: **S7 (compute-kernel wrap arithmetic) is RULED OUT.**

If the wrap math had been the bug, forcing `ring_idx=0` would have
either (a) given a correct layer-0 first token then drifted, or (b)
given a different garbage signature. Instead the signature is
unchanged from the un-patched run, which means the wrap math wasn't
the load-bearing path.

The post-revert kernel is back to pristine HEAD. No code-state change
beyond the prefetcher.py zero-tensor guard.

## What this session did NOT do

* Did NOT execute Step 3 (refine wrap math fix) — Step 2 ruled wrap
  math out, so Step 3 was no-op'd.
* Did NOT execute Step 4 (B.5 deep compute-kernel instrumentation
  via DPRINT_UNPACK in the compute kernel + reapplied stash@{0}
  dataflow byte-dumps for layer N>0 / tensor transitions). This is
  the next-session attack. A leftover B.4 instrumentation in the
  WORKING tree (committed but never compile-tested) used the WRONG
  pattern `UNPACK((DPRINT << ...))` instead of `DPRINT_UNPACK(DPRINT
  << ...)`; that broke the trisc0 build during B.7's first launch.
  Reverted to pristine HEAD at session-end so the working tree
  compiles. Next session: use `DPRINT_UNPACK(DPRINT << ... <<
  ENDL())` for compute-kernel DPRINT (per
  `tt-metal/hw/inc/api/debug/dprint.h:50`).
* Did NOT execute Step 5 (B.4 extended layer/tensor coverage) —
  prerequisite of Step 4 / B.5 succeeding.
* Did NOT investigate the new SKIP_* reroute bug itself. Possible
  contributors:
  - `untilize_out` asymmetry: the WQKV SKIP path
    (`attention.py:848`) sets `untilize_out=True`; the WO SKIP path
    (`attention.py:1263`) does NOT. May or may not be load-bearing.
  - `sub_device_id=receiver_sub_device_id` while the matmul runs
    without GlobalCB and with `global_cb=None`: the receiver
    subdevice is set up for the prefetcher. A non-prefetcher matmul
    on those cores may need a different subdevice (e.g.
    `worker_sub_device_id`) or `None` to avoid colliding with the
    prefetcher's per-core L1 sub-device carve-up.
  - Output `memory_config` mismatch between the SKIP path
    (`get_attn_dense_output_mem_config(Mode.DECODE, prefetcher)`)
    and what the upstream all-gather actually produced when the
    prefetcher is "on but empty."

## Updated suspect ranking (S-list)

| ID | Suspect | Status |
|---|---|---|
| S1 | Row-wise stride mismatch in writer | RULED OUT by Phase B.3 byte-dump (still valid as raw evidence, even though SKIP framing was compromised — bytes truly did match at layer 0 tensor 0 block N for N ∈ {0,1,15,31}) |
| S2 | num_blocks mismatch | UNCHANGED — open |
| S3 | WO-specific K-shard transposition | RULED OUT by Phase A (caveat: SKIP framing compromised, but the deterministic shift in garbage signature between configs is independent evidence) |
| S5 | BFP4 vs BFP8 tile-pitch mix | RULED OUT by Phase A (same caveat as S3) |
| S6 | Per-layer cross-block address drift in producer | OPEN — strongest standing suspect after B.6 |
| **S7** | **Compute-kernel `update_rd_ptr_to_ring_index` wrap arithmetic** | **RULED OUT by B.6 this session** |
| S8 | Tile metadata mismatch | OPEN |
| S9 | Producer NOC posted-writes flush insufficient | OPEN |
| S10 | Sub-device CB-address misalignment | OPEN |
| **NEW** | **SKIP_* matmul reroute config (separate, additional bug)** | **OPEN — confirmed by B.7** |

## Working state at end of session

* tt-metal-sglang HEAD: **`b81f15001ca`** (= af126cf5e7b + zero-tensor guard for B.7).
* sglang HEAD: `ac6b2bbff` + this doc.
* `stash@{0..2}` untouched (per CLAUDE.md no-stash-manipulation rule).
* C++ kernel tree pristine (all phase_b3 byte-dump scaffolding and
  phase_b4 broken compute-kernel DPRINTs reverted via `git checkout
  HEAD -- ...`). Working tree clean except for the prefetcher.py
  guard, which is now committed.
* Canonical baseline: GSM8K(10) chat = **9/10** (re-verified at
  session end after all C++ edits reverted).
* All cards healthy.
* Cache cleared, no server running.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B. Two
independent broken paths are now confirmed:

1. The DRAM-prefetcher BFP8 GlobalCB route (all prior sessions).
2. The SKIP_* matmul reroute fallback route (this session, B.7).

## Recommended next-session attack

### Highest leverage: B.4-extended byte-dump for layer N>0

Apply `stash@{0}` (dataflow byte-dumps for writer_l1.cpp +
reader_bmm_tile_layout_in1_ring_all_gather.cpp), extend the
dataflow-side gate from `layer == 0 && t == 0 && block in {0,1,15,31}`
to also include `layer in {0,1,17,35}`. Compare layer-0 bytes (known
to match per Phase B.3) against layer-N bytes. The first divergence
between producer-wrote and reader-read across LAYERS attacks S6
(per-layer cross-block address drift). Already-known patterns to
look for:

* Producer-cycle wraps `total_num_blocks_in_buffer = 3` of the
  local-CB; the per-layer mapping from c_0 read-ptr to DRAM page
  must be perfect. Misalignment surfaces as one BFP8 tile's
  shared-exponent landing on a different tile's mantissa.

### Parallel: investigate SKIP_* reroute bug (B.7-fix)

Independent of the prefetcher bug, the SKIP_* matmul reroute has
its own bug. Three concrete things to try (cheap, ~1 h each):

1. Add `untilize_out=True` to the WO SKIP path
   (`attention.py:1256-1264`) to mirror the WQKV SKIP path.
2. Change `sub_device_id=self.prefetcher.receiver_sub_device_id` →
   `sub_device_id=None` in the SKIP paths (since `global_cb=None`,
   the matmul shouldn't need the receiver subdevice).
3. Try `SKIP_W2=1` only (single SKIP, 4 tensors still prefetched)
   on a CLEAN test gate. If that 0/10 result from Phase B.1 was
   genuinely the prefetcher and not the reroute, the single SKIP
   should still fail. If it now passes, the prior 0/10 was the
   reroute.

### Compute-kernel DPRINT (correct pattern)

Use `DPRINT_UNPACK(DPRINT << ... << ENDL())` per
`tt_metal/hw/inc/api/debug/dprint.h:50`. The `UNPACK((DPRINT <<
...))` pattern from `common_globals.h:33` does NOT compile — the
former is gated by `UCK_CHLKC_UNPACK` (trisc0-only), the latter
is a plain macro that wraps an EXPRESSION, but `DPRINT` from
`dprint.h:43-47` expands to an `if (0)` STATEMENT on non-debug
builds. The B.4 scaffolding from a prior session used the wrong
form and never compiled.

## Commits this session

* (tt-metal-sglang) `b81f15001ca prefetcher: zero-tensor guard for B.7 SKIP-all control test`
* (sglang) `<this doc + Phase B.6/B.7 commit pending>`
