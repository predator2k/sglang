# TT Qwen3-8B prefetcher — Phase B.8 (SKIP_* reroute ring-grid fix + S8 BFP4 re-eval) — 2026-05-23

Status: **PARTIAL — Path B's SKIP_* reroute ring-grid fix engaged but did
NOT yield correctness (Steps 1+2 = 0/10 with CHANGED garbage signatures
vs Phase B.7 baseline). Step 3 S8 BFP4 test couldn't isolate the BFP4
hypothesis because the reroute path itself is still broken — W1+W3 (the
two BFP4 weights) trigger immediate NaN sampling crash through the
partially-fixed reroute. Path A's S8-as-stated (reader_dram.cpp:95
per-tensor stride hypothesis) is RULED OUT by code analysis: producer
and writer both advance by max_block_size in c_0 — the 24576-byte
inter-block gap for BFP4 is ignored by both, not propagated to the
receiver. Canonical (no prefetcher) re-verified 9/10 = 90% post-commit.**

Continuation of `tt_qwen3_8b_prefetcher_path_a_b4ext_2026-05-23.md`
(S6/S7 ruled-out, S8 raised) and
`tt_qwen3_8b_prefetcher_phase_b67_2026-05-23.md` (reroute brokenness).
Executes the Step 0-7 plan from the dispatch.

## Bottom line

| Step | Config | GSM8K(10) | Q1 signature | Verdict |
|---|---|---|---|---|
| 1 | `SKIP_W2=1` + Path B fix | 0/10 | `房东benh火力ifty主权悠悠 bambS齊 Canon 新规:'.$n Cox Gand Bennett` | partial fix; different from B.7 SKIP_W2's `观摩 thankfully心意风尚风尚 unveiled Nast` ⇒ engagement confirmed |
| 2 | SKIP_{WQKV,WO,W1,W3,W2}=1 + Path B fix | 0/10 | `<think>辘 Huff崇拜 Huff Huff Huff Huff Huffisma Huff Huff崇拜崇拜崇拜 Huff Huff崇拜崇拜崇拜 Huff Huff Huff Huff Huf` | partial fix; different from B.7 SKIP_ALL's `.browserpod RESOURCE socks socks蘑菇.pk threaten` ⇒ engagement confirmed |
| 3 | `SKIP_W1=W3=1` (S8 BFP4 test, with reroute fix) | 0/10 | immediate NaN crash, no token printed | reroute still broken on BFP4; S8 test inconclusive |
| 6 | canonical re-verify (no envs) | **9/10** | clean math reasoning | ship-path UNCHANGED |

Path B's ring-grid fix is committed to `tt-metal-sglang` as
`c1e3437b78e` because:

1. It does NOT regress canonical (variants only built when SKIP_*
   env is set).
2. It CHANGES the reroute's garbage signature decisively, proving the
   bank-order permutation IS a necessary ingredient (just not
   sufficient).
3. The infrastructure is reusable for future reroute experiments.

## Path B fix mechanics (what it does)

For each SKIP_* env that gates a weight (WQKV, WO, W1, W3, W2),
construct a second tensor variant `{weight_name}_skip_ring` on the
permuted DRAM grid `prefetcher.to_core_range_set(
prefetcher.dram_banks())` — on Blackhole this is the 8-bank
permuted order `[1,3,2,0,5,7,6,4]`. This mirrors the
`lm_head.py:117-133` ring-mm weight pattern, which uses the same grid
for its `prefetch=False, num_global_cb_receivers=1` matmul.

The matmul call sites in `attention.py:909-919` and
`mlp.py:264-289, 414-427` switch the weight argument from
`self.{wqkv,wo,w1,w3,w2}` to `self.{...}_skip_ring` when the
corresponding `_skip_*` flag is set.

Variants are None by default; the `if not skip` paths short-circuit
construction. Cache filenames carry the `_skip_ring` suffix to avoid
collision with canonical-grid cache.

## Why the fix is partial (analysis from this dispatch)

Per the kernel source
(`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2466-2604`),
the `prefetch=False, num_global_cb_receivers=1` matmul's in1 reader
expects:

1. `in1_is_dram_sharded == True` (Path B: ✓ — both canonical and
   ring-grid `create_dram_sharded_mem_config` give DRAM-sharded).
2. Per-worker `bank_id` lookup via `worker_y_to_dram_bank_*[core.y]`
   from `device->get_optimal_dram_bank_to_logical_worker_assignment(
   in1_noc)`. This means **the kernel itself reads bank IDs from
   the device API**, NOT from the tensor's `dram_grid` parameter.

So `dram_grid` controls **where the shards are PLACED at write time**.
The kernel reads shard at `bank_id = lookup(core.y)`. For
correctness, shard `i` (in some kernel-defined ordering) must land on
bank `bank_id = lookup(worker_cores[i].y)`.

Path B placed shard `i` on bank `i` (per the
`prefetcher.dram_banks()` iteration order). This matches lm_head's
pattern. But lm_head ALSO uses different mesh_mapper (`ShardTensor
ToMesh(dim=-1)`, 1-D), n-dim padding to power-of-2, and `untilize_out=
True` for output. Some of those are still mismatched in Path B.

The decisive piece of evidence: **garbage signatures CHANGED** between
B.7 (canonical grid) and B.8 (permuted grid). Both produce garbage,
but different garbage. This rules out "fix didn't engage" and rules in
"fix engaged but kernel sees different but still-incorrect data."

Likely remaining issues (untested):
- mesh_mapper dim ordering: Path B uses `ShardTensor2dMesh(dims=w*_dims,
  mesh_shape=cluster_shape)` while lm_head uses `ShardTensorToMesh(dim=-1)`.
  These produce different per-device tensor layouts.
- n-dim padding: lm_head pads to power of 2 (e.g., 6144 → 8192);
  Path B doesn't. For W2 n=4096 (already pow2) this shouldn't matter,
  but for W1/W3 n=6144 it could.
- Output `memory_config` / `sub_device_id` mismatch (was already
  flagged in Phase B.7's `What this session did NOT do` list as a
  candidate).

## Step 3 — S8 BFP4 hypothesis test (couldn't isolate)

Config: `SKIP_W1=W3=1` + full prefetcher (WQKV+WO+W2 BFP8 still in
prefetcher). With Path B's reroute fix in place, W1/W3 use the
ring-grid path. Result: server boots, gets to first decode step,
crashes immediately with `RuntimeError: probability tensor contains
either inf, nan or element < 0`. No tokens printed.

Comparison to Steps 1+2 (which gave garbage tokens BEFORE crashing):
SKIP_W1+W3 is WORSE. Possible interpretations:

1. The Path B fix is even MORE broken for BFP4 dtype (W1/W3 are BFP4,
   while W2 was BFP8). BFP4 tile size = 576 B vs BFP8 1088 B; the
   `_ring_mem_config(k=4096, n=6144)`'s shard-spec stride math
   (which assumes `padded_size // dram_cores` regardless of dtype)
   might silently miscompute for BFP4.
2. The prefetcher's per-layer cycle with only WQKV+WO+W2 (3 tensors)
   exposes a different bug than the 5-tensor cycle.
3. The reroute compound failure (W1 broken AND W3 broken in same
   forward) accumulates faster than single-weight SKIP.

**S8 test inconclusive.** The S8 BFP4 hypothesis CANNOT be tested
under any SKIP-config until the reroute path itself is correct.

## S8 BFP4 hypothesis — re-evaluation via code analysis (RULED OUT as stated)

Path A's S8 hypothesis (per
`tt_qwen3_8b_prefetcher_path_a_b4ext_2026-05-23.md` §S8):

> The producer's c_0 reader (reader_dram.cpp:95) advances
> `l1_write_addr += max_block_size` between blocks where
> `max_block_size = 52224 B`. For BFP4 blocks, actual data is
> `27648 B`, leaving a 24576-byte GAP. If the writer's reads use the
> OLD layout assumption (reading 13056 B per receiver from c_0, but
> the BFP4 block only occupies 6912 B per receiver), bytes 6913..13056
> are STALE.

Re-checked the code this dispatch (`reader_dram.cpp`, `writer_l1.cpp`,
`dram_prefetcher_program_factory.cpp`):

- Producer reader: advances `l1_write_addr += max_block_size` AND
  internally `temp_l1_write_addr += curr_page_size` per page. Per-block
  total bytes written = `curr_block_num_pages × curr_page_size` =
  per-tensor block size. For BFP4 W1: 4 × 6912 = 27648 B. The 24576-B
  gap exists but is uninitialized within the same 52224-B cb slot.
- Writer reads from `local_cb_addr = get_read_ptr(local_cb_id)`. The
  c_0 cb has `page_size = max_tile_size` and `cb_pop_front(cb_id,
  max_block_num_tiles)` advances by `max_block_size` per block. So
  writer's `local_cb_addr` also advances by `max_block_size` per block
  — synced with producer.
- Writer uses `coalesced_num_pages × coalesced_page_size × num_rows`
  per receiver = `1728 × 4 = 6912 B` per receiver = `27648 B` total
  per block. **Writer reads ONLY the valid bytes; the 24576-B gap is
  skipped, never propagated to the receiver.**

**S8 as Path A described it is INCORRECT.** The producer's advance by
`max_block_size` leaves an uninitialized gap, but neither producer
nor writer reads from that gap. Path A's B.4-extended byte-dump at
L0, L1, L17, L35 already verified this empirically: bytes at the
receiver match bytes at the producer.

The TRUE root cause of the full-prefetcher 0/10 failure is therefore
NOT in `reader_dram.cpp:95`. The Path A doc's "Fix surface" at the
end is invalid; do NOT pursue that fix.

## What remains open (suspect ranking after this dispatch)

| ID | Suspect | Status after B.8 |
|---|---|---|
| S1 | Row-wise stride mismatch in writer | RULED OUT (B.3 byte-dump) |
| S2 | num_blocks mismatch | OPEN, unlikely (would be tensor-invariant) |
| S3 | WO K-shard transposition | RULED OUT (Phase A) |
| S5 | BFP4/BFP8 mixed tile pitch (program-config sense) | RULED OUT (Phase A) |
| S6 | Per-layer cross-block address drift in producer | RULED OUT (Path A B.4-extended) |
| S7 | Compute-kernel `update_rd_ptr_to_ring_index` wrap | RULED OUT (Phase B.6 + Path A B.5) |
| **S8** | **Per-tensor block_size in reader_dram.cpp** | **RULED OUT (this dispatch, code analysis: producer/writer both ignore the gap)** |
| S9 | Producer NOC posted-writes flush insufficient | OPEN |
| S10 | Sub-device CB-address misalignment | OPEN |
| NEW: SKIP_* reroute config | one of: mesh_mapper, n-padding, output_mem_config | **PARTIALLY DIAGNOSED — bank-grid is necessary but not sufficient** |

S9 and S10 are the only remaining "prefetcher-side" suspects. The
SKIP_* reroute bug is now distinct (also unresolved) and not the
prefetcher.

## What this dispatch did NOT do

- Did NOT land a working reroute fix. Path B's bank-grid is necessary
  but insufficient.
- Did NOT modify any C++ kernel code (after determining S8 is
  RULED OUT, the planned reader_dram.cpp edit was invalidated).
- Did NOT validate Step 4 (SKIP_W2 alone triangulation per plan) —
  already covered by Step 1.
- Did NOT bench TPOT for SKIP_W1+SKIP_W3 — no correct outputs to
  bench against.

## Path forward — recommended next session

Two orthogonal, parallel attacks:

### A. Reroute completion (in `attention.py` / `mlp.py`)

Iterate on the SKIP_* reroute fix toward a clean test:
1. Match lm_head's mesh_mapper exactly: switch from `ShardTensor2dMesh`
   to per-weight `ShardTensorToMesh(dim=-1)`. Validate cluster_shape
   is consistent.
2. Add `pad_to_power_of_2` for `n` dimension in `_ring_mem_config`,
   to match lm_head.
3. Try `untilize_out=True` for all SKIP paths (not just WQKV).
4. Audit `sub_device_id` — try `None` instead of `receiver_sub_device_id`
   when `global_cb=None`.
5. Audit output `memory_config` for the SKIP path — is it compatible
   with what the downstream all-reduce / reshape expects?

Gate per iteration: SKIP_W2=1 GSM8K(10) ≥ 7/10. Once clean, the SKIP
matrix becomes a valid diagnostic tool again, and S8 BFP4 / S9 NOC
flush / S10 sub-device can be tested via the SKIP isolation.

### B. NOC flush / sub-device probe (in tt-metal C++)

Even with the reroute broken, the full-prefetcher path is still
failing. Test S9 directly: replace `noc_async_posted_writes_flushed()`
with `noc_async_writes_flushed()` in `writer_l1.cpp:65`. Rebuild,
GSM8K. If passes ≥ 7/10, S9 confirmed. Risk: perf may regress (posted
→ unposted writes adds latency); the test is correctness-only.

For S10: add DPRINT in writer_l1.cpp dumping `remote_cb.fifo_start_addr`
AND in matmul's reader dumping the same. If they differ, S10 confirmed.

## Working state at end of session

- tt-metal-sglang HEAD: **`c1e3437b78e`** (Phase B.8 reroute ring-grid
  variants; canonical-preserving).
- sglang HEAD: pending this doc + commit message.
- `stash@{0..2}` untouched.
- C++ kernel tree pristine.
- Canonical baseline: GSM8K(10) chat = **9/10** (re-verified post-commit).
- All cards healthy.
- Cache cleared. No server running.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B. Two
independent broken paths remain:

1. The DRAM-prefetcher BFP8 GlobalCB route (full prefetcher, all
   prior sessions). S8 RULED OUT as Path A described; S9/S10 remain.
2. The SKIP_* matmul reroute fallback (Phase B.7 onward). Path B
   partially fixes; further iteration needed per "Path forward A."

## Commits this session

- (tt-metal-sglang) `c1e3437b78e prefetcher: SKIP_* reroute ring-grid
  variants (Phase B.8, partial fix)` — Path B's edits as committed.
- (sglang) `<this doc>` — pending commit.
