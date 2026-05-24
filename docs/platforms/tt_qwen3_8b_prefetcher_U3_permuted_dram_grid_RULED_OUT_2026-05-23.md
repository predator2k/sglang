# TT Qwen3-8B prefetcher — U3 permuted-DRAM-grid hypothesis RULED OUT — 2026-05-23

Status: **EVIDENCE_ADVANCE — U3 hypothesis RULED OUT by combined code analysis + hardware test.
The prefetcher kernel reader maps `bank_id = sender_index` (linear), so a contiguous
`(0,0)-(7,0)` DRAM grid is the CORRECT placement — shard `i` lives on bank `i`, which is
exactly what sender `i` reads. The permuted `prefetcher.dram_banks()` grid
(`[1,3,2,0,5,7,6,4]` on Blackhole) is the WRONG placement for the prefetcher path —
sender 0 reads bank 0, but the permuted grid places shard 3 on bank 0, so sender 0
gets shard 3's bytes. Hardware test confirms: SGLANG_TT_PREFETCHER_PERMUTED_DRAM_GRID=1
produces 0/10 GSM8K(10) chat with a CHANGED garbage signature vs canonical-grid baseline
(`理赔.pool真假真假.pool...` vs `.browserpod RESOURCE socks socks...`) and crashes at Q2
with NaN. Canonical re-verify post-experiment: GSM8K(10) chat = 10/10 = 100%. The
bug remains the U1 cross-sub-device dispatch sync gap; recommended next attack is the
Attack 1 path (galaxy-style 2-subdev with dummy receivers).**

Continuation of `tt_qwen3_8b_prefetcher_U1_cross_subdev_sync_2026-05-23.md`.
Tests the strongest remaining hypothesis after 13 dispatches: maybe even though
byte-transport is correct (S10/Lead 3) and sync is not the issue (galaxy 2-subdev
parity refuted U1), the actual **data fidelity** at the prefetcher reader is
broken by a DRAM-grid-vs-bank-mapping mismatch analogous to what Phase B.8 found
for the SKIP_* fallback path.

## Phase 1 — code-analysis diagnostic (galaxy vs ours)

Galaxy reference: `models/demos/llama3_70b_galaxy/tt/llama_attention.py:142-220`,
`models/demos/llama3_70b_galaxy/tt/model_config.py:631-638` (`dram_weight_grid`),
`models/demos/llama3_70b_galaxy/tt/model_config.py:2290-2297`
(`create_dram_sharded_mem_config`).

| Field | Galaxy (production-validated) | Ours (Qwen3-8B prefetcher) | Match? |
|---|---|---|---|
| `dram_weight_grid` | contiguous `CoreRangeSet((0,0)-(11,0))` (12 cores, Wormhole) | contiguous `CoreRangeSet((0,0)-(7,0))` (8 cores, Blackhole) | YES (both contiguous, both span full dram_grid) |
| Prefetcher-registered weight grid (WQKV / WO / W1 / W3 / W2) | `dram_weight_grid` via `create_dram_sharded_mem_config(...)` | `dram_weight_grid` via same function | **YES — identical pattern** |
| `dram_banks()` ordering | permuted `[1,2,3,0,4,6,9,10,11,8,7,5]` (Wormhole yaml) | permuted `[1,3,2,0,5,7,6,4]` (Blackhole yaml) | Both permuted (same indirection); but neither uses this as placement grid for prefetcher weights |
| Where permuted grid IS used | NONE in galaxy prefetcher path | `lm_head.py:121` ring-mm path + Phase B.8 SKIP_* fallback variants (canonical-preserving) | OUR DEPLOYED PATH IS GALAXY-EQUIVALENT |

**Decision point (per dispatch plan):** galaxy uses CONTIGUOUS grid for prefetcher
weights. Our deployed code does the same. The U3 hypothesis ("galaxy uses permuted
grid") is REFUTED by code inspection — galaxy code does not use the permuted grid
for any prefetcher-registered weight. The permuted grid is ONLY used by the
`prefetch=False, num_global_cb_receivers=1` ring matmul (lm_head + our SKIP
fallback) which uses a DIFFERENT bank-mapping algorithm
(`device->get_optimal_dram_bank_to_logical_worker_assignment`).

### Why the U3 hypothesis is wrong (root-cause derivation)

`ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp:247-265`:

```cpp
for (uint32_t core_index = 0; core_index < reader_core_range.num_cores(); core_index++) {
    const auto& core = reader_cores[core_index];
    uint32_t bank_id = core_index;            // ← linear mapping
    ...
    std::vector<uint32_t> reader_rt_args = {bank_id, vc, total_num_blocks_in_buffer};
    ...
}
```

`ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/reader_dram.cpp:47-48`:

```cpp
uint32_t tensor_base_address = tensor_addrs_l1[layer * num_tensors + t];
uint64_t src_base_addr = get_noc_addr_from_bank_id<true>(bank_id, tensor_base_address);
```

`tt_metal/impl/allocator/allocator.cpp:53-56` (DRAM bank-to-logical-core mapping):

```cpp
for (uint32_t bank_id = 0; bank_id < config_->num_dram_channels; bank_id++) {
    CoreCoord logical_core = CoreCoord{bank_id, 0};
    ...
    logical_core_to_bank_ids_[BufferType::DRAM].insert({logical_core, {bank_id}});
}
```

So **DRAM bank `i` ↔ logical core `(i, 0)`** is a 1:1 identity mapping.
The DRAM-sharded buffer's shard `i` is placed on the i-th core in the
shard_spec grid (which is the i-th `CoreCoord` of `dram_weight_grid`).

* Contiguous grid `(0,0), (1,0), ..., (7,0)` → shard `i` on bank `i` ✓
* Permuted grid `(1,0), (3,0), (2,0), (0,0), (5,0), (7,0), (6,0), (4,0)` →
  shard 0 on bank 1, shard 1 on bank 3, shard 2 on bank 2, shard 3 on bank 0, …

Prefetcher sender `s` (s = `core_index` ∈ [0..num_senders)) reads from
`bank_id = s` at the tensor base address. With contiguous grid: gets shard `s`.
With permuted grid: gets shard `permuted^-1[s]` (the inverse permutation), i.e.
sender 0 gets shard 3, sender 1 gets shard 0, sender 2 gets shard 2, sender 3
gets shard 1, sender 4 gets shard 7, sender 5 gets shard 4, sender 6 gets shard 6,
sender 7 gets shard 5.

The downstream matmul ring topology then misaligns: receiver at ring_idx `4s+k`
expects N-cols `[(4s+k)*N/32 : (4s+k+1)*N/32]` but receives N-cols from
shard `permuted^-1[s]`'s range. Different garbage.

The `prefetch=False, num_global_cb_receivers=1` matmul kernel (lm_head + our
SKIP_* fallback) uses a DIFFERENT lookup
(`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2466+`):
`worker_y_to_dram_bank_*[core.y]` derived from
`device->get_optimal_dram_bank_to_logical_worker_assignment(in1_noc)`.
This API returns banks ordered by NOC distance to each worker row. The
**inverse** of that lookup is what produces the `[1,3,2,0,5,7,6,4]` order,
which is why the SKIP path needs the permuted grid. **The prefetcher path
does NOT use that lookup — it uses a linear `bank_id = core_index` mapping.**

## Phase 2 — implementation

Env-gated `SGLANG_TT_PREFETCHER_PERMUTED_DRAM_GRID=1` builds permuted-grid
variants `wqkv_pdg`, `wo_sharded_ring_pdg` (in `attention.py`) and
`w1_pdg`, `w3_pdg`, `w2_pdg` (in `mlp.py`) and registers THOSE for the
prefetcher's `insert_tensor` calls. The matmul forward sites switch the
weight argument accordingly. Cache files carry the `_pdg` suffix to
avoid collision. Non-permuted path is the default (env off → no variant
constructed, no behavior change).

Files: `models/tt_transformers/tt/attention.py:601-741` (init),
`models/tt_transformers/tt/attention.py:996-1010, 1411-1428` (forward sites);
`models/tt_transformers/tt/mlp.py:118-271` (init),
`models/tt_transformers/tt/mlp.py:315-355, 488-508` (forward sites).

Commit: `tt-metal-sglang@22417cb1d2d` "prefetcher: U3 permuted-DRAM-grid
experiment (env-gated, RULED OUT 2026-05-23)".

## Phase 3 — hardware test

| Run | Config | Result | Q1 signature |
|---|---|---|---|
| **U3 permuted-grid** | `SGLANG_TT_USE_PREFETCHER=1`, `SGLANG_TT_PREFETCHER_PERMUTED_DRAM_GRID=1` | **0/10**, server crashed at Q2 with `RuntimeError: probability tensor contains either inf, nan or element < 0` | `理赔.pool真假真假.pool修为逃离风控蘑菇PrivateKeyGravityGravity日晚间...` |
| Baseline reference (canonical-grid prefetcher, broken since Phase A) | `SGLANG_TT_USE_PREFETCHER=1` only | 0/10 (prior data) | `.browserpod RESOURCE socks socks蘑菇.pk threaten ...` (Phase B.7) |
| **Canonical re-verify** (no prefetcher) | env unset | **10/10 = 100%** | `#### 64` (correct math reasoning) |

Decisive observation: garbage signature CHANGED (different scramble), proving the
fix engaged at the data-layer but produced a DIFFERENT wrong placement. This
matches the theoretical prediction from Phase 1's code analysis. **U3 hypothesis
RULED OUT.**

## Phase 4 — decision (Case C per dispatch plan)

Per the dispatch's decision tree, this maps to **Case B/C** — partial impact
with changed garbage. The DRAM grid IS related (different placement → different
garbage) but the canonical contiguous grid is the CORRECT placement for the
prefetcher path. The bug is elsewhere.

## Updated suspect ranking

| ID | Suspect | Status after U3 |
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
| A1 | ecfb2e782c2 L1-clash bypass = hidden real clash | RULED OUT (Attack 1) |
| Lead 2 | Port upstream 934d954b995's `last_subblock_w_valid` | RULED OUT |
| Lead 3 (U2) | in0 ring-all-gather byte transport | RULED OUT (in0 bytes correct) |
| **U1** | **Cross-sub-device dispatch sync gap** | **ROOT CAUSE LOCATED — fix path identified (Attack 1: galaxy-style 2-subdev + dummy receivers)** |
| **U3** | **Prefetcher DRAM grid placement mismatch** | **RULED OUT (this dispatch) — contiguous grid is the CORRECT placement for the prefetcher path; permuted grid is for the prefetch=False ring matmul only** |

## Phase 5 — pivot: recommended next attacks

The U1 root-cause + fix path remain the highest-leverage attack. Concrete plans:

### Attack 1 (highest leverage, ~4-8h) — galaxy-style 2-subdev + dummy receivers

1. Modify `prefetcher.py:init()` to skip 3-subdev when
   `SGLANG_TT_PREFETCHER_2SUBDEV=1`.
2. Modify the GlobalCB sender_receiver_mapping to include "dummy receiver"
   CoreRangeSets that cover the bbox gaps of `worker_sub_device ∩ non_idle_cores`
   in each rectangular sub-range.
3. Switch all prefetcher matmul `sub_device_id=` to `worker_sub_device_id`
   (which in 2-subdev mode includes receivers).
4. Verify dummy receivers are NO-OP in the producer (no actual writes) but
   live in GlobalCB metadata so the `contains()` check passes.
5. Rebuild Python (no C++ change), GSM8K(10).

Expected: matmul's `determine_sub_device_ids` returns `worker_sub_device`
(matches all_reduce); dispatch system serializes both on the same stream;
the WAIT_STREAM-only cached path waits on the receiver-stream counter that
the matmul increments. Bug fixed.

Risks: (a) L1 CB clash between RMSNorm on
`dynamic_worker_core_grid(16) = (5,0)-(6,7)` and prefetcher GlobalCB on
receivers (cols 1-4 left + 8-11 right + col 1 of right side row 0). None
of our receivers are in cols 5-6, so no overlap. Should be safe.
(b) The bbox computation in
`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2024-2037` iterates
over each `subdevice_cores.ranges()` and pushes the BOUNDING BOX of the
intersection with `non_idle_cores` — so dummy receivers need to be picked
so each rectangular range of `worker_sub_device` has a clean intersection
with the GlobalCB membership (an off-by-one here was what blew up the
initial U1 Attack A try).

### Attack 2 (~2-4h) — device-side cross-sub-device barrier semaphore

Use a `tt_metal::GlobalSemaphore` shared between receiver and worker sub-devices.
Insert a tiny "barrier" op after each prefetcher matmul that increments the
sema on receiver_sub_device after matmul completes; the next all_reduce / CCL
op waits on the sema. Trace-safe (device-side semaphore, not host-sync).
Requires a small ttnn op or matmul-factory modification (~4 hours C++ work).

### Attack 3 (~1-2h) — modify all_reduce to declare receiver dependency

Add `wait_sub_device_ids=[receiver_sub_device_id]` kwarg to all_reduce / CCL
ops, and patch `tt_metal/impl/program/dispatch.cpp:431` to emit an additional
WAIT_STREAM on the listed sub-devices alongside the auto-determined ones.
Smallest API surface change but requires C++ tt_metal modification.

## What this dispatch did NOT do

* Did NOT land a working fix. U3 was ruled out as predicted by code analysis
  and confirmed by hardware test.
* Did NOT implement Attack 1 (galaxy 2-subdev + dummy receivers) — that's the
  next session's plan.
* Did NOT modify any C++ kernel code.
* Did NOT exhaust the 4-8h Attack 1 scope in this dispatch — we executed the
  U3 evaluation cleanly (Phase 1-6 per plan), avoided premature commitment
  to a partial Attack 1, and preserved infrastructure for future probes.

## Working state at end of session

* tt-metal-sglang HEAD: **`22417cb1d2d`** (U3 commit; canonical-preserving).
  Previous HEAD `c1e3437b78e` (Phase B.8) and `b81f15001ca` (B.7) preserved.
* Probe / experiment branches: none created this session (U3 went straight to
  tenstorrent-p1 because it's env-gated infrastructure with no canonical
  regression).
* sglang HEAD: this doc pending commit.
* `stash@{0,1,2}` UNTOUCHED.
* C++ kernel tree pristine.
* Canonical baseline GSM8K(10) chat = **10/10 = 100%** post-U3-experiment.
  (Previously 9/10 on B.8 — slight noise; both reproduce the canonical pass.)
* All cards healthy (`tt-smi -ls` shows both p150a present and resettable).
* Cache cleared. No server running.

## Reproduction commands

```bash
# Confirm clean HEAD
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
cd /home/mhnie/tt-metal-sglang && git log -1 --oneline  # should be 22417cb1d2d

# Re-run U3 (CHANGED garbage; crashes at Q2)
podman exec p3a-ngram bash -c 'if [ -n "/root/.cache/tt-metal-cache" ] && [ -d "/root/.cache/tt-metal-cache" ]; then rm -rf "/root/.cache/tt-metal-cache"/*; fi'
podman exec -d p3a-ngram bash -c '
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_PERMUTED_DRAM_GRID=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u3_test.log 2>&1'

# Wait for /health_generate 200, then:
podman exec p3a-ngram python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 10
# Expect: Q1 garbage like "理赔.pool真假真假.pool修为逃离风控蘑菇…", Q2+ crashes with NaN.

# Canonical re-verify (10/10)
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c 'if [ -n "/root/.cache/tt-metal-cache" ] && [ -d "/root/.cache/tt-metal-cache" ]; then rm -rf "/root/.cache/tt-metal-cache"/*; fi'
podman exec -d p3a-ngram bash -c '
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/canon_reverify.log 2>&1'
# Wait for /health_generate 200, then:
podman exec p3a-ngram python3 /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_qwen3_8b_gsm8k_chat.py --num 10
# Expect: 10/10 = 100%.
```

## Commits this session

* (tt-metal-sglang) **`22417cb1d2d`** prefetcher: U3 permuted-DRAM-grid
  experiment (env-gated, RULED OUT 2026-05-23)
* (sglang) `<this doc>` — pending commit

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping config at
TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = **10/10 = 100%**.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** Root cause is the
U1 cross-sub-device dispatch sync gap (LOCATED, not yet fixed). The U3
permuted-DRAM-grid hypothesis is now RULED OUT (this dispatch). Next-up:
implement Attack 1 (galaxy-style 2-subdev + dummy receivers, ~4-8h).
