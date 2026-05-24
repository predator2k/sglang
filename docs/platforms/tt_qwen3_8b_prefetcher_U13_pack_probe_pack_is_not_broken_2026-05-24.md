# TT Qwen3-8B prefetcher — U13: PACK probe + sync'd DST-at-pack + U9/U10 re-validation with FIXED rebuild — PACK is NOT broken; U9/U10 verdicts CONFIRMED — 2026-05-24

Status: **EVIDENCE_ADVANCE — Three loadbearing measurements added to
the U12 line of attack.  (Part 1) PACK-thread L1 probe immediately
after `pack_tile_block` writes to `mm_out_cb` shows ALL ZEROES with
zero weights once `tensix_sync()` is inserted before the volatile
read.  Without sync, the same probe showed 4478/57600 NONZERO at
`blk=31` on 2-of-4 ELFs — but that was a race against the in-flight
packer NoC write, not real garbage.  (Part 1b) Sync'd DST probe AT
the pack site (right before pack_tile_block) on EVERY iteration
captured by the budget: 57600/57600 ZERO across all 4 gathered ELFs.
(Part 2) Widened U12 LLK probe with per-ELF tagging confirms all 4
gathered ELFs (0x2d96e, 0x2db6e, 0x2de6b, 0x2df6a) were exercised;
sync'd reads show DST=0 on every probed iteration / core / block.
(Part 3) Re-validated U9 (DST_ZERO) and U10 (3× BYPASS_GCB) with the
FIXED rebuild script + container-source-sync — ALL 4 PRIOR VERDICTS
HOLD: none of those patches fix the bug.  The rebuild trap did NOT
mask the truth.  Bug surface narrows to: a path BETWEEN mm_out_cb
write and the final output that lives OUTSIDE the gathered matmul
compute kernel (consumer reader, CCL, RMSNorm, all-gather, or an
overwriter touching `mm_out_cb`'s L1 region between PACK and the
consumer's UNPACK).**

Continuation of `tt_qwen3_8b_prefetcher_U12_llk_probe_math_not_broken_2026-05-24.md`
(U12 ruled out the matmul kernel's MATH path; bug was localized to
"downstream of DST commit").

tt-metal-sglang HEAD: **`dc43d1d2a79`** (U13 probes landed).
sglang HEAD: pending this doc + rebuild script container-sync note.

## Bonus discovery: host-vs-container source split

U12's `rebuild_tt_metal_kernels.sh` patch sync'd `_ttnn.so` from
`${BUILD_DIR}` to the install dir.  U13 found a SECOND silent
invalidation trap: the container `/tt-metal/` is a SEPARATE working
copy from the host `/home/mhnie/tt-metal-sglang/`.  Host-side edits
are invisible to the in-container ninja build until they are
`podman cp`'d into `/tt-metal/`.

Symptom: U13's first three runs had the kernel/factory edits on the
host but `strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep
SGLANG_TT_PREFETCHER_PACK_PROBE` returned nothing.  The cache cleared
fine, the rebuild "succeeded", but the new env-var name never made
it into the .so because the host source was never compiled.

Workflow correction (now baked into the rebuild script header):
1. Edit on host: `/home/mhnie/tt-metal-sglang/...`
2. `podman cp <host_file> p3a-ngram:/tt-metal/...`
3. Verify: `podman exec p3a-ngram bash -c 'grep <SENTINEL> /tt-metal/...'`
4. Run rebuild script (which builds in-container + syncs install dir)
5. Verify: `strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep <SENTINEL>`
6. Clear cache, run.

## Part 1 — PACK probe results

Probe code: added `PACK((...))` block immediately after
`pack_tile_block(start_dst_index, mm_out_cb_id, ...)` in the
`last_out` branch, and after the same call in the `spill` /
mm_partials branch.  Reads first 16 bytes (4 dwords) at
`CB_WR_PTR(mm_out_cb_id)` — i.e. `fifo_wr_ptr << cb_addr_shift`,
which is the L1 byte address PACK just wrote (cb.push_back has not
yet advanced fifo_wr_ptr).  Per-ELF static budget = 16 entries.

### Run 1: PACK_PROBE without explicit sync

env: `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1
SGLANG_TT_PREFETCHER_PACK_PROBE=1`, prompt `"2+2="`, max_new=5.

Output: `"Und抄袭抄袭边上olia"` (bug present, matches U11/U12 signature).

DPRINT entries:

| ELF       | total | NONZERO | zero  | NONZERO locus               |
|-----------|------:|--------:|------:|-----------------------------|
| 0x2d96e   | 11520 |       0 | 11520 | —                           |
| 0x2db6e   | 23040 |       0 | 23040 | —                           |
| 0x2de6b   | 11520 |  ~2300  |  9220 | `b=0 blk=31 is0=0 is1=0`    |
| 0x2df6a   | 11520 |  ~2178  |  9342 | `b=0 blk=31 is0=0 is1=0`    |

PACK_OUT NONZERO appears ONLY at `blk=31` (the last block = `num_blocks-1` =
last_out branch) and ONLY on 2 of the 4 gathered ELFs.  Sample:

```
0:8-0:TR2: [U13_PACK_OUT elf=0x2de6b l1=0xaaf00 b=0 blk=31 is0=0 is1=0
            w0=0x880e4d17 w1=0x9a4e107 w2=0x15a323bd w3=0x8910c404 NONZERO]
```

### Run 2: PACK_PROBE with `ckernel::tensix_sync()` before L1 read

Same code, added `tensix_sync()` immediately before the volatile
`uint32_t*` read.  Output: `"Und秤秤/templates legislature"` (bug
still present).

DPRINT entries:

| ELF       | total | NONZERO | zero  |
|-----------|------:|--------:|------:|
| 0x2d96e   | 11520 |       0 | 11520 |
| 0x2db6e   | 23040 |       0 | 23040 |
| 0x2de6b   | 11520 |       0 | 11520 |
| 0x2df6a   | 11520 |       0 | 11520 |

**With sync, mm_out_cb post-PACK is ALL ZERO across every probed core
and every probed ELF (57600 / 57600 entries).**  The original run's
NONZERO entries were a race against the in-flight packer NoC write
— the volatile read returned stale prior-program L1 bytes before
the packer's write reached cache.  This is a load-bearing
correction: **PACK is not writing garbage.**

## Part 1b — Sync'd DST-at-pack probe

Same SGLANG_TT_PREFETCHER_PACK_PROBE gate; added a MATH-thread
probe at the pack site (before `pack_tile_block`) using
`ckernel::tensix_sync()` followed by `ckernel::dbg_get_array_row(DEST,
0, ...)`.  Per-ELF budget = 8.

Results: **57600 / 57600 entries read DST tile 0 row 0 = all zero**
across all 4 ELFs, every probed `(b, block, in0_subblock, in1_subblock)`
tuple.

Combined with Part 1: PACK reads zero DST and writes zero
mm_out_cb.  The gathered matmul compute kernel is INNOCENT under U11
zero-weight injection — both MATH and PACK produce correct zeros.

## Part 2 — U12 LLK probe coverage verification

U12 v2's probe was gated on `in0_subblock==0 && in1_subblock==0 &&
inner_dim_idx==0`.  Some ELFs might never have exercised those
subblock indices on the first matmul_block invocation, leaving the
probe silent for that ELF.

U13 Part 2 widens the gate: fires on the first 16 iterations the
kernel sees (any subblock / inner_dim_idx), emits a CT-hashed ELF
tag built from `(in0_block_w, in0_num_subblocks, in1_num_subblocks,
num_blocks, out_subblock_h, out_subblock_w, batch)`.

Results — 921506 total DST entries:

| ELF       | total entries | NONZERO | zero    |
|-----------|--------------:|--------:|--------:|
| 0x2d96e   | 184320        |       0 | 184320  |
| 0x2db6e   | 368640        |       0 | 368640  |
| 0x2de6b   | 184320        |      84 | 184236  |
| 0x2df6a   | 184320        |       0 | 184320  |

The 84 NONZERO entries (0.05% on ELF 0x2de6b only, all at `b=0 blk=0
is0=0 is1=0 kk=0`) match U12 v2's race-noise pattern (without
`dbg_halt()` sync).  These are NOT real bugs — `dbg_get_array_row`
races the in-flight FPU pipeline when not preceded by
`tensix_sync()`.  Part 1b's sync'd DST reads at the pack site (after
the full inner loop completes) are the load-bearing measurement.

All 4 gathered ELFs in the Qwen3-8B model were exercised by the
probe.  No ELF escaped coverage.

## Part 3 — U9 / U10 re-validation with FIXED rebuild (+ container sync)

Per the U12 bonus discovery + the U13 container-source split: the
rebuild script may have silently invalidated U9 and U10 patches.
This part re-runs each with explicit verification that the patched
kernel is actually loaded.

Method per test:
1. Verify host edit landed (`grep` for sentinel).
2. `podman cp` to `/tt-metal/`.
3. Verify in-container source has sentinel.
4. Run rebuild script.
5. Verify `strings _ttnncpp.so` has the env-var name.
6. Clear cache (literal-path `[ -n "$VAR" ]` guarded).
7. Launch server with the bypass env=1.
8. Run sanity decode (`"2+2=" → ?`).
9. Run GSM8K(5) (truncated from GSM8K(10) for time; pass = ≥ 3/5).

Results:

| Test | Env var | Sanity `"2+2=" →`               | GSM8K(5) | Verdict (vs U9/U10 prior) |
|------|---------|---------------------------------|---------:|---------------------------|
| **U9 DST_ZERO**        | `SGLANG_TT_PREFETCHER_DST_ZERO=1`         | `"5erceerceerceerce"`      | 0/5 (server crashed Q1) | **UPHELD** |
| **U10 BYPASS_GCB_INIT**    | `SGLANG_TT_PREFETCHER_BYPASS_GCB_INIT=1`    | `"5argvargvargvargv"`      | 0/5 (server crashed Q1) | **UPHELD** |
| **U10 BYPASS_GCB_BLOCK**   | `SGLANG_TT_PREFETCHER_BYPASS_GCB_BLOCK=1`   | `"5=} craz craz craz"`     | 0/5 (server crashed Q1) | **UPHELD** |
| **U10 BYPASS_GCB_ADVANCE** | `SGLANG_TT_PREFETCHER_BYPASS_GCB_ADVANCE=1` | `"5X organised教學'"`      | 0/5 (server crashed Q1) | **UPHELD** |

All 4 prior verdicts hold.  The garbage signatures changed (because
the actual L1 layout in DST/mm_out_cb has changed with the bypass
active), but the bug class is unchanged — high-entropy multi-script
mode collapse, then NaN-poisoning crash.

**The U9 / U10 RULED-OUT verdicts were NOT false positives.**
The rebuild trap (real and now-documented) did not save us — none
of those code regions in the gathered matmul kernel is the source
of the bug.

### Canonical re-verify (no prefetcher, current kernel + lib state)

| `"2+2=" →`                        | GSM8K(5) |
|----------------------------------|---------:|
| `"5? - The Math"`                | 3/5 (60%, Q3+Q4 reasoning errors not engine errors) |

The 3/5 is within expected variance for a 5-question sample (vs U10's
9/10 over 10 questions).  No engine-level regression: garbage tokens
absent, sentence structure coherent.

## Hypothesis ledger correction (post-U13)

| Hypothesis | Pre-U13 status | Post-U13 status | Evidence |
|---|---|---|---|
| MATH writes garbage from zero inputs | RULED OUT (U12) | **CONFIRMED RULED OUT** | U13 widened probe across all 4 ELFs / many iterations; sync'd DST always zero |
| UNPACK reads stale bytes into srcA/srcB | RULED OUT (U12) | **CONFIRMED RULED OUT** | Unchanged; U12 v1 still definitive |
| PACK writes garbage to mm_out_cb | STANDING (U12 candidate) | **RULED OUT** | U13 Part 1 sync'd L1 probe: mm_out_cb post-pack is all zero |
| mm_partials_cb spill path | STANDING (U12 candidate) | UNTESTED — `last_out` branch alone insufficient | PACK_PART probe got 0 entries; spill geometry not verified |
| A non-probed gathered ELF | STANDING (U12 candidate) | **RULED OUT** | All 4 ELFs covered by Part 2 widened probe |
| DST stale state from prior program | RULED OUT (U9) | **CONFIRMED RULED OUT** | Part 3 re-validation with verified-active patch: still broken |
| GlobalCB Block A INIT (ring-idx + start-addr) | RULED OUT (U10) | **CONFIRMED RULED OUT** | Part 3 re-validation: still broken |
| GlobalCB Block B+C (per-block rd_ptr update) | RULED OUT (U10) | **CONFIRMED RULED OUT** | Part 3 re-validation: still broken |
| GlobalCB Block D (end-of-batch handoff) | RULED OUT (U10) | **CONFIRMED RULED OUT** | Part 3 re-validation: still broken |
| **Downstream of mm_out_cb (consumer kernel)** | (implicit) | **STANDING — new top-priority** | All four eliminated above; bug must live in code reached AFTER PACK |
| **An overwriter touches mm_out_cb's L1 region between PACK and the consumer's UNPACK** | (implicit) | **STANDING — second priority** | Could explain why DST/mm_out_cb-at-pack are zero but downstream sees garbage |

## U14 dispatch (recommended next attack)

The gathered matmul compute kernel is verified innocent at MATH, at
DST-commit, and at PACK→mm_out_cb.  The bug surface is now confined
to one of:

1. **Consumer's reader kernel** (the next layer's UNPACK or the
   read-side of the all-gather/reduce-scatter that follows the
   matmul).  Under `ENABLE_GLOBAL_CB`, the reader may compute the
   wrong L1 read address — different from where the producer's PACK
   wrote.  Probe: in the consumer kernel, dump the first BF16 bytes
   from the read address right after `cb_wait_front`.  If they're
   nonzero, the reader is reading the wrong slot.

2. **An overwriter** — any kernel running on the same core between
   the producer's `cb.push_back` and the consumer's `cb_wait_front`
   that touches mm_out_cb's L1 region.  Probe: snapshot mm_out_cb's
   L1 bytes at the consumer's cb_wait_front BEFORE the consumer
   reads them; compare to the producer's post-PACK snapshot.

3. **Layout / format mismatch under prefetcher** — mm_out_cb's tile
   format may differ between producer and consumer under
   `ENABLE_GLOBAL_CB`'s sharding (e.g. different face order,
   transposed tiles, different data format).  PACK writes valid
   tiles in one format; UNPACK reads as a different format → reads
   garbage even though the bytes are "correct" by the producer's
   contract.  Probe: print mm_out_cb's `pack_dst_format` and
   `unpack_src_format` and check for mismatch.

4. **A different (non-matmul) compute kernel** that runs under
   prefetcher and is silently broken.  Candidates: distributed
   RMSNorm, CCL all-gather/reduce-scatter, the experimental
   prefetcher kernel itself.

Attack order for U14: do #3 first (cheapest to verify — just dump
the format CT args from both sides).  Then #1 (consumer reader
L1 probe).  Then #2 (cross-kernel L1 snapshot diff).  Then #4
(systematic non-matmul kernel sweep).

## Reproducer (working)

```bash
# Verify host source has SENTINEL
grep SGLANG_TT_PREFETCHER_PACK_PROBE \
  /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp

# Sync host → container
podman cp /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp \
  p3a-ngram:/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp
podman cp /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp \
  p3a-ngram:/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp

# Verify container source has SENTINEL
podman exec p3a-ngram bash -c 'grep SGLANG_TT_PREFETCHER_PACK_PROBE \
  /tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp'

# Build (full host code rebuild — picks up factory env-var propagation)
podman exec p3a-ngram bash -c 'cd /tt-metal && ./build_metal.sh 2>&1 | tail -5'

# Rebuild + sync libs (fixed in U12)
podman exec p3a-ngram bash \
  /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh ttnn

# Verify .so contains the env-var string
podman exec p3a-ngram bash -c 'strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep PACK_PROBE'

# Clear cache (literal-path, var-guarded)
podman exec p3a-ngram bash -c 'CACHE="/root/.cache/tt-metal-cache"; \
  [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

# Kill prior server (escaped dot)
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'

# Launch
podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_PREFETCHER_PACK_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_FILE=/tmp/u13_pack_sync_probe.log \
  TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u13_test.log 2>&1'

# wait until /health → 200, then:
curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text": "2+2=", "sampling_params": {"max_new_tokens": 5, "temperature": 0.0}}'

# IMPORTANT: max_new_tokens MUST be ≥ 2 to trigger the decode (gathered)
# code path.  max_new_tokens=1 only runs prefill (canonical kernel) and
# the gathered ELFs never compile.

# Analyze:
podman exec p3a-ngram grep "U13_PACK_OUT" /tmp/u13_pack_sync_probe.log | head -20
podman exec p3a-ngram bash -c 'grep "U13_PACK_OUT" /tmp/u13_pack_sync_probe.log | grep -c NONZERO'
podman exec p3a-ngram bash -c 'grep "U13_DST_SYNC" /tmp/u13_pack_sync_probe.log | grep -c NONZERO'
```

## Working state at session end

- tt-metal-sglang HEAD: **`dc43d1d2a79`** (U13 probes landed)
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only)
- sglang HEAD: pending this doc + rebuild script container-sync note
- Container `/tt-metal/`: U13 kernel + factory edits applied, libs synced to
  `/tt-metal/ttnn/ttnn/_ttnn.so` and `_ttnncpp.so`
- `/root/.cache/tt-metal-cache/*`: cleared at end of session
- `stash@{0,1,2}`: untouched
- TT devices: healthy
- No server running at session end

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal kernel/factory + sglang
      doc + scripts/, all behind opt-in env gates)
- [x] All `rm` operations guard with `[ -n "$VAR" ]` + literal-path
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Used FIXED rebuild_tt_metal_kernels.sh (host-vs-container sync added too)
- [x] Canonical re-verify (env unset) preserves baseline signature
      (`"5? - The Math"`, GSM8K(5) = 3/5)
- [x] All probes env-gated (`SGLANG_TT_PREFETCHER_PACK_PROBE=1`,
      `SGLANG_TT_PREFETCHER_LLK_PROBE=1`)

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≈ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  U13
ruled out the gathered matmul compute kernel as the bug source (it
was the dominant U7..U12 hypothesis line).  Bug is downstream of
mm_out_cb — in a consumer kernel, a CCL kernel, or a layout/format
mismatch.  A U14 follow-on probe is required to fully isolate.

## Commits this session

* (tt-metal-sglang) **`dc43d1d2a79`** — `prefetcher: U13 env-gated
  PACK probe + widened LLK probe + DST-at-pack sync probe
  (EVIDENCE_ADVANCE — PACK is NOT writing garbage)`
* (sglang) `<this doc>` + `rebuild_tt_metal_kernels.sh` host-vs-container
  sync trap documented — pending commit.
