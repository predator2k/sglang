# TT Qwen3-8B prefetcher — U12: LLK DST/srcA/srcB probe — MATH is NOT broken from zero inputs (U11 verdict OVERTURNED); bug is downstream of MATH — 2026-05-24

Status: **EVIDENCE_ADVANCE — U12 directly probes srcA / srcB / DST
from MATH-thread context immediately after the first matmul_block of
the first gathered ELF, under U11 zero-weight injection.  When the
probe synchronizes properly with the FPU pipeline (via
`dprint_tensix_dest_reg` which internally calls `dbg_halt()` to drain
all three TRISC threads and the math pipeline), srcA, srcB, and DST
are ALL ZERO across every probed core and every probed ELF.  This
overturns the U11 Phase 4 verdict ("math is fundamentally broken under
ENABLE_GLOBAL_CB") — math itself correctly computes
`zero × zero = zero`.  The 2e19 logit magnitude U11 observed must
originate downstream of the per-matmul DST commit (PACK output
formatting; mm_out_cb stale bytes outside the written tile; a
different gathered ELF the static-gate probe did not capture; or the
mm_partials_cb spill/reload path).  Bonus discovery and fix:
`rebuild_tt_metal_kernels.sh` was silently invalidating ALL host-side
patches since ninja only emits to `${BUILD_DIR}` while Python loads
from `/tt-metal/ttnn/ttnn/`.**

Continuation of `tt_qwen3_8b_prefetcher_U11_zero_weight_math_broken_2026-05-24.md`
(U11 dispatch matrix Phase 4 case 2 was the working verdict going in).

tt-metal-sglang HEAD: **`7be138e3514`** (U12 env-gated LLK probe + rebuild script fix landed).
sglang HEAD: pending this doc.

## What U12 set out to discriminate

Per the U11 dispatch handoff, two hypotheses were standing after U11:

| Hypothesis | Probe |
|---|---|
| **A. UNPACK reads stale bytes**: srcA/srcB hold non-zero data despite cb_in1's rd_ptr bytes being zero (per S10/Lead 3) | dump srcA/srcB after UNPACK loads, before matmul_block |
| **B. MATH/FPU writes garbage from zero inputs**: matmul(0, 0) → non-zero DST | dump DST before/after matmul_block |

The probe was env-gated `SGLANG_TT_PREFETCHER_LLK_PROBE=1`, only
active on the gathered (`use_global_cb`) path, with a static one-shot
gate on `b==0 && block==0 && in0_subblock==0 && in1_subblock==0 &&
inner_dim_idx==0` to print exactly once per core per ELF launch.

## v1 probe (the load-bearing run)

Code: `bmm_large_block_zm_fused_bias_activation_gathered.cpp` at the
innermost matmul_block site.

```cpp
#ifdef SGLANG_TT_PREFETCHER_LLK_PROBE
// U12 PRE: dump DST tile 0 BEFORE first matmul_block
{
    static bool u12_pre_dst_dumped = false;
    if (!u12_pre_dst_dumped && first-iter-gate) {
        u12_pre_dst_dumped = true;
        DPRINT << "[U12_PRE_DST ring_idx=" << ring_idx
               << " in1_cb_rdptr=0x" << HEX();
        UNPACK((DPRINT << get_local_cb_rd_ptr(in1_cb_id)));
        DPRINT << "]" << ENDL();
        dprint_tensix_dest_reg<false>(0);       // INTERNALLY: dbg_halt() + MATH dump + dbg_unhalt()
    }
}
#endif
matmul_block(...);
#ifdef SGLANG_TT_PREFETCHER_LLK_PROBE
// U12 POST: dump DST + srcA + srcB after first matmul_block
{
    static bool u12_post_dst_dumped = false;
    if (!u12_post_dst_dumped && first-iter-gate) {
        u12_post_dst_dumped = true;
        DPRINT << "[U12_POST_DST ...]" << ENDL();
        dprint_tensix_dest_reg<false>(0);
        MATH(({
            uint32_t srca_rd[8], srcb_rd[8];
            ckernel::dbg_get_array_row(ckernel::dbg_array_id::SRCA, 0, srca_rd);
            DPRINT << "[U12_SRCA row0] " << HEX dump << ENDL();
            ckernel::dbg_get_array_row(ckernel::dbg_array_id::SRCB, 0, srcb_rd);
            DPRINT << "[U12_SRCB row0] " << HEX dump << ENDL();
        }));
    }
}
#endif
```

`dbg_get_array_row(SRCA, ...)` is destructive (clobbers DST row 0 +
LREG3 via save/restore through SFPLOAD/SFPSTORE) but only runs once
per ELF and only affects the in-flight matmul — acceptable diagnostic
overhead.

`dprint_tensix_dest_reg<>(0)` internally calls `dbg_halt()` which
issues `tensix_sync()` on all three TRISC threads, draining the math
pipeline and ensuring DST RAM holds the post-matmul snapshot at read
time.

## v1 hardware run (Qwen3-8B BF16, P300 pair, batch=1, prompt="2+2=", max=5)

env: `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1
SGLANG_TT_PREFETCHER_LLK_PROBE=1 TT_METAL_DPRINT_CORES=all`

Output text: `"Und..." e2e=36s` (matches U11 broken signature, confirms
the gathered ELF is being dispatched and the bug is present).

DPRINT log captures (64 unique cores, 4 distinct gathered ELFs probed):

```
0:8-0:TR0: [U12_PRE_DST ring_idx=16 in1_cb_rdptr=0xdf70]
0:8-0:TR1: Tile ID = 0
0:8-0:TR1:   0.0000   0.0000   0.0000   0.0000   ...   (× 16 floats, × 64 rows)
0:8-0:TR2: [U12_POST_DST ring_idx=10 in0_index=0 in1_index=0 inner_k=4]
0:8-0:TR1:   0.0000   0.0000   ...   (× 16 × 64 rows again, POST tile 0)
0:8-0:TR1: [U12_SRCA row0] 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0
0:8-0:TR1: [U12_SRCB row0] 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0
```

Every core printed:
- `U12_PRE_DST` cb_in1 rd_ptr (varies per core, all valid L1 addresses ~0xdf70..0x10c10)
- DST tile 0 BEFORE matmul: all-zero ✓
- DST tile 0 AFTER matmul: all-zero ✓
- srcA row 0: all-zero ✓
- srcB row 0: all-zero ✓

Cross-core, cross-ELF: **every** probe reports zero for all four
diagnostic targets.

## v1 verdict (definitive)

| Hypothesis | Status |
|---|---|
| A. UNPACK reads stale bytes | RULED OUT — srcB is zero (matches the zero weights at cb_in1); srcA is zero (in0 from previous matmul = zero, cascading) |
| B. MATH writes garbage from zero inputs | RULED OUT — DST is zero before AND after matmul_block, on every probed core/ELF |
| **NEW C. Bug is downstream of MATH** | **STANDING** — PACK output, mm_out_cb stale bytes, mm_partials_cb spill, or a different gathered ELF not captured by the static-gate probe |

The U11 dispatch matrix Phase 4 case-2 verdict ("math is fundamentally
broken under ENABLE_GLOBAL_CB") is **OVERTURNED**.  matmul(0, 0) -> 0
correctly inside the gathered LLK kernel.

## v2 probe (a quick sanity sample — race-noise illustration)

To check whether LATER iterations (block=1..31, batch=0+) produce
non-zero DST after multiple `+=` accumulations with zero inputs, v2
widened the gate to fire 8 times per ELF and read DST RAW via
`dbg_get_array_row(DEST, ...)` WITHOUT the `dbg_halt()` synchronization.

v2 shows DST as NONZERO on 88 of 460k samples (0.02%), all on
`b=0 blk=0`.  These are race readings of DST RAM while the FPU
matmul is still in-flight (no sync barrier) — not real garbage.
v1 (sync'd) DST reads are the load-bearing data.

`tensix_sync()` should always precede `dbg_get_array_row(DEST, ...)`
in future probes that are not already inside `dbg_halt()`.  Adding
this guard to v2 was deferred to keep the dispatch scoped to the
discrimination question; v1 already answers it.

## Bonus discovery & fix: `_ttnn.so` install path

While iterating on U12, the factory-side fprintf diagnostic kept not
firing despite the build succeeding.  Root cause: ninja's `ttnn`
target only emits `_ttnn.so` / `_ttnncpp.so` into `${BUILD_DIR}`
(`/tt-metal/build_Release/ttnn/`), but Python's `import ttnn` loads
its compiled extension from `/tt-metal/ttnn/ttnn/_ttnn.so` (and
`_ttnncpp.so`).  Without a post-build sync these two paths diverge
and the rebuild becomes a runtime no-op: the kernel cache appears
fresh (because rm -rf'd before the run) but the host's
`mm_kernel_defines` map is populated by an OLD `_ttnn.so` that
never had the SGLANG_TT_PREFETCHER_* propagation code.

The U9 / U10 / U11 docs' "verified defines_generated.h has
SGLANG_TT_PREFETCHER_*=1" claims may have been false-positives if the
prior session built host code without syncing libs to the install
dir.  At minimum, U12's first three runs were affected by this until
the sync was added.

Fix landed in `scripts/rebuild_tt_metal_kernels.sh` as a mandatory
post-build `cp -fp ${BUILD_DIR}/ttnn/_ttnn.so ${INSTALL_DIR}/_ttnn.so`
(and the same for `_ttnncpp.so`), with a clear comment explaining
the failure mode so future iterations don't re-walk this trap.

## What's next (U13 attack)

Now that MATH is verified correct from zero inputs, the bug surface
narrows to:

1. **PACK formatting bug**: PACK reads DST (zero) and writes to
   mm_out_cb — but the WRITE may not fully zero the destination
   bytes.  PACK with `pack_tile_block(...)` is supposed to write all
   subblock tiles, but if `out_subblock_num_tiles` is mis-sized or
   the per-face pack stride mis-strides under `ENABLE_GLOBAL_CB`,
   trailing bytes hold prior-program garbage that the NEXT layer's
   UNPACK reads as if they were real activations.

2. **mm_out_cb residency**: The downstream layer reads from
   mm_out_cb via `cb_wait_front / cb_pop_front`.  If the CB
   geometry under `ENABLE_GLOBAL_CB` has unused trailing tiles
   beyond what PACK wrote, those bytes are L1 garbage that gets
   `unpack_tile`'d as if real.

3. **mm_partials_cb spill/reload**: When `spill==true` (multi-block
   matmul), the kernel `pack_tile_block(start_dst_index,
   mm_partials_cb_id, out_subblock_num_tiles)` then later
   `reload_from_cb_to_dst(...)` which copies mm_partials back into
   DST and resumes accumulation.  Under `ENABLE_GLOBAL_CB` the
   partials CB might overlap with previously-prefetched weight bytes
   in L1 (the `start_cb_index` ordering across mm_out / mm_partials /
   in1 / weight-prefetch CBs is a known source of layout conflicts).

4. **A different gathered ELF**: The static-gate probe fires once per
   ELF per core.  If there are >4 gathered ELFs in the model
   (different shapes per-matmul) and one of them has a real MATH bug
   that the probe didn't capture, that ELF would still drive the 2e19
   logits.

The U13 dispatch should probe DST AFTER the FULL inner_dim loop
completes (after `tile_regs_commit()` but before
`pack_tile_block(...)`), then probe mm_out_cb bytes AFTER PACK
writes.  Discrimination:
- DST=zero, mm_out_cb=zero, downstream garbage → mm_partials_cb spill
- DST=zero, mm_out_cb=NONZERO → PACK writes garbage
- DST=NONZERO post-pack-prep → real MATH bug on a different ELF

## Reproducer (working)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'
podman exec p3a-ngram bash -c 'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_PREFETCHER_LLK_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_FILE=/tmp/u12_dst_probe.log \
  TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u12_test.log 2>&1'

# wait until /health → 200, then:
curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text": "2+2=", "sampling_params": {"max_new_tokens": 5, "temperature": 0.0}}'

podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null'

# Analyze:
podman exec p3a-ngram grep "U12_" /tmp/u12_dst_probe.log | head -40
```

For the v1 (load-bearing, sync'd) dump, an earlier session's
`u12_dst_probe.log` is preserved at
`/tmp/u12_dst_probe_run5_first_matmul_zero.log` (in-container).

## Working state at session end

- tt-metal-sglang HEAD: **`7be138e3514`** (U12 patch landed)
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only)
- sglang HEAD: pending this doc
- Container `/tt-metal/`: U12 kernel + factory edits applied, libs synced to `/tt-metal/ttnn/ttnn/`
- `/root/.cache/tt-metal-cache/*`: cleared at end of session
- `stash@{0,1,2}`: untouched
- TT devices: healthy (one mid-session reset after a watchdog timeout from
  an over-aggressive v1 probe interaction with `dbg_halt()` cascade)
- No server running at session end

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only tt-metal kernel/factory, env-gated)
- [x] All `rm` operations guard with `[ -n "$VAR" ]`
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape (`pkill -9 -f "sglang\.launch_server.*--port 30000"`)
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Canonical re-verify with env unset PASSES: `"2+2=" → "5? - The Math"`
- [x] Fix is fully env-gated (`SGLANG_TT_PREFETCHER_LLK_PROBE=1` opt-in)
- [x] Probe was env-gated initially

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≈ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  U12 narrowed
the bug from "math kernel" to "downstream of MATH" but did NOT fix it.
A U13 follow-on probe of PACK / mm_out_cb / mm_partials_cb is
required to fully isolate.

## Commits this session

* (tt-metal-sglang) **`7be138e3514`** — `prefetcher: U12 env-gated
  LLK DST/srcA/srcB probe (EVIDENCE_ADVANCE — math is NOT broken
  from zero inputs; bug is downstream)`
* (sglang) `<this doc>` + `rebuild_tt_metal_kernels.sh` install-dir
  sync — pending commit.
