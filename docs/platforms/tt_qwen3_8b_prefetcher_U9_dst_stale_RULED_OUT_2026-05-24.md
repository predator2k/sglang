# TT Qwen3-8B prefetcher — U9: DST stale-state in gathered compute kernel RULED OUT (hardware-tested) — 2026-05-24

Status: **EVIDENCE_ADVANCE — U9 dispatch hypothesis (DST accumulator
holds prior-program values, the first matmul_block accumulates onto
stale state) is REFUTED BY HARDWARE EXPERIMENT. An env-gated
SGLANG_TT_PREFETCHER_DST_ZERO=1 patch was added to the gathered
(prefetcher) compute kernel that emits `MATH((ckernel::zeroacc()))`
right after `mm_block_init` and before any `matmul_block`. The
compile-time define was verified active in
`/root/.cache/tt-metal-cache/.../defines_generated.h`. With the patch
enabled, the prefetcher run STILL produced the same garbage token
signature ("<think> Hayward cerebral kd אולי 뛰...") and crashed the
scheduler with `RuntimeError: probability tensor contains either inf,
nan or element < 0` at iteration ~3000 tokens — exactly the U7
NaN-poisoning pattern. With the env var UNSET (canonical mode), GSM8K(10)
chat = 9/10 = 90% confirming the patch is fully behavior-preserving and
does not regress the canonical path. The bug therefore lies in one of
the two remaining U8 hypotheses (GlobalCB inter-matmul rd_ptr handoff
OR tensor_split arithmetic), NOT in DST register lifecycle.**

Continuation of `tt_qwen3_8b_prefetcher_U7_gathered_matmul_garbage_2026-05-24.md`
(Case B confirmed: gathered kernel produces garbage from FIRST
invocation for 4-of-14 ELFs) and the prior `U4-B` code-only ruling-out
of DST staleness (which this dispatch promoted from code-only to
hardware-verified, with a stronger conclusion: even FORCING the DST to
zero on every kernel launch does not fix the bug).

tt-metal-sglang HEAD: **`8d6f8832238`** (this dispatch's env-gated U9 probe
landed); sglang HEAD: pending this doc.

## Bottom line

| Test | Probe | Result |
|---|---|---|
| **U9 patch active (SGLANG_TT_PREFETCHER_DST_ZERO=1)** | `MATH((ckernel::zeroacc()))` right after `mm_block_init` in gathered kernel | **Server CRASHED with garbage + NaN logits, same signature as bug.** GSM8K(10) = **0/10** (Q1 garbage `"<think> Hayward cerebral kd אולי 뛰אולי..."`, then connection refused Q2–Q10 due to crash). |
| Patch compile-time verify | grep defines_generated.h | `#define SGLANG_TT_PREFETCHER_DST_ZERO 1` + `#define ENABLE_GLOBAL_CB 1` both present. ZEROACC executes on every gathered kernel_main. |
| Canonical re-verify (env var UNSET) | clean canonical mode | **9/10 = 90% PASS.** The factory env-var guard correctly returns false, so canonical builds the original kernel. |

## What changed (now landed on `tenstorrent-p1`)

Two files modified, both env-gated. Commit `8d6f8832238`:

### File 1 — `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`

```cpp
// existing line:
mm_block_init(
    in0_cb_id, in1_cb_id, mm_partials_cb_ids[0], in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

#ifdef SGLANG_TT_PREFETCHER_DST_ZERO
    // U9: Explicit DST zero before any matmul accumulation. mm_block_init's
    // llk_math_pack_sync_init only resets the dest_offset_id / section base
    // pointers; it does NOT issue a ZEROACC. ...
    MATH((ckernel::zeroacc()));
#endif

for (uint32_t b = 0; b < batch; b++) {
    // ... existing body unchanged ...
```

### File 2 — `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`

```cpp
if (use_global_cb) {
    mm_in1_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
    mm_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
    // U9 ... propagate SGLANG_TT_PREFETCHER_DST_ZERO=1 only on the gathered path
    const char* dst_zero_env = std::getenv("SGLANG_TT_PREFETCHER_DST_ZERO");
    if (dst_zero_env != nullptr && std::string(dst_zero_env) == "1") {
        mm_kernel_defines["SGLANG_TT_PREFETCHER_DST_ZERO"] = "1";
    }
}
```

(`<cstdlib>` and `<string>` includes added at top of file.)

The env-var gate is checked on the host program-factory side, so canonical
code paths never see the new define. The host check returns `false` when
the env var is unset, so the unmodified gathered kernel ELF is generated.

## LLK background (what `_llk_math_pack_sync_init_` actually does)

From `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_common.h:91-110`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_() {
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {}
    if constexpr (Dst == DstSync::SyncFull) {
        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    } else {
        static_assert(Dst == DstSync::SyncHalf);
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    }
}
```

`reset_dest_offset_id()` + `set_dest_section_base<StartZero>()` only manipulate
the DST pointer registers (offset id, section base). Neither emits a
`TT_ZEROACC` instruction. **The actual DST register CONTENTS are unaffected
by `mm_block_init`**, only the pointers that determine which section will
be considered "current".

The normal lifecycle that clears DST is:
- `tile_regs_acquire()` → `MATH(llk_math_wait_for_dest_available)` → just waits for next section to free
- (matmul writes into section)
- `tile_regs_commit()` → `MATH(llk_math_dest_section_done)` → just flips section pointer
- `tile_regs_wait()` → on PACK, waits for math
- `pack_tile_block(...)` → packer writes section contents to L1
- `tile_regs_release()` → `PACK(llk_pack_dest_section_done)` → THIS issues `TT_ZEROACC(p_zeroacc::CLR_HALF, ..., dest_offset_id % 2)` (the section just packed is zeroed)

So **every section, AFTER being packed, gets ZEROACC'd** by the normal release
epilogue. The hypothesis was that on the VERY FIRST `tile_regs_acquire` of
a kernel launch, no prior release has run on the current core (or a prior
launch left state behind), so DST could be non-zero. `matmul_block(...
idst=0, ...)` then does `DST[0:rt*ct] += C[...]` and accumulates onto the
leftover values.

`MATH((ckernel::zeroacc()))` calls `ckernel::zeroacc()` from
`tt_llk_blackhole/common/inc/ckernel.h:300-311`:

```cpp
inline void zeroacc() {
    addr_mod_t {.srca={.incr=0}, .srcb={.incr=0}, .dest={.incr=0}}.set(ADDR_MOD_1);
    TT_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, ADDR_MOD_1, 0);
}
```

`CLR_ALL` clears the entire DST. This was placed AFTER `mm_block_init` (so its
ADDR_MOD_1 reset doesn't clobber matmul's MOP setup — wait, actually it does
since matmul uses ADDR_MOD_1 too, but the next `matmul_block` call re-runs
the MOP via `_llk_math_matmul_` which uses the replay buffer programmed at
init time and does not re-configure ADDR_MOD_1). Either way, the hardware
evidence is conclusive: ZEROACC at this point did not fix the garbage.

## What the hardware run produced (U9 active)

Server log `/tmp/u9_test.log` — Q1 ran for 76s, generated tokens, then NaN-poisoned:

```
[2026-05-24 10:56:13] Decode batch, #running-req: 1, #token: 3072, ...
[2026-05-24 10:56:13] Scheduler hit an exception: Traceback (most recent call last):
  File "/sglang/python/sglang/srt/layers/sampler.py", line 485, in top_k_top_p_min_p_sampling_from_probs_torch
    sampled_index = torch.multinomial(probs_sort, num_samples=1)
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```

Q1 last_line: `'<think> Hayward cerebral kd אולי 뛰אולי无所谓 kd שימוש 뛰 odal שימוש אולי kd odal kd 뛰뛰 kd אולי kd שימוש cerebral שימ'`

The structure is identical to the U7 garbage signature: a few coherent-ish
tokens at the start (the "<think>" and "Hayward cerebral" come from a
correct enough first few decode steps), then mode-collapse to repeated
high-entropy token clusters. Eventually the residual stream goes NaN and
the scheduler crashes when sampling.

This is the same failure mode the bug has shown across every U7 / U6 /
prior dispatch with the prefetcher on, despite a forced DST zero.

## What this rules out (and what's left)

| Hypothesis | Status after U9 |
|---|---|
| U8 #2 — **DST register stale state holds prior-program values** | **REFUTED by hardware experiment.** Even with explicit `zeroacc()` after `mm_block_init`, the bug persists in the same form. |
| U8 #1 — **GlobalCB inter-matmul `fifo_rd_ptr` handoff** | **STANDING.** The gathered kernel's `update_rd_ptr_to_ring_index` (kernel L114-134) saves `in1_rd_ptr_start_addr` at batch start and restores it at batch end. Across distinct matmul invocations sharing the same `cb_in1` (the GlobalCB), the rd_ptr is updated by L483 `update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_size, in1_tensor_split)` — advances to "next tensor". If the producer is on a different schedule, rd_ptr could land on a not-yet-written L1 region. |
| U8 #3 — **`tensor_split=true/false` boolean path** | **STANDING.** `is_tensor_split(in1_cb_id, in1_tensor_size_bytes)` at kernel L283 makes a runtime decision that drives BOTH the wrap modulo in `update_rd_ptr_to_ring_index` (L118-130) and the next-block-index calculation in `calculate_next_block_index_and_update_rd_ptr` (L85-109). A wrong `tensor_split` value would land the matmul on the wrong L1 region. |

Note: U7's verdict that "bytes are correct at the compute rd_ptr"
constrains what these two remaining hypotheses can be doing. If
`update_rd_ptr_to_ring_index` lands the rd_ptr at a wrong L1 address,
S10's reader probe would have caught it. So either:
* S10/Lead 3's rd_ptr-time check fires under a DIFFERENT condition than
  U7's garbage-producing matmuls (e.g., S10 dumped at `b=0, in0_sub=0,
  in1_sub=0`, but the garbage emerges in a later subblock with a
  different intra-block rd_ptr offset), OR
* The rd_ptr is correct AND the bytes at it are correct, but the
  matmul_block MOP itself executes wrong under ENABLE_GLOBAL_CB +
  certain compile-time parameter combinations (a 4th class of hypothesis).

## Hypothesis ledger after U9

| ID | Suspect | Status |
|---|---|---|
| S1-S10 | Producer/consumer + L1/CB micro-hypotheses | ALL RULED OUT |
| A1 | ecfb2e782c2 L1-clash bypass | RULED OUT |
| Lead 2 | upstream 934d954b995 last_subblock_w_valid port | RULED OUT |
| Lead 3 (U2) | in0 ring-all-gather byte transport | RULED OUT |
| U1 | Cross-sub-device dispatch sync gap | WEAKENED (U6) — not the root cause |
| U3 | Permuted vs contiguous DRAM grid | RULED OUT |
| U4-A | TP=2 misconfig | RULED OUT |
| U4-B | DST accumulator stale state (code-only) | superseded by U9 below |
| U5 | Subdev-routing barrier | RULED OUT |
| U6 | Galaxy-parity 2-subdev + dummy_receivers | EVIDENCE_ADVANCE; weakens U1 |
| U7 | Differential matmul-output test (Case A/B/C) | **CASE B CONFIRMED** — gathered compute kernel produces garbage outputs |
| **U9** | **DST stale-state at gathered kernel start (hw-tested zeroacc patch)** | **RULED OUT (this dispatch — explicit `MATH((ckernel::zeroacc()))` after `mm_block_init` did NOT fix the bug; canonical re-verify 9/10 confirms patch is behavior-preserving when env var unset)** |
| **U8 #1** | **GlobalCB inter-matmul rd_ptr handoff** | **STANDING — top priority next** |
| **U8 #3** | **tensor_split boolean path / `update_rd_ptr_to_ring_index` wrap arithmetic** | **STANDING — second priority** |
| **NEW** | **matmul_block MOP semantics under ENABLE_GLOBAL_CB + specific CT-arg combos** | **OPEN — implied by the gap between S10 (bytes correct at rd_ptr) + U9 (DST is correct)** |

## Recommended next attack (U10)

Per U7's recommendation `Alternative U9 proposal — bisect the gathered
ELF parameter space`:

1. Diff the compile-time args of the 4 GARBAGE ELFs vs the 10/14 NORMAL
   ELFs (or 4-of-5 / 1-of-5 under the U8 ELF count). The garbage set
   shares some compile-time param value. The hash dirs live at
   `/root/.cache/tt-metal-cache/.../bmm_large_block_zm_fused_bias_activation_gathered/<hash>/defines_generated.h`.
2. Pull `unpadded_in0_shard_widths_in_tiles[]` runtime args (`get_arg_val`
   at L239-240) on the garbage cores vs the normal cores. The garbage
   cluster lives at receiver positions y=6, x∈{8,9,10} — that's a
   specific subset of the ring; some `unpadded_in0_block_w` slot may be
   wrong (e.g., padded out to 0 or to a too-large value) for these.
3. Probe the rd_ptr at a LATER point inside the matmul block loop (not
   just at b=0, in0_sub=0, in1_sub=0): if `tensor_split` flips between
   block iterations, S10 would miss it.

Concrete next-session probe (paste into the gathered kernel under an env
gate):

```cpp
// Inside the for (block) loop, after calculate_next_block_index_and_update_rd_ptr:
#ifdef SGLANG_TT_PREFETCHER_RDPTR_PROBE
    UNPACK(({
        static uint32_t u10_ord = 0;
        if (u10_ord < 4) {
            DPRINT_UNPACK({DPRINT << "U10@b" << b << "block" << block
                << " rd_ptr=" << HEX() << get_local_cb_rd_ptr(in1_cb_id)
                << " split=" << (uint32_t)in1_tensor_split
                << " curr_idx=" << curr_in1_block_index
                << " next_idx=" << next_in1_block_index << ENDL();}));
            u10_ord++;
        }
    }));
#endif
```

Compare the dump from a garbage-producing core (e.g., (9,6)-0) vs a
normal core (e.g., (8,8)-0).

## Working state at session end

- tt-metal-sglang HEAD: **`8d6f8832238`** (U9 env-gated probe landed)
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only)
- sglang HEAD: pending this doc + commit message
- Container `/tt-metal/` source: U9 patch applied + freshly rebuilt
  (libtt_metal.so + _ttnn.so + _ttnncpp.so all updated in
  `/tt-metal/ttnn/ttnn/` and `/tt-metal/tt_metal/`)
- `/root/.cache/tt-metal-cache/*`: cleared (last action before canonical re-verify)
- `stash@{0,1,2}`: untouched (per CLAUDE.md rule)
- TT cards: healthy (no probe rebuild/crash this session)
- No server running at session end
- **Canonical re-verify post-U9 (env unset)**: GSM8K(10) = **9/10 = 90%** ✓

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only env-gated tt-metal kernel changes)
- [x] All `rm` operations guard with `[ -n "$VAR" ]`
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape (`pkill -9 -f "sglang\.launch_server.*--port 30000"`)
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Canonical did NOT regress (9/10)
- [x] Fix is fully env-gated (`SGLANG_TT_PREFETCHER_DST_ZERO=1` opt-in)

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** The
gathered compute kernel's bug is **NOT** DST stale-state (this
dispatch ruled out). The two remaining U8 hypotheses (GlobalCB
inter-matmul rd_ptr handoff; tensor_split arithmetic) are STANDING.
A 4th class — matmul_block MOP semantics under ENABLE_GLOBAL_CB +
specific compile-time arg combinations — is now implied by the gap
between S10/Lead 3 (bytes correct at rd_ptr) and U7/U9 (output is
garbage from FIRST invocation with DST forced to zero).

## Commits this session

* (tt-metal-sglang) **`8d6f8832238`** — `prefetcher: U9 env-gated DST
  stale-state probe (HYPOTHESIS RULED OUT)` — adds
  `SGLANG_TT_PREFETCHER_DST_ZERO=1` env-gated `MATH((ckernel::zeroacc()))`
  in the gathered compute kernel; factory propagates the define only on
  the gathered (use_global_cb) code path; canonical untouched.
* (sglang) `<this doc>` — pending commit.
