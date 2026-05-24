# TT Qwen3-8B prefetcher — U10: GlobalCB block-bisect (3 surgical bypasses) ALL RULED OUT — 2026-05-24

Status: **EVIDENCE_ADVANCE — All 3 surgical ENABLE_GLOBAL_CB block
bypasses (Block A INIT, Blocks B+C BLOCK, Block D ADVANCE) were
implemented as env-gated CT defines, plumbed into the factory only on
the gathered path, hardware-tested individually with full kernel cache
clear, and verified active via `defines_generated.h` inspection. Each
bypass produced wrong GSM8K Q1 with a DIFFERENT garbage signature, but
all 3 stayed in the same bug class (mode-collapsed multi-script
high-entropy token strings) — none restored correctness. Notable
secondary observation: with any of the 3 bypasses active, Q1 decoded
all 3000 tokens WITHOUT NaN-poisoning (vs U9 baseline which crashed
at ~3000 with `RuntimeError: probability tensor contains either inf,
nan or element < 0`); Q2 always hung at `[V2.5] pending DECODE
restore`. Canonical re-verify (env unset) is the gate for behavior-
preservation. The bug class is therefore NOT localized to any of the
3 ENABLE_GLOBAL_CB-gated `for (b…)` regions. Per Phase 4 of the
dispatch matrix, the bug surface now extends to LLK-level differences
under `RemoteSenderCBInterface`/`RemoteReceiverCBInterface` vs
`LocalCBInterface`, per-receiver-core behavior (galaxy=2 vs ours=4),
or `noc_async_writes_flushed` semantics under shared-receiver
patterns.**

Continuation of `tt_qwen3_8b_prefetcher_U9_dst_stale_RULED_OUT_2026-05-24.md`
(U9 ruled out DST stale-state on hardware). U7 Case B remains the
load-bearing characterization (4 of 14 gathered ELFs produce
~2^60..2^109 magnitude outputs from first invocation).

tt-metal-sglang HEAD: **`bd2f43660ac`** (U10 env-gated 3-way bypass landed).
sglang HEAD: pending this doc.

## Map of the 4 ENABLE_GLOBAL_CB-gated blocks in gathered kernel

`bmm_large_block_zm_fused_bias_activation_gathered.cpp` — line numbers
shown are post-U9 (the U9 DST_ZERO insertion at L259-269 shifted everything
below by ~13 lines). Pre-U9 dispatch's spec line ranges (259-272,
323-334, 471-484) are off by this offset.

| ID | Lines | Trigger | Purpose | Variables it sets / mutates |
|---|---|---|---|---|
| **A** (INIT) | L272-285 | `for (b … batch)` start | Capture start-of-this-matmul rd_ptr; probe tensor_split path; advance rd_ptr to this core's ring slot | `in1_cb_start_addr`, `in1_rd_ptr_start_addr`, `curr_in1_block_index = ring_idx`, `in1_tensor_split`; mutates `local_cb.fifo_rd_ptr` by `ring_idx * block_size_bytes` |
| **B** (BLOCK calc) | L336-347 | `for (block … num_blocks)` start | Compute next block index/ptr that Block C will apply | `next_in1_block_index`, `next_in1_rd_ptr_addr` (locals only; no CB mutation) |
| **C** (BLOCK advance) | L484-487 | `for (block …)` end | Apply Block B's next ptr to this block's CB rd_ptr | `curr_in1_block_index = next_in1_block_index`; mutates `local_cb.fifo_rd_ptr = next_in1_rd_ptr_addr` |
| **D** (END handoff) | L490-497 | `for (b …)` end (after sync_buf.push_back) | Reset rd_ptr to saved start; advance by `ring_size` to "next tensor"; INTER-matmul handoff | mutates `local_cb.fifo_rd_ptr` (restore + advance by `ring_size * block_size_bytes`) |

The dispatch directive listed 3 blocks (INIT, BLOCK, ADVANCE). Block C
is conceptually paired with Block B (C consumes the locals B writes —
bypassing B without C would read uninitialized values). So the
mapping `BYPASS_GCB_BLOCK ↔ skip {B, C}` is correct as a single
bisect probe.

## What each bypass does to the runtime

| Bypass | Effect on CB rd_ptr | Effect on inter-matmul handoff |
|---|---|---|
| **INIT** | No `ring_idx` offset; cores read same ring slot. Block A's locals still init (`in1_rd_ptr_start_addr` captures the pre-offset value), so Block D's restore + advance still works correctly. | Preserved — Block D restores to saved start + advances by `ring_size`. |
| **BLOCK (B+C)** | rd_ptr stays at Block-A position for all `num_blocks` iterations — kernel reads block 0 num_blocks times. | Preserved — Block A and D unaffected. |
| **ADVANCE** | rd_ptr leaks past the matmul's tensor (left wherever Block C put it = mid-tensor at end of loop). | BROKEN — next matmul's Block A captures wrong `in1_rd_ptr_start_addr` and adds another `ring_idx` offset on top. Cascading drift across matmuls. |

INIT and BLOCK are non-cascading (single-matmul-scope effects). ADVANCE
is cascading. If any single block contained the bug, we'd expect that
bypass to either (a) FIX correctness if the block introduces the
corruption, or (b) make things catastrophically worse if the block
is a safety mechanism. NEITHER happened.

## Hardware iteration table (4 runs)

| Test | env_extra | Boot | Cache verified | Q1 result | Q1 garbage signature | Q2 fate |
|---|---|---|---|---|---|---|
| **U10-INIT** | `SGLANG_TT_PREFETCHER_BYPASS_GCB_INIT=1` | OK (40s) | `defines_generated.h` ✓ | WRONG, 3000 toks, no NaN | `<think>咬KEבסופו wspבסופו.MATCH wspבסופו.MATCH.MATCH wsp.MATCH.MATCHKE.MATCH wspKEבסופו.MATCHבסופו.MA` | Hung at `[V2.5] pending DECODE restore` |
| **U10-BLOCK** | `SGLANG_TT_PREFETCHER_BYPASS_GCB_BLOCK=1` | OK (40s) | `defines_generated.h` ✓ | WRONG, 3000 toks, no NaN | `全能 (*(cents Bearing孫两千運用气血()['国会的带领下留住;%規劃 SoxQuddenessenessen宽敞 Ble插入前三季度_te Hyp imgUrl�記事你会发现 dark` | Hung at `[V2.5] pending DECODE restore` |
| **U10-ADVANCE** | `SGLANG_TT_PREFETCHER_BYPASS_GCB_ADVANCE=1` | OK (40s) | `defines_generated.h` ✓ | WRONG, 3000 toks, no NaN | `RgbactersRgbRgbRgbactersRgbRgbactersRgbRgbactersRgbactersactersactersactersRgbactersRgbRgbactersacte` | Hung at `[V2.5] pending DECODE restore` |
| **U10-canonical re-verify** | _none_ (no prefetcher, no bypass) | OK (35s) | n/a | **9/10 = 90% PASS** | `#### 64` (Q1 correct) | Completed cleanly |

Canonical re-verify confirmed: U10 patches are fully behavior-preserving
when env vars are unset. The 3 `#ifndef`-guards correctly fall through
to the original blocks when no `SGLANG_TT_PREFETCHER_BYPASS_GCB_*` define
is propagated; the host factory check returns false on missing env var.

Full canonical Q-by-Q (5-min run): Q1✓ Q2✓ Q3✗(reasoning correct but
answer extraction got "80" not "160") Q4✓ Q5✓ Q6✓ Q7✓ Q8✓ Q9✓ Q10
implicit (FINAL line printed before Q10 was logged but FINAL=9/10
indicates Q10✓). Matches U9 canonical baseline 9/10 exactly.

## Cross-comparison vs U9 baseline

| Probe | U9 zeroacc | U10 INIT | U10 BLOCK | U10 ADVANCE |
|---|---|---|---|---|
| Q1 reaches 3000 toks | NO (NaN-crash) | YES | YES | YES |
| Q1 garbage class | "<think> Hayward cerebral …" | "<think>咬KE…" | "全能(*(cents…" | "Rgbactersacters…" |
| NaN poisoning | YES (~3000 toks) | NO | NO | NO |
| Q2+ usable | NO (scheduler crashed) | NO (Python hang) | NO (Python hang) | NO (Python hang) |

All 4 (U9 + U10×3) ship the prefetcher's underlying bug; none fix it.
The U10 runs DID reliably get further into the decode than U9 — i.e.
the garbage magnitudes from the 4 garbage ELFs (per U7) are
LESS extreme when any of the 3 GlobalCB blocks is bypassed, even
though the bug class persists. This implies the 3 blocks DO
contribute to the magnitude (probably by compounding wrong values
across iterations), but they are NOT the SOURCE of the bug.

## What this rules out

| Hypothesis (from U9 ledger) | U10 verdict |
|---|---|
| **U8 #1 — GlobalCB inter-matmul rd_ptr handoff via Block D** | **RULED OUT.** Bypassing Block D's `update_rd_ptr_to_ring_index(…, ring_size, …)` did NOT fix the bug. The cascading inter-matmul drift only changes the garbage signature, not the bug class. |
| **U8 #3 — `tensor_split` boolean path / wrap arithmetic in `update_rd_ptr_to_ring_index`** | **RULED OUT.** Bypassing Block A (the only site that PROBES tensor_split) forces the value to false; this didn't restore correctness. Likewise BLOCK bypass routes around `calculate_next_block_index_and_update_rd_ptr` (the other tensor_split consumer) and didn't help. |
| **Block A's `update_rd_ptr_to_ring_index(…, ring_idx, …)` — wrong start offset** | **RULED OUT.** Bypassing the offset (cores all read slot 0) → garbage, not correct. |
| **Block B/C's `calculate_next_block_index_and_update_rd_ptr` — wrong intra-matmul advance** | **RULED OUT.** Bypassing the per-block advance (cores read block 0 num_blocks times) → garbage, not correct. |

## What's NOW the suspect surface

Per Phase 6 of the dispatch ("If U10 ruled out — pivot") and per the
"new" 4th-class hypothesis seeded by U9:

1. **LLK / firmware-level differences under `Remote*CBInterface`.**
   The gathered kernel uses `get_local_cb_interface(in1_cb_id)` for the
   GlobalCB on the compute side, but the GlobalCB is set up on the
   producer (writer_l1.cpp) using `RemoteSenderCBInterface`. The
   matmul LLK ops (`matmul_block`, `mm_block_init`,
   `pack_tile_block`) take the SAME compile-time paths in gathered
   vs canonical builds — the only difference IS `ENABLE_GLOBAL_CB`
   gating the 4 blocks above. But `mm_block_init` calls
   `_llk_math_pack_sync_init_` which is templated on `DstSync` enum;
   under `tt_metal/api/tt_metal/tile_regs_modes.hpp` the default is
   `SyncHalf` (which differs from `SyncFull` only in semaphore index).
   No obvious gate based on CB-setup.

2. **Per-receiver-core behavior asymmetry.** Galaxy uses 2 receivers
   per sender; our P300-pair fabric uses 4 receivers per sender
   (per U6's galaxy-parity probe). The 4-garbage-ELFs cluster from
   U7 lives at receivers (X,6) for X∈{8,9,10} on both devices — a
   subset of the 4-receiver layout. The 1-of-5 LM-head ELF (the
   only correct one, per U8) is on the non-prefetcher path entirely.
   A possible bug class: the in1 GlobalCB hardware-level CB write
   wraps around the L1 region in a way that ASSUMES 2 receivers
   per sender (no fragmentation across receivers).

3. **`noc_async_writes_flushed` / async-write completion semantics
   under shared-receiver pattern.** Producer's
   `noc_async_writes_flushed` waits for ALL inflight writes on the
   issuing core, but with shared receivers the producer's perceived
   "done" may not mean the receiver's L1 is consistent with the
   compute kernel's view. The S10/Lead 3 probes are point-in-time
   at the compute rd_ptr — they wouldn't catch a torn write that
   happens slightly later.

## Recommended next attack (U11)

1. **Direct LLK probe**: instrument `_llk_math_matmul_` inner-loop
   (in tt_llk_blackhole) to DPRINT the first 4 tiles of input it
   sees on a garbage core (e.g. (9,6)-0) vs a normal core (e.g.
   (8,8)-0). If the LLK sees DIFFERENT input bytes than what S10
   observed at the rd_ptr, the bug is between rd_ptr-time read and
   LLK input register load — i.e. somewhere in the unpacker /
   srcA/srcB tile path. If LLK sees the SAME bytes, the bug is in
   `matmul_block`'s math.

2. **Diff the 4-garbage vs 10-normal compile-time arg dictionaries**
   (from U7's "Alternative U9 proposal" — still standing). The
   garbage cluster shares some CT-arg value. Concrete script:

   ```bash
   for d in /root/.cache/tt-metal-cache/.../bmm_large_block_zm_fused_bias_activation_gathered/*; do
       hash=$(basename $d)
       echo "=== $hash ==="
       cat $d/defines_generated.h
       cat $d/runtime_args.h 2>/dev/null
   done | tee /tmp/u11_gathered_args.log
   ```

   Then cross-reference with the U7 garbage-vs-normal core list.

3. **Switch from `RemoteSenderCBInterface` (2-receiver pattern) to
   single-receiver per sender** (host-side reconfigure). If the bug
   disappears with 1 receiver per sender, the bug is in
   multi-receiver hardware writeback semantics.

## Hypothesis ledger after U10

| ID | Suspect | Status |
|---|---|---|
| S1-S10 | Producer/consumer + L1/CB micro-hypotheses | ALL RULED OUT |
| A1 | ecfb2e782c2 L1-clash bypass | RULED OUT |
| Lead 2 | upstream 934d954b995 last_subblock_w_valid port | RULED OUT |
| Lead 3 (U2) | in0 ring-all-gather byte transport | RULED OUT |
| U1 | Cross-sub-device dispatch sync gap | WEAKENED (U6) |
| U3 | Permuted vs contiguous DRAM grid | RULED OUT |
| U4-A | TP=2 misconfig | RULED OUT |
| U4-B | DST accumulator stale state (code-only) | superseded by U9 |
| U5 | Subdev-routing barrier | RULED OUT |
| U6 | Galaxy-parity 2-subdev + dummy_receivers | EVIDENCE_ADVANCE; weakens U1 |
| U7 | Differential matmul-output test | CASE B CONFIRMED (load-bearing) |
| U9 | DST stale-state (hw-tested zeroacc) | RULED OUT |
| U8 #1 | GlobalCB inter-matmul rd_ptr handoff (Block D) | **RULED OUT (this dispatch — U10 ADVANCE bypass)** |
| U8 #2 | DST stale-state | RULED OUT (U9) |
| U8 #3 | tensor_split boolean / wrap arithmetic | **RULED OUT (this dispatch — U10 INIT + BLOCK bypasses both routed around it; bug persists)** |
| U10 #A | Block A ring-index advance | **RULED OUT (this dispatch)** |
| U10 #B | Block B/C per-block rd_ptr update | **RULED OUT (this dispatch)** |
| **NEW #1** | **LLK-level matmul behavior under `Remote*CBInterface` shared-receiver pattern** | **OPEN (highest priority)** |
| **NEW #2** | **Per-receiver-core asymmetry (4-rx vs galaxy 2-rx)** | **OPEN** |
| **NEW #3** | **CT-arg discriminator between 4-garbage vs 10-normal ELFs (U7 alternative proposal)** | **OPEN** |

## Working state at session end

- tt-metal-sglang HEAD: **`bd2f43660ac`** (U10 3-way bypass landed)
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only)
- sglang HEAD: pending this doc + commit
- Container `/tt-metal/` source: U10 patches applied and rebuilt
  (libtt_metal.so + _ttnn.so all updated)
- `/root/.cache/tt-metal-cache/*`: cleared between each test
- `stash@{0,1,2}`: untouched (per CLAUDE.md rule)
- TT cards: healthy (single `tt-smi -r 0,1` per run between bypass tests
  because BLOCK / ADVANCE runs left Device 0/1 firmware in a state that
  required reset before the next bringup)
- No server running at session end
- Cache artifacts: `/tmp/u10_init/`, `/tmp/u10_block/`, `/tmp/u10_advance/`,
  `/tmp/u10_canonical/` each contain `server.log` + `gsm.log` + `result.txt`

## Hard constraints checked

- [x] No upstream PR (predator2k/* only)
- [x] No SGLang core behavior changes (only env-gated tt-metal kernel + factory)
- [x] All `rm` operations guard with `[ -n "$VAR" ]`
- [x] No stash pop/drop
- [x] Port-clear used `\.` escape (`pkill -9 -f "sglang\.launch_server.*--port 30000"`)
- [x] `HF_MODEL=/models/Qwen3-8B` local path
- [x] Cache clear literal absolute path with var guard
- [x] Canonical re-verify: **9/10 = 90% PASS** (env vars unset; matches U9 baseline)
- [x] Fix is fully env-gated (3 separate `SGLANG_TT_PREFETCHER_BYPASS_GCB_*=1`
      opt-in defines; defaults preserve U9's behavior precisely)

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≈ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** All 3
ENABLE_GLOBAL_CB block-bisect bypasses confirmed wrong, in 3 different
garbage signatures but the same bug class. The bug is elsewhere in
the gathered execution path — most likely at the LLK / firmware level
where the 4 `#ifdef ENABLE_GLOBAL_CB` blocks above did NOT have a
discoverable hook.

## Commits this session

* (tt-metal-sglang) **`bd2f43660ac`** — `prefetcher: U10 env-gated
  GlobalCB block-bisect probes (ALL 3 RULED OUT)` — adds 3 env-gated
  `SGLANG_TT_PREFETCHER_BYPASS_GCB_{INIT,BLOCK,ADVANCE}=1` defines in
  the gathered compute kernel; factory propagates them only on the
  gathered (use_global_cb) code path; canonical untouched.
* (sglang) `<this doc>` — pending commit.
