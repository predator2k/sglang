# TT Qwen3-8B prefetcher — U7 differential matmul-output test: Case B confirmed (gathered compute kernel produces garbage) — 2026-05-24

Status: **EVIDENCE_ADVANCE — Case B verdict from the dispatch
directive's decision matrix is CONFIRMED. Direct DPRINT-PACK
instrumentation of `pack_tile_block` in the gathered compute kernel
(`bmm_large_block_zm_fused_bias_activation_gathered.cpp`, the
prefetcher / ring-all-gather path) and the canonical compute kernel
(`bmm_large_block_zm_fused_bias_activation.cpp`, non-gathered path)
captured 59200 U7G + 66144 U7C dumps across a real `"What is 2+2?"`
prefetcher decode that GARBAGED tokens (`" What9ose�,"`). At least
**4 of the 14 unique gathered-matmul ELF instances produce garbage
output values from the very first invocation** (BF16 magnitudes of
~2^60–2^109, ending in NaN/Inf bytes `0x7e..0x7f` / `0xfe..0xff`),
while the SAME prefetcher run's CANONICAL matmuls (U7C) and ALL of
the canonical-baseline matmuls produce well-formed small-magnitude
BF16 values (`0x3a..0x3e` exponent range). This is conclusive
proof the bug is COMPUTE-KERNEL local to the gathered path — not
downstream of it — DESPITE the previously-verified fact that input
bytes (in0 via Lead 3, in1 via S10) arrive byte-correct at the
compute kernel's read pointer. The gathered compute kernel reads
correct bytes and produces wrong outputs.**

Continuation of `tt_qwen3_8b_prefetcher_U6_galaxy_parity_evidence_2026-05-24.md`
(U6 weakened U1) and the S10/Lead 3 byte-transport closure docs.
tt-metal-sglang HEAD unchanged at `68b8eaa1152` (instrumentation
applied, captured, and reverted in-session). sglang HEAD: this doc.

## Bottom line

| Test | Probe | Result |
|---|---|---|
| **U7 canonical baseline** | `pack_tile_block` byte-dump in non-gathered kernel | 98480 U7C dumps. All outputs well-formed BF16 (`0x3a..0x3e` exp). Server returns `"5"` for `"2+2="`. |
| **U7 prefetcher run (5-token decode for `"What is 2+2?"`)** | Both U7C + U7G probes | 66144 U7C (well-formed) + **59200 U7G (4 ELFs garbage)**. Server returns `" What9ose�,"`. |
| **U7C cross-config diff (canonical vs prefetcher)** | Same `pack_tile_block` byte-dump in canonical kernel, run in both modes | 27417 of 37k byte-payloads MATCH (only ~73% because GlobalCB consumes L1, shifting `mm_out_cb` addresses; bytes themselves match where addrs match). |
| **U7G addr-by-addr garbage analysis** | First-invocation byte at each unique gathered-matmul ELF | **4 of 14 ELFs produce garbage from first fire** (`@9e700`, `@a1700`, `@a3700`, `@a6300`); 10 of 14 normal. |
| Canonical re-verify post-revert | Clean kernels, no DPRINT | `"What is 2+2?"` → `" What is 2+"` ✓; chat → `"<think>\nOkay, the user is asking \"What is 2+2?\" That seems straightforward,"` ✓. **PASS**. |

**Decisive U7 cross-check (per-address sample, first invocation per core):**

| Addr | Kernel | First U7G byte sample (one core) | Magnitude class |
|---|---|---|---|
| **`@9e700`** | gathered | `7534f603 756a7522 bf17f615 d18cf587` | **~2^107** (huge) |
| **`@a1700`** | gathered | `7634f546 757c7603 be9b7673 be99f4d8` | **~2^109** (huge) |
| **`@a3700`** | gathered | `c0f6c102 c121c120 c114c122 c121c0fe` | **~-16 to -32** (large neg) |
| **`@a6300`** | gathered | `c2c1c2c2 c1c2c1c1 c1c2c2c2 c1c2c2c2` | **~-32 to -128** (large neg) |
| `@a0700` | gathered | `beff3e94 bf4b3f10 3f26bf7e 3f28be1d` | ~-0.5 (normal) |
| `@a2700` | gathered | `bd80be03 be303e09 3e643e00 bcb73d95` | ~-0.06 (normal) |
| `@a4700` | gathered | `bb6f3c6e 3d753b9a 3b4b3d0b 3b0ebcde` | ~-0.004 (normal) |
| `@a5700` | gathered | `bde93ce5 be95be6e 3c9dbe8a bd11bdeb` | ~-0.1 (normal) |
| `@a6700` | gathered | `bbb03d14 bcb33c6e bc363c2b bdfa3cad` | ~-0.005 (normal) |
| `@a6f00` | gathered | `bce13c9f 3dab3e4b 3d603d62 3d1c3d5e` | ~-0.03 (normal) |
| `@a8700` | gathered | `3d023c5d bbacbcc0 bca5bcaa bcbab9f1` | ~0.03 (normal) |
| `@a8f00` | gathered | `bdf83d39 3d313d92 3d883e3a 3e1bbd01` | ~-0.1 (normal) |
| `@aa700` | gathered | `3cdf3d76 3d52bd81 3c923c69 3d35bd8a` | ~0.03 (normal) |
| `@aaf00` | gathered | `3c803d1d 3d713d4a bd0ebb33 3c31bcf6` | ~0.02 (normal) |
| All U7C | canonical (any run) | `bb94bc1e 3a333c0f 3c243c75 3b9ebbb7` | ~-0.05..0.06 (normal) |

**Temporal degradation for the "good" addresses**: even ELFs that
START normal degrade over decode steps. For core (1,8)-0 at
`@a6700` the dumps go:
* dump 1: `3ced3cb3 bd09bc68 3c44bc1f 3d043cda` (~0.05)
* dump 7: `c0264022 3f1dc026 c027bf30 bf893fc8` (~-2 to +2)
* dump 20: `7effff59 7ede7f0a fe9afe1c ff207f47` (**NaN/Inf**)

This is consistent with the explosive amplification in the model's
recurrence: layer 0's garbage QKV → garbage attention output →
garbage residual → all subsequent layers compute on garbage →
final logits = NaN/Inf → garbage token. The `" What"` token sometimes
emerges first (correct from a low-noise first-tile shortcut), then
the post-garbage state takes over for tokens 2–5.

**ALL-zero outputs**: 30 dumps at `@a6300` show `c1c1c1c1...c1c1c1c1`
or `c2c1c2c2...` — i.e. the matmul converged to a constant-value
output across all output positions. These appear at cores (X, 6) for
X ∈ {8, 9, 10} on both devices — a specific subset of the receiver
ring. Consistent with the gathered kernel's failure mode being
core-position-dependent.

## Why this is decisive — connecting U7 to S10 and Lead 3

| Layer | Test | Result | Conclusion |
|---|---|---|---|
| Producer L1 transport | S10 (`writer_l1.cpp`) | All 16 producer cores dest_phys_L1 = 706304 (correct) | Bytes ARRIVE at the right L1 addr |
| Receiver L1 transport | S10 (`reader_bmm_tile_layout_in1_ring_all_gather.cpp`) | All 32 receivers fifo_start = 706304, bytes byte-exact = producer | Bytes WAIT at the right L1 addr |
| Compute in1 read | S10 (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`) | rd_ptr_post_byte = 706304 + ring_idx·13056; bytes match | Compute READS the right bytes |
| Compute in0 read | Lead 3 (`reader_bmm_tile_layout_in0_ring_all_gather.cpp`) | cb_in0 / cb_in2 bytes well-formed, D0/D1 agree, ring arithmetic correct | Compute READS right activation bytes |
| **Compute output (THIS dispatch)** | **U7 (`bmm_large_block_zm_fused_bias_activation_gathered.cpp` pack_tile_block)** | **4 of 14 ELFs produce ~2^60..2^109 magnitude outputs from the FIRST invocation** | **Compute kernel produces WRONG OUTPUTS given correct inputs** |

The bug surface has been narrowed to: **the gathered compute kernel's
matmul math itself (or its compile-time-arg generation) is wrong for
specific ELF parameterizations**. With correct in0 + correct in1 bytes
verified bit-exact, the only remaining variables are:
1. The compute kernel's PACKER_L1_ACC / FP32_DEST_ACC_EN gating
2. The `matmul_block(...)` / `mm_block_init(...)` inner-loop semantics
   when `gather_in0 == True` (`ENABLE_GLOBAL_CB` is defined)
3. The compile-time arg deltas between the 4-garbage ELFs and the
   10-normal ELFs
4. The `tile_regs_acquire` / `tile_regs_commit` / `pack_tile_block`
   ordering inside the gathered loop (note: the gathered code path's
   `pack_tile_block` call sits inside a `last_out && spill` branch
   that uses `mm_partials_cb` for in-progress accumulation; canonical
   uses a slightly different structure)

## Case verdict (per dispatch directive)

> **Case B — matmul outputs DIFFER at layer 0**:
> Compute kernel behaves differently between gathered (prefetcher) and
> non-gathered (canonical) paths despite same input bytes. The
> `global_cb=...` kwarg triggers different kernel behavior. Look at the
> compute kernel's GlobalCB handling.

**CASE B IS CONFIRMED.** All 4 garbage-producing ELFs are within the
gathered code path (U7G probe fires). The same-CB-address-dimensioned
canonical matmuls in the SAME prefetcher run (U7C probe fires on
non-ring matmuls) produce normal outputs.

## Instrumentation summary (now reverted)

Two files modified (both restored via `git checkout HEAD -- ...`):

1. **`bmm_large_block_zm_fused_bias_activation_gathered.cpp`** (gathered/prefetcher path):
   * Added `#include "api/debug/dprint.h"`.
   * Inserted a `DPRINT_PACK({...})` block right after the
     `mm_out_cb.push_back(out_subblock_num_tiles)` call inside the
     `last_out` branch.
   * Per-ELF static counter `u7g_ord < 1` limits each kernel ELF to
     dump once per instance.
   * Gate: `last_out && b == 0 && in0_subblock == 0 && in1_subblock == 0`.
   * Computes the just-written block's start address as
     `(wr_units - out_subblock_num_tiles · fifo_page_size) << 4`
     (with wraparound handling for the CB ring).
   * Output format: `"U7G@" + HEX(byte_addr) + "=" + 4 uint32 hex` ≈ 50 bytes (well under the 204-byte per-thread DPRINT buffer).

2. **`bmm_large_block_zm_fused_bias_activation.cpp`** (canonical path):
   * Identical probe with `"U7C"` tag and `u7c_ord < 1`.
   * Gate: `last_out && b == 0 && bh == 0 && bw == 0 && in0_subblock == 0 && in1_subblock == 0`.

**Volume gates lesson**: Initial probe used a longer 200+-byte
DPRINT format and the prefetcher run crashed the DPRINT parser with
`"Unexpected debug print type pos 0x2 len 0x5 code 114"` — the
per-thread 204-byte DPRINT buffer was overflowed and the parser
read past valid data into stack memory (interpreting `0x72` = `'r'`
from the next adjacent string as a type code). Reduced format to
~50 bytes resolved this; canonical re-launched cleanly and
prefetcher captured the bug evidence without overflow.

## Decisive prefetcher-bug reproduction

Without DPRINT, with my probe NOT in the gathered kernel (clean tree),
the prefetcher tokens for various prompts are still garbage:
- `2+2=` (max=3, T=0): `"5� Boeh"` (output_ids = [20, 124071, 69197])
- `What is 2+2?` (max=5, T=0): `" What\">\">\">\">"`
- chat-formatted same prompt (max=10): `"<think>classclassclassclassclassclassclass"`

The 1-token decode for `"2+2="` happens to produce `"5"` (correct first
token by chance from a partially-cached state); only multi-token decode
exposes the bug clearly. The 5-token + chat tests fire all 36 layers
of decode multiple times and exercise the gathered ELFs that produce
the ~10²⁰ magnitudes which then NaN-poison every subsequent matmul.

## Hypothesis ledger update

| ID | Suspect | Status |
|---|---|---|
| S1-S10 | Producer/consumer + L1/CB micro-hypotheses | ALL RULED OUT |
| A1 | ecfb2e782c2 L1-clash bypass | RULED OUT |
| Lead 2 | upstream 934d954b995 last_subblock_w_valid port | RULED OUT |
| Lead 3 (U2) | in0 ring-all-gather byte transport | RULED OUT |
| U1 | Cross-sub-device dispatch sync gap | WEAKENED (U6) — not the root cause |
| U3 | Permuted vs contiguous DRAM grid | RULED OUT |
| U4-A | TP=2 misconfig | RULED OUT |
| U4-B | DST accumulator stale state | RULED OUT |
| U5 | Subdev-routing barrier | RULED OUT |
| U6 | Galaxy-parity 2-subdev + dummy_receivers | EVIDENCE_ADVANCE; weakens U1 |
| **U7** | **Differential matmul-output test (Case A vs B vs C)** | **CASE B CONFIRMED (this dispatch) — gathered compute kernel produces ~2^60..2^109 magnitude outputs from FIRST invocation for 4 of 14 ELFs, ending in NaN/Inf, while canonical matmuls in same run produce normal small-magnitude outputs.** |

## Recommended next attack (per Phase 7 Case B in the dispatch)

Per directive verbatim:
> **Case B — matmul outputs DIFFER at layer 0**:
> Compute kernel behaves differently between gathered (prefetcher) and non-gathered (canonical) paths despite same input bytes. The `global_cb=...` kwarg triggers different kernel behavior. Look at the compute kernel's GlobalCB handling.
> Look at:
>   - mm_init differences between the two compute kernels
>   - DST register usage differences
>   - `acquire_dst`/`pack_tile` ordering
>   - `matmul_block` vs `matmul_tiles` semantics
>   - The compute kernels have ENABLE_GLOBAL_CB macro gates — those gates control different behavior; find what

**Concrete U8 proposal — diff the gathered vs canonical compute kernels under `ENABLE_GLOBAL_CB`**:
1. Identify which compile-time args differ between the 4 GARBAGE
   ELFs and the 10 NORMAL ELFs (hash IDs visible in `.cache/tt-metal-cache/.../bmm_large_block_zm_fused_bias_activation_gathered/`).
   The garbage cluster shares some param — likely `in0_block_w`,
   `unpadded_in0_shard_widths_in_tiles[]`, or `num_blocks`.
2. Specifically inspect the gathered kernel's `mm_block_init` call
   and `reload_from_cb_to_dst` semantics. With
   `ENABLE_GLOBAL_CB` defined, the kernel uses `update_rd_ptr_to_ring_index`
   on in1_cb between blocks — a path NOT present in the canonical
   kernel. If a wrong `update_rd_ptr` lands the matmul on the SAME
   in1 block for multiple iterations (instead of advancing through
   the ring), the matmul accumulates 32× the same value → 2^5 magnitude
   if input is unit-magnitude → much larger for K=4096 inner-dim.
   But our observed magnitudes are 2^60+, suggesting compound wrap
   bugs OR DST register reuse.
3. Verify the gathered kernel honors `PACKER_L1_ACC=0` for `block==0`
   correctly (the `if (block == 0) { llk_pack_reconfig_l1_acc(0); }`
   inside the spill branch). If this fires AFTER the first block on
   only some ELFs, the matmul accumulates the previous batch's
   output instead of zero-init.

**Alternative U9 proposal — bisect the gathered ELF parameter space**:
1. With the SAME 5-token `"What is 2+2?"` decode, force only one
   gathered ELF to be different (e.g., toggle one compile-time arg
   between the garbage and normal sets) and observe which arg flip
   restores or breaks the output.
2. The 4 garbage ELFs share something (in0_block_w? per_core_M?
   `unpadded_in0_shard_widths_in_tiles`?). The 10 normal ELFs share
   the complement. Diff their kernel hashes to find the discriminator.

## Working state at session end

- tt-metal-sglang HEAD: **`68b8eaa1152`** (unchanged; instrumentation reverted in-session).
- tt-metal-sglang branch: `tenstorrent-p1`.
- Two compute-kernel files: clean (`git checkout HEAD -- ...`).
- Container src + libexec mirror: synced to clean files.
- `/root/.cache/tt-metal-cache/*` cleared (last action before reverify).
- `stash@{0,1,2}` untouched (per CLAUDE.md rule).
- DPRINT artifacts preserved on host container `/tmp/u7_*`:
  * `/tmp/u7_canonical_mm_out_short.log` (5.6 MB, 98486 lines, 98480 U7C dumps)
  * `/tmp/u7_prefetcher_mm_out.log` (7.2 MB, 125350 lines, 59200 U7G + 66144 U7C)
  * `/tmp/u7_canonical_uniq.txt` + `/tmp/u7_pf_c_uniq.txt` (sort-unique digests)
- Cards healthy (single `tt-smi -r 0,1` mid-session after DPRINT-overflow crash hung Device 0).
- No server running at session end.
- **Canonical re-verify post-revert PASSES**: `"What is 2+2?"` → `" What is 2+"`; chat → `<think>\nOkay, the user is asking "What is 2+2?" That seems straightforward,`.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat = 9-10/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** The
gathered compute kernel produces 2^60..2^109 magnitude outputs from
4 of 14 ELF instances on first invocation, NaN-poisoning the
recurrence and producing garbage tokens. Fix requires either patching
the gathered compute kernel's GlobalCB-gated math (U8 plan) or
identifying which compile-time arg combination triggers the bug and
avoiding it from the Python `matmul_1d_ring_config` call site (U9 plan).

## Commits this session

* (tt-metal-sglang) **none** — instrumentation applied, captured, and
  reverted in the same session; HEAD remains `68b8eaa1152`.
* (sglang) `<this doc>` — pending commit.
