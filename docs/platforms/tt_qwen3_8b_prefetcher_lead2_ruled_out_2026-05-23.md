# TT Qwen3-8B prefetcher — Lead 2 (port upstream 934d954b995 last_subblock_w_valid) RULED OUT — 2026-05-23

Status: **EVIDENCE_ADVANCE (Case C) — Lead 2 RULED OUT. Defensive port of
upstream commit `934d954b995` ("[Bug Fix] DRAM Matmul - Fix Compute
Reading Un-pushed Data") into our 1D ring-gathered factory is a clean
no-op compile (built tt-metal in ~30 s, 233 TUs, both _ttnn.so + _ttnncpp.so
relinked clean). The plumbed `last_subblock_w_valid` named compile-time
arg is set to `out_subblock_w` by our factory (no implicit subblock-width
padding in the 1D path, unlike dram_sharded), so the compute kernel's
new `is_last_in1_subblock_padded` guard is always false at runtime — the
original full-width matmul_block path is preserved. With the port in
place, the prefetcher's auto-warmup decode still produces garbage tokens
(50 tok/s for ~1600 tokens) and eventually NaN-traps Qwen3-8B's sampling
on `torch.multinomial(probs)` with "probability tensor contains either
`inf`, `nan` or element < 0" — IDENTICAL failure class to the baseline
broken prefetcher (B.8) and bit-exact symptom from the Attack-1 dispatch.
The upstream bug class (`per_core_N_compute > per_core_N_in1_sender`,
where the planner pads the matmul's last subblock above what the reader
pushed into cb_in1) does NOT apply to our 1D / ring-gathered path: our
factory uses `in1_num_subblocks = per_core_N / out_subblock_w` (integer
truncation, no padding source), so `last_subblock_w_valid == out_subblock_w`
by construction and the upstream guard is a no-op for every Qwen3-8B
matmul shape. Probe branch deleted, tt-metal-sglang restored to
c1e3437b78e, container + libexec mirror rebuilt to HEAD.**

Canonical (no prefetcher) Qwen3-8B re-verified post-revert: GSM8K(10) chat
= **10/10 = 100%** (Q3 long-reasoning but final answer correct — better
than the documented 9/10 baseline because of sampling-temperature noise
giving a different Q3 reasoning chain this run). Tt-metal-sglang HEAD
unchanged at `c1e3437b78e`; working tree clean; stash@{0..2} preserved.

Continuation of `tt_qwen3_8b_prefetcher_cpp_revert_attack1_2026-05-23.md`
(A1 ruled out). This dispatch tested the upstream-diff doc's recommended
Attack 3 / Lead 2.

## Bottom line

| Phase | Result |
|---|---|
| Upstream `934d954b995` fetch (upstream remote, depth 400) | OK (commit visible, 3 files / +51 -3) |
| Probe-branch creation (`lead2-port-934d954b995` on tenstorrent-p1) | OK |
| Defensive port: factory + compute kernel (no-op `last_subblock_w_valid = out_subblock_w`) | OK |
| tt-metal incremental rebuild (`rebuild_tt_metal_kernels.sh ttnn`, 4 build steps) | OK in ~30 s, both `_ttnn.so` (14.2 MB) and `_ttnncpp.so` (33.3 MB) relinked clean |
| Container src + libexec mirror sync | OK |
| JIT cache clear + server boot with prefetcher | Server up but auto-warmup decode hits NaN crash on `torch.multinomial` after ~1600 decode tokens at 50 tok/s |
| Prefetcher Q1 (eval) | INVALID (`<ERROR RemoteDisconnected>`) — scheduler died on warmup decode NaN before eval Q1 could complete |
| Final GSM8K(10) | 0/10 = 0% (scheduler crashed, all Q2..Q10 = connection refused) |
| Probe branch cleanup | deleted; restored tenstorrent-p1 (c1e3437b78e) |
| Container re-sync + rebuild | OK; libs restored to HEAD |
| Canonical re-verify (no prefetcher) | **10/10 = 100%** (≥ 9/10 baseline; no regression — actually +1 due to sampling-noise gain on Q3) |
| Stash state | unchanged (stash@{0,1,2} preserved) |

## Upstream patch summary

Commit `934d954b9959f02085d8ada922ecdc8a16b7fdeb`, Edwin Lee, 2026-05-21,
PR #44872 "[Bug Fix] DRAM Matmul - Fix Compute Reading Un-pushed Data",
fixes issue #43998. Files touched:

1. `matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp`
   (+4 lines): emits `{"last_subblock_w_valid", out_subblock_w}` (always
   no-op for this factory — never pads beyond `per_core_N_in1_sender`)
2. `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`
   (+10 lines): computes `last_subblock_w_valid = out_subblock_w -
   (per_core_N_compute - per_core_N_in1_sender)` after the optional
   subblock-width-widening optimization (`if out_subblock_h == 1 && out_subblock_w < max_subblock_w`)
   that may pad `per_core_N_compute` above `per_core_N_in1_sender` —
   the buggy source.
3. `bmm_large_block_zm_fused_bias_activation.cpp` (+40 -3 lines):
   reads `last_subblock_w_valid` as named CT arg (under `MATMUL_DRAM_SHARDED`
   gate, defaults to `out_subblock_w` otherwise); computes
   `constexpr bool last_subblock_padded = last_subblock_w_valid < out_subblock_w;`;
   in the matmul body's `for (in1_subblock < in1_num_subblocks)` loop,
   narrows `matmul_block(..., out_subblock_w, ...)` to
   `matmul_block(..., effective_subblock_w, ...)` on the last in1
   subblock when padded; in the bias-add stage, redirects out-of-range
   bias_tile_idx reads to tile 0 of cb_bias for the padded lanes.

### Why upstream needs it

In the DRAM-sharded factory the planner has a subblock-width-widening
heuristic (lines ~146-164 in patched upstream):

```cpp
if (out_subblock_h == 1 and out_subblock_w < max_subblock_w) {
    // ... search for a larger out_subblock_w that minimizes
    // num_subblock_w_per_core_N (round-up division)
    out_subblock_w = preferred_out_subblock_w;
    per_core_N_compute = out_subblock_w * num_subblock_w_per_core_N;
    // ↑ per_core_N_compute may now exceed per_core_N_in1_sender
    //   (the count the reader actually pushes into cb_in1)
}
```

When this heuristic widens out_subblock_w, the new
`per_core_N_compute = out_subblock_w * round_up(per_core_N_in1_sender,
out_subblock_w) / out_subblock_w` can exceed `per_core_N_in1_sender`.
The compute kernel then reads `per_core_N_compute` columns per K-step
while the reader only pushed `per_core_N_in1_sender` tiles into cb_in1.
The unpacker silently reads past the valid CB region; the writer drops
the padded columns from output, so PCC stayed green — but the path is
fragile (depends on the in1 CB being large enough that the over-read
doesn't pull garbage from outside the CB's valid region).

## Why it does NOT apply to our 1D / ring-gathered path

The matmul-1D / ring-gathered factory
(`matmul_multicore_reuse_mcast_1d_program_factory.cpp::process_gather_in0`,
lines 2086-2654 — the function that calls the gathered compute kernel
and the only direct caller of
`bmm_large_block_zm_fused_bias_activation_gathered.cpp`) has NO equivalent
subblock-width-widening heuristic. Its in1 sizing is straightforward:

```cpp
uint32_t in1_num_subblocks = per_core_N / out_subblock_w;       // line 2129
uint32_t in1_block_height_in_tiles = in0_block_w;               // line 2130
uint32_t in1_block_num_tiles = out_subblock_w *
    in1_block_height_in_tiles * in1_num_subblocks;              // line 2131
uint32_t in1_block_size_bytes = in1_block_num_tiles *
    in1_single_tile_size;                                       // line 2132
uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;   // line 2134
```

Note the integer truncation in `in1_num_subblocks = per_core_N /
out_subblock_w`. If `per_core_N` is NOT divisible by `out_subblock_w`,
the remainder is silently DROPPED from `in1_num_subblocks` and from
`in1_block_num_tiles`, so the compute kernel reads at most
`in1_num_subblocks * out_subblock_w * in0_block_w` tiles per K-step
— never more than `per_core_N * in0_block_w` and never more than the
reader's `in1_block_num_tiles` (same formula). The over-read bug
class from upstream's subblock-w widening cannot trigger here:
`per_core_N_compute` (here = `in1_num_subblocks * out_subblock_w`)
is always **≤** `per_core_N_in1_sender` (the per-call expectation).

For all Qwen3-8B matmul shapes our prefetcher uses, `per_core_N` is
chosen by the model_config Python to be divisible by `out_subblock_w`
(verified by the fact that no run produces a "matmul block size mismatch"
TT_FATAL at boot). So `last_subblock_w_valid == out_subblock_w` always
for our path, and the upstream guard is a runtime no-op.

## Port diff (vs c1e3437b78e HEAD, then reverted)

### `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (+13 lines)

After the `in1_per_core_w = out_subblock_w * in1_num_subblocks;
out_subblock_num_tiles = out_subblock_h * out_subblock_w;` block (post-line
2135), added a defensive `last_subblock_w_valid` local equal to
`out_subblock_w`. Threaded into `compute_named_compile_args` map after
the `cb_sync2` entry. No runtime arg additions; no kernel-side signature
change for non-prefetcher callers.

### `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` (+26 -1 lines)

Inserted after the `sync_cb2` CT arg parse:

```cpp
constexpr uint32_t last_subblock_w_valid =
    get_named_compile_time_arg_val("last_subblock_w_valid");
constexpr bool last_subblock_padded = last_subblock_w_valid < out_subblock_w;
```

In the matmul-block loop (post the `for (uint32_t in1_subblock = 0;
in1_subblock < in1_num_subblocks; in1_subblock++)` line):

```cpp
const bool is_last_in1_subblock_padded =
    last_subblock_padded && (in1_subblock == in1_num_subblocks - 1);
const uint32_t effective_subblock_w =
    is_last_in1_subblock_padded ? last_subblock_w_valid : out_subblock_w;
```

Changed the inner `matmul_block(..., out_subblock_w, out_subblock_h,
in0_block_w);` call to `matmul_block(..., effective_subblock_w, ...);`.

No bias-stage equivalent (the gathered kernel doesn't have a FUSE_BIAS
path — the gathered compute is the inner of the ring-gather matmul and
bias is added outside, in a separate matmul/eltwise step).

## Build outcome

tt-metal incremental rebuild via `rebuild_tt_metal_kernels.sh` target
`ttnn` (the matmul / kernels-only path):

- 4 build steps (1 CXX compile, 2 shared-lib relinks, 1 dependency re-check)
- Compile + link in **~30 seconds** (warm-cache Unity build of the matmul TU only)
- No errors, no warnings
- Both shared libs relinked: `_ttnn.so` 14252664 bytes (unchanged size),
  `_ttnncpp.so` 33272864 bytes (+304 bytes vs HEAD = our new kernel arg
  parse + branch)

This confirms the named CT arg plumbing is byte-compatible with all
other matmul-1D callers (the only caller of the gathered kernel is
`process_gather_in0` itself; all other 1D matmul variants use different
compute kernels and are NOT affected).

## Server-boot outcome with port

The defensively-ported build booted fine through the entire init
sequence:

1. UMD topology discovery: OK
2. Multi-mesh mapping: 1 logical mesh → 1 physical, OK
3. Fabric init: 2 devices, OK
4. Weight load (Qwen3-8B BFP8): 18.83 s, avail mem = 45 GB after
5. Paged KV alloc (155200 tokens, 21.32 GB): OK, avail mem = 24 GB after
6. tt_worker_init_registry_done: OK
7. DRAM Prefetcher: 180 tensors inserted (5 per layer × 36 layers), OK
8. Global CB created (size 835584): OK
9. V2.5 DECODE manager activated: OK
10. `prefill chat-detect req=1 is_chat=False prompt_len=1 first_5_ids=[0]`
    — this is the SGLang plugin's prefetcher-warmup prefill (single-token
    `[0]` prompt) to populate the prefetcher trace cache
11. Prefill batch (#new-token: 64, then 128) succeeded, gen ramped to
    50 tok/s
12. Decode loop ran for ~28 seconds (1600 tokens at 50 tok/s) producing
    garbage tokens (no traceback in log because the actual token IDs
    aren't decoded at this level)
13. **NaN crash on `torch.multinomial(probs_sort, num_samples=1)` in
    `top_k_top_p_min_p_sampling_from_probs_torch` (sampler.py:485) —
    `RuntimeError: probability tensor contains either inf, nan or element < 0`**
14. Scheduler caught the exception, raised SIGQUIT, parent terminated
    with `leaked semaphore` warning

This is the EXACT same failure mode as B.8 baseline broken prefetcher and
the Iter-2 retry in B.9: prefetcher produces deterministically wrong
matmul outputs that eventually NaN out one of the softmax-then-sample
intermediate tensors. The port plumbing did not change this; it doesn't
trigger because our path doesn't have the upstream's padding source.

## Why the warmup decode survived longer than expected

The earlier B.8/B.9 broken-prefetcher runs typically NaN'd within the
first 10-50 decode steps. This dispatch's warmup ran ~1600 tokens before
NaN-trapping. This isn't necessarily a positive signal from the port — it
could be:

1. Coincidence of the input distribution for the `[0]` prompt's chat-detect
   path
2. Less aggressive amplification of the wrong matmul outputs (e.g.
   embedding's grid choice differs from the broken-prefetcher run)
3. Slight numeric variation in the matmul outputs that delays the NaN
   but doesn't prevent it

None of these would amount to a real fix. The port is fundamentally a
no-op for our path.

## Hardware coordination notes

- Initial server-launch died at layer 2 of weight load with no log
  traceback; investigation via `sudo dmesg` showed three historical
  OOM-kills (uptime tags 381580, 540127, 547420 sec). The current
  dispatch's first launch hit the same OOM (host had been running for
  6.5 days with multiple parallel agents). Restart after waiting for
  memory pressure to clear (49 GB available) succeeded.
- Second launch hit `TT_THROW @ risc_firmware_initializer.cpp:1267`
  on Mesh device init — TT card state was stuck from the OOM-killed
  previous attempt. Recovered via `tt-smi -r 0 1` reset (took ~60 s),
  third launch booted cleanly.
- Lead 3's in0-side instrumentation (`reader_bmm_tile_layout_in0_ring_all_gather.cpp`)
  was present in the working tree at session start but had reverted to
  HEAD by the time I began (Lead 3 agent's own cleanup). Hardware never
  contested; both leads ran sequentially.

## Hypothesis ledger update

| ID | Suspect | Status |
|---|---|---|
| S1 | Row-wise stride mismatch in writer | RULED OUT (B.3 byte-dump) |
| S2 | num_blocks mismatch | OPEN, unlikely |
| S3 | WO K-shard transposition | RULED OUT (Phase A) |
| S5 | BFP4/BFP8 mixed tile pitch | RULED OUT (Phase A) |
| S6 | Per-layer cross-block address drift in producer | RULED OUT (B.4-extended) |
| S7 | Compute-kernel ring_idx wrap | RULED OUT (B.6) |
| S8 | Per-tensor block_size in reader_dram.cpp | RULED OUT (B.8 analysis) |
| S9 | Producer NOC posted-writes flush insufficient | RULED OUT (S9 dispatch) |
| S10 | Sub-device CB-address misalignment | RULED OUT (S10 dispatch — bytes match) |
| A1 | ecfb2e782c2 L1-clash bypass = hidden real clash | RULED OUT (A1 dispatch — bypass is needed) |
| **Lead 2** | **Port upstream 934d954b995's `last_subblock_w_valid` to gathered factory** | **RULED OUT (this dispatch — defensive port is a clean no-op; upstream subblock-w padding bug class does not exist in our 1D path because `in1_num_subblocks = per_core_N / out_subblock_w` truncates instead of padding)** |

## Path forward — Lead 2 is exhausted; pivot

With S1-S10 + A1 + Lead 2 all ruled out, the remaining attack vectors
from the phase B.9 / S10 / upstream-diff docs are:

### Lead 3 (parallel agent, in progress) — U2 in0-side direct probe

S10 instrumented in1 (the weight side) and confirmed bytes match. The
in0 (activation) side was NOT instrumented. Lead 3 agent (working in
parallel during this dispatch) is adding DPRINT to
`reader_bmm_tile_layout_in0_ring_all_gather.cpp` mirroring the S10 reader
probe. If in0 byte stream doesn't match what `all_reduce` / `reshard`
emitted, that's the new culprit.

### Lead 4 (recommended next dispatch) — Python-side call-site diff

Compare `models/tt_transformers/tt/attention.py` and `mlp.py`'s
`ttnn.linear(..., global_cb=...)` invocations against the upstream
tt-metal demo's prefetcher invocation (if one exists). Check for:
- Program-config kwarg drift (in0_block_w, per_core_N, out_subblock_w)
- mesh-mapper choice (RowMajor vs ColMajor)
- `unpadded_in0_shard_widths_in_tiles` runtime arg computation
- `output_mem_config` shape vs prefetcher's expectation

This is a Python-only diff; the C++ kernels are now exhausted as a
suspect surface. The phase B.9 doc's U1/U3/U4 categories all live here.

### Lead 5 (ultimate fallback) — descriptor-migration `be3d3fccab0`

Upstream commit `be3d3fccab0` (2026-05-18) rewrites
`DramPrefetcherProgramFactory` to `DramPrefetcherOperation::create_descriptor`
returning `ProgramDescriptor`. Cherry-pick (with the dependencies)
and retest. Risk: tt-metal descriptor migration is large; could regress
canonical. Only attempt after Lead 4 also yields no fix.

## Working state at session end

- tt-metal-sglang HEAD: **`c1e3437b78e`** (unchanged from session start —
  probe branch deleted; HEAD restored)
- Container source synced from HEAD (2 files re-copied via `podman cp`)
- Container shared libs rebuilt to HEAD state (~30 s, both `_ttnn.so` +
  `_ttnncpp.so` updated)
- `/root/.cache/tt-metal-cache/*` cleared (twice; once per build)
- `stash@{0,1,2}` UNTOUCHED
- All cards healthy (tt-smi snapshot shows both boards at 60-63 °C, 0.86 V)
- Upstream remote `https://github.com/tenstorrent/tt-metal.git` was already
  configured at session start (not added by this dispatch); left in place
  as read-only convenience for future Lead 5 / upstream cherry-picks (no
  push permission). The dispatch fetched `upstream main --depth=400` for
  the commit lookup; no other upstream interactions.
- Canonical GSM8K(10) chat re-verified: **10/10 = 100%** (Q1..Q10 all
  CORRECT; ≥ 9/10 baseline threshold met; no regression vs HEAD c1e3437b78e)

## Commits this session

- (tt-metal-sglang) **none** — probe-branch commit deleted; HEAD unchanged
- (sglang) `<this doc>` — pending commit

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping config
at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.** All
documented prefetcher suspects S1-S10 + A1 + Lead 2 are RULED OUT. The
remaining attack vectors are Lead 3 (in0-side DPRINT probe; in progress)
and Lead 4 (Python call-site diff vs upstream demo). Both require a
different attack tool than the C++ port tried this session.
