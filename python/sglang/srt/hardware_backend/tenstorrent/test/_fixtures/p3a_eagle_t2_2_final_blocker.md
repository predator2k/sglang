# P3a.2 T2.2 EAGLE — Final State (2026-05-14)

## Progress (all committed to `predator2k/sglang` `tenstorrent-p1`)

- 809fc7f19 — ETH+ROW+MUX dispatch override + TT_DISPATCH_PROBE
- 9089edea0 — bootstrap_container.sh (initial)
- c29593e51 — bootstrap_container.sh — privileged + uvloop + --device note
- a1ba960f3 — force page_size=64 + attention_backend=torch_native defaults
- 4a5b3ecc4 — CPU paged-KV allocator (no Triton dependency)
- 823adbb7a — launch_eagle.sh

## What now works on this branch

EAGLE 2-model cohost (Qwen3-8B target + Qwen3-1.7B draft) **boots all the way
to Uvicorn running on http://0.0.0.0:30000** with cohost mesh sharing.
See `p3a_eagle_t2_2_post_dispatch_v3.log` ... `v9_full_patch.log` for the
boot logs across iterations.

## Remaining structural blocker

Every boot attempt (v3, v4, v6, v7, v8, v9) crashes at the FIRST inference
request OR during prefill trace capture with:

  TT_FATAL: Circular buffer core range [0-0 - 7-9] in program 70 exceeds
  device compute grid (12x9)

A matmul kernel asks for an 8 cols × 10 rows core grid; P300 with MUX
dispatch exposes 12 cols × 9 rows. The 10th row is reserved for ETH
dispatch traffic — there is no kernel-time slot for it.

## What was tried and failed

1. Patched `model_config.py:get_attn_qkv_program_config` lines 1596+1600
   `(8, 10) if is_blackhole() else (8, 8)` → `(8, 8) if is_blackhole() else (8, 8)`.
2. Patched `attention_1d.py:1624` `(8, 10)` → `(8, 8)`.
3. Cleared `/root/.cache/tt-metal-model-cache/P300/` + all .pyc.
4. Tried `TT_METAL_OPTIMIZATIONS=accuracy`.
5. Tried `--skip-server-warmup`.
6. Tried `--speculative-num-draft-tokens 1 --speculative-num-steps 1`.

None of those reach the failing kernel. The crashing grid request is built
in tt-metal C++ at `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp`
which queries `device->compute_with_storage_grid_size()` and picks an
8×10 grid. That C++ default isn't reachable from a Python-level patch.

## Path forward (out of scope for this session)

A. Patch `matmul_program_config.cpp` to clamp grid.y to ≤ 9 on Blackhole
   under MUX dispatch, then rebuild `localhost/local-tt-metal:dev`. ~hours
   of build time.

B. Drop MUX dispatch entirely. ROW dispatch needs MUX on Blackhole, so this
   means going back to WORKER+COL — which re-introduces the original iter-8
   bmm kernel-placement crash.

C. Add proper `model_params/Qwen3-8B/` config in tt-metal fork (custom
   matmul configs that don't fall through to the 8×10 default). Days of
   tuning work per-model.

D. Wait for tt-metal upstream to add tuned Qwen3 + P300 support.

## Single-model paged status

Same root cause — every paged Qwen3 (or Llama-3.1-8B) launch hits the same
matmul 8×10 wall. Single-model paged is currently broken on this branch
on this hardware until one of A/B/C/D above is taken.

## tt-metal C++ rebuild attempted (v13)

Patched `tt_metal/impl/device/device.cpp::compute_with_storage_grid_size()`
to clamp `grid.y` to ≤8 (forcing the device-reported grid to be 11×8 on
P300/MUX so any auto-config reading the device grid never picks the
dispatch row). Rebuilt `libtt_metal.so` and `_ttnncpp.so` via ninja
inside the container; replaced the loaded .so files. Probe confirmed
device now reports 11×8.

Result: v13 boots cleanly to `Uvicorn running` but the first inference
request STILL fails with the same TT_FATAL — `Circular buffer core range
[0-0 - 7-9] in program 70 exceeds device compute grid (12x8)`.

Python-side trace confirms the failing kernel is the **LM head's
DRAM-sharded matmul** at `lm_head.py:158`, not the QKV / SDPA / MLP path
I patched earlier. DRAM-sharded matmul derives its CoreRangeSet from
the input tensor's DRAM-sharding layout (8 columns × N rows where N is
chosen by tt-metal C++ to evenly divide the vocab dim), not from
`compute_with_storage_grid_size`. So the device-grid clamp does not
reach this code path.

To unblock here would require either patching
`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`
(or a sibling DRAM-sharded factory) to clamp the row count, or providing
a tuned `dram_matmul_config` in `tt_transformers/tt/model_config.py` for
Qwen3 on P300 that hardcodes a fitting grid. Both are non-trivial.

## Proven catch-22 (v10 evidence)

Tested `SGLANG_TT_DISPATCH=legacy` (WORKER+COL dispatch — 10 rows
visible) WITH all the `(8, 10)` → `(8, 8)` Python-side patches still
applied. Result: the SAME `bmm_large_block_zm_fused_bias_activation`
kernel that iter-8 originally crashed on now crashes again:

  TT_FATAL: Illegal kernel placement for bmm_large_block_zm_fused_bias_activation,
  Kernels cannot be placed on dispatch cores!

This proves the two failure modes are TWO ENDS OF THE SAME CONFLICT:

  - ETH+ROW+MUX dispatch (my fix) frees worker cores from dispatch but
    leaves only 9 rows; matmul kernels asking for 10 rows fail with CB
    overflow.
  - WORKER+COL dispatch (legacy) keeps 10 rows of workers, but the bmm
    kernel's CoreGrid lands on the dispatch cores → placement crash.

Both branches hit the same tt-metal C++ matmul auto-config picking an
8×10 grid that exceeds available workers under any dispatch layout on
P300. The 8×10 default was tuned for single-chip Blackhole p150 where
all 10 rows are workers; on 2-chip P300 under MUX it doesn't fit.

Log evidence:
  - `p3a_eagle_t2_2_post_dispatch_v9_full_patch.log` — MUX path, CB-overflow crash
  - `p3a_eagle_t2_2_post_dispatch_v10_legacy.log`     — legacy path, placement crash

## Recommendation

Land the 6 commits in `predator2k/sglang` `tenstorrent-p1` and pause P3a.2
until tt-metal C++ matmul auto-config tuning is done. The Python-side work
in this session is solid — the structural wall is below it.
