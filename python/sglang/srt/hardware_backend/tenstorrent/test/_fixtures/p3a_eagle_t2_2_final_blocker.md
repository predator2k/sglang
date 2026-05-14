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

## Breakthrough: tt-metal C++ investigation revealed the real Layout-B blocker

The original "device.cpp clamp" was applied UNCONDITIONALLY to clamp
grid.y ≤ 8. That was correct for Layout-A under MUX (12x9 → 12x8) but
**broke single-chip Layout-B**, which natively has an 11x10 grid that
matmul kernels need all 10 rows of.

v27 (single-model + my unconditional clamp) crashed at the same 8x10
matmul kernel grid issue we were chasing for Layout-A's "wo aliasing".
v28 (single-model + conditional clamp — only clamp under ETH/MUX
dispatch) cleared the kernel-grid check. `/health` returned **200**
for the first time across all 29 iterations.

The "wo aliasing" hypothesis from T2.2.G was **WRONG**. The actual
cause of v3-v9 crashes was the same 8x10 kernel grid being requested
on the 12x9 grid Layout-A exposed. Once my conditional clamp gives
no-MUX single-chip the full 11x10, the issue disappears.

**v29 progress (single-model Qwen3-8B, solo chip 0):**
  - boots cleanly
  - `/health` → HTTP 200 (first time!)
  - inference enters real forward
  - prefill completes
  - decode crashes at `distributed_norm.py:108 → rmsnorm.py:154` with
    `RuntimeError: bad optional access` (C++ std::bad_optional_access)

The new blocker (T2.2.I) is a missing `std::optional` value somewhere
in `ttnn.rms_norm`'s sharded program_config path. Probably the norm's
sharded config references a multi-device-assumed field (e.g.,
`cluster.ring_size`, prefetcher fields) that's empty on single chip.

## TODO list (post-session)

### TODO-1: Layout-A wo aliasing fix (T2.2.G)

**Problem:** With shared (1,2) cohost mesh, the draft's `ttnn.as_tensor`
for the wo weight returns the target's per-device shape rather than the
draft's. Root cause not pinned: either tt-metal's DRAM-sharded buffer
allocator aliases across as_tensor calls with matching mem_config
layout, or `cache_file_name` keys collide despite different paths.

**Investigation needed:**
  - Read ttnn C++ `as_tensor` + `MemoryConfig` allocator paths to see if
    there's a buffer-handle cache keyed on layout-only.
  - Try forcing `cache_file_name=None` for the draft's wo (skip the
    on-disk cache lookup) to see if that breaks the aliasing.
  - Try `ttnn.DRAM_MEMORY_CONFIG` (interleaved, not sharded) for the
    draft's wo specifically — bypasses the DRAM-sharded code path.

**Reward:** Layout-A would have zero per-inference overhead vs the
Layout-B CPU bounce.

### TODO-2: Layout-B single-chip warmup hang (T2.2.H follow-up)

**Problem:** With `SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT=split` Layout-B
boots cleanly to Uvicorn, but the first inference hangs in the target
model's tt-metal lazy warmup (`prefill_forward_text` → tt-metal C++
opaque). py-spy can't merge native frames; we don't have the exact
C++ call site.

**Hypotheses** (ordered by probability):
  1. `trace_region_size=50_000_000` is too large for solo Blackhole's
     DRAM allocator — silently locks. Try 10_000_000 or omit.
  2. tt-metal kernel grid configs computed for `num_devices=1` are
     unsupported on Blackhole (e.g., the QKV / SDPA / WO matmul
     auto-configs that we patched for `num_devices=2` produce
     different shapes).
  3. Some warmup op (e.g., all-gather for CCL) is called
     unconditionally and waits for the multi-device fabric that
     single-chip mesh doesn't have.
  4. tt-metal's program cache / dispatch state from the FIRST mesh
     open (target on chip 0) leaks into the SECOND mesh open (draft on
     chip 1), corrupting both.

**Investigation steps:**
  - Reproduce with single-model (no EAGLE) at `mesh_shape=(1,1)`,
    physical_device_ids=[0]. If it hangs identically → it's
    single-chip warmup, not EAGLE.
  - ~~Drop trace_region_size to 10 MB, retry.~~ **TRIED in v25, still
    hangs at the same `rotary_embedding_llama` warning. Hypothesis 1
    eliminated.**
  - Add `MAX_PREFILL_CHUNK_SIZE` env clamping.
  - Try `TT_METAL_WATCHER` env to see kernel-level events leading up
    to the hang.
  - Try `--max-prefill-tokens 32` + `--chunked-prefill-size 64` to
    force tiny prefill shapes (we patched server_args.page_size=64
    earlier so this should pass validation).

**Note re: CPU bounce:** The "CPU bounce for EAGLE handoff" idea was
based on a wrong hypothesis (cross-mesh tensor read at the EAGLE
level). At the SGLang Python boundary we already convert
ttnn→torch at every model boundary, so the EAGLE protocol handoff
is implicitly CPU-mediated. The real hang is below SGLang, inside
tt-metal's warmup on a single-chip mesh.

## Layout-B split-mesh implemented (v24) — boots; inference hangs on cross-mesh

The original Layout-B "infeasible" finding (T0.3) tested mesh_shape=(1,2)
on both models — both claimed ALL chips and ETH-timed-out. Proper
Layout-B uses `mesh_shape=(1,1) + physical_device_ids=[N]` per model so
target and draft own disjoint chips.

Wired `SGLANG_TT_SPEC_DRAFT_MESH_LAYOUT=split` into
`tt_llm.TTModels.__init__`: first model opens chip 0, second opens
chip 1. Committed at `091d63b12`.

v24 result: BOTH meshes open successfully (no ETH timeout), BOTH models
load, KV alloc completes for both, **warmup completes for both, Uvicorn
comes up**. First inference request via `/v1/chat/completions` then
hangs for the full 300 s watchdog timeout. py-spy dump shows
`forward_batch_generation` (eagle_worker.py:461) stuck deep in ttnn
C++ — no Python-level exception.

**Probable cause:** EAGLE's protocol needs to move tensors between
target's mesh (chip 0) and draft's mesh (chip 1) — at minimum the
draft consumes the target's hidden_states; with split meshes there is
no ETH fabric between the two chips so the inter-mesh tensor read
blocks indefinitely.

**Path forward** (next session):
  - **CPU-bounce transfer for EAGLE handoff** — copy hidden_states /
    accept-mask through host RAM between target.mesh and draft.mesh.
    Costs a host-device roundtrip per spec step but is the only fix
    that keeps cohost without solving the Layout-A wo aliasing.
  - **Lazy clone of small per-model buffers** — patch `attention.py`
    so each model allocates its own per-mesh-isolated wo (revisit
    Layout-A with explicit unique cache_file_name+mem_config).
  - **Llama-3.1-8B + Llama-3.2-1B** when Llama-3.2-1B becomes
    available — only officially-tuned tt_transformers EAGLE pair.

## WO matmul cohost aliasing diagnosed (v20–v23) — Layout-A wall

Probed both `ttnn.as_tensor` input (pt_wo) and output (self.wo) for each
Attention instance:

  Target Qwen3-8B layer 0:
    BEFORE: pt_wo = (1, 1, 4096, 4096)
    AFTER:  self.wo.shape = (1, 1, 2048, 4096)  ← correct per-device after dim-2 shard

  Draft Qwen3-1.7B layer 0:
    BEFORE: pt_wo = (1, 1, 2048, 2048)  ← state_dict + transposes correct
    AFTER:  self.wo.shape = (1, 1, 2048, 4096)  ← WRONG, should be (1, 1, 1024, 2048)

The same `(1, 1, 2048, 4096)` shape persists for every draft layer's wo.
Draft's pt_wo input is correct (verified per-layer); `ttnn.as_tensor`
itself is what produces the wrong shape — the draft's wo allocation
appears to alias the target's prior wo buffer on the shared cohost
mesh.

**This is a structural Layout-A limitation:** sharing one mesh between
target + draft means DRAM-sharded buffer allocations from both models
collide. The 7 distinct shape/grid mismatches we've worked around so
far (dispatch, page_size, attention_backend, allocator, LM head,
QK-norm/SDPA, and now WO) increasingly look like surface symptoms of
the same root: cohost without per-model isolation breaks Qwen3's
tt_transformers wiring in ways that grow with each new layer of
shared state.

**Path forward** (out of scope for this session):
  - **Layout B revisited** — re-attempt split-mesh cohost (separate
    mesh per model) despite the early-finding that it's infeasible on
    P300. The current evidence weights against Layout-A more than
    the original concern weighted against Layout-B.
  - **Different mesh_mapper / mem_config** for wo specifically — force
    a fresh non-DRAM-sharded allocation per model to break aliasing.
  - **Llama-3.2-1B + Llama-3.1-8B** when Llama-3.2-1B becomes
    available — the only officially-supported tt_transformers EAGLE
    pair with proper P300 tunings, sidesteps this Qwen3 path entirely.

## QK-norm + SDPA shape strip landed (v17–v19) — exposed WO matmul mismatch

Probed Q/K/V shapes right before QK-norm and confirmed the GQA-fused
padding in `nlp_create_qkv_heads`:

  padded_head_dim = head_dim * (1 + n_local_kv_heads / n_local_heads)

For Qwen3-1.7B on (1,2) mesh: 128 * (1 + 4/8) = 192. The logical Q/K/V
shape is `[1, 8, seq, 128]` but padded shape is `[1, 8, seq, 192]`.
RMSNorm and SDPA reject this padding.

Patched `attention.py:forward_prefill` to slice Q, K, V last dim back
to `head_dim` via `ttnn.slice` after `nlp_create_qkv_heads`. Committed
to tt-metal fork at `ea7b7f3061`.

v18 cleared QK-norm. v19 cleared SDPA. Now blocked at WO output matmul
(attention.py:1169) with `TT_FATAL: width=1024 height=2048` — attn_output
post-concat_heads is 1024-wide (8 Q heads × 128 per device) but `wo`
matrix has height 2048. Likely an all-gather expected but not happening
on the 2-device fabric (vs ring topology assumption). See P3a.2 T2.2.G.

This is the **6th distinct shape/grid mismatch** in the Qwen3+P300
path. Each fix reveals the next. The pattern strongly suggests Qwen3
has no `model_params/Qwen3/P300/` tuning in tt-metal — every shape
calc falls through to a default that's only correct for one (untested)
hardware/model combination.

## LM head workaround landed (v14–v16) — exposed QK-norm shape mismatch

After v13's C++ device-grid clamp didn't help (LM head's DRAM-sharded
matmul derives its grid from DRAM banks, not device grid), v14–v16
worked around the LM head specifically:

  - `get_lm_head_program_config` now uses regular
    `MatmulMultiCoreReuseMultiCastProgramConfig` with explicit `(8, 8)`
    grid instead of `dram_matmul_config` (which picks 8×10 via DRAM
    bank-to-reader assignment).
  - `get_lm_head_input_mem_config` PREFILL → `DRAM_MEMORY_CONFIG`
    (regular matmul rejects WIDTH_SHARDED input).
  - `get_lm_head_output_mem_config` PREFILL → `DRAM_MEMORY_CONFIG`
    (regular matmul rejects WIDTH_SHARDED output).

Committed to tt-metal fork at `predator2k/tt-metal` tenstorrent-p1
`09eab40b87`. Boot now reaches PAST the LM head (further than any prior
iteration).

**Next blocker (v16)**: warmup forward pass crashes in Qwen3's QK-norm
at `rmsnorm.py:154`:

  TT_FATAL: Gamma's last padded dim needs to equal tile width and
  gamma's volume needs to align with last padded dim of input.
  gamma.padded_shape: Shape([1, 1, 4, 32])  (volume 128)
  a.padded_shape:    Shape([1, 8, 128, 192]) (last dim 192)

QK-norm gamma sized for head_dim=128, but input last dim is 192 (=
128 + 64?). Shape-mismatch in Qwen3's QKV pre-norm tensor layout
on P300 — separate from the matmul grid issue. Likely needs either
a tt_transformers patch matching the gamma to the actual post-QKV
shape, or a Qwen3-specific config in `model_params/Qwen3-8B/P300/`.

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
