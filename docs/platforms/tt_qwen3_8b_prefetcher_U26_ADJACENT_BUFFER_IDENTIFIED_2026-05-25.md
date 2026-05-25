# TT Qwen3-8B prefetcher — U26: 0xa8700 sibling buffer IDENTIFIED as previous layer's RS reduced output; REALLOCATE_W2 with move_sharded fix still fails under real weights; Path C pre-alloc blocked on trace-capture lifetime (EVIDENCE_ADVANCE) — 2026-05-25

Status: **EVIDENCE_ADVANCE — Path A (Python adjacent-buffer
ownership probe) RESOLVES the U25 mystery: the 0xa8700 "sibling
buffer" is the PREVIOUS DECODER LAYER's reduce-scatter reduced
output (`w2_out_reduced`) still alive at the next layer's MLP
dispatch time.  The W2 output / RS reduced addresses follow a
deterministic 6-layer rotation cycle across all 36 layers.
0xa8700 is NOT a stomper candidate — it is part of the residual
stream the model holds across the layer boundary.

A second key probe re-tests U22's REALLOCATE_W2 path under REAL
weights, now with the `move_sharded` per-core src/dst fix
(commit `5a5b3b5cdd6`) IN PLACE.  Result: w2_out correctly moves
to 0xaa700 (per [U22_REALLOCATE_W2] addr probe), but real-weight
output is STILL garbage (`What"*:3.'92(+.C/`, tokens
3555,6,1,9,25,18,13,...).  The move_sharded fix is a prerequisite
but not by itself sufficient.

Path C — pre-allocate w2_out via `ttnn.allocate_tensor_on_device`
+ pass via `optional_output_tensor=` — fails with `TT_THROW:
Tensor is not allocated` during trace capture.  Pre-alloc
lifetime is not threaded through trace-capture; the pre-alloc
storage is invalidated between compile-pass and capture-pass.
Deferring fix to U27.**

Continuation of `tt_qwen3_8b_prefetcher_U25_RCB_AND_BYTE_HUNT_RULE_OUT_LAST_CANDIDATES_2026-05-25.md`.

tt-metal-sglang HEAD: **`a60b0d0058c`** (U26 commit; env-gated;
default-off; canonical bytewise-equal at 9.09s e2e_latency).
Branch `tenstorrent-p1`; NOT pushed (predator2k/* fork only per
project policy).
sglang HEAD: pending this doc commit.

## TL;DR forensic chain (post-U26)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| A.1 | What allocates the L1 0xa8700 buffer at W2 dispatch time on receiver (2,7)? | `SGLANG_TT_U26_ADJ_BUFFER_PROBE=1` enumerates L1 buffers in [0xa6000, 0xb0000] at PRE_W2, POST_W2, PRE_RS, POST_RS, PRE_WO, POST_WO per-layer-per-iteration | **PREVIOUS LAYER's `w2_out_reduced`** — the RS reduced output rotates through 0xaa700 → 0xa8700 → 0xa4700 (3-cycle); the layer N+1 PRE_W2 probe sees layer N's reduced still alive at whichever address it landed at. |
| A.2 | Across which iterations / layers does the cursed 0xa6700 hold w2_out? | same probe, parsed per-layer | W2 deterministic 6-layer cycle: `a6700 a4700 a6700 a6700 a4700 a6700 a6700 a4700 ...` (period 6 across 36 layers). |
| A.3 | Does the RS reduced output ever land at the cursed 0xa6700? | same probe | Yes — when w2_out is reallocated off 0xa6700 (U22 REALLOCATE_W2 path), the RS reduced instead lands at 0xa6700.  Whatever lives at 0xa6700 inherits the stomp class. |
| B | Does U22 REALLOCATE_W2 work under REAL weights now that the `move_sharded` per-core fix (`5a5b3b5cdd6`) has landed? | `SGLANG_TT_U22_REALLOCATE_W2=1` (no zero weights) | NO — output `What"*:3.'92(+.C/` (tokens 3555,6,1,9,25,18,13,6,24,17,7,10,13,34,14).  REALLOCATE moves w2_out 0xa6700 → 0xaa700 cleanly but model output is still garbage. |
| C | Does pre-allocating w2_out via `ttnn.allocate_tensor_on_device` + `optional_output_tensor=` succeed? | `SGLANG_TT_U26_PREALLOC_W2=1` | NO — allocation succeeds at dispatch (layer 0 w2_out.addr=0xa7600 for bfp8, 0xa6700 for bf16), but trace capture later throws `TT_THROW: Tensor is not allocated` from `storage.cpp:186` because the pre-alloc storage is invalidated between compile-pass and capture-pass. |

**Punchline:** The U25 "0xa8700 sibling buffer of unknown
identity" is identified as the previous-layer residual.  No
"sibling op" candidate (no off-by-one shard map, no double-write
mechanism between 0xa6700 and 0xa8700).  The U21 verdict stands:
the stomp is at FIXED L1 0xa6700 and does NOT follow the
allocation.  But under REAL weights, even moving w2_out off the
cursed slot (REALLOCATE_W2 / Path B) does NOT restore output —
something else in the prefetcher path also breaks under
real-weights relocation.

## Phase A — adjacent-buffer ownership probe

### Probe design

Added env-gated `SGLANG_TT_U26_ADJ_BUFFER_PROBE=1` Python probe
that, per layer, per decode iteration, enumerates ALL L1 buffers
in [0xa6000, 0xb0000] at SIX bracketed dispatch points:

| Probe point | File | What it captures |
|---|---|---|
| `PRE_W2` | mlp.py before W2 matmul | L1 buffers in range BEFORE W2 allocates w2_out |
| `POST_W2` | mlp.py after W2 + w2_in dealloc | Buffers AFTER W2's output lands |
| `PRE_RS` | mlp.py before tt_all_reduce | Buffers AT RS dispatch time |
| `POST_RS` | mlp.py after tt_all_reduce returns | Buffers including the RS reduced output |
| `PRE_WO` | attention.py before WO matmul | Buffers AT attention WO dispatch time |
| `POST_WO` | attention.py after WO matmul | Buffers AFTER WO produces dense_out_sharded |

Each probe prints `(addr, buffer_type, buffer_layout, sz_per_bank)`
for every buffer.  Filters keep output bounded.

### Results — buffer ownership table (iter=2, post-trace-capture)

The 6-layer rotation cycle, sampled across all 36 layers:

| Layer | POST_W2 w2_out.addr | POST_RS reduced.addr |
|---|---|---|
| 0  | 0xa6700 | 0xaa700 |
| 1  | 0xa4700 | 0xa8700 |
| 2  | 0xa6700 | 0xa4700 |
| 3  | 0xa6700 | 0xaa700 |
| 4  | 0xa4700 | 0xa8700 |
| 5  | 0xa6700 | 0xa4700 |
| 6  | 0xa6700 | 0xaa700 |
| 7  | 0xa4700 | 0xa8700 |
| ... | (period 6) | (period 3) |

PRE_W2 enumeration shows for any layer N, the L1 contains (in
addition to GlobalCB at 0xac700):

| Slot | Identity at PRE_W2 |
|---|---|
| 0xa4700 | layer N-1's reduced (if it landed there) or stale free |
| 0xa6700 | layer N-1's w2_out (kept alive by tt_all_reduce input ref) OR free |
| 0xa8700 | **layer N-2's or N-3's reduced** (residual-stream-rotated) |
| 0xaa700 | similar — another previous-layer reduced |
| 0xa8d80 / 0xaad80 | attention output cat or similar 6528 B/bank slot (variant layers) |
| 0xab700 | attention WO output 4096 B/bank (variant layers) |
| 0xac700 | GlobalCB receiver (always present) |

**0xa8700 IDENTITY: previous-layer (N-1 or N-3) RS reduced output
(`w2_out_reduced` from `tt_all_reduce`), 8192 B/bank WIDTH_SHARDED,
identical shape/layout to current-layer w2_out because both are
`[1,1,32,4096/ring_size]`.  These are not "siblings co-allocated"
— they are the SAME class of tensor at different layer
positions in the residual-stream rotation.**

PRE_WO / POST_WO probes show similar rotation for attention
outputs.

### Implication

U25's hypothesis "an op writes to 0xa8700 and accidentally ALSO
targets 0xa6700" is RULED OUT.  No single op produces both
0xa6700 and 0xa8700 outputs at the same time — they are
allocated by different layers at different times in the rotation.

The 0xa6700 stomp must originate from somewhere OTHER than a
sibling-buffer-aliasing mechanism.

## Phase B — REALLOCATE_W2 re-test with move_sharded per-core fix

### Setup

The U22 REALLOCATE_W2 path was previously blocked because
`move_sharded` used bank-averaged src/dst addresses to compute a
single chunk size, which corrupted shards on cores with per-core
allocation deltas different from the bank average (see U22 doc).
Commit `5a5b3b5cdd6` (already in tenstorrent-p1 branch) fixes
this by threading per-core src/dst addresses through the kernel
RT args.

### Run

```bash
SGLANG_TT_USE_PREFETCHER=1
SGLANG_TT_U22_REALLOCATE_W2=1
SGLANG_TT_U22_PRINT_ADDR=1
# NO PREFETCHER_ZERO_WEIGHTS this time — REAL weights
```

### Result

```
[U22_REALLOCATE_W2] old=0xa6700 new=0xaa700 pass_mc=True   ✓
[U22_REALLOCATE_W2] old=0xa4700 new=0xaa700 pass_mc=True   ✓
...                                                          ✓
[U26_ADJ_POST_W2] iter=2 layer=0 w2_out.addr=0xa6700        (pre-reallocate)
[U26_ADJ_POST_RS] iter=2 layer=0 reduced.addr=0xa6700       (RS reduced now AT 0xa6700)
```

Output (real weights): ` What"*:3.'92(+.C/` (tokens
3555,6,1,9,25,18,13,6,24,17,7,10,13,34,14) — fully garbage,
same pattern as U22's pre-fix observation.

### Sub-finding — RS reduced inherits the cursed slot

When REALLOCATE moves w2_out from 0xa6700 → 0xaa700, the RS
reduced (which was previously at 0xaa700) instead lands at
0xa6700.  Per U21's "stomp is at FIXED L1 0xa6700" verdict, the
stomp now targets WHATEVER lives at 0xa6700 — which is now the
RS reduced output.  This explains why REALLOCATE alone doesn't
fix things: it just shifts which tensor inherits the stomp.

### Sub-finding — U17 PRE_RS NONZERO is uninformative under real weights

Under REAL weights + REALLOCATE_W2 + U17 PRE_RS at 0xaa700:
2880 NONZERO / 0 zero.  This is consistent with W2's genuine
non-zero output AT 0xaa700 (the new location) — it doesn't tell
us whether the stomp also fires.  The U17 probe (designed pre-U21
when bug was thought to be address-bound) can only distinguish
zero-vs-nonzero, not "correct W2 bytes vs stomped garbage".

## Phase C — pre-allocate w2_out via `ttnn.allocate_tensor_on_device`

### Approach

Add a helper `_u26c_prealloc_w2_out()` on the MLP module that
lazily allocates a persistent device tensor with the exact
mem_config / shape / dtype that `ttnn.linear(w2)` would otherwise
produce.  Pass this tensor via `optional_output_tensor=` so the
matmul writes INTO it instead of allocating a new buffer each
iteration.

```python
def _u26c_prealloc_w2_out(self):
    if getattr(self, "_u26c_w2_out", None) is not None:
        return self._u26c_w2_out
    _mc = self.args.get_mlp_ff2_mem_config(Mode.DECODE, self.prefetcher)
    _dtype = self.args.ccl_dtype if self.args.is_galaxy else ttnn.bfloat16
    _shape = (1, 1, 32, self.args.dim)
    _t = ttnn.allocate_tensor_on_device(
        ttnn.Shape(_shape), _dtype, ttnn.TILE_LAYOUT, self.mesh_device, _mc,
    )
    self._u26c_w2_out = _t
    return _t
```

Hook in W2 ttnn.linear:
```python
optional_output_tensor=_u26c_out_t,
```

### Result

Allocation succeeds at first MLP dispatch (layer 0
w2_out.addr=0xa6700 for bf16 dtype — note: it lands AT the
cursed address because the allocator picks lowest-free).  Trace
capture begins, then THROWS:

```
TT_THROW: Tensor is not allocated (assert.hpp:104)
RuntimeError: TT_THROW @ /tt-metal/ttnn/core/tensor/storage.cpp:186
backtrace: ttnn.linear → matmul_device_operation.cpp
```

### Diagnosis

The pre-allocated tensor's storage is invalidated between the
compile pass and the capture pass.  `_capture_decode_trace_text`
in `generator.py` runs the model TWICE: a compile pass for
program-config caching, then a capture pass that hard-codes
addresses into the trace.  Between the two passes, captured-time
buffer references can be deallocated, and a tensor allocated
BEFORE the compile pass has no protection against this.

To make pre-alloc work, the pre-allocation lifecycle must be
threaded into trace-capture explicitly — either by re-allocating
just before capture begins, or by using a trace-aware persistent
buffer API (akin to `persistent_output_buffer=` on the CCL ops).

### Sub-finding — even fresh pre-alloc lands at 0xa6700 anyway

A side observation: `ttnn.allocate_tensor_on_device` picks the
lowest-free L1 slot in the allocator's freelist, which is the
SAME slot the in-trace allocator picks.  Without an allocator
hint that forces a non-rotating slot (e.g., pinning to 0xb0000+),
pre-allocation alone does NOT move the buffer off the cursed
address — the rotation pattern would still emerge.

## Hypothesis ledger (post-U26)

| ID | Suspect | Pre-U26 | Post-U26 |
|---|---|---|---|
| (all U17–U25 ruled-outs) | … | unchanged | unchanged |
| **U26 A — 0xa8700 sibling buffer is a per-iteration mystery allocation requiring its own forensic chain** | (per U25) | **RULED OUT** | identified as previous-layer `w2_out_reduced` (residual rotation) |
| **U26 A — Some single op writes to BOTH 0xa6700 and 0xa8700 due to off-by-one shard map** | (per U25 top hypothesis) | **RULED OUT** | the two addresses are owned by DIFFERENT layers' MLPs at different times in the rotation; no co-temporal aliasing op |
| **U26 B — REALLOCATE_W2 with move_sharded per-core fix works under real weights** | (new) | **RULED OUT** | output still garbage; w2_out moves cleanly to 0xaa700 but RS reduced inherits 0xa6700 + stomp class |
| **U26 C — Pre-allocate w2_out via ttnn.allocate_tensor_on_device + optional_output_tensor= works** | (new) | **BLOCKED** | TT_THROW Tensor not allocated during trace capture; pre-alloc lifetime not threaded |
| **U26 STANDING — what writes garbage to receiver-core (2,7) L1 0xa6700 between W2 PACK exit and RS reader entry?** | (per U25) | **STANDING — TOP PRIORITY** | candidate set unchanged from U25 |
| **U26 NEW STANDING — when REALLOCATE moves w2_out off 0xa6700, why does the model still produce garbage under real weights even though the U21 verdict says stomp at 0xa6700 should now be harmless?** | (new) | **NEW STANDING** | suggests either: (a) stomp targets MULTIPLE addresses, or (b) REALLOCATE introduces an independent perturbation, or (c) the RS reader / writer is now reading FROM the stomped 0xa6700 (RS reduced inherited the slot) and propagating it through subsequent layers' residual stream |

## What U26 lands (all env-gated; canonical bytewise-equal)

| File | Change |
|---|---|
| `models/tt_transformers/tt/mlp.py` | `SGLANG_TT_U26_ADJ_BUFFER_PROBE`-gated PRE_W2, POST_W2, PRE_RS, POST_RS L1 enumeration blocks; `SGLANG_TT_U26_PREALLOC_W2`-gated `_u26c_prealloc_w2_out()` + `optional_output_tensor=` wire-up (allocation works; trace capture crashes; kept for U27 hand-off). |
| `models/tt_transformers/tt/attention.py` | `SGLANG_TT_U26_ADJ_BUFFER_PROBE`-gated PRE_WO and POST_WO L1 enumeration blocks. |

## Reproducer

### A. U26 Phase A — adjacent-buffer ownership probe

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c '
  TT_CACHE_HOME=/root/.cache/tt-metal-cache
  [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ] && rm -rf "${TT_CACHE_HOME}"/*
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_U26_ADJ_BUFFER_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  nohup python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u26a_server.log 2>&1 &
'
podman exec p3a-ngram bash -c '
  until curl -sf http://127.0.0.1:30000/health_generate > /dev/null 2>&1; do sleep 5; done
  curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
    -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":3,\"temperature\":0.0}}"'
grep -E "U26_ADJ_(POST_W2|POST_RS)\] iter=2 layer=" /tmp/u26a_server.log | head -10
# Confirms the 6-layer / 3-cycle rotation pattern.
```

### B. U22 REALLOCATE_W2 under REAL weights (no zero weights)

```bash
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U22_REALLOCATE_W2=1 \
SGLANG_TT_U22_PRINT_ADDR=1 \
... (no PREFETCHER_ZERO_WEIGHTS)
# Output: ' What"*:3.'92(+.C/' garbage tokens.
```

### C. Path C pre-alloc crash

```bash
SGLANG_TT_USE_PREFETCHER=1 \
SGLANG_TT_U26_PREALLOC_W2=1 \
...
# Crashes with TT_THROW: Tensor is not allocated.
```

### D. Canonical re-verify

```bash
SGLANG_TT_USE_PREFETCHER=0 ...
# Output: " What is 2+2? What is 2+2? What" at 9.09s e2e_latency.
```

## U27 dispatch recommendation

### U27 Phase A — pre-alloc lifecycle threading

Make `SGLANG_TT_U26_PREALLOC_W2=1` actually work by either:
- Re-allocating the persistent buffer inside the capture pass
  (hook into `_capture_decode_trace_text` via a model-level
  callback registered with the prefetcher).
- Use the `persistent_output_buffer=` style API that already
  exists for `reduce_scatter_minimal_async` and `all_gather_async`
  for trace-aware persistent buffers, but at the matmul level.
  Investigate if `ttnn.linear` / `ttnn.matmul` can accept a
  `persistent_output_buffer=` analogue.

### U27 Phase B — explore "what fails when RS reduced lands at 0xa6700"

Under REALLOCATE_W2 + REAL weights, the RS reduced inherits the
cursed 0xa6700 slot.  Add a probe to read the RS reduced bytes
immediately after RS, compare them across layers, see if there's
a corruption signature in the reduced output that propagates
through subsequent residual adds.

### U27 Phase C — investigate the OTHER 0xa6700-class addresses

Maybe the stomp also hits 0xa4700 (where w2_out sometimes lands).
Add probes at 0xa4700 mirroring the U17 PRE_RS probe at 0xa6700.
If 0xa4700 also shows stomp under zero weights, the stomp is
NOT at single fixed L1 address but on a class of addresses
(maybe per-receiver-core specific offsets, but invariant across
the rotation cycle).

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (only tt-metal C++ in prior
      commits + Python probes in mlp.py / attention.py, all
      env-gated).
- [x] All `rm` operations guard with `[ -n "$VAR" ]` (TT_CACHE_HOME)
      + literal-path absolute.
- [x] No stash pop/drop.
- [x] Port-clear used `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — used `podman cp` for source
      sync (Python-only changes; no rebuild needed).
- [x] All new probes / Path C variant env-gated (`SGLANG_TT_U26_*`).
- [x] Canonical Qwen3-8B re-verified bytewise: " What is 2+2? What
      is 2+2? What" at 9.09s e2e_latency (no regression).
- [x] Server stopped at session end.
- [x] TT devices: healthy.

## Shipping verdict (unchanged from U19/U20/U21/U22/U23/U24/U25)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat ~ 9-10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**
  The L1 0xa6700 stomp persists; no real-weight-preserving
  workaround found.
- **DO NOT ship any `SGLANG_TT_U26_*` env var.**  Diagnostic only.

The U26 probes are safe to leave as default-off probes — they
are strictly NO-OP for canonical builds (the probe blocks are
Python `if env == "1":` gates, no kernel-level changes added in
U26).

## Commits this session

- (tt-metal-sglang) **`a60b0d0058c`** —
  `prefetcher: U26 — Path A adjacent-buffer ownership probe identifies 0xa8700 as PREVIOUS LAYER's RS reduced output; Path C pre-alloc workaround crashes (EVIDENCE_ADVANCE)`
- (sglang) `<this doc>` — pending commit.

## Working state at session end

- tt-metal-sglang HEAD: **`a60b0d0058c`** (U26 commit; env-gated;
  default-off; canonical bytewise-equal).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed;
  predator2k/* fork only per project policy).
- sglang HEAD: pending this doc commit.  No sglang Python
  changes in U26.
- Container `/tt-metal/`: synced (mlp.py + attention.py with U26
  probes applied).  No C++ changes in U26 — no rebuild needed.
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: bytewise " What is 2+2? What is 2+2? What"
  at 9.09s e2e_latency.
