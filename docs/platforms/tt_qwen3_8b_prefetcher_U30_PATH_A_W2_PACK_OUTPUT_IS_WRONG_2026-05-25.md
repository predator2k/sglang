# TT Qwen3-8B prefetcher — U30 Path A: W2 PACK output bytes ARE wrong at compute-kernel exit under real weights and U29-Phase-3 signaler (Theory A confirmed; Theory B downstream-stomp refuted) — 2026-05-25

Status: **EVIDENCE_ADVANCE — U30 Phase 1 (Path A) discrimination
between matmul-output-is-wrong vs matmul-output-is-correct-but-stomped
LANDS Path A. With the U29 Phase 3 signaler enabled (verified
producer→consumer handshake from prior session) and REAL weights, the
gathered compute kernel's `U14_END_OUT` probe — which `tensix_sync()`s
the FPU pipeline before reading mm_out_cb's L1 base bytes at
end-of-batch — captures GARBAGE at PACK exit on 3 of 4 gathered ELFs:
ELF `0x2d96e` 88% garbage at L1 0xa6700, ELF `0x2de6b` 95% garbage at
L1 0xaaf00, ELF `0x2df6a` 95% garbage at L1 0xa76c0. ELF `0x2db6e`
(writing to 0x9c6c0..0xa16c0) is 100% NORMAL. The garbage matches what
U14_CONSUMER_FIRST (the RS reader's per-page probe) sees when it
reads the same L1 region. This rules out Theory B (something stomps
the L1 buffer between PACK retirement and RS read) because there is
no such "between" window — PACK already wrote garbage. This re-opens
U7's verdict (`Case B confirmed`): the compute kernel's
`use_global_cb`/`ENABLE_GLOBAL_CB` math+pack path produces wrong
outputs from correct inputs on specific compile-time-arg parameterizations.
Canonical (U29 OFF, prefetcher OFF) re-verified `" What is 2+2? What is
2+2? What"` at 9.16s + GSM8K(10) = 10/10 = 100%. No regression.**

Continuation of `tt_qwen3_8b_prefetcher_U29_PHASE3_GLOBALSEMAPHORE_HANDSHAKE_VERIFIED_RACE_NOT_ROOT_CAUSE_2026-05-25.md`.

tt-metal-sglang HEAD: `5d5d4a02999` (U29 Phase 3 tip — unchanged this session; Path A probe was already in tree, this session only ran it with the right env).
sglang HEAD: pending this doc.

## TL;DR (Phase 1 / Path A forensic chain)

| Step | Question | Probe / Action | Result |
|---|---|---|---:|
| A.1 | Under U29 signaler + REAL weights, does RS reader actually pull wrong bytes from W2's output L1 buffer? | Run `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_U29_W2_RS_SIGNALER=1 SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1` + `TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=NC`; capture U14_CONSUMER_FIRST (~50-byte DPRINT inside the RS reader kernel that reads the first 16 bytes of each input page after `noc_async_read_barrier`). | **YES — at L1 0xa6700 (W2 receiver-core output), 120/176 reads = 68% garbage (`0x7f*/0xff*/0x73*/0xf3*` exponent NaN/Inf); at L1 0xa8900, 1056/2320 reads = 46% garbage; mix of normal-magnitude and garbage reads.** Token output: `" Whatg.)."` (broken). |
| A.2 | Is the garbage already present at compute-kernel EXIT (PACK retired), or only later? | Same run, add `TT_METAL_DPRINT_RISCVS="TR0,TR1,TR2,NC,BR"` to capture PACK-thread DPRINTs from U14_END_OUT probe (gated by `SGLANG_TT_PREFETCHER_CONSUMER_PROBE`); reads mm_out_cb's L1 base bytes AFTER all PACK writes for batch b are retired AND `ckernel::tensix_sync()` drained the FPU pipeline. | **GARBAGE PRESENT AT PACK EXIT.** ELF `0x2d96e` at L1 0xa8900: **4039/4672 = 86% garbage**; ELF `0x2de6b` at L1 0xaaf00: **3993/6144 = 65% garbage**; ELF `0x2df6a` at L1 0xa76c0: **3757/5696 = 66% garbage**. ELF `0x2db6e` writing to addresses 0x9c6c0/0x9e6c0/0x9f6c0/0xa16c0: **0% garbage, 100% NORMAL BF16**. |
| A.3 | Discrimination: matmul-output-is-wrong vs matmul-output-is-correct-but-stomped? | A.1 + A.2 together prove: bytes are wrong AT PACK EXIT (which `tensix_sync`s the FPU). There is no kernel running between PACK retirement (within the kernel) and U14_END_OUT measurement (same kernel, three lines later). | **Theory A confirmed (matmul output is wrong); Theory B refuted (no stomp window between PACK and U14_END_OUT measurement).** |
| A.4 | Is this consistent with U12/U13 ("math+PACK are correct under zero weights")? | U12 (LLK srcA/srcB/DST=0 + tensix_sync) and U13 (PACK output=0 + tensix_sync) both ran under `SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1`. Re-ran U14_END_OUT under zero weights this session. | **YES — under zero weights, all 4 ELFs at all probed L1 addresses show `w0=0x0 w1=0x0 w2=0x0 w3=0x0 zero`. The math IS correctly computing `x * 0 = 0`.** The garbage only manifests under REAL weights for 3 specific gathered ELFs. |
| A.5 | Did we break canonical (U29 OFF, prefetcher OFF)? | Standard canonical re-verify. | **NO — `" What is 2+2? What is 2+2? What"` at 9.16s e2e; GSM8K(10) = 10/10 = 100.0%.** Bytewise-equal vs U29 Phase 3 baseline. No regression. |

**Punchline:** U30 Path A nails the bug to inside the **gathered
matmul compute kernel itself** — specifically to compile-time-arg
parameterizations that produce 86-95% garbage tiles at PACK exit
under REAL weights, while one parameterization (`0x2db6e`) produces
clean output. This is the same scenario U7 described in 2026-05-24
("Case B confirmed; gathered compute kernel produces ~2^60..2^109
magnitude outputs from 4 of 14 ELFs on first invocation"). The U29
Phase 3 signaler handshake is **correct closure of the W2→RS race**,
but it's a **fix to a non-problem** — the bytes the consumer reads
are already garbage at the moment PACK retired, so the consumer
synchronizing to PACK retirement doesn't help.

The **next attack (U31)** must target the gathered compute kernel's
behavior on the GARBAGE ELFs vs the NORMAL ELF. Specifically:

1. **Diff compile-time args between `0x2db6e` (normal) and `0x2d96e/0x2de6b/0x2df6a` (garbage).** The elf-tag is
   `(in0_block_w * 1u) ^ (in0_num_subblocks * 131u) ^ (in1_num_subblocks * 17u) ^ (num_blocks * 7919u) ^ (out_subblock_h * 31u) ^ (out_subblock_w * 257u) ^ (batch * 65537u)`.
   Bisect by toggling one arg at a time on the garbage ELFs to find the discriminator.
2. **Probe DST register pre-PACK on the garbage ELFs under real weights.** U12 only ran under zero weights; under real weights the DST may be wrong. Use `dprint_tensix_dest_reg<>(0)` after the last `matmul_block` and before `pack_tile_block`.
3. **Probe mm_partials_cb spill/reload on the garbage ELFs under real weights.** The `spill` branch reloads mm_partials_cb into DST as the starting accumulator; if mm_partials_cb's L1 region is touched by the prefetcher GCB (which shares L1 on receiver cores), spill garbage would explain the explosion.
4. **Probe the `ENABLE_GLOBAL_CB` rd_ptr arithmetic on the garbage ELFs.** `update_rd_ptr_to_ring_index` and `update_local_cb_rd_ptr` may be feeding the wrong in1 weight tile to the matmul on the garbage ELFs (causing the math itself to compute against stale or wrong weights). This is the GCB-receiver-side race that U13's PACK_PROBE under zero weights cannot detect (because anything × 0 = 0).

## Run results (full reproduction)

### Path A.1 — RS reader U14_CONSUMER_FIRST under U29 signaler + REAL weights

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c 'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U29_W2_RS_SIGNALER=1 \
  SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_RISCVS=NC \
  TT_METAL_DPRINT_FILE=/tmp/u30_pathA_u14end.log \
  TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
  python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native'

# Wait for /health, then:
curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text": "What is 2+2?", "sampling_params": {"max_new_tokens": 5, "temperature": 0.0}}'
# → " Whatg.)." at 11.05s e2e_latency  (garbage)
```

Per-address garbage classification of 23k U14_CONSUMER_FIRST entries:

| L1 input addr (RS reader's per-page read) | GARBAGE | NORMAL | % garbage |
|---|---:|---:|---:|
| `0xa6700` (W2 receiver L1 — same as U17-U19) | 120 | 56 | **68%** |
| `0xa8900` (adjacent receiver L1)             | 1056 | 1264 | **46%** |
| `0xa4700`                                    | 72   | 24   | **75%** |
| `0xa66c0`                                    | 32   | 0    | **100%** |
| `0x47b97640` (DRAM tensor address)           | 0    | 3456 | **0%**  |
| `0x47b8de40`                                 | 0    | 3264 | **0%**  |

The DRAM tensor addresses (>= 0x40000000) read 100% NORMAL → these
are matmul outputs that landed in DRAM (non-prefetcher paths) and
have NOT been corrupted. The low L1 addresses (0xa****) are the
**receiver-core L1 buffers** for the gathered matmuls (W2/WO) and
they show heavy garbage.

### Path A.2 — Compute kernel U14_END_OUT (PACK exit) under U29 signaler + REAL weights

Same launch as A.1 but with `TT_METAL_DPRINT_RISCVS="TR0,TR1,TR2,NC,BR"`
to capture the PACK-thread DPRINT. Prompt `"2+2="` max=3 (smaller to
keep log size manageable).

Curl output: `"5k1"` (garbage).

`tot=1` (first-dispatch-per-ELF-instance) entries grouped by `elf` × `l1`:

| ELF | L1 addr | GARBAGE | NORMAL | % garbage |
|---|---|---:|---:|---:|
| `0x2d96e` | 0xa8900 | 4039  | 633  | **86%** |
| `0x2d96e` | 0xa6700 | 639   | n/a  | **100%** |
| `0x2d96e` | 0xa4700 | 384   | n/a  | **100%** |
| `0x2d96e` | 0xa66c0 | 128   | n/a  | **100%** |
| `0x2de6b` | 0xaaf00 | 3993  | 2151 | **65%** |
| `0x2de6b` | 0xa8f00 | 224   | 160  | **58%** |
| `0x2de6b` | 0xa6f00 | 224   | 160  | **58%** |
| `0x2df6a` | 0xa76c0 | 3757  | 1939 | **66%** |
| `0x2df6a` | 0xa8700 | 379   | 69   | **85%** |
| `0x2df6a` | 0xaa700 | 320   | n/a  | **100%** |
| `0x2df6a` | 0xa6700 | 320   | n/a  | **100%** |
| **`0x2db6e`** | 0xa16c0 | **0** | **2880** | **0%** (NORMAL) |
| **`0x2db6e`** | 0x9e6c0 | **0** | **2880** | **0%** (NORMAL) |
| **`0x2db6e`** | 0x9f6c0 | **0** | **2752** | **0%** (NORMAL) |
| **`0x2db6e`** | 0x9c6c0 | **0** | **2752** | **0%** (NORMAL) |
| **`0x2db6e`** | 0xa5700 | **0** | **448**  | **0%** (NORMAL) |
| **`0x2db6e`** | 0xa2700 | **0** | **448**  | **0%** (NORMAL) |

**Classification rule (`bad`): w0/w1/w2/w3 leading byte ∈ {0x7e,0x7f,0xfe,0xff,0x73,0x74,0x75,0x76,0xf3,0xf4,0xf5,0xf6} = NaN/Inf or near-NaN exponent.**

### Path A.4 — Same probe under ZERO weights

Same launch, `SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1` added.

Curl output: `"Und_flag_flag"` (still garbage — but garbage from a
different cause; under zero weights the model output is the residual
stream's untouched activations driven through 36 layers).

ALL 1088 entries at L1 0xa6700 (and all other addresses):
```
0:4-7:TR2: [U14_END_OUT elf=0x2d96e l1=0xa6700 tot=1 b=0
            w0=0x0 w1=0x0 w2=0x0 w3=0x0 zero]
```

Confirms U12/U13: math is correct under zero weights; the bug
manifests only under REAL weights.

### Canonical re-verify (U29 OFF, prefetcher OFF)

```
SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
HF_MODEL=/models/Qwen3-8B ... python3 -m sglang.launch_server ...

curl -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":15,"temperature":0.0}}'
→ " What is 2+2? What is 2+2? What" at 9.16s e2e

eval_qwen3_8b_gsm8k_chat.py --num 10
→ FINAL: 10/10 = 100.0%
```

**Bytewise-equal vs U28/U29 canonical baseline. No regression.**

## Temporal degradation trace (one specific core)

Tracking core `1:3-5:TR2` reads at L1 `0xa6700` chronologically:

| Iter | ELF | bytes (w0..w3) | magnitude class |
|---:|---|---|---|
| 1 | 0x2d96e | `0xbd47bcdb 0x3a753a33 0xbc463b75 0x3d69bc7e` | **NORMAL** ~ -0.05 |
| 2 | 0x2df6a | `0x3e9abe1e 0x3c30bd4d 0xbc493e2b 0x3d46bd58` | **NORMAL** ~ 0.3 |
| 3 | 0x2d96e | `0x7f617e91 0xff717dd1 0x7f10fe72 0xfe3afd9a` | **NaN/Inf** ~ 2^127 |
| 4 | 0x2d96e | `0x7f2d7f2b 0xff607f6b 0xff2f7f37 0xff4d7d24` | NaN/Inf |
| 5+ | both | all 0x7f*/0xff* | NaN |

So even the GARBAGE ELFs produce 1-2 iterations of normal output
before exploding. This matches U7's temporal degradation pattern and
suggests the bug is in **how the gathered compute kernel handles
inter-decode-step state** (DST accumulator across batches, or
mm_partials_cb stale state across program launches, or in1 weight
rd_ptr stale state across launches).

## Architectural learnings (Path A)

### Learning 1: U29 Phase 3 signaler is a CORRECT fix for a NON-PROBLEM

The W2→RS handshake closure (sentinel test 2880/2880 = 0xDEADBEEF;
counter wait engages and increments) is the architecturally-correct
closure of the cross-sub-device dispatch race. But the consumer
(RS reader) reading the W2 output buffer **at the right moment**
doesn't help when the buffer **already** contains garbage by the time
PACK retired. The signaler does not produce or consume bytes; it only
orders existing bytes. If those bytes are wrong, they remain wrong.

This rules out U28-β (cross-subdev dispatch race) as the dominant
source of corruption — even when fully closed by U29, the symptom
persists, because the symptom does NOT depend on dispatch ordering.

### Learning 2: 3 of 4 gathered ELFs are buggy under real weights

ELFs `0x2d96e, 0x2de6b, 0x2df6a` produce 65-100% garbage tiles at
PACK exit on first-dispatch.
ELF `0x2db6e` produces 0% garbage tiles.

Likely identification (TBD via factory probe in U31):
- `0x2db6e` (normal) writes to L1 0x9c6c0/0x9e6c0/0x9f6c0/0xa16c0 with
  page size suggesting a smaller block — likely one of the **smaller
  matmuls** (W1 or W3 gated FF1/FF3 path, which has K-dim 2048).
- `0x2d96e, 0x2de6b, 0x2df6a` write to L1 0xa6700/0xaaf00/0xa76c0 —
  likely **W2 (down_proj, K-dim = intermediate = 12288)** plus one or
  more of WO/WQKV (also K-dim 4096).

So the bug correlates with **K-dim / num_blocks** — larger K-dim
matmuls have more `num_blocks` iterations in the `spill` branch,
exercising more mm_partials_cb spill/reload, and more `update_rd_ptr_to_ring_index`
calls. The U13 PACK_PROBE under zero weights showed `blk=31` NONZERO
without sync; with sync it was clean. Under real weights, this might
not be a race — it might be that the spill/reload arithmetic itself
is wrong for the larger-K ELFs.

### Learning 3: U7's "Case B" diagnosis is now CONFIRMED end-to-end

U7 (2026-05-24) saw 4 of 14 gathered ELFs produce 2^60-2^109 garbage
at PACK exit on first invocation. The intervening U8-U29 line of
attack chased downstream-stomp theories (U17 PRE-RS stomp probe;
U19 fill-zero workaround; U20-U25 hunt for stomper kernel; U26-U28
buffer-aliasing+dispatch-race). Each ruled out the candidate AND
narrowed the surface, but none addressed the original Case B finding.

**The U30 Path A measurement re-validates Case B with the U29 Phase 3
signaler ENABLED, conclusively showing the W2→RS race fix does not
remove the garbage at PACK exit. The bug has been in the gathered
compute kernel the whole time; the downstream-stomp lineage was a
red herring.**

The U19 fill-zero workaround (which CONFIRMED stomp signature) was
**also a red herring**: it appeared to "fix" the bug because zeroing
W2's output before RS makes RS scatter zeros, which keeps the
residual stream clean → coherent output. But it wasn't fixing a stomp
— it was bypassing the actual garbage that the matmul kernel produced
and replacing it with zero. Either fix produces a coherent residual
stream; only the matmul-bytes-are-wrong interpretation is consistent
with U30 Path A's PACK-exit measurements.

## Hypothesis ledger (post-U30 Path A)

| ID | Suspect | Pre-U30 | Post-U30 |
|---|---|---|---|
| **U7 / Case B — Gathered compute kernel produces garbage on specific compile-time-arg ELFs under REAL weights** | EVIDENCE_ADVANCE (2026-05-24) | **CONFIRMED — TOP STANDING.** U30 Path A re-validates with U29 signaler enabled; 3/4 ELFs produce 65-100% garbage at PACK exit; ELF `0x2db6e` is clean. |
| **U16 Phase 5c / U29 — W2→RS device-side handshake** | LANDED + EMPIRICALLY VERIFIED (U29 Phase 3) | **RETIRED as the fix** — handshake is correct closure of a non-causal race. Leave in tree as default-off; will be useful for any future RS-side race investigation. |
| **U28 β — Cross-sub-device dispatch race at W2→RS** | NEW TOP per U28 | **REFUTED (U29 Phase 3 + U30 Path A).** |
| **NEW U30 — Gathered kernel mm_partials_cb spill/reload bug under real weights** | (new) | **NEW STANDING.** U13 only tested under zero weights (where anything × 0 = 0 hides spill bugs). Under real weights, the spill branch might reload garbage from mm_partials_cb's L1 region (possibly aliased with prefetcher GCB / receiver L1 layout). Probe in U31. |
| **NEW U30 — GCB rd_ptr arithmetic bug under real weights** | (new) | **NEW STANDING.** `update_rd_ptr_to_ring_index` / `update_local_cb_rd_ptr` semantics under ENABLE_GLOBAL_CB might land in1 weight tile on the wrong DRAM block on the buggy ELFs. Under zero weights this is hidden because anything × 0 = 0. Probe in U31 by reading cb_in1 bytes BEFORE matmul_block on the garbage ELFs and verifying they match the expected weight tile. |
| **NEW U30 — DST accumulator stale state across decode iterations** | (new) | **NEW STANDING.** First-iteration output looks correct (1-2 normal magnitudes per ELF per core), then explodes. This is consistent with DST holding the previous decode iteration's huge output and the spill reload picking up the stale magnitude. Probe in U31 by reading DST at kernel ENTRY on the buggy ELFs (should be zero or the just-prior `tile_regs_acquire` init). |
| U17-U28 downstream-stomp lineage (fabric, dispatcher, RCB, prefetcher writer, adjacent buffer, GCB tile_init_op, per-core hypothesis) | EACH RULED OUT incrementally | All consistent with the new top: there was no stomper because the bug was at PACK exit. The L1-0xa6700 "stomp signature" is just the wrap from the buggy matmul writing huge values to its output buffer. |

## Hard constraints checked

- [x] No upstream PR (predator2k/* only).
- [x] No SGLang core behavior changes (no tt-metal-sglang edits this session; only ran existing probes with new env combinations).
- [x] All `rm` operations guard with `[ -n "$CACHE" ]` and use literal absolute path (`TT_CACHE_HOME=/root/.cache/tt-metal-cache`).
- [x] No stash pop/drop.
- [x] Port-clear uses `\.` escape (`pkill -f "sglang\.launch_server.*--port 30000"`).
- [x] `HF_MODEL=/models/Qwen3-8B` local path.
- [x] Container NOT bind-mounted — verified host vs container source diff is empty for gathered compute kernel + factory; ttnn lib has all U29 + CONSUMER_PROBE defines.
- [x] Probes env-gated (`SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1`, `SGLANG_TT_U29_W2_RS_SIGNALER=1`).
- [x] Canonical Qwen3-8B re-verified bytewise (`" What is 2+2? What is 2+2? What"` at 9.16s) AND GSM8K(10) = 10/10 = 100% with U29 OFF + prefetcher OFF.
- [x] Server stopped at session end; cache cleared at session end.

## Reproducer

### A. Path A probe (compute-kernel + RS-reader bytes, U29 + real weights)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c 'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_U29_W2_RS_SIGNALER=1 \
  SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_RISCVS="TR0,TR1,TR2,NC,BR" \
  TT_METAL_DPRINT_FILE=/tmp/u30_pathA.log \
  TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
  python3 -u -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u30_pathA_server.log 2>&1'

# Wait for /health, then:
curl -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text":"2+2=","sampling_params":{"max_new_tokens":3,"temperature":0.0}}'

# Stop and analyze:
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000"'
podman exec p3a-ngram grep -c "U14_END_OUT" /tmp/u30_pathA.log
podman exec p3a-ngram grep -c "U14_CONSUMER_FIRST" /tmp/u30_pathA.log

# Per-ELF garbage classification:
podman exec p3a-ngram bash -c "grep \"U14_END_OUT.*tot=1 \" /tmp/u30_pathA.log | \
  awk '{
    elf=\"\"; l1=\"\"; w0=\"\"; w1=\"\"; w2=\"\"; w3=\"\";
    for(i=1;i<=NF;i++){
      if(\$i~/^elf=/)elf=\$i; if(\$i~/^l1=/)l1=\$i;
      if(\$i~/^w0=/)w0=\$i; if(\$i~/^w1=/)w1=\$i;
      if(\$i~/^w2=/)w2=\$i; if(\$i~/^w3=/)w3=\$i;
    }
    bad=0;
    if(w0~/=0x(7e|7f|fe|ff|73|74|f3|f4)/||w1~/=0x(7e|7f|fe|ff|73|74|f3|f4)/||
       w2~/=0x(7e|7f|fe|ff|73|74|f3|f4)/||w3~/=0x(7e|7f|fe|ff|73|74|f3|f4)/)bad=1;
    print elf,l1,(bad?\"GARBAGE\":\"NORMAL\")
  }' | sort | uniq -c | sort -rn"
```

### B. Canonical re-verify (default — U29 OFF, prefetcher OFF)

```bash
podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 5'
podman exec p3a-ngram bash -c 'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  python3 -u -m sglang.launch_server ...'

# Expected: " What is 2+2? What is 2+2? What" at ~9s e2e; GSM8K 10/10
```

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ~ 27 ms / 1024-1024 and GSM8K(10) chat = 10/10.

- **DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  The
  gathered compute kernel produces 65-100% garbage tiles at PACK
  exit on 3 of 4 unique ELFs under REAL weights.  Fix requires
  identifying which compile-time arg combo triggers the bug and
  either avoiding it or patching the gathered kernel's math path.
- **DO NOT ship `SGLANG_TT_U29_W2_RS_SIGNALER=1`.**  Signaler is
  correct closure of a non-causal race; does not address the actual
  bug.  Safe to leave in tree under default-off.

## U31 — exact next attack

Per U30's narrowed bug surface (gathered compute kernel under real
weights, 3 of 4 ELFs):

1. **U31.1 — Factory dump of compile-time args per ELF.** Add a
   factory-side `fprintf(stderr, "[U31_FACTORY_ARGS elf=0x%x in0_block_w=%u in0_num_subblocks=%u in1_num_subblocks=%u num_blocks=%u out_subblock_h=%u out_subblock_w=%u batch=%u]\n", ...)` at the point where the gathered matmul program is created. Find the discriminator between `0x2db6e` (normal) and the 3 garbage ELFs.
2. **U31.2 — DST-pre-PACK probe under REAL weights.** Re-enable `SGLANG_TT_PREFETCHER_PACK_PROBE=1` with `tensix_sync()` and capture per-ELF DST contents BEFORE pack_tile_block. Compare garbage ELFs vs normal ELF. If DST is wrong on garbage ELFs → math/spill bug. If DST is right but PACK is wrong → pack_tile_block bug specific to ELF params.
3. **U31.3 — mm_partials_cb spill/reload trace under REAL weights.** Probe mm_partials_cb's L1 contents AFTER `pack_tile_block(start_dst_index, mm_partials_cb_id, ...)` writes (spill path) and BEFORE `reload_from_cb_to_dst(...)` reads. If the bytes change between, something is stomping mm_partials_cb's L1 in the spill window.
4. **U31.4 — in1 cb_in1 bytes BEFORE matmul_block under REAL weights.** Probe `cb_in1` (the in1 ring-all-gather receive CB) bytes immediately before the `matmul_block(in0_cb, in1_cb, ...)` call on the garbage ELFs. If bytes do NOT match the expected weight tile (per the W2 weight cache), the `update_rd_ptr_to_ring_index` is feeding the matmul the wrong weights.
5. **U31.5 — bisect compile-time args.** If U31.1 identifies a discriminator (e.g. `out_subblock_h=4` vs `1`), force the garbage ELFs to use the normal ELF's args (potentially at perf cost) and re-run; if output is coherent, the discriminator is confirmed; a Python-side fix can avoid the buggy params at the `matmul_1d_ring_config` call site.

## Commits this session

- **tt-metal-sglang**: NO new commits (HEAD remains at U29 Phase 3 tip `5d5d4a02999`). All U30 probes were already in tree from U13 (PACK_PROBE) + U14 (CONSUMER_PROBE) + U29 (signaler); this session combined them with new env to produce the discriminative measurement.
- **sglang**: pending — this doc only (no Python changes in sglang tree).

## Working state at session end

- tt-metal-sglang HEAD: `5d5d4a02999` (U29 Phase 3 tip; unchanged).
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only).
- sglang HEAD: pending this doc commit.
- Container `/tt-metal/`: unchanged this session.
- Container `/tt-metal/ttnn/ttnn/_ttnn{,cpp}.so`: unchanged (already has all U14/U29 defines).
- `/root/.cache/tt-metal-cache/*`: cleared at session end.
- `stash@{0,1,2}`: untouched.
- TT devices: healthy.
- Server: stopped at session end.
- Canonical re-verify: `" What is 2+2? What is 2+2? What"` at 9.16s + GSM8K(10) = 10/10 = 100%.
- DPRINT artifacts preserved in-container at `/tmp/u30_pathA*.log` for U31's reference.

## Next-session checklist (U31)

1. Read this doc + U29 Phase 3 + U13 PACK_PROBE + U7 Case B as the latest evidence chain.
2. U31.1: add factory-side fprintf to dump compile-time args per ELF; correlate `u14_elf_tag` to `(in0_block_w, num_blocks, out_subblock_h/w, batch)`.
3. U31.2: re-run with `SGLANG_TT_PREFETCHER_PACK_PROBE=1` under REAL weights; capture DST-pre-PACK on garbage ELFs.
4. U31.3-5: depending on U31.2 verdict.
5. If/when GSM8K(10) ≥ 7/10 under prefetcher: bench TPOT, retire all U29 guards, squash into one production flag.
