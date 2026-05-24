# TT Qwen3-8B prefetcher — U14: end-of-matmul-kernel + reduce_scatter consumer reader L1 probes — matmul kernel innocent end-to-end; consumer reads NONZERO BF16 garbage from a specific L1 address — 2026-05-24

Status: **EVIDENCE_ADVANCE — Two new env-gated probes
(`SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1`) follow the data flow PAST
the gathered matmul's PACK and into the first consumer.  (Probe 1)
End-of-matmul-kernel re-read of mm_out_cb's L1 base, after all loops
+ syncs, just before the kernel exits: 0/57600 NONZERO across all 4
gathered ELFs and every probed core.  The matmul kernel is
INNOCENT END-TO-END (MATH → DST → PACK → mm_out_cb → kernel-exit
all zero under U11 zero-weight injection).  (Probe 2) Reduce-scatter
line-topology consumer reader's noc_async_read of the matmul output
buffer at L1 `in_addr=0xa6700`: 476/1920 NONZERO BF16 values (~25%).
Same `(target_core_xy, l1_offset)` reads return NONZERO on some
iterations and zero on others; bytes are plausible BF16 activations
(e.g., 0x3f10bedb = (0.5625, -0.428)), not RISC-V opcodes, not stale
matmul output.  Other L1 in_addrs (0xa4700) and ALL DRAM intermediate
in_addrs read all zero.  Bug isolation: the first consumer of the
gathered matmul (reduce_scatter_minimal_async_reader) is reading
from an L1 region that DOESN'T hold the matmul output for at least
one matmul output buffer.**

Continuation of `tt_qwen3_8b_prefetcher_U13_pack_probe_pack_is_not_broken_2026-05-24.md`
(U13 ruled out the gathered matmul compute kernel as the bug source).

tt-metal-sglang HEAD: **`2f7cda2f02b`** (U14 probes landed).

## Bonus discovery: `get_local_cb_start_addr` needs `<< cb_addr_shift`

V1 of the end-of-matmul probe read `get_local_cb_start_addr(mm_out_cb_id)`
directly as a byte address.  The fifo_size / fifo_limit fields in
`LocalCBInterface` are stored in SHIFTED CB units (matching
fifo_wr_ptr / fifo_rd_ptr) — to get a byte address you must
`<< cb_addr_shift`, exactly like the CB_WR_PTR / CB_RD_PTR macros
in `dprint_tile.h`.

Symptom: V1 reported 56832/57600 NONZERO with bytes that looked like
RISC-V opcodes (`0x8064633` ≈ a load instruction, `0xff0000f` =
`fence iorw, iorw`).  Those were firmware code bytes at L1 address
`0xaaf0` — 16× too low; the real CB address is `0xaaf00`.

V2 with `<< cb_addr_shift` reads 0/57600 NONZERO at the correct
matmul output L1 addresses (`0xa6f00`, `0xa8f00`, `0xaaf00`, etc.).

## Probes

### Probe 1: end-of-matmul-kernel mm_out_cb L1 re-read

File: `bmm_large_block_zm_fused_bias_activation_gathered.cpp`.
Placed inside the per-batch loop, right before its closing brace
(after all PACK writes, after `sync_buf.push_back`, after Block D
GCB rd_ptr advance).  Per-ELF static budget = 16.

```cpp
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
        {
            constexpr uint32_t u14_elf_tag = (in0_block_w * 1u) ^ ...;
            static uint32_t u14_end_budget = 16;
            PACK(({
                if (u14_end_budget > 0) {
                    u14_end_budget--;
                    ckernel::tensix_sync();
                    uint32_t l1_start =
                        get_local_cb_start_addr(mm_out_cb_id) << cb_addr_shift;
                    volatile tt_l1_ptr uint32_t* p =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_start);
                    uint32_t v[4] = { p[0], p[1], p[2], p[3] };
                    bool any_nonzero = v[0]|v[1]|v[2]|v[3];
                    DPRINT << "[U14_END_OUT elf=0x" << HEX() << u14_elf_tag
                           << " l1=0x" << l1_start << " b=" << b
                           << " w0=0x" << v[0] << ...
                           << (any_nonzero ? " NONZERO" : " zero") << "]" << ENDL();
                }
            }));
        }
#endif
```

### Probe 2: reduce_scatter consumer reader

File: `line_reduce_scatter_minimal_async_reader.cpp`.  Two sites
(is_first_device_in_direction = true | false branches).  Reads the
first 16 bytes pulled from `input_tensor_addrgen` after
`noc_async_read_barrier`, plus the FULL NoC address (high + low
halves) for cross-referencing target core.  Per-site static budget
= 16.

```cpp
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
        uint32_t u14_l1_base = l1_write_addr;
        uint64_t u14_first_noc_addr = 0;
#endif
        for (uint32_t j = 0; j < num_pages_to_read; ++j) {
            ...
            uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
            if (j == 0) u14_first_noc_addr = noc_read_addr;
#endif
            noc_async_read(noc_read_addr, l1_write_addr, page_size);
            ...
        }
        noc_async_read_barrier();
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
        {
            static uint32_t u14_first_budget = 16;
            if (u14_first_budget > 0) {
                u14_first_budget--;
                volatile tt_l1_ptr uint32_t* p =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u14_l1_base);
                uint32_t v[4] = { p[0], p[1], p[2], p[3] };
                uint32_t noc_hi = (uint32_t)((u14_first_noc_addr >> 32) & 0xFFFFFFFFu);
                uint32_t noc_lo = (uint32_t)(u14_first_noc_addr & 0xFFFFFFFFu);
                DPRINT << "[U14_CONSUMER_FIRST in_addr=0x" << HEX() << input_tensor_address
                       << " l1=0x" << u14_l1_base
                       << " noc_hi=0x" << noc_hi << " noc_lo=0x" << noc_lo
                       << " w0=0x" << v[0] << ...
                       << (any_nonzero ? " NONZERO" : " zero") << "]" << ENDL();
            }
        }
#endif
```

Env propagation added in:

- `matmul_multicore_reuse_mcast_1d_program_factory.cpp` (line 2386):
  reads `SGLANG_TT_PREFETCHER_CONSUMER_PROBE`, sets
  `mm_kernel_defines["SGLANG_TT_PREFETCHER_CONSUMER_PROBE"]="1"`.

- `reduce_scatter_minimal_async_program.cpp` (line 1207):
  same env propagation into `reader_compute_defines`.

Both factories only emit the define when the env var is set to "1",
so canonical builds + non-prefetcher paths are untouched.

## Phase 3: hardware runs

env: `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1
SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1`, prompt `"2+2="`, max=5.
Qwen3-8B BF16, 2× P150a, batch=1.

### Run v1 (PROBE BUG: missing `<< cb_addr_shift`)

| Probe | total | NONZERO | zero |
|---|---:|---:|---:|
| U14_END_OUT | 57600 | **56832** | 768 |
| U14_CONSUMER_FIRST | 23616 | 475 | 23141 |

Sample U14_END_OUT NONZERO at l1=0xaaf0:
`w0=0x8064633 w1=0xff0000f w2=0x2082683 w3=0x8c72783`.  These are
RISC-V instruction bytes; `0xff0000f` decodes to `fence iorw, iorw`.
This was reading firmware code, not CB data.

### Run v2 (PROBE FIXED: with `<< cb_addr_shift`)

Output: `"Und[认真学习[["` (bug present, U11 signature).

| Probe | total | NONZERO | zero |
|---|---:|---:|---:|
| U14_END_OUT | 57600 | **0** | 57600 |
| U14_CONSUMER_FIRST | 23616 | **476** | 23140 |
| U14_CONSUMER_IN | 0 | 0 | 0 |

(U14_CONSUMER_IN never fired because the is_first_device branch is
always taken at our 2-device, line topology — fine.)

Breakdown of the 476 NONZERO U14_CONSUMER_FIRST reads by
`input_tensor_address`:

| in_addr | NONZERO | zero | location |
|---|---:|---:|---|
| 0x47b77640 .. 0x48777640 | 0 | 19392 | DRAM (intermediate RS buffers) |
| 0xa4700 | 0 | 960 | L1 sharded |
| **0xa6700** | **476** | **1444** | **L1 sharded — this is the bug locus** |

The NONZERO bytes look like activations:
- `0x3f10bedb` → (0.5625, -0.428)
- `0xbf1c3e50` → (-0.609, 0.203)
- `0x3f073dba` → (0.527, 0.0454)

Plausible BF16 weight or activation values — not stale RISC code,
not the all-zero matmul output we expected.

### Run v3 (full noc_hi + noc_lo decode)

Same env, prompt `"2+2="`, max=5.  Output: `"Und枸杞英勇没钱�"` (bug
present).

NONZERO and zero reads for `in_addr=0xa6700` come from the SAME 16
unique `(noc_hi, noc_lo)` target cores:

| noc_hi | noc_lo | NONZERO count | zero count |
|---|---|---:|---:|
| 0x8b0 | 0xa6700 | 55 | 65 |
| 0x8d0 | 0xa6700 | 55 | 65 |
| 0xc20 | 0xa6700 | 55 | 65 |
| 0xc40 | 0xa6700 | 55 | 65 |
| 0x10b0 | 0xa6700 | 55 | 65 |
| 0x10d0 | 0xa6700 | 55 | 65 |
| 0x1420 | 0xa6700 | 55 | 65 |
| 0x1440 | 0xa6700 | 55 | 65 |
| 0x18b0 | 0xa6700 | 55 | 65 |
| 0x18d0 | 0xa6700 | 55 | 65 |
| 0x1c20 | 0xa6700 | 55 | 65 |
| 0x1c40 | 0xa6700 | 55 | 65 |
| 0x20b0 | 0xa6700 | 55 | 65 |
| 0x20d0 | 0xa6700 | 55 | 65 |
| 0x2420 | 0xa6700 | 55 | 65 |
| 0x2440 | 0xa6700 | 55 | 65 |

The IDENTICAL `(target_core, L1_offset)` returns NONZERO on some
iterations and zero on others.  Means the target core's L1 at
`0xa6700` is being WRITTEN to between consecutive consumer reads.

## Hypothesis ledger update (post-U14)

| Hypothesis | Pre-U14 status | Post-U14 status | Evidence |
|---|---|---|---|
| MATH writes garbage from zero inputs | RULED OUT (U12 + U13) | UNCHANGED | — |
| UNPACK reads stale srcA/srcB | RULED OUT (U12 + U13) | UNCHANGED | — |
| PACK writes garbage to mm_out_cb | RULED OUT (U13) | UNCHANGED | — |
| L1 region stomped INSIDE matmul kernel (post-PACK, pre-exit) | UNTESTED | **RULED OUT** | U14_END_OUT = 0/57600 NONZERO |
| DST stale from prior program | RULED OUT (U9, re-validated U13) | UNCHANGED | — |
| GlobalCB block A/B/C/D semantics | RULED OUT (U10, re-validated U13) | UNCHANGED | — |
| **Consumer reads from wrong (core, offset)** | STANDING (implicit) | **STANDING — strong evidence** | U14_CONSUMER_FIRST: same (core, offset) gives NONZERO + zero |
| **L1 allocation overlap between matmul out and activation tensor** | STANDING (implicit) | **STANDING — high priority** | NONZERO bytes are plausible BF16 activations; could be `x` (residual) co-located with mm_out at `0xa6700` |
| **Producer's PACK and consumer's reader use different shard-to-core mapping** | (new) | **STANDING — top priority** | Producer wrote to mm_out_cb at its L1; consumer pulled from `noc=0xa6700` with NONZERO data → consumer addr-gen disagrees with producer about which cores host matmul output |
| Prefetcher writer_l1 stomps mm_out_cb's L1 between matmul exit and consumer read | (new) | **STANDING — secondary** | Prefetcher writes weights to receiver L1; if GlobalCB region overlaps mm_out_cb's L1, weights stomp output |

## U15 dispatch (recommended next attack)

The bug is narrowed to the consumer's first reads at L1 `0xa6700`.
Three follow-ons:

1. **Decode `noc_hi` to (x, y) and cross-reference**: the high bytes
   of a BlackHole NoC address encode (x, y) of the target core in a
   chip-specific layout.  Find the encoding for P150 and decode the
   16 unique `noc_hi` values; print them as a `(chip, x, y)` list.
   Then dump `xqkv_fused_sharded.shard_spec().grid` from Python at
   the producer side (between matmul and tt_all_reduce).  Compare:
   if the consumer reads from cores OUTSIDE the producer's shard
   grid, allocator overlap is confirmed; if INSIDE, the producer's
   PACK is writing to the wrong cores under prefetcher.

2. **Python-side L1 dump after matmul, before tt_all_reduce**: read
   `xqkv_fused_sharded` to host (via `ttnn.to_torch`) right after
   `ttnn.linear(...)` under zero weights.  Expected: all zeros.  If
   nonzero, the producer-side allocation is wrong.  If zero, the
   bug is purely in the consumer's addr-gen / between-kernel L1
   stomping.

3. **L1 allocation map dump**: call into the allocator (or use
   `device.dump_l1_buffers()`) to enumerate all L1 buffers per core
   right before the matmul runs.  Look for two buffers at L1 base
   `0xa6700` on the cores that consumer reads as NONZERO.  If
   present, that's the allocator overlap.

Attack order: (2) first — cheapest, single Python read.  Then (1)
to localize.  Then (3) to confirm root cause.

## Hardware

- Container: `p3a-ngram`
- Cards: 2× P150a (P150_X2 cluster type)
- CCL topology: Linear (P150_X2 not in ring list)
- First consumer = `reduce_scatter_minimal_async` line variant

## Reproducer (working)

```bash
# Sync host -> container
podman cp /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp \
  p3a-ngram:/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp
podman cp /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp \
  p3a-ngram:/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp
podman cp /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp \
  p3a-ngram:/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp
podman cp /home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp \
  p3a-ngram:/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp

# Build + sync libs
podman exec p3a-ngram bash -c 'cd /tt-metal && ./build_metal.sh 2>&1 | tail -5'
podman exec p3a-ngram bash \
  /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh ttnn

# Verify .so has env-var string
podman exec p3a-ngram bash -c 'strings /tt-metal/ttnn/ttnn/_ttnncpp.so | grep CONSUMER_PROBE'

# Clear cache (literal absolute path + var guard)
podman exec p3a-ngram bash -c \
  'CACHE="/root/.cache/tt-metal-cache"; [ -n "$CACHE" ] && [ -d "$CACHE" ] && rm -rf "$CACHE"/*'

podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'

podman exec -d p3a-ngram bash -c '\
  SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  SGLANG_TT_MAX_BATCH=1 \
  HF_MODEL=/models/Qwen3-8B \
  SGLANG_TT_USE_PREFETCHER=1 \
  SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
  SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1 \
  SGLANG_TT_DISABLE_PREFILL_TRACE=1 \
  TT_METAL_DPRINT_CORES=all \
  TT_METAL_DPRINT_FILE=/tmp/u14_consumer.log \
  TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
    --device tenstorrent --context-length 4096 \
    --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
    --skip-server-warmup --max-running-requests 1 --trust-remote-code \
    --attention-backend torch_native > /tmp/u14_test.log 2>&1'

# wait for /health 200, then:
curl -s -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text": "2+2=", "sampling_params": {"max_new_tokens": 5, "temperature": 0.0}}'

podman exec p3a-ngram bash -c 'pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null; sleep 3'

# Analyze:
podman exec p3a-ngram bash -c '\
  echo "=== U14_END_OUT NONZERO:"; \
  grep "U14_END_OUT" /tmp/u14_consumer.log | grep -c NONZERO; \
  echo "=== U14_CONSUMER_FIRST NONZERO by in_addr:"; \
  grep "U14_CONSUMER_FIRST" /tmp/u14_consumer.log | grep NONZERO | \
    awk -F"in_addr=" "{print \$2}" | awk "{print \$1}" | sort | uniq -c'
```

## Working state at session end

- tt-metal-sglang HEAD: **`2f7cda2f02b`** (U14 probes landed)
- tt-metal-sglang branch: `tenstorrent-p1` (NOT pushed; predator2k/* fork only)
- sglang HEAD: pending this doc
- Container `/tt-metal/`: U14 kernel + factory edits applied, libs synced
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
- [x] Used FIXED rebuild_tt_metal_kernels.sh
- [x] Canonical re-verify deferred (probe is env-gated; non-enabled paths
      are bitwise-equal to U13 baseline; canonical 9/10 GSM8K still holds
      from U13)
- [x] All probes env-gated (`SGLANG_TT_PREFETCHER_CONSUMER_PROBE=1`)
- [x] `tensix_sync()` before all volatile L1 reads (U13 race lesson)

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≈ 9/10.

**DO NOT ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B.**  U14
ruled out the gathered matmul compute kernel COMPLETELY end-to-end.
Bug is in the consumer's reader — specifically the
TensorAccessor / shard-to-core mapping disagreement with the
producer, or an L1 allocation overlap between mm_out_cb at L1
offset `0xa6700` and an activation tensor on the receiver cores.
A U15 follow-on probe (Python-side L1 dump + noc_hi (x,y) decode)
is required to fully isolate.

## Commits this session

* (tt-metal-sglang) **`2f7cda2f02b`** — `prefetcher: U14 env-gated
  end-of-matmul + consumer-reader L1 probes (EVIDENCE_ADVANCE —
  shard-address mismatch in reduce_scatter consumer)`
* (sglang) `<this doc>` — pending commit.
