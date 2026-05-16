# v141: EAGLE Fast-Verify Hardware Test

## Commit under test
`4a405e9af` — EAGLE fast-verify: 1 decode per cycle instead of dn

## What was done

### Code changes (committed as `94019b691`)
1. **Triton kernel fallbacks** (`tt_eagle_kernels.py`): Added pure-torch
   fallbacks for 7 Triton JIT kernels that crash on TT hardware
   (RuntimeError: 0 active drivers):
   - `assign_draft_cache_locs`
   - `assign_req_to_token_pool` / `assign_req_to_token_pool_func`
   - `create_extend_after_decode_spec_info`
   - `get_target_cache_loc`
   - `filter_finished_cache_loc_kernel`
   - `align_evict_mask_to_page_size`
   - `create_flashinfer_kv_indices_triton`

2. **KV cache copy gate** (`model_runner_kv_cache_mixin.py`): Disabled
   `enable_kv_cache_copy` on tenstorrent/cpu devices where CUDA copy
   streams are unavailable.

### Hardware test attempts
- 7 launch attempts on 2x Blackhole P150a (60GB host RAM)
- Container: p3a-ngram (local-tt-metal:dev image, fork-patched)
- Server args: `--speculative-algorithm EAGLE3 --speculative-num-draft-tokens 2`
  `--speculative-eagle-topk 1 --speculative-num-steps 1`

### Results
| Attempt | Fallback status | Failure point | Error |
|---------|----------------|---------------|-------|
| 1 | No fallbacks | eagle_worker.assign_draft_cache_locs | Triton 0 active drivers |
| 2 | assign_draft_cache_locs only | eagle_info.assign_req_to_token_pool | Triton 0 active drivers |
| 3 | All 7 fallbacks | eagle_info.align_evict_mask_to_page_size | Triton 0 active drivers |
| 4 | All 7 fallbacks + eagle_info patched | FAST VERIFY seen in warmup | TT_FATAL read during trace |
| 5-6 | All fallbacks | Device weight loading | SIGKILL (exit -9) OOM |
| 7 | All fallbacks | SERVER HEALTHY — benchmark ran | Server crashed after 3 runs |

### Benchmark measurements (attempt 7, dn=2)

Warmup:
- warmup 1: 100 tokens in 9.42s = 10.6 tok/s
- warmup 2: 100 tokens in 15.83s = 6.3 tok/s
- warmup 3: 100 tokens in 16.12s = 6.2 tok/s

Measured runs:
- run 1: 100 tokens in 6.97s = **14.3 tok/s**
- run 2: 100 tokens in 8.07s = **12.4 tok/s**
- run 3: 100 tokens in 6.99s = **14.3 tok/s**
- run 4: server crashed (connection reset)

Average: **13.7 tok/s** (3 runs)

### Device-level verify timing (from server log)
```
verify cycle (converged): this=36.8ms avg=42.7ms fast=True dn=2 bs=1
```
Device-level: 1 token per 37ms = ~27 tok/s (matches baseline decode speed).

### Analysis
- **EAGLE overhead is ~50% of wall time.** The TT device verify cycle
  takes 37ms (matching the 27 tok/s no-spec baseline), but the total
  cycle including CPU-side EAGLE draft inference, tree building, token
  dispatch, and Python overhead doubles the per-token latency.
- **accept_length ~1.0** (bonus only). The fast-verify tiling means
  positions 1..dn-1 see approximated logits, so draft tokens are always
  rejected. The optimization's premise (1 decode cycle vs dn=2 decode
  cycles) is correct, but the savings are consumed by EAGLE overhead.
- The fast-verify optimization does reduce verify latency from ~72ms
  (2 decodes) to ~37ms (1 decode), but the EAGLE framework overhead
  (draft model forward, tree construction, verification protocol)
  adds ~30-40ms per cycle.

## Conclusion
- **EAGLE fast-verify does NOT beat the 27 tok/s baseline.**
  Measured 13.7 tok/s avg (3 runs) — 49% slower than baseline.
- **Root cause:** EAGLE CPU-side overhead (draft inference + tree ops +
  protocol dispatch) adds ~30-40ms per token cycle on top of the
  37ms TT device verify step. The fast-verify optimization correctly
  reduces verify from 2 to 1 decode per cycle, but the overhead that
  was invisible on GPU (overlapped with device compute) dominates on
  the CPU-bound TT path.
- **All 7 Triton fallbacks validated and committed** — the EAGLE
  speculative path no longer crashes on missing Triton drivers.
- **Recommendation:** EAGLE spec decoding is not viable on TT hardware
  until the draft model can run on-device (not CPU) or the EAGLE
  protocol overhead is reduced below the decode time (~37ms).
