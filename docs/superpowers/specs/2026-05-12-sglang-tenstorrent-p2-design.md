# SGLang on Tenstorrent — Phase 2 Implementation Spec

> **Date**: 2026-05-12 (amended 2026-05-13)
> **Status**: Draft amended after Tenstorrent plugin discovery; pre-paged-implementation.
> **Scope**: Phase 2 (P2a + P2b sub-phases). Phase 1.5 hygiene patches are a parallel deliverable, included by reference.
> **Predecessor**: [`2026-05-11-sglang-tenstorrent-p1-design.md`](2026-05-11-sglang-tenstorrent-p1-design.md). P1 has shipped (47 commits on `tenstorrent-p1` branch in `predator2k/sglang`; all §9 acceptance gates passed).
> **Anchored on**: [`../brainstorming/2026-05-12-sglang-tenstorrent-p2-brainstorm.md`](../brainstorming/2026-05-12-sglang-tenstorrent-p2-brainstorm.md). The brainstorm framed two directions (P2-paged vs P2-coverage); this spec collapses them into one path after the `generator_sglang.py` discovery.
> **Target hardware**: 2× Tenstorrent Blackhole p150a (validated), mesh-shape parametric to 4× p150a (design-only, not validated in P2).
>
> **2026-05-13 amendment** — Tenstorrent's `tt-inference-server/tt-sglang-plugin` (`tt_llm.py` / `tt_utils.py` / `worker_setup.py` / `patching_model_registry.py`) was discovered. It implements the model-registration pattern in ~500 LoC, already covers 4 model families (Llama / Qwen / Mistral / GptOss) and uses `generator_sglang.py`'s SGLang-flavored API. **This spec amends to ABSORB the plugin's architecture (model-registry path, not custom TTExecutionBackend ABC) and concentrate our independent contribution on RadixAttention + Blackhole + dual-track + §9 acceptance.** Net effect: ~1900 LoC scope cut, ~2 weeks saved, P2a re-estimates to 3 weeks. See §A2 Amendment notes at the end of this document for INV-level deltas. P1 simple backend port (T0.7 already shipped) remains as the dual-track fallback path per G5a / §5.3c.

---

## 0. Executive Summary

Phase 2 transforms P1's single-user contiguous-KV path into a **paged KV + RadixAttention + batched + multi-model** path. The implementation approach is to **absorb Tenstorrent's upstream `tt-sglang-plugin`** (which already integrates `generator_sglang.py`'s SGLang-flavored API) and concentrate our independent value on three areas the plugin has not delivered:

1. **RadixAttention validation** — plugin admits ("Potential improvements (3)") it has not examined SGLang prefix caching; our G4a is the first measurement
2. **Blackhole (p150a) validation** — plugin tested only on Wormhole (n150/n300/t3k); our P2 target is 2× Blackhole p150a
3. **Dual-track + production discipline** — preserving P1 simple-user backend as fallback (G5a / §5.3c), §9 acceptance gates, INV-level documentation, hardware-issue runbooks

P2 is split into two sub-phases:
- **P2a — Plugin Absorption + Paged Baseline** (~3 weeks, was 5 before amendment): port plugin into our namespace (`Tenstorrent*ForCausalLM` per INV-6); wire dual-track switch (env var) preserving simple path; validate RadixAttention; verify Blackhole; full §9 acceptance.
- **P2b — Multi-Model + Long-Context** (~4 weeks, gated on P2a §9 pass): per-model max_seq_len matrix, 128K chunked-prefill on Llama, P300 chunk-size table. (Models register out-of-box from plugin; per-model verification is the work.)

**P1.5 hygiene patches** (SIGQUIT handler, mesh watchdog, persistent cache path doc, P300 chunk-size patch sketch) ship as **4 independent commits in week 1**, all merged before P2a starts.

P2 estimate: 8 weeks total (P1.5 W1 + P2a W2-W4 + P2b W5-W8), sequential gating. Decision gate at end of P2a determines whether and how P2b proceeds.

> **Implementation discovery (2026-05-13)**: fresh tt-metal main (commit `89686ee7`, UMD-bump 2026-05-12) compiled from source resolves the host firmware-19.6.0 ↔ image-UMD-18.10.0 mismatch that blocked earlier hardware verification. Image `localhost/local-tt-metal:dev` (sha256:`973e972bddf5`) — mesh open + tt_transformers import + `generator_sglang.LlamaForCausalLM` import all confirmed working. This is the **third image pin** added to §7 cross-references.

---

## 1. Goals & Non-goals

### 1.1 P2a Goals (must ship)

| # | Goal | Verification |
|---|---|---|
| G1a | **(AMENDED 2026-05-13)** Adopt Tenstorrent plugin's model-registration architecture (`Tenstorrent*ForCausalLM(nn.Module)` registered via SGLang `ModelRegistry`; standard `model_runner.forward(input_ids, positions, forward_batch)` path). P1 simple backend (T0.7 shipped via `_do_*` private pattern) preserved as `tt_transformers_single` fallback for G5a dual-track. | Unit test: importing `TenstorrentLlamaForCausalLM` succeeds + `ModelRegistry.models["LlamaForCausalLM"]` resolves to ours; existing P1 hardware tests still pass against `tt_transformers_single`. |
| G2a | **(AMENDED 2026-05-13)** Plugin's KV-cache flow: `tt_transformers` owns KV bytes via `allocate_kv_cache(num_blocks, num_kv_heads, block_size, head_size)`; SGLang's default allocator owns slot indices; mapping is `block_id = token_index // block_size` (INV-3 preserved). No custom `PagedTokenToKVPoolAllocator` subclass needed. | Confirm: SGLang allocator + plugin page-table math co-index consistently under stress (§9.16 replaced by §9.16r: round-trip slot→block_id→KV-write→read-back consistency). |
| G3a | Paged Llama-3.1-8B inference at **B ≥ 4 concurrent measured, ≤ 32 design ceiling**, 8K context. | Hardware test (§9.3 batched correctness) |
| G4a | **SGLang RadixAttention enabled** with `--enable-radix-cache`. With shared 1K-token system prompt across 4 user questions, hit-rate ≥ 50% AND effective tok/s ≥ 1.5× no-prefix baseline. Plugin upstream has NOT validated this — our G4a is the first measurement. | Hardware test (§9.4) |
| G5a | Dual-track: `SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged` (plugin path, default) / `tt_transformers_single` (P1 fallback, T0.7 simple backend). Switch requires server restart, not hot-swap. | §9.8 |
| G6a | Mesh-shape parametric via `SGLANG_TT_MESH_SHAPE` env (`1x2`, `1x4`). P2a validates `1x2` on real Blackhole hardware; `1x4` is static-lint only. Plugin's `_configure_fabric` already handles `FABRIC_1D` for ≥2 devices. | §9.9 |
| G7a | OOM admission (queue-full → in-stream `AbortReq`, NOT HTTP 503), request cancellation (via `/abort_request` AND client disconnect), paged-block utilization metrics. SGLang scheduler handles admission/cancel out-of-box once plugin is wired; we verify. | §9.5, §9.6, §9.10 |
| G8a | **(NEW 2026-05-13)** **Blackhole p150a validation**: confirm plugin's `tt_utils._configure_fabric` + `ttnn.device.is_blackhole()` branches work on 2× p150a + KMD 2.8.0 + FW 19.6.0. Plugin's tested hardware matrix is Wormhole-only (n150/n300/t3k); P2a is the first Blackhole validation. | §9.1 smoke + §9.3 batched both run on actual Blackhole hardware. |

### 1.2 P2b Goals (gated on P2a §9 pass)

| # | Goal | Verification |
|---|---|---|
| G1b | `TenstorrentQwenForCausalLM` + `TenstorrentMistralForCausalLM` registered in SGLang's model registry under **separate arch namespace** (no collision with SGLang's own `LlamaForCausalLM` / `MistralForCausalLM`). | Per-model smoke + greedy (§9.b1) |
| G2b | `TenstorrentGptOssForCausalLM` registered. Uses the GptOss-specific loader (`models/demos/gpt_oss/tt/common.create_tt_model`), not the shared `tt_transformers` loader. | §9.b1 |
| G3b | **Per-model max_seq_len matrix** (not universal 128K): Llama-3.1-8B 32K, Qwen3-7B 32K, Qwen3-32B 4K (upstream hang), Mistral-7B 32K, GptOss best-effort. | §9.b2 |
| G4b | Chunked-prefill working through paged path on Llama; 128K-input + 1K-output end-to-end pass. | §9.b3 |
| G5b | P300 chunk-size patch (in our fork) populated per-model after empirical tuning. | §9.b4 |

### 1.3 Non-goals

| # | Item | Rationale |
|---|---|---|
| N1 | `TTXLAExecutionBackend` implementation | `tt_transformers` covers all target models; tt-xla deferred to P3+ |
| N2 | vllm adapter (despite `generator_vllm.py` existing upstream) | Out of scope by user instruction |
| N3 | 4× p150a hardware validation | Parametric design only; physical validation deferred |
| N4 | `tt_data_parallel > 1` (data parallelism within mesh) | Interface kwarg reserved, P2 forces `dp=1` |
| N5 | Speculative decoding / draft model | P3+ |
| N6 | LoRA / Multi-LoRA serving | P3+ |
| N7 | BF16-throughout precision (vs BFP8 default) | Known precision drift documented; P2 keeps BFP8 |
| N8 | Multimodal models | `generator_sglang.py` has no multimodal adapter |
| N9 | Upstream PRs to `sgl-project/sglang` or tt-metal | All patches in our fork (`predator2k/sglang`) |
| N10 | G7 hygiene fixes moved out of P2 | Shipped as P1.5 (week 1, 4 independent commits) |
| N11 | **GptOss / Mistral coverage conditional on W0 licensing resolution.** If unresolved at P2b W0 → that model becomes a `placeholder` registration (instantiation raises NotImplementedError), not in §9.b acceptance gates. | License gating is real-world dependency |
| N12 | **Qwen3-32B max_seq_len > 4K** | Upstream `tt_transformers/tt/model_config.py` caps at 4096 due to documented hang |

### 1.4 Dropped from P1 spec §11

- **P2-coverage / TTXLAExecutionBackend** — dropped per N1
- **Plan Q (write our own ttnn Llama)** — dropped; `tt_transformers` proved sufficient in P1

---

## 2. Architecture (AMENDED 2026-05-13)

### 2.1 Incremental map P1 → P2 (plugin-based)

```
                      [ apply_server_args_defaults ]
                                |
                                v
                    [ TTSRTPlatform.activate() ]
                                |
                                v
       [ MeshDeviceCtx(shape=env or (1,2)) ]   ← P1.5: SIGQUIT handler
                                |              ← P2:   mesh-shape parametric
                                v
        ┌───────────────────────┴───────────────────────┐
        │                                               │
        │ SGLANG_TT_EXECUTION_BACKEND=                  │
        │   tt_transformers_single                       │   tt_transformers_paged (default)
        │                                               │
        v                                               v
[ TTTpModelWorker (P1, B=1) ]              [ Standard SGLang ModelRunner ]
        │                                               │
        v                                               v
[ TTExecutionBackend ABC                  [ ModelRunner.model = TenstorrentLlamaForCausalLM ]
   _do_new_request / _do_extend /            (registered via SGLang ModelRegistry; INV-6)
   _do_decode_step / _do_reset_all ]         │
        │                                    │   forward(input_ids, positions, forward_batch)
        v                                    v
[ Generator.prefill_forward_text /        [ tt_model.prefill_forward / decode_forward
   decode_forward ]                          (LlamaForCausalLM from generator_sglang.py) ]
        │                                    │
        v                                    v
[ tt_transformers (B=1, no RadixAttn) ]  [ tt_transformers (paged + Radix-capable) ]
```

**Two paths co-exist** via env var (G5a, INV-5). Left = P1 simple (T0.7 shipped). Right = plugin-based paged. **Plugin path is the default; simple path is the fallback** per §5.3c.

### 2.2 New modules in P2a (AMENDED 2026-05-13 — ~500 LoC, down from ~1900)

All paths are relative to `python/sglang/srt/hardware_backend/tenstorrent/`.

| Path | Responsibility | Size |
|---|---|---|
| `models/__init__.py` | Plugin registration hook (calls SGLang `ModelRegistry.models.update(TT_MODEL_REGISTRY)` on import); namespace marker for INV-6. | ~30 |
| `models/tt_llm.py` | **Adapted from Tenstorrent's `tt-sglang-plugin/sglang_tt_plugin/models/tt_llm.py`**. `TenstorrentLlamaForCausalLM` etc. (4 model families). Same forward logic (`_build_page_table`, `_flatten_to_padded`, `_pad_decode_batch`) and `allocate_on_device`. Class renamed per INV-6. | ~300 |
| `models/worker_setup.py` | **Adapted from plugin's `worker_setup/worker_setup.py`**. Worker process env setup (`TT_METAL_CACHE`, `TT_CACHE_PATH`, `_DP\d+` parsing). | ~80 |
| `models/tt_utils.py` | **Adapted from plugin's `utils/tt_utils.py`**. `BaseMetalDeviceRunner` for mesh device open + fabric config. Includes Blackhole branch (`ttnn.device.is_blackhole()`). | ~150 |
| `platform.py` extension | Wire `TenstorrentLlamaForCausalLM` model registration on platform activate; mesh-shape parametric. | +30 |
| **Preserved from T0.6/T0.7** (P1 simple path, dual-track fallback): | |
| `execution/base.py` | TTExecutionBackend ABC (`_do_*` private methods); now scoped to `tt_transformers_single` only. INV-1 amended to scope to simple path. | (shipped) |
| `execution/tt_transformers_backend.py` | P1 simple backend + new ABC wiring (T0.7 shipped). Used only when env var = `tt_transformers_single`. | (shipped) |
| **P2a NEW total** | | **~590 LoC new + 4 files preserved as simple fallback** |

**Cancelled from pre-amendment plan**:
- ~~`execution/tt_transformers_paged_backend.py`~~ — plugin's `TTModels.forward()` replaces this
- ~~`kv_pool/paged.py` `TTPagedKVAdapter` + `TTBackedKVCache`~~ — plugin uses tt_transformers' own `allocate_kv_cache`, SGLang's default allocator owns indices (INV-2 amended)
- ~~`attn_meta/tt_paged.py` `TTPagedMetadataBackend`~~ — page-table math inlined in plugin's `_build_page_table`
- ~~`tp_worker.py` paged extensions~~ — plugin uses standard `ModelRunner.forward()`; only the simple-backend worker remains specialized

**Cross-module patches (still in our fork only, per N9)**:
- `python/sglang/srt/managers/scheduler.py` / `utils.py` for `GenerationBatchResult.bypass_chunked_req` — still required for chunked-prefill failure recovery (§3.7). Plugin doesn't address chunked-prefill failure; this is OUR contribution on top.

### 2.3 New modules in P2b (≈ 1500 lines)

| Path | Responsibility | Size |
|---|---|---|
| `models/qwen_tt.py`, `models/mistral_tt.py` | Two more arch wrappers, same pattern as `llama_tt.py` | ~150 each |
| `models/gpt_oss_tt.py` | GptOss wrapper using the separate `models/demos/gpt_oss/tt/common.create_tt_model` loader (distinct from `generator_sglang.py`) | ~200 |
| Chunked-prefill enablement in paged backend | Hook into SGLang's chunk segmentation; wire to `tt_transformers` paged forward per chunk | ~300 |
| `p300_chunk_table.py` (our fork patch) | Per-model chunk-size table populated empirically | ~150 |
| `models/__init__.py` registration | Arch namespace setup (see INV-6) | ~50 |
| Test scaffolding (per-model fixtures, 128K stress, chunked-prefill) | | ~450 |
| **P2b total** | | **~1500** |

### 2.4 Architecture invariants (referenced by §5.2b escape hatch)

These are explicitly numbered so that an implementation discovery violating any of them triggers a brainstorming-skill replay (§5.2b), not an ad-hoc plan-level edit.

| ID | Invariant |
|---|---|
| **INV-1** | `TTExecutionBackend.forward(forward_batch) → LogitsProcessorOutput` is **model-level** forward (not per-layer attention). Naming MUST NOT alias to SGLang's `AttentionBackend` class — that's per-layer. |
| **INV-2** | `TTPagedKVAdapter` subclasses `PagedTokenToKVPoolAllocator` with `device="cpu"` and a custom `TTBackedKVCache` wrapper. `alloc_extend` / `alloc_decode` are rewritten in pure CPU torch (SGLang's mainline uses Triton kernels). The adapter does NOT subclass `MHATokenToKVPool` (which requires CUDA-tensor `.data_ptr()`). |
| **INV-3** | `page_table` tensor values are **block IDs** in `[0, num_pages)`, computed as `token_index // block_size`. dtype must be `torch.int32`. |
| **INV-4** | SGLang owns the page table. `tt_transformers` receives KV cache tensors and page table as parameters; `PagedAttentionConfig` path inside `tt_transformers` (which allocates internally) is NOT used. |
| **INV-5** | Two execution backends co-exist via `SGLANG_TT_EXECUTION_BACKEND`. Switching requires server restart (mesh re-init). Hot-swap is not supported. |
| **INV-6** | Model arches are registered under `Tenstorrent*` namespace (e.g. `TenstorrentLlamaForCausalLM`), NOT colliding with SGLang's own `LlamaForCausalLM`. SGLang `ModelRegistry.register` raises on duplicates. |
| **INV-7** | `MeshDeviceCtx` accepts mesh shapes `(1,2)` and `(1,4)` parametrically. P2 validates only `(1,2)`; `(1,4)` is static-lint only. |

### 2.5 SGLang integration points

1. **Model registry**: `tenstorrent/models/__init__.py` calls `ModelRegistry.register(arch_string, cls)` for each `Tenstorrent*ForCausalLM`. Architecture matching follows SGLang standard flow.
2. **KV pool wiring**: `TTTpModelWorker._init_model_runner` constructs `TTPagedKVAdapter` and injects it as `self.token_to_kv_pool_allocator`. Scheduler's admission control queries `available_size()` on this adapter.
3. **Attention metadata wiring**: `TTModelRunner` exposes `self.attn_backend = TTPagedMetadataBackend(...)`. SGLang scheduler calls `init_forward_metadata(forward_batch)` per forward; our metadata backend builds the ttnn page-table.
4. **RadixAttention**: Enabled via `--enable-radix-cache` (SGLang default). RadixCache calls `allocator.alloc()` and `allocator.free()` — those go through our adapter's free list. `tt_transformers` is unaware of RadixAttention; it only sees the page table.

   **Key distinction from vllm-style block-aligned prefix caches**: SGLang's scheduler and RadixAttention are NOT vllm-equivalents — the spec must be implemented against SGLang's APIs only, not by analogy to vllm patterns.
   - **Prefix granularity**: RadixAttention matches prefixes at **token-level** granularity within pages. A prefix may end mid-block (e.g., 30 tokens of a 32-token block). vllm's hashed prefix cache only matches whole blocks.
   - **Shared-page write safety**: When two requests share a 30-token prefix in block_size=32 layout, both `req_to_token` rows reference the same flat token indices for `[0, 30)`. Positions `[30, 31]` of that physical page are then filled independently by each request's `alloc_extend` (which uses `last_loc` to continue within a partial block before allocating a fresh page). tt_transformers' `paged_fill_cache` writes at `(start_pos + offset_in_chunk)` granularity within pages, not block boundaries — partial-block writes are supported.
   - **Page-table construction (§3.2)**: `row[::block_size]` reads only the first-token index of each `block_size`-strided position, which always maps to a valid page ID regardless of whether the prefix ends on a block boundary. The ceil-divide for the page-table row length covers the trailing partial block.
   - **Tree-node sharing**: RadixCache may keep finished-request tokens as a tree node without freeing the slots, so adapter free-list size ≠ "unique KV bytes" — it's the SGLang-side bookkeeping view only. See R5 mitigation for the invariants enforced via §9.11/§9.15.

---

## 3. Data Flow

### 3.1 Pipeline

```
HTTP /v1/completions
        │
        ▼
[ Tokenizer → SGLang Scheduler ]
   - RadixCache.match_prefix → matched_kv_indices, matched_token_count
   - token_to_kv_pool_allocator.available_size → admission via PrefillAdder
   - alloc_extend / alloc_decode → out_cache_loc[total_new_tokens]
        │
        ▼
[ ScheduleBatch → ForwardBatch ]
   input_ids (new tokens only, flat across batch), req_pool_indices,
   extend_seq_lens (FULL seq lens, includes prefix),
   extend_prefix_lens (prefix-cache hit counts),
   out_cache_loc, seq_lens, forward_mode
        │
        ▼
[ TTTpModelWorker.forward_batch_generation ]
        │
        ▼
[ TTExecutionBackend.init_forward_metadata(fb) ]    ← TTPagedMetadataBackend
   - Reads req_to_token_pool.req_to_token[req_pool_idx, :seq_lens] for full
     token-pool index sequence per req
   - page_table[i, :ceil(seq_len[i]/block_size)] = (row[::block_size] // block_size).to(int32)
   - Uploads page_table to ttnn (host→device, once per forward)
        │
        ▼
[ TTExecutionBackend.forward(fb) ]                   ← model-level forward
   - EXTEND: build full `tokens[max_batch, padded_prefill_len]` per
     `prefill_forward_text` requirement (tokens includes prefix; start_pos
     marks where prefix ends); call generator.prefill_forward_text(
         tokens=tokens, page_table=page_table, kv_cache=adapter.kv_tensors,
         prompt_lens=full_seq_lens, start_pos=prefix_lens, empty_slots=list(range(B)),
     )
   - DECODE: append one slot to each req's page_table tail; call
     generator.decode_forward_text(input_ids=last_tokens[B], page_table=...,
         kv_cache=...)
   - Returns logits[B, vocab] on host CPU
        │
        ▼
[ Greedy / top-k sampling (host CPU, P1-compatible path) ]
        │
        ▼
   next_token_ids → back to Scheduler → RadixCache.cache_finished_req
```

### 3.2 Prefill (EXTEND mode)

**Terminology**:
- `extend_seq_lens[i]` = **full** sequence length for req `i` (includes cached prefix)
- `extend_prefix_lens[i]` = cached prefix length (from RadixCache hit)
- `extend_num_tokens` = `sum(extend_seq_lens - extend_prefix_lens)` = total new tokens to compute
- `ForwardBatch.input_ids` in EXTEND mode contains **only new tokens**, flattened across the batch

**`tt_transformers.prefill_forward_text` signature** (verified in `tt_transformers/tt/generator.py:451`):
- `tokens`: full prompt INCLUDING cached prefix (NOT just new tokens), shape `[max_batch_size, padded_prefill_len]`
- `prompt_lens`: list/tensor of full prompt lengths per slot
- `start_pos`: **Python list of int** (not tensor) — cached prefix lengths per slot
- `empty_slots`: index list of slot IDs in `[0, max_batch_size)`; for P2a B≤4 with no retract, this is `list(range(B))` (identity mapping)
- `page_table`: `torch.int32[max_batch_size, max_blocks_per_seq]` of block IDs

**Per-forward construction**:

```python
# Build `tokens` from req.fill_ids (full sequence, padded per row)
padded_len = round_up(max(prompt_lens), pad_step)
tokens = torch.full((max_batch_size, padded_len), pad_id, dtype=torch.int32)
for i in range(B):
    s = extend_prefix_lens[i]
    e = extend_seq_lens[i]
    tokens[i, s:e] = req.fill_ids[s:e]   # only new tokens populated; [0:s) area irrelevant per start_pos
```

### 3.3 Decode (DECODE mode)

For each active req, scheduler appends 1 new token slot. Workflow:

1. `allocator.alloc_decode(seq_lens, last_loc)` → `out_cache_loc[B]`
2. `init_forward_metadata` extends each req's page_table tail
3. `generator.decode_forward_text(input_ids=last_tokens[B], page_table, kv_cache, ...)` → logits `[B, vocab]`

### 3.4 Free / Eviction

RadixCache owns the lifecycle. `TTPagedKVAdapter` is **read-through**:

- `cache_finished_req(req)` → may insert as cache node (no free) OR evict via `allocator.free(token_indices)`
- `allocator.free(token_indices)` → `TTPagedKVAdapter._add_to_free_list(token_indices)`
- Does NOT zero ttnn KV (next allocator-write overwrites the tile)
- Does NOT touch `page_table` (next forward rewrites)

**Invariant**: RadixCache's view of KV state = adapter's free list state. No cache sync.

### 3.5 OOM Admission

```python
# SGLang's existing PrefillAdder calls these:
adapter.available_size() -> int      # CPU-side bookkeeping
adapter.alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
                     last_loc, extend_num_tokens, num_new_pages=None) -> out_indices or None
# If alloc_extend returns None, SGLang abort path engages:
#   - in-stream AbortReq with abort_message="The request queue is full." (NOT HTTP 503)
#   - 503 is health-check-only in SGLang (http_server.py:518)
```

We do NOT implement an admission gate. SGLang's `PrefillAdder` already handles it.

### 3.6 Cancel mid-prefill

P2a minimal:
1. Client disconnects (TCP RST) OR `/abort_request` POST → Scheduler marks `req.canceled = True`
2. Current forward call runs to completion (no ttnn interruption)
3. Worker checks `req.canceled` after forward; calls `req.set_finish_with_abort("client_canceled")`
4. `cache_finished_req(req)` evicts; `page_table` slots become reusable

P2b stretch: chunk-boundary cancel (~4K-token granularity) on chunked-prefill.

### 3.7 Error recovery

**tt_transformers raises inside `forward`**:
```python
try:
    logits = generator.prefill_forward_text(...)
except Exception as exc:
    for req in fb.reqs:
        req.set_finish_with_abort(f"tt_backend_error: {exc!r}")
    return LogitsProcessorOutput(next_token_logits=None)
# NOTE: req.set_finish_with_abort does NOT short-circuit the CURRENT step's
# sampling (Q2 — verify in implementation P2a.1). If it doesn't, return
# zero-logits instead so sampler produces a valid (if meaningless) next_token
# that's immediately discarded by the abort path on the next iter.
```

**Chunked-prefill mid-chunk failure** — adapter performs all 5 steps:
```python
def on_chunked_prefill_failure(req, out_cache_loc_this_chunk):
    # 1. Return KV slots from this chunk
    self.allocator.free(out_cache_loc_this_chunk)
    # 2. Revert req_to_token writes for this chunk
    self.req_to_token_pool.req_to_token[
        req.req_pool_idx,
        len(req.prefix_indices) + len(req.fill_ids) - len(out_cache_loc_this_chunk):
        len(req.prefix_indices) + len(req.fill_ids),
    ] = 0   # 0 is the unallocated sentinel
    # 3. Prevent partial-prefix insertion into RadixCache
    req.skip_radix_cache_insert = True
    # 4. Mark abort
    req.set_finish_with_abort("tt_backend_chunked_prefill_failure")
    # 5. Surface to scheduler via GenerationBatchResult flag
    return GenerationBatchResult(..., bypass_chunked_req=True)
# Scheduler observes bypass_chunked_req and clears its own chunked_req state.
# GenerationBatchResult.bypass_chunked_req is a NEW field added in our fork
# (Risk R6: requires upstream-class patch, rebased monthly).
```

**ttnn C++ abort (SIGABRT)** → MeshDeviceCtx atexit + SIGQUIT handler (P1.5) cleans up → process exits → scheduler parent restarts worker.

### 3.8 Implementation-discovery items (Q1-Q9)

Eight signatures and semantics are best verified during implementation P2a.0–P2a.3, not pre-baked into the spec. See §5.2.

---

## 4. Tests + Acceptance Gates

### 4.1 Test pyramid

```
   Hardware-gated (live mesh + SGLang server, SGLANG_PLATFORM=tenstorrent)
   ─────────────────────────────────────────────────────────────────────
   smoke / greedy / batched / radix prefix-cache / OOM admission /
   abort flow / stability / dual-track / mesh-shape / perf-log
                    ~12 tests per phase
                                  │
   Integration (CPU, mocked mesh + mocked backend)
   ──────────────────────────────────────────
   ForwardBatch end-to-end through TTTpModelWorker
   Adapter ↔ allocator ↔ RadixCache contract
   Page-table translation accuracy
                    ~8 tests
                                  │
   Unit (CPU, no mesh, no SGLang server)
   ────────────────────────────────────────
   ABC contract (forward / init_metadata / free)
   TTPagedKVAdapter alloc_extend/decode CPU implementation
   Page-table block-id math (// block_size)
   FINISH_ABORT + skip_radix_cache_insert flow
   Chunked-prefill failure invariant (5-step)
   Model registry namespace isolation
                    ~30 tests
```

### 4.2 P1 test migration

| P1 existing | P2a treatment |
|---|---|
| 31 unit tests (`test_wrapper_lifecycle`, `test_execution_backend_registry`, `test_forward_mode_guard`, `test_platform_activate`, etc.) | Signatures updated to new model-level ABC; test structure preserved |
| 13 hardware-gated tests (`test_smoke`, `test_greedy_correctness`, `test_mmlu_mini`, `test_stability`, `test_perf_log`, `test_p1_acceptance`) | **Two copies**: one targeting `tt_transformers_single` (P1 fallback regression net), one targeting `tt_transformers_paged` (P2 default). Tagged with `@pytest.mark.simple_backend` / `@pytest.mark.paged_backend` for selective CI |

### 4.3 P2a §9 Acceptance Gates

| # | Gate | Test file |
|---|---|---|
| §9.1 | Paged smoke (B=1) returns "Paris" via `/v1/completions` | `test_smoke_paged.py` |
| §9.2 | Greedy correctness on paged path; per-prompt isolated re-run matches batched run (NOT bit-exact across runs — BFP8 drift) | `test_greedy_correctness_paged.py` |
| §9.3 | Batched correctness B=4: 4 concurrent prompts; each prompt's tokens (deterministic temp=0) match its isolated-run output | `test_batched_correctness.py` |
| §9.4 | RadixAttention prefix-cache: shared 1K-token system prompt + 4 distinct user questions. Prometheus `sglang:cache_hit_rate ≥ 0.5`, effective tok/s ratio (warm/cold) ≥ 1.5 | `test_radix_prefix_cache.py` |
| §9.5 | Queue-full admission: with `--max-queued-requests N`, submit `N+1` long requests. The `N+1`th receives in-stream AbortReq with `meta_info.finish_reason.type == "abort"` AND `abort_message` mentions "queue" | `test_queue_full_admission.py` |
| §9.6 | Abort flow: two tests — `/abort_request` POST clears KV (`/get_internal_state.available_size` returns to baseline); client TCP RST also clears | `test_abort_via_endpoint.py`, `test_abort_via_disconnect.py` |
| §9.7 | Stability (compressed 15-min, B=4): ITL p99 windowed (baseline 5-10min vs tail 5min in SAME run), drift < 10%. Baseline is intra-run, NOT vs B=1. | `test_stability_paged.py` |
| §9.8 | Dual-track switch: two server-launch fixtures (one per env var value), each runs its full suite. Restart between, no hot-swap. | `test_dual_track_switch.py` |
| §9.9 | Mesh-shape parametric: `SGLANG_TT_MESH_SHAPE=1x2` runs on hardware; `1x4` passes static-lint (no real hardware required) | `test_mesh_shape_param.py` |
| §9.10 | Perf log records: batched tok/s @ B={1,2,4}, prefix-cache hit/miss latency, per-mode timing, KV-pool utilization. Sanity floors: batched B=4 ≥ 1.2× B=1 decode tok/s; prefix-hit lookup < 10ms; KV-pool steady-state < 95% | `test_perf_log_paged.py` |
| §9.11 | Eviction-replay: load prefix → fill RadixCache to force evict that prefix → re-issue → logits stable | `test_eviction_replay.py` |
| §9.12 | Chunked-prefill failure recovery — 5 sub-tests, all must pass: | `test_chunked_failure_recovery.py` |
| §9.12.a |  ↳ `allocator.free` called with exact slot list of failed chunk | |
| §9.12.b |  ↳ `req_to_token[req_pool_idx, slice]` zeroed in the failed-chunk range | |
| §9.12.c |  ↳ `req.skip_radix_cache_insert == True` | |
| §9.12.d |  ↳ `req.finished_reason` is `FINISH_ABORT` instance | |
| §9.12.e |  ↳ `GenerationBatchResult.bypass_chunked_req == True` and scheduler clears `self.chunked_req` | |
| §9.13 | Shutdown teardown: `ttnn.get_num_tensors() == 0` after server shutdown | `test_shutdown_teardown.py` |
| §9.14 | Token-pool index static bound: assert at startup that `num_pages × block_size < 2^31`, fail-fast otherwise | `test_token_pool_overflow.py` |
| §9.15 | **Free-list invariant** unit test (covers R5 mitigation): after any sequence of `alloc_extend`/`alloc_decode`/`free` calls, `len(free_pages) + len(allocated_pages) == capacity`. Synchronous post-condition check, no hardware needed. | `test_free_list_invariant.py` |
| §9.16 | **G2a alloc/free/page-table math** unit tests (covers G2a verification): `alloc_extend` returns right count of slots; `free` restores free list; `(token_idx // block_size).to(int32)` matches expected page IDs | `test_paged_kv_adapter_math.py` |

### 4.4 P2b §9.b Acceptance Gates

| # | Gate |
|---|---|
| §9.b1 | Qwen3-7B / Mistral-7B / GptOss-20B (if W0 license resolved) — smoke + greedy per model |
| §9.b2 | Per-model max_seq_len matrix: Llama 32K / Qwen3-7B 32K / Qwen3-32B 4K / Mistral-7B 32K / GptOss best-effort |
| §9.b3 | Chunked-prefill on Llama-3.1-8B: 128K input + 1K output end-to-end |
| §9.b4 | P300 chunk-size table covers all in-scope model families (Llama + Qwen + Mistral; GptOss conditional) |
| §9.b5 | HBM fragmentation: 24h+ batched (B=4), KV pool free blocks ≥ 95% of initial. Sampled every 30 minutes; final sample after 5-minute idle drain. |
| §9.b6 | Model registry collision test: after a hypothetical SGLang upgrade adding new arch names, registration still succeeds (covers risk R14) |

### 4.5 Perf log additions

Beyond P1 (cold/warm prefill, warm decode):
- Batched tok/s @ B={1, 2, 4}
- Prefix-cache hit/miss latency split
- KV-pool utilization time series
- Per-mode timing breakdown (`init_forward_metadata` / `forward` / `sample`)
- Chunked-prefill chunk durations (P2b)

### 4.6 P1.5 hygiene tests (independent of P2)

| # | Patch | Test |
|---|---|---|
| P1.5-1 | SIGQUIT handler in MeshDeviceCtx | `test_sigquit_cleanup.py` — subprocess + signal (not pytest fork) |
| P1.5-2 | P300 chunk-size patch sketch | static lint |
| P1.5-3 | Mesh watchdog (< 5s ready check) | `test_mesh_watchdog.py` |
| P1.5-4 | Persistent kernel cache path documented | doc-only, no test |

---

## 5. Risks / Open Questions / Migration

### 5.1 Risks (grouped by category)

#### Category A — Allocator / Host-side

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R2 | CPU-torch `alloc_extend` latency > 1ms @ B=4 | MEDIUM | **2-step ladder**: (1) vectorize CPU torch implementation, (2) accept latency and force `B≤4` cap (G3a measured-floor still met; design-ceiling suspended pending P3). Triton-CPU port / C++ extension explicitly OUT of P2 scope. |
| R9 | 8 implementation-discovery items (Q1-Q9) surprise | MEDIUM | spec labels items explicitly; Plan reserves dedicated P2a.0 "signature lock" tasks |

#### Category B — Cache coherence / paged-evict

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R5 | RadixAttention vs paged-KV evict path inconsistency or deadlock | **HIGH** | §9.11 eviction-replay (end-to-end) + 24h long-run (§9.b5) + free-list invariant unit test (post-condition check). Three orthogonal verifications. Q7 explicitly tracks this; HIGH-severity item with an unanswered Q must be answered in P2a. |
| R11 | HBM fragmentation under sustained B=4 over 24h | MEDIUM | §9.b5 gate: sampled every 30min, final sample after 5-min idle drain; free_blocks/capacity_initial ≥ 0.95 |
| R12 | `_chunked_req_scheduled_last_iter` ↔ `bypass_chunked_req` interaction | MEDIUM | §9.12 covers; Q9 answer validates; spec invariant in §3.7 explicit |

#### Category C1 — License / Legal

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R10 | Qwen / Mistral / GptOss weight licensing or download access | per N11 | P2b W0 explicit licensing-resolution task. Unresolved models become `placeholder` registration (NotImplementedError on instantiation), NOT in §9.b acceptance. |

#### Category C2 — Upstream / Dependency drift

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | `generator_sglang.py` signature drift on tt-metal upgrade | HIGH | P2a.0 signature lock + adapter version check + **pin tt-metal docker image SHA** (not just upstream commit) at implementation start |
| R6 | `GenerationBatchResult.bypass_chunked_req` upstream-class patch | MEDIUM | Monthly rebase task (quarterly is contingency floor). If conflict on SGLang bump: temporarily disable chunked-prefill, rewrite patch. |
| R13 | tt-metal docker image drift | NOTE | Pin docker image SHA + version comment in Dockerfile / launch script |
| R14 | Model registry collision on SGLang upgrade | NOTE | §9.b6 registration-conflict unit test |

#### Category D — Process / merge / time

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R3 | `empty_slots` non-identity after retract | MEDIUM | P2a enforces B≤4 (no retract triggered); P2b W1 explicit retract test |
| R4 | Per-model max_seq_len surprises in P2b | MEDIUM | "Empirically determined" matrix; 1-week buffer in W7-W10 |
| R7 | 4× p150a parametric design unvalidated bug | LOW | Spec explicitly says 4× not validated; future expansion doesn't trust P2 code |
| R8 | Dual-track maintenance cost > expected | LOW | Review 4 weeks after P2a ship; if simple-backend unused, delete |

### 5.2 Open Questions (verified during implementation)

| # | Question | Best guess | When verified |
|---|---|---|---|
| Q1 | `prefill_forward_text` real batch tokens shape — `[32, max_pad]` or `[B, per_user_pad]`? | `[max_batch_size, max_pad]` | P2a.0 |
| Q2 | `req.set_finish_with_abort` short-circuits same-iter sampling? | No (next iter only) | P2a.1 |
| Q3 | `empty_slots` is identity at B=4 (no retract)? | Yes | P2a.2 (batched test) |
| Q4 | CPU-torch `alloc_extend` p99 latency at B=4? | < 500us | P2a.3 (micro-benchmark) |
| Q5 | `paged_attention_config` must be passed to `create_tt_model`? | No (page table fed externally per INV-4) | P2a.0 |
| Q6 | `req_to_token[:slice]` dtype requires `.to(int32)` for ttnn? | Yes | P2a.1 |
| Q7 | RadixCache `cache_finished_req(canceled=True)` ↔ adapter free-list consistent? | Yes (read-through) | P2b (abort-flow test) |
| Q8 | GptOss separate-loader coupling depth with paged path? | Unknown | P2b W0 |
| Q9 | `TTBackedKVCache` shim covers all SGLang KV call-sites in {attention-backend, allocator, radix_cache} modules? | Yes — to verify | P2a.0 (Plan deliverable: call-site CSV) |

#### 5.2b Implementation-discovery escape hatch

If any Q1-Q9 answer **violates** an INV-1 through INV-7 invariant from §2.4, implementation MUST pause and re-enter brainstorming-skill replay scope:

- Violations of **INV-1** (ABC shape) or **INV-3** (page-table value semantics) → full §2 + §3 replay
- Violations of **INV-2, INV-4, INV-5, INV-6, INV-7** → replay the §2 + §3 paragraphs touching that specific invariant

Implementation MUST NOT silently mutate §2 architecture to accommodate a discovery — that's spec-level, not plan-level.

### 5.3 Migration timeline (AMENDED 2026-05-13 — was 10 weeks, now 4)

```
Week  : 1   2   3   4
P1.5  : ████
P2a   :     ████████          (~2 weeks: plugin port + Blackhole verify + RadixAttention probe + §9-min acceptance)
P2b   :              ████      (~1 week: per-model max_seq_len matrix + Llama 32K stress; NO 128K chunked-prefill — pushed to P3)
P3    :                    ████ (P3 starts here — see §A2 for scope)
```

> **Acceleration rationale**: plugin absorption removed ~1900 LoC of planned implementation work (entire Stage-1 of original plan deleted: TTPagedKVAdapter, TTBackedKVCache, TTPagedMetadataBackend). Plugin also pre-solves multi-model (G1b/G2b) and mesh-shape parametric (G6a). What's left in P2 is verification + dual-track + RadixAttention probe — work that fits in ~3 weeks total (P2a + P2b).

> **Banner assumes Blackhole + plugin work**: if plugin's untested Blackhole branches in `tt_utils.py` need patching, P2a slips by 1 week (still under 5w original budget). If RadixAttention is dead-on-arrival with plugin's KV layout, G4a downgrades to "documented infeasibility, deferred to P3" rather than blocking P2a.

**P1.5 merge order**: P1.5 must be **fully merged** (all 4 logical changes landed on `main`, regardless of squash topology) before P2a branches off the post-merge SHA. P2a MUST NOT start while any P1.5 change is in-flight (avoids 4-way rebase race).

**P2a (W2-W6)**:
- W2: P2a.1 plugin port — Q1, Q5, Q6, Q2, Q9 signature lock + Q9 call-site CSV deliverable. ABC rewrite. P1 simple backend port.
- W3: `TTPagedKVAdapter` + `TTBackedKVCache` + page-table translation.
- W4: RadixAttention integration + batched forward + admission + abort.
- W5: Test suite + §9.1-§9.14 gates.
- W6: Buffer / catch-up / decision gate evaluation.

**P2b (W7-W10)**:
- W7: Qwen + Mistral registration + per-model max_seq_len matrix + licensing resolution (gate W0 task).
- W8: GptOss separate-loader integration.
- W9: Chunked-prefill + 128K Llama validation.
- W10: P300 chunk-size full table + §9.b acceptance.

#### 5.3a User-visible transitions

| Event | User impact |
|---|---|
| P1.5 merged | Transparent (cleanup-only patches) |
| P2a merged, default flips to `tt_transformers_paged` | `max_running_requests` 1 → 4; prefix cache auto-enabled; per-request latency may shift ±10% |
| P2b merged | Qwen / Mistral / GptOss selectable via `--model-path`; Llama supports up to 128K context |

#### 5.3b P2a → P2b decision gate (quantified)

P2b starts only if **all** of:

1. §9.1-§9.14 pass.
2. ≥ 7 of Q1-Q9 have explicit answers (≤ 2 may remain "deferred").
3. R5 (RadixAttention deadlock) shows no reproduction in 24h stability run.
4. No §9 gate has failed ≥ 2 times.
5. Q8 (GptOss coupling) is answered; if coupling depth invalidates P2b W0 budget → P2b is descoped to (Qwen + Mistral + Llama-128K) only, GptOss → placeholder.

**Failure handling**:
- First W6-gate failure → 2-week fix loop (W7-W8 converted to repair work, P2b not started).
- Second failure (W9 re-test) → P2b downgraded to **best-effort**: §1 G1b/G2b/G3b/G4b/G5b drop out of acceptance, ship as "best-effort" placeholder for P3 re-spec.
- No third attempt. P2b is not allowed to be open-ended.

#### 5.3c Rollback paths

| Scenario | Rollback |
|---|---|
| P2a paged path has bug, need fast cutover | `SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single` — reverts to **single-user contiguous-KV semantics** (the rewritten simple backend, NOT P1 code as-is). RadixAttention off. B=1. |
| Need to roll back to literal P1 code | `git revert` to the pre-P2a-merge commit. Env-var flip is insufficient. |
| P2b model-specific bug | Don't register that model arch in `tenstorrent/models/__init__.py`; other models stay live. No code rollback needed. |

#### 5.3d Descope order (if W6 buffer insufficient)

In order, prefer cutting:
1. §9.10 perf-log auxiliary dimensions (per-mode timing, KV-pool utilization series) — leave as "intentionally skipped — descoped"
2. §9.9 mesh-shape `1x4` static lint (defer to P2.5 or P3)
3. Dual-track automated test suite (preserve manual smoke)

**Cannot be cut**: §9.1-§9.7 (functional baseline); §9.11-§9.16 (spec consistency gates — §9.11 eviction-replay, §9.12.a-e chunked-prefill 5-step, §9.13 teardown, §9.14 token-pool overflow, §9.15 free-list invariant for R5, §9.16 G2a math).

### 5.4 Note on §1 timing claims

§1 G-items do NOT contain absolute time SLAs. All week-budget references live in §5.3. Implementation timeline drift from §5.3 does NOT constitute a spec violation, but does trigger the §5.3b decision gate.

---

## 6. Appendix A — Review history

Spec went through 4 rounds of subagent-driven review during brainstorming:

| Round | Sections | Findings (BLOCKER / CONCERN / NOTE) | Outcome |
|---|---|---|---|
| 1 | §1+§2 | 5 / 6 / 4 | Phased P2a/P2b (was single-phase); KV adapter is parallel class (not MHATokenToKVPool subclass); model registry separate namespace; ABC renamed to model-level |
| 2 | §3 v1 | 2 / 6 / 2 | Use `req_to_token_pool.req_to_token` (not concat of match_prefix + out_cache_loc); `prefill_forward_text` wants ALL tokens with `start_pos`; SGLang admission via `PrefillAdder` (no hand-rolled gate) |
| 3 | §3 v3 | 4 / 6 / 2 | page_table values are block IDs (`// block_size`); `alloc_extend`/`alloc_decode` rewritten in pure CPU torch (Triton kernels not available); `set_finish_with_abort` is the right abort idiom; `skip_radix_cache_insert` is the right cache-bypass flag |
| 4 | §4 v1 | 2 / 6 / 3 | §9.5 reframed as queue-full admission (not pool-OOM); `cache_hit_rate` Prometheus gauge is queryable; new §9.11 eviction-replay gate |
| 5 | §5 v1, v2, v3 | 4 / 5 / 4 across rounds | INV-1..INV-7 invariants; quantified decision gate; descope order; license risk demoted to W0 task |

---

## 7. Appendix B — Cross-references

**Specs / plans / brainstorm**
- P1 spec: [`2026-05-11-sglang-tenstorrent-p1-design.md`](2026-05-11-sglang-tenstorrent-p1-design.md)
- P1 plan: [`../plans/2026-05-11-sglang-tenstorrent-p1-plan.md`](../plans/2026-05-11-sglang-tenstorrent-p1-plan.md)
- P2 brainstorm (this spec's input): [`../brainstorming/2026-05-12-sglang-tenstorrent-p2-brainstorm.md`](../brainstorming/2026-05-12-sglang-tenstorrent-p2-brainstorm.md)

**Implementation tooling**
- `superpowers:writing-plans` skill — produces the implementation plan from this spec
- `superpowers:subagent-driven-development` skill — executes the plan
- Git remote: `predator2k/sglang` (`origin`). NEVER push to `sgl-project/sglang` per N9
- tt-metal docker images (two pins per R1/R13):
  - **P2a.0 / P1-simple-path** (UMD-compatible with host KMD 2.8.0; no `generator_sglang.py`): `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-55fd115-aa4ae1e` (sha256:`d4116d2a7b20383ec42c990b08f9c6159462f9f2b84f9a25448db678f17e41aa`)
  - **P2a.1+ / paged path** (has `generator_sglang.py`; UMD mismatch with current host — must resolve before P2a.1 starts): `ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-models-amd64:latest-rc` (sha256:`40dcdcdabb5ea87a0700d7bdeeada290fe2a09246d4890237b8cd6828c1e360c`)

**Upstream code (tt-metal, read-only)**
- `tt_transformers` (bundled in tt-metal docker image): `models/tt_transformers/tt/generator_sglang.py`, `common.py`, `attention.py`, `model.py`, `decoder.py`, `generator.py`
- GptOss separate loader: `models/demos/gpt_oss/tt/common.py`

**SGLang integration files**
- `python/sglang/srt/managers/scheduler.py` (and possibly `managers/utils.py` for `GenerationBatchResult`)
- `python/sglang/srt/mem_cache/allocator.py` (subclassed by `TTPagedKVAdapter`)
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/managers/schedule_batch.py` (`Req.set_finish_with_abort`, `req.skip_radix_cache_insert`)

**P1 module root for new files in this spec**
- All `tenstorrent/...` paths in §2.2/§2.3/§5.3c are relative to `python/sglang/srt/hardware_backend/tenstorrent/` (P1 convention)

**Tenstorrent plugin source (P2a absorption base, 2026-05-13)**
- Upstream: [`tenstorrent/tt-inference-server`](https://github.com/tenstorrent/tt-inference-server) — repo cloned at `/home/mhnie/tt-inference-server/`
- Plugin subdir: `tt-sglang-plugin/` (~500 LoC across 5 files)
- License: SPDX `Apache-2.0`, © 2026 Tenstorrent USA, Inc. — our adapted copy MUST preserve SPDX header per N9 fork-only policy.

**Built tt-metal image (P2a runtime, 2026-05-13)**
- Tag: `localhost/local-tt-metal:dev` — image ID sha256:`973e972bddf5` (built from tt-metal commit `89686ee7`, "UMD Bump 2026-05-12", verified mesh-open + tt_transformers import + `generator_sglang.LlamaForCausalLM` import working against host KMD 2.8.0 / FW 19.6.0).
- Build context: `/home/mhnie/tt-metal/` (~547 MB clone) with `dockerfile/Dockerfile --target release-models --build-arg GIT_REF=89686ee7`. ~30 min build on AMD Ryzen 9 9900X 24-thread.
- Replaces both earlier image pins (which had UMD < FW mismatch).

---

## A2. Amendment Notes — 2026-05-13 (plugin pivot + P2 minimization)

### A2.1 What changed and why

After ~14 commits of P2a Phase-0 work landing the new TTExecutionBackend ABC, P1 simple-backend port, and Phase-0 signature/Q-answer evidence, we discovered Tenstorrent's `tt-inference-server/tt-sglang-plugin` (~500 LoC) which implements the **same integration objective via SGLang's standard model-registration pattern** rather than our planned custom ABC. The plugin already uses `generator_sglang.py`'s SGLang-flavored wrapper (`LlamaForCausalLM.initialize_sglang_model` + `prefill_forward` + `decode_forward` + `allocate_kv_cache`) and covers all 4 target model families (Llama / Qwen / Mistral / GptOss).

### A2.2 INV deltas

| Invariant | Pre-amendment | Post-amendment | Reason |
|---|---|---|---|
| INV-1 | `TTExecutionBackend.forward(forward_batch) → LogitsProcessorOutput` is model-level | **Scoped to `tt_transformers_single` fallback only.** Plugin path uses `TenstorrentLlamaForCausalLM(nn.Module).forward(input_ids, positions, forward_batch)` (standard SGLang model signature). | Plugin uses ModelRegistry path, not a custom ABC. T0.6 ABC retained as simple-path contract. |
| INV-2 | `TTPagedKVAdapter` subclasses `PagedTokenToKVPoolAllocator` with `device=cpu` | **Removed.** SGLang's default allocator owns slot indices; `tt_transformers.allocate_kv_cache(num_blocks,...)` owns KV bytes on device; mapping is `block_id = index // block_size`. | Plugin's simpler "co-indexed" model. Drops ~600 LoC of planned adapter work. |
| INV-3 | `page_table` int32, block IDs from `// block_size` | **Preserved verbatim** — plugin's `_build_page_table` does exactly this. | No change. |
| INV-4 | SGLang owns page table | **Amended.** SGLang owns the slot-index space → page-table block IDs derived; tt_transformers receives `page_table` + `kv_cache` externally per forward call. | Spirit preserved; ownership boundary clarified. |
| INV-5 / INV-6 / INV-7 | Backend switch via env var / `Tenstorrent*` namespace / mesh-shape parametric | **All preserved.** Plugin uses model-registry namespace patching which is compatible with INV-6. | No change. |

### A2.3 Scope cut from P2 (moved to P3 or dropped)

The following items, originally planned for P2a, are **moved to P3**:

- **128K chunked-prefill on Llama** (was G4b) — P2b drops to "8K-32K baseline" only; long-context is a P3 stress goal
- **P300 chunk-size table** (was G5b) — plugin's `_get_max_tokens_for_hardware` is a starting point; per-model empirical tuning is P3
- **HiRadixCache** — P3
- **Disaggregation** — P3
- **Speculative decoding** — P3
- **LoRA / multi-LoRA** — P3
- **BF16-throughout precision** — P3 (plugin defaults to BFP8)
- **Multimodal** — P3 (plugin doesn't address; would need `generator_vllm.py`-style vision path)
- **tt-xla backend** — P3 (fallback for models outside `tt_transformers` coverage)
- **Performance kernel-level tuning** — P3 (plugin is functional baseline; P3 optimizes)

### A2.4 What stays in P2 (the minimum that justifies a Phase 2 at all)

1. **Plugin port + Blackhole validation** — G1a, G2a, G3a, G8a. The plugin has never run on Blackhole; we are the first.
2. **Dual-track preservation** — G5a. P1 simple backend remains live as `tt_transformers_single`. No production should ever lose its fallback.
3. **RadixAttention probe** — G4a. Plugin admits non-validation; our measurement establishes whether SGLang prefix caching meaningfully accelerates TT inference. Result either way is valuable input to P3.
4. **§9 acceptance gates (minimum set)** — §9.1, §9.2, §9.3, §9.4, §9.5, §9.6, §9.7 (stability), §9.8 (dual-track), §9.10 (perf log baseline). §9.9 mesh-shape, §9.11 eviction-replay, §9.12.a-e chunked-failure, §9.13 teardown, §9.14 token-pool overflow lint — **all KEPT** as they validate plugin behavior we depend on. §9.15 / §9.16 — **dropped** (no `TTPagedKVAdapter` to validate).
5. **Per-model smoke** (P2b) — G1b, G2b registration of Qwen / Mistral / GptOss; verify each one boots and emits coherent output. No 128K stress.

### A2.5 P3 scope (greatly expanded)

P3 inherits everything cut from P2 plus its original scope:

- 128K chunked-prefill on Llama (was P2b G4b)
- P300 chunk-size empirical table (was P2b G5b)
- Speculative decoding integration with SGLang
- LoRA + Multi-LoRA serving
- BF16-throughout precision (vs BFP8 default)
- Multimodal inference (vision text models)
- HiRadixCache + disaggregation explorations
- tt-xla backend (model-coverage extension beyond `tt_transformers`)
- Performance kernel-level tuning (overlap, fuse, batched parallelism)
- 4× p150a hardware validation (plugin code has stubs, untested)
- Galaxy mesh shape (`8,4`) — plugin claims support but untested
- Long-running stability (24h+) under sustained load
- Production observability (Prometheus dashboards, alerting, queue depth tracking)

P3 is now the **innovation phase** rather than P2 being it. P2 is plumbing-verification.
