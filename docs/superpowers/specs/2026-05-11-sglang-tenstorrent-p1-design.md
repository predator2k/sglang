# SGLang on Tenstorrent — Phase 1 Implementation Spec

> **Date**: 2026-05-11
> **Status**: Draft (brainstorming output, pre-implementation)
> **Scope**: Phase 1 only. Phase 2/3 outlined as deferred future work.
> **Supersedes**: [`docs/platforms/tenstorrent_design.md`](../../platforms/tenstorrent_design.md) for the P1 implementation path. That doc remains a broader design reference.
> **Target hardware**: 2× Tenstorrent Blackhole p150a (`01:00.0`, `06:00.0`)

---

## 0. Executive Summary

Phase 1 (P1) delivers an end-to-end **single-prompt** Llama-3.1-8B BF16 inference server on SGLang, running TP=2 across two p150a cards via ttnn `mesh_device` and ETH fabric. P1 is a **black-box integration**: SGLang acts as the scheduler / HTTP server / sampler, while a `tt_transformers` model instance (built via `models.tt_transformers.tt.common.create_tt_model` returning a `Generator` / `LlamaForCausalLM`) owns the model forward and KV cache internally. P1 deliberately **disables** continuous batching, RadixAttention prefix cache, chunked prefill, and CUDA graphs.

P1 → P2/P3 is an **architectural inversion**, not a continuation. After P1, the decision whether to invest in P2 (paged KV + per-layer SGLang attention) must be re-scoped based on observed performance.

**P1 estimate**: 1–1.5 months for one engineer with prior ttnn familiarity, or 2–3 months without.

---

## 1. Hardware & Software Context

### 1.1 Hardware (verified on this machine)

| Card | BDF | PCIe | Comment |
|---|---|---|---|
| Device 0 | `0000:01:00.0` | Gen5 x16 (~64 GB/s) | Primary |
| Device 1 | `0000:06:00.0` | **Gen3 x1 (~0.98 GB/s, downgraded)** | Motherboard limitation, not fixable |

**Implication**: all inter-card data **must** traverse ETH fabric (4× QSFP-DD 800GbE per card), because Device 1's PCIe bandwidth is 63× lower than Device 0. Host-bridged collectives are not viable.

ETH fabric is brought up by `tt_transformers` / ttnn at `open_mesh_device` time (not pre-flashed at idle). Confirmed working: user has successfully run `tt-metal/models/tt_transformers/demo/simple_text_demo.py` with Qwen3-1.7B on these two cards via TP=2.

### 1.2 Software stack

| Component | Version |
|---|---|
| Tenstorrent KMD | 2.8.0 (installed) |
| `tt-smi` / `tt-flash` / `tt-umd` | installed, working |
| Docker image (dev env) | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-55fd115-aa4ae1e` |
| ttnn / tt-metal | bundled in docker image |
| `tt_transformers` | bundled in docker image |
| SGLang | from this repo, mounted into container with `-v` |

**Dev workflow**:
```bash
docker run --rm \
    --device /dev/tenstorrent \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --ipc host \
    -v $HOME/tt-models:/models \
    -v /home/mhnie/sglang:/sglang \
    --entrypoint bash \
    ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-55fd115-aa4ae1e \
    -c "source /home/container_app_user/tt-metal/python_env/bin/activate && \
        cd /sglang/python && pip install -e . && \
        SGLANG_PLATFORM=tenstorrent python -m sglang.launch_server [...] "
```

---

## 2. Prerequisites (must verify before starting implementation)

These are not implementation tasks; they are go/no-go conditions. **If any of these fail, P1 should pause until resolved.**

1. **tt_transformers supports Llama-3.1-8B on Blackhole at the pinned commit, AND exposes a reusable public API surface.** **(Partially verified 2026-05-11 — see `phase0-evidence.txt` and the post-evidence corrections to this section.)** Two things must verify:
   a. **Demo run** — execute the demo with Llama-3.1-8B inside docker. Model selection is via the `HF_MODEL` env var (there is no `--model` pytest flag, confirmed by `conftest.py:pytest_addoption`):
      ```bash
      HF_MODEL=/models/Llama-3.1-8B-Instruct \
      pytest -s -v models/tt_transformers/demo/simple_text_demo.py \
        -k "performance and batch-1"
      ```
      Confirmed at line 1053 of `simple_text_demo.py`: `"Llama-3.1-8B"` is in `supported_models`. Confirmed at line 1086: `"P300_Llama-3.1-8B": 38` decode toks/s target on this exact host (P300 = two p150a).
   b. **Public class surface** — inside the docker image, verify the actual API. **The earlier draft of this spec referenced `tt_transformers.models.llama3.Llama_3_1`; that module/class does NOT exist** in `vllm-tt-metal-src-release-...:0.10.0-55fd115-aa4ae1e`. Actual surface:
      ```bash
      python -c "
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM  # Generator subclass
        from models.tt_transformers.tt.common import create_tt_model           # factory, common.py:665
        import inspect
        print(Generator)
        print(LlamaForCausalLM)
        print(inspect.signature(create_tt_model))"
      ```
      Capture `create_tt_model`'s signature and `Generator`'s public methods (`prefill`, `decode_forward`, etc. — to be verified). These drive Phase F's `TTLlamaWrapper` glue. There is no per-family Python class; model identity comes from the `HF_MODEL` directory contents.

   If either check fails, see Risk #1 — re-spec, don't silently swap models.

2. **Llama-3.1-8B-Instruct downloaded.** Path: `$HOME/tt-models/Llama-3.1-8B-Instruct/`, HF format (safetensors + config.json + tokenizer files).

3. **HuggingFace access for Llama-3.1.** It's a gated model. Either:
   - `huggingface-cli login` on host with an account that has access, then `huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir $HOME/tt-models/Llama-3.1-8B-Instruct`, OR
   - Weights pre-staged from a non-gated mirror.
   Inside docker, no further auth needed (read from mounted path).

4. **Docker image pulled and reproduces user's working Qwen3 demo.** Baseline that proves stack is healthy before any SGLang code runs.

5. **`pip install -e /sglang/python` does not break the docker image's pre-installed environment.** The image already contains `vllm` + tt-metal Python env. SGLang's install may attempt to install conflicting `torch` / `flashinfer` versions. Verify:
   - After install, `python -c "import sglang, ttnn, vllm; print('ok')"` succeeds
   - If not, use `pip install -e /sglang/python --no-deps` and manually reconcile.

6. **ETH fabric working between the two p150a's.** Already proven; no action needed.

7. **`sudo tt-smi -r` works for device reset** (passwordless sudo confirmed). Required for recovery from wedged mesh state.

8. **`SGLANG_PLATFORM=tenstorrent` env var requirement** is conditional: if `ttnn` is the only platform plugin that activates on this machine, SGLang's auto-discovery selects TT automatically. If other plugins also activate (e.g. a co-installed NPU or CUDA plugin reports itself active), the env var becomes mandatory. Inside our docker image (no CUDA, no NPU), it should not be needed; document as defensive-only.

---

## 3. Architecture

### 3.1 Decision summary

| Decision | Choice | Rationale |
|---|---|---|
| Target | Llama-3.1-8B BF16, TP=2 on 2× p150a | User-specified |
| Code organization | **In-tree**: `python/sglang/srt/hardware_backend/tenstorrent/` | User-specified |
| Multi-card execution | **Single-process** + `ttnn.open_mesh_device([0, 1])` + ETH fabric | Only viable model given p150a hardware + ttnn maturity; matches Tenstorrent's own reference impls |
| Model code source | Reuse `tt_transformers` via `models.tt_transformers.tt.common.create_tt_model` factory + `Generator` / `LlamaForCausalLM` classes (verified 2026-05-11) | Avoids per-op Blackhole compatibility risk |
| Execution-backend surface | `TTExecutionBackend` ABC (5 lifecycle methods) + registry; **P1 implements one backend (`tt_transformers`)**; tt-xla slot reserved for post-P1 | Tenstorrent deprecated tt-torch in favor of tt-xla in <12 months — keep the impl thin and replaceable; mirrors SGLang's `--attention-backend` / `--moe-runner-backend` registry pattern |
| SGLang TP view | `tp_size = 1` (TP=2 hidden inside mesh) | P1 only; P2 will need re-design |
| Integration pattern | **MLX-style**: worker is forward entry; ModelRunner is bookkeeping stub | Required because `attn_backend=None` is not safe with standard `ModelRunner.forward_batch` path |
| Phasing | P1 black-box now; P2/P3 deferred and rescoped after P1 | Honest about architectural inversion required for P2 |

### 3.2 Module-level architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  SGLang HTTP server, tokenizer, scheduler           (unchanged)     │
│  Scheduler dispatches to TTTpModelWorker via `use_tt()` branches    │
│  (see §4.2 — analogous to existing MLX `use_mlx()` branches)        │
│                                                                     │
│  Server-args defaults (TT-injected, see §6.1 for full list):        │
│    max_running_requests = 1                                         │
│    chunked_prefill_size = -1   (disables chunked prefill)           │
│    disable_radix_cache = True                                       │
│    disable_overlap_schedule = True  (CRITICAL — ttnn is sync)       │
│    pre_warm_nccl = False                                            │
│    cpu_offload_gb = 0                                               │
│    sampling_backend = "pytorch" (but see invariant #2 below)        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ in-process worker.forward_batch(batch)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TTTpModelWorker(TpModelWorker)         (NEW, mirrors MLX exactly)  │
│   ★ Subclasses TpModelWorker; overrides forward_batch_generation()  │
│     to route through TTExecutionBackend instead of standard         │
│     ModelRunner.forward(). Returns GenerationBatchResult with       │
│     next_token_ids already sampled. Exact method shape: copy        │
│     MlxTpModelWorker.forward_batch_generation line-by-line and      │
│     swap MLX-specific calls for TT equivalents.                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TTExecutionBackend (ABC, 5 lifecycle methods)                      │
│   .new_request(req_id, prompt_tokens) -> None                       │
│   .extend(req_id) -> last-token logits (host torch tensor)          │
│   .decode_step(req_id, last_token) -> next-token logits             │
│   .free(req_id) -> None                                             │
│   .reset_all() -> None                                              │
│                                                                     │
│  Registry: TT_EXECUTION_BACKENDS = { name -> class }                │
│  Selection: env SGLANG_TT_EXECUTION_BACKEND (default "auto" →       │
│             "tt_transformers" in P1)                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ P1 impl
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TTTransformersExecutionBackend  (NEW, glue around tt_transformers) │
│   - per-request KV state in self._req_state[req_id]                 │
│   - wraps the Generator/LlamaForCausalLM instance returned by       │
│     models.tt_transformers.tt.common.create_tt_model                │
│   - registered as "tt_transformers"                                 │
│                                                                     │
│  (Post-P1) TTXLAExecutionBackend slot — registered as "tt_xla"      │
│  but currently raises NotImplementedError on construction. See      │
│  §11 — tt-xla is the natural model-coverage path after P1 ships.    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ttnn / tt-metal runtime                                            │
│   open_mesh_device([0, 1], fabric=Eth)                              │
│   tt_transformers Generator (built by create_tt_model from HF_MODEL;│
│     weights mesh-sharded, TP=2 internal)                            │
└─────────────────────────────────────────────────────────────────────┘
```

**Key invariants**:
1. `TTTpModelWorker(TpModelWorker)` is the forward entry — it inherits from the standard `TpModelWorker` and overrides `forward_batch_generation()`. The standard `ModelRunner.forward()` is never called on the TT path. Scheduler reaches this worker via a new `use_tt()` dispatch branch added to `scheduler.py` alongside the existing `use_mlx()` ones — see §4.2.
2. On TTModelRunner: `attn_backend = None`, `graph_runner = None`, **`sampler = None`** (matches MLX exactly — `MlxModelRunnerStub.initialize()` sets `self.sampler = None`). **Sampling is performed inside the TT path, not via SGLang's `model_runner.sample()`** — see §5.1. **P1 is GREEDY-ONLY**: `argmax(logits)` inside the worker or wrapper. Temperature, top-k, top-p, min-p, repetition penalty are **out of scope for P1** even though `sampling_backend = "pytorch"` is set defensively. Adding sampling features means implementing them at the worker level (host-side torch on CPU) — deferred to P2/P3.
3. `model_runner.device = "cpu"` — even though `server_args.device = "tenstorrent"` (user-facing identity), the worker's `self.device` (which `scheduler.init_overlap` uses via `torch.get_device_module(self.device)`) must be "cpu" because `torch` has no device module called "tenstorrent". `TTModelRunner.__init__` must override `self.device = "cpu"` after `super().__init__()`.
4. No `torch.distributed`, no NCCL, no Gloo, no ProcessGroup, no per-rank workers.
5. SGLang's KV pool is `_DummyKVCache` (zero device allocation); KV lives inside `tt_transformers` per-request state. The dummy pool is constructed **directly inside `TTModelRunner.initialize()`** with the matching constructor signature — the `get_mha_kv_pool_cls()` factory is **never invoked and raises NotImplementedError defensively** (see §6.1).
6. Allowed `forward_mode` values: `EXTEND`, `DECODE`, `IDLE`. **IDLE returns an empty `GenerationBatchResult(logits_output=LogitsProcessorOutput(next_token_logits=None), can_run_cuda_graph=False)`** — MLX handles this at `tp_worker.py:136-140` and we must mirror it (scheduler produces IDLE batches even with max_running_requests=1 for sync / draining). MIXED/SPLIT_PREFILL/DLLM_EXTEND raise `NotImplementedError`.
7. **`TTExecutionBackend` is the only seam the worker depends on.** The 5-method ABC (`new_request` / `extend` / `decode_step` / `free` / `reset_all`) is the boundary; everything below it (ttnn calls, tt_transformers method dispatch, page tables, etc.) is implementation detail of `TTTransformersExecutionBackend`. P1 ships exactly one implementation; the registry + ABC exist so a future `TTXLAExecutionBackend` (post-P1, see §11) is a backend-swap rather than a Phase-F rewrite. Worker and ModelRunner code must **NOT** import `tt_transformers` symbols directly — go through the backend interface.

---

## 4. Component List

### 4.1 New files (in-tree)

All under `python/sglang/srt/hardware_backend/tenstorrent/`.

| # | File | Role | LoC est. |
|---|---|---|---|
| 1 | `__init__.py` | Module init | ~5 |
| 2 | `platform.py` | `TTSRTPlatform(SRTPlatform)` + `activate_tt_platform()` entry-point fn + `MeshDeviceCtx` singleton (open/close, program cache config) | ~250 |
| 3 | `tp_worker.py` | `TTTpModelWorker(TpModelWorker)` — inherits standard worker; overrides `_init_model_runner`, `get_pad_input_ids_func`, `forward_batch_generation`. Adds `_forward_batch_generation_tt(model_worker_batch)` private method analogous to MLX's `_forward_batch_generation_mlx`. **Copy MLX's file as starting point, swap MLX runner references for TT.** | ~200 |
| 4 | `model_runner.py` | `TTModelRunner(ModelRunner)` — bookkeeping stub. Overrides `__init__` to set `self.device = "cpu"`. Overrides `initialize()`: `attn_backend=None`, `graph_runner=None`, **`sampler=None`** (mirrors MLX), constructs `_DummyKVCache` directly with matching ctor sig. P1 sampling is greedy and happens in the worker, not via `self.sampler` | ~150 |
| 5 | `model_runner_stub.py` | `_DummyKVCache`, `_DummyModel` (copy MLX pattern) | ~80 |
| 6a | `execution/__init__.py` | Registry: `TT_EXECUTION_BACKENDS: dict[str, type[TTExecutionBackend]]`, decorator `@register_tt_execution_backend(name)`, factory `get_tt_execution_backend(name)` that resolves `"auto"` → P1's default `"tt_transformers"` | ~40 |
| 6b | `execution/base.py` | `TTExecutionBackend(abc.ABC)` — the 5-method lifecycle contract worker code depends on | ~60 |
| 6c | `execution/tt_transformers_backend.py` | `TTTransformersExecutionBackend(TTExecutionBackend)` wrapping the `Generator` instance built by `tt_transformers.tt.common.create_tt_model(HF_MODEL=...)`. Per-request KV state mgmt via `Generator`'s prefill/decode methods (exact method names captured during Phase 0.2). Registered as `"tt_transformers"`. | ~300 |
| 6d | `execution/tt_xla_backend.py` | `TTXLAExecutionBackend(TTExecutionBackend)` placeholder — registered as `"tt_xla"`, raises `NotImplementedError` from `__init__` with a pointer to §11. Lands in P1 so the registry shape is exercised in tests; the real impl is post-P1. | ~30 |
| 7 | `warmup.py` | Decode-path warmup driver (runs dummy `[1,1]` decode to JIT program cache) | ~80 |
| 8 | `scripts/reset_devices.sh` | `sudo tt-smi -r 0000:01:00.0 0000:06:00.0` (documented recovery) | ~10 |
| 9 | `test/test_platform_activate.py` | Activation returns None when ttnn unavailable | ~50 |
| 10 | `test/test_wrapper_lifecycle.py` | KV state lifecycle no-leak | ~100 |
| 11 | `test/test_server_args_defaults.py` | apply_server_args_defaults sets expected fields | ~60 |
| 12 | `test/test_forward_mode_guard.py` | Worker raises on unsupported forward_mode | ~50 |
| 13 | `test/test_greedy_correctness.py` | 10-prompt greedy vs HF reference (`@requires_tt`) | ~150 |
| 14 | `test/test_execution_backend_registry.py` | `TT_EXECUTION_BACKENDS` has `tt_transformers` and `tt_xla` entries; `get_tt_execution_backend("auto")` resolves to `tt_transformers` in P1; `tt_xla` factory raises `NotImplementedError` with a clear pointer | ~40 |

Total P1 new code: ~1500 LoC (unchanged — the ABC + registry adds ~100 LoC but the wrapper LoC budget absorbs it).

### 4.2 SGLang core modifications

| File | Change |
|---|---|
| `python/pyproject.toml` | Add `[project.entry-points."sglang.srt.platforms"]` → `tenstorrent = "sglang.srt.hardware_backend.tenstorrent.platform:activate_tt_platform"` |
| `python/sglang/srt/utils/tensor_bridge.py` (where `use_mlx()` actually lives — verified) | **Add `use_tt()` helper** with the same call-shape as `use_mlx()`. **NOTE: `use_mlx()` is env-var-gated** (`bool(envs.SGLANG_USE_MLX.get()) and _MLX_AVAILABLE`). TT uses platform-plugin activation instead — `use_tt()` should check `isinstance(current_platform, TTSRTPlatform)` (or the OOT enum), NOT introduce a `SGLANG_USE_TT` env var. The structural analogy is "one boolean dispatch primitive"; the gating semantics differ. |
| `python/sglang/srt/environ.py` | Add `SGLANG_TT_EXECUTION_BACKEND = EnvStr("")` — selects which `TTExecutionBackend` implementation to load. `""` / `"auto"` → P1 default `"tt_transformers"`. `"tt_xla"` lands post-P1 and currently raises `NotImplementedError`. Env-var instead of `server_args` keeps the choice contained to the TT backend (vs. polluting upstream SGLang's dataclass with TT-specific options). |
| `python/sglang/srt/managers/scheduler.py` | (a) **Add `use_tt()` dispatch branches alongside existing `use_mlx()` branches** at the ~4 worker-instantiation sites. Line numbers as of SGLang HEAD on 2026-05-11: 384–385, 653, 1312, 1511 — re-verify against current HEAD before editing; (b) guard NCCL-specific overlap paths with `is_cuda()` |
| `python/sglang/srt/server_args.py` | (a) Document `tenstorrent` as a valid `--device` value in the argparse help text (no `choices=` constraint exists — argparse accepts any string today); (b) guard ~5 `torch.cuda.*` capability detections with `is_cuda()` |
| `python/sglang/srt/managers/schedule_batch.py` | Line 342 `torch.cuda.current_device()` guard |
| `python/sglang/srt/managers/io_struct.py` | CUDA usage guard (verify sites with grep) |
| `python/sglang/srt/managers/utils.py` | CUDA usage guard |
| `python/sglang/srt/managers/scheduler_pp_mixin.py` | CUDA usage guard |
| `python/sglang/srt/managers/mm_utils.py` | CUDA usage guard |
| `python/sglang/srt/managers/scheduler_profiler_mixin.py` | CUDA usage guard |
| `python/sglang/srt/model_executor/model_runner.py` | Guard `torch.cuda.empty_cache/synchronize/get_device_capability` (~4 sites) |
| `python/sglang/srt/utils/offloader.py` | Make offloader paths conditional on `is_cuda()` |
| `python/sglang/srt/utils/multi_stream_utils.py` | Same guard pattern |
| `python/sglang/srt/utils/nvtx_pytorch_hooks.py` | Try/except top-level CUDA imports |
| `python/sglang/srt/utils/profile_utils.py` | Same |
| `python/sglang/srt/utils/device_timer.py` | Same |
| `python/sglang/srt/utils/bench_utils.py` | Same |
| `python/sglang/srt/utils/patch_torch.py` | Same |
| `python/sglang/srt/utils/cuda_ipc_transport_utils.py` | Same |

Each individual change is small (5–15 lines). **Revised total: ~150 LoC of edits across ~17 files.** The original ~80 LoC / 9 files estimate was optimistic per the independent review (see Appendix B); implementer should run `grep -rln "torch.cuda" python/sglang/srt/{managers,utils}` before estimating to catch additional sites.

### 4.3 Files explicitly reused unchanged

- HTTP / gRPC entrypoints
- Tokenizer
- Scheduler core
- Sampler (pytorch path)
- RadixAttention (present but inert in P1 since KV pool is dummy)
- Model registry / chat template
- All benchmark / observability tools

### 4.4 Out of scope for P1

- CUDA graph or equivalent (TT relies on ttnn program cache)
- Paged KV cache (P2)
- Continuous batching / multi-request batched forward (P3)
- Speculative decoding
- MoE / MLA / NSA / multimodal
- Quantization (BFP8 / BFP4)
- Disaggregation (PD separation)
- TP > 2 / pipeline parallelism / data parallelism
- Models other than Llama-3.1-8B
- Production deployment form (Docker image, systemd, etc.)

---

## 5. Data Flow

### 5.1 Request path

```
[POST /v1/completions]
     │
     ▼
managers/scheduler.py            (unchanged)
   produces ScheduleBatch
     forward_mode ∈ {EXTEND, DECODE}
     |input_ids| determined by mode
     │
     ▼  in-process call (no IPC)
hardware_backend/tenstorrent/tp_worker.py: TTTpModelWorker(TpModelWorker)

   # CONTRACT: copy MLX's tp_worker.py structure line-by-line. The shape
   # below is a faithful summary; exact arg-default values + return-field
   # population (GenerationBatchResult has many optional fields) must
   # mirror MlxTpModelWorker.forward_batch_generation and
   # MlxTpModelWorker._forward_batch_generation_mlx as of the SGLang
   # revision being built against.

   def forward_batch_generation(
       self, model_worker_batch: ModelWorkerBatch,
       forward_batch=None, pp_proxy_tensors=None,
       is_verify=False, skip_attn_backend_init=False,
   ) -> GenerationBatchResult:
     if model_worker_batch is None:
       return super().forward_batch_generation(
           model_worker_batch, forward_batch, pp_proxy_tensors,
           is_verify, skip_attn_backend_init)
     return self._forward_batch_generation_tt(model_worker_batch)

   def _forward_batch_generation_tt(self, mwb) -> GenerationBatchResult:
     # --- IDLE: mirror MLX tp_worker.py:136-140 ---
     if mwb.forward_mode.is_idle():
       return GenerationBatchResult(
           logits_output=LogitsProcessorOutput(next_token_logits=None),
           can_run_cuda_graph=False)  # + other fields per MLX

     # --- P1 guard: only EXTEND/DECODE besides IDLE ---
     if not (mwb.forward_mode.is_extend() or mwb.forward_mode.is_decode()):
       raise NotImplementedError(
           f"P1 supports EXTEND/DECODE/IDLE only, got {mwb.forward_mode}. "
           f"Chunked prefill / mixed / DLLM must stay disabled via "
           f"apply_server_args_defaults.")

     # --- Per-request id: mwb.reqs is List[Req], NOT mwb.req_ids ---
     # P1: max_running_requests=1 ⇒ len(mwb.reqs) == 1 at most
     assert mwb.reqs is not None and len(mwb.reqs) == 1
     req_id = mwb.reqs[0].rid

     if mwb.forward_mode.is_extend():
       prompt_tokens = mwb.input_ids  # torch.Tensor[L] on CPU
       self.wrapper.new_request(req_id, prompt_tokens)
       logits_ttnn = self.wrapper.extend(req_id)
     else:  # DECODE
       last_tok = int(mwb.input_ids[-1].item())
       logits_ttnn = self.wrapper.decode_step(req_id, last_tok)

     # --- Greedy sampling inside the worker (mirrors MLX argmax path) ---
     # P1 is GREEDY-ONLY (see §3.2 invariant #2). We do NOT call
     # self.model_runner.sample() because self.model_runner.sampler is
     # None. To add temperature/top-p in P2/P3, either set sampler to a
     # real instance AND build a ForwardBatch.init_new(mwb, runner)
     # (which itself requires attn_backend != None ⇒ another redesign),
     # or implement top-k/top-p directly here on host torch tensors.
     logits = ttnn.to_torch(logits_ttnn).float()              # [vocab]
     next_token_ids = torch.argmax(logits, dim=-1, keepdim=True).long()
     # ★ MLX returns next_token_logits=None in GenerationBatchResult
     #   because sampling already happened — do the same here:
     return GenerationBatchResult(
         logits_output=LogitsProcessorOutput(next_token_logits=None),
         next_token_ids=next_token_ids,
         can_run_cuda_graph=False)
     # Per the round-2 verification, MLX leaves all other
     # GenerationBatchResult fields at their dataclass defaults
     # (see managers/utils.py:26-32) — no further fields needed.
     │
     ▼
hardware_backend/tenstorrent/execution/tt_transformers_backend.py:
  TTTransformersExecutionBackend (impl of TTExecutionBackend, resolved
                                  via TT_EXECUTION_BACKENDS registry)
   self._req_state[req_id] holds per-request KV handle
   .extend()       — runs prefill, returns last-token logits (host)
   .decode_step()  — advances by one token, returns last-token logits
     │
     ▼
ttnn / tt-metal runtime
   mesh_device (open once at server start, closed at shutdown)
   tt_transformers Generator.prefill/decode_forward — TP=2 sharding internal
     │
     ▼  ttnn.Tensor → torch.Tensor (D2H over ETH gather + PCIe)
sampling/sampler.py             (unchanged, pytorch path)
   top-k / top-p / min-p sampling on CPU
     │
     ▼  int token_id
managers/scheduler.py
   appends to req.output_ids
   re-batches → DECODE step OR finalizes if EOS / max_tokens
```

### 5.2 Per-token time budget (P1 estimate, subject to measurement)

| Step | Latency | Notes |
|---|---|---|
| **Server cold-start: weight load to Device 0** | ~0.5 s | 8 GB over PCIe Gen5 x16 (64 GB/s) |
| **Server cold-start: weight load to Device 1** | **~10–15 s** | 8 GB over PCIe Gen3 x1 (~0.8 GB/s) — **physical bottleneck**, unavoidable on this machine |
| **Server cold-start: mesh + ETH bring-up** | 5–30 s | tt_transformers `open_mesh_device` + fabric init |
| **Cold-start first prefill** | 10–60 s | tt-metal program cache JIT, per unique prefill shape |
| Warm prefill (1k tokens) | 300–800 ms | shape already cached |
| Warm decode (per token) | 30–70 ms | ttnn synchronous, no async pipelining |
| └ ttnn forward (TP=2 via ETH) | 25–55 ms | the dominant cost |
| └ `ttnn.to_torch(logits)` D2H | 5–15 ms | mesh ETH gather + PCIe Gen5 x16 hop (Device 0 path) |
| └ host sampler | <1 ms | torch.topk / multinomial on CPU |
| └ Python / scheduler overhead | 3–7 ms | no cuda-graph equivalent to amortize |
| **Single-request steady throughput** | **~10–25 tok/s** | P1 estimate only; gated on tt_transformers Blackhole maturity |

These numbers are estimates pending measurement. They define the magnitude, not a contract.

**Note on Device 1's PCIe Gen3 x1 bottleneck**: it dominates *initial weight load* (~10–15 s) but does **not** appear in the steady-state decode budget — because runtime traffic between the two cards travels over ETH, not PCIe. The slow PCIe slot only carries control plane (sync, status) and KV reads/writes that originate on Device 1's local DRAM (rare; KV is sharded so each card writes to its own DRAM via NoC, not PCIe).

---

## 6. Server-args defaults, observability, concurrency

### 6.1 Server-args defaults injected by `TTSRTPlatform.apply_server_args_defaults`

```python
def apply_server_args_defaults(self, server_args):
    # P1 hard constraints
    server_args.max_running_requests = 1
    server_args.chunked_prefill_size = -1   # -1 disables chunked prefill
    server_args.disable_radix_cache = True

    # CRITICAL: overlap scheduler creates FutureMap + copy streams +
    # device Events that the synchronous ttnn forward path cannot
    # satisfy. Without this, FutureMap allocates -1 sentinels that get
    # written into req.output_ids and the run produces garbage tokens.
    server_args.disable_overlap_schedule = True

    # Sampling backend setting is largely defensive — P1 actually does
    # greedy in the worker (see §5.1) and never invokes model_runner.sample().
    # We still set "pytorch" so that any code path that does check this
    # value gets a non-CUDA-specific selection.
    server_args.sampling_backend = "pytorch"

    # Disable CUDA-only paths
    server_args.pre_warm_nccl = False
    server_args.cpu_offload_gb = 0          # 0 = offloader disabled (also default)
    server_args.enable_torch_compile = False

    # User-facing device identity. Note: scheduler.init_overlap calls
    # torch.get_device_module(self.device), and self.device flows from
    # model_runner.device — NOT from this string. We override
    # model_runner.device = "cpu" in TTModelRunner.__init__ to satisfy
    # torch's device module lookup while keeping this user-facing label.
    server_args.device = "tenstorrent"

    # TP visibility: SGLang sees tp_size = 1 in P1. The "TP=2" we
    # talk about throughout this spec is entirely inside ttnn's
    # mesh_device — invisible to SGLang. Users invoke with --tp 1
    # (or omit, defaulting to 1). If user explicitly passes --tp 2,
    # raise at startup; do not silently override.
    if server_args.tp_size not in (None, 1):
        raise ValueError(
            f"TT backend requires --tp 1 (TT-internal TP=2 is hidden); "
            f"got --tp {server_args.tp_size}.")
    server_args.tp_size = 1

    # No NCCL / disaggregation
    server_args.enable_dp_attention = False
```

Plus class-level overrides:
```python
class TTSRTPlatform(SRTPlatform):
    _enum = PlatformEnum.OOT
    device_name = "tenstorrent"
    device_type = "cpu"       # critical: torch.device("tt") would error

    def get_default_attention_backend(self) -> str:
        # Returns "tenstorrent" only because the SRTPlatform contract
        # requires a string. TTModelRunner.initialize() overrides the
        # default ModelRunner init to set self.attn_backend = None and
        # never invoke the attention registry, so this string is never
        # actually looked up in P1. We still register a stub
        # TTAttnBackend that raises NotImplementedError on any forward()
        # so that a future refactor accidentally hitting the registry
        # fails loudly.
        return "tenstorrent"
    def get_graph_runner_cls(self) -> type:
        # P1 sets support_cuda_graph()=False, so the framework will not
        # call this method. Inherit the base NotImplementedError —
        # if ever called, the trace tells us a code path snuck through.
        raise NotImplementedError("P1 does not use graph runners")
    def get_mha_kv_pool_cls(self):
        # TTModelRunner.initialize() constructs _DummyKVCache directly;
        # this factory should never be invoked. The framework's expected
        # ctor passes many kwargs (page_size, head_num, head_dim, layer_num,
        # enable_memory_saver, start_layer, end_layer; see
        # model_executor/model_runner_kv_cache_mixin.py:410-423) which
        # _DummyKVCache(size, dtype, device) does NOT accept — a TypeError
        # would result. We raise NotImplementedError here so the failure
        # is louder and earlier than a TypeError deep in the framework.
        raise NotImplementedError(
            "TT backend constructs _DummyKVCache inside "
            "TTModelRunner.initialize(); the factory should not be called. "
            "If you see this error, a code path bypassed the stub init.")
    def get_mla_kv_pool_cls(self): raise NotImplementedError("MLA not in P1")
    def get_nsa_kv_pool_cls(self): raise NotImplementedError("NSA not in P1")
    def support_cuda_graph(self) -> bool: return False
    def support_piecewise_cuda_graph(self) -> bool: return False
    def supports_fp8(self) -> bool: return False
```

### 6.2 Observability and logging

The TT backend writes structured log lines at the following points (logger name `sglang.srt.hardware_backend.tenstorrent`):

| Event | Level | Fields |
|---|---|---|
| `mesh_open` | INFO | bdfs, ETH live count, mesh shape |
| `weights_load_start` / `weights_load_done` | INFO | model_path, elapsed_s, per-card MB transferred |
| `warmup_start` / `warmup_done` | INFO | warmup shape, JIT elapsed |
| `req.new` / `req.extend` / `req.decode` / `req.free` | DEBUG | req_id, prompt_len, current_offset |
| `req.error` | ERROR | req_id, exception, stage (prefill / decode) |
| `mesh_close` / `mesh_close_timeout` | INFO / WARN | shutdown elapsed |
| `forward_mode_unsupported` | ERROR | observed forward_mode (defensive — should not fire) |

Performance counters (exposed via SGLang's existing metrics endpoint, no new endpoints):
- `tt_decode_latency_ms` (histogram)
- `tt_extend_latency_ms` (histogram)
- `tt_d2h_latency_ms` (histogram)
- `tt_active_requests` (gauge, 0 or 1 in P1)

Out of scope for P1: per-layer profiling, ttnn op-level timing, GPU memory utilization tracking.

### 6.3 Concurrency semantics under `max_running_requests = 1`

SGLang accepts concurrent HTTP requests regardless of the running-request cap. With cap=1, the scheduler queues incoming requests and serves them serially:
- Request A arrives, starts EXTEND → DECODE loop
- Request B arrives mid-A → enters scheduler's waiting queue
- B is held until A reaches EOS / max_tokens / cancellation
- B then runs

The HTTP server does not return 429 or queue full; clients see latency proportional to queue depth. Document this in the P1 readme; recommend clients implement client-side concurrency=1 to avoid surprising tail latency.

---

## 7. Error Handling

### 7.1 Startup errors

| Scenario | Detection | Behavior |
|---|---|---|
| `import ttnn` fails on non-TT host | try/except in `activate_tt_platform()` | Return `None`. SGLang falls back to base SRTPlatform (or other plugin). No exception. |
| `ttnn.open_mesh_device([0, 1])` fails | catch on init | Raise `RuntimeError("Mesh device init failed. Try: sudo tt-smi -r 0000:01:00.0 0000:06:00.0")`. Abort. |
| HF weights path missing | `os.path.isdir` check in `TTTransformersExecutionBackend.__init__` | `FileNotFoundError` with explicit instruction to mount via `-v` |
| `tt_transformers.create_tt_model` rejects HF_MODEL config | catch | `NotImplementedError` + suggest verifying tt_transformers commit and that the HF_MODEL directory matches a supported architecture (see `simple_text_demo.py:supported_models`) |
| Weights schema mismatch | tt_transformers raises | Re-raise with context |
| Warmup forward fails | catch in warmup driver | Log full traceback. Abort startup. **Never serve traffic with broken forward.** |

### 7.2 Runtime errors

| Scenario | Behavior |
|---|---|
| ttnn op raises during prefill | catch → `wrapper.free(req_id)` → HTTP 500 sanitized → mesh stays open |
| ttnn op raises during decode | same; only this request affected |
| Mesh OOM (prompt too long) | catch ttnn OOM → free → HTTP 413 "prompt exceeds device memory" |
| Prompt > model context_len | rejected by SGLang scheduler (unchanged) |
| Unexpected `forward_mode` (MIXED/SPLIT_PREFILL/etc) | `NotImplementedError` → HTTP 500. Should be unreachable given server-args defaults; defensive. |
| Client disconnect mid-decode | SGLang scheduler cancels req → worker calls `wrapper.free(req_id)` before next step |

### 7.3 Shutdown

**Process placement matters.** SGLang spawns multiple processes (TokenizerManager, Scheduler, DetokenizerManager). The `mesh_device` handle lives in the **Scheduler** subprocess — the only one that calls into ttnn. Signal handlers and `atexit` must be registered **in the Scheduler subprocess**, not in `TTSRTPlatform` at plugin-import time (which runs in the parent process, where `atexit` would never see the mesh handle).

Concrete: register the handlers inside `TTTpModelWorker._init_model_runner` (which runs in the Scheduler subprocess after the mesh has been opened) — or in `MeshDeviceCtx.__init__` if that singleton lives in the same process.

| Scenario | Behavior |
|---|---|
| SIGTERM / SIGINT to Scheduler subprocess | Handler registered in worker init: `wrapper.reset_all()` → `ttnn.close_mesh_device(mesh)` → exit |
| Unhandled exception in Scheduler subprocess main loop | `atexit` handler registered in worker init: best-effort `close_mesh_device` |
| `close_mesh_device` hangs > 5 s | Timeout. Force exit. Log requires `sudo tt-smi -r` before next start. |
| Docker hard kill (SIGKILL) | Mesh leaks. Next start fails at mesh open → user runs reset script. |
| Parent process killed but Scheduler subprocess survives | Scheduler is orphaned but still holds mesh. `tt-smi` will show device in-use. Worker has no SIGCHLD detection in P1 — accept as known issue, document. |

### 7.4 Recovery script

`hardware_backend/tenstorrent/scripts/reset_devices.sh`:
```bash
#!/bin/bash
set -e
echo "Resetting Tenstorrent devices 01:00.0 and 06:00.0..."
sudo tt-smi -r 0000:01:00.0 0000:06:00.0
echo "Done. Mesh state cleared."
```
Documented in spec, not auto-invoked by server (reset is destructive — user decides).

---

## 8. Testing Strategy

P1 has **no CI**. This machine is not part of SGLang's CI runner pool. All tests are manual or scripted.

### 8.1 Smoke test (every dev iteration)

Inside the tt-metal docker:
```bash
SGLANG_PLATFORM=tenstorrent python -m sglang.launch_server \
  --model-path /models/Llama-3.1-8B-Instruct \
  --port 30000 \
  --max-running-requests 1 \
  --chunked-prefill-size -1 --disable-radix-cache \
  --sampling-backend pytorch \
  > /tmp/sglang-tt.log 2>&1 &
SERVER_PID=$!

# poll for /health up to 5 min before trying inference
for i in {1..60}; do
  if curl -sf localhost:30000/health >/dev/null; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server died during startup. Last 50 log lines:"
    tail -50 /tmp/sglang-tt.log
    exit 1
  fi
  sleep 5
done

curl -X POST localhost:30000/v1/completions \
  -d '{"model":"llama","prompt":"The capital of France is",
       "max_tokens":10,"temperature":0}'
```
**Pass**: HTTP 200, response body includes "Paris", end-to-end ≤ 5 minutes, server process still alive after the request returns.

### 8.2 Greedy correctness

`test/test_greedy_correctness.py` — runs on actual hardware (`@requires_tt`):
- 10 fixed prompts (committed): short English, long English, code, multilingual, chat-formatted
- Generate 50 tokens at temp=0
- Compare to HF reference (`transformers` lib on CPU or remote GPU) on identical prompts
- Metric: per-prompt top-1 token agreement %
- **Pass**: mean ≥ 90%, no single prompt < 70%

### 8.3 lm-eval-harness subset (optional, recommended)

- MMLU 5-shot, 100-question subset (STEM 50 + humanities 50)
- Run same subset with HF reference
- **Pass**: TT accuracy ≥ HF reference - 1pp

### 8.4 Stability test

- `bench_serving.py` at concurrency=1 for 1 hour
- **Pass**: no crashes; no mesh OOM; **ITL p99 measured over the final 5 minutes differs from ITL p99 measured over minutes 5–10 (after warmup) by < 10%**. Skip the very first 5 minutes of the run to exclude cold-start JIT.

### 8.5 Performance log (no thresholds)

Record and report (not gate):
- Cold first-prefill TTFT
- Warm prefill latency vs prompt length (100 / 500 / 1k / 4k)
- Warm decode steady-state tok/s
- D2H measured latency

Results land in spec's "P1 observed performance" addendum after first measurement, feeding into P2 decision.

### 8.6 Unit tests (CPU-runnable, no TT hardware required)

| Test | Purpose |
|---|---|
| `test_platform_activate.py` | `activate_tt_platform()` returns `None` when `import ttnn` raises (mocked) |
| `test_wrapper_lifecycle.py` | `new_request → extend → decode_step × 3 → free` round-trip with mocked tt_transformers; `reset_all` clears state |
| `test_server_args_defaults.py` | `apply_server_args_defaults` sets expected constraints |
| `test_forward_mode_guard.py` | Worker raises `NotImplementedError` on MIXED/SPLIT_PREFILL/DLLM_EXTEND. IDLE must NOT raise — verify it returns an empty `GenerationBatchResult(logits_output=LogitsProcessorOutput(next_token_logits=None), can_run_cuda_graph=False)` (mirrors MLX `tp_worker.py:136-140`). |

---

## 9. P1 Acceptance Criteria

P1 is considered complete when all of:
1. Smoke test (§8.1) passes
2. Greedy correctness (§8.2) passes
3. Stability test (§8.4) passes
4. Performance log (§8.5) produced and committed to spec addendum
5. All unit tests (§8.6) pass
6. Reset script (§7.4) works on a wedged mesh
7. Spec self-review + user review complete

P1 explicitly does **not** require:
- Any throughput threshold
- Continuous batching
- Multi-request serving
- RadixAttention prefix cache benefit
- TP > 2

---

## 10. Risks

| # | Risk | Severity | Probability | Mitigation |
|---|---|---|---|---|
| 1 | tt_transformers does not support Llama-3.1-8B on Blackhole at chosen commit | **High** | Medium | Verified in Prerequisites §2.1 BEFORE coding starts. If verification fails, **pause this spec and renegotiate target**: either contribute Llama support upstream (out of scope for P1), wait for upstream, or rewrite this spec for a proven model (e.g. Qwen3) — do not silently swap models mid-implementation. |
| 2 | ttnn synchronous execution caps decode throughput at ~10–25 tok/s | Medium | High | Document as P1 ceiling. P3 batching may help. ttnn trace mode (beta) is a future avenue. |
| 3 | D2H per-token cost (5–15 ms) dominates at small batch sizes | Medium | High | P1 acceptable; P3 needs batching to amortize. |
| 4 | Mesh device leaks on crash → manual reset required | Low | Medium | Document recovery (§7.4). Consider auto-reset on startup with `--force-reset` flag (deferred). |
| 5 | tt-metal API drift between docker image versions | Medium | Medium | Pin docker image hash in spec. Don't track tt-metal main. |
| 6 | First-prefill JIT compile per unique shape (10–60s) confuses early benchmarking | Low | High | Warmup driver covers decode; document prefill JIT cost in observability output. |
| 7 | SGLang upstream changes break TT integration during rebase | Low | Medium | In-tree but minimal SGLang core diff (~150 LoC across ~17 files; see §4.2); easy to re-rebase. |
| 8 | P1 black-box approach masks subsystem issues (silent KV bugs in tt_transformers) | Low | Low | Greedy correctness test (§8.2) is end-to-end check; lm-eval subset gives broader signal. |
| 9 | Mesh OOM on long prompts. Memory math per card: 28 GB total − ~8 GB weights (16 GB BF16 ÷ TP=2) − ttnn framework overhead ~2 GB → **~18 GB KV budget**. Llama-3.1-8B GQA 8 KV heads × 32 layers × 128 head_dim × 2 (K+V) × 2B = 128 KB/token total → 64 KB/token per card. 18 GB / 64 KB ≈ 280k tokens KV per card → well above context_len 128k. | Low | Low | Bounded by context_len (128k); scheduler rejects beyond. |
| 10 | Llama-3.1-8B weights gated, HF token expiry mid-development | Low | Low | Documented as setup step. |
| 11 | **Silent-wrong-output hazard A**: implementer wires `model_runner.sample(logits_output, ForwardBatch.init_new(mwb, runner))` but `attn_backend=None` makes `ForwardBatch.init_new` produce zero positions / `support_triton` checks fail silently → sampler returns garbage tokens. | **High** if it happens | Low | §3.2 invariant #2 + §5.1 pseudocode mandate greedy-in-worker. Test (§8.2) catches via greedy mismatch with HF reference. Implementer must NOT add `model_runner.sample()` in P1. |
| 12 | **Silent-wrong-output hazard B**: `disable_overlap_schedule` not set → `FutureMap` allocates `-1` sentinel indices that the synchronous TT path never resolves → those `-1`s get appended to `req.output_ids` as token ids → output is malformed. | **High** if it happens | Medium (default is overlap=on) | §6.1 forces `disable_overlap_schedule = True`. Test (§8.1) smoke catches via "Paris" assertion. |

---

## 11. Phase 2 / Phase 3 Outline (deferred)

**P2 and P3 are NOT committed by this spec.** Their feasibility, value, and timing must be reassessed after P1 ships, based on observed P1 performance and stability.

### Phase 2 — TWO independent directions (pick one based on P1 results)

**P2-paged: Paged KV + per-layer attention (architectural inversion).** Replace tt_transformers' internal KV with SGLang-owned paged KV pool; enable RadixAttention prefix cache.

This requires:
- Forking or monkey-patching `tt_transformers.tt.attention` (the model's Attention layer) so it reads/writes our paged KV instead of its internal contiguous KV
- Implementing `TTPagedKVPool(MHATokenToKVPool)` backed by `ttnn.Tensor` mesh-sharded pages
- Re-doing the `tp_size` math: SGLang must see `tp_size=2` (for correct per-card KV sizing) while still single-process; this requires non-trivial work in worker init
- Registering a real `TTAttnBackend(AttentionBackend)` and adopting SGLang's per-layer attention dispatch

**Discovery (Phase 0.2):** `models.tt_transformers.tt.common.create_tt_model` natively accepts a `paged_attention_config` kwarg. P2-paged may be much smaller scope than originally feared — possibly just passing a `PagedAttentionConfig` through, not a fork/monkey-patch. Re-evaluate before committing.

P2-paged estimate (placeholder, not committed): 2-3 months — but may shrink to weeks if `paged_attention_config` does what we hope.

**P2-coverage: Add `TTXLAExecutionBackend` (model-coverage expansion).** Implement the second slot in the execution-backend registry to compile arbitrary PyTorch/JAX models via tt-xla (PJRT → StableHLO → TT-MLIR → TT-Metal). The wrapper interface (§3.2) doesn't change — only `execution/tt_xla_backend.py` and `execution/__init__.py` are touched. SGLang's worker/scheduler/etc. are platform-stable.

P2-coverage requires:
- Studying `tt-xla` PJRT contract + how `torch_xla` builds StableHLO graphs from `nn.Module`
- Implementing `TTXLAExecutionBackend.__init__` to JIT-compile a Llama (or arbitrary) model and persist the compiled artifact
- Mapping our 5-method lifecycle onto PJRT calls (prefill + decode are separate graph entry points)
- Performance comparison against the P1 `tt_transformers` baseline

P2-coverage estimate (placeholder, not committed): unknown — depends heavily on how mature tt-xla is by then. The win is generality: any HF PyTorch model becomes runnable, not just hand-ported tt_transformers ones.

**Decision rule:** if P1 perf is acceptable and Llama is the only model required, P2-paged for prefix caching. If model coverage is the priority, P2-coverage via tt-xla.

### Phase 3 — batched + continuous batching

Builds on P2:
- Multi-request batched forward in tt_transformers' Attention (or our re-impl)
- Scheduler tweaks for tile-size alignment (Blackhole 32-tile)
- Per-request KV gather in attention layer

P3 estimate (placeholder, not committed): 1–2 months on top of P2.

### Decision gate

Before committing to P2, re-spec based on P1 results. Specifically:
- If P1's 1-request tok/s is acceptable for the user's actual workload, P2 may not be needed.
- If P1 throughput is unacceptable but ttnn batching shows promise without paged KV, consider a lighter P2a.
- If tt_transformers proves too fragile, the answer may be Q (write our own ttnn Llama) — but that doubles the budget.

---

## 12. Open Questions

1. **Exact tt_transformers commit** to pin against — must be verified per Prerequisite §2.1.
2. **Decode warmup shape coverage** — `[1, 1]` is fixed, but does tt_transformers' decode path JIT once per layer or once total? Affects warmup wall time.
3. **Whether `ttnn.from_torch` for int32 prompt tokens** on Blackhole requires special handling (some integer ops may not be implemented).
4. **HF tokenizer locale handling** — Llama 3.1 uses tiktoken-style tokenizer; SGLang's existing path expected to work but unverified on this exact build.
5. **`reset_devices.sh`** safety — does `tt-smi -r` while a Python process holds the device handle leave kernel state consistent? Document the answer.

6. **SGLang API drift — must re-verify before coding each module.** All `server_args.X = Y` lines, class names, method names, kwarg names, and return types in this spec were validated against SGLang main at the commit checked out on 2026-05-11 via `grep` (see §6.1, §5.1, §6.2, §7). SGLang's internals drift across releases; the implementer **MUST** re-verify each load-bearing identifier against the SGLang revision they are building on before writing the corresponding TT code. High-risk identifiers, with grep anchors for verification:
   - `server_args.chunked_prefill_size` / `cpu_offload_gb` / `disable_radix_cache` / `sampling_backend` / `pre_warm_nccl` / `enable_dp_attention` / `enable_torch_compile` / `max_running_requests` / `tp_size` — `grep -n "^\s*<name>:" python/sglang/srt/server_args.py`
   - `class TpModelWorker` / `class BaseTpWorker` / `class GenerationBatchResult` — `grep -rn "^class <Name>" python/sglang/srt/managers/`
   - `forward_batch_generation` method shape and inherited signature — `grep -n "def forward_batch_generation" python/sglang/srt/managers/tp_worker.py`
   - `ForwardBatch.init_new` signature — `grep -n "def init_new" python/sglang/srt/model_executor/forward_batch_info.py`
   - `model_runner.sample` signature — `grep -n "def sample" python/sglang/srt/model_executor/model_runner.py`
   - `LogitsProcessorOutput` constructor fields — `grep -n "class LogitsProcessorOutput" python/sglang/srt/`
   - `ForwardMode.{EXTEND,DECODE,MIXED,SPLIT_PREFILL,IDLE,DLLM_EXTEND}` — `grep -nE "ForwardMode\.\w+ =|class ForwardMode" python/sglang/srt/`
   - CLI flag names: `--chunked-prefill-size`, `--disable-radix-cache`, etc. — `grep -nE "add_argument.\"--" python/sglang/srt/server_args.py | grep -E "<flag>"`

   If any check fails, fix the spec inline before writing the corresponding code, do not silently substitute a guess.

7. **Hidden assumptions** flagged by subagent review (verify each before coding):
   - `sampling_info`, `extend_prefix_lens`, `extend_seq_lens` are still populated on `ModelWorkerBatch` even though P1 skips `ForwardBatch.init_new` and sampling — needed if a future code path uses them
   - `req.fill_ids` / `req.prefix_indices` are populated for `EXTEND` even with `disable_radix_cache = True`
   - `ttnn.to_torch(logits_ttnn)` returns a CPU torch tensor (not pinned, not device-resident) suitable for direct host argmax
   - The `tt-metal` docker image's pre-installed torch version is compatible with the `pip install -e /sglang/python` torch dep. **§2.5 should be promoted to hard go/no-go**: if torch versions mismatch, get a known-good torch wheel pinned before coding starts, not after.

8. **`use_tt()` helper location**: verified — `use_mlx()` lives at `python/sglang/srt/utils/tensor_bridge.py:38`. Add `use_tt()` alongside it. **Critical distinction**: `use_mlx()` returns `bool(envs.SGLANG_USE_MLX.get()) and _MLX_AVAILABLE` (env-var gated). `use_tt()` must NOT introduce `SGLANG_USE_TT` — it should check platform-plugin activation: `isinstance(current_platform, TTSRTPlatform)` or equivalent. The gating semantics differ from MLX even though the function signature is the same.

9. **`tt_transformers` `Generator` API surface — capture in Phase 0.2.** Spec was originally anchored on a non-existent `Llama_3_1` class (corrected post-Phase-0 investigation 2026-05-11). The actual surfaces are `Generator` / `LlamaForCausalLM` / `create_tt_model`. **Still unverified**: which exact method names on `Generator` perform prefill vs decode vs KV-state-allocate-free. Phase 0.2 captures these and feeds them to Phase F's `TTLlamaWrapper` glue. **If `Generator` exposes only a `generate()` loop and no per-step entry points, the wrapper design needs to be redone or `tt_transformers` needs upstream hooks** — this is the live form of Risk #1 after the Llama_3_1 correction.

---

## 13. References

- Prior design doc (broader scope, superseded for P1): [`docs/platforms/tenstorrent_design.md`](../../platforms/tenstorrent_design.md)
- Tenstorrent vLLM fork: `github.com/tenstorrent/vllm`
- tt-metal: `github.com/tenstorrent/tt-metal`
- tt_transformers Llama demo: `models/tt_transformers/demo/simple_text_demo.py`
- **tt-xla** (alternative execution path; deprecates tt-torch): `github.com/tenstorrent/tt-xla` — reserved as a registered slot in `TT_EXECUTION_BACKENDS`; real impl post-P1 (see §11)
- SGLang MLX backend (architecture reference): `python/sglang/srt/hardware_backend/mlx/`
- SGLang platform interface: `python/sglang/srt/platforms/interface.py`
- SGLang attention-backend registry (pattern reference): `python/sglang/srt/layers/attention/attention_registry.py`

---

## Appendix A — File map

```
python/sglang/srt/hardware_backend/tenstorrent/
├── __init__.py
├── platform.py                  # TTSRTPlatform + activate_tt_platform + MeshDeviceCtx
├── tp_worker.py                 # TTTpModelWorker (forward entry)
├── model_runner.py              # TTModelRunner (bookkeeping stub)
├── model_runner_stub.py         # _DummyKVCache, _DummyModel
├── execution/                   # Swappable execution-backend layer (Phase F)
│   ├── __init__.py              # TT_EXECUTION_BACKENDS registry + factory + env resolve
│   ├── base.py                  # TTExecutionBackend(abc.ABC) — 5-method contract
│   ├── tt_transformers_backend.py  # TTTransformersExecutionBackend — P1's real impl
│   └── tt_xla_backend.py        # TTXLAExecutionBackend — post-P1 placeholder
├── warmup.py                    # decode-path warmup driver
├── scripts/
│   └── reset_devices.sh
└── test/
    ├── test_platform_activate.py
    ├── test_wrapper_lifecycle.py
    ├── test_execution_backend_registry.py
    ├── test_server_args_defaults.py
    ├── test_forward_mode_guard.py
    └── test_greedy_correctness.py     # @requires_tt
```

Plus edits to SGLang core (see §4.2 for the full ~17-file list with rationale per file).

---

## Appendix B — Subagent independent review record

A general-purpose subagent was tasked with an independent fresh-eyes review on 2026-05-11, after 5 rounds of author self-review. The review surfaced 5 🔴 critical findings and 7 🟡 medium findings, all of which have been incorporated into this spec. The most consequential corrections were:

1. **Sampler reality**: MLX uses `sampler = None` and does greedy in its own runner — `model_runner.sample()` is never invoked. P1 is greedy-only.
2. **`mwb.reqs[0].rid`** (not `mwb.req_ids[0]`) is the actual API.
3. **Scheduler dispatch wiring**: MLX is selected via `use_mlx()` branches in `scheduler.py`, not via entry-point alone. TT needs analogous `use_tt()` branches.
4. **`disable_overlap_schedule = True`** is mandatory; otherwise FutureMap creates `-1` sentinels that become garbage tokens.
5. **`server_args.device = "tenstorrent"` is unsafe** without overriding `model_runner.device = "cpu"`, because `scheduler.init_overlap` calls `torch.get_device_module(self.device)`.

The author's 5 prior self-review rounds caught architectural-level mistakes (e.g. P1→P2 framing, MLX-style integration choice, sampler/graph_runner=None requirement) but missed all 5 critical findings above. **Takeaway**: spec-level review by the author alone is insufficient for catching API-shape and dispatch-wiring errors. An independent reviewer with `grep` access is materially additive.

The medium-severity findings concerned: `IDLE` mode handling (must be allowed, not raised), CUDA-guard file count (revised from 9 to 17 files), `_DummyKVCache` factory should raise rather than silently succeed, signal handler process placement (Scheduler subprocess, not parent), and `Llama_3_1` import path requires independent verification beyond the existing demo-script proof. All have been incorporated above.
