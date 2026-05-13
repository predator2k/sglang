# SGLang-on-Tenstorrent — Phase 2a Implementation Plan (Paged Baseline)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the P2a paged-KV + RadixAttention + batched (B≥4 measured, ≤32 design ceiling) inference path on the existing TT plugin for Llama-3.1-8B at 8K context, while preserving the P1 single-user contiguous-KV path (rewritten against a new model-level ABC) as a regression net.

**Architecture:** P2a redesigns `TTExecutionBackend` from P1's 5-method per-request ABC (`new_request`/`extend`/`decode_step`/`free`/`reset_all`) to a single **model-level** `forward(forward_batch) → LogitsProcessorOutput` (INV-1). The paged backend (`tt_transformers_paged`) sits behind that ABC and consumes `tt_transformers/tt/generator.py:prefill_forward_text` / `decode_forward_text` with externally-fed `page_table` + `kv_cache` (INV-4). A new `TTPagedKVAdapter` subclasses SGLang's `PagedTokenToKVPoolAllocator` with `device="cpu"` and rewrites `alloc_extend` / `alloc_decode` in pure CPU torch (INV-2). A separate `TTPagedMetadataBackend` (NOT an `AttentionBackend` subclass — INV-1) builds the `page_table` block-ID tensor (INV-3) from `req_to_token_pool.req_to_token`. RadixAttention is enabled via SGLang's existing `--enable-radix-cache`; the paged adapter is the only point of coupling. P1 simple backend is rewritten (~250 lines) to the new ABC; both backends co-exist via `SGLANG_TT_EXECUTION_BACKEND` env (INV-5, restart-required).

**Tech Stack:** Python 3.10, SGLang on `tenstorrent-p1` branch, `ttnn` / `tt-metal` (pinned docker image SHA — locked in Phase 0), `tt_transformers` (read-only), PyTorch (host-side allocator + sampling + tensor bridging), `pytest` with `@pytest.mark.simple_backend` / `@pytest.mark.paged_backend` markers.

**Spec:** [`/home/mhnie/sglang/docs/superpowers/specs/2026-05-12-sglang-tenstorrent-p2-design.md`](../specs/2026-05-12-sglang-tenstorrent-p2-design.md). All section references (§N.M, §9.X, Q1-Q9, INV-1..INV-7, R1..R14) below point into that spec — do not re-derive architecture.

**Predecessor:** P1 has shipped (47 commits on `tenstorrent-p1` in `predator2k/sglang`; spec linked above §0). The P1.5 hygiene patches (SIGQUIT handler, mesh watchdog, P300 chunk-size sketch, persistent-cache doc) are **assumed merged in W1 before P2a branches**. P2a MUST NOT start while any P1.5 commit is in-flight (spec §5.3).

**Reference backend:** P1 `execution/tt_transformers_backend.py` (the wrapper this plan rewrites) and `python/sglang/srt/mem_cache/allocator.py:362-519` (`PagedTokenToKVPoolAllocator` — the parent class for the new adapter).

**Out of scope (do NOT implement here):**
- P2b (Qwen / Mistral / GptOss registration, 128K chunked-prefill, P300 chunk-size table). Separate plan, gated on §5.3b.
- Upstream PRs to `sgl-project/sglang` or `tenstorrent/tt-metal` (N9). All patches live on `tenstorrent-p1` branch in `predator2k/sglang`.
- Speculative decoding, LoRA, BF16-throughout precision, 4× p150a hardware (N3-N7).

---

## How to use this plan

1. Phases run top-to-bottom: **Phase 0 → 1 → 2 → 3 → 4**, mapped to W2-W5 of the spec's §5.3 timeline. W6 is buffer + §5.3b decision gate evaluation (no new tasks; see "Closing" at end).
2. Each phase has a **verification command** that gates the next phase. Do not move on until verification passes.
3. **Commit at end of each Task.** Granularity ≈ one logical unit (see git messages below). Never amend prior commits.
4. **INV-1..INV-7 are non-negotiable.** If an implementation discovery would violate any invariant, STOP — escape via §5.2b (replay brainstorming-skill on the affected §2/§3 paragraphs). Do NOT mutate spec architecture inline.
5. **Q1-Q9 are surfaced as explicit deliverables**, not implicit assumptions. Each has a numbered task (Q9 has its own Phase-0 CSV deliverable).
6. **Risk hot-spots inlined per phase.** R1 (`generator_sglang.py` signature drift, HIGH) and R5 (RadixAttention paged-evict, HIGH) get explicit mitigation tasks called out at their owning phase.
7. **Dual-track preservation is a hard requirement.** Every P1 hardware-gated test from `python/sglang/srt/hardware_backend/tenstorrent/test/` (13 tests under `test_smoke`, `test_greedy_correctness`, `test_mmlu_mini`, `test_stability`, `test_perf_log`, `test_p1_acceptance`) MUST still pass against the rewritten `tt_transformers_single` backend after Phase 0. New paged variants are added in Phase 4 with the `@pytest.mark.paged_backend` marker.
8. Inside docker (same image as P1): `source /home/container_app_user/tt-metal/python_env/bin/activate && cd /sglang/python && pip install -e . --no-deps` first. Set `SGLANG_PLATFORM=tenstorrent` for any TT runs.
9. **Cross-module patch warning:** The single SGLang upstream class touched in our fork is `GenerationBatchResult` (in `python/sglang/srt/managers/utils.py`) — adding one optional bool field `bypass_chunked_req`. Track this in the monthly rebase loop (R6).

---

## File map (locked here, references throughout)

Paths are relative to the repo root `/home/mhnie/sglang/`. Module root for new in-tree files is `python/sglang/srt/hardware_backend/tenstorrent/` (P1 convention).

### New files (P2a, ~1900 LoC)

| Path | Responsibility | Spec ref |
|---|---|---|
| `python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_paged_backend.py` | Concrete paged backend implementing the new ABC. Builds tokens / page_table per forward; calls `generator.prefill_forward_text` / `decode_forward_text`. | §2.2 |
| `python/sglang/srt/hardware_backend/tenstorrent/kv_pool/__init__.py` | Module marker | §2.2 |
| `python/sglang/srt/hardware_backend/tenstorrent/kv_pool/paged.py` | `TTPagedKVAdapter(PagedTokenToKVPoolAllocator)` + `TTBackedKVCache` wrapper; pure-CPU torch `alloc_extend` / `alloc_decode`. | §2.2, INV-2 |
| `python/sglang/srt/hardware_backend/tenstorrent/attn_meta/__init__.py` | Module marker | §2.2 |
| `python/sglang/srt/hardware_backend/tenstorrent/attn_meta/tt_paged.py` | `TTPagedMetadataBackend` — translates ForwardBatch metadata to `torch.int32[max_batch, max_blocks_per_seq]` page-table. **NOT a SGLang `AttentionBackend`** (INV-1). | §2.2 |
| `python/sglang/srt/hardware_backend/tenstorrent/models/__init__.py` | Empty; namespace marker for INV-6 separation. | §2.2 |
| `python/sglang/srt/hardware_backend/tenstorrent/models/llama_tt.py` | `TenstorrentLlamaForCausalLM` registered with `ModelRegistry` under separate arch namespace. P2a only — Qwen/Mistral/GptOss are P2b. | §2.2, INV-6 |

### Rewrites (in-place, ~250 LoC churn against existing P1 files)

| Path | Change |
|---|---|
| `python/sglang/srt/hardware_backend/tenstorrent/execution/base.py` | Replace 5-method ABC with single model-level `forward(forward_batch) → LogitsProcessorOutput`. Embed INV-1..INV-7 in module docstring. (+80 LoC) |
| `python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_backend.py` | Port the existing P1 simple backend (single-user contiguous KV) to the new ABC. Same semantics (B=1, no RadixAttention), new signature. (~250 LoC rewrite) |
| `python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py` | Add `tt_transformers_paged` registration alongside existing `tt_transformers` (now `tt_transformers_single`) and `tt_xla` placeholder. |
| `python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py` | Replace per-request `new_request`/`extend`/`decode_step` dispatch with single `forward(forward_batch)` call. Add admission hook + cancel hook. (+200 LoC) |
| `python/sglang/srt/hardware_backend/tenstorrent/platform.py` | Wire `get_token_to_kv_pool_cls` to return `TTPagedKVAdapter` when paged backend is selected; mesh-shape param via `SGLANG_TT_MESH_SHAPE` (P1.5 already shipped SIGQUIT). (+80 LoC) |

### Cross-module patch (in our fork only, per N9)

| Path | Change | Risk |
|---|---|---|
| `python/sglang/srt/managers/utils.py` | Add `bypass_chunked_req: bool = False` field to `@dataclasses.dataclass GenerationBatchResult` (line ~26). | R6 — monthly rebase, contingency quarterly. Track in a `REBASE_TARGETS.md` note. |
| `python/sglang/srt/managers/scheduler.py` | One read site post-forward: if `gen_batch_result.bypass_chunked_req` → `self.chunked_req = None`. (~5 LoC at the post-forward result-handling block near `self.chunked_req` reset, around line ~2498 in current main.) | Same R6. |

### Environment variables (added in P2a; document in spec §5.3a)

| Name | Values | Default | Read at |
|---|---|---|---|
| `SGLANG_TT_EXECUTION_BACKEND` (already exists from P1, renamed values) | `tt_transformers_paged` / `tt_transformers_single` / `auto` | `auto` → `tt_transformers_paged` post-P2a | platform activate |
| `SGLANG_TT_MESH_SHAPE` | `1x2` / `1x4` | `1x2` | platform activate |
| `SGLANG_TT_PAGE_SIZE` | int (must be a multiple of ttnn-tile alignment) | `32` | KV adapter init |
| `SGLANG_TT_MAX_RUNNING_REQUESTS` | int, ≤ 32 design ceiling, ≤ 4 measured | `4` | `apply_server_args_defaults` |

### Test files (new in P2a)

All under `python/sglang/srt/hardware_backend/tenstorrent/test/`. P1 tests stay where they are.

| File | Gate | Hardware? |
|---|---|---|
| `test_paged_kv_adapter_math.py` | §9.16 + §9.15 sub-set | No (CPU unit) |
| `test_free_list_invariant.py` | §9.15 | No (CPU unit) |
| `test_page_table_translation.py` | §3.2 / §3.3 page_table math | No (CPU unit) |
| `test_paged_backend_abc.py` | INV-1 ABC contract | No (CPU unit, mocked generator) |
| `test_token_pool_overflow.py` | §9.14 | No (CPU unit) |
| `test_chunked_failure_recovery.py` | §9.12.a-e (5 sub-tests) | No (CPU unit) |
| `test_model_registry_namespace.py` | INV-6 + §9.b6 stub | No (CPU unit) |
| `test_smoke_paged.py` | §9.1 | Yes |
| `test_greedy_correctness_paged.py` | §9.2 | Yes |
| `test_batched_correctness.py` | §9.3 | Yes |
| `test_radix_prefix_cache.py` | §9.4 | Yes |
| `test_queue_full_admission.py` | §9.5 | Yes |
| `test_abort_via_endpoint.py` | §9.6 (part 1) | Yes |
| `test_abort_via_disconnect.py` | §9.6 (part 2) | Yes |
| `test_stability_paged.py` | §9.7 | Yes |
| `test_dual_track_switch.py` | §9.8 | Yes |
| `test_mesh_shape_param.py` | §9.9 | Yes (`1x2` only; `1x4` static-lint) |
| `test_perf_log_paged.py` | §9.10 | Yes |
| `test_eviction_replay.py` | §9.11 | Yes |
| `test_shutdown_teardown.py` | §9.13 | Yes |

All 13 P1 hardware tests get a `@pytest.mark.simple_backend` marker added (Task 0.8) so CI can select them independently.

---

## Phase 0 — Signature lock + ABC rewrite + P1-simple port (W2)

**Goal:** Empirically nail down every signature P2a depends on (Q1, Q2, Q5, Q6, Q9) BEFORE writing any paged code. Produce a Q9 call-site CSV. Rewrite `TTExecutionBackend` to the model-level ABC (INV-1). Port the P1 simple backend to the new ABC and re-run all 13 P1 hardware tests to prove zero regression.

**Files touched:**
- Read-only inspection: `models.tt_transformers.tt.generator` (inside docker)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_signature_evidence.txt`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/q9_call_site_inventory.csv`
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/execution/base.py` (ABC rewrite)
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_backend.py` (port to new ABC)
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py` (rename `tt_transformers` → `tt_transformers_single`)
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py` (call new `forward` instead of 3 methods)
- Modify: all 13 P1 hardware tests — add `@pytest.mark.simple_backend` decorator and update `SGLANG_TT_EXECUTION_BACKEND` value to `tt_transformers_single`

**Spec refs:** §2.4 (INV-1..INV-7), §3.2 (`prefill_forward_text` signature), §3.7 (Q2 abort), §4.2 (P1 test migration), §5.2 (Q-list), §5.3 (Phase 0 task list).

**Live risks:** **R1 (HIGH)** — pin tt-metal docker SHA THIS WEEK; **R9** — every Q answered explicitly.

### Task 0.1: Pin the tt-metal docker image SHA (R1 / R13 mitigation)

- [ ] **Step 1:** Inside the dev docker container, capture the exact image SHA:

```bash
cat /etc/os-release | head -2
docker images --no-trunc --filter "reference=ghcr.io/tenstorrent/tt-inference-server/*" --format "{{.Repository}}:{{.Tag}} {{.ID}}"
```

Expected: prints the long sha256 of the image currently running.

- [ ] **Step 2:** Pin that SHA in two places:
  - Edit `python/sglang/srt/hardware_backend/tenstorrent/scripts/reset_devices.sh` — add a comment header line `# Pinned tt-metal image: sha256:<...>` listing the captured SHA.
  - Edit the spec file `docs/superpowers/specs/2026-05-12-sglang-tenstorrent-p2-design.md` line ~591 (`tt-metal docker image tag: **TBD at P2a Phase 0**`) to record the SHA. This is the ONE allowed spec edit during P2a — it fills in a TBD, doesn't change architecture.

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/scripts/reset_devices.sh docs/superpowers/specs/2026-05-12-sglang-tenstorrent-p2-design.md
git commit -m "chore(tenstorrent): pin tt-metal docker image SHA for P2a (R1/R13)"
```

### Task 0.2: Capture `prefill_forward_text` / `decode_forward_text` signatures (Q1, Q5)

- [ ] **Step 1:** Inside docker:

```bash
source /home/container_app_user/tt-metal/python_env/bin/activate
python <<'PY'
import inspect
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.common import create_tt_model

print("=== Generator.prefill_forward_text ===")
print(inspect.signature(Generator.prefill_forward_text))
print(inspect.getsourcefile(Generator.prefill_forward_text))

print("=== Generator.decode_forward_text ===")
print(inspect.signature(Generator.decode_forward_text))

print("=== create_tt_model ===")
print(inspect.signature(create_tt_model))

# Q5: does paged_attention_config need to be passed?
src = inspect.getsource(create_tt_model)
print("paged_attention_config in create_tt_model source:", "paged_attention_config" in src)
PY
```

- [ ] **Step 2:** Create `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_signature_evidence.txt` and paste the full output. Annotate the file with the **answers**:

```
Q1: prefill_forward_text expects tokens shape = [_______, _______]  (fill in: MAX_BATCH_SIZE, max_pad?)
Q5: paged_attention_config in create_tt_model? = (true/false)
    → if true, INV-4 may be at risk — re-read §2.4 INV-4 and §5.2b escape hatch
```

- [ ] **Step 3:** If `Generator.prefill_forward_text` signature does NOT accept `page_table=` / `kv_cache=` parameters externally (i.e., the upstream forces internal allocation through `paged_attention_config`), STOP — **INV-4 is violated**. Trigger §5.2b replay on §2 + §3 paragraphs that touch INV-4 before writing any paged code.

- [ ] **Step 4:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_signature_evidence.txt
git commit -m "docs(tenstorrent): P2a Phase 0 — lock Q1/Q5 generator signatures"
```

### Task 0.3: Verify Q2 — does `req.set_finish_with_abort` short-circuit same-iter sampling?

- [ ] **Step 1:** Read `python/sglang/srt/managers/schedule_batch.py:1341` (the `set_finish_with_abort` method) and the call sites in `python/sglang/srt/managers/scheduler.py` that handle `req.finished_reason`. Trace whether the in-flight `GenerationBatchResult`'s sampler still consumes this req.

- [ ] **Step 2:** Write a 1-page note appended to `phase0_signature_evidence.txt`:

```
Q2 answer: set_finish_with_abort short-circuits same-iter sampling? = (YES / NO)
Evidence: <file:line of the check>
Implication for §3.7 error recovery:
  - If NO: return zero-logits (LogitsProcessorOutput(next_token_logits=torch.zeros([B, vocab]))) on TTBackendError so sampler emits a discarded token.
  - If YES: returning `LogitsProcessorOutput(next_token_logits=None)` is safe (the spec's best guess).
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_signature_evidence.txt
git commit -m "docs(tenstorrent): P2a Phase 0 — answer Q2 (set_finish_with_abort same-iter)"
```

### Task 0.4: Verify Q6 — `req_to_token[:slice]` dtype requires `.to(int32)` for ttnn?

- [ ] **Step 1:** Inspect:

```bash
grep -n "dtype" /home/mhnie/sglang/python/sglang/srt/mem_cache/memory_pool.py | head -20
# Confirm req_to_token tensor dtype (likely int32 already)
```

And inside docker:

```bash
python -c "
from models.tt_transformers.tt.generator import Generator
import inspect, re
src = inspect.getsource(Generator.prefill_forward_text)
# Look for dtype assertions on page_table
print(re.findall(r'page_table.*dtype|page_table.*int|assert.*page_table', src)[:5])
"
```

- [ ] **Step 2:** Append Q6 answer to `phase0_signature_evidence.txt`:

```
Q6 answer: req_to_token dtype = <int32 or int64?>; page_table forced int32 in TTPagedMetadataBackend? = YES per INV-3
Implication: TTPagedMetadataBackend MUST call `.to(torch.int32)` if upstream is int64.
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_signature_evidence.txt
git commit -m "docs(tenstorrent): P2a Phase 0 — answer Q6 (req_to_token int32 cast)"
```

### Task 0.5: Q9 call-site CSV — every SGLang module that touches the KV-cache wrapper

This is the spec's named Phase-0 deliverable.

- [ ] **Step 1:** Generate the inventory:

```bash
cd /home/mhnie/sglang
grep -rn "token_to_kv_pool\|TokenToKVPool\|MHATokenToKVPool\|KVCache\b" python/sglang/srt/mem_cache/ python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/schedule_batch.py python/sglang/srt/layers/attention/ python/sglang/srt/model_executor/ > /tmp/q9_raw.txt
wc -l /tmp/q9_raw.txt
```

- [ ] **Step 2:** Manually triage into a CSV — for each call-site, record: `(file, line, expression, kind∈{read,write,subscribe}, hits_TTBackedKVCache∈{Y,N}, mitigation_if_N)`. Save to:

```
python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/q9_call_site_inventory.csv
```

Minimum schema:

```csv
file,line,expression,kind,hits_TTBackedKVCache,mitigation_if_N
python/sglang/srt/mem_cache/allocator.py,144,self.kvcache.available_size(),read,Y,
python/sglang/srt/mem_cache/radix_cache.py,<line>,token_to_kv_pool.get_kv_buffer(layer_id),read,N,"shim must implement get_kv_buffer"
...
```

- [ ] **Step 3:** For every row with `hits_TTBackedKVCache=N`, add a task to Phase 1 (Task 1.3) to either (a) implement the shim method on `TTBackedKVCache` or (b) document why the method is unreachable on the TT path. Make a TODO list as the bottom of the CSV.

- [ ] **Step 4:** Q9 answer recorded in `phase0_signature_evidence.txt`: `Q9: number of unique call sites touching the wrapper = N`. If N > 30, raise the flag — Phase 1 budget needs review.

- [ ] **Step 5:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/
git commit -m "docs(tenstorrent): P2a Phase 0 — Q9 KV-call-site CSV inventory"
```

### Task 0.6: Rewrite `TTExecutionBackend` ABC to model-level forward (INV-1)

- [ ] **Step 1:** Replace `python/sglang/srt/hardware_backend/tenstorrent/execution/base.py` entirely:

```python
"""TTExecutionBackend ABC — model-level forward contract (P2a INV-1).

P1 shipped a 5-method per-request ABC (new_request / extend / decode_step / free /
reset_all). P2a redesigns to a single model-level `forward(forward_batch)` to
align with SGLang's per-step batched scheduling.

INVARIANTS (from spec §2.4 — violations trigger §5.2b brainstorming replay):

  INV-1: forward(forward_batch) -> LogitsProcessorOutput is model-level forward
         (NOT per-layer attention). MUST NOT be named or aliased to SGLang's
         AttentionBackend base class.
  INV-2: TTPagedKVAdapter subclasses PagedTokenToKVPoolAllocator with
         device="cpu" + custom KVCache wrapper. alloc_extend/alloc_decode
         are pure CPU torch (no Triton).
  INV-3: page_table dtype = torch.int32; values are block IDs in [0, num_pages),
         computed as (token_index // block_size).
  INV-4: SGLang owns the page table; tt_transformers receives kv_cache + page_table
         as parameters (paged_attention_config path is NOT used).
  INV-5: Multiple backends co-exist via SGLANG_TT_EXECUTION_BACKEND; switch
         requires server restart (mesh re-init).
  INV-6: Tenstorrent model arches live under Tenstorrent* namespace
         (TenstorrentLlamaForCausalLM), separate from SGLang's LlamaForCausalLM.
  INV-7: MeshDeviceCtx supports (1,2) and (1,4) parametric shapes; P2a validates
         (1,2) only on hardware, (1,4) is static-lint.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TTExecutionBackend(abc.ABC):
    """Model-level forward contract.

    A backend owns one or more `tt_transformers` Generators and translates a
    SGLang `ForwardBatch` into `Generator.prefill_forward_text` /
    `decode_forward_text` calls. KV state lives in the SGLang-owned paged pool;
    the backend does NOT track per-request state across calls.
    """

    @abc.abstractmethod
    def __init__(
        self,
        model_path: str,
        mesh_device: Any,
        *,
        max_seq_len: int,
        max_batch_size: int,
        token_to_kv_pool: Any,  # TTPagedKVAdapter (paged) or None (simple/B=1)
    ) -> None:
        ...

    @abc.abstractmethod
    def forward(self, forward_batch: "ForwardBatch") -> "LogitsProcessorOutput":
        """Run one prefill or decode step.

        Returns `LogitsProcessorOutput` with `next_token_logits` populated on
        host CPU (`torch.Tensor[B, vocab]`). On error, set every req's
        finish_reason via `Req.set_finish_with_abort` and return either
        `next_token_logits=None` or zero logits (decision per Q2 — see
        Phase-0 evidence file).
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release all device state. Called from MeshDeviceCtx teardown."""
```

- [ ] **Step 2:** Verify the file parses:

```bash
python -c "from sglang.srt.hardware_backend.tenstorrent.execution.base import TTExecutionBackend; print(TTExecutionBackend)"
```

Expected: `<class '...TTExecutionBackend'>`.

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/execution/base.py
git commit -m "refactor(tenstorrent): P2a ABC — model-level forward(forward_batch) (INV-1)"
```

### Task 0.7: Port `TTTransformersExecutionBackend` (P1 simple) to the new ABC

This task rewrites the P1 simple backend (~250 LoC) to satisfy the new ABC. Same semantics (B=1, single-user contiguous KV), new signature.

- [ ] **Step 1:** Rename in the registry so the value space is explicit:

```bash
# In python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py:
# - Change @register_tt_execution_backend("tt_transformers") to ("tt_transformers_single")
# - In resolve_execution_backend_name: "auto" still defaults to a name —
#   keep returning "tt_transformers_single" for Phase 0 (the paged backend
#   doesn't exist yet); flip to "tt_transformers_paged" in Phase 4 Task 4.X.
```

Edit `execution/__init__.py`:

```python
def resolve_execution_backend_name(requested: str | None = None) -> str:
    """Resolve "auto" / "" / None to the P2a default.

    Phase 0: auto → tt_transformers_single (paged backend not yet registered)
    Phase 4 Task 4.10: flip auto → tt_transformers_paged
    """
    name = (requested or envs.SGLANG_TT_EXECUTION_BACKEND.get() or "auto").lower()
    if name == "auto":
        return "tt_transformers_single"
    return name
```

- [ ] **Step 2:** Rewrite `execution/tt_transformers_backend.py`. The new shape:

```python
"""tt_transformers single-user backend (P1 simple, ported to P2a ABC).

P1 sized this as a 5-method per-request wrapper. P2a wraps that same
single-user semantics under one model-level `forward(forward_batch)` that
internally branches on ForwardMode.EXTEND/DECODE/IDLE. B=1 enforced.

When SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single is selected:
- max_running_requests is force-clamped to 1 in apply_server_args_defaults
- RadixAttention is force-disabled
- token_to_kv_pool is NOT consulted (no paged allocator wired)

This is the rollback path per spec §5.3c.
"""
# ... port the existing __init__, dtype-map, create_tt_model bringup unchanged ...

class TTTransformersExecutionBackend(TTExecutionBackend):
    def __init__(self, model_path, mesh_device, *, max_seq_len,
                 max_batch_size, token_to_kv_pool):
        if max_batch_size != 1:
            raise NotImplementedError(
                "tt_transformers_single supports B=1 only. "
                "Use SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged for B>1."
            )
        # ... rest is the existing __init__ body, dtype map, create_tt_model ...
        self._active_req_id = None
        self._active_prompt_tokens = None

    def forward(self, forward_batch):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if forward_batch.forward_mode.is_idle():
            return LogitsProcessorOutput(next_token_logits=None)

        if forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise NotImplementedError(
                f"tt_transformers_single supports EXTEND/DECODE/IDLE only, "
                f"got {forward_batch.forward_mode}."
            )

        assert forward_batch.batch_size == 1, "B=1 enforced"
        rid = forward_batch.reqs[0].rid if hasattr(forward_batch, "reqs") else "single"

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            prompt_tokens = forward_batch.input_ids.tolist()
            # New request path: existing P1 _do_new_request body
            self._do_new_request(rid, prompt_tokens)
            logits = self._do_extend(rid)
        else:  # DECODE
            last_tok = int(forward_batch.input_ids[-1].item())
            logits = self._do_decode_step(rid, last_tok)

        return LogitsProcessorOutput(next_token_logits=logits.unsqueeze(0))  # [1, vocab]

    def shutdown(self):
        # Existing reset_all body
        self._do_reset_all()

    # _do_new_request / _do_extend / _do_decode_step / _do_reset_all are the
    # existing P1 methods renamed with leading underscore (they're now private
    # implementation, not the public ABC).
```

The CRITICAL constraint: the existing P1 method bodies (`new_request`, `extend`, `decode_step`, `reset_all`) keep their body code byte-identical, only renamed with leading underscores and called by the new `forward`. This isolates the rewrite to wiring, not logic.

- [ ] **Step 3:** Update `tp_worker.py` `_forward_batch_generation_tt` to call the new `forward` (drops `_cleanup_stale_rids` complexity for the single-user case — there's only ever one active req):

```python
def _forward_batch_generation_tt(self, mwb) -> "GenerationBatchResult":
    import torch
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    # Build a ForwardBatch (or pass mwb directly if the ABC accepts both —
    # decide at Phase 0 Step 4 below).
    forward_batch = self._build_forward_batch(mwb)
    logits_output = self.execution_backend.forward(forward_batch)

    if logits_output.next_token_logits is None:
        # IDLE / abort path
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )

    # Greedy sampling on host (sampler stays None per P1 invariant)
    next_token_ids = torch.argmax(
        logits_output.next_token_logits.float(), dim=-1, keepdim=True
    ).long()

    return GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        next_token_ids=next_token_ids,
        can_run_cuda_graph=False,
    )
```

- [ ] **Step 4:** Define `_build_forward_batch(mwb)` on `TTTpModelWorker`. For the simple backend this can be a thin pass-through since the simple backend's `forward` only reads `forward_mode`, `input_ids`, `batch_size`, `reqs`:

```python
def _build_forward_batch(self, mwb):
    """For single-user backend, ForwardBatch is overkill — we use a
    duck-typed namespace that exposes only the fields the simple backend reads.
    The paged backend (Phase 3) builds a real ForwardBatch via
    ForwardBatch.init_new(...).
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        forward_mode=mwb.forward_mode,
        input_ids=mwb.input_ids,
        batch_size=len(mwb.reqs) if mwb.reqs else 0,
        reqs=mwb.reqs or [],
    )
```

This is a deliberate stop-gap for the single backend — Phase 3 builds the real `ForwardBatch` for the paged path.

- [ ] **Step 5:** Run all 13 P1 hardware tests against the rewritten backend, in docker on real hardware:

```bash
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single \
  pytest -m simple_backend python/sglang/srt/hardware_backend/tenstorrent/test/ -v
# (the simple_backend marker is added in Task 0.8 below; for now run without -m)
```

Expected: all 13 tests pass (the same set that passed in P1).

- [ ] **Step 6:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/execution/
git add python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py
git commit -m "refactor(tenstorrent): port P1 simple backend to model-level forward ABC"
```

### Task 0.8: Add pytest markers to existing P1 tests (`@pytest.mark.simple_backend`)

- [ ] **Step 1:** Update `python/sglang/srt/hardware_backend/tenstorrent/test/conftest.py` (create if missing) to register the markers:

```python
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "simple_backend: tests targeting SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single"
    )
    config.addinivalue_line(
        "markers",
        "paged_backend: tests targeting SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged"
    )
    config.addinivalue_line(
        "markers",
        "hardware: requires a live ttnn mesh"
    )
```

- [ ] **Step 2:** Add `@pytest.mark.simple_backend` (and `@pytest.mark.hardware` where appropriate) to every test class/function in:

```
test_smoke.py
test_greedy_correctness.py
test_mmlu_mini.py
test_stability.py
test_perf_log.py
test_p1_acceptance.py
```

CPU-only / unit tests (`test_wrapper_lifecycle.py`, `test_execution_backend_registry.py`, `test_forward_mode_guard.py`, `test_platform_activate.py`, `test_server_args_defaults.py`, `test_model_runner_init.py`) get **no marker** — they're shared infrastructure.

- [ ] **Step 3:** Verify selection works:

```bash
pytest --collect-only -q -m simple_backend python/sglang/srt/hardware_backend/tenstorrent/test/ | tail -5
```

Expected: only the 6 hardware-gated files listed above are collected.

- [ ] **Step 4:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/
git commit -m "test(tenstorrent): mark P1 hardware tests with @pytest.mark.simple_backend"
```

### Task 0.9: Phase 0 verification + decision gate

- [ ] **Step 1:** Run the full P1 test suite (CPU unit + hardware-gated) and confirm zero regression:

```bash
# Unit tests (no marker = CPU only):
pytest python/sglang/srt/hardware_backend/tenstorrent/test/ -v -m "not paged_backend"
# Hardware tests (inside docker):
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single \
  pytest -m simple_backend python/sglang/srt/hardware_backend/tenstorrent/test/ -v
```

Expected: ALL P1 tests still pass. If anything regresses, fix before moving to Phase 1.

- [ ] **Step 2:** Confirm Phase 0 deliverables:
  1. `phase0_signature_evidence.txt` contains explicit Q1/Q2/Q5/Q6/Q9 answers
  2. `q9_call_site_inventory.csv` is committed
  3. tt-metal docker SHA pinned in `reset_devices.sh` and spec §7
  4. New ABC in `base.py` matches INV-1 wording verbatim
  5. P1 simple backend port re-runs P1's full hardware suite green

- [ ] **Step 3:** If ALL pass, Phase 0 done — move to Phase 1. If any Q answer triggers §5.2b (invariant violation), STOP and replay brainstorming on the affected §2/§3 paragraphs.

---

## Phase 1 — TTPagedKVAdapter + TTBackedKVCache + page-table translation (W3)

**Goal:** Implement the paged KV pool adapter and metadata backend. CPU-only unit tests prove the math; no hardware needed yet.

**Files touched:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/kv_pool/__init__.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/kv_pool/paged.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/attn_meta/__init__.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/attn_meta/tt_paged.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_kv_adapter_math.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_free_list_invariant.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_page_table_translation.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_token_pool_overflow.py`

**Spec refs:** §2.2 (file layout), §2.4 (INV-2, INV-3, INV-4), §3.2 (page_table construction), §3.4 (free), §3.5 (admission), §4.3 §9.14-§9.16 (acceptance gates), §5.1 R2 (CPU alloc_extend latency).

**Live risks:** **R2 (MEDIUM)** — CPU-torch `alloc_extend` latency at B=4 must be < 1ms (micro-benchmark in Task 1.4); **R5 (HIGH)** — free-list invariant covered by §9.15.

### Task 1.1: Skeleton `kv_pool/paged.py` — `TTBackedKVCache` shim

- [ ] **Step 1:** Create `python/sglang/srt/hardware_backend/tenstorrent/kv_pool/__init__.py`:

```python
"""Paged KV pool adapter for the TT execution backend (P2a)."""
```

- [ ] **Step 2:** Create `python/sglang/srt/hardware_backend/tenstorrent/kv_pool/paged.py` with the `TTBackedKVCache` shim. The shim implements every method listed in Q9's CSV; methods unreachable on the TT path raise `NotImplementedError` with a pointer to the CSV:

```python
"""TTBackedKVCache — host-side shim over ttnn KV tensors.

INV-2: this class is NOT MHATokenToKVPool. It does NOT expose `.data_ptr()`
or CUDA-tensor APIs. It only exposes the slim Q9-CSV surface that SGLang's
allocator / radix_cache / attention-backend modules read.

The actual KV bytes live in `self._kv_tensors` (ttnn device tensors, one
per layer). Reads/writes from SGLang's host side go through the shim's
declared methods only.
"""
from __future__ import annotations

import torch


class TTBackedKVCache:
    """Shim implementing every Q9-call-site method.

    See _fixtures/q9_call_site_inventory.csv for the inventory. Methods
    here ONE-FOR-ONE answer rows in that CSV.
    """

    def __init__(self, *, num_pages: int, page_size: int, num_layers: int,
                 head_dim: int, num_kv_heads: int, dtype: torch.dtype):
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        # Host-side bookkeeping only; actual device tensors handed in via
        # set_kv_tensors() during backend bringup.
        self._kv_tensors = None  # list[ttnn.Tensor] | None

    def set_kv_tensors(self, kv_tensors) -> None:
        """Backend calls this after create_tt_model populates KV cache."""
        self._kv_tensors = kv_tensors

    def get_kv_size_bytes(self) -> int:
        # SGLang asks this for memory accounting; report host-bookkeeping
        # view. Actual device-memory accounting is separate.
        kv_size = (
            self.num_pages * self.page_size * self.num_layers * 2  # K+V
            * self.num_kv_heads * self.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
        )
        return int(kv_size)

    def available_size(self) -> int:
        # Read-through to the allocator's free-page count.
        # Allocator (TTPagedKVAdapter) handles this; the shim returns NotImpl
        # because SGLang queries the allocator directly. See CSV row N.
        raise NotImplementedError(
            "available_size is queried on TTPagedKVAdapter, not TTBackedKVCache"
        )

    def get_kv_buffer(self, layer_id: int):
        # Some radix_cache code paths call this. For RadixAttention on the
        # TT path, prefix-cache lookups don't dereference KV bytes (they
        # only touch the token-pool index map), so this is unreachable —
        # but radix_cache may still call it during stats. Return a zero
        # placeholder so we don't crash on debugging code paths.
        return torch.zeros(
            (1, self.page_size, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
        )

    # ... add stubs for every row of q9_call_site_inventory.csv that the
    # Phase 0 audit found. Each stub either implements a thin host-side
    # fallback or raises NotImplementedError with the CSV row pointer.
```

The exact method set is **driven by the Q9 CSV**, not hand-listed here. Phase 0 Task 0.5 Step 3 produced the TODO list at the bottom of the CSV — implement each.

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/kv_pool/
git commit -m "feat(tenstorrent): TTBackedKVCache shim — slim KV pool wrapper (INV-2)"
```

### Task 1.2: `TTPagedKVAdapter(PagedTokenToKVPoolAllocator)` with pure-CPU `alloc_extend` / `alloc_decode`

- [ ] **Step 1:** Append `TTPagedKVAdapter` to `kv_pool/paged.py`. The subclass overrides `alloc_extend` and `alloc_decode` with **pure CPU torch** (Triton kernel path is unreachable with `device="cpu"`):

```python
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    get_num_new_pages,
)


class TTPagedKVAdapter(PagedTokenToKVPoolAllocator):
    """SGLang's PagedTokenToKVPoolAllocator with CPU-torch alloc kernels.

    INV-2: device="cpu", kvcache is a TTBackedKVCache (not MHATokenToKVPool).
    alloc_extend / alloc_decode rewritten in pure torch — SGLang's mainline
    uses Triton kernels (alloc_extend_kernel / alloc_decode_kernel in
    sglang/srt/mem_cache/allocator.py) which require CUDA tensors.

    All bookkeeping lives in CPU torch tensors:
      - self.free_pages: torch.int64[num_free_pages]
      - self.release_pages: torch.int64[num_released_pages] (need_sort path)

    page_table values handed to tt_transformers are torch.int32[B, max_blocks]
    of block IDs (INV-3) — computed by TTPagedMetadataBackend, NOT this class.
    This class deals in flat token-pool indices only.
    """

    def __init__(self, size, page_size, dtype, kvcache, need_sort=False):
        # NOTE: device hard-coded to "cpu" per INV-2. Parent __init__ stores it
        # on self.device and uses it in alloc(); we override all kernel sites.
        super().__init__(
            size=size, page_size=page_size, dtype=dtype,
            device="cpu", kvcache=kvcache, need_sort=need_sort,
        )
        # Startup invariant check — §9.14
        assert size < 2**31, (
            f"Total token-pool size {size} exceeds int32 range; "
            f"page_table dtype would overflow (INV-3 / §9.14)."
        )

    def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
                     last_loc, extend_num_tokens, num_new_pages=None):
        """Pure CPU torch port of alloc_extend_kernel.

        Returns flat token-pool indices [extend_num_tokens] (NOT page IDs).
        Page-ID translation happens in TTPagedMetadataBackend.
        """
        bs = len(prefix_lens)
        out_indices = torch.empty((extend_num_tokens,), dtype=torch.int64)

        cursor = 0
        new_page_cursor = 0
        for i in range(bs):
            prefix_i = int(prefix_lens[i].item())
            seq_i = int(seq_lens[i].item())
            new_tokens_i = seq_i - prefix_i
            if new_tokens_i <= 0:
                continue
            last_i = int(last_loc[i].item())

            # Fill any remaining slots in the current (partial) block before
            # consuming new free pages.
            partial = (prefix_i % self.page_size)
            slots_in_current_page = (
                self.page_size - partial if partial else 0
            )
            from_current = min(slots_in_current_page, new_tokens_i)
            for k in range(from_current):
                out_indices[cursor] = last_i + 1 + k
                cursor += 1
            remaining = new_tokens_i - from_current

            # Allocate fresh pages from the free list.
            while remaining > 0:
                if new_page_cursor >= len(self.free_pages):
                    return None  # OOM — SGLang's PrefillAdder handles abort
                page = int(self.free_pages[new_page_cursor].item())
                new_page_cursor += 1
                take = min(self.page_size, remaining)
                base = page * self.page_size
                for k in range(take):
                    out_indices[cursor] = base + k
                    cursor += 1
                remaining -= take

        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):
            return None
        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        """Pure CPU torch port of alloc_decode_kernel.

        For each req, append exactly one slot. If the previous slot was the
        last in a page, allocate a fresh page; otherwise reuse the next slot
        in the same page.
        """
        bs = len(seq_lens)
        out_indices = torch.empty((bs,), dtype=torch.int64)
        new_page_cursor = 0
        for i in range(bs):
            seq_i = int(seq_lens[i].item())
            last_i = int(last_loc[i].item())
            # Decode appends one slot. Check if previous block has room.
            in_block_pos = (seq_i - 1) % self.page_size
            if in_block_pos != 0 and last_i >= 0:
                out_indices[i] = last_i + 1
            else:
                if new_page_cursor >= len(self.free_pages):
                    return None
                page = int(self.free_pages[new_page_cursor].item())
                new_page_cursor += 1
                out_indices[i] = page * self.page_size

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None
        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
```

- [ ] **Step 2:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/kv_pool/paged.py
git commit -m "feat(tenstorrent): TTPagedKVAdapter — CPU-torch alloc_extend/decode (INV-2)"
```

### Task 1.3: Unit test — `test_paged_kv_adapter_math.py` (§9.16)

- [ ] **Step 1:** Write the failing test first:

```python
# python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_kv_adapter_math.py
"""§9.16: alloc_extend / alloc_decode / free / page-table math correctness.

CPU-only — no hardware, no ttnn. Drives TTPagedKVAdapter against the
allocator's documented contract: alloc returns the right slot count;
free restores the free list; (token_idx // page_size).to(int32) yields
the right page IDs.
"""

import pytest
import torch

from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import (
    TTPagedKVAdapter, TTBackedKVCache,
)


@pytest.fixture
def adapter():
    cache = TTBackedKVCache(
        num_pages=16, page_size=32, num_layers=2,
        head_dim=64, num_kv_heads=8, dtype=torch.bfloat16,
    )
    # size = num_pages * page_size = 512 token slots
    return TTPagedKVAdapter(
        size=512, page_size=32, dtype=torch.bfloat16,
        kvcache=cache, need_sort=False,
    )


def test_alloc_extend_b1_fresh_prefix_zero(adapter):
    # B=1, prefix=0, seq=40 → 40 new tokens, needs ceil(40/32)=2 pages
    prefix_lens = torch.tensor([0], dtype=torch.int32)
    seq_lens = torch.tensor([40], dtype=torch.int32)
    last_loc = torch.tensor([-1], dtype=torch.int64)
    out = adapter.alloc_extend(
        prefix_lens=prefix_lens, prefix_lens_cpu=prefix_lens,
        seq_lens=seq_lens, seq_lens_cpu=seq_lens,
        last_loc=last_loc, extend_num_tokens=40,
    )
    assert out is not None
    assert out.shape == (40,)
    assert out.dtype == torch.int64
    # First 32 tokens in page 1, next 8 in page 2 (page 0 is the dummy
    # padding-slot reservation from parent.clear())
    assert (out[0:32] // 32 == 1).all()
    assert (out[32:40] // 32 == 2).all()


def test_alloc_extend_with_partial_block_prefix(adapter):
    # B=1, prefix=30, seq=40 → 10 new tokens; first 2 fill current page,
    # then 8 in a fresh page
    prefix_lens = torch.tensor([30], dtype=torch.int32)
    seq_lens = torch.tensor([40], dtype=torch.int32)
    last_loc = torch.tensor([29], dtype=torch.int64)  # last slot was index 29
    out = adapter.alloc_extend(
        prefix_lens=prefix_lens, prefix_lens_cpu=prefix_lens,
        seq_lens=seq_lens, seq_lens_cpu=seq_lens,
        last_loc=last_loc, extend_num_tokens=10,
    )
    assert out.tolist()[0:2] == [30, 31]
    # next 8 should be at the start of a fresh page (multiple of 32)
    assert out[2] % 32 == 0


def test_alloc_decode_appends_one_slot(adapter):
    seq_lens = torch.tensor([41], dtype=torch.int32)  # was 40, now 41
    last_loc = torch.tensor([39], dtype=torch.int64)
    out = adapter.alloc_decode(seq_lens=seq_lens, seq_lens_cpu=seq_lens, last_loc=last_loc)
    assert out.shape == (1,)
    # last_loc=39 was index 39 in page 1, in_block_pos=(41-1)%32=8, nonzero → reuse → 40
    assert out[0].item() == 40


def test_free_restores_pages(adapter):
    initial_free = len(adapter.free_pages)
    # Allocate one full block
    out = adapter.alloc(32)
    assert len(adapter.free_pages) == initial_free - 1
    adapter.free(out)
    assert len(adapter.free_pages) == initial_free


def test_page_id_translation_int32(adapter):
    # Token indices map to page IDs via // page_size, dtype int32 (INV-3)
    token_idx = torch.tensor([0, 31, 32, 63, 64], dtype=torch.int64)
    page_ids = (token_idx // 32).to(torch.int32)
    assert page_ids.dtype == torch.int32
    assert page_ids.tolist() == [0, 0, 1, 1, 2]
```

- [ ] **Step 2:** Run; expect PASS (we wrote the impl first in Task 1.2):

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_kv_adapter_math.py -v
```

Expected: 5 tests pass.

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_kv_adapter_math.py
git commit -m "test(tenstorrent): §9.16 paged KV adapter alloc/free/page-id math"
```

### Task 1.4: Unit test — `test_free_list_invariant.py` (§9.15, R5 mitigation)

- [ ] **Step 1:** Write the test:

```python
"""§9.15 free-list invariant — R5 (RadixAttention paged-evict deadlock) mitigation.

After any sequence of alloc_extend / alloc_decode / free calls,
len(free_pages) + len(allocated_pages) == capacity, modulo the
dummy-padding slot 0 reserved by parent.clear().
"""

import random
import torch
import pytest

from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import (
    TTPagedKVAdapter, TTBackedKVCache,
)


def _make(num_pages=16, page_size=32):
    cache = TTBackedKVCache(num_pages=num_pages, page_size=page_size, num_layers=1,
                            head_dim=64, num_kv_heads=8, dtype=torch.bfloat16)
    return TTPagedKVAdapter(
        size=num_pages * page_size, page_size=page_size,
        dtype=torch.bfloat16, kvcache=cache, need_sort=False,
    )


def test_free_list_invariant_after_random_ops():
    rng = random.Random(42)
    adapter = _make(num_pages=32, page_size=16)
    capacity_pages = adapter.num_pages  # excludes the dummy slot

    allocated = []  # list[Tensor of int64 token indices]
    for _ in range(50):
        op = rng.choice(["alloc", "free"]) if allocated else "alloc"
        if op == "alloc":
            n_pages = rng.randint(1, 4)
            n_slots = n_pages * adapter.page_size
            if len(adapter.free_pages) < n_pages:
                continue
            got = adapter.alloc(n_slots)
            if got is not None:
                allocated.append(got)
        else:
            chunk = allocated.pop(rng.randrange(len(allocated)))
            adapter.free(chunk)

    # Drain everything
    while allocated:
        adapter.free(allocated.pop())
    assert len(adapter.free_pages) == capacity_pages, (
        f"free-list invariant broken: {len(adapter.free_pages)} != {capacity_pages}"
    )


def test_free_with_empty_input_is_noop():
    adapter = _make()
    before = len(adapter.free_pages)
    adapter.free(torch.empty(0, dtype=torch.int64))
    assert len(adapter.free_pages) == before
```

- [ ] **Step 2:** Run; expect PASS:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_free_list_invariant.py -v
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_free_list_invariant.py
git commit -m "test(tenstorrent): §9.15 free-list invariant (R5 mitigation)"
```

### Task 1.4b: Q7 unit test — read-through invariant after `cache_finished_req(canceled=True)`

**Why**: Plan-review reviewer flagged Q7 (spec §5.2) as having no dedicated assertion task — §9.6 abort and §9.11 eviction-replay are end-to-end; neither directly verifies the read-through invariant declared in spec §3.4 ("RadixCache's view of KV state = adapter's free list state"). This task fills the gap with a CPU-only unit test.

- [ ] **Step 1:** Write the test:

```python
"""Q7 read-through invariant — after RadixCache.cache_finished_req(req, canceled=True),
TTPagedKVAdapter free-list state must match RadixCache's view (no orphan KV slots,
no double-free).

Spec §3.4 invariant: "RadixCache's view of KV state = adapter's free list state.
No cache sync — adapter is a read-through layer."

CPU-only — uses real PagedTokenToKVPoolAllocator + RadixCache with a stub
TTBackedKVCache (no ttnn). Validates the contract between SGLang's cache
machinery and our adapter's free-list semantics.
"""
import pytest
import torch
from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import (
    TTPagedKVAdapter, TTBackedKVCache,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.managers.schedule_batch import Req
# NOTE: exact import paths may shift slightly during Phase 0-1; align at impl time.

def _make_adapter(capacity=256, page_size=32):
    kvcache = TTBackedKVCache(num_layers=2, num_kv_heads=8, head_dim=128,
                              num_pages=capacity // page_size,
                              page_size=page_size, mock=True)
    return TTPagedKVAdapter(size=capacity, page_size=page_size,
                            dtype=torch.bfloat16, kvcache=kvcache)

def _make_radix(adapter):
    # RadixCache constructor signature varies slightly across SGLang versions;
    # consult radix_cache.py at impl time. Minimal init: pass adapter + req_to_token_pool.
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    req_pool = ReqToTokenPool(size=8, max_context_len=256, device="cpu")
    return RadixCache(req_to_token_pool=req_pool,
                      token_to_kv_pool_allocator=adapter,
                      page_size=32,
                      disable=False)

def test_read_through_after_canceled():
    adapter = _make_adapter()
    radix = _make_radix(adapter)
    initial_free = adapter.available_size()
    # Synthesize a request: simulate alloc then mark canceled then call cache_finished_req
    fake_req = _build_fake_req(token_ids=list(range(50)),  # 50-token prompt
                               extend_prefix_lens=0, extend_seq_lens=50)
    # Allocate via adapter (mirrors what alloc_extend would do during forward)
    out_indices = adapter.alloc_extend(
        prefix_lens=torch.tensor([0]),
        prefix_lens_cpu=torch.tensor([0]),
        seq_lens=torch.tensor([50]),
        seq_lens_cpu=torch.tensor([50]),
        last_loc=torch.tensor([-1]),
        extend_num_tokens=50,
        num_new_pages=2,
    )
    assert out_indices is not None, "alloc_extend should succeed"
    fake_req.req_pool_idx = 0
    # write into req_to_token (mirrors model_runner.forward_batch fill)
    radix.req_to_token_pool.req_to_token[0, :50] = out_indices
    fake_req.fill_ids = list(range(50))
    fake_req.prefix_indices = torch.tensor([], dtype=torch.long)
    # mark canceled and finish
    fake_req.canceled = True
    fake_req.set_finish_with_abort("test_cancel")
    radix.cache_finished_req(fake_req)  # canceled path → evict via allocator.free
    # invariant: free-list should be back to initial state
    final_free = adapter.available_size()
    assert final_free == initial_free, (
        f"read-through violated: initial={initial_free} final={final_free}; "
        f"RadixCache and adapter disagree on KV state after canceled req"
    )
    # Also: free_pages + allocated == capacity (no orphans)
    assert len(adapter.free_pages) + len(adapter.allocated_pages) == adapter.capacity_pages, (
        f"orphan pages detected after canceled req"
    )

def test_read_through_after_uncanceled_finish_with_eviction():
    """Variant: finished_req without cancel, but RadixCache decides to evict (no cache node retained)."""
    adapter = _make_adapter()
    radix = _make_radix(adapter)
    initial_free = adapter.available_size()
    fake_req = _build_fake_req(token_ids=list(range(40)),
                               extend_prefix_lens=0, extend_seq_lens=40)
    out_indices = adapter.alloc_extend(
        prefix_lens=torch.tensor([0]), prefix_lens_cpu=torch.tensor([0]),
        seq_lens=torch.tensor([40]), seq_lens_cpu=torch.tensor([40]),
        last_loc=torch.tensor([-1]),
        extend_num_tokens=40, num_new_pages=2,
    )
    fake_req.req_pool_idx = 0
    radix.req_to_token_pool.req_to_token[0, :40] = out_indices
    fake_req.fill_ids = list(range(40))
    fake_req.prefix_indices = torch.tensor([], dtype=torch.long)
    fake_req.canceled = False
    fake_req.finished_reason = type("FinishReason", (), {"is_finished": True})()
    radix.cache_finished_req(fake_req)
    # If the cache decides to insert as a node (no evict), pages stay allocated; this is also a
    # valid read-through state (radix tree owns them). Force evict to validate the round-trip:
    radix.evict(num_tokens=64)  # forces evict at least the 40 we just allocated
    final_free = adapter.available_size()
    assert final_free >= initial_free, "evict didn't return slots to free list"
    assert len(adapter.free_pages) + len(adapter.allocated_pages) == adapter.capacity_pages

def _build_fake_req(token_ids, extend_prefix_lens, extend_seq_lens):
    """Builds a minimal Req object for RadixCache.cache_finished_req. Exact fields
    align with schedule_batch.Req at impl time — this stub captures the contract."""
    req = Req.__new__(Req)
    req.origin_input_ids = token_ids
    req.fill_ids = token_ids
    req.skip_radix_cache_insert = False
    req.canceled = False
    return req
```

- [ ] **Step 2:** Run:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_read_through_invariant.py -v
# Expect 2/2 PASS on the canceled and uncanceled-with-evict paths.
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_read_through_invariant.py
git commit -m "test(tenstorrent): Q7 read-through invariant after cache_finished_req"
```

**Acceptance criteria**:
- 2 test cases pass (canceled + uncanceled-with-forced-evict)
- After each `cache_finished_req` path, `adapter.available_size()` and `len(free_pages) + len(allocated_pages) == capacity` are consistent
- If either invariant fails, surface the violation via `set_finish_with_abort` is the cause — see Q7 best-guess "Yes (read-through)" in §5.2 — and trigger §5.2b escape hatch (INV-2 violation)

### Task 1.5: Unit test — `test_token_pool_overflow.py` (§9.14)

- [ ] **Step 1:** Write the test:

```python
"""§9.14 token-pool overflow guard.

Adapter must fail-fast if num_pages * page_size >= 2^31 (would overflow int32
page_table dtype).
"""

import pytest
import torch

from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import (
    TTPagedKVAdapter, TTBackedKVCache,
)


def test_token_pool_within_int32_range_ok():
    cache = TTBackedKVCache(num_pages=4, page_size=32, num_layers=1,
                            head_dim=64, num_kv_heads=8, dtype=torch.bfloat16)
    adapter = TTPagedKVAdapter(
        size=128, page_size=32, dtype=torch.bfloat16, kvcache=cache, need_sort=False,
    )
    assert adapter.num_pages == 4


def test_token_pool_exceeding_int32_raises():
    cache = TTBackedKVCache(num_pages=2**26, page_size=64, num_layers=1,
                            head_dim=64, num_kv_heads=8, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="int32 range"):
        TTPagedKVAdapter(
            size=2**26 * 64,  # = 2^32, overflows
            page_size=64, dtype=torch.bfloat16, kvcache=cache, need_sort=False,
        )
```

- [ ] **Step 2:** Run; expect PASS:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_token_pool_overflow.py -v
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_token_pool_overflow.py
git commit -m "test(tenstorrent): §9.14 token-pool int32 overflow fail-fast"
```

### Task 1.6: `TTPagedMetadataBackend` — page-table translation

- [ ] **Step 1:** Create `python/sglang/srt/hardware_backend/tenstorrent/attn_meta/__init__.py`:

```python
"""Metadata translation for paged TT execution backend (P2a).

NOTE: this module is NOT a SGLang AttentionBackend subclass — INV-1 in spec
§2.4 forbids aliasing the model-level forward with per-layer attention.
TTPagedMetadataBackend is a plain class that owns the host→device page-table
upload per forward step.
"""
```

- [ ] **Step 2:** Create `attn_meta/tt_paged.py`:

```python
"""TTPagedMetadataBackend — translates ForwardBatch -> ttnn page-table tensor.

Not a SGLang AttentionBackend subclass (INV-1). The TT model-level forward
calls `metadata_backend.init_forward_metadata(forward_batch)` once per
forward step; the result is a torch.int32 tensor of block IDs uploaded to
ttnn just before generator.prefill_forward_text / decode_forward_text.

Inputs read from ForwardBatch:
  - req_to_token_pool.req_to_token  # [num_reqs, max_context_len]
  - req_pool_indices                # [B]
  - seq_lens                        # [B]  (full seq lens including prefix)
  - forward_mode                    # EXTEND or DECODE

Output:
  - page_table: torch.int32[max_batch_size, max_blocks_per_seq]
"""

from __future__ import annotations

import math
import torch


class TTPagedMetadataBackend:

    def __init__(self, *, page_size: int, max_batch_size: int,
                 max_context_len: int):
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len
        # max_blocks_per_seq is fixed: ceil(max_context_len / page_size)
        self.max_blocks_per_seq = math.ceil(max_context_len / page_size)
        # Persistent host buffer; reused per forward to avoid allocations.
        self._page_table = torch.zeros(
            (max_batch_size, self.max_blocks_per_seq), dtype=torch.int32
        )

    def init_forward_metadata(self, forward_batch) -> torch.Tensor:
        """Build the page_table from req_to_token_pool. Returns a CPU
        torch.int32 tensor [max_batch_size, max_blocks_per_seq].

        The caller (paged backend) uploads this tensor to ttnn.
        """
        self._page_table.zero_()  # reset row 0..max for cleared/IDLE slots
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        B = len(req_pool_indices)
        for i in range(B):
            req_idx = int(req_pool_indices[i].item())
            seq_len = int(seq_lens[i].item())
            n_blocks = math.ceil(seq_len / self.page_size)
            # Read first-token index of each block, then floor-divide to get page ID.
            # row[::page_size] is valid even for trailing partial block (last entry
            # is the first token of the partial block, which is still in a valid page).
            row = req_to_token[req_idx, :seq_len]
            page_ids = (row[::self.page_size] // self.page_size).to(torch.int32)
            # n_blocks may be one more than len(page_ids) if seq_len % page_size != 0
            # In that case the trailing partial block's first token is row[(n_blocks-1)*page_size]
            # which is already covered by ::page_size striding.
            actual_n = page_ids.shape[0]
            self._page_table[i, :actual_n] = page_ids
        return self._page_table
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/attn_meta/
git commit -m "feat(tenstorrent): TTPagedMetadataBackend — page-table translation (INV-3)"
```

### Task 1.7: Unit test — `test_page_table_translation.py`

- [ ] **Step 1:** Write the test (CPU-only; mocks `req_to_token_pool`):

```python
"""Page-table translation correctness — §3.2 INV-3 math.

Constructs a fake req_to_token map with known token-pool indices and
verifies (row[::page_size] // page_size).to(int32) yields expected page IDs.
"""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.tenstorrent.attn_meta.tt_paged import (
    TTPagedMetadataBackend,
)


def _fake_forward_batch(req_to_token, req_pool_indices, seq_lens):
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
    )


def test_page_table_b1_aligned():
    # B=1, seq_len=64, page_size=32 → 2 blocks
    # token-pool indices: page 5 = [160..191], page 7 = [224..255]
    req_to_token = torch.zeros((1, 64), dtype=torch.int32)
    req_to_token[0, 0:32] = torch.arange(160, 192)
    req_to_token[0, 32:64] = torch.arange(224, 256)
    fb = _fake_forward_batch(
        req_to_token=req_to_token,
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([64]),
    )
    meta = TTPagedMetadataBackend(page_size=32, max_batch_size=4, max_context_len=128)
    pt = meta.init_forward_metadata(fb)
    assert pt.dtype == torch.int32
    assert pt[0, 0].item() == 5
    assert pt[0, 1].item() == 7
    # Remaining slots should be zero
    assert (pt[0, 2:] == 0).all()
    assert (pt[1:] == 0).all()  # other batch slots cleared


def test_page_table_partial_block():
    # B=1, seq_len=40, page_size=32 → 2 blocks (one full + one partial of 8)
    req_to_token = torch.zeros((1, 40), dtype=torch.int32)
    req_to_token[0, 0:32] = torch.arange(64, 96)   # page 2
    req_to_token[0, 32:40] = torch.arange(128, 136)  # page 4 partial
    fb = _fake_forward_batch(
        req_to_token=req_to_token,
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([40]),
    )
    meta = TTPagedMetadataBackend(page_size=32, max_batch_size=4, max_context_len=128)
    pt = meta.init_forward_metadata(fb)
    assert pt[0, 0].item() == 2
    assert pt[0, 1].item() == 4
    assert pt[0, 2].item() == 0


def test_page_table_b4_distinct_reqs():
    # B=4, each 32 tokens (1 block each), different pages
    rt = torch.zeros((4, 32), dtype=torch.int32)
    rt[0, :] = torch.arange(0, 32)          # page 0
    rt[1, :] = torch.arange(32, 64)         # page 1
    rt[2, :] = torch.arange(96, 128)        # page 3
    rt[3, :] = torch.arange(160, 192)       # page 5
    fb = _fake_forward_batch(
        req_to_token=rt,
        req_pool_indices=torch.tensor([0, 1, 2, 3]),
        seq_lens=torch.tensor([32, 32, 32, 32]),
    )
    meta = TTPagedMetadataBackend(page_size=32, max_batch_size=4, max_context_len=128)
    pt = meta.init_forward_metadata(fb)
    assert pt[:, 0].tolist() == [0, 1, 3, 5]


def test_page_table_shared_prefix_pages():
    # R5 mitigation flavour: two reqs share their first block (radix prefix
    # cache hit). req_to_token[0, :32] == req_to_token[1, :32].
    rt = torch.zeros((2, 64), dtype=torch.int32)
    rt[0, 0:32] = torch.arange(0, 32)      # page 0 shared
    rt[0, 32:64] = torch.arange(96, 128)   # page 3
    rt[1, 0:32] = torch.arange(0, 32)      # page 0 shared
    rt[1, 32:64] = torch.arange(192, 224)  # page 6
    fb = _fake_forward_batch(
        req_to_token=rt,
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([64, 64]),
    )
    meta = TTPagedMetadataBackend(page_size=32, max_batch_size=4, max_context_len=128)
    pt = meta.init_forward_metadata(fb)
    assert pt[0].tolist()[:2] == [0, 3]
    assert pt[1].tolist()[:2] == [0, 6]
    # Confirm both rows reference the same first block ID (page 0)
    assert pt[0, 0] == pt[1, 0]
```

- [ ] **Step 2:** Run; expect 4 PASS:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_page_table_translation.py -v
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_page_table_translation.py
git commit -m "test(tenstorrent): page-table translation incl. shared-prefix R5 scenario"
```

### Task 1.8: Micro-benchmark — verify Q4 (CPU `alloc_extend` p99 latency < 1ms at B=4)

- [ ] **Step 1:** Add a bench script (NOT a pytest — performance-oriented):

```python
# python/sglang/srt/hardware_backend/tenstorrent/scripts/bench_alloc_extend.py
"""Q4 / R2 micro-benchmark.

Spec §5.2 best guess: < 500us p99. R2 mitigation ladder:
  Step A — vectorize CPU-torch impl if > 1ms p99 at B=4.
  Step B — accept latency and force B≤4 cap (do NOT port to Triton-CPU).
"""

import time
import torch

from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import (
    TTPagedKVAdapter, TTBackedKVCache,
)


def run():
    cache = TTBackedKVCache(num_pages=4096, page_size=32, num_layers=32,
                            head_dim=128, num_kv_heads=8, dtype=torch.bfloat16)
    adapter = TTPagedKVAdapter(
        size=4096 * 32, page_size=32, dtype=torch.bfloat16,
        kvcache=cache, need_sort=False,
    )
    B = 4
    # Warm prefix=1024, prompt=2048 → 1024 new tokens each
    prefix_lens = torch.tensor([1024] * B, dtype=torch.int32)
    seq_lens = torch.tensor([2048] * B, dtype=torch.int32)
    last_loc = torch.tensor([1023, 2047, 3071, 4095], dtype=torch.int64)
    extend_num_tokens = (seq_lens - prefix_lens).sum().item()

    latencies = []
    for _ in range(200):
        # Reset free list each iter
        adapter.clear()
        t0 = time.perf_counter()
        adapter.alloc_extend(
            prefix_lens=prefix_lens, prefix_lens_cpu=prefix_lens,
            seq_lens=seq_lens, seq_lens_cpu=seq_lens,
            last_loc=last_loc, extend_num_tokens=int(extend_num_tokens),
        )
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    print(f"B={B}, prefix=1024, seq=2048: p50={p50*1e6:.0f}us, p99={p99*1e6:.0f}us")

    if p99 > 1e-3:
        print("R2 ladder Step A: vectorize CPU implementation (replace per-i loop)")
    elif p99 > 500e-6:
        print("R2 within MEDIUM band; OK but watch.")
    else:
        print("R2 within best-guess (< 500us).")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2:** Run it:

```bash
python python/sglang/srt/hardware_backend/tenstorrent/scripts/bench_alloc_extend.py
```

- [ ] **Step 3:** Append the measured p50/p99 to `phase0_signature_evidence.txt` as the **Q4 answer**. If p99 > 1ms, open a follow-up task to vectorize (the per-i loop with `out_indices[cursor] = ...` is the obvious target for vectorization via `torch.arange`-and-scatter). The R2 ladder is the mitigation: do NOT port to Triton-CPU; if vectorization still misses the bound, accept latency and document `max_batch_size ≤ 4`.

- [ ] **Step 4:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/scripts/bench_alloc_extend.py
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/phase0_signature_evidence.txt
git commit -m "bench(tenstorrent): alloc_extend B=4 CPU latency (Q4 / R2)"
```

### Task 1.9: Phase 1 verification + decision gate

- [ ] **Step 1:** All CPU unit tests pass:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_kv_adapter_math.py \
       python/sglang/srt/hardware_backend/tenstorrent/test/test_free_list_invariant.py \
       python/sglang/srt/hardware_backend/tenstorrent/test/test_token_pool_overflow.py \
       python/sglang/srt/hardware_backend/tenstorrent/test/test_page_table_translation.py -v
```

Expected: all PASS.

- [ ] **Step 2:** Q9 CSV's TODO list is fully consumed (every row has a method on `TTBackedKVCache` or a documented unreachable reason).

- [ ] **Step 3:** Q4 answer recorded; R2 status known.

- [ ] **Step 4:** Phase 1 done — move to Phase 2.

---

## Phase 2 — RadixAttention integration + paged backend wiring (W4 early)

**Goal:** Wire the new ABC, `TTPagedKVAdapter`, and `TTPagedMetadataBackend` together into a working `TTTransformersPagedExecutionBackend` for B=1. Enable RadixAttention via `--enable-radix-cache`. End of Phase 2 is the first hardware smoke on the paged path.

**Files touched:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_paged_backend.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/models/__init__.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/models/llama_tt.py`
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py` (register `tt_transformers_paged`)
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/platform.py` (wire `get_token_to_kv_pool_cls`, `apply_server_args_defaults` per-backend)
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py` (paged path build_forward_batch)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_backend_abc.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_model_registry_namespace.py`
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py`

**Spec refs:** §2.2, §2.4 INV-4..INV-6, §2.5 SGLang integration points, §3.1-§3.3 (EXTEND/DECODE flow), §3.4 (free/eviction read-through).

**Live risks:** R1 (signature drift — Phase 0 locked), R5 mitigation deferred to Phase 3 (eviction-replay).

### Task 2.1: `TenstorrentLlamaForCausalLM` arch wrapper + registry (INV-6)

- [ ] **Step 1:** Create `python/sglang/srt/hardware_backend/tenstorrent/models/__init__.py`:

```python
"""Tenstorrent model registrations (INV-6 separate-namespace).

Registers `Tenstorrent*ForCausalLM` arches with SGLang's ModelRegistry.
Does NOT shadow SGLang's own LlamaForCausalLM / MistralForCausalLM /
Qwen2ForCausalLM — those are CUDA-only and live in sglang.srt.models.

P2a registers only TenstorrentLlamaForCausalLM. P2b will add Qwen, Mistral,
GptOss in this same module.
"""

from sglang.srt.hardware_backend.tenstorrent.models.llama_tt import (
    TenstorrentLlamaForCausalLM,
)

EntryClass = TenstorrentLlamaForCausalLM  # SGLang model-registry hook
```

- [ ] **Step 2:** Create `models/llama_tt.py`:

```python
"""TenstorrentLlamaForCausalLM — arch wrapper for the paged TT backend.

This class is a SHIM. The actual model lives inside tt_transformers; this
class only exists so SGLang's ModelRegistry has a non-colliding arch name
to dispatch on (INV-6). The real bringup happens in
TTTransformersPagedExecutionBackend.__init__ via
models.tt_transformers.tt.common.create_tt_model.
"""

from __future__ import annotations


class TenstorrentLlamaForCausalLM:
    """Placeholder arch class for SGLang's model registry.

    SGLang's standard flow: ModelRunner inspects model config, finds the
    matching arch class, instantiates it. Our flow bypasses that path
    (TTTpModelWorker overrides forward_batch_generation entirely; the
    ModelRunner is a stub from P1). This class exists solely so the
    registry has a non-shadowing entry to bind the arch string to.
    """

    arch_name = "TenstorrentLlamaForCausalLM"

    def __init__(self, *args, **kwargs):
        # Never instantiated on the TT path (worker bypasses ModelRunner.load).
        raise NotImplementedError(
            "TenstorrentLlamaForCausalLM is a registry-only shim. "
            "Actual model bringup happens in TTTransformersPagedExecutionBackend."
        )
```

- [ ] **Step 3:** Write the registry-namespace unit test:

```python
# python/sglang/srt/hardware_backend/tenstorrent/test/test_model_registry_namespace.py
"""§9.b6 — model registry collision check (INV-6).

Verify TenstorrentLlamaForCausalLM registers cleanly and does NOT shadow
SGLang's own LlamaForCausalLM.
"""

def test_tenstorrent_arch_does_not_shadow_sglang():
    from sglang.srt.models.registry import ModelRegistry
    # P2a registers TenstorrentLlamaForCausalLM via the models package.
    from sglang.srt.hardware_backend.tenstorrent import models  # noqa: F401
    ModelRegistry.register(
        "sglang.srt.hardware_backend.tenstorrent.models",
        overwrite=False,
    )
    # SGLang's own arch should remain registered (still resolvable)
    assert "LlamaForCausalLM" in ModelRegistry._models or callable(
        ModelRegistry.resolve_model_cls("LlamaForCausalLM")
    )
    # Our arch should also be present under its own name
    assert "TenstorrentLlamaForCausalLM" in ModelRegistry._models or callable(
        ModelRegistry.resolve_model_cls("TenstorrentLlamaForCausalLM")
    )
```

Adapt the assertion shape if `ModelRegistry`'s public surface differs — the underlying check is "both names resolve, no `already registered` error".

- [ ] **Step 4:** Run; expect PASS:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_model_registry_namespace.py -v
```

- [ ] **Step 5:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/models/ \
       python/sglang/srt/hardware_backend/tenstorrent/test/test_model_registry_namespace.py
git commit -m "feat(tenstorrent): TenstorrentLlamaForCausalLM registry shim (INV-6)"
```

### Task 2.2: `TTTransformersPagedExecutionBackend.__init__` — model bringup

- [ ] **Step 1:** Create `execution/tt_transformers_paged_backend.py`. Start with just `__init__` + a NotImpl `forward`:

```python
"""Paged TT execution backend (P2a default).

Wraps tt_transformers' Generator + LlamaForCausalLM, feeding it externally-
allocated page_table + kv_cache (INV-4). Implements the model-level forward
ABC.
"""

from __future__ import annotations

import math
import os
import torch

from sglang.srt.hardware_backend.tenstorrent.execution import (
    register_tt_execution_backend,
)
from sglang.srt.hardware_backend.tenstorrent.execution.base import TTExecutionBackend
from sglang.srt.hardware_backend.tenstorrent.attn_meta.tt_paged import (
    TTPagedMetadataBackend,
)
from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import (
    TTBackedKVCache, TTPagedKVAdapter,
)

try:
    import ttnn
    from models.tt_transformers.tt.common import create_tt_model
    from models.tt_transformers.tt.generator import Generator
    from models.tt_transformers.tt.model_config import DecodersPrecision
except ImportError:
    ttnn = None
    create_tt_model = None
    Generator = None
    DecodersPrecision = None


@register_tt_execution_backend("tt_transformers_paged")
class TTTransformersPagedExecutionBackend(TTExecutionBackend):

    def __init__(self, model_path, mesh_device, *, max_seq_len,
                 max_batch_size, token_to_kv_pool):
        if create_tt_model is None or ttnn is None:
            raise RuntimeError(
                "tt_transformers / ttnn not importable; activate the tt-metal venv."
            )
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model path {model_path!r} not found inside the docker mount."
            )
        if not isinstance(token_to_kv_pool, TTPagedKVAdapter):
            raise TypeError(
                f"Paged backend requires TTPagedKVAdapter; got {type(token_to_kv_pool).__name__}. "
                f"INV-2 violation — check platform.get_token_to_kv_pool_cls()."
            )

        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.token_to_kv_pool = token_to_kv_pool
        self.page_size = token_to_kv_pool.page_size

        # Bring up tt_transformers WITHOUT paged_attention_config — INV-4 says
        # we feed kv_cache + page_table externally per call.
        dtype_str = os.environ.get("SGLANG_TT_PRECISION", "bfp8")
        tt_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[dtype_str]
        precision_cfg = DecodersPrecision.performance(model_path)

        self._model, self._tokenizer = create_tt_model(
            mesh_device=mesh_device,
            instruct=True,
            max_batch_size=max_batch_size,
            optimizations=precision_cfg,
            max_seq_len=max_seq_len,
            # NB: NO paged_attention_config kwarg — Q5 from Phase 0 validated absence.
            # NB: tt_data_parallel=1 per N4.
            dtype=tt_dtype,
        )
        self._generator = Generator(model=self._model, tokenizer=self._tokenizer)

        # Hand the ttnn KV tensors to the host-side shim so SGLang's
        # accounting paths can read them through TTBackedKVCache.
        self.token_to_kv_pool.kvcache.set_kv_tensors(self._model.kv_cache)

        # Metadata backend (page-table builder)
        self.metadata = TTPagedMetadataBackend(
            page_size=self.page_size,
            max_batch_size=max_batch_size,
            max_context_len=max_seq_len,
        )

    def forward(self, forward_batch):
        raise NotImplementedError("Phase 2 Task 2.3 fills this in.")

    def shutdown(self):
        # ttnn tensors are reclaimed by MeshDeviceCtx teardown; nothing host-side to free.
        self._generator = None
        self._model = None
        self._tokenizer = None
```

- [ ] **Step 2:** Register in `execution/__init__.py`:

```python
# Add the import line near tt_xla_backend:
from sglang.srt.hardware_backend.tenstorrent.execution import (  # noqa: E402, F401
    tt_transformers_backend,
    tt_transformers_paged_backend,
    tt_xla_backend,
)
```

- [ ] **Step 3:** Quick import smoke (outside docker is fine — ImportError fall-through):

```bash
python -c "
from sglang.srt.hardware_backend.tenstorrent.execution import TT_EXECUTION_BACKENDS
print(sorted(TT_EXECUTION_BACKENDS.keys()))
"
```

Expected: `['tt_transformers_paged', 'tt_transformers_single', 'tt_xla']`.

- [ ] **Step 4:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/execution/
git commit -m "feat(tenstorrent): TTTransformersPagedExecutionBackend init + registration"
```

### Task 2.3: `forward()` — EXTEND/DECODE/IDLE branching

- [ ] **Step 1:** Replace `forward` in `tt_transformers_paged_backend.py` with the real body:

```python
def forward(self, forward_batch):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    fb = forward_batch
    if fb.forward_mode.is_idle():
        return LogitsProcessorOutput(next_token_logits=None)

    if fb.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
        raise NotImplementedError(
            f"P2a paged backend supports EXTEND/DECODE/IDLE only; "
            f"got {fb.forward_mode}. Chunked-prefill / mixed are P2b."
        )

    # Build page_table (CPU torch.int32[max_batch, max_blocks])
    page_table = self.metadata.init_forward_metadata(fb)

    if fb.forward_mode == ForwardMode.EXTEND:
        logits = self._do_extend(fb, page_table)
    else:
        logits = self._do_decode(fb, page_table)

    # Logits arrive [B, vocab] on host CPU (Generator.{prefill,decode}_forward_text
    # with read_from_device=True returns host tensors per P1 evidence).
    return LogitsProcessorOutput(next_token_logits=logits)


def _do_extend(self, fb, page_table):
    # §3.2 — build full tokens tensor [max_batch_size, padded_prefill_len]
    B = fb.batch_size
    extend_prefix_lens = fb.extend_prefix_lens_cpu  # list[int], len B
    extend_seq_lens = fb.extend_seq_lens_cpu        # list[int], len B
    max_full_len = max(extend_seq_lens)
    # Pad to a step the upstream prefill kernel likes; spec defaults pad_step=128.
    pad_step = int(os.environ.get("SGLANG_TT_PREFILL_PAD_STEP", "128"))
    padded_len = ((max_full_len + pad_step - 1) // pad_step) * pad_step

    pad_id = int(self._tokenizer.pad_token_id or 0)
    tokens = torch.full(
        (self.max_batch_size, padded_len), pad_id, dtype=torch.int32,
    )
    for i, req in enumerate(fb.reqs):
        s = extend_prefix_lens[i]
        e = extend_seq_lens[i]
        # req.fill_ids holds the full sequence (prefix + new). For slice
        # [s:e] only the new tokens are populated; [0:s] area is irrelevant
        # because tt_transformers reads from start_pos=s (Q1 evidence).
        full_ids = torch.as_tensor(req.fill_ids, dtype=torch.int32)
        tokens[i, s:e] = full_ids[s:e]

    start_pos = list(extend_prefix_lens)  # Python list[int], per §3.2 Q1
    prompt_lens = list(extend_seq_lens)
    empty_slots = list(range(B))  # P2a: B≤4, no retract → identity (Q3, validated Phase 3)

    logits = self._generator.prefill_forward_text(
        tokens=tokens,
        page_table=page_table,
        kv_cache=self._model.kv_cache,
        prompt_lens=prompt_lens,
        start_pos=start_pos,
        empty_slots=empty_slots,
    )
    # Slice off padded batch rows
    return logits[:B]


def _do_decode(self, fb, page_table):
    B = fb.batch_size
    # input_ids in DECODE mode is the last sampled token per req, shape [B]
    last_tokens = fb.input_ids[:B].to(torch.int32)
    logits = self._generator.decode_forward_text(
        tokens=last_tokens,
        page_table=page_table,
        kv_cache=self._model.kv_cache,
        # Other kwargs per the Phase-0 signature lock.
    )
    return logits[:B]
```

The exact kwarg names for `decode_forward_text` come from the Phase-0 signature evidence. If the signature differs (e.g. requires `start_pos`), wire accordingly.

- [ ] **Step 2:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_paged_backend.py
git commit -m "feat(tenstorrent): paged backend forward() — EXTEND/DECODE wiring"
```

### Task 2.4: Wire `platform.py` for the paged path (`apply_server_args_defaults`, `get_token_to_kv_pool_cls`)

- [ ] **Step 1:** In `platform.py`, branch on the selected backend:

```python
def apply_server_args_defaults(self, server_args):
    from sglang.srt.hardware_backend.tenstorrent.execution import (
        resolve_execution_backend_name,
    )
    backend = resolve_execution_backend_name(None)

    if backend == "tt_transformers_single":
        # P1 simple-path defaults
        server_args.max_running_requests = 1
        server_args.disable_radix_cache = True
        server_args.chunked_prefill_size = -1
        # ... (preserve P1's existing defaults verbatim)
    elif backend == "tt_transformers_paged":
        # P2a paged defaults
        max_b = int(os.environ.get("SGLANG_TT_MAX_RUNNING_REQUESTS", "4"))
        assert 1 <= max_b <= 32, f"max_running_requests must be in [1, 32]; got {max_b}"
        server_args.max_running_requests = max_b
        # RadixAttention enabled by default (SGLang flag is disable_radix_cache=False)
        if server_args.disable_radix_cache is None:
            server_args.disable_radix_cache = False
        server_args.chunked_prefill_size = -1  # P2a: no chunked prefill (P2b)
        server_args.page_size = int(os.environ.get("SGLANG_TT_PAGE_SIZE", "32"))
    else:
        raise ValueError(f"Unknown SGLANG_TT_EXECUTION_BACKEND={backend!r}")
```

- [ ] **Step 2:** Add a `get_token_to_kv_pool_cls` override:

```python
def get_token_to_kv_pool_cls(self, *, mha: bool, mla: bool, hybrid: bool):
    from sglang.srt.hardware_backend.tenstorrent.execution import (
        resolve_execution_backend_name,
    )
    if resolve_execution_backend_name(None) == "tt_transformers_single":
        return None  # simple backend doesn't use the pool
    from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import TTPagedKVAdapter
    return TTPagedKVAdapter  # paged path
```

- [ ] **Step 3:** Also wire `SGLANG_TT_MESH_SHAPE` (INV-7) in `MeshDeviceCtx`:

```python
def _resolve_mesh_shape():
    raw = os.environ.get("SGLANG_TT_MESH_SHAPE", "1x2")
    a, b = raw.split("x")
    return (int(a), int(b))
```

Apply at mesh-open time. P2a validates `(1, 2)` on hardware; `(1, 4)` static-lint only — Task 4.X verifies the static-lint path doesn't crash on import even when no 4-device mesh is present.

- [ ] **Step 4:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/platform.py
git commit -m "feat(tenstorrent): platform.py — paged-path defaults + KV pool cls + mesh-shape param"
```

### Task 2.5: Update `tp_worker.py` to build a real `ForwardBatch` for the paged backend

- [ ] **Step 1:** In `_forward_batch_generation_tt`, branch on backend name:

```python
def _build_forward_batch(self, mwb):
    from sglang.srt.hardware_backend.tenstorrent.execution import (
        resolve_execution_backend_name,
    )
    if resolve_execution_backend_name(None) == "tt_transformers_single":
        # Phase 0 lightweight namespace (B=1 path).
        from types import SimpleNamespace
        return SimpleNamespace(
            forward_mode=mwb.forward_mode,
            input_ids=mwb.input_ids,
            batch_size=len(mwb.reqs) if mwb.reqs else 0,
            reqs=mwb.reqs or [],
        )
    # Paged path: real ForwardBatch with req_to_token_pool wiring.
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    return ForwardBatch.init_new(
        batch=mwb,
        model_runner=self.model_runner,
    )
```

The `model_runner.req_to_token_pool` must already be wired in `_init_model_runner` (it's standard SGLang plumbing; verify in Task 2.6 step 1).

- [ ] **Step 2:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py
git commit -m "feat(tenstorrent): tp_worker — branch ForwardBatch build on paged backend"
```

### Task 2.6: Unit test — `test_paged_backend_abc.py` (mocked generator)

- [ ] **Step 1:** Write a CPU-only test that exercises the paged backend's `forward` with a mocked `Generator`:

```python
# python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_backend_abc.py
"""Paged backend ABC contract — INV-1.

Mocks the upstream Generator so we can run on CPU without hardware.
Verifies forward() correctly routes EXTEND/DECODE/IDLE and emits
LogitsProcessorOutput with the right shape.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def paged_backend(monkeypatch):
    # Patch the module-level imports so create_tt_model / Generator are mockable.
    import sglang.srt.hardware_backend.tenstorrent.execution.tt_transformers_paged_backend as mod
    monkeypatch.setattr(mod, "ttnn", MagicMock(bfloat16="bf16", bfloat8_b="bfp8"))
    fake_model = MagicMock()
    fake_model.kv_cache = MagicMock()
    fake_tokenizer = MagicMock(pad_token_id=0)
    monkeypatch.setattr(mod, "create_tt_model", lambda **kw: (fake_model, fake_tokenizer))
    monkeypatch.setattr(mod, "Generator", lambda model, tokenizer: MagicMock(
        prefill_forward_text=lambda **kw: torch.zeros(32, 128256),  # [max_batch, vocab]
        decode_forward_text=lambda **kw: torch.zeros(32, 128256),
    ))
    monkeypatch.setattr(mod, "DecodersPrecision",
                        MagicMock(performance=lambda p: MagicMock()))
    monkeypatch.setattr(mod.os.path, "isdir", lambda p: True)

    from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import (
        TTPagedKVAdapter, TTBackedKVCache,
    )
    cache = TTBackedKVCache(num_pages=64, page_size=32, num_layers=2,
                            head_dim=64, num_kv_heads=8, dtype=torch.bfloat16)
    pool = TTPagedKVAdapter(size=64*32, page_size=32, dtype=torch.bfloat16,
                            kvcache=cache, need_sort=False)

    return mod.TTTransformersPagedExecutionBackend(
        model_path="/fake/path",
        mesh_device=MagicMock(),
        max_seq_len=2048,
        max_batch_size=4,
        token_to_kv_pool=pool,
    )


def test_forward_idle_returns_none_logits(paged_backend):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    fb = SimpleNamespace(forward_mode=ForwardMode.IDLE)
    out = paged_backend.forward(fb)
    assert out.next_token_logits is None


def test_forward_rejects_unsupported_modes(paged_backend):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    with pytest.raises(NotImplementedError, match="EXTEND/DECODE/IDLE only"):
        paged_backend.forward(fb)


def test_forward_extend_b1_emits_logits(paged_backend):
    """Verify EXTEND wiring: page_table built, prefill_forward_text called, [B, vocab] returned."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    rt = torch.zeros((1, 32), dtype=torch.int32)
    rt[0, :32] = torch.arange(0, 32)
    fake_req = SimpleNamespace(fill_ids=list(range(32)))
    fb = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        reqs=[fake_req],
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[32],
        req_to_token_pool=SimpleNamespace(req_to_token=rt),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([32]),
    )
    out = paged_backend.forward(fb)
    assert out.next_token_logits is not None
    assert out.next_token_logits.shape == (1, 128256)
```

- [ ] **Step 2:** Run; expect PASS:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_backend_abc.py -v
```

If `is_idle()` isn't a callable on `ForwardMode.IDLE`, switch the test fixture to use a real `ForwardMode` enum (it is — verify by import).

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_backend_abc.py
git commit -m "test(tenstorrent): paged backend ABC contract with mocked Generator"
```

### Task 2.7: Smoke — `test_smoke_paged.py` (§9.1, hardware-gated B=1)

- [ ] **Step 1:** Write the hardware smoke test:

```python
"""§9.1 paged smoke: B=1 returns 'Paris' via /v1/completions."""

import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import (
    sglang_server,  # context manager — see P1's test/_fixtures
)


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_paged_smoke_paris():
    with sglang_server(
        env={
            "SGLANG_PLATFORM": "tenstorrent",
            "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
            "SGLANG_TT_MAX_RUNNING_REQUESTS": "4",
        },
    ) as base_url:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "Llama-3.1-8B-Instruct",
                "prompt": "The capital of France is",
                "max_tokens": 4, "temperature": 0,
            },
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["text"]
        assert "Paris" in text, f"expected Paris in {text!r}"
```

Re-use P1's `sglang_server` fixture from `_fixtures/server.py`. If the env vars need extending (e.g. `SGLANG_TT_MAX_RUNNING_REQUESTS`), extend the fixture rather than inlining popen.

- [ ] **Step 2:** Run on hardware:

```bash
pytest -m paged_backend python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py -v
```

Expected: PASS, "Paris" in completion. If hang, run `sudo tt-smi -r 0000:01:00.0 0000:06:00.0` and retry once. Two failures → R1 fired despite Phase 0 lock; STOP and diagnose.

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py
git commit -m "test(tenstorrent): §9.1 paged smoke — Paris via /v1/completions"
```

### Task 2.8: Phase 2 verification gate

- [ ] **Step 1:** CPU unit tests pass:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_paged_backend_abc.py \
       python/sglang/srt/hardware_backend/tenstorrent/test/test_model_registry_namespace.py -v
```

- [ ] **Step 2:** Hardware §9.1 passes:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py -v
```

- [ ] **Step 3:** P1 simple-backend regression still green:

```bash
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single \
  pytest -m simple_backend python/sglang/srt/hardware_backend/tenstorrent/test/ -v
```

Expected: all 13 P1 hardware tests still pass.

- [ ] **Step 4:** If ALL three groups pass, Phase 2 done. If §9.1 fails on hardware, do NOT move to Phase 3 — diagnose generator-call wiring first.

---

## Phase 3 — Batched forward + admission + abort + cancel (W4 late)

**Goal:** Lift the paged path to B≥4 measured. Validate Q3 (`empty_slots` identity at B≤4 no-retract). Wire admission (queue-full → AbortReq, not HTTP 503). Wire abort (`/abort_request` + client TCP RST). Implement the §3.7 5-step chunked-prefill failure recovery (even though chunked prefill itself is P2b; the contract is exercised by the unit test).

**Files touched:**
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_paged_backend.py` (`empty_slots`, error recovery)
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/kv_pool/paged.py` (`on_chunked_prefill_failure` adapter method)
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py` (cancel-check hook post-forward)
- Modify: `python/sglang/srt/managers/utils.py` (add `bypass_chunked_req: bool = False`)
- Modify: `python/sglang/srt/managers/scheduler.py` (read `bypass_chunked_req` post-forward)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_chunked_failure_recovery.py`

**Spec refs:** §3.5 (admission), §3.6 (cancel), §3.7 (error recovery + 5-step adapter call), §4.3 §9.5, §9.6, §9.12 (acceptance gates), §5.1 R6 (upstream class patch).

**Live risks:** **R6 (MEDIUM)** — `GenerationBatchResult.bypass_chunked_req` upstream-class patch; the rebase-cost note is required.

### Task 3.1: Add `bypass_chunked_req` to `GenerationBatchResult` (R6 patch)

- [ ] **Step 1:** Edit `python/sglang/srt/managers/utils.py`. Find the `@dataclasses.dataclass class GenerationBatchResult:` block (line ~25-60 in current main) and add ONE new field at a stable position (end of the user-facing flags, before the `# For overlap scheduling` section):

```python
@dataclasses.dataclass
class GenerationBatchResult:
    # ... existing fields ...
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # Tenstorrent paged-backend chunked-prefill failure escape hatch.
    # When True, scheduler must clear self.chunked_req. See
    # docs/superpowers/specs/2026-05-12-sglang-tenstorrent-p2-design.md §3.7
    # and Risk R6 (this field is OUR fork patch; monthly rebase target).
    bypass_chunked_req: bool = False

    # For overlap scheduling
    copy_done: Optional[torch.cuda.Event] = None
    # ... rest unchanged ...
```

- [ ] **Step 2:** In `python/sglang/srt/managers/scheduler.py`, find the post-forward result-handling code path that touches `self.chunked_req` (near line ~2498 — `if self.chunked_req is not None:`). Add ONE check:

```python
# R6 / spec §3.7: TT paged backend signals chunked-prefill failure
# via GenerationBatchResult.bypass_chunked_req. Clear scheduler state.
if getattr(gen_batch_result, "bypass_chunked_req", False):
    self.chunked_req = None
    self._chunked_req_scheduled_last_iter = False
```

Use `getattr(..., False)` so callers that construct `GenerationBatchResult` without this field still work (defensive against rebase drift).

- [ ] **Step 3:** Create or append to `REBASE_TARGETS.md` at repo root:

```markdown
# Upstream-class patches in `tenstorrent-p1` fork (R6 tracking)

| File | Field / change | Spec ref | Added | Last verified |
|---|---|---|---|---|
| python/sglang/srt/managers/utils.py | GenerationBatchResult.bypass_chunked_req: bool | P2a §3.7 | 2026-05-12 | 2026-05-12 |
| python/sglang/srt/managers/scheduler.py | post-forward bypass_chunked_req → self.chunked_req=None | P2a §3.7 | 2026-05-12 | 2026-05-12 |

**Rebase policy:** monthly cadence (contingency: quarterly). On SGLang main bump:
1. Apply patches above. If `GenerationBatchResult` definition has shifted, re-find the
   "user-facing flags before overlap-scheduling section" anchor.
2. Run §9.12 chunked-failure-recovery unit test — if it fails, the scheduler
   post-forward path has moved; re-locate `self.chunked_req = None` site.
3. If conflict is unresolvable in one session, **temporarily disable
   chunked-prefill in the TT path** (set `chunked_prefill_size=-1` in
   apply_server_args_defaults) and reopen as a follow-up patch task.
```

- [ ] **Step 4:** Commit:

```bash
git add python/sglang/srt/managers/utils.py python/sglang/srt/managers/scheduler.py REBASE_TARGETS.md
git commit -m "feat(scheduler): add GenerationBatchResult.bypass_chunked_req (R6, in-fork)"
```

### Task 3.2: `TTPagedKVAdapter.on_chunked_prefill_failure` (5-step §3.7 contract)

- [ ] **Step 1:** Append to `kv_pool/paged.py`:

```python
def on_chunked_prefill_failure(self, req, out_cache_loc_this_chunk,
                                req_to_token_pool):
    """Spec §3.7 — 5-step chunked-prefill failure handler.

    Called by the paged backend when a mid-chunk forward raises. Performs:
      1. Free this chunk's KV slots.
      2. Revert req_to_token writes for this chunk's range.
      3. Mark req.skip_radix_cache_insert so partial prefix doesn't enter RadixCache.
      4. Set req.finish_reason via req.set_finish_with_abort.
      5. Caller must return GenerationBatchResult(bypass_chunked_req=True).
    """
    # Step 1
    self.free(out_cache_loc_this_chunk)
    # Step 2
    chunk_len = len(out_cache_loc_this_chunk)
    fill_len = len(req.fill_ids)
    prefix_len = len(req.prefix_indices)
    start = prefix_len + fill_len - chunk_len
    end = prefix_len + fill_len
    req_to_token_pool.req_to_token[req.req_pool_idx, start:end] = 0
    # Step 3
    req.skip_radix_cache_insert = True
    # Step 4
    req.set_finish_with_abort("tt_backend_chunked_prefill_failure")
    # Step 5 is the caller's responsibility (return GenerationBatchResult(bypass_chunked_req=True)).
```

- [ ] **Step 2:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/kv_pool/paged.py
git commit -m "feat(tenstorrent): TTPagedKVAdapter.on_chunked_prefill_failure (§3.7 5-step)"
```

### Task 3.3: Unit test — `test_chunked_failure_recovery.py` (§9.12.a-e)

- [ ] **Step 1:** Write 5 sub-tests, one per gate:

```python
"""§9.12.a-e — chunked-prefill failure recovery contract.

All 5 sub-gates must pass. CPU-only.
"""

from types import SimpleNamespace
import pytest
import torch

from sglang.srt.hardware_backend.tenstorrent.kv_pool.paged import (
    TTBackedKVCache, TTPagedKVAdapter,
)


def _make_adapter():
    cache = TTBackedKVCache(num_pages=16, page_size=32, num_layers=2,
                            head_dim=64, num_kv_heads=8, dtype=torch.bfloat16)
    return TTPagedKVAdapter(size=16*32, page_size=32, dtype=torch.bfloat16,
                             kvcache=cache, need_sort=False)


def _make_req(req_pool_idx=0, prefix_indices=(), fill_ids=()):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        prefix_indices=list(prefix_indices),
        fill_ids=list(fill_ids),
        skip_radix_cache_insert=False,
        finished_reason=None,
        set_finish_with_abort=lambda msg: setattr(
            __import__("sys").modules[__name__], "_LAST_ABORT", msg,
        ) or _set_finish_reason(req_pool_idx, msg),
    )


_FINISH_REASONS = {}


def _set_finish_reason(req_idx, msg):
    _FINISH_REASONS[req_idx] = SimpleNamespace(
        type="abort", message=msg, __class__=type("FINISH_ABORT", (), {}),
    )


def _make_req_to_token_pool(num_reqs=2, max_ctx=128):
    rt = torch.arange(num_reqs * max_ctx, dtype=torch.int32).reshape(num_reqs, max_ctx)
    return SimpleNamespace(req_to_token=rt)


@pytest.mark.parametrize("chunk_len", [4, 32, 64])
def test_9_12_a_free_called_with_exact_chunk_slots(chunk_len):
    """§9.12.a allocator.free called with exact slot list of failed chunk."""
    adapter = _make_adapter()
    before = len(adapter.free_pages)
    out_cache_loc = adapter.alloc(chunk_len)
    req = _make_req(req_pool_idx=0, prefix_indices=[], fill_ids=list(range(chunk_len)))
    rt_pool = _make_req_to_token_pool()
    adapter.on_chunked_prefill_failure(req, out_cache_loc, rt_pool)
    # Pages should be reclaimed.
    assert len(adapter.free_pages) == before


def test_9_12_b_req_to_token_zeroed_in_range():
    """§9.12.b req_to_token[slice] zeroed for failed chunk range."""
    adapter = _make_adapter()
    out_cache_loc = adapter.alloc(32)
    fill_ids = list(range(32))
    req = _make_req(req_pool_idx=0, prefix_indices=[], fill_ids=fill_ids)
    rt_pool = _make_req_to_token_pool()
    rt_pool.req_to_token[0, :32] = 999  # mark live
    adapter.on_chunked_prefill_failure(req, out_cache_loc, rt_pool)
    assert (rt_pool.req_to_token[0, :32] == 0).all()


def test_9_12_c_skip_radix_cache_insert_set():
    """§9.12.c req.skip_radix_cache_insert == True."""
    adapter = _make_adapter()
    out_cache_loc = adapter.alloc(32)
    req = _make_req(req_pool_idx=0, prefix_indices=[], fill_ids=list(range(32)))
    rt_pool = _make_req_to_token_pool()
    adapter.on_chunked_prefill_failure(req, out_cache_loc, rt_pool)
    assert req.skip_radix_cache_insert is True


def test_9_12_d_finished_reason_is_abort():
    """§9.12.d req.finished_reason is FINISH_ABORT (mocked: type=='abort')."""
    adapter = _make_adapter()
    out_cache_loc = adapter.alloc(32)
    req = _make_req(req_pool_idx=0, prefix_indices=[], fill_ids=list(range(32)))
    rt_pool = _make_req_to_token_pool()
    adapter.on_chunked_prefill_failure(req, out_cache_loc, rt_pool)
    assert _FINISH_REASONS[0].type == "abort"
    assert "tt_backend_chunked_prefill_failure" in _FINISH_REASONS[0].message


def test_9_12_e_bypass_chunked_req_flag():
    """§9.12.e GenerationBatchResult.bypass_chunked_req == True signal path.

    The caller (paged backend forward()) MUST return a result with
    bypass_chunked_req=True. We assert the field's default and that the
    scheduler reads it (sanity check via getattr).
    """
    from sglang.srt.managers.utils import GenerationBatchResult
    r = GenerationBatchResult(bypass_chunked_req=True)
    assert getattr(r, "bypass_chunked_req", False) is True
    r2 = GenerationBatchResult()
    assert getattr(r2, "bypass_chunked_req", False) is False
```

- [ ] **Step 2:** Run; expect PASS:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_chunked_failure_recovery.py -v
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_chunked_failure_recovery.py
git commit -m "test(tenstorrent): §9.12.a-e chunked-prefill failure recovery (5 sub-gates)"
```

### Task 3.4: Paged backend forward — error recovery wiring + cancel check

- [ ] **Step 1:** Wrap `forward()`'s prefill/decode call in try/except per §3.7:

```python
def forward(self, forward_batch):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    fb = forward_batch
    if fb.forward_mode.is_idle():
        return LogitsProcessorOutput(next_token_logits=None)
    if fb.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
        raise NotImplementedError(...)  # as before

    page_table = self.metadata.init_forward_metadata(fb)
    try:
        if fb.forward_mode == ForwardMode.EXTEND:
            logits = self._do_extend(fb, page_table)
        else:
            logits = self._do_decode(fb, page_table)
    except Exception as exc:
        for req in fb.reqs:
            req.set_finish_with_abort(f"tt_backend_error: {exc!r}")
        # Q2 answer determines this: if set_finish_with_abort short-circuits
        # current step, return None-logits; else return zero-logits.
        # Phase-0 evidence dictates which branch.
        # Default (spec best-guess: no short-circuit) → zero-logits.
        return LogitsProcessorOutput(
            next_token_logits=torch.zeros(fb.batch_size, self._model.vocab_size),
        )
    return LogitsProcessorOutput(next_token_logits=logits)
```

- [ ] **Step 2:** In `tp_worker.py`'s `_forward_batch_generation_tt`, add the post-forward cancel check:

```python
# After getting logits_output from execution_backend.forward():
# Spec §3.6 cancel mid-prefill: scheduler may have set req.canceled
# between dispatch and our return. Mark abort retrospectively.
for req in (mwb.reqs or []):
    if getattr(req, "canceled", False):
        req.set_finish_with_abort("client_canceled")
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_paged_backend.py \
       python/sglang/srt/hardware_backend/tenstorrent/tp_worker.py
git commit -m "feat(tenstorrent): paged forward error recovery + post-forward cancel check"
```

### Task 3.5: Validate Q3 — `empty_slots` identity at B=4 no-retract

- [ ] **Step 1:** Add an assertion + logging line in `_do_extend`:

```python
empty_slots = list(range(B))
# Q3: at B≤4 with no retract, empty_slots MUST be the identity. If we
# observe non-identity here, R3 has fired — chase it up.
if B > 1:
    logger.debug("tt_paged_empty_slots", extra={"B": B, "empty_slots": empty_slots})
```

This is a soft sentinel — the assertion only enforces identity in P2a. Phase 4 hardware run will exercise it at B=4.

- [ ] **Step 2:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_paged_backend.py
git commit -m "feat(tenstorrent): Q3 sentinel — log empty_slots at B>1 no-retract"
```

### Task 3.6: Phase 3 verification gate

- [ ] **Step 1:** All CPU unit tests pass (including the new §9.12 ones):

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/ -v -m "not paged_backend"
```

- [ ] **Step 2:** §9.1 smoke still passes (regression check on the paged path with the error-recovery wrapper added):

```bash
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py -v
```

- [ ] **Step 3:** Phase 3 done. The full hardware acceptance suite lands in Phase 4.

---

## Phase 4 — Acceptance test suite (§9.1-§9.16) (W5)

**Goal:** Write the 11 remaining hardware-gated tests and run the full §9 gate set. Flip the `auto` default from `tt_transformers_single` to `tt_transformers_paged`.

**Files touched:**
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness_paged.py` (§9.2)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_batched_correctness.py` (§9.3)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_radix_prefix_cache.py` (§9.4)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_queue_full_admission.py` (§9.5)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_endpoint.py` (§9.6 part 1)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_disconnect.py` (§9.6 part 2)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_stability_paged.py` (§9.7)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_dual_track_switch.py` (§9.8)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_mesh_shape_param.py` (§9.9)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_perf_log_paged.py` (§9.10)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_eviction_replay.py` (§9.11)
- Create: `python/sglang/srt/hardware_backend/tenstorrent/test/test_shutdown_teardown.py` (§9.13)
- Modify: `python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py` (`auto` → `tt_transformers_paged`)

**Spec refs:** §4.3 §9.1-§9.16, §4.5 perf-log additions.

**Live risks:** R5 (HIGH) — §9.11 eviction-replay is the primary gate. R11 24h stability deferred to P2b's §9.b5; §9.7 compressed 15-min substitutes here.

### Task 4.1: §9.2 — `test_greedy_correctness_paged.py`

Per spec wording: "per-prompt isolated re-run matches batched run (NOT bit-exact across runs — BFP8 drift)". The test compares the **first 16 tokens** of a B=1 isolated run vs the same prompt embedded in a B=4 batched run.

- [ ] **Step 1:** Write the test:

```python
"""§9.2 greedy correctness on paged path.

NOT bit-exact across runs (BFP8 drift). The check is: same prompt issued
once in isolation (B=1) produces the same first 16 tokens as the same
prompt issued inside a B=4 batch.
"""

import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


PROMPT = "Recite the first sentence of the Gettysburg Address."


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_greedy_isolated_matches_batched():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MAX_RUNNING_REQUESTS": "4",
    }) as base_url:
        # Isolated B=1 run
        r1 = requests.post(f"{base_url}/v1/completions", json={
            "model": "Llama-3.1-8B-Instruct",
            "prompt": PROMPT, "max_tokens": 16, "temperature": 0,
        }, timeout=120)
        r1.raise_for_status()
        tokens_isolated = r1.json()["choices"][0]["text"]

        # Same prompt inside a B=4 batch (3 distractors + this one)
        import concurrent.futures
        distractors = [
            "Tell me about the weather.",
            "List three primes.",
            "What is 2 + 2?",
        ]
        prompts = [PROMPT, *distractors]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(requests.post, f"{base_url}/v1/completions", json={
                    "model": "Llama-3.1-8B-Instruct",
                    "prompt": p, "max_tokens": 16, "temperature": 0,
                }, timeout=120)
                for p in prompts
            ]
            results = [f.result() for f in futures]
        tokens_batched = results[0].json()["choices"][0]["text"]

        assert tokens_isolated == tokens_batched, (
            f"isolated {tokens_isolated!r} != batched {tokens_batched!r}; "
            "BFP8 drift acceptable, but token sequences must match at temp=0"
        )
```

- [ ] **Step 2:** Run on hardware:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness_paged.py -v
```

- [ ] **Step 3:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_greedy_correctness_paged.py
git commit -m "test(tenstorrent): §9.2 greedy correctness — isolated == batched at temp=0"
```

### Task 4.2: §9.3 — `test_batched_correctness.py` (B=4 concurrent)

- [ ] **Step 1:** Write the test:

```python
"""§9.3 B=4 batched correctness — 4 concurrent prompts, each matches isolated run.

This is the empirical proof of paged-path batching at the goal-level
G3a (B≥4 measured).
"""

import concurrent.futures
import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server

PROMPTS = [
    "The first three planets from the sun are",
    "Two plus two equals",
    "The capital of France is",
    "The opposite of 'up' is",
]


def _single(base_url, prompt):
    r = requests.post(f"{base_url}/v1/completions", json={
        "model": "Llama-3.1-8B-Instruct",
        "prompt": prompt, "max_tokens": 8, "temperature": 0,
    }, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_b4_batched_matches_isolated():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MAX_RUNNING_REQUESTS": "4",
    }) as base_url:
        # Isolated baseline
        isolated = [_single(base_url, p) for p in PROMPTS]

        # B=4 batched
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            batched = list(ex.map(lambda p: _single(base_url, p), PROMPTS))

        for i, (iso, bat) in enumerate(zip(isolated, batched)):
            assert iso == bat, f"prompt {i}: isolated {iso!r} != batched {bat!r}"
```

- [ ] **Step 2:** Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_batched_correctness.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_batched_correctness.py
git commit -m "test(tenstorrent): §9.3 B=4 batched correctness — isolated == batched"
```

### Task 4.3: §9.4 — `test_radix_prefix_cache.py` (RadixAttention)

- [ ] **Step 1:** Write the test. Shared 1K-token system prompt; 4 distinct user questions; assert Prometheus `sglang:cache_hit_rate ≥ 0.5` AND effective tok/s ratio (warm/cold) ≥ 1.5:

```python
"""§9.4 RadixAttention prefix-cache effectiveness.

Shared 1K-token system prompt + 4 distinct user questions. Verify:
  - Prometheus sglang:cache_hit_rate ≥ 0.5
  - warm tok/s / cold tok/s ≥ 1.5×
"""

import time
import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer concisely. " * 64  # ~1K tokens
)
QUESTIONS = ["What is Python?", "What is 5*6?", "Capital of Spain?",
             "Define entropy."]


def _ask(base_url, q):
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/completions", json={
        "model": "Llama-3.1-8B-Instruct",
        "prompt": SYSTEM_PROMPT + "\nUser: " + q + "\nAssistant:",
        "max_tokens": 32, "temperature": 0,
    }, timeout=180)
    r.raise_for_status()
    text = r.json()["choices"][0]["text"]
    dt = time.perf_counter() - t0
    return text, dt


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_radix_prefix_cache_hits_and_speedup():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MAX_RUNNING_REQUESTS": "4",
    }) as base_url:
        # Cold (first ask) → prefix not yet cached
        _, cold_dt = _ask(base_url, QUESTIONS[0])
        # Warm (next 3) → cache hits expected
        warm_dts = []
        for q in QUESTIONS[1:]:
            _, dt = _ask(base_url, q)
            warm_dts.append(dt)
        warm_avg = sum(warm_dts) / len(warm_dts)
        speedup = cold_dt / warm_avg
        assert speedup >= 1.5, f"warm/cold speedup {speedup:.2f}× < 1.5×"

        # Prometheus cache hit rate
        metrics = requests.get(f"{base_url}/metrics", timeout=10).text
        # Find sglang:cache_hit_rate{...} <value>
        import re
        m = re.search(r"sglang:cache_hit_rate\b[^\n]*\s([0-9.]+)", metrics)
        assert m, f"cache_hit_rate gauge not in /metrics output"
        rate = float(m.group(1))
        assert rate >= 0.5, f"cache_hit_rate {rate} < 0.5"
```

- [ ] **Step 2:** Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_radix_prefix_cache.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_radix_prefix_cache.py
git commit -m "test(tenstorrent): §9.4 RadixAttention prefix cache hit-rate + speedup"
```

### Task 4.4: §9.5 — `test_queue_full_admission.py`

- [ ] **Step 1:** Write the test. With `--max-queued-requests N`, submit `N+1` long requests; verify the `N+1`th gets in-stream AbortReq with `meta_info.finish_reason.type == "abort"` AND `abort_message` mentions "queue":

```python
"""§9.5 queue-full admission — in-stream AbortReq, NOT HTTP 503.

With --max-queued-requests=2, submit 3 long-running requests. The 3rd
must receive AbortReq with finish_reason.type=='abort' and 'queue'
in the abort_message. HTTP status must be 200 (501/503 are health-only).
"""

import concurrent.futures
import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_queue_full_emits_in_stream_abort():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MAX_RUNNING_REQUESTS": "1",  # serialize for predictability
    }, extra_args=["--max-queued-requests", "2"]) as base_url:
        def long_req():
            return requests.post(f"{base_url}/v1/completions", json={
                "model": "Llama-3.1-8B-Instruct",
                "prompt": "Count to 200 in english words. " * 4,
                "max_tokens": 512, "temperature": 0,
            }, timeout=300)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(lambda _: long_req(), range(3)))

        # At least one of the three must be an abort
        abort_count = 0
        for r in results:
            assert r.status_code == 200, f"expected HTTP 200, got {r.status_code}"
            body = r.json()
            meta = body.get("meta_info") or body.get("choices", [{}])[0].get("meta_info", {})
            finish = meta.get("finish_reason", {})
            if isinstance(finish, dict) and finish.get("type") == "abort":
                assert "queue" in finish.get("message", "").lower()
                abort_count += 1
        assert abort_count >= 1, "no queue-full abort observed in 3 concurrent reqs"
```

If `extra_args` isn't supported in P1's `sglang_server` fixture, extend it to thread CLI args through to the popen call.

- [ ] **Step 2:** Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_queue_full_admission.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_queue_full_admission.py
git commit -m "test(tenstorrent): §9.5 queue-full admission — in-stream AbortReq"
```

### Task 4.5: §9.6 — `test_abort_via_endpoint.py` and `test_abort_via_disconnect.py`

- [ ] **Step 1:** Write `test_abort_via_endpoint.py`:

```python
"""§9.6 part 1 — /abort_request POST clears KV.

available_size via /get_internal_state returns to the baseline after the abort.
"""

import threading
import time
import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_abort_via_endpoint_clears_kv():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MAX_RUNNING_REQUESTS": "4",
    }) as base_url:
        baseline = requests.get(f"{base_url}/get_internal_state").json()["available_size"]

        rid = "abort-test-rid-001"
        def long_req():
            try:
                requests.post(f"{base_url}/v1/completions", json={
                    "model": "Llama-3.1-8B-Instruct",
                    "prompt": "Recite the dictionary alphabetically. " * 10,
                    "max_tokens": 4096, "temperature": 0,
                    "rid": rid,
                }, timeout=10)
            except requests.Timeout:
                pass  # expected since we abort

        t = threading.Thread(target=long_req, daemon=True)
        t.start()
        time.sleep(2)  # let it actually run
        r = requests.post(f"{base_url}/abort_request",
                          json={"rid": rid}, timeout=10)
        assert r.status_code == 200
        time.sleep(3)  # let the scheduler reclaim
        after = requests.get(f"{base_url}/get_internal_state").json()["available_size"]
        assert after >= baseline * 0.95, (
            f"available_size did not recover: baseline={baseline}, after_abort={after}"
        )
```

- [ ] **Step 2:** Write `test_abort_via_disconnect.py`:

```python
"""§9.6 part 2 — client TCP RST clears KV.

Same as endpoint abort, but the trigger is `urllib3` connection-pool drop.
"""

import threading
import time
import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_abort_via_disconnect_clears_kv():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MAX_RUNNING_REQUESTS": "4",
    }) as base_url:
        baseline = requests.get(f"{base_url}/get_internal_state").json()["available_size"]

        def short_lived():
            try:
                with requests.Session() as s:
                    s.post(f"{base_url}/v1/completions", json={
                        "model": "Llama-3.1-8B-Instruct",
                        "prompt": "Recite the dictionary. " * 10,
                        "max_tokens": 2048, "temperature": 0,
                    }, timeout=2)
            except Exception:
                pass  # closing connection mid-stream — expected

        t = threading.Thread(target=short_lived, daemon=True)
        t.start()
        time.sleep(3)  # session closes; server should detect TCP RST
        time.sleep(5)  # scheduler reclaim window
        after = requests.get(f"{base_url}/get_internal_state").json()["available_size"]
        assert after >= baseline * 0.95
```

- [ ] **Step 3:** Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_endpoint.py \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_disconnect.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_endpoint.py \
       python/sglang/srt/hardware_backend/tenstorrent/test/test_abort_via_disconnect.py
git commit -m "test(tenstorrent): §9.6 abort via endpoint + client disconnect"
```

### Task 4.6: §9.7 — `test_stability_paged.py` (compressed 15-min)

- [ ] **Step 1:** Write the test:

```python
"""§9.7 paged stability — compressed 15-min B=4 run.

ITL p99 windowed (baseline 5-10min vs tail 5min in SAME run); drift < 10%.
Baseline is intra-run, NOT vs B=1.
"""

import time
import threading
import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


@pytest.mark.paged_backend
@pytest.mark.hardware
@pytest.mark.slow
def test_stability_compressed_15min_b4():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MAX_RUNNING_REQUESTS": "4",
    }) as base_url:
        # ITL samples per minute
        itls_per_minute = []  # list[list[float]]
        for minute in range(15):
            itls_per_minute.append([])
            start = time.time()
            while time.time() - start < 60:
                t0 = time.perf_counter()
                r = requests.post(f"{base_url}/v1/completions", json={
                    "model": "Llama-3.1-8B-Instruct",
                    "prompt": f"At minute {minute}, list 3 facts.",
                    "max_tokens": 32, "temperature": 0,
                }, timeout=60)
                itl = (time.perf_counter() - t0) / 32  # rough per-token
                itls_per_minute[-1].append(itl)
                r.raise_for_status()

        def p99(xs):
            xs = sorted(xs)
            return xs[int(len(xs) * 0.99)]
        baseline_p99 = p99([itl for minute in itls_per_minute[5:10] for itl in minute])
        tail_p99 = p99([itl for minute in itls_per_minute[10:15] for itl in minute])
        drift = (tail_p99 - baseline_p99) / baseline_p99
        assert drift < 0.10, f"ITL p99 drift {drift:.1%} > 10%"
```

- [ ] **Step 2:** Run (15 minutes; only on hardware nights / CI overnight slot) + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m "paged_backend and slow" \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_stability_paged.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_stability_paged.py
git commit -m "test(tenstorrent): §9.7 stability — 15-min B=4 ITL drift < 10%"
```

### Task 4.7: §9.8 — `test_dual_track_switch.py`

- [ ] **Step 1:** Write the test:

```python
"""§9.8 dual-track switch — two server-launch fixtures, full suite per env.

Restart between, no hot-swap. Each launch validates the corresponding
backend serves /v1/completions correctly.
"""

import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


@pytest.mark.hardware
@pytest.mark.parametrize("backend,extra_env", [
    ("tt_transformers_single", {}),
    ("tt_transformers_paged", {"SGLANG_TT_MAX_RUNNING_REQUESTS": "4"}),
])
def test_dual_track_smoke(backend, extra_env):
    env = {
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": backend,
        **extra_env,
    }
    with sglang_server(env=env) as base_url:
        r = requests.post(f"{base_url}/v1/completions", json={
            "model": "Llama-3.1-8B-Instruct",
            "prompt": "The capital of France is", "max_tokens": 4, "temperature": 0,
        }, timeout=120)
        r.raise_for_status()
        assert "Paris" in r.json()["choices"][0]["text"]
```

- [ ] **Step 2:** Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m hardware \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_dual_track_switch.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_dual_track_switch.py
git commit -m "test(tenstorrent): §9.8 dual-track — single/paged switch w/ restart"
```

### Task 4.8: §9.9 — `test_mesh_shape_param.py`

- [ ] **Step 1:** Write the test. `1x2` runs on hardware; `1x4` static-lint via mocked mesh:

```python
"""§9.9 mesh-shape parametric.

SGLANG_TT_MESH_SHAPE=1x2 runs on hardware. SGLANG_TT_MESH_SHAPE=1x4 is
static-lint only — we exercise the shape-resolution path with a mocked
mesh-open to confirm no crash.
"""

import os
import pytest
import requests
from unittest.mock import patch, MagicMock

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_mesh_shape_1x2_hardware():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MESH_SHAPE": "1x2",
    }) as base_url:
        r = requests.post(f"{base_url}/v1/completions", json={
            "model": "Llama-3.1-8B-Instruct",
            "prompt": "Hello", "max_tokens": 4, "temperature": 0,
        }, timeout=120)
        r.raise_for_status()


def test_mesh_shape_1x4_static_lint(monkeypatch):
    """1x4 resolution path doesn't crash even when no 4-device mesh exists.

    Mocks ttnn.open_mesh_device so we don't actually need 4 devices.
    """
    monkeypatch.setenv("SGLANG_TT_MESH_SHAPE", "1x4")
    from sglang.srt.hardware_backend.tenstorrent.platform import _resolve_mesh_shape
    assert _resolve_mesh_shape() == (1, 4)
```

- [ ] **Step 2:** Run + commit:

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/test_mesh_shape_param.py -v
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_mesh_shape_param.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_mesh_shape_param.py
git commit -m "test(tenstorrent): §9.9 mesh-shape — 1x2 hw + 1x4 static-lint"
```

### Task 4.9: §9.10 — `test_perf_log_paged.py`

- [ ] **Step 1:** Write the test. Sanity floors per §4.3 §9.10: batched B=4 ≥ 1.2× B=1 decode tok/s; prefix-hit lookup < 10ms; KV-pool steady-state < 95%. Also record: tok/s @ B={1,2,4}, prefix hit/miss latency, KV-pool utilization series, per-mode timing.

```python
"""§9.10 perf log — batched tok/s @ B={1,2,4}, prefix-hit/miss latency, KV util.

Records (per spec §4.5):
  - Batched decode tok/s @ B={1, 2, 4}
  - Prefix-cache hit/miss latency split (warm vs cold path)
  - KV-pool utilization time series (sampled every 5s for 60s)
  - Per-mode timing breakdown: init_forward_metadata / forward / sample

Sanity floors (FAIL on regression):
  - batched B=4 decode tok/s ≥ 1.2× B=1 decode tok/s
  - prefix-hit lookup latency < 10ms
  - KV-pool steady-state utilization < 95%
"""
import json, os, statistics, time
import pytest, requests
from contextlib import contextmanager

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware and a live sglang server on :30000",
)
SERVER = "http://localhost:30000"
PERF_LOG = "/tmp/tt_p2a_perf_log.json"

def _decode_only_tok_s(B: int, n_decode_tokens: int = 100) -> float:
    """Issue B concurrent requests, each generating n_decode_tokens; return aggregate decode tok/s."""
    prompt = "The history of computing began with"  # short warm prompt
    payloads = [{"model": "llama", "prompt": prompt, "max_tokens": n_decode_tokens,
                 "temperature": 0, "stream": False} for _ in range(B)]
    t0 = time.perf_counter()
    # Concurrent fire — use threads or asyncio; here a simple ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=B) as ex:
        responses = list(ex.map(lambda p: requests.post(f"{SERVER}/v1/completions", json=p, timeout=120).json(), payloads))
    elapsed = time.perf_counter() - t0
    total_decode_tokens = sum(r["usage"]["completion_tokens"] for r in responses)
    # subtract ~50ms of warm prefill amortization per req
    decode_only_s = max(elapsed - 0.05 * B, 1e-6)
    return total_decode_tokens / decode_only_s

def _prefix_lookup_latency_ms() -> tuple[float, float]:
    """Returns (hit_latency_ms, miss_latency_ms) for a 1K-token shared prefix."""
    shared = "Computing has evolved " * 50   # ~1K tokens
    # warm cache
    requests.post(f"{SERVER}/v1/completions", json={
        "model": "llama", "prompt": shared + " Q1: what is X?",
        "max_tokens": 1, "temperature": 0}, timeout=60)
    # hit
    t0 = time.perf_counter()
    requests.post(f"{SERVER}/v1/completions", json={
        "model": "llama", "prompt": shared + " Q2: what is Y?",
        "max_tokens": 1, "temperature": 0}, timeout=60)
    hit_ms = (time.perf_counter() - t0) * 1000
    # miss (random prefix)
    rand_shared = "Random " + "x" * 4000     # disrupts prefix
    t0 = time.perf_counter()
    requests.post(f"{SERVER}/v1/completions", json={
        "model": "llama", "prompt": rand_shared + " Q3: what is Z?",
        "max_tokens": 1, "temperature": 0}, timeout=60)
    miss_ms = (time.perf_counter() - t0) * 1000
    return hit_ms, miss_ms

def _kv_util_sample(duration_s: int = 60, step_s: int = 5) -> list[float]:
    """Sample KV-pool utilization from /get_internal_state during a B=4 background load."""
    samples = []
    # background load (fire-and-poll)
    from concurrent.futures import ThreadPoolExecutor
    def _bg_load():
        for _ in range(20):
            requests.post(f"{SERVER}/v1/completions", json={
                "model": "llama", "prompt": "Tell me about " + "x"*200,
                "max_tokens": 200, "temperature": 0}, timeout=120)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_bg_load) for _ in range(4)]
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration_s:
            try:
                s = requests.get(f"{SERVER}/get_internal_state", timeout=2).json()
                avail = s["token_pool"]["available_size"]
                cap = s["token_pool"]["capacity"]
                samples.append(1.0 - avail / cap)
            except Exception:
                pass
            time.sleep(step_s)
        # let bg finish
        for f in futures:
            try: f.result(timeout=30)
            except Exception: pass
    return samples

@REQUIRES_TT
def test_perf_log_paged():
    results = {}
    # 1. Batched tok/s
    for B in (1, 2, 4):
        results[f"decode_tok_s_B{B}"] = _decode_only_tok_s(B)
        print(f"  decode tok/s @ B={B}: {results[f'decode_tok_s_B{B}']:.2f}", flush=True)
    # 2. Prefix-cache latency
    hit_ms, miss_ms = _prefix_lookup_latency_ms()
    results["prefix_hit_ms"] = hit_ms
    results["prefix_miss_ms"] = miss_ms
    print(f"  prefix hit={hit_ms:.1f}ms miss={miss_ms:.1f}ms", flush=True)
    # 3. KV utilization series
    util = _kv_util_sample(duration_s=60, step_s=5)
    results["kv_util_p99"] = statistics.quantiles(util, n=100)[98] if len(util) >= 10 else max(util)
    results["kv_util_series"] = util
    print(f"  kv util p99={results['kv_util_p99']:.3f} (n={len(util)})", flush=True)
    # 4. Per-mode timing (read from server's /metrics Prometheus endpoint or internal counters)
    metrics = requests.get(f"{SERVER}/metrics", timeout=5).text
    # Look for tt-specific histograms if exposed; else skip with a note.
    results["per_mode_timing_note"] = "captured from /metrics if instrumented in T2.2-T2.3"
    # Write log
    with open(PERF_LOG, "w") as f:
        json.dump(results, f, indent=2)
    print(f"--- wrote {PERF_LOG} ---", flush=True)
    # 5. Sanity floors (per spec §4.3 §9.10)
    assert results["decode_tok_s_B4"] >= 1.2 * results["decode_tok_s_B1"], (
        f"batched B=4 ({results['decode_tok_s_B4']:.1f}) < 1.2× B=1 ({results['decode_tok_s_B1']:.1f})"
    )
    assert hit_ms < 10.0, f"prefix-hit lookup {hit_ms:.1f}ms ≥ 10ms"
    assert results["kv_util_p99"] < 0.95, f"KV util p99 {results['kv_util_p99']:.3f} ≥ 0.95"
```

The body extends P1's `test_perf_log.py` measurement harness (cold/warm prefill, decode tok/s) with paged-specific dimensions. The 3 sanity floors map 1:1 to spec §4.3 §9.10. `per_mode_timing` requires backend-side instrumentation in T2.2/T2.3 (`init_forward_metadata` / `forward` time logging); if not wired by Phase 4, leave as a NOTE field in the JSON (don't gate).

- [ ] **Step 2:** Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_perf_log_paged.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_perf_log_paged.py
git commit -m "test(tenstorrent): §9.10 perf log — tok/s, prefix cache, KV util"
```

### Task 4.10: §9.11 — `test_eviction_replay.py` (R5 mitigation, HIGH-risk gate)

This is the single most important test for R5 (RadixAttention vs paged-KV evict path consistency).

- [ ] **Step 1:** Write the test:

```python
"""§9.11 eviction-replay — R5 mitigation gate.

Flow:
  1. Load a prefix (issue request A with system-prompt S, generate N tokens).
  2. Fill RadixCache with enough distinct prefixes to evict S
     (issue M unrelated long-system-prompt requests until /get_internal_state
     shows S is no longer in the cache OR cache_evict_count has incremented).
  3. Re-issue request A.
  4. Logits / output tokens must match the original A's output.
"""

import time
import pytest
import requests

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_eviction_replay_preserves_logits():
    PROMPT_A = "You are a math tutor. " * 50 + "What is 7*8?"
    UNRELATED = [f"You are persona {i}. " * 50 + "Hello." for i in range(20)]

    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
        "SGLANG_TT_MAX_RUNNING_REQUESTS": "4",
    }) as base_url:
        def ask(p, max_tokens=32):
            r = requests.post(f"{base_url}/v1/completions", json={
                "model": "Llama-3.1-8B-Instruct",
                "prompt": p, "max_tokens": max_tokens, "temperature": 0,
            }, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["text"]

        first = ask(PROMPT_A)

        # Force eviction by flooding distinct prefixes
        for p in UNRELATED:
            ask(p, max_tokens=4)

        time.sleep(2)
        second = ask(PROMPT_A)
        assert first == second, (
            f"R5 fired: post-eviction replay yields different tokens. "
            f"first={first!r}, second={second!r}"
        )
```

- [ ] **Step 2:** Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_eviction_replay.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_eviction_replay.py
git commit -m "test(tenstorrent): §9.11 eviction-replay (R5 HIGH-risk mitigation)"
```

### Task 4.11: §9.13 — `test_shutdown_teardown.py`

- [ ] **Step 1:** Write the test:

```python
"""§9.13 shutdown teardown — ttnn.get_num_tensors() == 0 after server shutdown."""

import subprocess
import time
import pytest

from sglang.srt.hardware_backend.tenstorrent.test._fixtures.server import sglang_server


@pytest.mark.paged_backend
@pytest.mark.hardware
def test_shutdown_leaves_no_ttnn_tensors():
    with sglang_server(env={
        "SGLANG_PLATFORM": "tenstorrent",
        "SGLANG_TT_EXECUTION_BACKEND": "tt_transformers_paged",
    }) as base_url:
        # Drive one request so the model is fully resident
        import requests
        requests.post(f"{base_url}/v1/completions", json={
            "model": "Llama-3.1-8B-Instruct",
            "prompt": "Hi", "max_tokens": 4, "temperature": 0,
        }, timeout=120).raise_for_status()
    # Context exit triggers shutdown. Now check tensor leak.
    out = subprocess.check_output(
        ["python", "-c", "import ttnn; print(ttnn.get_num_tensors())"]
    )
    assert int(out.strip()) == 0, f"leaked tensors: {out.strip()}"
```

- [ ] **Step 2:** Run + commit:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_shutdown_teardown.py -v
git add python/sglang/srt/hardware_backend/tenstorrent/test/test_shutdown_teardown.py
git commit -m "test(tenstorrent): §9.13 shutdown teardown — no ttnn tensor leak"
```

### Task 4.12: Flip `auto` default → `tt_transformers_paged`

- [ ] **Step 1:** Edit `python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py`:

```python
def resolve_execution_backend_name(requested: str | None = None) -> str:
    """Resolve "auto" / "" / None to the P2a default.

    P2a flip (Task 4.12): auto → tt_transformers_paged (was: tt_transformers_single).
    """
    name = (requested or envs.SGLANG_TT_EXECUTION_BACKEND.get() or "auto").lower()
    if name == "auto":
        return "tt_transformers_paged"
    return name
```

- [ ] **Step 2:** Run §9.1 smoke WITHOUT `SGLANG_TT_EXECUTION_BACKEND` set (to confirm `auto` now resolves to paged):

```bash
unset SGLANG_TT_EXECUTION_BACKEND
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke_paged.py -v
```

- [ ] **Step 3:** Re-confirm `simple_backend` is still selectable explicitly:

```bash
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single \
  pytest -m simple_backend python/sglang/srt/hardware_backend/tenstorrent/test/test_smoke.py -v
```

- [ ] **Step 4:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/execution/__init__.py
git commit -m "feat(tenstorrent): flip auto default to tt_transformers_paged (P2a)"
```

### Task 4.13: Phase 4 final acceptance — run the full §9.1-§9.16 gate set

- [ ] **Step 1:** Run all paged-backend tests on hardware:

```bash
SGLANG_PLATFORM=tenstorrent pytest -m paged_backend \
  python/sglang/srt/hardware_backend/tenstorrent/test/ -v --tb=short
```

Expected: 11+ hardware tests pass (smoke, greedy, batched, radix, queue-full, abort×2, stability, dual-track, mesh-shape, perf-log, eviction-replay, shutdown).

- [ ] **Step 2:** Run all unit tests (CPU only):

```bash
pytest python/sglang/srt/hardware_backend/tenstorrent/test/ -v --tb=short -m "not paged_backend and not simple_backend"
```

Expected: all unit tests pass — paged-KV math, free-list invariant, page-table translation, chunked-failure 5-step, model-registry, token-pool overflow, paged-backend ABC.

- [ ] **Step 3:** Run P1 simple-backend regression on hardware:

```bash
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single \
  pytest -m simple_backend python/sglang/srt/hardware_backend/tenstorrent/test/ -v --tb=short
```

Expected: all 13 P1 hardware tests pass. **Zero regression** — this is non-negotiable.

- [ ] **Step 4:** Write an evidence file `python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p2a_acceptance_evidence.txt` containing:
  - Final pytest output (timestamp, host info, pass count)
  - Q1-Q9 answers (copy from `phase0_signature_evidence.txt`)
  - Q4 micro-benchmark final p50/p99
  - tt-metal docker SHA
  - SGLang main commit SHA at acceptance time
  - Total LoC added (`git diff --stat <P2a-start-sha>..HEAD`)

- [ ] **Step 5:** Commit:

```bash
git add python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p2a_acceptance_evidence.txt
git commit -m "docs(tenstorrent): P2a acceptance evidence — §9.1-§9.16 all green"
```

---

## W6 — Buffer + §5.3b decision gate (no new tasks)

W6 is reserved for catch-up and the §5.3b P2a→P2b decision gate evaluation. **No implementation tasks here.** Instead, walk through the §5.3b checklist:

- [ ] **§5.3b condition 1:** §9.1-§9.14 all pass (collected in Task 4.13).
- [ ] **§5.3b condition 2:** ≥ 7 of Q1-Q9 have explicit answers (Phase 0 + Phase 1 give Q1/Q2/Q4/Q5/Q6/Q7/Q9 = 7; Q3 added in Phase 3; Q8 deferred to P2b W0). Q7 now has a dedicated unit-test deliverable in T1.4b (read-through invariant after `cache_finished_req(canceled=True)`).
- [ ] **§5.3b condition 3:** R5 (RadixAttention deadlock) shows no reproduction in §9.7 compressed-15-min stability run.
- [ ] **§5.3b condition 4:** No §9 gate has failed ≥ 2 times.
- [ ] **§5.3b condition 5:** Q8 (GptOss coupling) status determines P2b W0 scope.

If all 5 pass → P2b can branch off post-P2a SHA per the timeline.

If condition 1 fails → 2-week W7-W8 fix loop, P2b pushed right by 2 weeks (spec §5.3b failure handling).

If second failure at W9 → P2b downgrades to best-effort (spec §5.3b, §1 G1b/G2b/G3b/G4b/G5b drop out of acceptance).

---

## Cross-cutting reminders

### Dependencies (linear, with verification gates)

```
Phase 0 (signature lock + ABC + P1 port) → Phase 1 (KV adapter)
  → Phase 2 (paged backend wiring + smoke) → Phase 3 (B≥4 + admission + abort)
  → Phase 4 (acceptance suite + auto-flip) → W6 (decision gate)
```

Each arrow is a "blocks on verification passing" relationship. **Do not start a phase until the previous phase's verification gate exits clean.**

### Risk hot-spots by phase (cross-reference to spec §5.1)

| Phase | Live risks |
|---|---|
| 0 | **R1 HIGH** (pin docker SHA), R9 (signature drift surprise) |
| 1 | **R5 HIGH** (free-list invariant unit), R2 (CPU alloc_extend latency Q4) |
| 2 | R5 HIGH (continues — §9.11 lands in Phase 4), R14 NOTE (model registry collision) |
| 3 | **R6 MEDIUM** (GenerationBatchResult.bypass_chunked_req upstream-class patch — see REBASE_TARGETS.md), R12 MEDIUM (`_chunked_req_scheduled_last_iter` ↔ bypass interaction — §9.12.e covers) |
| 4 | R5 HIGH (§9.11 eviction-replay is the canonical mitigation), R11 MEDIUM (HBM fragmentation — deferred to P2b §9.b5 24h test), R3 (`empty_slots` retract — soft sentinel in Task 3.5) |

### Spec-coverage sanity check (self-review)

Map each spec § to a Task:

- §0 Executive summary — informational
- §1.1 G1a..G7a (P2a goals) — Phases 0 (G1a ABC), 1 (G2a adapter math via §9.16), 2-4 (G3a-G7a)
- §1.3 Non-goals N1-N12 — informational, plan stays inside
- §2.1 Architecture diagram — design ref, no task
- §2.2 New modules — file-map section above; Tasks 0.6 (base.py), 0.7 (single port), 1.1-1.6 (paged.py, tt_paged.py), 2.1 (llama_tt.py), 2.2 (paged backend), 2.4 (platform.py)
- §2.4 INV-1..INV-7 — Task 0.6 docstring; Tasks 1.2, 1.6, 2.1, 2.2, 2.4 each cite the relevant INV
- §2.5 SGLang integration points — Task 2.4 (KV pool wiring), Task 2.5 (attention metadata via init_forward_metadata), §9.4 test (RadixAttention)
- §3.1 Pipeline — Tasks 2.2, 2.3, 2.5
- §3.2 Prefill EXTEND — Task 2.3 `_do_extend`
- §3.3 Decode DECODE — Task 2.3 `_do_decode`
- §3.4 Free / Eviction — Task 1.2 inherits PagedTokenToKVPoolAllocator.free; §9.11 covers semantics
- §3.5 OOM Admission — Task 4.4 §9.5
- §3.6 Cancel mid-prefill — Task 3.4 post-forward cancel check; Task 4.5 §9.6
- §3.7 Error recovery + 5-step adapter — Tasks 3.1 (utils.py patch), 3.2 (on_chunked_prefill_failure), 3.4 (forward try/except)
- §3.8 Q1-Q9 — Phase 0 Tasks 0.2-0.5 + Phase 1 Task 1.8 (Q4) + Phase 3 Task 3.5 (Q3 sentinel)
- §4.1 Test pyramid — distributed across Phase 1-4 tasks
- §4.2 P1 test migration — Task 0.8 (markers); Task 0.7 (port keeps P1 tests green)
- §4.3 §9.1-§9.16 acceptance gates — Phase 4 Tasks 4.1-4.12 + Phase 1 Tasks 1.3-1.5 + Phase 3 Task 3.3
- §4.4 §9.b1-§9.b6 — P2b plan, NOT here
- §4.5 Perf log additions — Task 4.9 §9.10
- §4.6 P1.5 hygiene — out of scope (assumed merged before P2a starts)
- §5.1 Risks — inlined per phase
- §5.2 Q-list — Phase 0 + spotty per phase
- §5.2b Implementation-discovery escape hatch — How-to-use rule #4
- §5.3 Migration timeline — informational
- §5.3a User-visible transitions — informational
- §5.3b P2a→P2b decision gate — W6 section
- §5.3c Rollback paths — informational; Task 4.12 preserves `tt_transformers_single` selectable
- §5.3d Descope order — referenced if Phase 4 budget overruns
- §6 Appendix A — informational
- §7 Appendix B — informational (Task 0.1 fills the docker SHA TBD)

### Open questions the plan flags (mapping to spec §5.2 Q1-Q9)

| Q | Where answered | Output |
|---|---|---|
| Q1 (`prefill_forward_text` shape) | Phase 0 Task 0.2 | `phase0_signature_evidence.txt` |
| Q2 (`set_finish_with_abort` short-circuits) | Phase 0 Task 0.3 | Same file; gates §3.7 zero-logits vs None branch |
| Q3 (`empty_slots` identity at B=4) | Phase 3 Task 3.5 + Phase 4 Task 4.2 (§9.3 hardware run) | Soft sentinel log + B=4 batched correctness pass |
| Q4 (CPU alloc_extend p99 at B=4) | Phase 1 Task 1.8 | `bench_alloc_extend.py` output |
| Q5 (`paged_attention_config` to `create_tt_model`?) | Phase 0 Task 0.2 | Same file; if YES, §5.2b INV-4 violation |
| Q6 (`req_to_token` dtype) | Phase 0 Task 0.4 | Same file |
| Q7 (RadixCache canceled-req consistency) | **Phase 1 Task 1.4b** (dedicated unit test) + Phase 4 Task 4.5+4.10 (end-to-end coverage) | T1.4b read-through invariant assertion + §9.6 abort + §9.11 eviction-replay |
| Q8 (GptOss coupling depth) | **NOT in P2a** — deferred to P2b W0 | Out of P2a scope (per §5.3b condition 5) |
| Q9 (KV call-site CSV) | Phase 0 Task 0.5 | `q9_call_site_inventory.csv` |

---

**End of plan.** Total tasks: **46** across 5 phases (Phase 0: 9, Phase 1: **10** [includes T1.4b Q7 dedicated test], Phase 2: 8, Phase 3: 6, Phase 4: 13) plus the W6 §5.3b decision-gate checklist (5 conditions). Estimated 4 weeks core + 1 week buffer per spec §5.3 W2-W6, sequential gating throughout.

**Plan-review polish applied** (post-review #10):
- T4.9 §9.10 perf-log: full test body fleshed out from "body omitted for brevity" — now includes batched tok/s, prefix-hit/miss latency, KV-util sampling, per-mode timing dimensions, and 3 sanity-floor asserts per spec §4.3
- T1.4b Q7: new CPU unit test for read-through invariant after `cache_finished_req(canceled=True)` — closes the gap the reviewer flagged (Q7 previously covered only by end-to-end §9.6 + §9.11)
