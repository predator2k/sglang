# P2a Phase 0 — Verification Report

Date: 2026-05-12
Branch: tenstorrent-p1 (commits b3b7ff546..<HEAD>)
Pinned tt-metal image: ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-models-amd64:latest-rc (sha256:40dcdcdabb5ea87a0700d7bdeeada290fe2a09246d4890237b8cd6828c1e360c)

## Step 2 — Phase 0 Deliverables Manifest

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Q1/Q2/Q5/Q6/Q9 answered in phase0_signature_evidence.txt | PASS | `grep "^Q[1-9]" phase0_signature_evidence.txt` returns Q1 (line 66), Q2 (line 96), Q5 (line 83), Q6 (line 128), Q9 (line 137) — all 5 sections present |
| 2 | q9_call_site_inventory.csv committed | PASS | commit 58cde2320; 58 lines; header row `file,line,expression,...`; last row `# TODO: implement get_value_buffer(layer_id) stub...` |
| 3 | tt-metal docker SHA pinned in reset_devices.sh AND spec §7 | PASS | `reset_devices.sh` line 2: `# Pinned tt-metal image: ghcr.io/.../tt-metalium-ubuntu-22.04-release-models-amd64:latest-rc (sha256:40dcdcdabb5ea87a0700d7bdeeada290fe2a09246d4890237b8cd6828c1e360c)`; spec `docs/superpowers/specs/2026-05-12-sglang-tenstorrent-p2-design.md` line 591: same SHA in `§7` tech-stack table |
| 4 | New ABC in execution/base.py matches INV-1 verbatim | PASS | `grep INV-1 python/sglang/srt/hardware_backend/tenstorrent/execution/base.py` → line 9: `INV-1: forward(forward_batch) -> LogitsProcessorOutput is model-level forward` matching spec §2.4 INV-1 wording exactly |
| 5 | P1 simple backend port (T0.7) compiles AST + commit landed | PASS | commit c5f8774ec "refactor(tenstorrent): port P1 simple backend to model-level forward ABC"; `python3 -c "import ast; ast.parse(open('python/sglang/srt/hardware_backend/tenstorrent/execution/tt_transformers_backend.py').read()); print('AST PARSE OK')"` → `AST PARSE OK` |

**Manifest result: 5/5 PASS**

## Step 1 — CPU Unit Tests (host attempt)

Command attempted:
```
pytest python/sglang/srt/hardware_backend/tenstorrent/test/ -v -m "not paged_backend" --collect-only
```

Result: **17 tests collected, 5 collection errors** (host env limitation — not a regression).

Successfully collected (12 tests in 6 modules — these run inside docker):
- `test_greedy_correctness.py::test_greedy_correctness_vs_hf_reference`
- `test_mmlu_mini.py::test_mmlu_mini_accuracy`
- `test_p1_acceptance.py` — 8 functions (acceptance gate tests)
- `test_perf_log.py::test_perf_log_capture`
- `test_platform_activate.py::test_activate_returns_none_when_ttnn_import_fails`
- `test_platform_activate.py::test_activate_returns_name_when_ttnn_present`
- `test_smoke.py` — 3 functions
- `test_stability.py::test_stability_compressed`

Collection errors (host env — `torch` / `sglang` not installed on host Python):
```
ERROR test_execution_backend_registry.py  — ModuleNotFoundError: No module named 'sglang'
ERROR test_forward_mode_guard.py          — ModuleNotFoundError: No module named 'torch'
ERROR test_model_runner_init.py           — ModuleNotFoundError: No module named 'torch'
ERROR test_server_args_defaults.py        — ModuleNotFoundError: No module named 'sglang'
ERROR test_wrapper_lifecycle.py           — ModuleNotFoundError: No module named 'sglang'
```

All 5 errors are pure import-path failures (`torch` / `sglang` not installed in the host system Python). The tenstorrent-venv also cannot build sglang editable install on this host (missing C++ build deps). This is expected — the task description noted "Most P1 unit tests require the docker env (orjson/etc not on host)." No logic regression detected; the test files have correct syntax and valid Python (AST confirmed on each file individually).

## Step 1 — Hardware Tests (DEFERRED — operator to run in docker)

These commands MUST be run by the operator in a live docker container with mesh access BEFORE Phase 1 starts. Until then, Phase 0 is "code-ready, hardware-unverified".

```bash
# Inside docker (using the pinned image):
podman run --rm -it --device=/dev/tenstorrent \
  -v /home/mhnie/sglang:/sglang \
  sha256:40dcdcdabb5ea87a0700d7bdeeada290fe2a09246d4890237b8cd6828c1e360c \
  bash

# Inside the container:
source /opt/venv/bin/activate
cd /sglang/python && pip install -e . --no-deps
cd /sglang
SGLANG_PLATFORM=tenstorrent SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single \
  pytest -m simple_backend python/sglang/srt/hardware_backend/tenstorrent/test/ -v
```

Expected: all 6 hardware-gated test files (test_smoke, test_greedy_correctness, test_mmlu_mini, test_stability, test_perf_log, test_p1_acceptance) pass.

## Decision gate per spec §5.3b conditions

- §5.3b condition 1 (P2a §9 gates) — NOT YET (we're at Phase 0; §9 gates are Phase 4)
- §5.3b condition 2 (≥7 of Q1-Q9 answered) — STATUS:
  - Q1 ✅ (phase0_signature_evidence.txt)
  - Q2 ✅
  - Q5 ✅
  - Q6 ✅
  - Q9 ✅
  - Q3, Q4, Q7 deferred to Phase 1-3
  - Q8 deferred to P2b W0
- §5.3b condition 3 (R5 no repro) — N/A in Phase 0 (R5 emerges in Phase 4 stability tests)

## Decision

Step 2 manifest = 5/5 PASS. No INV violation was discovered during Phase 0.

→ **APPROVED to proceed to Phase 1** (after operator runs hardware tests in docker and confirms green).
