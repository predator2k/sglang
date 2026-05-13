# P2b §9.b2 — Per-model max_seq_len matrix (2026-05-13)

Configuration ceiling (HuggingFace `max_position_embeddings`) and P2a/P2b verified
context_length for each available model on 2× Tenstorrent Blackhole p150a.

| Model | HF ceiling | Verified context_length | Status |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 131072 | 16384 (P2a §9.1–§9.14) | PASS at 16K |
| Qwen3-8B | 40960 | 16384 (T4.2 §9.b1) | PASS at 16K |
| Qwen3-1.7B | 40960 | not validated | DEFERRED — config inspection only |
| Qwen3-14B | 40960 | not validated | DEFERRED — large model, separate test fixture |
| Mistral-7B-v0.3 | n/a (weights unavailable) | n/a | PLACEHOLDER per T4.1 |
| GptOss-20B | n/a (weights unavailable) | n/a | PLACEHOLDER per T4.1 |

## HuggingFace config sources

`max_position_embeddings` read from `/home/mhnie/tt-models/<model>/config.json`
on 2026-05-13 (no server launch required):

- Llama-3.1-8B-Instruct: 131072 (`config.json` top-level field)
- Qwen3-8B: 40960
- Qwen3-1.7B: 40960
- Qwen3-14B: 40960

Spec N12 caps Qwen3-32B at 4K (upstream hang); not locally available so not listed.

## P2b validation summary

Two model families validated end-to-end at 16K context on Blackhole hardware
(`p2a-smoke` container, `local-tt-metal:dev`, sha256:`973e972bddf5`):

- **Llama-3.1-8B-Instruct @ 16384** — P2a §9.1–§9.14 full acceptance gate suite
- **Qwen3-8B @ 16384** — T4.2 §9.b1 smoke: server healthy, `2+2=4` arithmetic correct
  (response `<think>4</think>` confirmed on `--context-length 16384`)

The Qwen3 family required tt-metal patch 02 update (commit from T4.3) to handle
`rope_parameters` as a nested dict in `model_config.py`.

## How to actually verify higher seq lens

Each row marked "DEFERRED" requires:

1. `pkill sglang` (stop current server)
2. Reset devices (`scripts/reset_devices.sh`)
3. Relaunch with `--context-length <new_value> --model-path /models/<model>`
4. Wait for ready (~3–5 min; longer for 14B due to JIT + larger KV pool)
5. POST a long-context request to verify operation
6. Tear down

Exhaustive matrix verification is deferred to P3. Current P2b validates that the
plugin path supports up to 16K context across two model families (Llama-3,
Qwen3) on 2× Blackhole p150a.

## Target context lengths for P3 binary-search

Rough targets based on HF ceilings, hardware DRAM budget, and spec §A2.5:

| Model | P3 target | Notes |
|---|---|---|
| Llama-3.1-8B-Instruct | 32K | Spec G3b; 2× current validated 16K |
| Qwen3-8B | 32K | HF ceiling 40K; try 32K first |
| Qwen3-1.7B | 32K | Smaller model — likely fits; HF ceiling 40K |
| Qwen3-14B | 8K–16K | Larger; DRAM budget tighter; binary-search from 8K |
| Qwen3-32B | 4K (capped) | Spec N12 upstream hang; hard cap at 4K |
