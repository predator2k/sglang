# P2b W0 — Model availability checklist (2026-05-13)

Verified at `ls /home/mhnie/tt-models/`:

| Model | Status | Local path | P2b action |
|---|---|---|---|
| Llama-3.1-8B-Instruct | ✅ available | `/home/mhnie/tt-models/Llama-3.1-8B-Instruct` | P2a baseline (already validated) |
| Qwen3-8B | ✅ available | `/home/mhnie/tt-models/Qwen3-8B` | T4.2 smoke target |
| Qwen3-1.7B | ✅ available | `/home/mhnie/tt-models/Qwen3-1.7B` | T4.2 smoke alternate (smaller / faster) |
| Qwen3-14B | ✅ available | `/home/mhnie/tt-models/Qwen3-14B` | T4.3 max_seq_len test target |
| Mistral-7B-v0.3 | ❌ NOT available | — | **PLACEHOLDER per spec N11** — TenstorrentMistralForCausalLM stays registered but instantiation NOT validated in P2b. Defer to P3. |
| GptOss-20B | ❌ NOT available | — | **PLACEHOLDER per spec N11** — same. |

## Implications for §9.b acceptance

- §9.b1 per-model smoke: scope reduced to Llama + Qwen only (2 of 4)
- §9.b2 max_seq_len matrix: only Llama-3.1-8B (32K) + Qwen3-{1.7,8,14}B (best-effort)
- §9.b3 (128K chunked-prefill on Llama) — pushed to P3 per spec §A2
- §9.b4 (P300 chunk-size table) — pushed to P3 per spec §A2

## Recommendation

Proceed to T4.2 with **Qwen3-8B** as the multi-model smoke target. Mistral / GptOss
remain registered (their `TenstorrentMistralForCausalLM` / `TenstorrentGptOssForCausalLM`
classes are importable from `models/tt_llm.py`) but uninstantiable for lack of weights.
This satisfies spec G1b/G2b partially — class registration verified in T1.3
test_plugin_registration.py; instantiation deferred.
