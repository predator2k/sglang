#!/bin/bash
# Reproduce the v111-v113 EAGLE-3 unified-mode state on 2× Tenstorrent P150a.
#
# Usage:
#   bash repro_eagle3_2xp150a.sh [generate|chat|both]
#
# - "generate" (default): /generate factual-completion smoke (3 prompts)
# - "chat":              /v1/chat/completions reasoning smoke (Tokyo prompt)
# - "both":              single launch, alternates /generate and /chat
#                        (demonstrates v111 unified mode — one server, both endpoints)
#
# Unified mode (v111+): the server auto-detects chat-template requests at
# prefill time by scanning input_ids for Qwen3 chat tokens {151644,151645,151648}
# and engages tree-mask per-request. No env override is required. The
# SGLANG_TT_EAGLE_TREE_MASK={auto,0,1} env still works as a global override
# for debugging.
#
# Expected output (generate mode):
#   "capital of Japan is" → "Tokyo, and the capital..."
#   ~8 tok/s avg, spec_accept_rate ~0.30, 7/8 factual prompts correct
#
# Expected output (chat mode):
#   "What is the capital of Japan?" → "<think>\nOkay, ...Tokyo..."
#   Full reasoning chain reaching the correct answer.

set -uo pipefail  # NOT -e: pkill/tt-smi return nonzero on harmless conditions

MODE=${1:-generate}
case "$MODE" in
  generate|chat|both) ;;
  *) echo "Usage: $0 [generate|chat|both]" >&2; exit 2 ;;
esac
echo "Mode: $MODE (unified auto-detect; no tree-mask env override needed)"

# Pre-flight: reset TT cards and clear cache
/home/mhnie/.tenstorrent-venv/bin/tt-smi -r 0,1 2>&1 | tail -n 2 || true
podman exec p3a-ngram bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; rm -rf /root/.cache/tt-metal-model-cache/P300/* 2>/dev/null || true'

LOG=/tmp/eagle3_2xp150a_${MODE}.log
rm -f "$LOG"

# Launch (no tree-mask env override — prefill auto-detect handles chat vs generate)
podman exec \
  -e SGLANG_PLATFORM=tenstorrent \
  -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  -e SGLANG_TT_SPEC_DRAFT_PATH=/models/qwen3_8b_eagle3 \
  -e SGLANG_TT_SPEC_DRAFT_BACKEND=cpu \
  p3a-ngram bash -lc 'source /opt/venv/bin/activate; python -m sglang.launch_server \
    --model-path /models/Qwen3-8B --trust-remote-code \
    --host 0.0.0.0 --port 30000 --device tenstorrent \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /models/qwen3_8b_eagle3 \
    --speculative-eagle-topk 1 \
    --speculative-num-steps 1 \
    --speculative-num-draft-tokens 2 \
    --max-running-requests 1 \
    --context-length 2048 \
    --mem-fraction-static 0.5 \
    --attention-backend torch_native \
    --disable-cuda-graph \
    --enable-metrics' > "$LOG" 2>&1 &

# Wait for /health
until podman exec p3a-ngram bash -lc 'curl -s --max-time 2 http://localhost:30000/health 2>/dev/null | head -c 50' 2>&1 | grep -qE "."; do
  sleep 8
done
echo "Server up. Log: $LOG"
sleep 5

probe_generate() {
  for prompt in \
    "The capital of Japan is" \
    "World War 2 ended in" \
    "1 + 1 = "; do
    podman exec p3a-ngram bash -lc "curl -s --max-time 60 -X POST http://localhost:30000/generate \
      -H 'Content-Type: application/json' \
      -d '{\"text\":\"$prompt\",\"sampling_params\":{\"max_new_tokens\":16,\"temperature\":0.0},\"log_metrics\":true}'" 2>&1 | \
      python3 -c "
import sys, json
r = json.load(sys.stdin)
m = r['meta_info']
print(f'  {repr(\"$prompt\"):42s} → {repr(r[\"text\"][:60]):65s} tok/s={m[\"completion_tokens\"]/m[\"e2e_latency\"]:.2f} accept={m[\"spec_accept_rate\"]:.3f}')"
  done
}

probe_chat() {
  podman exec p3a-ngram bash -lc 'curl -s --max-time 120 -X POST http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"/models/Qwen3-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Japan?\"}],\"max_tokens\":64,\"temperature\":0.0}"' 2>&1 | \
    python3 -c "import sys, json; r=json.load(sys.stdin); print('content:', repr(r['choices'][0]['message']['content']))"
}

case "$MODE" in
  generate) probe_generate ;;
  chat)     probe_chat ;;
  both)
    echo "--- /generate ---"
    probe_generate
    echo "--- /v1/chat/completions ---"
    probe_chat
    ;;
esac

# Cleanup
podman exec p3a-ngram bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null'
echo "Done. Server stopped. Hardware free."
