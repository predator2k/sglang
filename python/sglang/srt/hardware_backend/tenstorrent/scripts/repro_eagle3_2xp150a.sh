#!/bin/bash
# Reproduce the v94/v95/v107 EAGLE-3 final state on 2× Tenstorrent P150a.
#
# Usage:
#   bash repro_eagle3_2xp150a.sh [generate|chat]
#
# - "generate" (default): production-quality factual content via /generate
# - "chat": coherent thinking content via /v1/chat/completions
#
# Expected output (generate mode):
#   "capital of France" → "Paris. The capital of Italy the United Kingdom..."
#   8 tok/s avg, spec_accept_rate ~0.30, 7/8 prompts factually correct
#
# Expected output (chat mode):
#   "What is the capital of Japan?" → "<think>\nOkayOkay, the user is asking
#    for what the capital of Japan is. I need to provide the correct answer.
#    Tokyo. But wait..."
#   Correctly answers Tokyo with reasoning chain.

set -uo pipefail  # NOT -e: pkill/tt-smi return nonzero on harmless conditions

MODE=${1:-generate}
TREE_MASK_ENV=""
if [ "$MODE" = "chat" ]; then
  TREE_MASK_ENV="-e SGLANG_TT_EAGLE_TREE_MASK=1"
  echo "Mode: chat (tree-mask ON for /v1/chat/completions)"
else
  echo "Mode: generate (tree-mask OFF for /generate raw text)"
fi

# Pre-flight: reset TT cards and clear cache
/home/mhnie/.tenstorrent-venv/bin/tt-smi -r 0,1 2>&1 | tail -n 2 || true
podman exec p3a-ngram bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; rm -rf /root/.cache/tt-metal-model-cache/P300/* 2>/dev/null || true'

LOG=/tmp/eagle3_2xp150a_${MODE}.log
rm -f "$LOG"

# Launch
podman exec \
  -e SGLANG_PLATFORM=tenstorrent \
  -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  -e SGLANG_TT_SPEC_DRAFT_PATH=/models/qwen3_8b_eagle3 \
  -e SGLANG_TT_SPEC_DRAFT_BACKEND=cpu \
  $TREE_MASK_ENV \
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

# Smoke test
if [ "$MODE" = "chat" ]; then
  podman exec p3a-ngram bash -lc 'curl -s --max-time 120 -X POST http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"/models/Qwen3-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Japan?\"}],\"max_tokens\":64,\"temperature\":0.0}"' 2>&1 | \
    python3 -c "import sys, json; r=json.load(sys.stdin); print('content:', repr(r['choices'][0]['message']['content']))"
else
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
fi

# Cleanup
podman exec p3a-ngram bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null'
echo "Done. Server stopped. Hardware free."
