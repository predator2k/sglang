#!/bin/bash
# HISTORICAL — v62 iteration milestone (torch tree-build/verify fallback first
# unblocked EAGLE boot). For the current production launch, use:
#   bash scripts/repro_eagle3_2xp150a.sh [generate|chat|both]
#
# This script uses pre-tuned spec config (num_steps=5, num_draft_tokens=6)
# that v94 later found suboptimal — production uses num_steps=1, num_draft_tokens=2.
# Kept for git history; do not run for production validation.
#
# Launch EAGLE-3 (target Qwen3-8B + draft Tengyunw/qwen3_8b_eagle3) on TT,
# now with the torch tree-build/verify fallback installed via TTTpModelWorker.
#
# Builds on v61's torch-native draft-attention stack:
#   - target on 2× P150a (paged path)
#   - draft on CPU torch (LlamaForCausalLMEagle3, 1 layer, HF embed)
#   - TorchNativeAttnBackend per draft step
#   - tt_eagle_kernels.install_tt_eagle_kernels() injects sgl_kernel fallbacks
#
# v62 should clear the NameError at eagle_utils.py:138 and produce real EAGLE
# verified output.

set -euo pipefail

LOG=/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_eagle_t2_2_post_dispatch_v62_eagle3_treebuild_fallback.log

podman exec p3a-ngram bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null || true; sleep 3'
/home/mhnie/.tenstorrent-venv/bin/tt-smi -r 0,1 || true

echo "Launching EAGLE-3 v62. Log: $LOG"

podman exec \
  -e SGLANG_PLATFORM=tenstorrent \
  -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  -e SGLANG_TT_SPEC_DRAFT_PATH=/models/qwen3_8b_eagle3 \
  -e SGLANG_TT_SPEC_DRAFT_BACKEND=cpu \
  p3a-ngram bash -lc '
source /opt/venv/bin/activate
python -m sglang.launch_server \
  --model-path /models/Qwen3-8B --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --device tenstorrent \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /models/qwen3_8b_eagle3 \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 5 \
  --speculative-num-draft-tokens 6 \
  --max-running-requests 1 \
  --context-length 2048 \
  --mem-fraction-static 0.5 \
  --attention-backend torch_native \
  --disable-cuda-graph
' > "$LOG" 2>&1 &

EAGLE_PID=$!
echo "EAGLE v62 launcher PID $EAGLE_PID"
echo "Tail: tail -f $LOG"
