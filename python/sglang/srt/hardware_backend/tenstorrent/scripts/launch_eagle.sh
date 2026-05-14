#!/bin/bash
# Launch the EAGLE 2-model cohost smoke for SGLang-on-Tenstorrent.
#
# Pre-conditions captured from the P3a.2 boot-iteration log (see
# `test/_fixtures/p3a_eagle_t2_2_evidence.txt`):
#   - target model: /models/Qwen3-8B (loaded first)
#   - draft model:  /models/Qwen3-1.7B (loaded second via eagle_draft.py)
#   - --speculative-eagle-topk 1   (not auto-populated by SGLang; required)
#   - --speculative-num-steps 5
#   - --speculative-num-draft-tokens 6
#   - --max-running-requests 1     (host-OOM avoidance: 2-model phantom KV at higher bs blew anon-rss past 52 GB)
#   - --context-length 4096        (matched to OOM ceiling)
#   - --mem-fraction-static 0.5
#   - --disable-cuda-graph         (we have no CUDA graphs on TT)
#
# What the recent paged-path fixes enable for this launch:
#   - ETH+ROW+MUX dispatch (commit 809fc7f19) — unblocks iter-8 bmm
#     kernel-placement TT_FATAL
#   - page_size=64 + attention_backend=torch_native defaults
#     (commit a1ba960f3) — paged path's prior implicit dependencies
#   - CPU paged allocator (commit 4a5b3ecc4) — no Triton driver needed
#
# Usage:
#   bash python/sglang/srt/hardware_backend/tenstorrent/scripts/launch_eagle.sh
#
# Tee target: test/_fixtures/p3a_eagle_t2_2_post_dispatch.log

set -euo pipefail

LOG=/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/p3a_eagle_t2_2_post_dispatch.log

# Make sure no stale launch is hogging the mesh.
podman exec p3a-ngram bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null || true; sleep 2'

# Reset devices in case the previous run left mesh state stuck.
/home/mhnie/.tenstorrent-venv/bin/tt-smi -r 0,1 || true

echo "Launching EAGLE 2-model cohost. Log: $LOG"

podman exec \
  -e SGLANG_PLATFORM=tenstorrent \
  -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
  -e SGLANG_TT_SPEC_DRAFT_PATH=/models/Qwen3-1.7B \
  p3a-ngram bash -lc '
source /opt/venv/bin/activate
python -m sglang.launch_server \
  --model-path /models/Qwen3-8B --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --device tenstorrent \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path /models/Qwen3-1.7B \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 5 \
  --speculative-num-draft-tokens 6 \
  --max-running-requests 1 \
  --context-length 4096 \
  --mem-fraction-static 0.5 \
  --disable-cuda-graph
' > "$LOG" 2>&1 &

EAGLE_PID=$!
echo "EAGLE launcher PID $EAGLE_PID"
echo "Monitor:"
echo "  tail -f $LOG"
echo "  curl http://localhost:30000/health  # once it says 'Uvicorn running'"
