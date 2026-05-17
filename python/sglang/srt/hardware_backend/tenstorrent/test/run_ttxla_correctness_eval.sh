#!/bin/bash
# Host-side orchestrator for v143 correctness evaluation.
# Runs each model in a fresh container after device reset.
# Resume support: skips models already marked PASS in the output JSON.

set -euo pipefail

CONTAINER_IMAGE="ghcr.io/tenstorrent/tt-xla-slim:latest"
CONTAINER_NAME="tt-xla-eval"
TT_SMI="/home/mhnie/.tenstorrent-venv/bin/tt-smi"
SCRIPT="/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/eval_ttxla_correctness.py"
FIXTURE="/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v143_ttxla_correctness_eval.json"

MODELS=(
    "TinyLlama-1.1B"
    "SmolLM2-135M"
    "Qwen2.5-0.5B"
    "Qwen2.5-3B"
    "Qwen3-0.6B"
    "Qwen3-1.7B"
    "Qwen3-4B"
    "Qwen3-8B"
    "Qwen3-14B"
    "Llama-3.1-8B"
    "Mistral-7B"
    "Nemotron-Mini-4B"
    "StableLM-2-1.6B"
)
# NOTE: Phi-4-mini excluded — recompiles every decode step (14.5s/tok)
# due to rope_scaling long_factor/short_factor. Would take ~3h for 25 prompts.

echo "=== v143 tt-xla Correctness Evaluation Orchestrator ==="
echo "Models: ${#MODELS[@]}"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Processing: $MODEL"
    echo "=========================================="

    # Check if already done
    if [ -f "$FIXTURE" ]; then
        if python3 -c "
import json, sys
with open('$FIXTURE') as f:
    data = json.load(f)
m = data.get('models', {}).get('$MODEL', {})
if m and m.get('status') == 'PASS':
    print('DONE')
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            echo "  Already complete (PASS), skipping."
            continue
        fi
    fi

    # Reset devices
    echo "  Resetting TT devices..."
    "$TT_SMI" -r 0,1 2>&1 | tail -3
    sleep 3

    # Stop and remove any existing eval container
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true
    sleep 1

    # Start fresh container
    echo "  Starting container..."
    podman run -d --name "$CONTAINER_NAME" \
        --device /dev/tenstorrent --privileged \
        --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
        -v /home/mhnie/sglang:/sglang -v /home/mhnie/tt-models:/models \
        --network host \
        "$CONTAINER_IMAGE" sleep infinity

    sleep 2

    # Run eval for this single model
    echo "  Running correctness eval for $MODEL..."
    timeout 300 podman exec -e "EVAL_MODELS=$MODEL" "$CONTAINER_NAME" \
        python3 "$SCRIPT" 2>&1 | while IFS= read -r line; do
            echo "  [$MODEL] $line"
        done || echo "  WARNING: $MODEL may have timed out or failed"

    # Cleanup container
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true

    echo "  $MODEL complete."
done

echo ""
echo "=== All models complete ==="
echo "Results: $FIXTURE"
