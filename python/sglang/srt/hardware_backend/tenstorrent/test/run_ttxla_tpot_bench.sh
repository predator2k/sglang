#!/bin/bash
# v142 tt-xla TPOT benchmark orchestrator.
# Runs bench_ttxla_tpot.py for each model in a fresh container with device reset.
# Aggregates results into v142_ttxla_tpot_benchmark.json.

set -uo pipefail

IMAGE="ghcr.io/tenstorrent/tt-xla-slim:latest"
CNAME="tt-xla-bench"
TT_SMI="/home/mhnie/.tenstorrent-venv/bin/tt-smi"
SCRIPT="/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/bench_ttxla_tpot.py"
FIXTURE="/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v142_ttxla_tpot_benchmark.json"

declare -A MODEL_PATHS
MODEL_PATHS["TinyLlama-1.1B"]="/models/TinyLlama-1.1B-Chat-v1.0"
MODEL_PATHS["SmolLM2-135M"]="/models/SmolLM2-135M-Instruct"
MODEL_PATHS["Qwen2.5-0.5B"]="/models/Qwen2.5-0.5B"
MODEL_PATHS["Qwen2.5-3B"]="/models/Qwen2.5-3B"
MODEL_PATHS["Qwen3-0.6B"]="/models/Qwen3-0.6B"
MODEL_PATHS["Qwen3-1.7B"]="/models/Qwen3-1.7B"
MODEL_PATHS["Qwen3-4B"]="/models/Qwen3-4B"
MODEL_PATHS["Qwen3-8B"]="/models/Qwen3-8B"
MODEL_PATHS["Qwen3-14B"]="/models/Qwen3-14B"
MODEL_PATHS["Llama-3.1-8B"]="/models/Llama-3.1-8B-Instruct"
MODEL_PATHS["Mistral-7B"]="/models/Mistral-7B-Instruct-v0.3"
MODEL_PATHS["Nemotron-Mini-4B"]="/models/Nemotron-Mini-4B-Instruct"
MODEL_PATHS["StableLM-2-1.6B"]="/models/stablelm-2-1_6b"
MODEL_PATHS["Phi-4-mini"]="/models/Phi-4-mini-instruct"

# Order: small first for faster initial feedback
MODELS=(
    "SmolLM2-135M"
    "Qwen2.5-0.5B"
    "TinyLlama-1.1B"
    "Qwen3-0.6B"
    "Qwen3-1.7B"
    "StableLM-2-1.6B"
    "Qwen2.5-3B"
    "Qwen3-4B"
    "Nemotron-Mini-4B"
    "Mistral-7B"
    "Llama-3.1-8B"
    "Qwen3-8B"
    "Qwen3-14B"
    "Phi-4-mini"
)

echo "=== v142 tt-xla TPOT Benchmark ==="
echo "Models: ${#MODELS[@]}"
echo "Output: $FIXTURE"
echo ""

# Initialize fixture if not exists
if [ ! -f "$FIXTURE" ]; then
    python3 -c "
import json, time
d = {
    'version': 'v142',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'protocol': 'Single inference per container, 30 decode tokens, proven v137 pattern',
    'max_cache_len': 64, 'pad_len': 32, 'num_decode': 30,
    'hw': '2x P150a Blackhole FW 19.6.0',
    'container': 'ghcr.io/tenstorrent/tt-xla-slim:latest',
    'models': {}
}
with open('$FIXTURE', 'w') as f:
    json.dump(d, f, indent=2)
print('Initialized fixture')
"
fi

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$MODEL]}"
    echo ""
    echo "=========================================="
    echo "  $MODEL  ($MODEL_PATH)"
    echo "=========================================="

    # Check if already done
    if python3 -c "
import json, sys
with open('$FIXTURE') as f: d = json.load(f)
m = d.get('models',{}).get('$MODEL',{})
if m.get('status') in ('PASS','PASS_SLOW','OOM'):
    print('SKIP: already done (' + m['status'] + ')')
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
        continue
    fi

    # Reset devices (best effort, continue even if it fails)
    echo "  Resetting devices..."
    if timeout 30 "$TT_SMI" -r 0,1 2>&1 | tail -2; then
        echo "  Device reset OK"
    else
        echo "  WARNING: Device reset failed, continuing anyway..."
    fi
    sleep 3

    # Clean container
    podman stop "$CNAME" 2>/dev/null || true
    podman rm "$CNAME" 2>/dev/null || true
    sleep 1

    # Start container
    echo "  Starting container..."
    podman run -d --name "$CNAME" \
        --device /dev/tenstorrent --privileged \
        --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
        -v /home/mhnie/sglang:/sglang -v /home/mhnie/tt-models:/models \
        --network host \
        "$IMAGE" sleep infinity
    sleep 2

    # Run benchmark (output to file inside container to avoid pipe buffering)
    echo "  Running benchmark..."
    LOGFILE="/tmp/bench_${MODEL}.log"
    CONTAINER_LOG="/tmp/bench_output.log"
    podman exec "$CNAME" bash -c \
        "PYTHONUNBUFFERED=1 python3 -u $SCRIPT $MODEL_PATH $MODEL > $CONTAINER_LOG 2>&1; echo EXIT_CODE=\$? >> $CONTAINER_LOG" \
        || true

    # Copy log from container
    podman cp "$CNAME:$CONTAINER_LOG" "$LOGFILE" 2>/dev/null || true
    if [ -f "$LOGFILE" ]; then
        echo "  Benchmark output ($(wc -l < "$LOGFILE") lines):"
        grep "\[" "$LOGFILE" | head -20
        echo "  ---"
    fi

    # Extract JSON result via temp file (avoids shell quoting issues)
    RESULT_LINE=$(grep "^RESULT_JSON:" "$LOGFILE" 2>/dev/null | tail -1 || true)
    if [ -n "$RESULT_LINE" ]; then
        echo "${RESULT_LINE#RESULT_JSON:}" > /tmp/bench_result.json
        python3 -c "
import json
with open('/tmp/bench_result.json') as f: result = json.load(f)
with open('$FIXTURE') as f: d = json.load(f)
d['models']['$MODEL'] = result
with open('$FIXTURE', 'w') as f: json.dump(d, f, indent=2)
status = result.get('status', '?')
tpot = result.get('steady_state_tpot_ms', 'N/A')
print(f'  Saved: {status}  steady_tpot={tpot}ms')
"
    else
        echo "  WARNING: No RESULT_JSON found in output"
        python3 -c "
import json
with open('$FIXTURE') as f: d = json.load(f)
d['models']['$MODEL'] = {'status': 'NO_RESULT', 'error': 'No RESULT_JSON in output'}
with open('$FIXTURE', 'w') as f: json.dump(d, f, indent=2)
"
    fi

    # Cleanup container
    podman stop "$CNAME" 2>/dev/null || true
    podman rm "$CNAME" 2>/dev/null || true

    echo "  >>> $MODEL COMPLETE <<<"
done

echo ""
echo "=========================================="
echo "  FINAL SUMMARY"
echo "=========================================="
python3 -c "
import json
with open('$FIXTURE') as f: d = json.load(f)
for name, m in d.get('models', {}).items():
    status = m.get('status', '?')
    tpot = m.get('steady_state_tpot_ms', 'N/A')
    first = m.get('first_decode_s', 'N/A')
    prefill = m.get('prefill_s', 'N/A')
    out = m.get('output_text', '')[:40]
    print(f'  {name:25s}  {status:10s}  tpot={tpot}ms  1st_decode={first}s  prefill={prefill}s  {out!r}')
"

echo ""
echo "Results: $FIXTURE"
