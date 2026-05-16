#!/bin/bash
# Sweep Qwen3-8B precision modes to find the optimal quality/speed tradeoff.
#
# Tests each mode with both /generate (completions) and /v1/chat/completions (thinking).
# Records tok/s and output quality for each mode.
#
# Usage:
#   bash sweep_precision.sh [mode]          # test single mode
#   bash sweep_precision.sh ALL             # test all modes
#   bash sweep_precision.sh lofi lofi_bf16kv  # test specific modes
#
# Modes: lofi, balanced, lofi_bf16kv, lofi_hifi2_sdpa, lofi_hifi2_attn,
#        bfp8_lofi, lofi_bf16kv_hifi2_sdpa, lofi_bf16kv_hifi2_attn, hifi

set -uo pipefail

CONTAINER=p3a-ngram
TT_SMI=/home/mhnie/.tenstorrent-venv/bin/tt-smi
RESULTS_DIR=/tmp/precision_sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"

# All available modes
ALL_MODES="lofi lofi_bf16kv lofi_hifi2_sdpa lofi_hifi2_attn bfp8_lofi lofi_bf16kv_hifi2_sdpa lofi_bf16kv_hifi2_attn balanced hifi"

if [ "${1:-}" = "ALL" ]; then
    MODES="$ALL_MODES"
elif [ $# -gt 0 ]; then
    MODES="$*"
else
    echo "Usage: $0 [mode|ALL]"
    echo "Available modes: $ALL_MODES"
    exit 1
fi

echo "=== Precision Sweep ==="
echo "Modes: $MODES"
echo "Results: $RESULTS_DIR"
echo ""

copy_model_config() {
    podman cp /home/mhnie/tt-metal-sglang/models/tt_transformers/tt/model_config.py \
        "$CONTAINER":/tt-metal/models/tt_transformers/tt/model_config.py
}

kill_server() {
    podman exec "$CONTAINER" bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 2; pkill -9 -f "sglang::" 2>/dev/null' || true
    sleep 3
}

reset_devices() {
    "$TT_SMI" -r 0,1 2>&1 | tail -n 2 || true
    sleep 5
}

clear_cache() {
    podman exec "$CONTAINER" bash -lc 'rm -rf /root/.cache/tt-metal-model-cache/P300/* 2>/dev/null' || true
}

launch_server() {
    local mode=$1
    local logfile="/tmp/sglang_${mode}.log"

    echo "  Launching server with SGLANG_TT_QWEN3_PRECISION=$mode ..."
    podman exec -d \
        -e SGLANG_PLATFORM=tenstorrent \
        -e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged \
        -e "SGLANG_TT_QWEN3_PRECISION=$mode" \
        "$CONTAINER" bash -lc "source /opt/venv/bin/activate; python -m sglang.launch_server \
            --model-path /models/Qwen3-8B --trust-remote-code \
            --host 0.0.0.0 --port 30000 --device tenstorrent \
            --max-running-requests 1 \
            --context-length 2048 \
            --mem-fraction-static 0.5 \
            --attention-backend torch_native \
            --disable-cuda-graph 2>&1 | tee $logfile"

    # Wait for server ready
    local waited=0
    local max_wait=300
    while [ $waited -lt $max_wait ]; do
        if podman exec "$CONTAINER" bash -lc "curl -s --max-time 3 http://localhost:30000/health" 2>/dev/null | grep -q .; then
            echo "  Server ready after ${waited}s"
            sleep 5  # extra warmup
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 30)) -eq 0 ]; then
            echo "  Still waiting... (${waited}s)"
        fi
    done
    echo "  ERROR: Server failed to start within ${max_wait}s"
    return 1
}

test_generate() {
    local mode=$1
    echo "  Testing /generate (completions) ..."

    local prompts=("The capital of Japan is" "World War 2 ended in" "1 + 1 =" "The speed of light is approximately")
    local results=""

    for prompt in "${prompts[@]}"; do
        local resp
        resp=$(podman exec "$CONTAINER" bash -lc "curl -s --max-time 60 -X POST http://localhost:30000/generate \
            -H 'Content-Type: application/json' \
            -d '{\"text\":\"$prompt\",\"sampling_params\":{\"max_new_tokens\":32,\"temperature\":0.0},\"log_metrics\":true}'" 2>&1)

        local parsed
        parsed=$(echo "$resp" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    m = r['meta_info']
    tps = m['completion_tokens']/m['e2e_latency']
    print(f'OK|{tps:.2f}|{r[\"text\"][:80]}')
except Exception as e:
    print(f'ERROR|0|{e}')
" 2>&1)
        results="$results\n    $prompt => $parsed"
    done

    echo -e "$results"
    echo -e "$results" >> "$RESULTS_DIR/${mode}_generate.txt"
}

test_generate_long() {
    local mode=$1
    echo "  Testing /generate long-form (128 tokens) ..."

    local resp
    resp=$(podman exec "$CONTAINER" bash -lc "curl -s --max-time 120 -X POST http://localhost:30000/generate \
        -H 'Content-Type: application/json' \
        -d '{\"text\":\"Explain the theory of general relativity in simple terms.\",\"sampling_params\":{\"max_new_tokens\":128,\"temperature\":0.0},\"log_metrics\":true}'" 2>&1)

    echo "$resp" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    m = r['meta_info']
    tps = m['completion_tokens']/m['e2e_latency']
    tpot = m['e2e_latency']/m['completion_tokens']*1000
    text = r['text'][:200]
    # Check for garbage patterns
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    quality = 'OK' if unique_ratio > 0.3 else 'GARBAGE'
    print(f'  generate-long: {tps:.1f} tok/s  TPOT={tpot:.1f}ms  quality={quality}  unique_ratio={unique_ratio:.2f}')
    print(f'  text: {text}')
except Exception as e:
    print(f'  generate-long: ERROR {e}')
" 2>&1 | tee -a "$RESULTS_DIR/${mode}_generate_long.txt"
}

test_chat() {
    local mode=$1
    echo "  Testing /v1/chat/completions (thinking mode) ..."

    local resp
    resp=$(podman exec "$CONTAINER" bash -lc 'curl -s --max-time 120 -X POST http://localhost:30000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"/models/Qwen3-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Japan? Answer in one sentence.\"}],\"max_tokens\":128,\"temperature\":0.0}"' 2>&1)

    echo "$resp" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    c = r['choices'][0]
    content = c['message']['content']
    usage = r.get('usage', {})
    comp_tokens = usage.get('completion_tokens', 0)

    # Check for garbage patterns
    words = content.split() if content else []
    unique_ratio = len(set(words)) / max(len(words), 1)
    has_think = '<think>' in (c['message'].get('reasoning_content', '') or '')

    quality = 'OK' if unique_ratio > 0.3 else 'GARBAGE'
    has_tokyo = 'tokyo' in content.lower() or 'Tokyo' in content
    print(f'  chat: quality={quality} unique_ratio={unique_ratio:.2f} has_tokyo={has_tokyo} has_think={has_think} tokens={comp_tokens}')
    print(f'  content: {repr(content[:200])}')
    if c['message'].get('reasoning_content'):
        rc = c['message']['reasoning_content']
        print(f'  thinking: {repr(rc[:200])}')
except Exception as e:
    print(f'  chat: ERROR {e}')
    print(f'  raw: {sys.stdin.read()[:200] if hasattr(sys.stdin, \"read\") else \"N/A\"}')
" 2>&1 | tee -a "$RESULTS_DIR/${mode}_chat.txt"
}

test_chat_nothink() {
    local mode=$1
    echo "  Testing /v1/chat/completions with enable_thinking=false ..."

    local resp
    resp=$(podman exec "$CONTAINER" bash -lc 'curl -s --max-time 120 -X POST http://localhost:30000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"/models/Qwen3-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Japan? Answer in one sentence.\"}],\"max_tokens\":64,\"temperature\":0.0,\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"' 2>&1)

    echo "$resp" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    c = r['choices'][0]
    content = c['message']['content']
    words = content.split() if content else []
    unique_ratio = len(set(words)) / max(len(words), 1)
    quality = 'OK' if unique_ratio > 0.3 else 'GARBAGE'
    has_tokyo = 'tokyo' in content.lower() or 'Tokyo' in content
    print(f'  chat-nothink: quality={quality} unique_ratio={unique_ratio:.2f} has_tokyo={has_tokyo}')
    print(f'  content: {repr(content[:200])}')
except Exception as e:
    print(f'  chat-nothink: ERROR {e}')
" 2>&1 | tee -a "$RESULTS_DIR/${mode}_chat_nothink.txt"
}

test_speed() {
    local mode=$1
    echo "  Speed benchmark (5 sequential requests, 128 tokens each) ..."

    local total_tps=0
    local count=0

    for i in 1 2 3 4 5; do
        local resp
        resp=$(podman exec "$CONTAINER" bash -lc "curl -s --max-time 120 -X POST http://localhost:30000/generate \
            -H 'Content-Type: application/json' \
            -d '{\"text\":\"Write a detailed paragraph about the history of computing. Include information about early pioneers.\",\"sampling_params\":{\"max_new_tokens\":128,\"temperature\":0.0},\"log_metrics\":true}'" 2>&1)

        local tps
        tps=$(echo "$resp" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    m = r['meta_info']
    print(f'{m[\"completion_tokens\"]/m[\"e2e_latency\"]:.2f}')
except:
    print('0')
" 2>&1)
        total_tps=$(python3 -c "print($total_tps + $tps)")
        count=$((count + 1))
        echo "    run $i: $tps tok/s"
    done

    local avg_tps
    avg_tps=$(python3 -c "print(f'{$total_tps / $count:.1f}')")
    echo "  === $mode avg: $avg_tps tok/s ==="
    echo "$avg_tps" > "$RESULTS_DIR/${mode}_speed.txt"
}

# Main loop
for mode in $MODES; do
    echo ""
    echo "=============================================="
    echo "  MODE: $mode"
    echo "=============================================="

    # Step 1: Kill existing server
    kill_server

    # Step 2: Reset devices
    reset_devices

    # Step 3: Clear cache
    clear_cache

    # Step 4: Copy model_config.py
    copy_model_config

    # Step 5: Launch server
    if ! launch_server "$mode"; then
        echo "  SKIP: $mode (server failed to start)"
        echo "FAILED" > "$RESULTS_DIR/${mode}_speed.txt"
        continue
    fi

    # Step 6: Run tests
    test_generate "$mode"
    test_generate_long "$mode"
    test_chat "$mode"
    test_chat_nothink "$mode"
    test_speed "$mode"

    echo ""
    echo "  $mode complete."
done

# Summary
echo ""
echo "=============================================="
echo "  SUMMARY"
echo "=============================================="
for mode in $MODES; do
    speed=$(cat "$RESULTS_DIR/${mode}_speed.txt" 2>/dev/null || echo "N/A")
    echo "  $mode: $speed tok/s"
done
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "TT vLLM baseline: 32.6 tok/s"
