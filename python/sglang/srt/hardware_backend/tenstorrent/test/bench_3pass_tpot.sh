#!/usr/bin/env bash
# 3-pass TPOT/TTFT benchmark for tt-xla via bench_serving.py
#
# PJRT limitation: torch.compile captures tensor identity, so the server
# must be restarted between passes. Each pass:
#   1. Start fresh server
#   2. Send 1 warmup request (bench_serving default)
#   3. Send 1 benchmark request
#   4. Kill server
#
# Pass 1: JIT warmup   — seed=42, first shape compilation (~50s for 8B)
# Pass 2: JIT cache hit — seed=123, same model weights, fresh compile
# Pass 3: KV cache hit  — seed=123, same input, should match pass 2
#
# Usage: bash bench_3pass_tpot.sh [model_path] [input_len] [output_len]
set -euo pipefail

MODEL="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
INPUT_LEN="${2:-2048}"
OUTPUT_LEN="${3:-2048}"
PORT=30000
CTX_LEN=$(( INPUT_LEN + OUTPUT_LEN + 512 ))
OUTDIR="/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${OUTDIR}/bench_3pass_ttxla_${TIMESTAMP}"
mkdir -p "${LOGDIR}"

echo "=== 3-pass TPOT/TTFT benchmark ==="
echo "Model: ${MODEL}"
echo "Input: ${INPUT_LEN}, Output: ${OUTPUT_LEN}, Context: ${CTX_LEN}"
echo "Logs: ${LOGDIR}"

start_server() {
    echo "[server] Starting..."
    export SGLANG_TT_EXECUTION_BACKEND=tt_xla
    cd /sglang
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --port ${PORT} --host 0.0.0.0 --device cpu \
        --context-length ${CTX_LEN} --mem-fraction-static 0.5 \
        --disable-cuda-graph --tp-size 1 --skip-server-warmup \
        > /tmp/sglang_server.log 2>&1 &
    SERVER_PID=$!
    echo "[server] PID=${SERVER_PID}, waiting for ready..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "[server] Ready after ${i}s"
            return 0
        fi
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "[server] DIED during startup"
            cat /tmp/sglang_server.log | tail -20
            return 1
        fi
        sleep 1
    done
    echo "[server] Timeout after 120s"
    return 1
}

stop_server() {
    echo "[server] Stopping..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
    sleep 2
}

run_pass() {
    local pass_num=$1
    local pass_name=$2
    local seed=$3

    echo ""
    echo "=== PASS ${pass_num}: ${pass_name} (seed=${seed}) ==="
    start_server || return 1

    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url http://localhost:${PORT} \
        --model "${MODEL}" \
        --dataset-name random \
        --random-input-len "${INPUT_LEN}" \
        --random-output-len "${OUTPUT_LEN}" \
        --random-range-ratio 0.0 \
        --num-prompts 1 \
        --seed ${seed} \
        --request-rate inf \
        --disable-tqdm \
        --output-file "${LOGDIR}/pass${pass_num}.jsonl" \
        2>&1 | tee "${LOGDIR}/pass${pass_num}.log"

    stop_server
}

run_pass 1 "JIT warmup" 42
run_pass 2 "JIT cache hit" 123
run_pass 3 "KV cache hit" 123

echo ""
echo "=== Results summary ==="
for f in "${LOGDIR}"/pass*.log; do
    echo ""
    echo "--- $(basename $f) ---"
    grep -E "Mean TTFT|Mean TPOT|Output token throughput|Mean E2E|Total generated" "$f" || true
done
echo ""
echo "Done. Logs: ${LOGDIR}"
