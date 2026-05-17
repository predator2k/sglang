#!/usr/bin/env bash
# C3 probe: measure TPOT through SGLang scheduler vs standalone.
#
# Standalone TinyLlama 2K = 68ms TPOT (v144 measurement).
# Goal: measure same workload through SGLang server + bench_serving.
# Gap reveals scheduler/HTTP/sampler/detokenizer overhead.
set -uo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_LEN=2048
OUTPUT_LEN=64       # short — 64×60ms = 4s decode, fast
PORT=30000
CTX_LEN=$(( INPUT_LEN + OUTPUT_LEN + 512 ))
OUTDIR="/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures"
LOGDIR="${OUTDIR}/probe_c3_scheduler_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGDIR}"

echo "=== C3 probe: bench_serving TPOT for TinyLlama 2K ==="

export SGLANG_TT_EXECUTION_BACKEND=tt_xla
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export PYTHONPATH="/sglang/python:${PYTHONPATH:-}"
cd /sglang

echo "[server] starting..."
python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --port ${PORT} --host 0.0.0.0 --device cpu \
    --context-length ${CTX_LEN} --mem-fraction-static 0.5 \
    --disable-cuda-graph --tp-size 1 --skip-server-warmup \
    > "${LOGDIR}/server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "[server] ready after ${i}s"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[server] DIED during startup"
        tail -30 "${LOGDIR}/server.log"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== PASS 1: JIT warmup (seed=42) ==="
python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:${PORT} \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio 0.0 \
    --num-prompts 1 \
    --seed 42 \
    --request-rate inf \
    --disable-tqdm \
    --output-file "${LOGDIR}/pass1_warmup.jsonl" 2>&1 | tee "${LOGDIR}/pass1_warmup.log"

echo ""
echo "=== PASS 2: JIT cache hit, real measurement (seed=123) ==="
python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:${PORT} \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio 0.0 \
    --num-prompts 1 \
    --seed 123 \
    --request-rate inf \
    --disable-tqdm \
    --output-file "${LOGDIR}/pass2_hit.jsonl" 2>&1 | tee "${LOGDIR}/pass2_hit.log"

echo ""
echo "[server] stopping..."
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true

echo ""
echo "=== Results ==="
for f in "${LOGDIR}"/pass*.log; do
    echo "--- $(basename $f) ---"
    grep -E "Mean TTFT|Mean TPOT|Output token throughput|Mean E2E|Median TPOT" "$f" || echo "  (no TPOT lines found)"
done

echo ""
echo "Standalone reference (v144): TinyLlama 2K = 68ms/tok"
echo "Logs: ${LOGDIR}"
