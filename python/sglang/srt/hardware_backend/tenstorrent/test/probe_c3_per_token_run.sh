#!/usr/bin/env bash
# Spin up server, hit it with the streaming per-token probe, tear down.
set -uo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CTX_LEN=2624
PORT=30000
OUTDIR="/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures"
LOGDIR="${OUTDIR}/probe_c3_per_token_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGDIR}"

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

python3 -u /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/probe_c3_per_token.py \
    2>&1 | tee "${LOGDIR}/per_token.log"

echo ""
echo "[server] stopping..."
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true

echo ""
echo "Log: ${LOGDIR}/per_token.log"
