#!/usr/bin/env bash
# Run the warmup-vs-cache probe in TWO server configurations:
#   (1) --skip-server-warmup (default in our other probes)
#   (2) Without --skip-server-warmup (let SGLang do its built-in warmup)
# Then run the 3-request sequence in each.
set -uo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CTX_LEN=2624
PORT=30000
OUTDIR="/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures"
LOGDIR="${OUTDIR}/probe_c3_warmup_vs_cache_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGDIR}"

export SGLANG_TT_EXECUTION_BACKEND=tt_xla
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export PYTHONPATH="/sglang/python:${PYTHONPATH:-}"
cd /sglang

start_server() {
    local extra_args="$1"
    local label="$2"
    echo ""
    echo "==============================================================="
    echo "[server] starting (${label}, args=${extra_args})..."
    echo "==============================================================="
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --port ${PORT} --host 0.0.0.0 --device cpu \
        --context-length ${CTX_LEN} --mem-fraction-static 0.5 \
        --disable-cuda-graph --tp-size 1 \
        ${extra_args} \
        > "${LOGDIR}/server_${label}.log" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 240); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "[server] ${label} ready after ${i}s"
            return 0
        fi
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "[server] DIED during startup"
            tail -30 "${LOGDIR}/server_${label}.log"
            return 1
        fi
        sleep 1
    done
    echo "[server] ${label} timeout"
    kill ${SERVER_PID} 2>/dev/null || true
    return 1
}

stop_server() {
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
    sleep 2
}

run_probe() {
    local label="$1"
    echo ""
    echo "[probe] running warmup-vs-cache probe for: ${label}"
    python3 -u /sglang/python/sglang/srt/hardware_backend/tenstorrent/test/probe_c3_warmup_vs_cache.py \
        2>&1 | tee "${LOGDIR}/probe_${label}.log"
}

# Round 1: with --skip-server-warmup (matches our other probes)
if start_server "--skip-server-warmup" "skip_warmup"; then
    run_probe "skip_warmup"
    stop_server
fi

# Round 2: WITHOUT --skip-server-warmup (let SGLang try its built-in warmup)
if start_server "" "with_warmup"; then
    run_probe "with_warmup"
    stop_server
fi

echo ""
echo "Logs: ${LOGDIR}"
