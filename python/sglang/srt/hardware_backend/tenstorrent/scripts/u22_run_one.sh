#!/usr/bin/env bash
#
# U22 helper: launch sglang server with a chosen U22 env profile,
# wait for readiness, issue a single /generate, kill the server,
# print the JSON response.  Use from inside p3a-ngram.
#
# Usage:
#   bash u22_run_one.sh <profile_tag> [extra_env_kv_pairs...]
#
# Example:
#   bash u22_run_one.sh u22_reshard_zero \
#     SGLANG_TT_USE_PREFETCHER=1 \
#     SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 \
#     SGLANG_TT_U22_RESHARD_W2=1 \
#     SGLANG_TT_U22_PRINT_ADDR=1
#
# The default-baked env vars (no prefetcher, no probes) are still
# applied; pass overrides on the command line.

set -uo pipefail

TAG="${1:-noprobe}"
shift || true

LOGDIR="/tmp/u22"
mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/${TAG}.log"
RESP="${LOGDIR}/${TAG}.resp.json"
ADDR_LOG="${LOGDIR}/${TAG}.addrs.txt"

# Best-effort kill any existing 30000 listener.
pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null || true
sleep 4

# Clear kernel cache (literal path; var-guarded).
TT_CACHE_HOME=/root/.cache/tt-metal-cache
if [ -n "${TT_CACHE_HOME}" ] && [ -d "${TT_CACHE_HOME}" ]; then
    rm -rf "${TT_CACHE_HOME}"/* 2>/dev/null || true
fi

# Compose the env declaration string.
ENV_DECL=""
for kv in "$@"; do
    ENV_DECL="${ENV_DECL} ${kv}"
done

echo "=== U22 run: ${TAG} ==="
echo "=== Extra env: ${ENV_DECL}"
echo "=== Log: ${LOG}"

# Launch server in background.  Inherit base env (so site-packages
# resolve), then export overrides.
export HF_HOME=/root/.cache/huggingface
export TT_METAL_HOME=/tt-metal
export TT_CACHE_HOME=/root/.cache/tt-metal-cache
export HF_MODEL=/models/Qwen3-8B
export SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged
export SGLANG_TT_DISABLE_PREFILL_TRACE=1

# Parse extra "K=V" pairs into env.
for kv in ${ENV_DECL}; do
    export "${kv?}"
done

(
    python3 -u -m sglang.launch_server \
        --model-path /models/Qwen3-8B --port 30000 --host 0.0.0.0 \
        --device tenstorrent --context-length 4096 \
        --mem-fraction-static 0.5 --disable-cuda-graph --tp-size 1 \
        --skip-server-warmup --max-running-requests 1 --trust-remote-code \
        --attention-backend torch_native
) > "${LOG}" 2>&1 &
SVR_PID=$!

# Wait for /health.
for i in $(seq 1 90); do
    sleep 5
    if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health 2>/dev/null | grep -q '^200$'; then
        echo "  Server ready in ${i} ticks (~${i}*5 s)."
        break
    fi
    if ! kill -0 ${SVR_PID} 2>/dev/null; then
        echo "  Server EXITED early.  Tail:"
        tail -40 "${LOG}"
        exit 2
    fi
done

if ! curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health | grep -q '^200$'; then
    echo "  Server NEVER became ready.  Tail:"
    tail -40 "${LOG}"
    kill -9 ${SVR_PID} 2>/dev/null || true
    exit 3
fi

# Single decode probe.
NTOK="${U22_NTOK:-15}"
curl -s -X POST http://127.0.0.1:30000/generate \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"What is 2+2?\",\"sampling_params\":{\"max_new_tokens\":${NTOK},\"temperature\":0.0}}" \
    > "${RESP}" || true

# Capture U22 address prints + U17 probe counts.
grep -E "U22_RESHARD_W2|U22_REALLOCATE_W2|U22_ASSIGN_W2|U17_PRE_RS in_addr=" "${LOG}" \
    | head -40 > "${ADDR_LOG}" || true

# Kill server.
kill -9 ${SVR_PID} 2>/dev/null || true
pkill -9 -f "sglang\.launch_server.*--port 30000" 2>/dev/null || true

echo "=== Response:"
cat "${RESP}"
echo ""
echo "=== Addr / probe lines (first 40):"
cat "${ADDR_LOG}"
echo ""
echo "=== Done."
