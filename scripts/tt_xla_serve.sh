#!/usr/bin/env bash
# tt_xla_serve.sh -- Build and launch SGLang with tt-xla backend in a container.
#
# Usage:
#   ./scripts/tt_xla_serve.sh [build]   # build image, then launch
#   ./scripts/tt_xla_serve.sh launch     # launch only (image must exist)
#   ./scripts/tt_xla_serve.sh test       # send a test request to running server
#   ./scripts/tt_xla_serve.sh bench      # quick throughput benchmark
#   ./scripts/tt_xla_serve.sh stop       # stop the container

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="tt-xla-serve"
CONTAINER_NAME="tt-xla-serve"
PORT="${TT_XLA_PORT:-30000}"
MODEL_PATH="${TT_XLA_MODEL:-/models/Qwen3-1.7B}"

# ── Helpers ──────────────────────────────────────────────────────────
log() { echo "[tt-xla-serve] $(date +%H:%M:%S) $*"; }

wait_for_health() {
    local max_wait=${1:-300}
    local elapsed=0
    log "Waiting for server health (max ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            log "Server healthy after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "ERROR: Server did not become healthy after ${max_wait}s"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -50
    return 1
}

# ── Commands ─────────────────────────────────────────────────────────
cmd_build() {
    log "Building image $IMAGE_NAME..."
    docker build -f "$REPO_ROOT/docker/tt-xla.Dockerfile" \
        -t "$IMAGE_NAME" \
        "$REPO_ROOT"
    log "Image built: $IMAGE_NAME"
}

cmd_launch() {
    # Stop existing container if any.
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    # Reset TT devices.
    log "Resetting TT devices..."
    /home/mhnie/.tenstorrent-venv/bin/tt-smi -r 0,1 2>/dev/null || true
    sleep 2

    log "Launching container $CONTAINER_NAME (model=$MODEL_PATH, port=$PORT)..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --privileged \
        --device /dev/tenstorrent \
        --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
        -v /home/mhnie/sglang:/sglang \
        -v /home/mhnie/tt-models:/models \
        -p "${PORT}:30000" \
        -e SGLANG_PLATFORM=tenstorrent \
        -e SGLANG_TT_EXECUTION_BACKEND=tt_xla \
        "$IMAGE_NAME" \
        --model-path "$MODEL_PATH" \
        --trust-remote-code \
        --host 0.0.0.0 --port 30000 \
        --device tenstorrent \
        --max-running-requests 1 \
        --context-length 2048 \
        --attention-backend torch_native \
        --disable-cuda-graph

    wait_for_health 600
}

cmd_test() {
    log "Sending test request..."
    curl -s "http://localhost:${PORT}/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 32
            }
        }' | python3 -m json.tool
}

cmd_bench() {
    log "Quick throughput benchmark (5 requests, 64 tokens each)..."
    local total_tokens=0
    local start=$(date +%s%N)

    for i in $(seq 1 5); do
        local resp
        resp=$(curl -s "http://localhost:${PORT}/generate" \
            -H "Content-Type: application/json" \
            -d '{
                "text": "Explain quantum computing in simple terms:",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64
                }
            }')
        local tok
        tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('meta_info',{}).get('completion_tokens', len(d.get('text','').split())))" 2>/dev/null || echo "64")
        total_tokens=$((total_tokens + tok))
        echo "  Request $i: ${tok} tokens"
    done

    local end=$(date +%s%N)
    local elapsed_ms=$(( (end - start) / 1000000 ))
    local elapsed_s=$(echo "scale=2; $elapsed_ms / 1000" | bc)
    local tps=$(echo "scale=2; $total_tokens / $elapsed_s" | bc)
    log "Total: ${total_tokens} tokens in ${elapsed_s}s = ${tps} tok/s"
}

cmd_stop() {
    log "Stopping container $CONTAINER_NAME..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    log "Stopped."
}

# ── Main ─────────────────────────────────────────────────────────────
case "${1:-build}" in
    build)
        cmd_build
        cmd_launch
        ;;
    launch)
        cmd_launch
        ;;
    test)
        cmd_test
        ;;
    bench)
        cmd_bench
        ;;
    stop)
        cmd_stop
        ;;
    *)
        echo "Usage: $0 {build|launch|test|bench|stop}"
        exit 1
        ;;
esac
