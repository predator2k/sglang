#!/usr/bin/env bash
# Prefill worker for the PD-disaggregation demo. Does NOT need our backend
# (its KV is sent to the decode worker over the disagg transport), but it
# still emits the hicache_storage_backend setting so the upstream backup-on-
# prefill path uses CompressedHiCacheFile.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PORT=${PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-30001}
CACHE_DIR=${CACHE_DIR:-/tmp/sglang_hicache_compressed_prefill}
POLICY=${POLICY:-$DIR/policy.yaml}
EXTRA_JSON=$(python - <<EOF
import json
print(json.dumps({"profiles_yaml": "${POLICY}"}))
EOF
)

mkdir -p "$CACHE_DIR"
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$CACHE_DIR"

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --enable-hierarchical-cache \
    --hicache-storage-backend compressed_file \
    --hicache-storage-backend-extra-config "$EXTRA_JSON"
