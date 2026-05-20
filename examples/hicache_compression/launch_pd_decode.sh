#!/usr/bin/env bash
# Decode worker for the PD disaggregation demo. This is the primary consumer
# of the compressed_file backend: decode-side KV cache is offloaded to disk
# via DecodeKVCacheOffloadManager which writes through the storage backend.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PORT=${PORT:-30100}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-30001}
CACHE_DIR=${CACHE_DIR:-/tmp/sglang_hicache_compressed_decode}
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
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --disaggregation-decode-enable-offload-kvcache \
    --hicache-storage-backend compressed_file \
    --hicache-storage-backend-extra-config "$EXTRA_JSON"
