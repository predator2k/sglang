#!/usr/bin/env bash
# Single-server hierarchical-cache demo for compressed_file.
#
# - All KV pages evicted from device pool fall through to host RAM (L2) and
#   then to disk (L3). Disk goes through CompressedHiCacheFile, which applies
#   the policy.yaml-described codec/layout per (layer, K|V) slab.
# - Tier-aware rules in policy.yaml are *inert* in this mode (no caller passes
#   tier hints yet); the policy still differentiates layer-0 V from the rest.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PORT=${PORT:-30000}
CACHE_DIR=${CACHE_DIR:-/tmp/sglang_hicache_compressed_demo}
POLICY=${POLICY:-$DIR/policy.yaml}
EXTRA_JSON=$(python - <<EOF
import json
print(json.dumps({"profiles_yaml": "${POLICY}"}))
EOF
)

mkdir -p "$CACHE_DIR"
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$CACHE_DIR"

echo "model: $MODEL"
echo "cache: $CACHE_DIR"
echo "policy: $POLICY"
echo

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-storage-backend compressed_file \
    --hicache-storage-backend-extra-config "$EXTRA_JSON" \
    --hicache-write-policy write_back \
    --hicache-mem-layout layer_first
