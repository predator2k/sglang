#!/usr/bin/env bash
# One-shot setup for tt-xla eval container.
# Usage: bash setup_ttxla_container.sh
set -euo pipefail

CONTAINER_NAME="tt-xla-eval"
IMAGE="ghcr.io/tenstorrent/tt-xla-slim:latest"
SGLANG_DIR="/home/mhnie/sglang"

echo "[1/4] Removing old container..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "[2/4] Creating container..."
docker run -d --name "$CONTAINER_NAME" \
    --device /dev/tenstorrent \
    -v "${SGLANG_DIR}:/sglang" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    --ipc=host --cap-add=SYS_ADMIN \
    "$IMAGE" sleep infinity

echo "[3/4] Installing sglang (no-deps, preserves torch 2.9.1+cpu)..."
docker exec "$CONTAINER_NAME" pip install -e /sglang/python --no-deps --no-build-isolation -q

echo "[3a/4] Installing torchvision (matched to torch 2.9.1+cpu) with --no-deps..."
docker exec "$CONTAINER_NAME" pip install --no-deps -q \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.24.0

echo "[3b/4] Installing runtime deps for sglang server + bench_serving (no-deps, lock-safe)..."
docker exec "$CONTAINER_NAME" pip install --no-deps -q \
    orjson multiprocess pyarrow dill xxhash fsspec \
    IPython traitlets jedi prompt_toolkit pygments stack_data executing \
    pure_eval matplotlib_inline decorator asttokens wcwidth

echo "[3c/4] Installing runtime deps that pull transitive deps (server framework, datasets, CI tooling)..."
docker exec "$CONTAINER_NAME" pip install -q \
    fastapi uvicorn msgpack msgspec prometheus-client pyzmq sentencepiece \
    tiktoken openai einops blobfile llguidance outlines interegular modelscope \
    partial_json_parser pybase64 datasets scipy aiohttp anthropic gguf \
    python-multipart setproctitle psutil packaging pydantic requests pillow \
    soundfile pytest

echo "[4/4] Verifying..."
docker exec "$CONTAINER_NAME" python3 -c "
import torch, torch_xla, sglang
print(f'torch={torch.__version__} xla={torch_xla.__version__} sglang=OK')
import torch_xla.runtime as xr
xr.set_device_type('TT')
print(f'devices={xr.global_runtime_device_count()}')
"
echo "Done. Container '$CONTAINER_NAME' ready."
