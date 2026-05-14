#!/bin/bash
# Bring up the p3a-ngram container for SGLang-on-Tenstorrent development.
#
# Idempotent: skips already-completed steps. Run after a host reboot, after
# `podman rm`, or after the container's overlay is wiped.
#
# Image:       localhost/local-tt-metal:dev  (sha256:973e972bddf5)
#              Built from /home/mhnie/tt-metal at commit 89686ee7
#              (UMD bump 2026-05-12), see scripts/reset_devices.sh.
# Container:   p3a-ngram
# Mounts:      /home/mhnie/sglang   -> /sglang
#              /home/mhnie/tt-models -> /models
#              /dev/hugepages-1G    -> /dev/hugepages-1G
# Devices:     /dev/tenstorrent     (all p150a's)
# Network:     host (port 30000 exposed direct)
# Privileged:  yes — required so /sys/kernel/mm/hugepages is unmasked
#              (tt-metal UMD reads nr_hugepages at device init). Rootless
#              podman masks /sys/kernel by default; neither a plain bind
#              mount nor `--security-opt unmask=...` lifts that mask, only
#              `--privileged` does. Dev box with direct TT device
#              passthrough — privileged adds no meaningful new attack
#              surface here.
#
# Launch note: sglang's server_args init requires `--device tenstorrent` on
# the CLI to enable the TT path. `SGLANG_PLATFORM=tenstorrent` alone is not
# enough — `get_device()` returns "No accelerator available" without the
# explicit flag.
#
# Inside container, the runtime venv is /opt/venv (uv-managed, no system pip).
# SGLang is installed editable from /sglang/python with --no-build-isolation
# (avoids the Rust toolchain requirement that vcs_versioning would otherwise
# pull in). Transformers must be >=5.0 (our model_config.py patch targets 5.x).
#
# Usage:
#   bash python/sglang/srt/hardware_backend/tenstorrent/scripts/bootstrap_container.sh

set -euo pipefail

NAME="p3a-ngram"
IMAGE="localhost/local-tt-metal:dev"
HOST_SGLANG="/home/mhnie/sglang"
HOST_MODELS="/home/mhnie/tt-models"

if ! podman image exists "$IMAGE"; then
  echo "ERROR: image $IMAGE not found locally. Build it from /home/mhnie/tt-metal first."
  exit 1
fi

if podman ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  STATUS=$(podman ps -a --format '{{.Names}} {{.Status}}' | awk -v n="$NAME" '$1==n{print $2}')
  if [ "$STATUS" != "Up" ] && [ "$STATUS" != "running" ]; then
    echo "Container $NAME exists but not running; starting..."
    podman start "$NAME" >/dev/null
  else
    echo "Container $NAME already running."
  fi
else
  echo "Creating container $NAME..."
  # Rootless podman masks /sys/kernel with an empty tmpfs by default, hiding
  # the hugepages sysfs that tt-metal's UMD reads at device init. Neither a
  # plain bind-mount nor `--security-opt unmask=...` lifts that mask; only
  # `--privileged` (or a fully unmasked /sys) makes it visible. This is a
  # dev box with direct TT device passthrough already in place, so the
  # privileged flag adds no meaningful new attack surface here.
  podman run -d --name "$NAME" \
    --privileged \
    --device /dev/tenstorrent \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "${HOST_SGLANG}:/sglang" \
    -v "${HOST_MODELS}:/models" \
    --network host \
    --shm-size=8g \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c "sleep infinity" >/dev/null
fi

echo "Applying tt-metal in-container patches..."
podman exec "$NAME" bash -c '
set -e
cd /tt-metal
for p in /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/tt_metal_patches/0*.patch; do
  if patch -p1 --dry-run --silent < "$p" >/dev/null 2>&1; then
    echo "applying $(basename "$p")"
    patch -p1 --silent < "$p" >/dev/null
  fi
done
'

echo "Installing/refreshing SGLang editable + minimum runtime deps..."
podman exec "$NAME" bash -c '
set -e
source /opt/venv/bin/activate
# pip is not in this venv (uv-managed). Use uv directly.
uv pip install --no-build-isolation -e /sglang/python --no-deps >/dev/null
# Runtime deps required for sglang import + launch_server entry path.
# CUDA-only packages intentionally omitted (cuda-python, flashinfer_*, flash-attn-4,
# nvidia-cutlass-dsl, sgl-deep-gemm, sglang-kernel, quack-kernels, tilelang).
# transformers>=5.0 is required by our tt_transformers/tt/model_config.py patch.
uv pip install --no-build-isolation \
  pybase64 fastapi uvicorn uvloop aiohttp msgspec setproctitle python-multipart \
  IPython modelscope einops gguf interegular llguidance partial_json_parser \
  openai-harmony pillow psutil py-spy requests scipy sentencepiece blobfile \
  compressed-tensors easydict timm soundfile build datasets ninja anthropic \
  openai outlines==0.1.11 packaging nvidia-ml-py triton \
  "transformers>=5.0" >/dev/null
python -c "import sglang; print(\"sglang ok:\", sglang.__file__)"
'

echo "Done. Inside-container quick test:"
echo "  podman exec -e SGLANG_PLATFORM=tenstorrent p3a-ngram bash -lc 'python -m sglang.launch_server --help | head -5'"
