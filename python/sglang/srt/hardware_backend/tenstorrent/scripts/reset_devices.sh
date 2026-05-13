#!/bin/bash
# Pinned tt-metal image (Phase 0 / P1-simple-path testing):
#   ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-55fd115-aa4ae1e
#   image ID: sha256:d4116d2a7b20383ec42c990b08f9c6159462f9f2b84f9a25448db678f17e41aa
#   - tt-metal under /home/container_app_user/tt-metal/ (NOT /tt-metal/)
#   - venv: /home/container_app_user/tt-metal/python_env/bin/activate
#   - tt_transformers present; generator_sglang.py NOT present (added in newer trees)
# UMD-compatible with host KMD 2.8.0 (P1-verified).
#
# Paged-path image (Phase 2+, generator_sglang.py available):
#   ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-models-amd64:latest-rc
#   image ID: sha256:40dcdcdabb5ea87a0700d7bdeeada290fe2a09246d4890237b8cd6828c1e360c
# WARNING: this image's tt-metal UMD does NOT match host KMD 2.8.0 — open_mesh_device
# fails with "Querying size for a host channel that does not exist" until either
# the host KMD is upgraded or a UMD-compatible image variant carrying generator_sglang.py
# is identified. See spec R1.
# Recovery script for Tenstorrent devices on this host.
#
# Run when a previous SGLang launch left the mesh in a bad state (e.g.
# open_mesh_device hangs, ETH fabric stuck, OOM not freed). Wipes the
# two p150a's via tt-smi.
#
# Usage: sudo bash python/sglang/srt/hardware_backend/tenstorrent/scripts/reset_devices.sh

set -e
echo "Resetting Tenstorrent devices 01:00.0 and 06:00.0..."
sudo tt-smi -r 0000:01:00.0 0000:06:00.0
echo "Done. Mesh state cleared."
