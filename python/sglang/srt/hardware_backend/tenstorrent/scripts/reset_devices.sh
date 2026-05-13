#!/bin/bash
# Pinned tt-metal image: ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-models-amd64:latest-rc (sha256:40dcdcdabb5ea87a0700d7bdeeada290fe2a09246d4890237b8cd6828c1e360c)
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
