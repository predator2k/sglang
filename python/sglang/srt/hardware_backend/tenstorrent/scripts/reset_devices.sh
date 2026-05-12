#!/bin/bash
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
