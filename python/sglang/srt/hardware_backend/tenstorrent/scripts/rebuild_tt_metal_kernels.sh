#!/usr/bin/env bash
#
# Incremental rebuild of tt-metal-sglang C++ kernels ONLY.
# Skips toolchain, Python bindings reconfigure, and full clean.
#
# WHY THIS EXISTS:
# Cold full rebuild of tt-metal in p3a-ngram is ~30-60 min.
# When iterating on a single .cpp (e.g. layernorm kernels for
# the prefetcher Option A work), we only need to incrementally
# rebuild the affected .o + relink the shared lib. This takes
# 2-5 min instead.
#
# USAGE (from host):
#   podman exec p3a-ngram bash \
#     /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_metal_kernels.sh
#
# Optional positional arg:
#   $1 = TARGET = "ttnn" (default) | "ttnn_op_normalization" | any phony target
#       Use a narrower target (e.g. ttnn_op_normalization) when iterating
#       on a single subsystem to skip relinking unrelated translation units.
#       Run: ninja -t targets all | grep ttnn_op_ to discover available ones.
#
# Modelled on rebuild_tt_xla_stack.sh in the same directory.
# See that script's header for the "must run inside container"
# reasoning — same applies here.

set -eo pipefail

TT_METAL_DIR="/tt-metal"
BUILD_DIR="${TT_METAL_DIR}/build_Release"
TARGET="${1:-ttnn}"
NPROC="$(nproc)"

# Confirm we're inside the container
if [[ ! -d "${TT_METAL_DIR}" ]]; then
    echo "ERROR: ${TT_METAL_DIR} not found."
    echo "       This script must run INSIDE the p3a-ngram container."
    echo "       Invoke via: podman exec p3a-ngram bash $0 [target]"
    exit 1
fi

if [[ ! -f "${BUILD_DIR}/build.ninja" ]]; then
    echo "ERROR: ${BUILD_DIR}/build.ninja not found."
    echo "       A cold full rebuild must precede the first incremental run."
    echo "       Run the full tt-metal build first, then re-invoke this script."
    exit 1
fi

echo "================================================================"
echo "  tt-metal kernel-only rebuild ($(date))"
echo "  BUILD_DIR = ${BUILD_DIR}"
echo "  TARGET    = ${TARGET}"
echo "  NPROC     = ${NPROC}"
echo "================================================================"

cd "${BUILD_DIR}"

# Incremental ninja: only rebuild whatever depends on changed sources.
# The 'ttnn' phony target rebuilds all of TTNN (kernels + ops) and relinks
# _ttnn.so + _ttnncpp.so, which is what the prefetcher path actually loads.
# For single-subsystem iteration, pass a narrower target like
# ttnn_op_normalization to skip unaffected translation units entirely.
ninja -j "${NPROC}" "${TARGET}"

echo "================================================================"
echo "  Done. Output libs:"
ls -la "${BUILD_DIR}/ttnn/_ttnn.so" "${BUILD_DIR}/ttnn/_ttnncpp.so" 2>/dev/null || \
    echo "  (no _ttnn.so/_ttnncpp.so found — check target name)"
echo "================================================================"
