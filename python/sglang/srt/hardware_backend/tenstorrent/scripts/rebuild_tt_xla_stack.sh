#!/usr/bin/env bash
#
# Rebuild the full tt-xla stack (tt-metal -> tt-mlir-sglang fork -> tt-xla
# plugin) inside the tt-xla-eval container, with every workaround learned
# during the 2026-05-17/18 Tilize-attack session baked in.
#
# WHY THIS EXISTS:
# Building any of these pieces on the HOST side fails for path-baked reasons:
#   - /tt-xla/venv was created inside the container; host check rejects it
#     because VIRTUAL_ENV's baked-in path doesn't match the host's
#     /home/mhnie/tt-xla/venv.
#   - tt-mlir-sglang's env/activate defaults TTMLIR_TOOLCHAIN_DIR to
#     /opt/ttmlir-toolchain (no dash), but the actual path is
#     /opt/tt-mlir-toolchain (with dash) — must export the correct value
#     BEFORE sourcing env/activate.
#   - Stale CMakeCache.txt files from prior container builds contain
#     /tt-xla/... absolute paths that CMake refuses to use from the host's
#     /home/mhnie/tt-xla/... view.
#   - Tracy.hpp lives under the canonical install tree's tt-metal subtree;
#     the host's symlinks/paths don't always resolve correctly.
# Conclusion: builds MUST happen inside the container.
#
# USAGE (from host):
#   docker exec -it tt-xla-eval bash \
#     /sglang/python/sglang/srt/hardware_backend/tenstorrent/scripts/rebuild_tt_xla_stack.sh
#
# Optional positional args:
#   $1 = mode = "incremental" (default) | "from-scratch"
#       incremental: trust existing build dirs, run incremental ninja
#       from-scratch: rm -rf build dirs and reconfigure from scratch
#                     (~90-180 min including tt-metal clone+build)
#
# After this script succeeds, verify with:
#   docker exec tt-xla-eval bash -c \
#     'sha256sum /tt-xla/third_party/tt-mlir/install/lib/*.so'
# and compare against a known canonical baseline.
#
# IMPORTANT: This script ONLY rebuilds. It does NOT swap any libraries that
# the active pjrt-plugin-tt.so loads at runtime via $ORIGIN/../lib64. After
# rebuild, you must copy fork outputs into the plugin's lib64 — see the
# "install fork libs into plugin lib64" section at the end.

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

MODE="${1:-incremental}"

TTMLIR_DIR="/tt-mlir-sglang"
TTXLA_DIR="/tt-xla"
TTMLIR_TOOLCHAIN_DIR="/opt/tt-mlir-toolchain"
TT_METAL_BUILD_LIB="${TTMLIR_DIR}/third_party/tt-metal/src/tt-metal/build_Release/lib"
TTXLA_INSTALL_LIB="${TTXLA_DIR}/third_party/tt-mlir/install/lib"
PLUGIN_LIB64="${TTXLA_DIR}/python_package/pjrt_plugin_tt/lib64"
NPROC="$(nproc)"

# Confirm we're inside the container, not on host.
if [[ ! -d "${TTMLIR_DIR}" || ! -d "${TTXLA_DIR}" ]]; then
    echo "ERROR: Expected paths ${TTMLIR_DIR} and ${TTXLA_DIR} not found."
    echo "       This script must run INSIDE the tt-xla-eval container."
    echo "       Invoke via: docker exec tt-xla-eval bash $0 [mode]"
    exit 1
fi

echo "================================================================"
echo "  tt-xla stack rebuild ($(date))"
echo "  mode = ${MODE}"
echo "  TTMLIR_DIR = ${TTMLIR_DIR}"
echo "  TTXLA_DIR  = ${TTXLA_DIR}"
echo "================================================================"

# ============================================================================
# Step 0: Activate both envs (in correct order, with corrected paths)
# ============================================================================

# CRITICAL: must export the correct TTMLIR_TOOLCHAIN_DIR (with dash) BEFORE
# sourcing env/activate, because env/activate's default is wrong.
export TTMLIR_TOOLCHAIN_DIR

cd "${TTMLIR_DIR}"
# shellcheck disable=SC1091
source env/activate >/dev/null 2>&1 || true

# tt-xla's venv/activate hardcodes TTXLA_ENV_ACTIVATED=1 — set it manually
# in case venv/activate balks on path-relocation check.
export TTXLA_ENV_ACTIVATED=1
export VLLM_TARGET_DEVICE=empty
export TT_METAL_LOGGER_LEVEL=ERROR
export ARCH_NAME=wormhole_b0
export LD_LIBRARY_PATH="${TTMLIR_TOOLCHAIN_DIR}/lib:${LD_LIBRARY_PATH:-}"
export TT_MLIR_HOME="${TTXLA_DIR}/third_party/tt-mlir/src/tt-mlir/"

# ============================================================================
# Step 1: Clean stale state if from-scratch
# ============================================================================

if [[ "${MODE}" == "from-scratch" ]]; then
    echo "[1/6] Cleaning stale build state..."
    rm -rf \
        "${TTMLIR_DIR}/build" \
        "${TTXLA_DIR}/build-local" \
        "${TTXLA_DIR}/build" \
        "${TTXLA_DIR}/third_party/tt-mlir/src/tt-mlir-stamp" \
        "${TTXLA_DIR}/third_party/tt-mlir/src/tt-mlir" \
        "${TTXLA_DIR}/third_party/loguru/src/loguru-build" \
        "${TTXLA_DIR}/third_party/loguru/src/loguru-stamp"
    echo "  cleaned"
else
    echo "[1/6] Incremental mode — keeping existing build dirs"
fi

# ============================================================================
# Step 2: Configure + build fork tt-mlir with FULL canonical-style flags
# ============================================================================

echo "[2/6] Configure + build fork tt-mlir at ${TTMLIR_DIR}/build"

if [[ ! -f "${TTMLIR_DIR}/build/build.ninja" ]]; then
    # Configure with the FULL canonical flag set — NOT just the minimal set
    # in the original build_and_install.sh which only set PERF_TRACE=OFF
    # and produced only libTTMLIRCompiler.so. We need all 9 .so files.
    cmake -G Ninja -B "${TTMLIR_DIR}/build" -S "${TTMLIR_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang-17 \
        -DCMAKE_CXX_COMPILER=clang++-17 \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_PREFIX_PATH="${TTMLIR_TOOLCHAIN_DIR}" \
        -DMLIR_DIR="${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir" \
        -DLLVM_DIR="${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm" \
        -DTT_RUNTIME_ENABLE_TTNN=ON \
        -DTTMLIR_ENABLE_STABLEHLO=ON \
        -DTTMLIR_ENABLE_RUNTIME=ON \
        -DTTMLIR_ENABLE_OPMODEL=ON \
        -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DTTMLIR_ENABLE_TESTS=OFF \
        -DTTMLIR_ENABLE_TOOLS=ON \
        -DTTMLIR_ENABLE_DOCS=OFF \
        -DTTMLIR_ENABLE_PERF_TRACE=OFF \
        -DTT_RUNTIME_DEBUG=OFF \
        -DTT_USE_SYSTEM_SFPI=OFF
fi

# Build fork tt-mlir. This triggers tt-metal clone+build the first time.
# Known failure mode: tt-metal git clone hits transient TLS / network
# errors. Retry up to 3 times if ninja fails — the gitclone.cmake step
# does `rm -rf .../tt-metal` before retry, so retries are safe.
TRIES=0
MAX_TRIES=3
until ninja -C "${TTMLIR_DIR}/build" -j "${NPROC}"; do
    TRIES=$((TRIES + 1))
    if (( TRIES >= MAX_TRIES )); then
        echo "ERROR: fork tt-mlir build failed after ${MAX_TRIES} attempts"
        exit 2
    fi
    echo "WARN: ninja failed (attempt ${TRIES}/${MAX_TRIES}) — retrying"
    # Apply mid-build workarounds before retrying.
    apply_known_workarounds() {
        # Workaround #1: CMake's -lX flag transforms _ttnncpp.so / _ttnn.so
        # filenames into -l_ttnncpp / -l_ttnn, which the linker can't find
        # because lib prefix is missing. Create lib_*.so symlinks.
        if [[ -d "${TT_METAL_BUILD_LIB}" ]]; then
            for soname in _ttnncpp _ttnn; do
                if [[ -f "${TT_METAL_BUILD_LIB}/${soname}.so" \
                    && ! -e "${TT_METAL_BUILD_LIB}/lib${soname}.so" ]]; then
                    ln -sfn "${soname}.so" "${TT_METAL_BUILD_LIB}/lib${soname}.so"
                fi
            done
        fi
        # Workaround #2: tt-metal install step expects tt_metal/pre-compiled
        # dir to exist; if missing, install fails with "file INSTALL cannot
        # find...".
        local pre_compiled="${TTMLIR_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/pre-compiled"
        if [[ ! -d "${pre_compiled}" ]]; then
            mkdir -p "${pre_compiled}"
        fi
    }
    apply_known_workarounds
done

echo "  fork tt-mlir build done"

# ============================================================================
# Step 3: Install fork tt-mlir into the canonical install dir
# ============================================================================

echo "[3/6] Install fork outputs into ${TTXLA_INSTALL_LIB%/lib}"

# Install with explicit --prefix because fork's CMakeCache caches /usr/local
# (we didn't set CMAKE_INSTALL_PREFIX above). cmake --install --prefix
# overrides at install time.
cmake --install "${TTMLIR_DIR}/build" \
    --prefix "${TTXLA_INSTALL_LIB%/lib}" \
    --component SharedLib

cmake --install "${TTMLIR_DIR}/build" \
    --prefix "${TTXLA_INSTALL_LIB%/lib}" \
    --component DistributedRuntime 2>/dev/null || true

echo "  installed:"
ls "${TTXLA_INSTALL_LIB}"/*.so

# ============================================================================
# Step 4: Set up canonical-source symlink + stamp files
# ============================================================================

# tt-xla's CMakeLists.txt has an ExternalProject_Add(tt-mlir) that wants
# the source at /tt-xla/third_party/tt-mlir/src/tt-mlir. Point that at the
# fork via symlink + touched stamps so the EP skips download/configure/build.

CANONICAL_SRC_PATH="${TTXLA_DIR}/third_party/tt-mlir/src"
mkdir -p "${CANONICAL_SRC_PATH}" "${CANONICAL_SRC_PATH}/tt-mlir-stamp"

if [[ ! -L "${CANONICAL_SRC_PATH}/tt-mlir" && -e "${CANONICAL_SRC_PATH}/tt-mlir" ]]; then
    echo "WARN: ${CANONICAL_SRC_PATH}/tt-mlir exists and is NOT a symlink."
    echo "      Removing to replace with symlink to fork."
    rm -rf "${CANONICAL_SRC_PATH}/tt-mlir"
fi

ln -sfn "${TTMLIR_DIR}" "${CANONICAL_SRC_PATH}/tt-mlir"

# Touch every stamp file the ExternalProject_Add would create so it skips
# all sub-steps. Order matters — build stamp must be newer than configure
# stamp, install newer than build, etc. We just touch them all now (same
# mtime); CMake's check is "newer-than" so equal timestamps satisfy.
for stage in mkdir download update patch configure build install; do
    touch "${CANONICAL_SRC_PATH}/tt-mlir-stamp/tt-mlir-${stage}"
done

echo "  symlink + stamps in place"

# ============================================================================
# Step 5: Configure + build tt-xla plugin
# ============================================================================

echo "[5/6] Configure + build tt-xla plugin at ${TTXLA_DIR}/build-local"

if [[ ! -f "${TTXLA_DIR}/build-local/build.ninja" ]]; then
    cmake -G Ninja -B "${TTXLA_DIR}/build-local" -S "${TTXLA_DIR}" \
        -DCMAKE_BUILD_TYPE=Release
fi

# ninja for tt-xla plugin. May also trigger an incremental rebuild of the
# fork through the EP wrapper if stamps were re-invalidated; if that
# happens, the same workarounds apply.
TRIES=0
until ninja -C "${TTXLA_DIR}/build-local" -j "${NPROC}"; do
    TRIES=$((TRIES + 1))
    if (( TRIES >= MAX_TRIES )); then
        echo "ERROR: tt-xla plugin build failed after ${MAX_TRIES} attempts"
        exit 3
    fi
    echo "WARN: tt-xla ninja failed (attempt ${TRIES}/${MAX_TRIES})"
    # Re-apply workarounds in case the EP triggered an incremental rebuild
    for soname in _ttnncpp _ttnn; do
        [[ -f "${TT_METAL_BUILD_LIB}/${soname}.so" \
           && ! -e "${TT_METAL_BUILD_LIB}/lib${soname}.so" ]] && \
            ln -sfn "${soname}.so" "${TT_METAL_BUILD_LIB}/lib${soname}.so"
    done
    mkdir -p "${TTMLIR_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/pre-compiled"
done

echo "  tt-xla plugin built at ${TTXLA_DIR}/build-local/pjrt_implementation/src/pjrt_plugin_tt.so"

# ============================================================================
# Step 6: Copy fork libs into plugin's lib64 (the runtime rpath location)
# ============================================================================

# The plugin .so has rpath = $ORIGIN:$ORIGIN/lib:$ORIGIN/lib64:$ORIGIN/../pjrt_plugin_tt.libs
# At runtime, it searches lib64 BEFORE the install prefix. We must ensure
# lib64 contains fork's content (the install step above may not propagate
# here automatically; previous sessions verified this).
echo "[6/6] Copy fork libs into plugin's lib64 at ${PLUGIN_LIB64}"

mkdir -p "${PLUGIN_LIB64}"
cp -fp "${TTMLIR_DIR}/build/lib/libTTMLIRCompiler.so"           "${PLUGIN_LIB64}/"
cp -fp "${TTMLIR_DIR}/build/runtime/lib/libTTMLIRRuntime.so"    "${PLUGIN_LIB64}/"
# tt-metal libs come from the in-tree build dir, not from install/.
cp -fp "${TT_METAL_BUILD_LIB}/_ttnn.so"                          "${PLUGIN_LIB64}/"
cp -fp "${TT_METAL_BUILD_LIB}/_ttnncpp.so"                       "${PLUGIN_LIB64}/"
cp -fp "${TT_METAL_BUILD_LIB}/libtt_metal.so"                    "${PLUGIN_LIB64}/"
cp -fp "${TT_METAL_BUILD_LIB}/libtt-umd.so"                      "${PLUGIN_LIB64}/"
cp -fp "${TT_METAL_BUILD_LIB}/libtt_stl.so"                      "${PLUGIN_LIB64}/"
cp -fp "${TT_METAL_BUILD_LIB}/libtracy.so.0.10.0"                "${PLUGIN_LIB64}/"
# Symlinks for soname compat
ln -sfn libtracy.so.0.10.0 "${PLUGIN_LIB64}/libtracy.so"

echo "  plugin lib64 populated:"
ls "${PLUGIN_LIB64}"/*.so* 2>/dev/null | head -20

# ============================================================================
# Final verification
# ============================================================================

echo "================================================================"
echo "  REBUILD COMPLETE"
echo "================================================================"
echo "Plugin .so:        ${TTXLA_DIR}/python_package/pjrt_plugin_tt/pjrt_plugin_tt.so"
echo "                    -> ${TTXLA_DIR}/build-local/pjrt_implementation/src/pjrt_plugin_tt.so"
echo "Install lib SHAs:"
sha256sum "${TTXLA_INSTALL_LIB}"/*.so 2>/dev/null | sed 's|^|  |'
echo
echo "Plugin lib64 SHAs (active runtime libs):"
sha256sum "${PLUGIN_LIB64}"/*.so 2>/dev/null | sed 's|^|  |'
echo
echo "To smoke-test, run from host:"
echo "  docker exec -d tt-xla-eval bash -c 'cd /sglang && BYPASS_PREWARM=1 \\"
echo "    SGLANG_TT_CACHE_MODE=index_copy \\"
echo "    TT_METAL_RUNTIME_ROOT=/tt-xla/third_party/tt-mlir/install/tt-metal \\"
echo "    python3 -u python/sglang/srt/hardware_backend/tenstorrent/test/bench_3run_server_alive.py \\"
echo "    --models Qwen3-8B --backend tt_xla --input-len 1024 --output-len 1024 \\"
echo "    --out-tag rebuild_smoke > /tmp/rebuild_smoke.log 2>&1'"
echo "Expected TPOT warm: ~132 ms (canonical baseline)."
