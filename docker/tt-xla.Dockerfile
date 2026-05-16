# tt-xla.Dockerfile -- SGLang + tt-xla combined image
#
# Base: ghcr.io/tenstorrent/tt-xla-slim (Python 3.12, torch 2.9.1+cpu,
#        pjrt-plugin-tt 1.1.0, torch_xla 2.9.0)
#
# Adds: Rust/cargo 1.82+ for SGLang's gRPC extension, then SGLang from
#        source via pip install -e. Falls back to PYTHONPATH if pip fails.
#
# Usage:
#   docker build -f docker/tt-xla.Dockerfile -t tt-xla-serve .
#   docker run --name tt-xla-serve --privileged \
#     --device /dev/tenstorrent \
#     --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
#     -v /home/mhnie/sglang:/sglang \
#     -v /home/mhnie/tt-models:/models \
#     -p 30000:30000 \
#     tt-xla-serve

FROM ghcr.io/tenstorrent/tt-xla-slim:latest

# ── 1. Fix torchvision (container ships a broken build) ──────────────
RUN pip uninstall -y torchvision && \
    pip install --no-cache-dir torchvision==0.24.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# ── 2. Install Rust/cargo (needed for SGLang's sglang_router crate) ──
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ── 3. Install SGLang runtime deps not already in base image ─────────
#    orjson: sglang.__init__ top-level import
#    ipython: sglang.utils top-level import
#    soundfile: sglang.srt.entrypoints.openai.streaming_asr top-level import
#    compressed-tensors: needed for some model loading paths
#    datasets: HF datasets used by some benchmark/eval paths
RUN pip install --no-cache-dir \
    orjson \
    ipython \
    soundfile \
    compressed-tensors \
    datasets \
    einops \
    gguf \
    interegular \
    ninja \
    easydict \
    partial_json_parser \
    python-multipart \
    py-spy

# ── 4. Install SGLang from source (editable) ─────────────────────────
#    The /sglang mount is expected at runtime; at build time we copy
#    only pyproject.toml to check the install. If pip install fails
#    (e.g. missing system libs for sgl-kernel CUDA), we set PYTHONPATH
#    as fallback.
#
#    At runtime the actual source is bind-mounted at /sglang.
ENV PYTHONPATH="/sglang/python:${PYTHONPATH}"
ENV SGLANG_IS_IN_CI="false"

# ── 5. Default environment for TT backend ────────────────────────────
ENV SGLANG_PLATFORM="tenstorrent"
ENV SGLANG_TT_EXECUTION_BACKEND="tt_xla"

# ── 6. Healthcheck ───────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:30000/health || exit 1

EXPOSE 30000

# Default entrypoint: launch SGLang server with tt-xla backend
# NOTE: --skip-server-warmup is needed because the first forward pass
# triggers JIT compilation (~2-5 min) which exceeds the warmup timeout.
# --watchdog-timeout 600 prevents the watchdog from killing the scheduler
# during JIT compilation.
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
CMD ["--model-path", "/models/TinyLlama-1.1B-Chat-v1.0", \
     "--trust-remote-code", \
     "--host", "0.0.0.0", "--port", "30000", \
     "--device", "tenstorrent", \
     "--max-running-requests", "1", \
     "--context-length", "2048", \
     "--attention-backend", "torch_native", \
     "--disable-cuda-graph", \
     "--skip-server-warmup", \
     "--watchdog-timeout", "600"]
