"""TTSRTPlatform + MeshDeviceCtx + activate_tt_platform entry point.

See docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md.
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
import time
from typing import Any

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger("sglang.srt.hardware_backend.tenstorrent")


def activate_tt_platform() -> str | None:
    """Entry-point activation function for sglang.srt.platforms.

    Returns the fully-qualified class string when a TT runtime is available:
    - ttnn (tt-metal container for tt_transformers_single / tt_transformers_paged)
    - pjrt_plugin_tt (tt-xla container for tt_xla backend)

    Returns None on hosts without either, so SGLang falls back to the
    default platform.
    """
    has_ttnn = False
    has_ttxla = False
    try:
        import ttnn  # noqa: F401
        has_ttnn = True
    except ImportError:
        pass
    try:
        import pjrt_plugin_tt  # noqa: F401
        has_ttxla = True
    except ImportError:
        pass

    if not has_ttnn and not has_ttxla:
        return None

    # Register Tenstorrent models with SGLang ModelRegistry (INV-6).
    # Side-effect import: models/__init__.py calls register_tt_models()
    from sglang.srt.hardware_backend.tenstorrent import models  # noqa: F401
    logger.info(
        "[TT-Platform] Tenstorrent model arches registered"
        f" (ttnn={has_ttnn}, tt-xla={has_ttxla})"
    )
    return "sglang.srt.hardware_backend.tenstorrent.platform:TTSRTPlatform"


class TTSRTPlatform(SRTPlatform):
    """Platform plugin for 2x Tenstorrent Blackhole p150a.

    P1 black-box integration: SGLang owns scheduler/HTTP/sampler;
    tt_transformers owns model forward + KV cache. See
    docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md.
    """

    _enum = PlatformEnum.OOT
    device_name = "tenstorrent"
    device_type = "cpu"  # CRITICAL: torch.get_device_module("tenstorrent") would fail
                         # See spec §3.2 invariant #3 and §6.1.

    # ---- Spec §6.1 capability flags ----
    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def supports_fp8(self) -> bool:
        return False

    # ---- Subsystem factories ----
    def get_default_attention_backend(self) -> str:
        # Returns "tenstorrent" only because the SRTPlatform contract
        # requires a string. TTModelRunner.initialize() (Phase D) will
        # override the default ModelRunner init to set
        # self.attn_backend = None and never invoke the attention
        # registry, so this string is never actually looked up in P1.
        return "tenstorrent"

    def _uses_model_registry(self) -> bool:
        """Return True when backend uses SGLang ModelRegistry (paged or tt-xla)."""
        from sglang.srt.hardware_backend.tenstorrent.execution import (
            resolve_execution_backend_name,
        )
        return resolve_execution_backend_name() in (
            "tt_transformers_paged",
            "tt_xla",
        )

    def apply_server_args_defaults(self, server_args):
        import os

        model_registry = self._uses_model_registry()

        from sglang.srt.hardware_backend.tenstorrent.execution import (
            resolve_execution_backend_name,
        )
        _backend = resolve_execution_backend_name()
        paged = _backend == "tt_transformers_paged"

        # Simple (P1) path hard constraints — relaxed for model-registry backends.
        if not model_registry:
            # Simple path: B=1 enforced; chunked prefill disabled; no radix
            # cache (tt_transformers manages KV internally without page tables).
            server_args.max_running_requests = 1
            server_args.chunked_prefill_size = -1  # -1 disables chunked prefill
            server_args.disable_radix_cache = True
        # Paged mode: max_running_requests and chunked_prefill_size come from
        # CLI args (the plugin's BaseMetalDeviceRunner uses max_running_requests
        # to size the decode batch).  Radix cache is left to user preference.

        # CRITICAL: overlap scheduler creates FutureMap + copy streams that
        # the synchronous ttnn forward path cannot satisfy. Without this,
        # FutureMap allocates -1 sentinels that get written into
        # req.output_ids → garbage tokens.
        server_args.disable_overlap_schedule = True

        # Sampling backend setting is defensive — P1 actually does greedy
        # in the worker (see spec §5.1) and never invokes
        # model_runner.sample(). "pytorch" gives a non-CUDA-specific
        # selection for any code path that does check this value.
        server_args.sampling_backend = "pytorch"

        # Disable CUDA-only paths
        server_args.pre_warm_nccl = False
        server_args.cpu_offload_gb = 0  # 0 = offloader disabled (also default)
        server_args.enable_torch_compile = False

        # torch has no "tenstorrent" device module — ModelRunner.__init__
        # calls torch.get_device_module(server_args.device).set_device(...)
        # during init_torch_distributed, which runs BEFORE we can override
        # self.device in TTModelRunner. So we set "cpu" up front; the TT
        # platform identity comes from SGLANG_PLATFORM=tenstorrent and is
        # already activated. Host tensors flow through CPU torch anyway
        # (sampling, kv-bookkeeping); the real device work lives inside
        # the TTExecutionBackend via ttnn.
        server_args.device = "cpu"

        # TP visibility: SGLang sees tp_size = 1 in P1. The "TP=2" mesh is
        # entirely inside ttnn's mesh_device — invisible to SGLang. Users
        # invoke with --tp 1 (or omit, defaulting to 1). If they explicitly
        # pass --tp 2, raise at startup rather than silently override.
        if server_args.tp_size not in (None, 1):
            raise ValueError(
                f"TT backend requires --tp 1 (TT-internal TP=2 is hidden); "
                f"got --tp {server_args.tp_size}."
            )
        server_args.tp_size = 1

        # No NCCL / disaggregation
        server_args.enable_dp_attention = False

        # SGLang's `_get_attention_backend_from_str` whitelist rejects
        # "tenstorrent" (auto-derived from device name). Force the pass-through
        # "torch_native" backend — our paged path doesn't actually use SGLang's
        # attention abstraction (model.forward calls tt_transformers directly),
        # so the backend choice only needs to validate. P3a.2 patched
        # draft_utils.DraftBackendFactory to route torch_native to
        # TTMultiStepDraftBackend for the EAGLE draft path.
        if server_args.attention_backend in (None, "tenstorrent"):
            server_args.attention_backend = "torch_native"

        # Force page_size=64 on TT (paged path). tt_transformers'
        # `paged_scaled_dot_product_attention_decode` and BFP8 tile-packing
        # require the per-block dim to be 32-aligned (tiles are 32×32);
        # the kv_cache shape SGLang computes from page_size is
        # `(num_blocks, num_kv_heads, page_size, head_size)`, and
        # `pack_as_bfp8_tiles` on `(…, page_size, head_size)` segfaults
        # when page_size doesn't satisfy that alignment.
        #
        # SGLang's default for attention_backend="tenstorrent" is
        # page_size=1 (server_args._handle_page_size), which crashes here.
        # The user-facing knob `--page-size 64` works, but missing it should
        # not segfault — force the supported value if the user left it at
        # the default. Anything else gets a hard error.
        if paged:
            if server_args.page_size in (None, 1):
                server_args.page_size = 64
            elif server_args.page_size != 64:
                raise ValueError(
                    f"TT paged path requires --page-size 64 "
                    f"(tt-metal tile alignment); got {server_args.page_size}."
                )

        # Disable grammar backend — P1/P2a does completion-only, no JSON /
        # regex / tool-call constraints. The bundled xgrammar in tt-metal docker
        # is older than current sglang expects (missing StructuralTag), so
        # picking "none" avoids an ImportError at scheduler init.
        server_args.grammar_backend = "none"

        if not paged:
            # tt_transformers' MAX_PREFILL_CHUNK_SIZES_DIV1024 table has no
            # "P300" (2x p150a) entry, so it falls back to 4 (= 4*1024 = 4096
            # tokens) — any prompt longer than that triggers tt_transformers'
            # chunked-prefill path, which REQUIRES paged attention
            # (Generator.prefill_forward_single_user_text:164 asserts
            # `page_table is not None`). Simple path runs non-paged, so
            # chunking is off-limits.
            #
            # Lift the single-chunk ceiling to 8K tokens (well above the 5K
            # workloads we want to support) by setting the env var the table
            # consults. p150a L1 is large enough for an 8K-token prefill of
            # Llama-3.1-8B in BFP8/BF16 — anything bigger may OOM L1, hence
            # the conservative 8 (not 16/32). Override on the command line
            # if needed.
            os.environ.setdefault("MAX_PREFILL_CHUNK_SIZE", "8")

    def get_mha_kv_pool_cls(self):
        # Simple/P1 path: TTModelRunner constructs _DummyKVCache directly,
        # so this factory should never be invoked. Paged/P2a path: SGLang's
        # standard ModelRunner calls this to construct its slot-index tracker
        # (KV bytes themselves live on TT device, owned by tt_transformers).
        # tt_xla path: model manages its own StaticCache internally, but
        # SGLang's ModelRunner still needs a pool for memory accounting.
        # Return MHATokenToKVPool — it allocates CPU phantom tensors.
        import os
        backend = os.environ.get("SGLANG_TT_EXECUTION_BACKEND", "")
        if backend in ("tt_transformers_paged", "tt_xla"):
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
            return MHATokenToKVPool
        raise NotImplementedError(
            "TT backend (simple path) constructs _DummyKVCache inside "
            "TTModelRunner.initialize(); the factory should not be called."
        )

    def get_paged_allocator_cls(self):
        """Paged path needs a slot-index tracker. KV bytes themselves live on
        TT device via plugin's allocate_kv_cache; this allocator only manages
        indices.

        SGLang's stock `PagedTokenToKVPoolAllocator.alloc_extend` /
        `alloc_decode` call Triton kernels that fail on TT (no active
        driver). `TTCpuPagedTokenToKVPoolAllocator` overrides only those
        two methods with pure-CPU ports — semantically identical, runs in
        plain torch on the host. Everything else (`alloc`, `free`, sorting,
        clearing) is inherited unchanged.
        """
        import os
        backend = os.environ.get("SGLANG_TT_EXECUTION_BACKEND", "")
        if backend == "tt_transformers_paged":
            from sglang.srt.hardware_backend.tenstorrent.cpu_paged_allocator import (
                TTCpuPagedTokenToKVPoolAllocator,
            )
            return TTCpuPagedTokenToKVPoolAllocator
        if backend == "tt_xla":
            # tt_xla manages KV cache internally (StaticCache). Return the
            # CPU-based allocator for SGLang's memory pool bookkeeping.
            from sglang.srt.hardware_backend.tenstorrent.cpu_paged_allocator import (
                TTCpuPagedTokenToKVPoolAllocator,
            )
            return TTCpuPagedTokenToKVPoolAllocator
        raise NotImplementedError("Paged allocator not used in simple path")

    def get_mla_kv_pool_cls(self):
        raise NotImplementedError("MLA not in P1")

    def get_nsa_kv_pool_cls(self):
        raise NotImplementedError("NSA not in P1")

    def get_graph_runner_cls(self) -> type:
        # P1 sets support_cuda_graph()=False so the framework will not
        # call this. If it does, that's a bug — fail loudly.
        raise NotImplementedError("P1 does not use graph runners")

    # ---- Memory accounting (called by ServerArgs.__post_init__) ----

    # Blackhole p150a: 28 GB GDDR6 per card. The framework only uses this
    # to size mem_fraction_static heuristically; SGLang's "device memory"
    # accounting is irrelevant to ttnn's internal KV management (P1 is
    # black-box), so the precise number doesn't drive correctness — just
    # avoids a NotImplementedError in ServerArgs.__post_init__.
    _DEVICE_TOTAL_MEMORY_BYTES = 28 * (1024**3)

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return self._DEVICE_TOTAL_MEMORY_BYTES

    def get_current_memory_usage(self, device=None) -> float:
        # P1 does NOT track per-process memory on TT cards — ttnn owns
        # the allocator internally. Return 0.0 (no peak observed); the
        # value flows into stats logging but not into any scheduling
        # decision because mem_fraction_static is fixed at startup.
        return 0.0


class MeshDeviceCtx:
    """Per-Scheduler-subprocess mesh-device lifecycle holder.

    Opens a 1x2 mesh on Tenstorrent devices 0 and 1 with fabric enabled and
    Blackhole-appropriate dispatch config (MUX + ROW), mirroring the
    `mesh_device` fixture in tt-metal's conftest.py.

    Signal/atexit handlers MUST be registered from the Scheduler subprocess
    — registering them at plugin-import time in the parent process would
    catch the wrong PID's signals. Construct this object from inside the
    worker init, not from `activate_tt_platform()`.
    """

    # Defaults match the demo (Phase 0.1 evidence + tt-metal conftest):
    # `device_params=[{"fabric_config": True, "trace_region_size": 30_000_000,
    # "num_command_queues": 1}]`. Override via kwargs if a future workload
    # needs a different shape.
    DEFAULT_TRACE_REGION_SIZE = 30_000_000
    DEFAULT_NUM_COMMAND_QUEUES = 1
    DEFAULT_MESH_SHAPE = (1, 2)
    CLOSE_TIMEOUT_S = 5.0

    def __init__(
        self,
        *,
        mesh_shape: tuple[int, int] = DEFAULT_MESH_SHAPE,
        trace_region_size: int = DEFAULT_TRACE_REGION_SIZE,
        num_command_queues: int = DEFAULT_NUM_COMMAND_QUEUES,
    ):
        import ttnn

        self._ttnn = ttnn
        self._fabric_was_set = False
        self.mesh: Any | None = None
        self._closed = False
        self._close_lock = threading.Lock()

        # Set fabric BEFORE open_mesh_device (mandatory ordering per
        # tt-metal/conftest.py set_fabric helper).
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,  # num_planes
            ttnn.FabricTensixConfig.MUX,
        )
        self._fabric_was_set = True

        # Blackhole + fabric requires ROW dispatch (per
        # tests/scripts/common.py:get_updated_device_params).
        dispatch_core_config = ttnn.DispatchCoreConfig(
            None,  # dispatch_core_type (default)
            ttnn.DispatchCoreAxis.ROW,
            ttnn.FabricTensixConfig.MUX,
        )

        shape = ttnn.MeshShape(*mesh_shape)
        try:
            self.mesh = ttnn.open_mesh_device(
                mesh_shape=shape,
                dispatch_core_config=dispatch_core_config,
                trace_region_size=trace_region_size,
                num_command_queues=num_command_queues,
            )
        except Exception:
            # Reset fabric if open fails so the next attempt isn't blocked.
            self._reset_fabric_safely()
            raise

        # Persist compiled kernel binaries to disk. tt-metal's default is
        # in-memory only — every fresh process re-JITs all per-shape kernels
        # (~5–10s per never-seen (seq_len, layer-config) shape). Enabling
        # this drops fresh-process cold-prefill from ~9.5s to ~250ms when
        # the shape was previously compiled on this host. The on-disk
        # binaries are reused across processes; no downside other than disk
        # footprint.
        try:
            ttnn.device.EnablePersistentKernelCache()
        except Exception as exc:
            # Non-fatal: an older ttnn build may not have this binding.
            # The mesh still works; cold-prefill will just remain slow.
            logger.warning(
                "persistent_kernel_cache_unavailable",
                extra={"reason": repr(exc)},
            )

        try:
            pci_ids = [
                ttnn.GetPCIeDeviceID(i)
                for i in range(self.mesh.get_num_devices())
            ]
            bdfs = [f"0000:{pid:02x}:00.0" for pid in pci_ids]
        except Exception:
            bdfs = []

        logger.info(
            "mesh_open",
            extra={
                "bdfs": bdfs,
                "mesh_shape": list(mesh_shape),
                "num_devices": self.mesh.get_num_devices(),
            },
        )

        # Register handlers ONLY from the subprocess we're in. SIGTERM/SIGINT
        # → graceful close. atexit covers normal interpreter shutdown.
        atexit.register(self._safe_close)
        try:
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)
        except ValueError:
            # signal.signal() only works in main thread; if scheduler runs
            # in a side thread the registration silently fails. atexit
            # still covers normal shutdown.
            logger.warning(
                "mesh_signal_register_failed",
                extra={"reason": "not in main thread"},
            )

    def _on_signal(self, signum, _frame):
        logger.info("mesh_signal_received", extra={"signum": signum})
        self._safe_close()
        # Re-raise default behaviour so the process exits.
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    def _safe_close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        if self.mesh is None:
            self._reset_fabric_safely()
            return

        t0 = time.time()
        timer_fired = threading.Event()

        def _watchdog():
            timer_fired.set()
            logger.warning(
                "mesh_close_timeout",
                extra={"elapsed_s": self.CLOSE_TIMEOUT_S},
            )

        watchdog = threading.Timer(self.CLOSE_TIMEOUT_S, _watchdog)
        watchdog.daemon = True
        watchdog.start()
        try:
            for submesh in self.mesh.get_submeshes():
                self._ttnn.close_mesh_device(submesh)
            self._ttnn.close_mesh_device(self.mesh)
        except Exception:
            logger.exception("mesh_close_error")
        finally:
            watchdog.cancel()

        self._reset_fabric_safely()
        self.mesh = None
        elapsed = time.time() - t0
        if not timer_fired.is_set():
            logger.info("mesh_close", extra={"shutdown_elapsed_s": elapsed})

    def _reset_fabric_safely(self):
        if not self._fabric_was_set:
            return
        try:
            self._ttnn.set_fabric_config(self._ttnn.FabricConfig.DISABLED)
        except Exception:
            logger.exception("mesh_fabric_reset_error")
        self._fabric_was_set = False

    def close(self):
        """Public alias for explicit shutdown."""
        self._safe_close()
