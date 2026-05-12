"""TTSRTPlatform skeleton + activate_tt_platform entry point.

See docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md.
"""

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


def activate_tt_platform() -> str | None:
    """Entry-point activation function for sglang.srt.platforms.

    Returns the fully-qualified class string when ttnn is importable
    (we're inside the tt-metal docker image); returns None on hosts
    without ttnn so SGLang falls back to the default platform.
    """
    try:
        import ttnn  # noqa: F401
    except ImportError:
        return None
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

    def apply_server_args_defaults(self, server_args):
        # P1 hard constraints
        server_args.max_running_requests = 1
        server_args.chunked_prefill_size = -1  # -1 disables chunked prefill
        server_args.disable_radix_cache = True

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

        # User-facing device identity. Note: scheduler.init_overlap calls
        # torch.get_device_module(self.device), and self.device flows from
        # model_runner.device — NOT from this string. TTModelRunner.__init__
        # overrides model_runner.device = "cpu" so torch's device-module
        # lookup works while we keep this user-facing label.
        server_args.device = "tenstorrent"

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

    def get_mha_kv_pool_cls(self):
        # TTModelRunner (Phase D) constructs _DummyKVCache directly;
        # this factory should never be invoked. Raise loudly if it is.
        raise NotImplementedError(
            "TT backend constructs _DummyKVCache inside "
            "TTModelRunner.initialize(); the factory should not be called."
        )

    def get_mla_kv_pool_cls(self):
        raise NotImplementedError("MLA not in P1")

    def get_nsa_kv_pool_cls(self):
        raise NotImplementedError("NSA not in P1")

    def get_graph_runner_cls(self) -> type:
        # P1 sets support_cuda_graph()=False so the framework will not
        # call this. If it does, that's a bug — fail loudly.
        raise NotImplementedError("P1 does not use graph runners")


class MeshDeviceCtx:
    """Per-Scheduler-subprocess mesh-device lifecycle holder.

    Real implementation lands in Phase E.2 (ttnn.open_mesh_device + signal
    handlers + program cache config + 5s close timeout). For Phase A this
    is a placeholder so the rest of the module can reference the name.
    """
    pass
