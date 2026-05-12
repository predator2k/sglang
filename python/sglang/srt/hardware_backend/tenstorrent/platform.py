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
        # Real implementation lands in Phase E.1 — full spec §6.1 block.
        raise NotImplementedError(
            "apply_server_args_defaults is implemented in Phase E.1"
        )

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
