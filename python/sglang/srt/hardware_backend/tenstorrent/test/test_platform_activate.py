"""Platform activation tests for TTSRTPlatform.

Verifies spec §7.1 row 1: when ttnn is not importable (CPU-only host),
activate_tt_platform() returns None so SGLang falls back to the base
SRTPlatform / next plugin without raising.
"""
import builtins
import sys
from unittest.mock import patch


def test_activate_returns_none_when_ttnn_import_fails():
    """If `import ttnn` raises ImportError, activate must return None
    (not raise), so the OOT plugin entry point degrades gracefully on
    non-TT hosts."""
    # First, ensure the platform module is freshly imported so the
    # patch hits its in-function `import ttnn`.
    sys.modules.pop("sglang.srt.hardware_backend.tenstorrent.platform", None)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ttnn" or name.startswith("ttnn."):
            raise ImportError("simulated: ttnn not available")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        from sglang.srt.hardware_backend.tenstorrent.platform import (
            activate_tt_platform,
        )
        assert activate_tt_platform() is None


def test_activate_returns_name_when_ttnn_present():
    """On a host where ttnn IS importable, activate returns the platform
    name string so SGLang registers the OOT platform. This is the
    happy-path counterpart to the failure test above.

    NOTE: This test runs only inside the tt-metal venv where ttnn is
    actually importable. Skip otherwise so CI on non-TT hosts isn't
    flaky.
    """
    try:
        import ttnn  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("ttnn not available on this host")

    from sglang.srt.hardware_backend.tenstorrent.platform import (
        activate_tt_platform,
    )

    result = activate_tt_platform()
    # The function returns a non-empty string (the platform name) on
    # success. We don't pin the exact value here — it's a private
    # implementation detail.
    assert result is not None
    assert isinstance(result, str) and len(result) > 0
