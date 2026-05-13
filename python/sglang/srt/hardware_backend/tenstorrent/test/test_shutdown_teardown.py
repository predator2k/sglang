# SPDX-License-Identifier: Apache-2.0
"""§9.13 TTNN teardown leak check — partial verification (Option A).

Strategy (Option A, as per task spec):
  - Confirm that ttnn.GetNumAvailableDevices() > 0 while the server is running,
    proving the device is actively open (mesh is live with hardware tensors).
  - Confirm the server is serving requests normally (not in a degraded state).
  - Document that full teardown verification (devices released after shutdown)
    is handled by the container lifecycle: `podman stop p2a-smoke` triggers
    SIGTERM → MeshDeviceCtx._on_signal() → _safe_close() → close_mesh_device().

Note: ttnn.get_num_tensors() does NOT exist in the installed ttnn build.
The task's Option A phrasing uses it as a placeholder for "any device-level
check that proves hardware is actively open". We use GetNumAvailableDevices()
instead — it returns the count of TT devices accessible to the process, which
is > 0 iff the driver is open (i.e., the mesh is live).

Full teardown verification deferred: container lifecycle test (P3) should:
  1. Start the server.
  2. Run a request (proves mesh is open).
  3. Stop the container (podman stop).
  4. Verify process exits cleanly (exit code 0 from SIGTERM handler).
  5. Confirm no lingering /dev/tenstorrent FD leaks after container exit.

Gated on SGLANG_PLATFORM=tenstorrent + live server :30000.
"""

import json
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + live sglang server on :30000",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]

_SERVER = "http://localhost:30000"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _server_alive() -> bool:
    """Return True if the server responds to /v1/models."""
    try:
        with urllib.request.urlopen(f"{_SERVER}/v1/models", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def _server_generate(prompt: str, max_tokens: int = 5) -> dict:
    """Issue a quick /v1/completions call and return the response."""
    payload = {
        "model": "llama",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{_SERVER}/v1/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        assert r.status == 200
        return json.loads(r.read())


# ── Tests ─────────────────────────────────────────────────────────────────────

@REQUIRES_TT
def test_tt_device_fds_held_while_server_running():
    """§9.13 (partial A): /dev/tenstorrent device FDs are held while server is up.

    Proves that TT hardware is actively held open by the server process. Checks
    /proc for open FDs on /dev/tenstorrent* — if the mesh is open, these exist.

    NOTE: ttnn.get_num_tensors() is not in the installed ttnn build. Using
    /proc/*/fd scanning instead — avoids opening a second ttnn handle (which
    would block on the mutex held by the server process).

    ttnn.GetNumAvailableDevices() is intentionally NOT used here because it
    calls the device driver, triggering a mutex wait for the server's lock.
    """
    import glob
    import subprocess

    print()

    # 1. Check that /dev/tenstorrent* devices exist on this host
    tt_devs = glob.glob("/dev/tenstorrent*")
    print(f"  /dev/tenstorrent* devices: {tt_devs}")
    assert tt_devs, (
        "No /dev/tenstorrent* devices found. "
        "Is this running on a TT hardware host?"
    )

    # 2. Find processes holding FDs to /dev/tenstorrent (lsof -t)
    # -t = terse (only PIDs), +D = recursive under path.
    # Use /proc scan instead of lsof for container portability.
    holding_pids = []
    for fd_link in glob.glob("/proc/*/fd/*"):
        try:
            target = os.readlink(fd_link)
            if "/dev/tenstorrent" in target:
                pid_str = fd_link.split("/")[2]
                if pid_str.isdigit():
                    holding_pids.append(int(pid_str))
        except (OSError, ValueError):
            continue

    holding_pids = list(set(holding_pids))
    print(f"  PIDs holding /dev/tenstorrent FDs: {holding_pids}")

    assert len(holding_pids) > 0, (
        f"No processes hold /dev/tenstorrent FDs — mesh may not be open. "
        f"Devices found: {tt_devs}"
    )
    print(f"  §9.13 PASS: {len(holding_pids)} process(es) hold device FDs "
          f"(mesh is live while server is running)")
    print(
        f"  NOTE: Full teardown verification (FDs released after shutdown) "
        f"deferred to container lifecycle test (P3). See docstring."
    )


@REQUIRES_TT
def test_server_serves_requests_while_mesh_held():
    """§9.13 (partial A): server can complete a generation while mesh is open.

    Confirms the mesh is not only open but actively functional — device tensors
    are allocated, prefill/decode pipeline runs, output is produced.
    """
    assert _server_alive(), "Server not responding to /v1/models"

    resp = _server_generate("Hello, world!", max_tokens=3)
    text = resp["choices"][0]["text"]
    completion_tokens = resp["usage"]["completion_tokens"]

    print()
    print(f"  Generated: {text!r} ({completion_tokens} tokens)")
    assert completion_tokens > 0, (
        f"Expected >0 completion tokens; got {completion_tokens}. "
        f"Server may be in a degraded state."
    )
    print(f"  §9.13 PASS: mesh is live and serving requests correctly")


@REQUIRES_TT
def test_mesh_close_handler_registered_in_platform_source():
    """§9.13 (static): MeshDeviceCtx registers atexit + SIGTERM close handlers.

    Reads platform.py source to verify the teardown handlers are registered
    (atexit.register + signal.signal SIGTERM). This is the mechanism that
    ensures clean mesh close on container shutdown.
    """
    platform_path = os.path.join(
        os.path.dirname(__file__),
        "..", "platform.py",
    )
    with open(platform_path) as f:
        source = f.read()

    print()

    # atexit handler
    assert "atexit.register(self._safe_close)" in source, (
        "MeshDeviceCtx does not register atexit handler for _safe_close — "
        "shutdown may leak device FDs."
    )
    print("  atexit.register(_safe_close) confirmed in MeshDeviceCtx")

    # SIGTERM handler
    assert "signal.SIGTERM" in source, (
        "MeshDeviceCtx does not register SIGTERM handler — "
        "podman stop (SIGTERM) will not trigger mesh close."
    )
    print("  signal.SIGTERM handler confirmed in MeshDeviceCtx")

    # close_mesh_device is called in _safe_close
    assert "self._ttnn.close_mesh_device(self.mesh)" in source, (
        "MeshDeviceCtx._safe_close does not call ttnn.close_mesh_device — "
        "mesh may not be released on shutdown."
    )
    print("  ttnn.close_mesh_device(self.mesh) call confirmed in _safe_close")

    print(
        f"  §9.13 PASS: teardown handler chain confirmed: "
        f"SIGTERM/atexit → _safe_close → close_mesh_device"
    )
    print(
        f"  Full teardown verification (no FD leaks after container stop) "
        f"deferred to container lifecycle test (P3)."
    )
