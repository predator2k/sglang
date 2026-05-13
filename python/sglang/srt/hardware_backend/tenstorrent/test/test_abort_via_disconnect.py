# SPDX-License-Identifier: Apache-2.0
"""§9.6 abort flow — client TCP disconnect mid-stream.

Verifies that:
  1. Closing the HTTP connection mid-stream triggers server-side abort.
  2. GET /get_load shows zero in-flight tokens after disconnect + settle time
     (KV pages reclaimed).

Uses ``requests`` with ``stream=True`` and an early ``resp.close()`` to
simulate a mid-stream TCP disconnect.

Requires:
  - SGLANG_PLATFORM=tenstorrent
  - Live SGLang server on :30000
"""
import os
import time

import pytest
import requests

BASE_URL = "http://localhost:30000"
MODEL = "/models/Llama-3.1-8B-Instruct"
LONG_PROMPT = (
    "Write a very detailed 500-word essay about the history of computing "
    "from the 1940s to the present day:"
)
# Seconds to wait after disconnect for server to detect and reclaim KV
SETTLE_SECS = 5

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + live sglang server",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


def _get_num_tokens() -> int:
    resp = requests.get(f"{BASE_URL}/get_load", timeout=10)
    resp.raise_for_status()
    return resp.json()[0]["num_tokens"]


@REQUIRES_TT
def test_abort_via_disconnect_reclaims_kv():
    """Open a streaming request, close the socket after a few tokens, then
    confirm the server reclaims KV pages within SETTLE_SECS seconds."""
    # Baseline: no tokens in flight
    baseline_tokens = _get_num_tokens()
    assert baseline_tokens == 0, f"Server not idle at test start: {baseline_tokens} tokens"

    payload = {
        "model": MODEL,
        "prompt": LONG_PROMPT,
        "max_tokens": 400,
        "temperature": 0,
        "stream": True,
    }

    tokens_seen = 0
    tokens_in_flight_at_close = 0

    with requests.post(
        f"{BASE_URL}/v1/completions", json=payload, stream=True, timeout=120
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            tokens_seen += 1
            if tokens_seen >= 5:
                # Capture in-flight token count just before disconnect
                tokens_in_flight_at_close = _get_num_tokens()
                # Close the connection — simulates TCP disconnect
                resp.close()
                break
        # Context manager __exit__ ensures underlying socket is closed

    assert tokens_seen >= 5, "Stream ended before 5 tokens; prompt may be too short"
    assert tokens_in_flight_at_close > 0, (
        "Expected non-zero tokens in flight at disconnect moment; "
        f"got {tokens_in_flight_at_close}"
    )

    # Wait for the server to detect the disconnect and reclaim KV pages
    deadline = time.monotonic() + SETTLE_SECS
    tokens_after = tokens_in_flight_at_close
    while time.monotonic() < deadline:
        tokens_after = _get_num_tokens()
        if tokens_after == 0:
            break
        time.sleep(0.5)

    assert tokens_after == 0, (
        f"KV not reclaimed {SETTLE_SECS}s after disconnect: "
        f"{tokens_after} tokens still in flight"
    )
