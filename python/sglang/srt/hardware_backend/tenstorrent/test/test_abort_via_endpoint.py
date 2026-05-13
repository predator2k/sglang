# SPDX-License-Identifier: Apache-2.0
"""§9.6 abort flow — POST /abort_request endpoint.

Verifies that:
  1. A streaming request can be aborted mid-stream via POST /abort_request.
  2. The final streaming chunk carries ``finish_reason == "abort"``.
  3. GET /get_load shows zero in-flight tokens after the abort (KV reclaimed).

Requires:
  - SGLANG_PLATFORM=tenstorrent
  - Live SGLang server on :30000
"""
import os
import threading
import time

import pytest
import requests

BASE_URL = "http://localhost:30000"
MODEL = "/models/Llama-3.1-8B-Instruct"
LONG_PROMPT = (
    "Write a very detailed 500-word essay about the history of computing "
    "from the 1940s to the present day:"
)

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
def test_abort_via_endpoint_reclaims_kv():
    """Submit a long streaming request, abort it via /abort_request, verify KV
    is reclaimed."""
    # Baseline: no tokens in flight
    baseline_tokens = _get_num_tokens()
    assert baseline_tokens == 0, f"Server not idle at test start: {baseline_tokens} tokens"

    rid_box: list[str | None] = [None]
    finish_reason_box: list[str | None] = [None]
    stream_started = threading.Event()

    def _stream():
        payload = {
            "model": MODEL,
            "prompt": LONG_PROMPT,
            "max_tokens": 400,
            "temperature": 0,
            "stream": True,
        }
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
                import json as _json

                chunk = _json.loads(line[6:])
                if rid_box[0] is None:
                    rid_box[0] = chunk.get("id")
                    stream_started.set()
                fr = chunk.get("choices", [{}])[0].get("finish_reason")
                if fr is not None:
                    finish_reason_box[0] = fr
                    break  # got terminal chunk

    stream_thread = threading.Thread(target=_stream, daemon=True)
    stream_thread.start()

    # Wait until the first chunk arrives (RID known)
    assert stream_started.wait(timeout=60), "Stream never started within 60s"
    time.sleep(0.5)  # let a few more tokens flow to ensure decode is active

    # Abort the request
    rid = rid_box[0]
    assert rid is not None
    abort_resp = requests.post(
        f"{BASE_URL}/abort_request", json={"rid": rid}, timeout=10
    )
    assert abort_resp.status_code == 200, f"Abort failed: {abort_resp.status_code}"

    stream_thread.join(timeout=30)
    assert not stream_thread.is_alive(), "Stream thread did not terminate after abort"

    # The final streaming chunk must carry finish_reason == "abort"
    assert finish_reason_box[0] == "abort", (
        f"Expected finish_reason='abort', got {finish_reason_box[0]!r}"
    )

    # KV pool must be fully reclaimed
    time.sleep(1)  # allow scheduler one pass to free pages
    tokens_after = _get_num_tokens()
    assert tokens_after == 0, (
        f"KV not reclaimed after abort: {tokens_after} tokens still in flight"
    )
