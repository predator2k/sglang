# SPDX-License-Identifier: Apache-2.0
"""§9.5 queue-full admission gate — PARTIAL implementation.

Strategy: Option A (deferred fixture).

The running server at :30000 was launched with ``--max-running-requests 4``
and ``--max-queued-requests`` unset (null / unlimited queue).  Triggering a
*queue-full* abort therefore requires relaunching the server with a small
``--max-queued-requests`` value.  Adding that subprocess/podman fixture is
deferred to P2a.3.

This file holds the test skeletons with documented skip reasons so that the
test IDs are registered in the suite and CI can gate on them explicitly when
the fixture lands.

Requires:
  - SGLANG_PLATFORM=tenstorrent
  - Live SGLang server on :30000
"""
import os

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + live sglang server",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]

_SKIP_QUEUE_FULL = pytest.mark.skip(
    reason=(
        "§9.5 queue-full abort requires --max-queued-requests=2 server fixture; "
        "deferred to P2a.3.  Running server has max_queued_requests=null (unlimited)."
    )
)


@REQUIRES_TT
@_SKIP_QUEUE_FULL
def test_queue_full_returns_abort():
    """N+1 concurrent requests where N = max_queued_requests should yield at
    least one response whose ``finish_reason`` is ``"abort"`` and whose
    associated message mentions the queue limit.

    When the P2a.3 fixture lands, replace the skip marker with a
    ``server_with_small_queue`` fixture that launches a side server on port
    30001 with ``--max-queued-requests 2``, runs the body below, and tears
    down.

    Implementation sketch (left in place for P2a.3)::

        import concurrent.futures, requests, json

        MAX_QUEUED = 2
        BASE_URL = "http://localhost:30001"

        def submit(prompt):
            payload = {
                "model": "/models/Llama-3.1-8B-Instruct",
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0,
                "stream": False,
            }
            resp = requests.post(f"{BASE_URL}/v1/completions", json=payload, timeout=120)
            return resp.json()

        long_prompts = [f"Write a 300-word essay on topic {i}:" for i in range(MAX_QUEUED + 2)]
        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = [ex.submit(submit, p) for p in long_prompts]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]

        abort_responses = [
            r for r in responses
            if r.get("choices", [{}])[0].get("finish_reason") == "abort"
        ]
        assert abort_responses, "Expected at least one abort due to queue full"
    """
