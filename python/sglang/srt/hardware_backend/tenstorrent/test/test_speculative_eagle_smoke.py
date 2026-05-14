# SPDX-License-Identifier: Apache-2.0
"""P3a §10.5 EAGLE smoke test on Tenstorrent Blackhole (P3a.2 T2.2).

Operator launches the SGLang server with --speculative-algorithm EAGLE and
--speculative-draft-model-path before running this test. Cohost layout is
Layout A (shared single (1,2) MeshDevice) per P3a.0 T0.3.

Per T0.3, we use the Qwen3 pair locally (Qwen3-1.7B draft + Qwen3-8B main)
since Llama-3.2-1B isn't available in /models/. The smoke check tokenizes
naturally to either name set.
"""
import json
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server with EAGLE spec on :30000",
)
pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


@REQUIRES_TT
def test_eagle_smoke_paris():
    payload = {
        "model": "llama",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ],
        "max_tokens": 10,
        "temperature": 0,
    }
    req = urllib.request.Request(
        "http://localhost:30000/v1/chat/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        assert r.status == 200
        resp = json.loads(r.read())
    content = resp["choices"][0]["message"]["content"]
    assert "Paris" in content, f"Expected Paris in answer, got: {content!r}"
