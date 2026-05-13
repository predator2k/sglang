# SPDX-License-Identifier: Apache-2.0
"""P3a §10.3 NGRAM smoke test on Tenstorrent Blackhole.

Operator launches the SGLang server with --speculative-algorithm NGRAM before
running this test. See plan T1.3 step 2 for the launch recipe.
"""
import json
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server with NGRAM spec on :30000",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


@REQUIRES_TT
def test_ngram_smoke_paris():
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
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.status == 200
        resp = json.loads(r.read())
    content = resp["choices"][0]["message"]["content"]
    assert "Paris" in content, f"Expected Paris in answer, got: {content!r}"
