# SPDX-License-Identifier: Apache-2.0
"""§9.1 paged smoke — plugin path returns 'Paris' for capital-of-France prompt.

Requires:
  - Built tt-metal image `local-tt-metal:dev` (sha256:973e972bddf5)
  - Live SGLang server on :30000 with SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged
"""
import json
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + live sglang server",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]


@REQUIRES_TT
def test_paged_smoke_paris():
    payload = {
        "model": "llama",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
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
    assert "Paris" in content, f"expected 'Paris', got {content!r}"
