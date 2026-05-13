# SPDX-License-Identifier: Apache-2.0
"""§9.b1 per-model smoke for paged path (G1b).

Per T4.1, scope reduced to Qwen3-8B (Llama already validated in P2a).
Mistral + GptOss are PLACEHOLDER (weights unavailable, deferred to P3).
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
def test_qwen3_smoke():
    """Qwen3-8B smoke — assumes server is launched with --model-path Qwen3-8B."""
    # Qwen3-8B is a thinking model; it emits <think>...</think> before answering.
    # 200 tokens is sufficient for the reasoning trace + the final "4".
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 200,
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
    assert len(content) > 0, "empty response"
    # Mild assertion: response contains "4" somewhere (since prompt asks for 2+2)
    assert "4" in content, f"expected '4' in response, got {content!r}"


@pytest.mark.skip(reason="PLACEHOLDER — Mistral weights unavailable, deferred to P3")
def test_mistral_smoke():
    """Mistral-7B smoke — PLACEHOLDER, deferred to P3."""
    pass


@pytest.mark.skip(reason="PLACEHOLDER — GptOss weights unavailable, deferred to P3")
def test_gptoss_smoke():
    """GptOss smoke — PLACEHOLDER, deferred to P3."""
    pass
