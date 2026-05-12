"""Spec §8.1 smoke test — verifies a live SGLang server on Tenstorrent
returns coherent generations containing the factual answer "Paris" for a
"capital of France" prompt.

Marked @requires_tt (skipped unless SGLANG_PLATFORM=tenstorrent and an
sglang server is reachable on localhost:30000). The CPU CI run skips
this file entirely; hardware runs gate on it.

NOTE on prompt format: the spec's verbatim example "The capital of France
is" elicits a descriptive continuation from Llama-3.1-8B-Instruct ("a
city that is steeped in history and culture") that doesn't contain the
literal word "Paris". The instruct tuning prefers continuation over
factual single-word recall in completion-style prompts. The chat-format
variant reliably returns "Paris." with finish_reason=stop — exercises
the same backend code path, simpler assertion.
"""

import json
import os
import urllib.error
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware and a live sglang server on :30000",
)


def _post(path: str, payload: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        f"http://localhost:30000{path}",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            assert r.status == 200, r.status
            return json.loads(r.read())
    except urllib.error.URLError as e:
        pytest.fail(f"sglang server not reachable on :30000 ({e}); start it first")


@REQUIRES_TT
def test_paris_via_chat_format():
    """Chat-formatted prompt to /v1/chat/completions. Llama-3.1-8B-Instruct
    answers a direct factual question concisely; we expect "Paris" in the
    assistant message content."""
    resp = _post(
        "/v1/chat/completions",
        {
            "model": "llama",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one word.",
                }
            ],
            "max_tokens": 10,
            "temperature": 0,
        },
    )
    content = resp["choices"][0]["message"]["content"]
    assert "Paris" in content, f"expected 'Paris' in response, got {content!r}"


@REQUIRES_TT
def test_paris_via_qa_completion():
    """Q/A continuation prompt to /v1/completions. The model's first
    response token after 'A:' should be ' Paris'."""
    resp = _post(
        "/v1/completions",
        {
            "model": "llama",
            "prompt": "Q: What is the capital of France?\nA:",
            "max_tokens": 5,
            "temperature": 0,
        },
    )
    text = resp["choices"][0]["text"]
    assert "Paris" in text, f"expected 'Paris' in completion, got {text!r}"


@REQUIRES_TT
def test_spec_verbatim_prompt_returns_coherent_text():
    """Spec §8.1 uses 'The capital of France is' as the example prompt.
    Llama-3.1-8B-Instruct continues this descriptively rather than naming
    the city. We only assert HTTP 200 + non-empty coherent English text;
    the Paris check is delegated to the two tests above which use a
    prompt format that elicits the factual answer.
    """
    resp = _post(
        "/v1/completions",
        {
            "model": "llama",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0,
        },
    )
    text = resp["choices"][0]["text"].strip()
    assert len(text) > 0
    # English ASCII letters present somewhere — sanity check that the
    # output isn't gibberish / -1 sentinels / nan.
    assert any(ch.isalpha() for ch in text), text
