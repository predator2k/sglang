# SPDX-License-Identifier: Apache-2.0
"""§9.2 paged greedy correctness on Blackhole.

Verifies deterministic decode under temp=0 via plugin path (P2a.1).
Accepts BFP8 quantization drift vs HF reference; only requires that
the SAME prompt produces the SAME output across re-runs.

Bit-exact HF reference comparison deferred to P2a.3 perf-log task.
"""
import importlib.util
import json
import os
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + live sglang server",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "_fixtures")


def _load_prompts():
    """Load PROMPTS from the fixture module without triggering the full sglang import chain."""
    spec = importlib.util.spec_from_file_location(
        "llama31_prompts",
        os.path.join(_FIXTURES_DIR, "llama31_prompts.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPTS


def _completion(prompt: str, max_tokens: int = 20) -> str:
    """POST /v1/completions with temperature=0 and return the generated text."""
    req = urllib.request.Request(
        "http://localhost:30000/v1/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": "llama",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            }
        ).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        assert r.status == 200, f"unexpected status {r.status}"
        return json.loads(r.read())["choices"][0]["text"]


@REQUIRES_TT
def test_greedy_determinism_paged():
    """For each prompt: run twice sequentially, assert outputs match.

    This is the minimum for §9.2: deterministic greedy at temp=0.
    Three prompts are used (cap) to keep hardware time reasonable.
    """
    prompts = _load_prompts()
    failures = []
    for i, prompt in enumerate(prompts[:3]):
        r1 = _completion(prompt)
        r2 = _completion(prompt)
        if r1 != r2:
            failures.append(
                {
                    "prompt_index": i,
                    "run1": r1,
                    "run2": r2,
                }
            )
    assert not failures, f"Non-deterministic outputs for prompts: {failures}"
