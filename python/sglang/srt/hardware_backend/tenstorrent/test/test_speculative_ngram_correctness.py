# SPDX-License-Identifier: Apache-2.0
"""P3a §10.3b NGRAM bit-exact correctness vs baseline.

Speculative decoding MUST be a perf optimization, never a quality regression.
At temperature=0, NGRAM output must token-for-token equal non-speculative.

Methodology:
  1. Operator runs ``test_ngram_outputs_recorded`` against the NGRAM server,
     captured to ``_fixtures/ngram_correctness_ngram.json``.
  2. Operator relaunches the server WITHOUT ``--speculative-algorithm`` and
     re-runs with ``NGRAM_CORRECTNESS_MODE=baseline``, captured to
     ``_fixtures/ngram_correctness_baseline.json``.
  3. ``test_ngram_matches_baseline`` then diffs the two recordings.
"""
import json
import os
import pathlib
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + sglang server",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]

PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "The first three prime numbers are",
    "Albert Einstein was born in",
    "The chemical symbol for gold is",
    "The speed of light in vacuum is",
    "Photosynthesis is the process by which",
    "The largest planet in our solar system is",
    "World War II ended in",
    "The author of 1984 is",
]


def _complete(prompt: str) -> str:
    payload = {
        "model": "llama",
        "prompt": prompt,
        "max_tokens": 30,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        "http://localhost:30000/v1/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["text"]


@REQUIRES_TT
def test_ngram_outputs_recorded():
    """Record outputs from currently-running server (ngram or baseline)."""
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    fixture_dir.mkdir(exist_ok=True)
    mode = os.environ.get("NGRAM_CORRECTNESS_MODE", "ngram")
    out_file = fixture_dir / f"ngram_correctness_{mode}.json"
    results = {p: _complete(p) for p in PROMPTS}
    out_file.write_text(json.dumps(results, indent=2))
    assert len(results) == 10


def test_ngram_matches_baseline():
    """Diff the two recordings. Skipped if either is missing."""
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    ngram_f = fixture_dir / "ngram_correctness_ngram.json"
    baseline_f = fixture_dir / "ngram_correctness_baseline.json"
    if not (ngram_f.exists() and baseline_f.exists()):
        pytest.skip("Both recordings missing; operator must run both")

    ngram = json.loads(ngram_f.read_text())
    baseline = json.loads(baseline_f.read_text())

    mismatches = []
    for prompt, ngram_out in ngram.items():
        baseline_out = baseline.get(prompt)
        if baseline_out != ngram_out:
            mismatches.append((prompt, ngram_out, baseline_out))

    assert not mismatches, (
        f"NGRAM bit-exactness violated for {len(mismatches)} prompts: "
        f"{mismatches[:3]}"
    )
