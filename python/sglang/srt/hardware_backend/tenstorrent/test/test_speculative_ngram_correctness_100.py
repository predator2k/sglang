# SPDX-License-Identifier: Apache-2.0
"""P3a §10.3c NGRAM 100-prompt strict bit-exact correctness vs baseline.

Methodology: 100 diverse prompts, temperature=0, max_tokens=30. Operator runs
the recording phase twice (NGRAM server, then baseline server) and the
comparison test asserts 100/100 byte-exact equality.
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
    # 10 factual openers (overlap with the 10-prompt suite for cross-check)
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
    # 30 factual continuations
    "Mount Everest is located in",
    "The Pacific Ocean is the",
    "Shakespeare wrote a play called",
    "The Great Wall of China was built to",
    "Mozart was born in the year",
    "The chemical formula for water is",
    "The Sahara Desert is in",
    "The Eiffel Tower is in",
    "The Mona Lisa was painted by",
    "The Roman Empire fell in",
    "The Earth orbits around",
    "The smallest country in the world is",
    "Penguins live in",
    "The Amazon River flows through",
    "The Statue of Liberty was a gift from",
    "Marie Curie discovered",
    "The Egyptian pyramids were built as",
    "The Beatles were a band from",
    "Mount Fuji is in",
    "The Renaissance began in",
    "The first human to walk on the moon was",
    "Antarctica is the",
    "The longest river in the world is",
    "DNA stands for",
    "The capital of Japan is",
    "Pi to two decimal places is",
    "The boiling point of water at sea level is",
    "Beethoven composed",
    "The Big Bang theory describes",
    "Penicillin was discovered by",
    # 20 question forms
    "What is the largest mammal?",
    "Who wrote Romeo and Juliet?",
    "When did the French Revolution begin?",
    "Where is the Colosseum?",
    "Why do leaves change color in autumn?",
    "How does a steam engine work?",
    "Which planet is known as the Red Planet?",
    "What does HTML stand for?",
    "Who painted the Sistine Chapel?",
    "When was the telephone invented?",
    "Where do polar bears live?",
    "Why is the sky blue?",
    "How tall is Mount Kilimanjaro?",
    "Which element has the symbol O?",
    "What is the longest bone in the human body?",
    "Who discovered America?",
    "When did the Berlin Wall fall?",
    "Where is the Great Barrier Reef?",
    "Why is recycling important?",
    "How many continents are there?",
    # 20 short prefix completions
    "Roses are red,",
    "To be or not to be,",
    "Once upon a time,",
    "In the beginning,",
    "The quick brown fox",
    "It was the best of times,",
    "All that glitters",
    "A journey of a thousand miles",
    "An apple a day",
    "Time flies",
    "Knowledge is",
    "Practice makes",
    "Honesty is",
    "Better late",
    "When in Rome,",
    "Actions speak",
    "Beauty is in",
    "Curiosity killed",
    "Don't judge a book",
    "Every cloud has",
    # 20 technical / programming
    "A binary tree is",
    "Recursion in programming is",
    "Object-oriented programming emphasizes",
    "A hash map provides",
    "Big O notation describes",
    "TCP/IP is",
    "A relational database stores",
    "Git is used for",
    "Machine learning is",
    "A neural network is",
    "Quantum computing uses",
    "Cloud computing offers",
    "Encryption is the process of",
    "An algorithm is",
    "A function in programming is",
    "Variables in programming are",
    "An array is a data structure that",
    "A loop is used to",
    "Compilers translate",
    "An IP address identifies",
]
assert len(PROMPTS) == 100, f"Need exactly 100 prompts, have {len(PROMPTS)}"


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
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["text"]


@REQUIRES_TT
def test_ngram_outputs_recorded_100():
    """Record outputs from the currently-running server (ngram or baseline)."""
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    fixture_dir.mkdir(exist_ok=True)
    mode = os.environ.get("NGRAM_CORRECTNESS_MODE", "ngram")
    out_file = fixture_dir / f"ngram_correctness_100_{mode}.json"
    results = {p: _complete(p) for p in PROMPTS}
    out_file.write_text(json.dumps(results, indent=2))
    assert len(results) == 100


def test_ngram_matches_baseline_100():
    """100-prompt strict 100% byte-exact gate."""
    fixture_dir = pathlib.Path(__file__).parent / "_fixtures"
    ngram_f = fixture_dir / "ngram_correctness_100_ngram.json"
    baseline_f = fixture_dir / "ngram_correctness_100_baseline.json"
    if not (ngram_f.exists() and baseline_f.exists()):
        pytest.skip("Both 100-prompt recordings missing; operator must run both")

    ngram = json.loads(ngram_f.read_text())
    baseline = json.loads(baseline_f.read_text())

    mismatches = []
    for prompt, ngram_out in ngram.items():
        baseline_out = baseline.get(prompt)
        if baseline_out != ngram_out:
            mismatches.append((prompt, ngram_out, baseline_out))

    assert not mismatches, (
        f"NGRAM byte-exactness violated for {len(mismatches)}/100 prompts. "
        f"First 3: {mismatches[:3]}"
    )
