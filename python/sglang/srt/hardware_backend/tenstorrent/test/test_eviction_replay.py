# SPDX-License-Identifier: Apache-2.0
"""§9.11 Eviction-replay — R5 mitigation (HIGH) acceptance gate.

Methodology (spec §9.11):
  1. Issue prompt P1 = S + Q with shared system prompt S → capture output O1.
     S fills ~3 RadixCache pages (page_size=64); Q is a single-token arithmetic
     chain that produces exactly one deterministic completion token.
  2. Issue N=100 distinct pressure prompts (different prefix SYSID-BETA vs S's
     SYSID-ALPHA, different question type) to displace S from the LRU cache.
  3. Re-issue P1 → capture output O2.
  4. Assert O1 == O2 (deterministic regardless of eviction).

Key design invariants:
  - P1's question Q uses a math chain ("1+1=2. 2+2=4. 3+3=") with a
    single-token answer ("6"). This is maximally stable against model context
    contamination: the answer is fully determined by the arithmetic sequence.
  - Pressure prompts use SYSID-BETA prefix and a task-number suffix entirely
    different from P1. This rules out KV context contamination as a confound.
  - max_tokens=1 eliminates all multi-token decoding variance.

The test verifies R5 (§5.1): RadixAttention paged-evict consistency.
If S is evicted, the server must re-prefill it fresh; the deterministic
(temperature=0, max_tokens=1) decode should produce the same output.

Writes a mitigation note to _fixtures/r5_mitigation_note.md per spec.

Gated on SGLANG_PLATFORM=tenstorrent + live server :30000.
"""

import json
import os
import time
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + live sglang server on :30000",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]

_SERVER = "http://localhost:30000"
_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "_fixtures")
_MITIGATION_NOTE_PATH = os.path.join(_FIXTURES_DIR, "r5_mitigation_note.md")

# Shared system prompt S: ~175 tokens (fills ~2-3 RadixCache pages at page_size=64).
# Uses "SYSID-ALPHA" marker so it is entirely distinct from pressure prompts.
_SYSTEM_PROMPT_S = (
    "SYSID-ALPHA: Mathematics is the language of the universe. "
    "Numbers and equations describe natural phenomena with precision. "
) * 25

# Deterministic completion task Q: single-token answer "6".
# The arithmetic chain forces the model to complete with "6" regardless of context.
_P1_QUESTION = "1+1=2. 2+2=4. 3+3="

# Expected answer: exactly the token "6"
_P1_EXPECTED_TOKEN = "6"

# Number of eviction-pressure prompts.
# Each distinct prefix is ~175 tokens → ~3 pages.
# N=100 → 300 pages pressure, sufficient to displace S's pages in LRU ordering.
_N_EVICTION = 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def _completion(prompt: str, max_tokens: int = 1, timeout: int = 120) -> str:
    """POST /v1/completions at temperature=0; return generated text."""
    payload = {
        "model": "llama",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{_SERVER}/v1/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        assert r.status == 200, f"unexpected status {r.status}"
        return json.loads(r.read())["choices"][0]["text"]


def _flush_cache() -> None:
    """POST /flush_cache to reset RadixCache state."""
    req = urllib.request.Request(
        f"{_SERVER}/flush_cache",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=b"{}",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        pass  # 200 OK expected


def _build_distinct_prefix(idx: int, target_tokens: int = 175) -> str:
    """Build an eviction-pressure prefix that is distinct per idx.

    Uses SYSID-BETA prefix (completely different from S's SYSID-ALPHA) so that
    RadixCache never confuses pressure-prompt pages with S's pages.
    First 64 tokens differ between any two pressure prompts (via idx embedding).
    """
    seed = (
        f"SYSID-BETA-{idx:06d}: "
        "Computer science is the study of algorithms and data structures. "
        "Programming languages enable humans to instruct computational systems. "
    )
    repeats = max(1, target_tokens // len(seed.split()))
    return seed * repeats


# ── Tests ─────────────────────────────────────────────────────────────────────

@REQUIRES_TT
def test_eviction_replay_output_stable():
    """§9.11 R5: single-token output is stable after RadixCache eviction.

    The test uses max_tokens=1 with an arithmetic chain (1+1=2, 2+2=4, 3+3=)
    whose answer is uniquely "6". This minimises model-context variance and
    focuses the test on KV correctness after eviction + re-prefill.
    """
    # Flush to start with clean cache
    _flush_cache()
    time.sleep(0.2)

    p1_prompt = _SYSTEM_PROMPT_S + _P1_QUESTION

    # Step 1: Issue P1 cold — S is inserted into RadixCache
    print(f"\n  Issuing P1 (cold)...", flush=True)
    t0 = time.perf_counter()
    output_1 = _completion(p1_prompt, max_tokens=1)
    t1 = time.perf_counter()
    print(f"  P1 output (cold): {output_1!r}  [{t1-t0:.3f}s]", flush=True)

    assert output_1.strip() == _P1_EXPECTED_TOKEN, (
        f"P1 cold output {output_1!r} != expected {_P1_EXPECTED_TOKEN!r}. "
        f"Prompt design may be wrong for this model config."
    )

    # Step 2: Issue N=100 distinct pressure prompts with SYSID-BETA prefix.
    # Question suffix is "TASK-n:" (completely different from P1's math chain).
    print(f"\n  Issuing {_N_EVICTION} distinct eviction-pressure prompts...", flush=True)
    t_pressure_start = time.perf_counter()
    for i in range(_N_EVICTION):
        prefix = _build_distinct_prefix(i, target_tokens=175)
        question = f" TASK-{i}: "
        _completion(prefix + question, max_tokens=1, timeout=120)
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t_pressure_start
            print(
                f"  ... {i+1}/{_N_EVICTION} pressure prompts done "
                f"[{elapsed:.1f}s elapsed]",
                flush=True,
            )

    t_pressure_end = time.perf_counter()
    print(
        f"  Eviction pressure complete: {_N_EVICTION} prompts in "
        f"{t_pressure_end - t_pressure_start:.1f}s",
        flush=True,
    )

    # Step 3: Re-issue P1 — S may have been evicted; server must re-prefill
    print(f"\n  Issuing P1 again (may trigger re-prefill after eviction)...", flush=True)
    t2 = time.perf_counter()
    output_2 = _completion(p1_prompt, max_tokens=1)
    t3 = time.perf_counter()
    print(f"  P1 output (post-eviction): {output_2!r}  [{t3-t2:.3f}s]", flush=True)

    # Step 4: Assert output is stable
    print(f"\n  Output comparison:")
    print(f"    O1 = {output_1!r}")
    print(f"    O2 = {output_2!r}")
    print(f"    match = {output_1 == output_2}")

    if output_1 == output_2:
        print(f"  §9.11 R5 PASS: output is stable after eviction (O1 == O2 == {_P1_EXPECTED_TOKEN!r})")
    else:
        print(
            f"  §9.11 R5 FAIL: output changed after eviction\n"
            f"    O1={output_1!r}\n"
            f"    O2={output_2!r}\n"
            f"  This indicates KV state inconsistency after eviction/re-prefill."
        )

    assert output_1 == output_2, (
        f"§9.11 R5 FAIL: eviction-replay output changed.\n"
        f"  O1={output_1!r}\n"
        f"  O2={output_2!r}\n"
        f"  Expected: both == {_P1_EXPECTED_TOKEN!r} (arithmetic chain completion)."
    )


@REQUIRES_TT
def test_eviction_replay_write_r5_mitigation_note():
    """§9.11: write R5 mitigation note to _fixtures/r5_mitigation_note.md."""
    os.makedirs(_FIXTURES_DIR, exist_ok=True)

    note = """\
# R5 (HIGH severity, RadixAttention paged-evict consistency) — mitigation acceptance reduction

Original spec §5.1 promised three orthogonal verifications:
  1. §9.11 eviction-replay (end-to-end) — DONE in T3.7
  2. §9.b5 24h+ long-run KV pool free-list consistency — DEFERRED to P3
  3. §9.15 free-list invariant unit test — DROPPED (no TTPagedKVAdapter)

R5 now relies on §9.11 alone. Documented acceptance reduction.
If R5 reproduces in P3+ workloads, the spec should be re-amended.
"""

    with open(_MITIGATION_NOTE_PATH, "w") as f:
        f.write(note)

    print(f"\n  Wrote R5 mitigation note to: {_MITIGATION_NOTE_PATH}")
    assert os.path.exists(_MITIGATION_NOTE_PATH), (
        f"R5 mitigation note not written: {_MITIGATION_NOTE_PATH}"
    )
    print(f"  §9.11 R5 mitigation note written successfully")
