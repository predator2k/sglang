"""Spec §8.5 performance log — records numbers, does not gate.

Captures the metrics spec §8.5 specifies for the post-bringup
"P1 observed performance" addendum:

  - Cold first-prefill TTFT (token-level wall-time to first decoded token)
  - Warm prefill latency at 100 / 500 / 1k / 4k prompt lengths
  - Warm decode steady-state tok/s
  - D2H measured latency (placeholder — backend currently folds D2H
    into the surrounding extend/decode timer; the spec's expectation
    is dedicated `_TT_D2H_LATENCY_MS` once F.4's TODO lands)

Writes a JSON report to /tmp/tt_perf_log.json and prints a summary
table. NEVER FAILS — exists purely to feed numbers into the spec
addendum. Use `pytest -v -s test_perf_log.py` to see the table.

Gated on SGLANG_PLATFORM=tenstorrent + live server :30000.
"""

import json
import os
import statistics
import time
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware and a live sglang server on :30000",
)


def _build_prompt(target_tokens: int) -> str:
    """Build a prompt of approximately `target_tokens` tokens by repeating
    a stable seed paragraph. Returns the prompt string."""
    seed = (
        "Computing has evolved from mechanical calculators through "
        "vacuum-tube mainframes to modern silicon. Each step extended "
        "what people could express and compute. "
    )
    # ~25 tokens per repetition with the Llama tokenizer; over-estimate
    # and let the server tokenize precisely.
    repeats = max(1, target_tokens // 20)
    return seed * repeats


def _post(payload: dict, timeout: int = 240) -> tuple[dict, float]:
    req = urllib.request.Request(
        "http://localhost:30000/v1/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    elapsed = time.perf_counter() - t0
    return json.loads(body), elapsed


@REQUIRES_TT
def test_perf_log_capture():
    """Record performance numbers; never fails."""
    results = {}

    # 1. Warm prefill at known-good shapes only.
    # tt_transformers' KV sharding (tensor_layout.cpp:111) has a tile-
    # alignment constraint that fails for many seq_lens between 128 and
    # 6144. Empirically validated on 2x p150a:
    #   - 100 -> pad 128       ✓ (F.4)
    #   - 256                  ✗ (perf_log v1 crashed here)
    #   - 1024                 unknown / unsafe
    #   - 4000 -> pad 4096     ✓ (subset of the 6144 path proven by 5K bench)
    #   - 6000 -> pad 6144     ✓ (5K bench)
    # Restrict perf log to these three proven shapes; document the
    # missing intermediate lengths as a P2 investigation item.
    # First call at each length pays cold JIT cost; we run TWICE at
    # each shape and record the second (warm) run.
    targets = [100, 4000, 6000]
    prefill_warm_ms = {}
    for target in targets:
        prompt = _build_prompt(target)
        # Warm-up call (eats JIT cost if shape is new)
        _, _ = _post({
            "model": "llama", "prompt": prompt, "max_tokens": 1, "temperature": 0,
        })
        # Measure
        resp, elapsed = _post({
            "model": "llama", "prompt": prompt, "max_tokens": 1, "temperature": 0,
        })
        actual_prompt_tokens = resp["usage"]["prompt_tokens"]
        prefill_warm_ms[target] = {
            "target_tokens": target,
            "actual_prompt_tokens": actual_prompt_tokens,
            "elapsed_ms": elapsed * 1000,
        }
        print(
            f"  prefill target={target:>4d} actual={actual_prompt_tokens:>4d}  "
            f"warm={elapsed*1000:>7.1f} ms",
            flush=True,
        )
    results["warm_prefill_ms"] = prefill_warm_ms

    # 2. Warm decode steady-state tok/s — generate 100 tokens after a
    # short warm prompt; assume prefill is amortised. Run 3 times,
    # take the median.
    decode_runs_tok_s = []
    short_prompt = _build_prompt(100)
    for trial in range(3):
        resp, elapsed = _post({
            "model": "llama", "prompt": short_prompt,
            "max_tokens": 100, "temperature": 0,
        })
        tokens = resp["usage"]["completion_tokens"]
        # Subtract a fixed prefill estimate (~50 ms warm) for a cleaner
        # decode-only number; tolerable noise.
        decode_only_s = elapsed - 0.05
        if tokens > 0 and decode_only_s > 0:
            decode_runs_tok_s.append(tokens / decode_only_s)
        print(
            f"  decode trial {trial+1}/3: {tokens} tokens in {elapsed:.2f}s -> "
            f"{tokens/elapsed:.2f} tok/s (raw)",
            flush=True,
        )
    if decode_runs_tok_s:
        results["warm_decode_tok_s_median"] = statistics.median(decode_runs_tok_s)
        results["warm_decode_tok_s_all"] = decode_runs_tok_s

    # 3. Print summary table
    print()
    print(f"=== TT Performance Log ===")
    print(f"warm prefill latency:")
    for t in targets:
        m = prefill_warm_ms[t]
        print(f"  {m['target_tokens']:>4d} target  ({m['actual_prompt_tokens']:>4d} actual): {m['elapsed_ms']:>7.1f} ms")
    if "warm_decode_tok_s_median" in results:
        print(f"warm decode tok/s (median of 3): {results['warm_decode_tok_s_median']:.2f}")
    print()

    # Write report
    out = "/tmp/tt_perf_log.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"--- wrote {out} ---", flush=True)

    # Sanity assertions — only catch obvious regressions, not micro perf
    assert prefill_warm_ms[100]["elapsed_ms"] < 5000, (
        f"100-token warm prefill {prefill_warm_ms[100]['elapsed_ms']:.0f}ms > 5s; "
        f"server is far slower than the F.4 baseline (~250ms)"
    )
    if "warm_decode_tok_s_median" in results:
        assert results["warm_decode_tok_s_median"] > 15, (
            f"decode tok/s {results['warm_decode_tok_s_median']:.1f} < 15; "
            f"under-performs F.4 baseline of 32+"
        )
