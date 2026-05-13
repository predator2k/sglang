# SPDX-License-Identifier: Apache-2.0
"""§9.10 Paged-path performance log — tok/s, prefix cache, KV utilization.

Measurements captured (per spec §4.3 §9.10):
  1. Batched decode tok/s at B={1, 2, 4}
     - Sanity floor: B=4 tok/s >= 1.2× B=1 tok/s
  2. Prefix hit/miss latency (2 cold + 2 warm requests)
     - Sanity floor: warm lookup latency delta < 10ms (relative to cold)
  3. KV pool utilization (3 sample points from /v1/loads)
     - Sanity floor: max_util < 95%
  4. Per-mode timing from /v1/loads gen_throughput (informational)

Design philosophy: test PASSES if measurements are captured. Floors that
fail are logged but do not cause the test to fail — the test separates
"did we measure?" (hard assertion) from "did floors meet?" (soft warning).

Results are written to:
  - /tmp/tt_perf_log_paged.json  (machine-readable)
  - python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/perf_log.json

Gated on SGLANG_PLATFORM=tenstorrent + live server :30000 (paged backend).
"""

import json
import os
import statistics
import threading
import time
import urllib.request

import pytest

REQUIRES_TT = pytest.mark.skipif(
    os.environ.get("SGLANG_PLATFORM") != "tenstorrent",
    reason="Requires Tenstorrent hardware + live sglang server on :30000",
)

pytestmark = [pytest.mark.paged_backend, pytest.mark.hardware]

_SERVER = "http://localhost:30000"
_LOG_PATH = "/tmp/sglang-paged.log"
_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "_fixtures")
_FIXTURE_PATH = os.path.join(_FIXTURES_DIR, "perf_log.json")
_TMP_PATH = "/tmp/tt_perf_log_paged.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _completion_sync(prompt: str, max_tokens: int = 50, timeout: int = 180) -> tuple:
    """POST /v1/completions (synchronous). Returns (text, usage, wall_time_s)."""
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
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        assert r.status == 200
        resp = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    return resp["choices"][0]["text"], resp.get("usage", {}), elapsed


def _get_loads() -> dict:
    """GET /v1/loads and return the first load entry."""
    with urllib.request.urlopen(f"{_SERVER}/v1/loads", timeout=10) as r:
        data = json.loads(r.read())
    return data["loads"][0]


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


def _build_prompt(n_tokens: int) -> str:
    """Build a prompt of approximately n_tokens by repeating a seed sentence."""
    seed = (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
    )
    repeats = max(1, n_tokens // 13)  # ~13 tokens/sentence for Llama tokenizer
    return seed * repeats


def _measure_decode_tok_s(batch_size: int, max_tokens: int = 30) -> dict:
    """Measure decode tok/s at a given batch_size.

    Strategy: issue `batch_size` requests concurrently using threads, measure
    wall time from first request start to last response end. Report total tokens
    / elapsed as aggregate tok/s, then divide by batch_size for per-request.

    Note: The paged backend supports max_running_requests=4. At B=4 all 4
    requests run in a single decode batch, giving a higher total tok/s but
    similar per-request tok/s as B=1.
    """
    prompt = _build_prompt(50)  # short prompt to minimize prefill noise
    results = [None] * batch_size
    errors = []

    def _worker(idx):
        try:
            text, usage, elapsed = _completion_sync(prompt, max_tokens=max_tokens, timeout=240)
            results[idx] = {
                "elapsed": elapsed,
                "tokens": usage.get("completion_tokens", 0),
            }
        except Exception as e:
            errors.append(str(e))

    t_start = time.perf_counter()
    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(batch_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    t_end = time.perf_counter()

    if errors:
        return {"batch_size": batch_size, "error": errors[0], "tok_s": 0.0}

    # Aggregate tok/s = total tokens generated / wall-clock window
    total_tokens = sum(r["tokens"] for r in results if r)
    wall_time = t_end - t_start
    tok_s = total_tokens / wall_time if wall_time > 0 else 0.0

    # Per-request median
    per_req_tok_s = [r["tokens"] / r["elapsed"] for r in results if r and r["elapsed"] > 0]
    per_req_median = statistics.median(per_req_tok_s) if per_req_tok_s else 0.0

    return {
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "wall_time_s": wall_time,
        "tok_s_aggregate": tok_s,
        "tok_s_per_req_median": per_req_median,
        "per_request_results": results,
    }


def _sample_kv_utilization(n_samples: int = 3, interval_s: float = 0.5) -> list:
    """Sample KV pool utilization from /v1/loads at `n_samples` points."""
    samples = []
    for i in range(n_samples):
        loads = _get_loads()
        samples.append({
            "sample_idx": i,
            "token_usage": loads.get("token_usage", 0.0),
            "num_used_tokens": loads.get("num_used_tokens", 0),
            "max_total_num_tokens": loads.get("max_total_num_tokens", 0),
            "gen_throughput": loads.get("gen_throughput", 0.0),
        })
        if i < n_samples - 1:
            time.sleep(interval_s)
    return samples


# ── Tests ──────────────────────────────────────────────────────────────────────

@REQUIRES_TT
def test_perf_log_batched_decode_tok_s():
    """§9.10: batched decode tok/s at B=1, B=2, B=4.

    Sanity floor: B=4 aggregate tok/s >= 1.2× B=1 (batching helps).
    Test passes if measurements complete; floor failure is logged not fatal.
    """
    results = {}
    floors_met = {}

    print()
    for bs in [1, 2, 4]:
        print(f"  Measuring decode tok/s at B={bs}...", flush=True)
        m = _measure_decode_tok_s(bs, max_tokens=30)
        results[f"B{bs}"] = m
        print(
            f"  B={bs}: aggregate={m.get('tok_s_aggregate', 0):.2f} tok/s  "
            f"per-req-median={m.get('tok_s_per_req_median', 0):.2f} tok/s  "
            f"wall={m.get('wall_time_s', 0):.2f}s",
            flush=True,
        )

    # Sanity floor: B=4 should be >= 1.2× B=1 in aggregate tok/s
    tok_s_b1 = results["B1"].get("tok_s_aggregate", 0)
    tok_s_b4 = results["B4"].get("tok_s_aggregate", 0)
    floor_ratio = 1.2
    if tok_s_b1 > 0 and tok_s_b4 > 0:
        ratio = tok_s_b4 / tok_s_b1
        floors_met["B4_vs_B1_ratio"] = ratio
        if ratio >= floor_ratio:
            print(f"  Floor PASS: B=4/B=1 ratio={ratio:.2f}x >= {floor_ratio}x")
        else:
            print(
                f"  Floor WARN: B=4/B=1 ratio={ratio:.2f}x < {floor_ratio}x "
                f"(batching benefit below threshold — tolerated)"
            )
    else:
        print(f"  Floor SKIP: one or both measurements returned 0 tok/s")

    # Hard assertion: measurements captured (not zero)
    assert tok_s_b1 > 0, f"B=1 measurement returned 0 tok/s — request failed?"
    assert tok_s_b4 > 0, f"B=4 measurement returned 0 tok/s — all requests failed?"

    # Save partial result for cross-test aggregation
    test_perf_log_batched_decode_tok_s._results = results
    test_perf_log_batched_decode_tok_s._floors_met = floors_met


@REQUIRES_TT
def test_perf_log_prefix_cache_latency():
    """§9.10: prefix hit vs miss latency — 2 cold + 2 warm requests.

    Sanity floor: warm - cold latency delta < 10ms (cache hit should not add
    latency vs fresh compute). Test passes if measurements complete.
    """
    _flush_cache()
    time.sleep(0.3)

    # Shared system prompt: ~200 tokens (enough for multiple cache pages)
    system_prompt = (
        "You are a helpful assistant specializing in mathematics. "
        "Please answer clearly and concisely. "
    ) * 12

    cold_times = []
    warm_times = []

    print()
    for i, q in enumerate(["What is 2+2?", "What is 3×3?"]):
        prompt = system_prompt + q
        _, _, elapsed = _completion_sync(prompt, max_tokens=5, timeout=120)
        cold_times.append(elapsed)
        print(f"  cold req {i+1}: {elapsed:.3f}s", flush=True)

    for i, q in enumerate(["What is 4+4?", "What is 5×5?"]):
        prompt = system_prompt + q
        _, _, elapsed = _completion_sync(prompt, max_tokens=5, timeout=120)
        warm_times.append(elapsed)
        print(f"  warm req {i+1}: {elapsed:.3f}s (should hit prefix cache)", flush=True)

    cold_avg = statistics.mean(cold_times)
    warm_avg = statistics.mean(warm_times)
    delta_ms = (warm_avg - cold_avg) * 1000

    print(f"\n  cold_avg={cold_avg:.3f}s  warm_avg={warm_avg:.3f}s  "
          f"delta={delta_ms:+.1f}ms")

    # Floor: warm should not be >10ms SLOWER than cold (cache shouldn't add latency)
    if delta_ms < 10:
        print(f"  Floor PASS: prefix-hit latency delta {delta_ms:+.1f}ms < 10ms")
    else:
        print(
            f"  Floor WARN: prefix-hit latency delta {delta_ms:+.1f}ms >= 10ms "
            f"(may indicate eviction overhead or decode-dominance; tolerated)"
        )

    result = {
        "cold_times_s": cold_times,
        "warm_times_s": warm_times,
        "cold_avg_s": cold_avg,
        "warm_avg_s": warm_avg,
        "delta_ms": delta_ms,
        "floor_10ms_met": delta_ms < 10,
    }
    test_perf_log_prefix_cache_latency._result = result

    # Hard assertion: measurements captured
    assert cold_avg > 0 and warm_avg > 0, "Prefix cache latency measurement failed"


@REQUIRES_TT
def test_perf_log_kv_utilization():
    """§9.10: KV pool utilization — 3 sample points from /v1/loads.

    Sanity floor: max_util < 95% (pool not saturated at steady state).
    Test passes if samples are captured.
    """
    print()

    # Sample idle state first
    idle_samples = _sample_kv_utilization(n_samples=1)
    print(f"  Idle KV util: {idle_samples[0]['token_usage']:.3f}")

    # Issue a request and sample during/after
    def _background_request():
        _completion_sync(_build_prompt(200), max_tokens=50, timeout=180)

    bg = threading.Thread(target=_background_request)
    bg.start()
    time.sleep(0.1)  # Let request start

    active_samples = _sample_kv_utilization(n_samples=3, interval_s=0.5)
    bg.join(timeout=180)

    # Post-completion sample
    time.sleep(0.2)
    post_samples = _sample_kv_utilization(n_samples=1)

    all_samples = idle_samples + active_samples + post_samples
    max_util = max(s["token_usage"] for s in all_samples)
    print(f"\n  KV utilization samples:")
    for s in all_samples:
        print(f"    sample {s['sample_idx']}: util={s['token_usage']:.4f} "
              f"used={s['num_used_tokens']}/{s['max_total_num_tokens']}")
    print(f"  max_util={max_util:.4f}")

    # Floor: max_util < 0.95
    if max_util < 0.95:
        print(f"  Floor PASS: max KV util {max_util:.4f} < 0.95")
    else:
        print(
            f"  Floor WARN: max KV util {max_util:.4f} >= 0.95 "
            f"(pool near saturation; tolerated)"
        )

    result = {
        "all_samples": all_samples,
        "max_util": max_util,
        "floor_95pct_met": max_util < 0.95,
    }
    test_perf_log_kv_utilization._result = result

    # Hard assertion: samples captured
    assert len(all_samples) >= 3, f"Only {len(all_samples)} KV samples captured"
    assert max_util >= 0, "max_util is negative — logic error"


@REQUIRES_TT
def test_perf_log_write_fixture():
    """§9.10: aggregate all perf measurements and write to fixture file.

    This test collects results from the preceding three tests and writes
    a JSON fixture to _fixtures/perf_log.json. It passes as long as the
    preceding tests ran (which is gated by the fixture attributes).
    """
    print()

    # Gather results from preceding tests (may be None if tests skipped)
    batched = getattr(test_perf_log_batched_decode_tok_s, "_results", {})
    prefix = getattr(test_perf_log_prefix_cache_latency, "_result", {})
    kv = getattr(test_perf_log_kv_utilization, "_result", {})
    floors = getattr(test_perf_log_batched_decode_tok_s, "_floors_met", {})

    fixture = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec_section": "§9.10",
        "batched_decode_tok_s": batched,
        "prefix_cache_latency": prefix,
        "kv_utilization": kv,
        "floors": {
            "B4_vs_B1_ratio": floors.get("B4_vs_B1_ratio"),
            "prefix_hit_delta_ms": prefix.get("delta_ms"),
            "prefix_hit_delta_floor_10ms_met": prefix.get("floor_10ms_met"),
            "kv_util_max": kv.get("max_util"),
            "kv_util_floor_95pct_met": kv.get("floor_95pct_met"),
        },
    }

    # Write fixture
    os.makedirs(_FIXTURES_DIR, exist_ok=True)
    with open(_FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    with open(_TMP_PATH, "w") as f:
        json.dump(fixture, f, indent=2)

    print(f"  Wrote perf fixture to: {_FIXTURE_PATH}")
    print(f"  Wrote perf fixture to: {_TMP_PATH}")

    # Print summary
    print(f"\n=== §9.10 Perf Log Summary ===")
    if batched:
        for key in ["B1", "B2", "B4"]:
            m = batched.get(key, {})
            print(
                f"  {key}: aggregate {m.get('tok_s_aggregate', 0):.2f} tok/s  "
                f"per-req-median {m.get('tok_s_per_req_median', 0):.2f} tok/s"
            )
    if prefix:
        print(
            f"  Prefix: cold={prefix.get('cold_avg_s', 0):.3f}s  "
            f"warm={prefix.get('warm_avg_s', 0):.3f}s  "
            f"delta={prefix.get('delta_ms', 0):+.1f}ms"
        )
    if kv:
        print(f"  KV max_util: {kv.get('max_util', 0):.4f}")
    print()

    assert os.path.exists(_FIXTURE_PATH), f"Fixture file not written: {_FIXTURE_PATH}"
