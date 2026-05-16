#!/usr/bin/env python3
"""
v136 Soak Test: 6-hour continuous throughput stability test.

Sends sequential /generate requests to SGLang (Qwen3-8B on 2xP150a,
lofi_hifi2_sdpa mode) for 6 hours, recording per-request tok/s.
Every 60 requests prints a summary. After completion, compares
first-hour avg vs last-hour avg to compute drift.

Target: drift < 5%.

Usage:
    python3 soak_test_6h.py [--duration-hours 6] [--url http://localhost:30000]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

# Diverse prompts to exercise different generation patterns
PROMPTS = [
    "The capital of Japan is",
    "Write a detailed paragraph about the history of computing.",
    "Explain the theory of general relativity in simple terms.",
    "World War 2 ended in",
    "The speed of light is approximately",
    "1 + 1 =",
    "Describe the process of photosynthesis step by step.",
    "What are the main differences between Python and C++?",
    "The largest ocean on Earth is",
    "Explain how a neural network learns from data.",
    "In mathematics, the Fibonacci sequence is defined as",
    "The periodic table of elements was first organized by",
    "Describe the water cycle in detail.",
    "What is the significance of the Turing test?",
    "The principle of conservation of energy states that",
    "Explain how encryption works in modern communication.",
]

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
SUMMARY_INTERVAL = 60  # requests between summaries


def send_request(url: str, prompt: str) -> dict:
    """Send a single generate request and return timing info."""
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
        "log_metrics": True,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{url}/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        t1 = time.monotonic()

        meta = data["meta_info"]
        completion_tokens = meta["completion_tokens"]
        e2e_latency = meta["e2e_latency"]
        tok_s = completion_tokens / e2e_latency if e2e_latency > 0 else 0

        return {
            "ok": True,
            "tok_s": tok_s,
            "completion_tokens": completion_tokens,
            "e2e_latency": e2e_latency,
            "wall_time": t1 - t0,
            "prompt": prompt[:60],
            "text_preview": data.get("text", "")[:100],
        }
    except Exception as e:
        t1 = time.monotonic()
        return {
            "ok": False,
            "tok_s": 0,
            "completion_tokens": 0,
            "e2e_latency": 0,
            "wall_time": t1 - t0,
            "error": str(e),
            "prompt": prompt[:60],
        }


def main():
    parser = argparse.ArgumentParser(description="6-hour soak test")
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument(
        "--output",
        default="/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v136_soak_test_6h.json",
    )
    args = parser.parse_args()

    duration_s = args.duration_hours * 3600
    url = args.url

    print(f"=== v136 Soak Test ===")
    print(f"Duration: {args.duration_hours}h ({duration_s:.0f}s)")
    print(f"URL: {url}")
    print(f"Output: {args.output}")
    print(f"Prompts: {len(PROMPTS)} diverse prompts, {MAX_NEW_TOKENS} max tokens each")
    print(f"Summary every {SUMMARY_INTERVAL} requests")
    print(f"Start: {datetime.now(timezone.utc).isoformat()}")
    print()
    sys.stdout.flush()

    # Health check
    try:
        r = requests.get(f"{url}/health", timeout=5)
        print(f"Health check: HTTP {r.status_code}")
    except Exception as e:
        print(f"WARNING: Health check failed: {e}")
        print("Proceeding anyway...")
    sys.stdout.flush()

    start_time = time.monotonic()
    start_wall = datetime.now(timezone.utc).isoformat()
    results = []
    errors = []
    crashes = []
    request_idx = 0
    prompt_idx = 0

    # Track per-hour buckets
    hourly_buckets = {}  # hour_num -> list of tok_s values

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= duration_s:
            break

        # Select prompt (round-robin)
        prompt = PROMPTS[prompt_idx % len(PROMPTS)]
        prompt_idx += 1

        result = send_request(url, prompt)
        result["request_idx"] = request_idx
        result["elapsed_s"] = elapsed
        result["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Bucket by hour
        hour_num = int(elapsed / 3600)
        if hour_num not in hourly_buckets:
            hourly_buckets[hour_num] = []

        if result["ok"]:
            results.append(result)
            hourly_buckets[hour_num].append(result["tok_s"])
        else:
            errors.append(result)
            # Check if this is a server crash (connection refused vs timeout)
            err_str = result.get("error", "")
            if "Connection refused" in err_str or "ConnectionError" in err_str:
                crash_info = {
                    "request_idx": request_idx,
                    "elapsed_s": elapsed,
                    "elapsed_h": elapsed / 3600,
                    "error": err_str,
                    "timestamp": result["timestamp"],
                }
                crashes.append(crash_info)
                print(f"\n!!! SERVER CRASH at request {request_idx} "
                      f"({elapsed/3600:.2f}h): {err_str}")
                print("Waiting 60s for potential restart...")
                sys.stdout.flush()
                time.sleep(60)
                # Try health check
                try:
                    r = requests.get(f"{url}/health", timeout=10)
                    print(f"Server recovered: HTTP {r.status_code}")
                except Exception:
                    print("Server still down. Waiting another 120s...")
                    time.sleep(120)
                    try:
                        r = requests.get(f"{url}/health", timeout=10)
                        print(f"Server recovered after extended wait: HTTP {r.status_code}")
                    except Exception:
                        print("Server did not recover. Ending test.")
                        break
                sys.stdout.flush()

        request_idx += 1

        # Print per-request one-liner
        if result["ok"]:
            print(
                f"  [{request_idx:5d}] {elapsed/3600:5.2f}h  "
                f"{result['tok_s']:5.1f} tok/s  "
                f"{result['completion_tokens']:3d}tok  "
                f"{result['e2e_latency']:.2f}s  "
                f"{result['prompt'][:40]}"
            )
        else:
            print(
                f"  [{request_idx:5d}] {elapsed/3600:5.2f}h  "
                f"ERROR: {result.get('error', 'unknown')[:60]}"
            )

        # Print summary every SUMMARY_INTERVAL requests
        if request_idx % SUMMARY_INTERVAL == 0 and len(results) > 0:
            recent = results[-SUMMARY_INTERVAL:]
            recent_ok = [r for r in recent if r["ok"]]
            if recent_ok:
                avg_tps = sum(r["tok_s"] for r in recent_ok) / len(recent_ok)
                min_tps = min(r["tok_s"] for r in recent_ok)
                max_tps = max(r["tok_s"] for r in recent_ok)
                all_avg = sum(r["tok_s"] for r in results) / len(results)
                print(
                    f"\n  --- Summary @ request {request_idx} "
                    f"({elapsed/3600:.2f}h) ---"
                )
                print(
                    f"  Last {SUMMARY_INTERVAL}: avg={avg_tps:.1f} "
                    f"min={min_tps:.1f} max={max_tps:.1f} tok/s"
                )
                print(
                    f"  Overall: avg={all_avg:.1f} tok/s, "
                    f"{len(results)} ok / {len(errors)} errors"
                )

                # Per-hour breakdown so far
                print("  Per-hour averages:", end="")
                for h in sorted(hourly_buckets.keys()):
                    vals = hourly_buckets[h]
                    if vals:
                        h_avg = sum(vals) / len(vals)
                        print(f"  H{h}={h_avg:.1f}", end="")
                print()
                print()
            sys.stdout.flush()

        sys.stdout.flush()

    # ---- Test complete ----
    end_time = time.monotonic()
    end_wall = datetime.now(timezone.utc).isoformat()
    total_elapsed = end_time - start_time

    print()
    print("=" * 60)
    print("  SOAK TEST COMPLETE")
    print("=" * 60)

    # Compute drift
    first_hour_vals = hourly_buckets.get(0, [])
    # Find the last hour that has data
    last_hour_num = max(hourly_buckets.keys()) if hourly_buckets else 0
    last_hour_vals = hourly_buckets.get(last_hour_num, [])

    first_hour_avg = (
        sum(first_hour_vals) / len(first_hour_vals) if first_hour_vals else 0
    )
    last_hour_avg = (
        sum(last_hour_vals) / len(last_hour_vals) if last_hour_vals else 0
    )

    drift_pct = (
        abs(last_hour_avg - first_hour_avg) / first_hour_avg * 100
        if first_hour_avg > 0
        else float("inf")
    )
    drift_direction = "down" if last_hour_avg < first_hour_avg else "up"

    overall_avg = sum(r["tok_s"] for r in results) / len(results) if results else 0
    overall_min = min(r["tok_s"] for r in results) if results else 0
    overall_max = max(r["tok_s"] for r in results) if results else 0
    overall_std = (
        (sum((r["tok_s"] - overall_avg) ** 2 for r in results) / len(results)) ** 0.5
        if results
        else 0
    )

    print(f"  Duration: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)")
    print(f"  Requests: {len(results)} ok, {len(errors)} errors, {len(crashes)} crashes")
    print(f"  Overall: avg={overall_avg:.1f} min={overall_min:.1f} "
          f"max={overall_max:.1f} std={overall_std:.2f} tok/s")
    print()

    # Per-hour breakdown
    print("  Per-hour breakdown:")
    hourly_summary = {}
    for h in sorted(hourly_buckets.keys()):
        vals = hourly_buckets[h]
        if vals:
            h_avg = sum(vals) / len(vals)
            h_min = min(vals)
            h_max = max(vals)
            h_std = (sum((v - h_avg) ** 2 for v in vals) / len(vals)) ** 0.5
            hourly_summary[f"hour_{h}"] = {
                "avg_tok_s": round(h_avg, 2),
                "min_tok_s": round(h_min, 2),
                "max_tok_s": round(h_max, 2),
                "std_tok_s": round(h_std, 2),
                "num_requests": len(vals),
            }
            print(
                f"    H{h}: avg={h_avg:.1f} min={h_min:.1f} "
                f"max={h_max:.1f} std={h_std:.2f} n={len(vals)}"
            )
    print()
    print(f"  First hour avg: {first_hour_avg:.1f} tok/s ({len(first_hour_vals)} reqs)")
    print(f"  Last hour avg:  {last_hour_avg:.1f} tok/s ({len(last_hour_vals)} reqs)")
    print(f"  Drift: {drift_pct:.2f}% ({drift_direction})")
    print(f"  Target: < 5.0%")
    print(f"  RESULT: {'PASS' if drift_pct < 5.0 else 'FAIL'}")
    print()
    sys.stdout.flush()

    # Build output JSON
    output = {
        "iteration": "v136 -- 6-hour soak test, Qwen3-8B on 2xP150a",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": "Qwen3-8B",
        "device": "2xP150a (P300_X2)",
        "precision_mode": "lofi_hifi2_sdpa",
        "config": {
            "max_running_requests": 1,
            "context_length": 2048,
            "attention_backend": "torch_native",
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "num_prompts": len(PROMPTS),
        },
        "duration": {
            "target_hours": args.duration_hours,
            "actual_seconds": round(total_elapsed, 1),
            "actual_hours": round(total_elapsed / 3600, 2),
            "start": start_wall,
            "end": end_wall,
        },
        "summary": {
            "total_requests_ok": len(results),
            "total_errors": len(errors),
            "total_crashes": len(crashes),
            "overall_avg_tok_s": round(overall_avg, 2),
            "overall_min_tok_s": round(overall_min, 2),
            "overall_max_tok_s": round(overall_max, 2),
            "overall_std_tok_s": round(overall_std, 2),
        },
        "drift": {
            "first_hour_avg_tok_s": round(first_hour_avg, 2),
            "first_hour_num_requests": len(first_hour_vals),
            "last_hour_num": last_hour_num,
            "last_hour_avg_tok_s": round(last_hour_avg, 2),
            "last_hour_num_requests": len(last_hour_vals),
            "drift_pct": round(drift_pct, 2),
            "drift_direction": drift_direction,
            "target_pct": 5.0,
            "result": "PASS" if drift_pct < 5.0 else "FAIL",
        },
        "hourly_summary": hourly_summary,
        "crashes": crashes,
        # Store per-request data compactly: just tok_s and elapsed_s
        "per_request": [
            {
                "idx": r["request_idx"],
                "elapsed_s": round(r["elapsed_s"], 1),
                "tok_s": round(r["tok_s"], 2),
                "tokens": r["completion_tokens"],
            }
            for r in results
        ],
    }

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.output}")

    # Return exit code based on result
    sys.exit(0 if drift_pct < 5.0 else 1)


if __name__ == "__main__":
    main()
