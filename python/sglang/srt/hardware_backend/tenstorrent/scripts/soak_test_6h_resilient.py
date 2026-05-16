#!/usr/bin/env python3
"""
v136 Resilient Soak Test: 6-hour continuous throughput stability test
with automatic crash recovery.

Handles container crashes (exit 137) by:
  1. Resetting TT devices via tt-smi
  2. Restarting the container
  3. Re-copying fork patches
  4. Relaunching the SGLang server
  5. Resuming the workload loop

Records all data across crash/restart cycles in a single output file.

Usage:
    python3 soak_test_6h_resilient.py [--duration-hours 6]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests as http_requests

# --- Configuration ---
CONTAINER = "p3a-ngram"
TT_SMI = "/home/mhnie/.tenstorrent-venv/bin/tt-smi"
FORK_DIR = "/home/mhnie/tt-metal-sglang/models/tt_transformers/tt"
FORK_FILES = ["model_config.py", "generator_sglang.py", "distributed_norm.py", "attention.py"]
CONTAINER_TT_DIR = "/tt-metal/models/tt_transformers/tt"

SERVER_URL = "http://localhost:30000"
MAX_SERVER_WAIT = 600  # seconds

# Diverse prompts
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
SUMMARY_INTERVAL = 60


def run_cmd(cmd, timeout=120, check=False):
    """Run a shell command and return (returncode, stdout)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, str(e)


def reset_devices():
    """Reset TT devices via tt-smi (one at a time to avoid bus errors)."""
    print("  [recovery] Resetting TT devices...")
    sys.stdout.flush()
    all_ok = True
    for dev_id in [0, 1]:
        rc, out = run_cmd(f"{TT_SMI} -r {dev_id}", timeout=60)
        if rc != 0:
            print(f"  [recovery] WARNING: tt-smi -r {dev_id} returned {rc}: {out}")
            all_ok = False
        time.sleep(3)
    time.sleep(5)
    return all_ok


def ensure_container_running():
    """Ensure the container is running."""
    rc, out = run_cmd(f"podman ps --filter name={CONTAINER} --format '{{{{.Names}}}}'")
    if CONTAINER in out:
        return True

    print(f"  [recovery] Container {CONTAINER} not running, starting...")
    sys.stdout.flush()
    rc, out = run_cmd(f"podman start {CONTAINER}", timeout=30)
    if rc != 0:
        print(f"  [recovery] Failed to start container: {out}")
        return False
    time.sleep(5)
    # Verify it actually started
    rc2, out2 = run_cmd(f"podman ps --filter name={CONTAINER} --format '{{{{.Names}}}}'")
    if CONTAINER not in out2:
        print(f"  [recovery] Container started but not showing as running")
        return False
    return True


def copy_fork_patches():
    """Copy fork patches into the container."""
    print("  [recovery] Copying fork patches...")
    sys.stdout.flush()
    for f in FORK_FILES:
        src = os.path.join(FORK_DIR, f)
        dst = f"{CONTAINER}:{CONTAINER_TT_DIR}/{f}"
        rc, out = run_cmd(f"podman cp {src} {dst}", timeout=30)
        if rc != 0:
            print(f"  [recovery] WARNING: Failed to copy {f}: {out}")
            return False
    return True


def clear_cache():
    """Clear TT model cache."""
    run_cmd(
        f"podman exec {CONTAINER} bash -lc 'rm -rf /root/.cache/tt-metal-model-cache/P300/* 2>/dev/null'",
        timeout=15,
    )


def kill_server():
    """Kill any existing SGLang server in the container."""
    run_cmd(
        f"podman exec {CONTAINER} bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null; "
        f"pkill -9 -f \"sglang::\" 2>/dev/null'",
        timeout=15,
    )
    time.sleep(3)


def launch_server():
    """Launch SGLang server inside the container."""
    print("  [recovery] Launching SGLang server...")
    sys.stdout.flush()
    kill_server()
    time.sleep(2)

    cmd = (
        f"podman exec -d "
        f"-e SGLANG_PLATFORM=tenstorrent "
        f"-e SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged "
        f"-e SGLANG_TT_QWEN3_PRECISION=lofi_hifi2_sdpa "
        f"{CONTAINER} bash -lc '"
        f"source /opt/venv/bin/activate; python -m sglang.launch_server "
        f"--model-path /models/Qwen3-8B --trust-remote-code "
        f"--host 0.0.0.0 --port 30000 --device tenstorrent "
        f"--max-running-requests 1 "
        f"--context-length 2048 "
        f"--mem-fraction-static 0.5 "
        f"--attention-backend torch_native "
        f"--disable-cuda-graph 2>&1 | tee /tmp/sglang_soak_test.log'"
    )
    rc, out = run_cmd(cmd, timeout=30)
    if rc != 0:
        print(f"  [recovery] Failed to launch server: {out}")
        return False
    return True


def wait_for_server(max_wait=MAX_SERVER_WAIT):
    """Wait for the server to become healthy."""
    print("  [recovery] Waiting for server health check...", end="", flush=True)
    waited = 0
    while waited < max_wait:
        try:
            r = http_requests.get(f"{SERVER_URL}/health", timeout=3)
            if r.status_code == 200:
                print(f" ready after {waited}s")
                sys.stdout.flush()
                # Warmup request
                try:
                    http_requests.post(
                        f"{SERVER_URL}/generate",
                        json={
                            "text": "warmup",
                            "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
                        },
                        timeout=60,
                    )
                except Exception:
                    pass
                time.sleep(2)
                return True
        except Exception:
            pass

        # Check container is still alive
        rc, out = run_cmd(f"podman ps --filter name={CONTAINER} --format '{{{{.Status}}}}'")
        if "Up" not in out and "running" not in out.lower():
            print(f" CONTAINER DIED during wait")
            return False

        time.sleep(10)
        waited += 10
        if waited % 60 == 0:
            print(f" {waited}s...", end="", flush=True)

    print(f" TIMEOUT after {max_wait}s")
    return False


def stop_container():
    """Stop the container (needed before device reset)."""
    print("  [recovery] Stopping container...")
    sys.stdout.flush()
    run_cmd(f"podman exec {CONTAINER} bash -lc 'pkill -9 -f python 2>/dev/null; pkill -9 -f sglang 2>/dev/null'", timeout=15)
    time.sleep(3)
    run_cmd(f"podman stop {CONTAINER}", timeout=30)
    time.sleep(5)


def full_recovery():
    """Perform full crash recovery: stop -> reset -> start -> patches -> launch -> wait."""
    print("\n=== CRASH RECOVERY ===")
    sys.stdout.flush()

    for attempt in range(3):
        print(f"  Recovery attempt {attempt + 1}/3")
        sys.stdout.flush()

        # Must stop container before device reset so devices are not held open
        stop_container()

        reset_devices()

        if not ensure_container_running():
            time.sleep(10)
            continue

        if not copy_fork_patches():
            time.sleep(5)
            continue

        clear_cache()

        if not launch_server():
            time.sleep(10)
            continue

        if wait_for_server():
            print("=== RECOVERY SUCCESSFUL ===\n")
            sys.stdout.flush()
            return True

        print(f"  Attempt {attempt + 1} failed, retrying...")
        sys.stdout.flush()
        time.sleep(15)

    print("=== RECOVERY FAILED after 3 attempts ===\n")
    sys.stdout.flush()
    return False


def send_request(prompt):
    """Send a single generate request."""
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
        resp = http_requests.post(
            f"{SERVER_URL}/generate", json=payload, timeout=120
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
        }


def is_crash_error(error_str):
    """Check if an error indicates a server/container crash."""
    crash_patterns = [
        "Connection refused",
        "ConnectionError",
        "RemoteDisconnected",
        "Connection aborted",
        "Connection reset",
    ]
    return any(p in error_str for p in crash_patterns)


def main():
    parser = argparse.ArgumentParser(description="6-hour resilient soak test")
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument(
        "--output",
        default="/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v136_soak_test_6h.json",
    )
    args = parser.parse_args()

    duration_s = args.duration_hours * 3600

    print(f"=== v136 Resilient Soak Test ===")
    print(f"Duration: {args.duration_hours}h ({duration_s:.0f}s)")
    print(f"Output: {args.output}")
    print(f"Prompts: {len(PROMPTS)} diverse, {MAX_NEW_TOKENS} max tokens each")
    print(f"Summary every {SUMMARY_INTERVAL} requests")
    print(f"Start: {datetime.now(timezone.utc).isoformat()}")
    print()
    sys.stdout.flush()

    # Initial health check
    try:
        r = http_requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"Server already running: HTTP {r.status_code}")
        # Quick warmup
        send_request("warmup")
        time.sleep(2)
    except Exception:
        print("Server not running. Performing initial setup...")
        if not full_recovery():
            print("FATAL: Cannot start server. Exiting.")
            sys.exit(1)

    start_time = time.monotonic()
    start_wall = datetime.now(timezone.utc).isoformat()
    results = []
    errors = []
    crashes = []
    crash_recovery_times = []
    request_idx = 0
    prompt_idx = 0
    consecutive_errors = 0
    hourly_buckets = {}

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= duration_s:
            break

        prompt = PROMPTS[prompt_idx % len(PROMPTS)]
        prompt_idx += 1

        result = send_request(prompt)
        result["request_idx"] = request_idx
        result["elapsed_s"] = elapsed
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["prompt"] = prompt[:60]

        hour_num = int(elapsed / 3600)
        if hour_num not in hourly_buckets:
            hourly_buckets[hour_num] = []

        if result["ok"]:
            results.append(result)
            hourly_buckets[hour_num].append(result["tok_s"])
            consecutive_errors = 0

            print(
                f"  [{request_idx:5d}] {elapsed/3600:5.2f}h  "
                f"{result['tok_s']:5.1f} tok/s  "
                f"{result['completion_tokens']:3d}tok  "
                f"{result['e2e_latency']:.2f}s  "
                f"{prompt[:40]}"
            )
        else:
            errors.append(result)
            consecutive_errors += 1
            err_str = result.get("error", "")

            print(
                f"  [{request_idx:5d}] {elapsed/3600:5.2f}h  "
                f"ERROR: {err_str[:60]}"
            )

            if is_crash_error(err_str):
                crash_info = {
                    "request_idx": request_idx,
                    "elapsed_s": elapsed,
                    "elapsed_h": round(elapsed / 3600, 3),
                    "error": err_str[:200],
                    "timestamp": result["timestamp"],
                }
                crashes.append(crash_info)
                print(
                    f"\n!!! SERVER CRASH #{len(crashes)} at request {request_idx} "
                    f"({elapsed/3600:.2f}h)"
                )
                sys.stdout.flush()

                # Perform full recovery
                recovery_start = time.monotonic()
                if full_recovery():
                    recovery_time = time.monotonic() - recovery_start
                    crash_recovery_times.append(recovery_time)
                    print(f"  Recovery took {recovery_time:.0f}s")
                    consecutive_errors = 0
                else:
                    print("FATAL: Recovery failed. Ending test.")
                    break
            elif consecutive_errors >= 5:
                print(f"\n!!! 5 consecutive errors. Attempting recovery...")
                sys.stdout.flush()
                recovery_start = time.monotonic()
                if full_recovery():
                    recovery_time = time.monotonic() - recovery_start
                    crash_recovery_times.append(recovery_time)
                    consecutive_errors = 0
                else:
                    print("FATAL: Recovery failed. Ending test.")
                    break

        request_idx += 1

        # Summary every N requests
        if request_idx % SUMMARY_INTERVAL == 0 and results:
            recent = [r for r in results[-SUMMARY_INTERVAL:] if r["ok"]]
            if recent:
                avg_tps = sum(r["tok_s"] for r in recent) / len(recent)
                min_tps = min(r["tok_s"] for r in recent)
                max_tps = max(r["tok_s"] for r in recent)
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
                    f"{len(results)} ok / {len(errors)} err / {len(crashes)} crashes"
                )
                print("  Per-hour:", end="")
                for h in sorted(hourly_buckets.keys()):
                    vals = hourly_buckets[h]
                    if vals:
                        h_avg = sum(vals) / len(vals)
                        print(f"  H{h}={h_avg:.1f}({len(vals)})", end="")
                print("\n")
            sys.stdout.flush()

        sys.stdout.flush()

    # ---- Complete ----
    end_time = time.monotonic()
    end_wall = datetime.now(timezone.utc).isoformat()
    total_elapsed = end_time - start_time

    print()
    print("=" * 60)
    print("  SOAK TEST COMPLETE")
    print("=" * 60)

    first_hour_vals = hourly_buckets.get(0, [])
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

    hourly_summary = {}
    print("  Per-hour breakdown:")
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
    print(f"  Last hour (H{last_hour_num}) avg: {last_hour_avg:.1f} tok/s ({len(last_hour_vals)} reqs)")
    print(f"  Drift: {drift_pct:.2f}% ({drift_direction})")
    print(f"  Target: < 5.0%")
    print(f"  RESULT: {'PASS' if drift_pct < 5.0 else 'FAIL'}")
    if crashes:
        print(f"\n  Crashes: {len(crashes)}")
        for i, c in enumerate(crashes):
            print(f"    #{i+1}: request {c['request_idx']} at {c['elapsed_h']:.2f}h")
        if crash_recovery_times:
            print(f"  Avg recovery time: {sum(crash_recovery_times)/len(crash_recovery_times):.0f}s")
    print()
    sys.stdout.flush()

    output = {
        "iteration": "v136 -- 6-hour resilient soak test, Qwen3-8B on 2xP150a",
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
        "crash_recovery_times_s": [round(t, 1) for t in crash_recovery_times],
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.output}")

    sys.exit(0 if drift_pct < 5.0 else 1)


if __name__ == "__main__":
    main()
