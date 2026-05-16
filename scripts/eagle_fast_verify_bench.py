#!/usr/bin/env python3
"""Benchmark EAGLE fast-verify throughput on Tenstorrent hardware.

Runs warmup + measured generation requests against a running SGLang server
and reports output tokens/second.
"""

import argparse
import json
import time
import requests
import sys


def generate_one(url: str, prompt: str, max_tokens: int, temperature: float = 0.0):
    """Send a single /generate request and return (output_tokens, elapsed_sec, text)."""
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/generate", json=payload, timeout=300)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    text = data.get("text", "")
    # Count output tokens from meta_info if available
    meta = data.get("meta_info", {})
    output_tokens = meta.get("completion_tokens", len(text.split()))
    accept_length = meta.get("accept_length", None)
    return output_tokens, elapsed, text, accept_length


def main():
    parser = argparse.ArgumentParser(description="EAGLE fast-verify benchmark")
    parser.add_argument("--url", default="http://localhost:30000", help="Server URL")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs")
    parser.add_argument("--max-tokens", type=int, default=100, help="Output tokens per request")
    parser.add_argument("--prompt", default="Explain the theory of general relativity in detail.", help="Input prompt")
    args = parser.parse_args()

    print(f"=== EAGLE Fast-Verify Benchmark ===")
    print(f"URL: {args.url}")
    print(f"Warmup: {args.warmup}, Measured: {args.runs}, Max tokens: {args.max_tokens}")
    print(f"Prompt: {args.prompt[:60]}...")
    print()

    # Warmup
    print("--- Warmup ---")
    for i in range(args.warmup):
        tokens, elapsed, text, accept_len = generate_one(
            args.url, args.prompt, args.max_tokens
        )
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"  warmup {i+1}: {tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s"
              f" (accept_length={accept_len})")

    # Measured runs
    print("\n--- Measured Runs ---")
    all_tokens = []
    all_elapsed = []
    all_tps = []
    all_accept = []
    for i in range(args.runs):
        tokens, elapsed, text, accept_len = generate_one(
            args.url, args.prompt, args.max_tokens
        )
        tps = tokens / elapsed if elapsed > 0 else 0
        all_tokens.append(tokens)
        all_elapsed.append(elapsed)
        all_tps.append(tps)
        if accept_len is not None:
            all_accept.append(accept_len)
        print(f"  run {i+1}: {tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s"
              f" (accept_length={accept_len})")

    print("\n--- Summary ---")
    avg_tps = sum(all_tps) / len(all_tps)
    avg_tokens = sum(all_tokens) / len(all_tokens)
    avg_elapsed = sum(all_elapsed) / len(all_elapsed)
    min_tps = min(all_tps)
    max_tps = max(all_tps)
    print(f"  Avg tok/s: {avg_tps:.1f}")
    print(f"  Min tok/s: {min_tps:.1f}")
    print(f"  Max tok/s: {max_tps:.1f}")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Avg elapsed: {avg_elapsed:.2f}s")
    if all_accept:
        avg_accept = sum(all_accept) / len(all_accept)
        print(f"  Avg accept_length: {avg_accept:.2f}")
    print(f"\n  BASELINE (no-spec): 27 tok/s")
    if avg_tps > 27:
        print(f"  RESULT: EAGLE fast-verify BEATS baseline by {avg_tps - 27:.1f} tok/s ({(avg_tps/27 - 1)*100:.1f}% faster)")
    else:
        print(f"  RESULT: EAGLE fast-verify SLOWER than baseline by {27 - avg_tps:.1f} tok/s ({(1 - avg_tps/27)*100:.1f}% slower)")


if __name__ == "__main__":
    main()
