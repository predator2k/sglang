#!/usr/bin/env python3
"""Long-context performance evaluation for Qwen3-8B on TT P300.

Tests with 4K/8K input and 4K/8K/16K output to measure throughput
at realistic workloads.
"""

import json
import time
import urllib.request
import sys

URL = "http://localhost:30000/v1/completions"
MODEL = "/models/Qwen3-8B"
HEADERS = {"Content-Type": "application/json"}


def generate_long_prompt(target_tokens: int) -> str:
    """Generate a prompt that's approximately target_tokens long."""
    base = (
        "Below is a detailed technical document about quantum computing. "
        "Please read it carefully and then provide a comprehensive summary "
        "with analysis.\n\n"
    )
    paragraph = (
        "Quantum computing represents a fundamentally different approach to "
        "computation that harnesses quantum mechanical phenomena such as "
        "superposition, entanglement, and quantum interference to process "
        "information in ways that classical computers cannot efficiently "
        "replicate. Unlike classical bits that exist in definite states of "
        "0 or 1, quantum bits or qubits can exist in superpositions of both "
        "states simultaneously, enabling quantum computers to explore many "
        "possible solutions in parallel. This parallelism, combined with "
        "quantum entanglement which creates correlations between qubits that "
        "have no classical analogue, gives quantum computers their "
        "extraordinary computational power for certain classes of problems. "
    )
    repeats = max(1, (target_tokens * 4) // len(paragraph))
    prompt = base + (paragraph * repeats)
    prompt += "\n\nNow provide an extremely detailed summary, analysis, and commentary on the above document. Cover every aspect thoroughly."
    return prompt


def bench(prompt: str, max_tokens: int, label: str) -> dict:
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()

    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data, HEADERS)
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    elapsed = time.perf_counter() - t0

    usage = resp["usage"]
    n_in = usage["prompt_tokens"]
    n_out = usage["completion_tokens"]
    tok_s = n_out / elapsed if elapsed > 0 else 0
    ttft_approx = elapsed * (n_in / (n_in + n_out)) if n_in + n_out > 0 else 0

    result = {
        "label": label,
        "input_tokens": n_in,
        "output_tokens": n_out,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(tok_s, 1),
        "ttft_approx_s": round(ttft_approx, 2),
        "text_preview": resp["choices"][0]["text"][:100],
    }
    print(f"  {label}: {n_in} in / {n_out} out / {elapsed:.1f}s / {tok_s:.1f} tok/s")
    return result


def main():
    print("=" * 60)
    print("Long-context performance evaluation — Qwen3-8B on 2×P150a")
    print("=" * 60)

    # Warmup
    print("\nWarmup...")
    data = json.dumps({"model": MODEL, "prompt": "Hello", "max_tokens": 10, "temperature": 0}).encode()
    for _ in range(2):
        req = urllib.request.Request(URL, data, HEADERS)
        with urllib.request.urlopen(req, timeout=300) as r:
            json.loads(r.read())

    results = []

    configs = [
        (128, 100, "short: 128in/100out"),
        (128, 4096, "short-in/4K-out"),
        (4096, 100, "4K-in/short-out"),
        (4096, 4096, "4K-in/4K-out"),
        (8192, 100, "8K-in/short-out"),
        (8192, 4096, "8K-in/4K-out"),
        (8192, 8192, "8K-in/8K-out"),
    ]

    for target_in, max_out, label in configs:
        print(f"\n--- {label} ---")
        prompt = generate_long_prompt(target_in)
        try:
            r = bench(prompt, max_out, label)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"label": label, "error": str(e)})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'In':>6} {'Out':>6} {'Time':>8} {'Tok/s':>8}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<25} {'ERROR':>6}")
        else:
            print(f"{r['label']:<25} {r['input_tokens']:>6} {r['output_tokens']:>6} {r['elapsed_s']:>7.1f}s {r['tok_s']:>7.1f}")

    out_path = "/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v87_acceptance_breakthrough/v124_long_context_perf.json"
    with open(out_path, "w") as f:
        json.dump({"date": "2026-05-16", "model": "Qwen3-8B", "device": "2xP150a", "results": results}, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
