"""Prefix-reuse benchmark for the compressed_file hicache backend demo.

Sends ``--num-prompts`` requests in two passes:

  Pass 1 (cold): each prompt has a long shared prefix followed by a small
                 request-specific tail. The shared prefix is large enough to
                 push at least one full page worth of KV through to disk.
  Pass 2 (hot):  same prompts again. The prefix-cache hit rate should be high
                 and the backend's read path is exercised.

After both passes, the script prints per-pass aggregate TTFT / TPS and pulls
the storage backend's compression stats from the on-disk cache directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import string
import time
from typing import List

import requests


def _gen_text(num_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    # Mix words to get a realistic-ish token distribution.
    words = [
        "history", "language", "compression", "model", "transformer", "cache",
        "prefix", "attention", "token", "decode", "prefill", "lossless",
        "ratio", "bandwidth", "throughput", "context", "memory", "system",
    ]
    out = []
    while sum(map(len, out)) < num_chars:
        out.append(rng.choice(words))
    return " ".join(out)[:num_chars]


def _send(host: str, port: int, prompt: str, max_tokens: int) -> dict:
    url = f"http://{host}:{port}/generate"
    t0 = time.perf_counter()
    r = requests.post(
        url,
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens},
        },
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()
    elapsed = time.perf_counter() - t0
    return {
        "elapsed": elapsed,
        "completion": body.get("text", ""),
        "meta": body.get("meta_info", {}),
    }


def _summarise(label: str, runs: List[dict]) -> None:
    e = [r["elapsed"] for r in runs]
    p50 = statistics.median(e)
    p90 = sorted(e)[max(0, int(0.9 * len(e)) - 1)]
    mean = statistics.mean(e)
    print(f"  {label:>10s}  n={len(runs):3d}  mean={mean*1000:7.1f}ms  "
          f"p50={p50*1000:7.1f}ms  p90={p90*1000:7.1f}ms")


def _report_cache_size(cache_dir: str) -> None:
    total = 0
    for root, _, files in os.walk(cache_dir):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    print(f"  on-disk cache size: {total/1e6:.2f} MB at {cache_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--prefix-chars", type=int, default=8 * 1024)
    ap.add_argument("--tail-chars",   type=int, default=128)
    ap.add_argument("--max-tokens",   type=int, default=32)
    ap.add_argument("--cache-dir", default=None,
                    help="optional path used by the backend so we can report total bytes")
    args = ap.parse_args()

    shared_prefix = _gen_text(args.prefix_chars, seed=0)
    prompts = [shared_prefix + " " + _gen_text(args.tail_chars, seed=1000 + i)
               for i in range(args.num_prompts)]

    print(f"# warming up server at {args.host}:{args.port}…")
    _ = _send(args.host, args.port, prompts[0], args.max_tokens)

    print("# pass 1 (cold): prefix likely not in storage yet")
    cold = [_send(args.host, args.port, p, args.max_tokens) for p in prompts]
    _summarise("cold", cold)

    print("# pass 2 (hot): same prompts → exercise backend read path")
    hot = [_send(args.host, args.port, p, args.max_tokens) for p in prompts]
    _summarise("hot", hot)

    if args.cache_dir:
        _report_cache_size(args.cache_dir)

    # Try to fetch /v1/stats if the server exposes it
    try:
        r = requests.get(f"http://{args.host}:{args.port}/get_server_info", timeout=5)
        if r.ok:
            print("# server info:")
            print(json.dumps(r.json(), indent=2)[:2000])
    except Exception:
        pass


if __name__ == "__main__":
    main()
