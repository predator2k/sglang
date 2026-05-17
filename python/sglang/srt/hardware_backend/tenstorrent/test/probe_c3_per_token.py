"""Profile per-token decode latencies via direct streaming HTTP call.

Hits /v1/chat/completions (or /generate) with stream=True so we can record
the wall-clock time between successive SSE events — that's the real
per-token latency from the client's perspective.

Standalone profile says steady-state 59-68ms.
bench_serving mean TPOT for TinyLlama 2K = 278ms, but median ITL = 70ms.
Goal: find which tokens are >100ms — confirm they're early ones (JIT compile).
"""
import time
import json
import urllib.request
import urllib.error
import sys

BASE = "http://localhost:30000"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_INPUT_TOKENS = 2048
NUM_OUTPUT_TOKENS = 64


def stream_generate(prompt: str, max_tokens: int):
    """Stream tokens, record per-event timing."""
    body = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
        },
        "stream": True,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    timings = []
    last_t = t0
    text_so_far = ""
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            now = time.perf_counter()
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            new_text = obj.get("text", "")
            # delta = new text since last event
            delta = new_text[len(text_so_far):]
            text_so_far = new_text
            timings.append({
                "elapsed_ms": (now - t0) * 1000,
                "since_last_ms": (now - last_t) * 1000,
                "delta_chars": len(delta),
            })
            last_t = now
    return timings, text_so_far


def main():
    # Build a long prompt — 2048 tokens worth (~8000 chars for English)
    # Use a single repeated word to make it predictable.
    word = "test "  # 5 chars / ~1 token each
    prompt = (word * 2400)[:8000]

    print(f"[c3-per-token] sending prompt with ~{len(prompt)//5} tokens")
    print(f"[c3-per-token] streaming up to {NUM_OUTPUT_TOKENS} tokens", flush=True)

    # Warmup request (small)
    print("\n[c3-per-token] === WARMUP request (short, just to get past TTFT JIT) ===", flush=True)
    try:
        wt, wtxt = stream_generate("Hello", 8)
        print(f"  warmup events: {len(wt)}, total time: {wt[-1]['elapsed_ms']:.0f}ms")
    except Exception as e:
        print(f"  warmup FAIL: {e}")

    # Real request
    print("\n[c3-per-token] === MEASUREMENT request ===", flush=True)
    try:
        events, final_text = stream_generate(prompt, NUM_OUTPUT_TOKENS)
    except Exception as e:
        print(f"  request FAIL: {e}")
        return

    print(f"\n[c3-per-token] total events: {len(events)}")
    print(f"[c3-per-token] final text length: {len(final_text)} chars (~{len(final_text)//5} tokens)")
    print(f"[c3-per-token] total time: {events[-1]['elapsed_ms']:.0f}ms")

    print("\n[c3-per-token] per-event breakdown:")
    print(f"{'idx':>4} {'elapsed_ms':>12} {'since_last_ms':>14} {'chars':>6}")
    for i, ev in enumerate(events):
        marker = "  <-- SLOW" if ev["since_last_ms"] > 200 else ""
        print(f"{i:>4} {ev['elapsed_ms']:>12.1f} {ev['since_last_ms']:>14.1f} {ev['delta_chars']:>6}{marker}")

    # Histogram of inter-token gaps
    gaps = [e["since_last_ms"] for e in events[1:]]  # skip first (TTFT)
    if gaps:
        gaps_sorted = sorted(gaps)
        n = len(gaps_sorted)
        print(f"\n[c3-per-token] inter-token-latency stats (n={n}):")
        print(f"  min  = {gaps_sorted[0]:>8.1f}ms")
        print(f"  p50  = {gaps_sorted[n//2]:>8.1f}ms")
        print(f"  p90  = {gaps_sorted[int(n*0.9)]:>8.1f}ms")
        print(f"  p99  = {gaps_sorted[min(n-1, int(n*0.99))]:>8.1f}ms")
        print(f"  max  = {gaps_sorted[-1]:>8.1f}ms")
        print(f"  mean = {sum(gaps)/n:>8.1f}ms")
        # Find slow events (>200ms)
        slow = [(i+1, e) for i, e in enumerate(events[1:]) if e["since_last_ms"] > 200]
        print(f"\n  slow events (>200ms): {len(slow)} of {n}")
        for i, e in slow[:10]:
            print(f"    event {i}: elapsed={e['elapsed_ms']:.0f}ms, gap={e['since_last_ms']:.0f}ms")


if __name__ == "__main__":
    main()
