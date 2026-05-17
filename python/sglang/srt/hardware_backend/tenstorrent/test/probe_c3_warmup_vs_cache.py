"""Diagnose: when request 2 is faster than request 1, is it JIT-cache or KV-cache?

The probe sends 3 long-prompt requests to one server, all 2048-input/16-output,
in sequence:
  R1: prompt A
  R2: prompt A (IDENTICAL) — should hit KV cache (if radix on) AND JIT cache
  R3: prompt B (same length, different content) — JIT cache hit, NO KV hit

For each:
  - TTFT (event 0 gap) — tells us if prefill ran or KV cache served it
  - first-decode latency (event 1 gap) — tells us if decode JIT compiled
  - steady-state latency (events 2+) — should be ~70ms
"""
import time, json, urllib.request

BASE = "http://localhost:30000"

def stream_generate(prompt: str, max_tokens: int):
    body = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0},
        "stream": True,
    }
    req = urllib.request.Request(
        f"{BASE}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    events = []
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
            delta = new_text[len(text_so_far):]
            text_so_far = new_text
            events.append({
                "elapsed_ms": (now - t0) * 1000,
                "since_last_ms": (now - last_t) * 1000,
                "delta_chars": len(delta),
            })
            last_t = now
    return events, text_so_far


def summarize(label, events):
    if not events:
        print(f"  {label}: no events")
        return
    ttft = events[0]["since_last_ms"]
    first_decode = events[1]["since_last_ms"] if len(events) > 1 else None
    steady = [e["since_last_ms"] for e in events[2:]]
    print(f"  {label}:")
    print(f"     event 0 (TTFT)            = {ttft:>8.1f} ms")
    if first_decode is not None:
        print(f"     event 1 (1st decode)      = {first_decode:>8.1f} ms")
    if steady:
        mn = min(steady); md = sorted(steady)[len(steady)//2]; mx = max(steady)
        print(f"     events 2+ ({len(steady)}) min/p50/max = {mn:.1f} / {md:.1f} / {mx:.1f} ms")


# Long prompts of identical length (~2048 input tokens worth)
# Use distinct word stems so tokenization length is similar but content differs
WORD_A = "test "        # 5 chars / ~1 token each
WORD_B = "data "        # same shape
PROMPT_A = (WORD_A * 2400)[:8000]
PROMPT_B = (WORD_B * 2400)[:8000]
MAX_TOKENS = 16

print(f"[warmup-test] sending small warmup so server is awake")
try:
    _, _ = stream_generate("Hi", 2)
except Exception as e:
    print(f"  warmup: {e}")

print()
print("[warmup-test] === R1: prompt A (cold-cache long shape) ===", flush=True)
ev_a1, _ = stream_generate(PROMPT_A, MAX_TOKENS)
summarize("R1 (prompt A, first time)", ev_a1)

print()
print("[warmup-test] === R2: prompt A AGAIN (KV cache + JIT cache test) ===", flush=True)
ev_a2, _ = stream_generate(PROMPT_A, MAX_TOKENS)
summarize("R2 (prompt A, same as R1)", ev_a2)

print()
print("[warmup-test] === R3: prompt B (same length, diff content — JIT cache only) ===", flush=True)
ev_b, _ = stream_generate(PROMPT_B, MAX_TOKENS)
summarize("R3 (prompt B, new content same shape)", ev_b)

print()
print("[warmup-test] === INTERPRETATION ===")
print("  If R2 TTFT << R1 TTFT:                    KV cache (radix) HIT — prefill skipped")
print("  If R2 event-1 << R1 event-1:               decode-JIT cache HIT")
print("  If R3 TTFT == R1 TTFT (slow):              KV cache MISS (different content)")
print("  If R3 TTFT << R1 TTFT (fast):              JIT cache HIT (same prefill bucket)")
print("  If R3 event-1 fast:                        decode-JIT cache HIT")
