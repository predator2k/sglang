"""3-run server-alive TPOT bench across all v145 models.

For each model:
  1. Launch sglang.launch_server with TT-XLA backend (kept alive across all 3 runs).
  2. Wait for /health.
  3. Run 1 — random prompt A: cold JIT + cold KV cache.
  4. Run 2 — random prompt B (different): JIT warm, KV miss.
  5. Run 3 — same prompt B again: JIT warm, KV hit (radix prefix-cache hit).
  6. Tear down server.

Measures TTFT and TPOT for each of the 3 runs via direct HTTP streaming.

Prompts A and B are shared across all models (seeded random token IDs in a
universal vocab range [1024, 4096) chosen to fit every tokenizer in the sweep).

Output:
  - JSON fixture: _fixtures/v146_3run_server_<timestamp>.json
  - Per-model log dir: _fixtures/bench_3run_<timestamp>/<model>.{server,run1,run2,run3}.log

Usage:
  python3 bench_3run_server_alive.py [--models tiny] [--input-len 1024] [--output-len 1024]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

FIXTURES = Path(__file__).parent / "_fixtures"
V145 = FIXTURES / "v145_ttxla_tpot_all_models.json"

DEFAULT_INPUT_LEN = 1024
DEFAULT_OUTPUT_LEN = 1024
HEALTH_TIMEOUT_S = int(os.environ.get("BENCH_HEALTH_TIMEOUT_S", "900"))
STARTUP_GRACE_S = int(os.environ.get("BENCH_STARTUP_GRACE_S", "60"))

# Shared prompts: two 1K-token sequences in safe vocab range.
def make_token_seed(seed: int, n: int) -> list[int]:
    r = random.Random(seed)
    return [r.randint(1024, 4096) for _ in range(n)]

PROMPT_A_SEED = 42
PROMPT_B_SEED = 1729


def launch_server(model_id: str, port: int, log_path: Path, ctx_len: int,
                  backend: str = "tt_xla", hf_id: str | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["SGLANG_TT_EXECUTION_BACKEND"] = backend
    # Single request bench: bs=1
    env["SGLANG_TT_MAX_BATCH"] = "1"
    # Optionally bypass server-side pre-warm hook (set BYPASS_PREWARM=1)
    if os.environ.get("BYPASS_PREWARM"):
        env["SGLANG_TT_DISABLE_PREWARM"] = "1"
    # tt_transformers wants the HF name in HF_MODEL even though weights come from /models/<x>
    if hf_id:
        env["HF_MODEL"] = hf_id
    # Optional Tracy/tt-metal profiler. Set TT_PROFILE=1 in outer shell.
    if os.environ.get("TT_PROFILE") == "1":
        env["TT_METAL_DEVICE_PROFILER"] = "1"
        env["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
        env["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"
    # tt_transformers paged path uses sglang's native --device tenstorrent
    # (real radix cache, real KV pool). tt-xla path drives compile via
    # --device cpu (custom compile flow).
    sglang_device = "tenstorrent" if backend.startswith("tt_transformers") else "cpu"
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_id,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--device", sglang_device,
        "--context-length", str(ctx_len),
        "--mem-fraction-static", "0.5",
        "--disable-cuda-graph",
        "--tp-size", "1",
        "--skip-server-warmup",  # we drive our own warmup via run 1
        "--max-running-requests", "1",
    ]
    if backend == "tt_transformers_paged":
        cmd += ["--trust-remote-code", "--attention-backend", "torch_native"]
    f = log_path.open("w")
    f.write(f"# launching: {' '.join(cmd)}\n")
    f.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc


def wait_for_health(port: int, proc: subprocess.Popen, timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
    url = f"http://localhost:{port}/health"
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False


def stream_one(port: int, input_ids: list[int], output_len: int, log_path: Path) -> dict:
    """Send /generate with stream=True; measure TTFT and per-token times."""
    url = f"http://localhost:{port}/generate"
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": output_len,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
        "return_logprob": False,
    }
    f = log_path.open("w")
    f.write(f"# input_len={len(input_ids)} output_len={output_len}\n")
    f.flush()

    chunk_times: list[float] = []
    t0 = time.perf_counter()
    first_token_at: Optional[float] = None
    completion_text = ""
    n_chunks = 0
    err = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as r:
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                line = raw[len("data: "):].strip()
                if line == "[DONE]":
                    break
                now = time.perf_counter()
                chunk_times.append(now)
                n_chunks += 1
                if first_token_at is None:
                    first_token_at = now
                try:
                    chunk = json.loads(line)
                    completion_text = chunk.get("text", completion_text)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    t_end = time.perf_counter()
    ttft_ms = (first_token_at - t0) * 1000.0 if first_token_at else None
    if first_token_at is not None and len(chunk_times) > 1:
        intervals = [(chunk_times[i] - chunk_times[i - 1]) * 1000.0
                     for i in range(1, len(chunk_times))]
        tpot_ms_mean = sum(intervals) / len(intervals)
        tpot_ms_min = min(intervals)
        tpot_ms_max = max(intervals)
    else:
        tpot_ms_mean = tpot_ms_min = tpot_ms_max = None

    f.write(f"# ttft_ms={ttft_ms} tpot_mean={tpot_ms_mean} chunks={n_chunks}\n")
    f.write(f"# completion_first_120chars={completion_text[:120]!r}\n")
    f.write(f"# error={err}\n")
    f.close()

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms_mean": tpot_ms_mean,
        "tpot_ms_min": tpot_ms_min,
        "tpot_ms_max": tpot_ms_max,
        "n_chunks": n_chunks,
        "wall_s": t_end - t0,
        "error": err,
    }


def bench_one_model(model_id: str, model_label: str, args, logdir: Path,
                    prompt_a: list[int], prompt_b: list[int], port: int,
                    hf_id: str | None = None) -> dict:
    server_log = logdir / f"{model_label}.server.log"
    print(f"\n[{model_label}] launching server (log: {server_log.name})", flush=True)
    proc = launch_server(model_id, port, server_log, args.context_len, args.backend, hf_id)
    t0 = time.perf_counter()
    ready = wait_for_health(port, proc)
    t_ready = time.perf_counter() - t0
    if not ready:
        print(f"[{model_label}] FAIL: server did not become ready in {HEALTH_TIMEOUT_S}s", flush=True)
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return {
            "model": model_label, "model_id": model_id,
            "status": "FAIL_LAUNCH", "t_ready_s": t_ready,
            "run1": None, "run2": None, "run3": None,
        }
    print(f"[{model_label}] ready in {t_ready:.1f}s, sleeping {STARTUP_GRACE_S}s for pre-warm", flush=True)
    time.sleep(STARTUP_GRACE_S)

    results = {"model": model_label, "model_id": model_id, "status": "PASS",
               "t_ready_s": t_ready,
               "input_len": args.input_len, "output_len": args.output_len}
    try:
        for name, prompt, run_idx in [
            ("run1_cold", prompt_a, 1),
            ("run2_warm_new_prompt", prompt_b, 2),
            ("run3_warm_kv_hit", prompt_b, 3),
        ]:
            print(f"[{model_label}] {name}...", flush=True)
            log = logdir / f"{model_label}.{name}.log"
            r = stream_one(port, prompt[: args.input_len], args.output_len, log)
            results[name] = r
            if r["error"]:
                print(f"[{model_label}] {name} ERROR: {r['error']}", flush=True)
                results["status"] = "FAIL_RUN"
                break
            print(f"[{model_label}] {name}: ttft={r['ttft_ms']:.0f}ms "
                  f"tpot={r['tpot_ms_mean']:.1f}ms ({r['n_chunks']} chunks)", flush=True)
    finally:
        print(f"[{model_label}] stopping server", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-len", type=int, default=DEFAULT_INPUT_LEN)
    ap.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    ap.add_argument("--context-len", type=int, default=None,
                    help="server --context-length. Defaults to input_len + output_len (no buffer; many small models max at 2048).")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--models", default="all", help="all|tiny|<comma list of model_labels>")
    ap.add_argument("--out-tag", default=None, help="tag for output dir; default uses timestamp")
    ap.add_argument("--backend", default="tt_xla",
                    help="SGLANG_TT_EXECUTION_BACKEND: tt_xla (default) or tt_transformers_single")
    args = ap.parse_args()

    v145 = json.loads(V145.read_text())
    # tt_transformers_single requires LOCAL model path (/models/<name>),
    # not the HF id. tt-xla accepts HF id directly. We carry hf_id alongside
    # so we can set HF_MODEL env for tt_transformers config lookup.
    def _resolve_path(label: str, mid: str) -> str:
        if args.backend == "tt_transformers_single":
            return f"/models/{label}"
        return mid
    full = [(r["model"], _resolve_path(r["model"], r["model_id"]), r["model_id"])
            for r in v145["results"]]
    if args.models == "tiny":
        targets = [r for r in full if r[0] == "TinyLlama-1.1B-Chat-v1.0"]
    elif args.models == "all":
        targets = full
    else:
        wanted = set(args.models.split(","))
        targets = [r for r in full if r[0] in wanted]
    if not targets:
        raise SystemExit(f"no matching models for --models {args.models!r}")

    if args.context_len is None:
        args.context_len = args.input_len + args.output_len
    tag = args.out_tag or time.strftime("%Y%m%d_%H%M%S")
    logdir = FIXTURES / f"bench_3run_{tag}"
    logdir.mkdir(parents=True, exist_ok=True)
    out_json = FIXTURES / f"v146_3run_server_{tag}.json"
    print(f"[bench] {len(targets)} models, logdir={logdir}, out={out_json}", flush=True)

    prompt_a = make_token_seed(PROMPT_A_SEED, args.input_len)
    prompt_b = make_token_seed(PROMPT_B_SEED, args.input_len)
    assert prompt_a != prompt_b, "prompts must differ"

    all_results = {
        "version": "v146",
        "backend": args.backend,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "context_len": args.context_len,
        "hw": "2x P150a (post-B.2 canonical wheel)",
        "max_batch_size": 1,
        "tag": tag,
        "logdir": str(logdir),
        "results": [],
    }
    for label, mid, hf_id in targets:
        try:
            r = bench_one_model(mid, label, args, logdir, prompt_a, prompt_b, args.port, hf_id)
        except Exception as e:
            r = {"model": label, "model_id": mid, "status": "FAIL_EXC",
                 "error": f"{type(e).__name__}: {e}"}
            print(f"[{label}] EXCEPTION: {r['error']}", flush=True)
        all_results["results"].append(r)
        out_json.write_text(json.dumps(all_results, indent=2))
        print(f"[bench] checkpointed -> {out_json}", flush=True)
    print(f"\n[bench] DONE -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
