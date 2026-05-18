"""Per-op decode profile via ttnn.profiler.get_all_programs_perf_data().

Loads a target model, does prefill + N decode steps, aggregates per-program
(per-op) wall time, and reports the top-N ops by total time.

Profiler env vars are set BEFORE any tt-metal/ttnn import so the profiler
captures from process start.

Usage:
  python3 probe_decode_op_profile.py --model Qwen/Qwen3-8B --input-len 1024 --decode 20

Output:
  - Stdout: top-30 ops table (op_name, count, total_ms, mean_us, % of decode time)
  - JSON: _fixtures/decode_op_profile_<tag>.json
"""
import os
# Profiler env vars must be set BEFORE importing ttnn/torch_xla.
if os.environ.get("PROBE_NO_PROFILER") != "1":
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
    os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
except (ImportError, AttributeError):
    pass

import torch
import torch_xla
import torch_xla.runtime as xr
xr.set_device_type("TT"); xr.use_spmd()
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import StaticCache

FIXTURES = Path(__file__).parent / "_fixtures"


def get_profiler():
    import ttnn  # noqa
    return ttnn._ttnn.profiler


def fetch_perf_snapshot():
    """Returns the latest per-program perf data. Shape varies by build."""
    prof = get_profiler()
    try:
        return prof.get_all_programs_perf_data()
    except Exception:
        return prof.get_latest_programs_perf_data()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=20)
    ap.add_argument("--skip-warmup", type=int, default=5,
                    help="drop the first N decode steps (JIT cliffs)")
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--weight-dtype", default="",
                    help="tt-xla experimental_weight_dtype: '' (default bf16), 'bfp_bf8', 'bfp_bf4'")
    args = ap.parse_args()

    tag = args.out_tag or args.model.split("/")[-1].replace("-", "_").lower()
    out_json = FIXTURES / f"decode_op_profile_{tag}.json"
    print(f"[profile] model={args.model} input_len={args.input_len} decode={args.decode}", flush=True)
    print(f"[profile] writing {out_json}", flush=True)

    if args.weight_dtype:
        torch_xla.set_custom_compile_options({"experimental_weight_dtype": args.weight_dtype})
        print(f"[profile] set experimental_weight_dtype={args.weight_dtype}", flush=True)
    device = torch_xla.device()
    config = AutoConfig.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="eager", use_cache=True
    )
    base.eval()
    base = base.to(device)
    n_kv = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    cache_size = args.input_len + args.decode + 32

    cache = StaticCache(config=config, max_batch_size=1, max_cache_len=cache_size,
                        device="cpu", dtype=torch.bfloat16)
    cache.early_initialization(batch_size=1, num_heads=n_kv, head_dim=head_dim,
                               dtype=torch.bfloat16, device="cpu")
    for layer in cache.layers:
        layer.keys = layer.keys.to(device); layer.values = layer.values.to(device)

    random.seed(42)
    rand_ids = torch.tensor([[random.randint(10, config.vocab_size - 1)
                              for _ in range(args.input_len)]])
    attn_mask = torch.ones((1, cache_size), dtype=torch.int32)
    compiled = torch.compile(base, backend="tt")

    print("[profile] prefill...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = compiled(
            input_ids=rand_ids.to(device), past_key_values=cache,
            cache_position=torch.arange(args.input_len).to(device),
            use_cache=True, attention_mask=attn_mask.to(device),
            position_ids=torch.arange(args.input_len, dtype=torch.long).unsqueeze(0).to(device),
        )
    torch_xla.sync()
    cur = out.logits[:, -1, :].argmax(-1).to("cpu").item()
    print(f"[profile] prefill done in {time.perf_counter()-t0:.1f}s, first_tok={cur}", flush=True)

    # Snapshot baseline (everything before decode loop).
    pre_decode_snapshot = fetch_perf_snapshot()
    n_pre = len(pre_decode_snapshot) if hasattr(pre_decode_snapshot, "__len__") else 0
    print(f"[profile] pre-decode perf records: {n_pre}", flush=True)

    times = []
    decode_pos = args.input_len
    for k in range(args.decode):
        attn_mask[:, decode_pos] = 1
        t_in = torch.tensor([[cur]]).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(
                input_ids=t_in, past_key_values=cache,
                cache_position=torch.tensor([decode_pos]).to(device),
                use_cache=True, attention_mask=attn_mask.to(device),
                position_ids=torch.tensor([[decode_pos]], dtype=torch.long).to(device),
            )
        cur = out.logits[:, -1, :].argmax(-1).to("cpu").item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        decode_pos += 1
        if k < 5 or k % 5 == 0:
            print(f"[profile] decode k={k} took {times[-1]:.1f}ms", flush=True)

    steady = times[args.skip_warmup:]
    mean = sum(steady) / len(steady)
    print(f"[profile] steady decode mean: {mean:.1f}ms ({1000/mean:.1f} tok/s)", flush=True)

    # Fetch final snapshot and diff.
    post_snapshot = fetch_perf_snapshot()
    n_post = len(post_snapshot) if hasattr(post_snapshot, "__len__") else 0
    print(f"[profile] post-decode perf records: {n_post}", flush=True)

    # Aggregate: many possible shapes. Inspect.
    new_records = post_snapshot[n_pre:] if hasattr(post_snapshot, "__getitem__") else post_snapshot
    print(f"[profile] new records during decode: {len(new_records) if hasattr(new_records, '__len__') else '?'}", flush=True)
    if new_records and hasattr(new_records, "__getitem__"):
        sample = new_records[0]
        print(f"[profile] sample record type: {type(sample).__name__}", flush=True)
        if hasattr(sample, "_fields"):
            print(f"[profile] sample fields: {sample._fields}", flush=True)
        elif hasattr(sample, "keys"):
            print(f"[profile] sample keys: {list(sample.keys())[:10]}", flush=True)

    # Aggregation: try multiple field-name conventions.
    by_op = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for rec in new_records:
        name = None; us = 0.0
        for nameattr in ("operation_name", "op_name", "name", "kernel_name"):
            if hasattr(rec, nameattr):
                name = getattr(rec, nameattr)
                break
            if isinstance(rec, dict) and nameattr in rec:
                name = rec[nameattr]
                break
        for timeattr in ("device_duration_us", "duration_us", "exec_time_us", "elapsed_us", "wall_us"):
            if hasattr(rec, timeattr):
                us = float(getattr(rec, timeattr))
                break
            if isinstance(rec, dict) and timeattr in rec:
                us = float(rec[timeattr])
                break
        if name is None:
            name = "unknown"
        by_op[name]["count"] += 1
        by_op[name]["total_us"] += us

    sorted_ops = sorted(by_op.items(), key=lambda kv: -kv[1]["total_us"])
    decode_total_ms = sum(steady)
    print(f"\n[profile] TOP-30 OPS BY TOTAL TIME (over {len(steady)} steady decode steps)")
    print(f"{'op_name':<40} {'count':>8} {'total_ms':>10} {'mean_us':>10} {'%':>6}")
    print("-" * 80)
    for name, stats in sorted_ops[:30]:
        total_ms = stats["total_us"] / 1000
        mean_us = stats["total_us"] / max(1, stats["count"])
        pct = total_ms / decode_total_ms * 100 if decode_total_ms else 0
        print(f"{name:<40} {stats['count']:>8} {total_ms:>9.1f}  {mean_us:>9.1f}  {pct:>5.1f}")

    out_data = {
        "model": args.model,
        "input_len": args.input_len,
        "decode": args.decode,
        "skip_warmup": args.skip_warmup,
        "steady_decode_mean_ms": mean,
        "tok_per_s": 1000 / mean,
        "decode_step_times_ms": times,
        "top_ops": [
            {"name": name, "count": stats["count"], "total_us": stats["total_us"]}
            for name, stats in sorted_ops
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"\n[profile] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
