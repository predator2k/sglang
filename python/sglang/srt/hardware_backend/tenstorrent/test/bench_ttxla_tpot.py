#!/usr/bin/env python3
"""
v142 -- TPOT benchmark for all tt-xla models on 2x P150a Blackhole.

Strategy: Run the PROVEN run_one_ttxla_model_v2.py pattern (single inference
per process, no cache reset) with 30 decode tokens instead of 5, to capture
steady-state TPOT after JIT warmup.

The host orchestrator (run_ttxla_tpot_bench.sh) handles:
  - Device reset between each model
  - Fresh container per model
  - Aggregating results into v142_ttxla_tpot_benchmark.json

Usage:
  python3 bench_ttxla_tpot.py <model_path> [model_name]

Output: JSON on stdout (RESULT_JSON: prefix on last line)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

MAX_CACHE_LEN = 64
PAD_LEN = 32
NUM_DECODE = 30  # Generate 30 tokens to measure steady-state TPOT

PROMPT = "The capital of France is"


def run(model_path: str, model_name: str = "unknown"):
    xr.set_device_type("TT")
    xr.use_spmd()
    device = torch_xla.device()
    ndev = xr.global_runtime_device_count()
    print(f"[init] device={device}, ndev={ndev}", flush=True)

    result = {
        "model_name": model_name,
        "model_path": model_path,
        "status": "PENDING",
        "prompt": PROMPT,
        "num_decode": NUM_DECODE,
        "max_cache_len": MAX_CACHE_LEN,
        "pad_len": PAD_LEN,
        "num_devices": ndev,
    }

    try:
        # Load
        print(f"[load] {model_path}", flush=True)
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        attn_impl = "eager"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, use_cache=True,
                trust_remote_code=True, attn_implementation="eager",
            )
        except Exception:
            attn_impl = "default"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, use_cache=True,
                trust_remote_code=True,
            )

        model.eval()
        config = model.config
        result["architecture"] = (config.architectures or ["?"])[0] if hasattr(config, "architectures") and config.architectures else "?"
        result["hidden_size"] = getattr(config, "hidden_size", 0)
        result["num_layers"] = getattr(config, "num_hidden_layers", 0)
        result["attn_implementation"] = attn_impl
        load_time = time.time() - t0
        result["load_time_s"] = round(load_time, 2)
        print(f"[load] {result['architecture']} h={result['hidden_size']} L={result['num_layers']} attn={attn_impl} ({load_time:.1f}s)", flush=True)

        # Compile
        print("[compile] to_device + torch.compile...", flush=True)
        t0 = time.perf_counter()
        model = model.to(device)
        compiled = torch.compile(model, backend="tt")
        compile_time = time.perf_counter() - t0
        result["compile_time_s"] = round(compile_time, 2)
        print(f"[compile] {compile_time:.1f}s", flush=True)

        # StaticCache on CPU then move to device
        nkv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        hd = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        print(f"[cache] kv_heads={nkv} head_dim={hd} max_len={MAX_CACHE_LEN}", flush=True)
        cache = StaticCache(
            config=config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN,
            device="cpu", dtype=torch.bfloat16,
        )
        cache.early_initialization(
            batch_size=1, num_heads=nkv, head_dim=hd,
            dtype=torch.bfloat16, device="cpu",
        )
        for layer in cache.layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)

        # Tokenize
        attn_mask = torch.ones((1, MAX_CACHE_LEN), dtype=torch.int32)
        tok = tokenizer(PROMPT, return_tensors="pt", max_length=PAD_LEN,
                        padding="max_length", padding_side="left",
                        return_attention_mask=True)
        prompt_ids = tok.input_ids
        plen = prompt_ids.shape[1]
        attn_mask[:, :plen] = tok.attention_mask
        cp = torch.arange(0, plen)
        result["prompt_tokens_padded"] = plen

        ids_dev = prompt_ids.to(device)
        cp_dev = cp.to(device)
        mask_dev = attn_mask.to(device)

        # Prefill
        print("[prefill]...", flush=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(input_ids=ids_dev, past_key_values=cache,
                           cache_position=cp_dev, use_cache=True,
                           attention_mask=mask_dev)
        t_prefill = time.perf_counter() - t0
        logits = out.logits.to("cpu")
        first_tok = logits[0, -1].argmax().item()
        first_word = tokenizer.decode(first_tok)
        result["prefill_s"] = round(t_prefill, 4)
        print(f"[prefill] {t_prefill:.2f}s first={first_word!r}", flush=True)

        # Decode loop
        generated = [first_tok]
        cur = first_tok
        cache_pos = plen
        decode_times = []

        max_decode = min(NUM_DECODE, MAX_CACHE_LEN - plen - 1)
        print(f"[decode] generating {max_decode} tokens...", flush=True)

        for i in range(max_decode):
            next_ids = torch.tensor([[cur]]).to(device)
            new_cp = torch.tensor([cache_pos]).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = compiled(input_ids=next_ids, past_key_values=cache,
                               cache_position=new_cp, use_cache=True,
                               attention_mask=mask_dev)
            dt = time.perf_counter() - t0
            decode_times.append(round(dt, 4))

            logits = out.logits.to("cpu")
            cur = logits[0, -1].argmax().item()
            generated.append(cur)
            cache_pos += 1

            if i < 3 or i == max_decode - 1:
                print(f"[decode {i}] {dt:.4f}s tok={tokenizer.decode(cur)!r}", flush=True)
            elif i == 3:
                print(f"[decode ...] (skipping intermediate output)", flush=True)

        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        full_output = PROMPT + gen_text

        result["output_text"] = full_output[:200]
        result["decode_times"] = decode_times
        result["total_decode_tokens"] = len(decode_times)

        # Compute TPOT metrics
        if decode_times:
            result["first_decode_s"] = decode_times[0]
            result["avg_decode_s"] = round(sum(decode_times) / len(decode_times), 4)

            if len(decode_times) > 1:
                # Steady-state: all decode steps AFTER the first (which includes JIT)
                steady = decode_times[1:]
                result["steady_state_tpot_s"] = round(sum(steady) / len(steady), 6)
                result["steady_state_tpot_ms"] = round(1000 * sum(steady) / len(steady), 3)
                result["steady_state_tok_s"] = round(len(steady) / sum(steady), 1)
                result["steady_min_ms"] = round(1000 * min(steady), 3)
                result["steady_max_ms"] = round(1000 * max(steady), 3)
                result["steady_p50_ms"] = round(1000 * sorted(steady)[len(steady) // 2], 3)

                # Also compute excluding first 2 (in case second also has JIT)
                if len(decode_times) > 2:
                    ultra_steady = decode_times[2:]
                    result["ultra_steady_tpot_ms"] = round(1000 * sum(ultra_steady) / len(ultra_steady), 3)

            result["jit_overhead_s"] = round(decode_times[0] - (sum(decode_times[1:]) / max(len(decode_times) - 1, 1)), 4) if len(decode_times) > 1 else None

        result["status"] = "PASS"
        print(f"[PASS] {full_output[:80]!r}", flush=True)
        if result.get("steady_state_tpot_ms"):
            print(f"[TPOT] steady={result['steady_state_tpot_ms']:.3f}ms  "
                  f"first_decode={result['first_decode_s']}s  "
                  f"range=[{result.get('steady_min_ms'):.3f},{result.get('steady_max_ms'):.3f}]ms", flush=True)

    except Exception as e:
        err = str(e)
        if "out of memory" in err.lower() or "OOM" in err or "RESOURCE_EXHAUSTED" in err:
            result["status"] = "OOM"
        else:
            result["status"] = "FAIL"
        result["error"] = err[:500]
        print(f"[{result['status']}] {err[:200]}", flush=True)
        traceback.print_exc()

    print(f"\nRESULT_JSON:{json.dumps(result)}", flush=True)
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/models/TinyLlama-1.1B-Chat-v1.0"
    name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(path)
    run(path, name)
