"""Test one model through tt-xla with eager attention fallback.

Usage:  python3 run_one_ttxla_model_v2.py <model_path> [family] [param_count]
Output: Single-line JSON on last line of stdout.
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
NUM_DECODE = 5
PROMPT = "The capital of France is"


def run(model_path: str, family: str = "unknown", param_count: str = "unknown"):
    xr.set_device_type("TT")
    xr.use_spmd()
    device = torch_xla.device()
    num_devices = xr.global_runtime_device_count()
    print(f"[init] device={device}, num_devices={num_devices}", flush=True)

    result = {
        "model_id": model_path,
        "family": family,
        "param_count": param_count,
        "local_path": model_path,
        "status": "PENDING",
        "prefill_s": None,
        "avg_decode_s": None,
        "decode_tok_s": None,
        "output_text": "",
        "prompt": PROMPT,
        "error": "",
        "decode_times": [],
        "architecture": "",
        "hidden_size": 0,
        "num_layers": 0,
        "attn_implementation": "",
    }

    try:
        # Load config first to check architecture
        print(f"[load] {model_path}...", flush=True)
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Try eager attention first (avoids SDPA issues on tt-xla)
        attn_impl = "eager"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                use_cache=True, trust_remote_code=True,
                attn_implementation="eager",
            )
            print(f"[load] Using eager attention", flush=True)
        except Exception as e1:
            print(f"[load] eager attention failed ({type(e1).__name__}: {e1}), trying default...", flush=True)
            attn_impl = "default"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                use_cache=True, trust_remote_code=True,
            )

        model.eval()
        config = model.config
        result["architecture"] = (config.architectures or ["?"])[0] if hasattr(config, "architectures") and config.architectures else "?"
        result["hidden_size"] = getattr(config, "hidden_size", 0)
        result["num_layers"] = getattr(config, "num_hidden_layers", 0)
        result["attn_implementation"] = attn_impl
        print(f"[load] {result['architecture']} h={result['hidden_size']} L={result['num_layers']} ({time.time()-t0:.1f}s)", flush=True)

        # Move + compile
        print("[compile] model.to(device) + torch.compile(backend='tt')...", flush=True)
        model = model.to(device)
        compiled = torch.compile(model, backend="tt")

        # StaticCache
        nkv = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", None))
        if nkv is None:
            nkv = config.num_attention_heads
        # Qwen3 sets head_dim explicitly (can differ from hidden_size/num_attention_heads)
        hd = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        print(f"[cache] StaticCache: kv_heads={nkv}, head_dim={hd}, max_len={MAX_CACHE_LEN}", flush=True)
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

        # Int32 mask
        attn_mask = torch.ones((1, MAX_CACHE_LEN), dtype=torch.int32)

        # Tokenize
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tok = tokenizer(PROMPT, return_tensors="pt", max_length=32,
                        padding="max_length", padding_side="left",
                        return_attention_mask=True)
        prompt_ids = tok.input_ids
        prompt_len = prompt_ids.shape[1]
        attn_mask[:, :prompt_len] = tok.attention_mask
        cp = torch.arange(0, prompt_len)

        ids_dev = prompt_ids.to(device)
        cp_dev = cp.to(device)
        mask_dev = attn_mask.to(device)

        # Prefill
        print("[prefill] Running...", flush=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(
                input_ids=ids_dev, past_key_values=cache,
                cache_position=cp_dev, use_cache=True,
                attention_mask=mask_dev,
            )
        t_prefill = time.perf_counter() - t0
        logits = out.logits.to("cpu")
        first_tok = logits[0, -1].argmax().item()
        first_word = tokenizer.decode(first_tok)
        print(f"[prefill] {t_prefill:.2f}s, first={first_word!r}", flush=True)
        result["prefill_s"] = round(t_prefill, 2)

        # Decode
        generated = [first_tok]
        cur = first_tok
        cache_pos = prompt_len
        decode_times = []

        for i in range(NUM_DECODE):
            next_ids = torch.tensor([[cur]]).to(device)
            new_cp = torch.tensor([cache_pos]).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = compiled(
                    input_ids=next_ids, past_key_values=cache,
                    cache_position=new_cp, use_cache=True,
                    attention_mask=mask_dev,
                )
            dt = time.perf_counter() - t0
            decode_times.append(round(dt, 4))

            logits = out.logits.to("cpu")
            cur = logits[0, -1].argmax().item()
            generated.append(cur)
            cache_pos += 1
            print(f"[decode {i}] {dt:.4f}s tok={tokenizer.decode(cur)!r}", flush=True)

        gen_text = tokenizer.decode(generated)
        full_output = PROMPT + gen_text
        avg_decode = sum(decode_times) / len(decode_times)
        tok_s = len(decode_times) / sum(decode_times) if sum(decode_times) > 0 else 0

        result["output_text"] = full_output
        result["decode_times"] = decode_times
        result["avg_decode_s"] = round(avg_decode, 4)
        result["decode_tok_s"] = round(tok_s, 2)
        result["status"] = "PASS"
        print(f"[PASS] {full_output!r} ({tok_s:.2f} tok/s)", flush=True)

    except Exception as e:
        err = str(e)
        if "out of memory" in err.lower() or "OOM" in err or "RESOURCE_EXHAUSTED" in err:
            result["status"] = "OOM"
        else:
            result["status"] = "FAIL"
        result["error"] = err[:500]
        print(f"[{result['status']}] {err[:300]}", flush=True)
        traceback.print_exc()

    # Final line: JSON
    print(f"\nRESULT_JSON:{json.dumps(result)}")
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/models/TinyLlama-1.1B-Chat-v1.0"
    fam = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    pc = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    run(path, fam, pc)
