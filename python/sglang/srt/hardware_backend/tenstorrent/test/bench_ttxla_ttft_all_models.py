"""Benchmark TTFT (prefill) for all tt-xla models on 2×P150a.

For each model: load, compile, prefill 2K tokens, report TTFT for
JIT warmup (first run) and JIT hit (second run).

Usage: python3 bench_ttxla_ttft_all_models.py
"""
import json, os, sys, time, traceback
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
except (ImportError, AttributeError):
    pass
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch, torch_xla, torch_xla.runtime as xr
xr.set_device_type("TT"); xr.use_spmd()
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache

device = torch_xla.device()
NUM_DEVICES = xr.global_runtime_device_count()
print(f"device={device}, n={NUM_DEVICES}", flush=True)

MODELS = [
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
    ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
    ("meta-llama/Llama-3.2-3B", "Llama-3.2-3B"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ("Qwen/Qwen3-4B", "Qwen3-4B"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("nvidia/Nemotron-Mini-4B-Instruct", "Nemotron-4B"),
    ("microsoft/Phi-4-mini-instruct", "Phi-4-mini"),
    ("stabilityai/stablelm-2-1_6b", "StableLM-1.6B"),
    ("google/gemma-3-1b-it", "Gemma3-1B"),
]

INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 64
NUM_DECODE = 5
results = []

for model_id, short_name in MODELS:
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {short_name} ({model_id})", flush=True)
    result = {"model": short_name, "model_id": model_id, "status": "PENDING"}

    try:
        # Load
        t_load0 = time.time()
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        arch = (config.architectures or ["?"])[0] if hasattr(config, "architectures") else "?"
        result["architecture"] = arch
        result["params_M"] = round(sum(
            p.numel() for p in AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
                use_cache=True, trust_remote_code=True
            ).parameters()
        ) / 1e6, 1) if False else 0  # skip param count for speed

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
            use_cache=True, trust_remote_code=True)
        model.eval()
        t_load1 = time.time()
        result["load_time_s"] = round(t_load1 - t_load0, 1)
        print(f"  Loaded in {result['load_time_s']}s, arch={arch}", flush=True)

        # Move + compile
        model = model.to(device)
        compiled = torch.compile(model, backend="tt")
        nkv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        hd = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)

        # Cache
        cache = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                            device="cpu", dtype=torch.bfloat16)
        cache.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                                   dtype=torch.bfloat16, device="cpu")
        for layer in cache.layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)

        attn_mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)

        # Generate random input
        import random; random.seed(42)
        vocab_size = config.vocab_size
        rand_ids = torch.tensor([[random.randint(10, vocab_size - 1) for _ in range(INPUT_LEN)]])

        # === Run 1: JIT warmup ===
        cp = torch.arange(INPUT_LEN).to(device)
        pos = torch.arange(INPUT_LEN, dtype=torch.long).unsqueeze(0).to(device)
        mask_d = attn_mask.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(input_ids=rand_ids.to(device), past_key_values=cache,
                          cache_position=cp, use_cache=True, attention_mask=mask_d,
                          position_ids=pos)
        torch_xla.sync()
        t1 = time.perf_counter()
        _ = out.logits[:, -1, :].to("cpu").argmax(-1).item()
        t2 = time.perf_counter()

        ttft_warmup = (t1 - t0) * 1000
        result["ttft_warmup_ms"] = round(ttft_warmup, 0)
        print(f"  TTFT warmup: {ttft_warmup:.0f}ms", flush=True)

        # === Run 2: JIT hit (fresh cache, same shape) ===
        cache2 = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                             device="cpu", dtype=torch.bfloat16)
        cache2.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                                    dtype=torch.bfloat16, device="cpu")
        for layer in cache2.layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)

        random.seed(123)
        rand_ids2 = torch.tensor([[random.randint(10, vocab_size - 1) for _ in range(INPUT_LEN)]])

        t0 = time.perf_counter()
        with torch.no_grad():
            out2 = compiled(input_ids=rand_ids2.to(device), past_key_values=cache2,
                           cache_position=cp, use_cache=True, attention_mask=mask_d,
                           position_ids=pos)
        torch_xla.sync()
        t1 = time.perf_counter()
        _ = out2.logits[:, -1, :].to("cpu").argmax(-1).item()
        t2 = time.perf_counter()

        ttft_hit = (t1 - t0) * 1000
        result["ttft_jit_hit_ms"] = round(ttft_hit, 0)
        result["speedup"] = round(ttft_warmup / ttft_hit, 1) if ttft_hit > 0 else 0
        result["status"] = "PASS"
        print(f"  TTFT JIT hit: {ttft_hit:.0f}ms (speedup: {result['speedup']}x)", flush=True)

        # Cleanup
        del model, compiled, cache, cache2, out, out2
        torch._dynamo.reset()

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)[:200]
        print(f"  FAIL: {e}", flush=True)
        traceback.print_exc()
        torch._dynamo.reset()

    results.append(result)

# Summary
print(f"\n{'='*60}")
print(f"{'Model':<20} {'Arch':<25} {'TTFT warm':>10} {'TTFT hit':>10} {'Speedup':>8} {'Status':>6}")
print(f"{'-'*20} {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
for r in results:
    print(f"{r['model']:<20} {r.get('architecture','?'):<25} "
          f"{r.get('ttft_warmup_ms','—'):>9}{'ms' if 'ttft_warmup_ms' in r else ''} "
          f"{r.get('ttft_jit_hit_ms','—'):>9}{'ms' if 'ttft_jit_hit_ms' in r else ''} "
          f"{r.get('speedup','—'):>7}{'x' if 'speedup' in r else ''} "
          f"{r['status']:>6}")

# Save JSON
out_path = "/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v145_ttxla_ttft_all_models.json"
with open(out_path, "w") as f:
    json.dump({"version": "v145", "input_len": INPUT_LEN, "hw": f"2x P150a, {NUM_DEVICES} devices",
               "results": results}, f, indent=2)
print(f"\nSaved: {out_path}")
