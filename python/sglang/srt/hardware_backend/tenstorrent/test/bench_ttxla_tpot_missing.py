"""Run TPOT benchmark for models not yet tested."""
import json, os, sys, time, traceback, random
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
except (ImportError, AttributeError):
    pass
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch, torch_xla, torch_xla.runtime as xr
xr.set_device_type("TT"); xr.use_spmd()
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import StaticCache

device = torch_xla.device()
N = xr.global_runtime_device_count()
print(f"device={device} n={N}", flush=True)

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "nvidia/Nemotron-Mini-4B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "stabilityai/stablelm-2-1_6b",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 64
NUM_DECODE = 20
SKIP_JIT = 2

# Load existing results
existing_path = "/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v145_ttxla_tpot_all_models.json"
try:
    with open(existing_path) as f:
        existing = json.load(f)
    results = existing["results"]
    done = {r["model_id"] for r in results if r["status"] == "PASS"}
    print(f"Already done: {len(done)} models", flush=True)
except:
    results = []
    done = set()

for model_id in MODELS:
    if model_id in done:
        print(f"\nSKIP {model_id} (already done)", flush=True)
        continue

    short = model_id.split("/")[-1]
    print(f"\n{'='*50}\n{short}", flush=True)
    r = {"model": short, "model_id": model_id, "status": "PENDING"}

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        r["arch"] = (config.architectures or ["?"])[0]
        r["hidden"] = config.hidden_size
        r["layers"] = config.num_hidden_layers

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
            use_cache=True, trust_remote_code=True)
        model.eval(); model = model.to(device)
        compiled = torch.compile(model, backend="tt")

        nkv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        hd = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)

        cache = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                            device="cpu", dtype=torch.bfloat16)
        cache.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                                   dtype=torch.bfloat16, device="cpu")
        for layer in cache.layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)

        attn_mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)
        random.seed(42)
        rand_ids = torch.tensor([[random.randint(10, config.vocab_size - 1) for _ in range(INPUT_LEN)]])
        cp = torch.arange(INPUT_LEN).to(device)
        pos = torch.arange(INPUT_LEN, dtype=torch.long).unsqueeze(0).to(device)
        mask_d = attn_mask.to(device)

        # Prefill
        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(input_ids=rand_ids.to(device), past_key_values=cache,
                          cache_position=cp, use_cache=True, attention_mask=mask_d,
                          position_ids=pos)
        torch_xla.sync()
        t1 = time.perf_counter()
        first_tok = out.logits[:, -1, :].to("cpu").argmax(-1).item()
        t2 = time.perf_counter()
        r["ttft_warmup_ms"] = round((t1 - t0) * 1000)
        r["ttft_logits_ms"] = round((t2 - t1) * 1000)
        print(f"  TTFT warmup: {r['ttft_warmup_ms']}ms + logits {r['ttft_logits_ms']}ms", flush=True)

        # Decode
        cur = first_tok; cache_pos = INPUT_LEN
        decode_times = []
        for i in range(NUM_DECODE):
            next_ids = torch.tensor([[cur]]).to(device)
            new_cp = torch.tensor([cache_pos]).to(device)
            new_pos = torch.tensor([[cache_pos]], dtype=torch.long).to(device)
            attn_mask[:, cache_pos] = 1
            md = attn_mask.to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = compiled(input_ids=next_ids, past_key_values=cache,
                              cache_position=new_cp, use_cache=True,
                              attention_mask=md, position_ids=new_pos)
            torch_xla.sync()
            t1 = time.perf_counter()
            cur = out.logits[:, -1, :].to("cpu").argmax(-1).item()
            t2 = time.perf_counter()
            cache_pos += 1
            decode_times.append((t2 - t0) * 1000)

        steady = decode_times[SKIP_JIT:]
        r["tpot_ms"] = round(sum(steady) / len(steady), 1)
        r["tok_s"] = round(1000 / r["tpot_ms"], 1)
        r["jit_decode_ms"] = [round(t, 0) for t in decode_times[:SKIP_JIT]]
        r["status"] = "PASS"
        print(f"  TPOT: {r['tpot_ms']}ms ({r['tok_s']} tok/s)", flush=True)

        del model, compiled, cache, out
        torch._dynamo.reset()
        import gc; gc.collect()

    except Exception as e:
        r["status"] = "FAIL"
        r["error"] = str(e)[:200]
        print(f"  FAIL: {e}", flush=True)
        torch._dynamo.reset()

    results.append(r)

# Save merged results
with open(existing_path, "w") as f:
    json.dump({"version": "v145", "input_len": INPUT_LEN, "decode": NUM_DECODE,
               "hw": f"2x P150a, {N} devices", "results": results}, f, indent=2)

# Print full table
print(f"\n{'='*80}")
print(f"{'Model':<25} {'Arch':<22} {'H':>5} {'L':>3} {'TTFT':>8} {'TPOT':>7} {'tok/s':>6} {'St':>4}")
print('-'*80)
for r in sorted(results, key=lambda x: x.get('tpot_ms', 9999)):
    print(f"{r['model']:<25} {r.get('arch','?'):<22} {r.get('hidden',''):>5} {r.get('layers',''):>3} "
          f"{str(r.get('ttft_warmup_ms','—'))+'ms':>8} {str(r.get('tpot_ms','—'))+'ms':>7} "
          f"{r.get('tok_s','—'):>6} {r['status']:>4}")
