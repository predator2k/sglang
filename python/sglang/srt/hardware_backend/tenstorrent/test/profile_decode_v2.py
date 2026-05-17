"""Profile tt-xla decode: 2K input, 100 decode, two runs (JIT warmup + JIT hit)."""
import os, time
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
except (ImportError, AttributeError):
    pass
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch, torch_xla, torch_xla.runtime as xr
xr.set_device_type("TT"); xr.use_spmd()
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 512
NUM_DECODE = 500

device = torch_xla.device()
print(f"device={device}, n={xr.global_runtime_device_count()}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True)
model.eval(); model = model.to(device)
compiled = torch.compile(model, backend="tt")
config = model.config
nkv, hd = config.num_key_value_heads, config.hidden_size // config.num_attention_heads

def make_cache():
    c = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    return c

import random

def run_pass(pass_name, seed):
    random.seed(seed)
    rand_ids = [random.randint(100, 31999) for _ in range(INPUT_LEN)]
    prompt_text = tokenizer.decode(rand_ids)
    tok = tokenizer(prompt_text, return_tensors="pt", max_length=INPUT_LEN, truncation=True)
    ids = tok.input_ids
    seq_len = ids.shape[1]

    cache = make_cache()
    attn_mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)

    # Prefill
    t_pf0 = time.perf_counter()
    with torch.no_grad():
        out = compiled(input_ids=ids.to(device), past_key_values=cache,
                       cache_position=torch.arange(seq_len).to(device),
                       use_cache=True, attention_mask=attn_mask.to(device),
                       position_ids=torch.arange(seq_len, dtype=torch.long).unsqueeze(0).to(device))
    torch_xla.sync()
    t_pf1 = time.perf_counter()
    first_tok = out.logits[:, -1, :].to("cpu").argmax(-1).item()
    t_pf2 = time.perf_counter()
    print(f"  [{pass_name}] Prefill {seq_len} tok: exec={1000*(t_pf1-t_pf0):.0f}ms copy={1000*(t_pf2-t_pf1):.0f}ms", flush=True)

    # Decode
    cur = first_tok; cache_pos = seq_len
    timings = []
    for i in range(NUM_DECODE):
        next_ids = torch.tensor([[cur]]).to(device)
        new_cp = torch.tensor([cache_pos]).to(device)
        new_pos = torch.tensor([[cache_pos]], dtype=torch.long).to(device)
        attn_mask[:, cache_pos] = 1
        mask_d = attn_mask.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(input_ids=next_ids, past_key_values=cache,
                          cache_position=new_cp, use_cache=True,
                          attention_mask=mask_d, position_ids=new_pos)
        t1 = time.perf_counter()
        torch_xla.sync()
        t2 = time.perf_counter()
        cur = out.logits[:, -1, :].to("cpu").argmax(-1).item()
        t3 = time.perf_counter()

        cache_pos += 1
        timings.append(((t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t3-t0)*1000))

    # Report (skip first 2 = JIT compile for decode shape)
    steady = timings[2:]
    for idx, label in enumerate(["fwd_queue", "xla_sync", "logits_cp", "TOTAL"]):
        vals = [t[idx] for t in steady]
        avg = sum(vals)/len(vals)
        print(f"  [{pass_name}] {label:>10}: avg={avg:.2f}ms  min={min(vals):.2f}  max={max(vals):.2f}", flush=True)
    total_avg = sum(t[3] for t in steady)/len(steady)
    print(f"  [{pass_name}] {NUM_DECODE} decode @ {total_avg:.1f} ms/tok = {1000/total_avg:.1f} tok/s", flush=True)
    return total_avg

print(f"\n=== RUN 1: JIT WARMUP (seed=42) ===", flush=True)
tpot1 = run_pass("WARMUP", 42)

print(f"\n=== RUN 2: JIT HIT (seed=123) ===", flush=True)
tpot2 = run_pass("JIT_HIT", 123)

print(f"\n=== SUMMARY: {INPUT_LEN} input, {NUM_DECODE} decode, {MODEL} ===")
print(f"  WARMUP: {tpot1:.1f} ms/tok ({1000/tpot1:.1f} tok/s)")
print(f"  JIT_HIT: {tpot2:.1f} ms/tok ({1000/tpot2:.1f} tok/s)")
print(f"  Speedup: {tpot1/tpot2:.2f}x")
