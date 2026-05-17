"""Profile decode overhead: where does the 222ms TPOT come from when model fwd is 8ms?"""
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
CACHE_SIZE = 2048
PROMPT = "The capital of France is"
NUM_DECODE = 50

device = torch_xla.device()
print(f"device={device}, n={xr.global_runtime_device_count()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True)
model.eval(); model = model.to(device)
compiled = torch.compile(model, backend="tt")
config = model.config
nkv, hd = config.num_key_value_heads, config.hidden_size // config.num_attention_heads

cache = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
cache.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
for layer in cache.layers:
    layer.keys = layer.keys.to(device)
    layer.values = layer.values.to(device)

attn_mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)

# Prefill
tok = tokenizer(PROMPT, return_tensors="pt")
input_ids = tok.input_ids
seq_len = input_ids.shape[1]
ids_dev = input_ids.to(device)
cp = torch.arange(0, seq_len).to(device)
pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(device)
mask_dev = attn_mask.to(device)

t0 = time.perf_counter()
with torch.no_grad():
    out = compiled(input_ids=ids_dev, past_key_values=cache, cache_position=cp,
                   use_cache=True, attention_mask=mask_dev, position_ids=pos)
t_prefill = time.perf_counter() - t0
logits = out.logits.to("cpu")
first_tok = logits[0, -1].argmax().item()
print(f"Prefill: {t_prefill:.2f}s, first={tokenizer.decode(first_tok)!r}")

# Decode with timing breakdown
cur = first_tok
cache_pos = seq_len
generated = [first_tok]
timings = []

for i in range(NUM_DECODE):
    t0 = time.perf_counter()

    next_ids = torch.tensor([[cur]]).to(device)
    new_cp = torch.tensor([cache_pos]).to(device)
    new_pos = torch.tensor([[cache_pos]], dtype=torch.long).to(device)
    attn_mask[:, cache_pos] = 1
    mask_d = attn_mask.to(device)

    t_prep = time.perf_counter()

    with torch.no_grad():
        out = compiled(input_ids=next_ids, past_key_values=cache,
                      cache_position=new_cp, use_cache=True,
                      attention_mask=mask_d, position_ids=new_pos)

    t_fwd = time.perf_counter()

    logits = out.logits.to("cpu")
    cur = logits[0, -1].argmax().item()

    t_end = time.perf_counter()

    generated.append(cur)
    cache_pos += 1
    timings.append({
        'prep': (t_prep - t0) * 1000,
        'fwd': (t_fwd - t_prep) * 1000,
        'logits': (t_end - t_fwd) * 1000,
        'total': (t_end - t0) * 1000,
    })

# Skip first 2 decode (JIT compile)
steady = timings[2:]
print(f"\nGenerated: {tokenizer.decode(generated)!r}")
print(f"\n=== Decode timing (ms), skipping first 2 JIT compiles ===")
print(f"{'':>5} {'prep':>8} {'fwd':>8} {'logits':>8} {'total':>8}")
for k in ['prep', 'fwd', 'logits', 'total']:
    vals = [t[k] for t in steady]
    avg = sum(vals) / len(vals)
    mn = min(vals)
    mx = max(vals)
    print(f"{k:>5} {avg:>8.2f} {mn:>8.2f} {mx:>8.2f}")

print(f"\nFirst 2 decode (JIT compile):")
for i, t in enumerate(timings[:2]):
    print(f"  [{i}] prep={t['prep']:.1f}ms fwd={t['fwd']:.1f}ms logits={t['logits']:.1f}ms total={t['total']:.1f}ms")
