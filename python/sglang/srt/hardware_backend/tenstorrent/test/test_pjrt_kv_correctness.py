"""Verify KV cache correctness after fill_(0) reset.

Sends the SAME prompt twice through the compiled model.
If fill_(0) works correctly, both should produce identical tokens.
"""
import os, time
from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch, torch_xla, torch_xla.runtime as xr
xr.set_device_type("TT"); xr.use_spmd()
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

CACHE_SIZE = 2048
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"
NUM_DECODE = 20

device = torch_xla.device()
print(f"[init] device={device}, n={xr.global_runtime_device_count()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True)
model.eval()
model = model.to(device)
compiled = torch.compile(model, backend="tt")
config = model.config
nkv = config.num_key_value_heads
hd = config.hidden_size // config.num_attention_heads

cache = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
cache.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
for layer in cache.layers:
    layer.keys = layer.keys.to(device)
    layer.values = layer.values.to(device)

attn_mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)

def run_request(prompt_text, request_num):
    """Run one request and return generated token IDs."""
    tok = tokenizer(prompt_text, return_tensors="pt")
    input_ids = tok.input_ids
    seq_len = input_ids.shape[1]

    ids_dev = input_ids.to(device)
    cp = torch.arange(0, seq_len).to(device)
    pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(device)
    mask = attn_mask.to(device)

    t0 = time.time()
    with torch.no_grad():
        out = compiled(input_ids=ids_dev, past_key_values=cache,
                      cache_position=cp, use_cache=True, attention_mask=mask,
                      position_ids=pos)
    dt = time.time() - t0
    logits = out.logits.to("cpu")
    first_tok = logits[0, -1].argmax().item()
    print(f"  Req {request_num} prefill: {dt:.2f}s, first={tokenizer.decode(first_tok)!r}")

    generated = [first_tok]
    cur = first_tok
    cache_pos = seq_len
    for i in range(NUM_DECODE - 1):
        next_ids = torch.tensor([[cur]]).to(device)
        new_cp = torch.tensor([cache_pos]).to(device)
        new_pos = torch.tensor([[cache_pos]], dtype=torch.long).to(device)
        attn_mask[:, cache_pos] = 1

        with torch.no_grad():
            out = compiled(input_ids=next_ids, past_key_values=cache,
                          cache_position=new_cp, use_cache=True,
                          attention_mask=attn_mask.to(device),
                          position_ids=new_pos)
        logits = out.logits.to("cpu")
        cur = logits[0, -1].argmax().item()
        generated.append(cur)
        cache_pos += 1

    text = tokenizer.decode(generated)
    print(f"  Req {request_num} output: {text!r}")
    return generated

# === Request 1 ===
print("\n=== Request 1 ===")
tokens1 = run_request(PROMPT, 1)

# === Reset with fill_(0) ===
print("\n=== Resetting cache with fill_(0) ===")
for layer in cache.layers:
    layer.keys.fill_(0)
    layer.values.fill_(0)
    if hasattr(layer, "cumulative_length"):
        layer.cumulative_length.fill_(0)
attn_mask.fill_(1)  # reset mask
print("  Reset done")

# === Request 2 (same prompt) ===
print("\n=== Request 2 (same prompt) ===")
tokens2 = run_request(PROMPT, 2)

# === Compare ===
match = tokens1 == tokens2
print(f"\n=== RESULT: tokens match = {match} ===")
if not match:
    for i, (a, b) in enumerate(zip(tokens1, tokens2)):
        if a != b:
            print(f"  Diverge at position {i}: {a} ({tokenizer.decode(a)!r}) vs {b} ({tokenizer.decode(b)!r})")
            break
