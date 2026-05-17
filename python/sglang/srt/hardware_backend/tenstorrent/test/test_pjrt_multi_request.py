"""Test multi-request sequence with different reset strategies.

Reproduces Error 13 from zero_() and tests alternatives.
"""
import os, time, traceback

from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])

os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch
import torch_xla
import torch_xla.runtime as xr

xr.set_device_type("TT")
xr.use_spmd()

from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache

device = torch_xla.device()
n = xr.global_runtime_device_count()
print(f"[init] device={device}, n={n}", flush=True)

CACHE_SIZE = 2048
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True,
)
model.eval()
model = model.to(device)
compiled = torch.compile(model, backend="tt")
config = model.config
nkv = config.num_key_value_heads
hd = config.hidden_size // config.num_attention_heads
print(f"[model] compiled, nkv={nkv}, hd={hd}", flush=True)

def make_cache():
    """Create a fresh StaticCache with tensors on device."""
    c = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    return c

cache = make_cache()
attn_mask = torch.zeros((1, CACHE_SIZE), dtype=torch.int32)
cache_pos_tracker = 0

def prefill(seq_len):
    global cache_pos_tracker
    input_ids = torch.randint(100, 1000, (1, seq_len)).to(device)
    cache_pos = torch.arange(0, seq_len).to(device)
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(device)
    attn_mask[:, :seq_len] = 1
    mask_dev = attn_mask.to(device)
    cache_pos_tracker = seq_len
    t0 = time.time()
    with torch.no_grad():
        out = compiled(input_ids=input_ids, past_key_values=cache,
                      cache_position=cache_pos, use_cache=True,
                      attention_mask=mask_dev, position_ids=position_ids)
    dt = time.time() - t0
    logits = out.logits.to("cpu")
    tok = logits[0, -1].argmax().item()
    print(f"  prefill({seq_len}): {dt:.2f}s, first_tok={tok}", flush=True)
    return tok

def decode(prev_tok):
    global cache_pos_tracker
    input_ids = torch.tensor([[prev_tok]]).to(device)
    cp = torch.tensor([cache_pos_tracker]).to(device)
    pos = torch.tensor([[cache_pos_tracker]], dtype=torch.long).to(device)
    attn_mask[:, cache_pos_tracker] = 1
    mask_dev = attn_mask.to(device)
    cache_pos_tracker += 1
    t0 = time.time()
    with torch.no_grad():
        out = compiled(input_ids=input_ids, past_key_values=cache,
                      cache_position=cp, use_cache=True,
                      attention_mask=mask_dev, position_ids=pos)
    dt = time.time() - t0
    logits = out.logits.to("cpu")
    tok = logits[0, -1].argmax().item()
    print(f"  decode(pos={cache_pos_tracker-1}): {dt:.4f}s, tok={tok}", flush=True)
    return tok

def reset_v1_zero():
    """Original: zero_() on device tensors — triggers Error 13."""
    global cache_pos_tracker
    for layer in cache.layers:
        layer.keys.zero_()
        layer.values.zero_()
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length.zero_()
    attn_mask.zero_()
    cache_pos_tracker = 0

def reset_v2_recreate():
    """Recreate cache tensors from scratch instead of zeroing."""
    global cache, cache_pos_tracker
    cache = make_cache()
    attn_mask.zero_()
    cache_pos_tracker = 0

def reset_v3_fill():
    """Use fill_(0) instead of zero_()."""
    global cache_pos_tracker
    for layer in cache.layers:
        layer.keys.fill_(0)
        layer.values.fill_(0)
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length.fill_(0)
    attn_mask.fill_(0)
    cache_pos_tracker = 0

def reset_v4_mul():
    """Multiply by zero instead of zero_()."""
    global cache_pos_tracker
    for layer in cache.layers:
        layer.keys.mul_(0)
        layer.values.mul_(0)
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length.mul_(0)
    attn_mask.mul_(0)
    cache_pos_tracker = 0

def reset_v5_new_tensors():
    """Replace cache tensor data with new zeros (same cache object)."""
    global cache_pos_tracker
    for layer in cache.layers:
        layer.keys = torch.zeros_like(layer.keys)
        layer.values = torch.zeros_like(layer.values)
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = torch.zeros_like(layer.cumulative_length)
    attn_mask.zero_()
    cache_pos_tracker = 0

# === Test each reset strategy ===
strategies = [
    ("v2_recreate", reset_v2_recreate),
    ("v3_fill", reset_v3_fill),
    ("v4_mul", reset_v4_mul),
    ("v5_new_tensors", reset_v5_new_tensors),
]

# First: do initial request
print("\n=== Initial request: prefill(2) + decode(3) ===", flush=True)
tok = prefill(2)
for i in range(3):
    tok = decode(tok)
print("Initial: PASS", flush=True)

for name, reset_fn in strategies:
    print(f"\n=== Testing {name}: reset + prefill(16) + decode(3) ===", flush=True)
    try:
        reset_fn()
        print(f"  {name}: reset done", flush=True)
        tok = prefill(16)
        for i in range(3):
            tok = decode(tok)
        print(f"{name}: PASS", flush=True)
    except Exception as e:
        print(f"{name}: FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
        # Device might be corrupted — try to recover
        try:
            cache = make_cache()
            attn_mask.zero_()
        except:
            print(f"  Cannot recover for next test", flush=True)
            break
