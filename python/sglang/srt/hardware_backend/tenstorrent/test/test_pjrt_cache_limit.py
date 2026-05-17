"""Investigate PJRT StaticCache limits on TT devices.

Usage: python3 test_pjrt_cache_limit.py [cache_size] [prompt_len]
"""
import sys, os, time, traceback

# Fix transformers 5.6.0 flash_attn KeyError
from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])

os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

import torch
import torch_xla
import torch_xla.runtime as xr

xr.set_device_type("TT")
xr.use_spmd()

from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache

def patch_static_cache_device(cache, device):
    """Move StaticCache internal tracking tensors to device so
    torch.compile doesn't see a CPU/XLA device mismatch when HF
    computes position_ids from past_seen_tokens."""
    # In transformers 5.x, StaticCache tracks seen tokens via
    # a _seen_tokens int or tensor. We need it on device.
    if hasattr(cache, '_seen_tokens') and isinstance(cache._seen_tokens, torch.Tensor):
        cache._seen_tokens = cache._seen_tokens.to(device)
    # Some versions use a plain int — wrap it
    if hasattr(cache, '_seen_tokens') and isinstance(cache._seen_tokens, int):
        cache._seen_tokens = torch.tensor(cache._seen_tokens, dtype=torch.long, device=device)

def test_cache_size(cache_size, prompt_len=10):
    device = torch_xla.device()
    n = xr.global_runtime_device_count()
    print(f"[init] device={device}, n={n}, cache_size={cache_size}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True,
    )
    model.eval()
    config = model.config
    nkv = config.num_key_value_heads
    hd = config.hidden_size // config.num_attention_heads
    print(f"[model] loaded, nkv={nkv}, hd={hd}", flush=True)

    model = model.to(device)
    compiled = torch.compile(model, backend="tt")
    print("[model] compiled", flush=True)

    cache = StaticCache(config=config, max_batch_size=1, max_cache_len=cache_size,
                        device="cpu", dtype=torch.bfloat16)
    cache.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                               dtype=torch.bfloat16, device="cpu")
    for layer in cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)

    # Inspect cache internals for debugging
    print(f"[cache] type={type(cache).__name__}", flush=True)
    for attr in ['_seen_tokens', 'seen_tokens', '_cache_position']:
        if hasattr(cache, attr):
            val = getattr(cache, attr)
            if isinstance(val, torch.Tensor):
                print(f"  cache.{attr}: tensor device={val.device} dtype={val.dtype} val={val}", flush=True)
            else:
                print(f"  cache.{attr}: {type(val).__name__} = {val}", flush=True)

    # Check what get_seq_length returns
    try:
        seq_len_val = cache.get_seq_length()
        print(f"  cache.get_seq_length() = {seq_len_val} (type={type(seq_len_val).__name__})", flush=True)
    except Exception as e:
        print(f"  cache.get_seq_length() error: {e}", flush=True)

    # Check past_seen_tokens if available
    try:
        pst = cache.get_past_seen_tokens() if hasattr(cache, 'get_past_seen_tokens') else None
        print(f"  cache.get_past_seen_tokens() = {pst}", flush=True)
    except Exception as e:
        print(f"  cache.get_past_seen_tokens() not available: {e}", flush=True)

    print(f"[cache] initialized: max_len={cache_size}", flush=True)

    attn_mask = torch.ones((1, cache_size), dtype=torch.int32).to(device)
    input_ids = torch.randint(100, 1000, (1, prompt_len)).to(device)
    cache_pos = torch.arange(0, prompt_len).to(device)
    # Explicit position_ids to avoid HF internal computation
    position_ids = torch.arange(0, prompt_len, dtype=torch.long).unsqueeze(0).to(device)

    print(f"[prefill] prompt_len={prompt_len}, cache_size={cache_size}...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = compiled(input_ids=input_ids, past_key_values=cache,
                      cache_position=cache_pos, use_cache=True,
                      attention_mask=attn_mask, position_ids=position_ids)
    dt = time.time() - t0
    logits = out.logits.to("cpu")
    print(f"[prefill] PASS: {dt:.2f}s, logits={logits.shape}", flush=True)

    # Decode one token
    next_id = logits[0, -1].argmax().item()
    next_ids = torch.tensor([[next_id]]).to(device)
    new_cp = torch.tensor([prompt_len]).to(device)
    new_pos = torch.tensor([[prompt_len]], dtype=torch.long).to(device)

    t0 = time.time()
    with torch.no_grad():
        out2 = compiled(input_ids=next_ids, past_key_values=cache,
                       cache_position=new_cp, use_cache=True,
                       attention_mask=attn_mask, position_ids=new_pos)
    dt2 = time.time() - t0
    print(f"[decode] PASS: {dt2:.4f}s", flush=True)
    print(f"RESULT: cache_size={cache_size} PASS", flush=True)

if __name__ == "__main__":
    cs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    pl = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    try:
        test_cache_size(cs, pl)
    except Exception as e:
        print(f"RESULT: FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
