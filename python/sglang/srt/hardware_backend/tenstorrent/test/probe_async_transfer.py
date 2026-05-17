"""Async probe: does tensor.to('cpu', non_blocking=True) actually overlap on tt-xla?

Test pattern A — blocking (current):
  out = compiled(...)
  tok = out.logits[:, -1, :].to('cpu').argmax(-1).item()   # synchronous transfer + host work
  # next iteration

Test pattern B — non_blocking:
  out = compiled(...)
  logits_cpu = out.logits[:, -1, :].to('cpu', non_blocking=True)   # queued
  # ... issue next iteration's prep ...
  tok = logits_cpu.argmax(-1).item()                                # forces sync

Test pattern C — overlap forward T+1 with logits transfer T:
  out_t = compiled(...)
  logits_t = out_t.logits[:, -1, :].to('cpu', non_blocking=True)   # queued
  # issue forward T+1 with token from T-1 (won't be bit-exact but tests if overlap works)
  out_t1 = compiled(...)
  tok_t = logits_t.argmax(-1).item()   # force sync of T
"""
import os, time, random
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
except (ImportError, AttributeError):
    pass
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch
import torch_xla, torch_xla.runtime as xr
xr.set_device_type("TT"); xr.use_spmd()
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import StaticCache

device = torch_xla.device()
print(f"[async] device={device}", flush=True)


# Inspect torch_xla for async APIs
print("\n[async] looking for async APIs in torch_xla", flush=True)
import torch_xla
print(f"  torch_xla.__version__: {torch_xla.__version__}")
async_names = [n for n in dir(torch_xla) if 'async' in n.lower() or 'queue' in n.lower()]
print(f"  torch_xla async-related names: {async_names}")
import torch_xla.core
xc_names = [n for n in dir(torch_xla.core) if 'async' in n.lower()]
print(f"  torch_xla.core async-related: {xc_names}")
try:
    import torch_xla.core.xla_model as xm
    xm_async = [n for n in dir(xm) if 'async' in n.lower() or 'non_block' in n.lower()]
    print(f"  xla_model async-related: {xm_async}")
except ImportError:
    print("  xla_model unavailable")


MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 64
N_DECODE = 20
SKIP_JIT = 3

config = AutoConfig.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True
)
base.eval(); base = base.to(device)
nkv = config.num_key_value_heads
hd = config.hidden_size // config.num_attention_heads
compiled = torch.compile(base, backend="tt")


def make_cache():
    c = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device); layer.values = layer.values.to(device)
    return c


def prefill_into(cache):
    random.seed(42)
    rand_ids = torch.tensor([[random.randint(10, config.vocab_size - 1) for _ in range(INPUT_LEN)]])
    mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)
    with torch.no_grad():
        out = compiled(input_ids=rand_ids.to(device), past_key_values=cache,
                       cache_position=torch.arange(INPUT_LEN).to(device),
                       use_cache=True, attention_mask=mask.to(device),
                       position_ids=torch.arange(INPUT_LEN, dtype=torch.long).unsqueeze(0).to(device))
    torch_xla.sync()
    return out.logits[:, -1, :].argmax(-1).to("cpu").item(), mask


def bench_blocking():
    cache = make_cache()
    first_tok, mask = prefill_into(cache)
    cur = first_tok; cp = INPUT_LEN; times = []
    for i in range(N_DECODE):
        t0 = time.perf_counter()
        next_ids = torch.tensor([[cur]]).to(device)
        cp_dev = torch.tensor([cp]).to(device)
        pos_dev = torch.tensor([[cp]], dtype=torch.long).to(device)
        mask[:, cp] = 1
        mask_dev = mask.to(device)
        with torch.no_grad():
            out = compiled(input_ids=next_ids, past_key_values=cache,
                           cache_position=cp_dev, use_cache=True,
                           attention_mask=mask_dev, position_ids=pos_dev)
        cur = out.logits[:, -1, :].to("cpu").argmax(-1).item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000); cp += 1
    return times


def bench_nonblocking():
    """Try non_blocking transfer."""
    cache = make_cache()
    first_tok, mask = prefill_into(cache)
    cur = first_tok; cp = INPUT_LEN; times = []
    for i in range(N_DECODE):
        t0 = time.perf_counter()
        next_ids = torch.tensor([[cur]]).to(device)
        cp_dev = torch.tensor([cp]).to(device)
        pos_dev = torch.tensor([[cp]], dtype=torch.long).to(device)
        mask[:, cp] = 1
        mask_dev = mask.to(device)
        with torch.no_grad():
            out = compiled(input_ids=next_ids, past_key_values=cache,
                           cache_position=cp_dev, use_cache=True,
                           attention_mask=mask_dev, position_ids=pos_dev)
        # non-blocking transfer
        logits_cpu = out.logits[:, -1, :].to("cpu", non_blocking=True)
        # Do a tiny bit of host work in the gap
        # ... nothing useful here, just measure if to() returned early
        cur = logits_cpu.argmax(-1).item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000); cp += 1
    return times


print("\n[async] === blocking baseline ===", flush=True)
t_b = bench_blocking()
mean_b = sum(t_b[SKIP_JIT:]) / len(t_b[SKIP_JIT:])
print(f"  blocking TPOT mean={mean_b:.2f}ms  min={min(t_b[SKIP_JIT:]):.2f}", flush=True)

print("\n[async] === non_blocking ===", flush=True)
try:
    t_nb = bench_nonblocking()
    mean_nb = sum(t_nb[SKIP_JIT:]) / len(t_nb[SKIP_JIT:])
    print(f"  non_blocking TPOT mean={mean_nb:.2f}ms  min={min(t_nb[SKIP_JIT:]):.2f}")
    print(f"  delta: {mean_b - mean_nb:+.2f}ms ({(mean_b - mean_nb)/mean_b*100:+.1f}%)")
except Exception as e:
    print(f"  non_blocking FAIL: {e}")
    import traceback; traceback.print_exc()
