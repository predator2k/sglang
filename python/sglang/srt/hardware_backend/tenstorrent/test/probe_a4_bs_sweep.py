"""A4-aligned probe: sweep batch size to find the throughput sweet spot.

bs=1 = ~59ms/tok = 17 tok/s (baseline)
bs=2 confirmed 1.19× (~50 ms/tok = 20 tok/s)
Sweep bs ∈ {1, 2, 4, 8} (and maybe 16 if mem allows).
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
print(f"[sweep] device={device}", flush=True)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 64
N_DECODE = 16     # decode 16 steps per batch
SKIP_JIT = 3

config = AutoConfig.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True
)
base.eval(); base = base.to(device)
nkv = config.num_key_value_heads
hd = config.hidden_size // config.num_attention_heads
compiled = torch.compile(base, backend="tt")


def make_cache(bs):
    c = StaticCache(config=config, max_batch_size=bs, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=bs, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device); layer.values = layer.values.to(device)
    return c


def prefill_into(cache, bs):
    random.seed(42)
    ids_one = [random.randint(10, config.vocab_size - 1) for _ in range(INPUT_LEN)]
    ids = torch.tensor([ids_one] * bs)
    mask = torch.ones((bs, CACHE_SIZE), dtype=torch.int32)
    cp = torch.arange(INPUT_LEN).to(device)
    pos = torch.arange(INPUT_LEN, dtype=torch.long).unsqueeze(0).expand(bs, -1).to(device)
    with torch.no_grad():
        out = compiled(input_ids=ids.to(device), past_key_values=cache,
                       cache_position=cp, use_cache=True, attention_mask=mask.to(device),
                       position_ids=pos)
    torch_xla.sync()
    return out.logits[:, -1, :].argmax(-1).to("cpu").tolist(), mask


def bench_bs(bs):
    try:
        cache = make_cache(bs)
        first_toks, mask = prefill_into(cache, bs)
    except Exception as e:
        return None, f"prefill: {e}"
    curs = list(first_toks); cp_pos = INPUT_LEN; times = []
    try:
        for i in range(N_DECODE):
            t0 = time.perf_counter()
            next_ids = torch.tensor([[c] for c in curs]).to(device)
            cp_dev = torch.tensor([cp_pos]).to(device)
            pos_dev = torch.tensor([[cp_pos]] * bs, dtype=torch.long).to(device)
            mask[:, cp_pos] = 1
            mask_dev = mask.to(device)
            with torch.no_grad():
                out = compiled(input_ids=next_ids, past_key_values=cache,
                               cache_position=cp_dev, use_cache=True,
                               attention_mask=mask_dev, position_ids=pos_dev)
            nexts = out.logits[:, -1, :].to("cpu").argmax(-1).tolist()
            t1 = time.perf_counter()
            curs = nexts; cp_pos += 1
            wall = (t1 - t0) * 1000
            per_tok = wall / bs
            times.append(wall)        # wall per step
    except Exception as e:
        return None, f"decode: {e}"
    return times, None


print(f"\n[sweep] decode-time vs batch size", flush=True)
print(f"{'bs':>4} {'wall/step':>10} {'tok/s':>8} {'per-tok':>8} {'multiplier':>10}")
print("-" * 50)

base_per_tok = None
for bs in [1, 2, 4, 8]:
    print(f"\n[sweep] bs={bs}", flush=True)
    times, err = bench_bs(bs)
    if err:
        print(f"  FAIL: {err}", flush=True)
        continue
    steady = times[SKIP_JIT:]
    mean_wall = sum(steady) / len(steady)
    per_tok = mean_wall / bs
    tok_s = 1000 * bs / mean_wall
    if base_per_tok is None:
        base_per_tok = per_tok
        mult = 1.0
    else:
        mult = base_per_tok / per_tok
    print(f"  wall/step={mean_wall:.2f}ms  per-tok={per_tok:.2f}ms  ({tok_s:.1f} tok/s)  {mult:.2f}×")
