"""A4-aligned probe: lockstep batch=2 decode.

If both sequences advance at the same cache_position each step, they share
one PJRT call per step. Throughput should be ~2× for the same total wall-time.

  baseline K=1 bs=1: ~58ms/tok
  aligned  K=1 bs=2: target ~29ms/tok per request
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
print(f"[a4] device={device}", flush=True)

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


def make_cache(batch_size):
    c = StaticCache(config=config, max_batch_size=batch_size, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=batch_size, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device); layer.values = layer.values.to(device)
    return c


def prefill_into(cache, batch_size):
    random.seed(42)
    rand_ids_one = [random.randint(10, config.vocab_size - 1) for _ in range(INPUT_LEN)]
    rand_ids = torch.tensor([rand_ids_one] * batch_size)        # [bs, INPUT_LEN]
    mask = torch.ones((batch_size, CACHE_SIZE), dtype=torch.int32)
    cp = torch.arange(INPUT_LEN).to(device)                     # [INPUT_LEN]
    pos = torch.arange(INPUT_LEN, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(device)
    with torch.no_grad():
        out = compiled(input_ids=rand_ids.to(device), past_key_values=cache,
                       cache_position=cp, use_cache=True, attention_mask=mask.to(device),
                       position_ids=pos)
    torch_xla.sync()
    return out.logits[:, -1, :].argmax(-1).to("cpu").tolist(), mask


def bench_bs(batch_size, label):
    cache = make_cache(batch_size)
    first_toks, mask = prefill_into(cache, batch_size)
    print(f"  prefill first toks: {first_toks}", flush=True)
    curs = list(first_toks); cp = INPUT_LEN; times = []
    for i in range(N_DECODE):
        t0 = time.perf_counter()
        next_ids = torch.tensor([[c] for c in curs]).to(device)            # [bs, 1]
        cp_dev = torch.tensor([cp]).to(device)
        pos_dev = torch.tensor([[cp]] * batch_size, dtype=torch.long).to(device)  # [bs, 1]
        mask[:, cp] = 1
        mask_dev = mask.to(device)
        with torch.no_grad():
            out = compiled(input_ids=next_ids, past_key_values=cache,
                           cache_position=cp_dev, use_cache=True,
                           attention_mask=mask_dev, position_ids=pos_dev)
        nexts = out.logits[:, -1, :].to("cpu").argmax(-1).tolist()
        t1 = time.perf_counter()
        curs = nexts
        cp += 1
        # Per-token time = wall time / batch_size (we get bs tokens per call)
        per_tok = (t1 - t0) * 1000 / batch_size
        for _ in range(batch_size):
            times.append(per_tok)
    return times, curs


print("\n[a4] === bs=1 reference ===", flush=True)
t1, _ = bench_bs(1, "bs=1")
steady1 = t1[SKIP_JIT*1:]
print(f"  bs=1 TPOT mean={sum(steady1)/len(steady1):.2f}ms  min={min(steady1):.2f}", flush=True)

print("\n[a4] === bs=2 lockstep ===", flush=True)
try:
    t2, _ = bench_bs(2, "bs=2")
    steady2 = t2[SKIP_JIT*2:]
    mean2 = sum(steady2)/len(steady2)
    print(f"  bs=2 TPOT per-token mean={mean2:.2f}ms  min={min(steady2):.2f}  ({1000/mean2:.1f} tok/s)")
except Exception as e:
    print(f"  bs=2 FAIL: {e}")
    import traceback; traceback.print_exc()
    t2 = None

print("\n" + "="*60)
if t1 and t2:
    m1 = sum(t1[SKIP_JIT:])/len(t1[SKIP_JIT:])
    m2 = sum(t2[SKIP_JIT*2:])/len(t2[SKIP_JIT*2:])
    print(f"bs=1 per-tok: {m1:.2f}ms")
    print(f"bs=2 per-tok: {m2:.2f}ms")
    print(f"A4 throughput multiplier: {m1/m2:.2f}× (ideal: 2.0×)")
