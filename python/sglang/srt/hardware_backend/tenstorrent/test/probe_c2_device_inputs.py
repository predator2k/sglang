"""C2 probe: how much TPOT do we save by keeping inputs device-resident?

Production decode step (tt_xla_model.py:331-337) does 4 .to(device) calls:
  - input_2d        (1 int32)
  - cache_pos_dev   (1 int32)
  - position_ids    (1 int64)
  - attn_mask_dev   ([1, CACHE_SIZE] int32, ~8KB)

If each is a PJRT call (~38ms hypothesis), C2 could save 100ms+. If H2D is
free / amortized, C2 buys nothing. Probe measures both.

Variants:
  V0 baseline:  4 .to(device) per step (current)
  V1 mask only: mask device-resident, updated via _mask[:, cp] = 1
  V2 inputs only: 3 small inputs in pre-allocated device buffers, fill_-ed
  V3 all:       V1 + V2 combined
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
print(f"[c2] device={device}", flush=True)

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


# =================================================================
def bench_v0_baseline():
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


def bench_v1_mask_only():
    """Mask kept on device, updated via on-device op."""
    cache = make_cache()
    first_tok, mask_cpu = prefill_into(cache)
    mask_dev = mask_cpu.to(device)            # one-time transfer
    cur = first_tok; cp = INPUT_LEN; times = []
    for i in range(N_DECODE):
        t0 = time.perf_counter()
        next_ids = torch.tensor([[cur]]).to(device)
        cp_dev = torch.tensor([cp]).to(device)
        pos_dev = torch.tensor([[cp]], dtype=torch.long).to(device)
        mask_dev[:, cp] = 1                   # on-device update
        with torch.no_grad():
            out = compiled(input_ids=next_ids, past_key_values=cache,
                           cache_position=cp_dev, use_cache=True,
                           attention_mask=mask_dev, position_ids=pos_dev)
        cur = out.logits[:, -1, :].to("cpu").argmax(-1).item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000); cp += 1
    return times


def bench_v2_inputs_only():
    """3 small inputs pre-allocated, fill_-ed each step. Mask still per-step .to(device)."""
    cache = make_cache()
    first_tok, mask = prefill_into(cache)
    input_buf = torch.zeros((1, 1), dtype=torch.int32, device=device)
    cp_buf = torch.zeros((1,), dtype=torch.int32, device=device)
    pos_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
    cur = first_tok; cp = INPUT_LEN; times = []
    for i in range(N_DECODE):
        t0 = time.perf_counter()
        input_buf.fill_(cur)
        cp_buf.fill_(cp)
        pos_buf.fill_(cp)
        mask[:, cp] = 1
        mask_dev = mask.to(device)
        with torch.no_grad():
            out = compiled(input_ids=input_buf, past_key_values=cache,
                           cache_position=cp_buf, use_cache=True,
                           attention_mask=mask_dev, position_ids=pos_buf)
        cur = out.logits[:, -1, :].to("cpu").argmax(-1).item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000); cp += 1
    return times


def bench_v3_all():
    """All inputs device-resident, all updates on-device."""
    cache = make_cache()
    first_tok, mask_cpu = prefill_into(cache)
    mask_dev = mask_cpu.to(device)
    input_buf = torch.zeros((1, 1), dtype=torch.int32, device=device)
    cp_buf = torch.zeros((1,), dtype=torch.int32, device=device)
    pos_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
    cur = first_tok; cp = INPUT_LEN; times = []
    for i in range(N_DECODE):
        t0 = time.perf_counter()
        input_buf.fill_(cur)
        cp_buf.fill_(cp)
        pos_buf.fill_(cp)
        mask_dev[:, cp] = 1
        with torch.no_grad():
            out = compiled(input_ids=input_buf, past_key_values=cache,
                           cache_position=cp_buf, use_cache=True,
                           attention_mask=mask_dev, position_ids=pos_buf)
        cur = out.logits[:, -1, :].to("cpu").argmax(-1).item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000); cp += 1
    return times


results = {}
for label, fn in [
    ("V0 baseline (4 ToD per step)", bench_v0_baseline),
    ("V1 mask device-resident",      bench_v1_mask_only),
    ("V2 inputs device-resident",    bench_v2_inputs_only),
    ("V3 all device-resident",       bench_v3_all),
]:
    print(f"\n[c2] {label}", flush=True)
    try:
        times = fn()
        steady = times[SKIP_JIT:]
        mean = sum(steady) / len(steady)
        results[label] = {"mean": mean, "min": min(steady), "max": max(steady),
                          "jit": times[:SKIP_JIT]}
        print(f"  tpot mean={mean:.2f}  min={min(steady):.2f}  max={max(steady):.2f}  ({1000/mean:.1f} tok/s)")
        print(f"  warmup: {[round(t,1) for t in times[:SKIP_JIT]]}")
    except Exception as e:
        results[label] = {"error": str(e)}
        import traceback; traceback.print_exc()
        print(f"  FAIL: {e}")

print("\n" + "="*70)
print(f"{'Variant':<35} {'TPOT mean':>12} {'min':>8} {'delta vs V0':>14}")
print("-"*70)
v0_mean = results.get("V0 baseline (4 ToD per step)", {}).get("mean")
for label, r in results.items():
    if "error" in r:
        print(f"{label:<35} FAIL")
    else:
        delta = ""
        if v0_mean and r["mean"]:
            d = v0_mean - r["mean"]
            delta = f"{d:+.2f}ms ({d/v0_mean*100:+.1f}%)"
        print(f"{label:<35} {r['mean']:>10.2f}ms {r['min']:>6.2f}ms {delta:>14}")
