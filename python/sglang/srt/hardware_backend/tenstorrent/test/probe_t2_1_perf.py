"""Phase 0.5 probe: TPOT timing + steady-state recompile check.

Runs the same TTFunctionalCache + MultiDecodeK from probe_t2_1_functional_cache.py
but with more iterations to measure steady-state TPOT and confirm no recompiles
after the first wrap.

Tests:
  - Baseline StaticCache K=1: 20 decode iters (drop first 2 as JIT warmup)
  - K=1 TTFunctionalCache: 20 decode iters
  - K=2 MultiDecodeK: 10 wraps (= 20 tokens)
  - K=4 MultiDecodeK: 5 wraps (= 20 tokens)

Per test:
  - TPOT min/mean/max ms (steady-state after JIT skip)
  - tok/s
  - dynamo frames_ok (target: stable after first wrap)
  - dynamo recompiles (target: 0 after warmup)
"""
import os, sys, time, random, traceback
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
except (ImportError, AttributeError):
    pass

os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch
from torch import nn
import torch_xla, torch_xla.runtime as xr
xr.set_device_type("TT"); xr.use_spmd()

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import StaticCache

device = torch_xla.device()
N = xr.global_runtime_device_count()
print(f"[perf] device={device}, n_devices={N}", flush=True)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 64
SEED = 42
N_DECODE = 20
SKIP_JIT = 2


# ====================================================================
class TTFunctionalCache(StaticCache):
    def __init__(self, config, max_batch_size, max_cache_len, device, dtype):
        super().__init__(config=config, max_batch_size=max_batch_size,
                         max_cache_len=max_cache_len, device=device, dtype=dtype)
        self._latest_k = [None] * len(self.layers)
        self._latest_v = [None] * len(self.layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[-2] != 1:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        cp = cache_kwargs["cache_position"]
        prior_k = self._latest_k[layer_idx] if self._latest_k[layer_idx] is not None else self.layers[layer_idx].keys
        prior_v = self._latest_v[layer_idx] if self._latest_v[layer_idx] is not None else self.layers[layer_idx].values
        max_len = prior_k.shape[2]
        pos = torch.arange(max_len, device=cp.device)
        write_mask = (pos == cp[0]).view(1, 1, max_len, 1)
        new_k = torch.where(write_mask, key_states.expand_as(prior_k), prior_k)
        new_v = torch.where(write_mask, value_states.expand_as(prior_v), prior_v)
        self._latest_k[layer_idx] = new_k
        self._latest_v[layer_idx] = new_v
        return new_k, new_v


class MultiDecodeK(nn.Module):
    def __init__(self, m, K):
        super().__init__()
        self.m = m
        self.K = K

    def forward(self, t_in, cps, poss, attention_mask, past_key_values):
        toks = []
        cur = t_in
        for k in range(self.K):
            out = self.m(input_ids=cur, past_key_values=past_key_values,
                         cache_position=cps[k], use_cache=True,
                         attention_mask=attention_mask, position_ids=poss[k])
            cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(cur)
        return torch.cat(toks, dim=-1)


def reset_counters():
    import torch._dynamo
    torch._dynamo.utils.counters.clear()
    torch._dynamo.reset()


def get_counters():
    import torch._dynamo
    c = torch._dynamo.utils.counters
    return {
        "frames_ok": dict(c.get("frames", {})).get("ok", 0),
        "recompiles": dict(c.get("stats", {})).get("recompiles", 0),
    }


# ====================================================================
print(f"[perf] loading {MODEL}", flush=True)
config = AutoConfig.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True
)
base.eval()
base = base.to(device)
nkv = config.num_key_value_heads
hd = config.hidden_size // config.num_attention_heads


def make_static_cache():
    c = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device); layer.values = layer.values.to(device)
    return c


def make_func_cache():
    c = TTFunctionalCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                          device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device); layer.values = layer.values.to(device)
    return c


def prefill(cache):
    random.seed(SEED)
    rand_ids = torch.tensor([[random.randint(10, config.vocab_size - 1) for _ in range(INPUT_LEN)]])
    attn_mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)
    compiled = torch.compile(base, backend="tt")
    with torch.no_grad():
        out = compiled(input_ids=rand_ids.to(device), past_key_values=cache,
                       cache_position=torch.arange(INPUT_LEN).to(device),
                       use_cache=True, attention_mask=attn_mask.to(device),
                       position_ids=torch.arange(INPUT_LEN, dtype=torch.long).unsqueeze(0).to(device))
    torch_xla.sync()
    return out.logits[:, -1, :].argmax(-1).to("cpu").item(), attn_mask


def bench_k1(cache_factory, label):
    reset_counters()
    cache = cache_factory()
    first_tok, attn_mask = prefill(cache)
    compiled = torch.compile(base, backend="tt")
    cur = first_tok
    cache_pos = INPUT_LEN
    times = []
    frames_history = []
    for i in range(N_DECODE):
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
        cur = out.logits[:, -1, :].to("cpu").argmax(-1).item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        frames_history.append(get_counters()["frames_ok"])
        cache_pos += 1
    return times, frames_history


def bench_kn(K, label):
    reset_counters()
    cache = make_func_cache()
    first_tok, attn_mask = prefill(cache)
    multi = MultiDecodeK(base, K=K)
    compiled = torch.compile(multi, backend="tt")
    cur = first_tok
    cache_pos = INPUT_LEN
    times = []
    frames_history = []
    n_wraps = N_DECODE // K
    for w in range(n_wraps):
        for k in range(K):
            attn_mask[:, cache_pos + k] = 1
        md = attn_mask.to(device)
        t_in = torch.tensor([[cur]]).to(device)
        cps = tuple(torch.tensor([cache_pos + k]).to(device) for k in range(K))
        poss = tuple(torch.tensor([[cache_pos + k]], dtype=torch.long).to(device) for k in range(K))
        t0 = time.perf_counter()
        with torch.no_grad():
            out_tokens = compiled(t_in, cps, poss, md, past_key_values=cache)
        out_cpu = out_tokens.to("cpu").tolist()[0]
        t1 = time.perf_counter()
        per_tok = (t1 - t0) * 1000 / K
        for _ in range(K):
            times.append(per_tok)
        frames_history.append(get_counters()["frames_ok"])
        cur = out_cpu[-1]
        cache_pos += K
    return times, frames_history


# ====================================================================
results = {}

for label, fn in [
    ("BASELINE StaticCache K=1", lambda: bench_k1(make_static_cache, "static")),
    ("TTFunctionalCache K=1",    lambda: bench_k1(make_func_cache,   "functional")),
    ("MultiDecodeK K=2",         lambda: bench_kn(K=2, label="K2")),
    ("MultiDecodeK K=4",         lambda: bench_kn(K=4, label="K4")),
]:
    print(f"\n[perf] {label}", flush=True)
    try:
        times, frames = fn()
        steady = times[SKIP_JIT:]
        mean = sum(steady) / len(steady)
        results[label] = {
            "tpot_ms": round(mean, 2),
            "min_ms":  round(min(steady), 2),
            "max_ms":  round(max(steady), 2),
            "tok_s":   round(1000 / mean, 1),
            "jit_calls_ms": [round(t, 1) for t in times[:SKIP_JIT]],
            "frames_first": frames[0] if frames else None,
            "frames_last":  frames[-1] if frames else None,
            "frame_grew_after_first": (frames[-1] > frames[0]) if frames else False,
        }
        print(f"  tpot={mean:.2f}ms  min={min(steady):.2f}  max={max(steady):.2f}  ({1000/mean:.1f} tok/s)")
        print(f"  jit warmup calls: {results[label]['jit_calls_ms']}")
        print(f"  frames_ok: first call={frames[0]}, last={frames[-1]}, grew after first={results[label]['frame_grew_after_first']}")
    except Exception as e:
        results[label] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  FAIL: {e}")
        traceback.print_exc()


print("\n" + "="*80)
print(f"{'Variant':<28} {'TPOT':>8} {'min':>7} {'max':>7} {'tok/s':>7} {'frames':>10}")
print("-"*80)
for label, r in results.items():
    if "error" in r:
        print(f"{label:<28} FAIL: {r['error'][:60]}")
    else:
        frames_str = f"{r['frames_first']}→{r['frames_last']}"
        print(f"{label:<28} {r['tpot_ms']:>6.1f}ms {r['min_ms']:>6.1f}  {r['max_ms']:>6.1f}  {r['tok_s']:>6}  {frames_str:>10}")

base_tpot = results.get("BASELINE StaticCache K=1", {}).get("tpot_ms")
k4_tpot = results.get("MultiDecodeK K=4", {}).get("tpot_ms")
if base_tpot and k4_tpot:
    speedup = base_tpot / k4_tpot
    print(f"\nA1 K=4 speedup vs baseline: {speedup:.2f}× ({base_tpot:.1f}ms → {k4_tpot:.1f}ms)")
