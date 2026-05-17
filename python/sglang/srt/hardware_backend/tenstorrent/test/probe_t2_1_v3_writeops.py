"""Probe v3: try different KV write ops to find what's fast on tt-xla.

Tests 4 functional-cache variants, all bit-exact equivalent:
  V1: torch.where + expand_as           (current — slow on tt-mlir?)
  V2: clone() + index_copy_             (private per step; may fuse correctly)
  V3: torch.index_put (functional)      (same idea, single op)
  V4: slice_scatter                     (functional alternative)
"""
import os, time, random, traceback
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
print(f"[v3] device={device}", flush=True)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 64
N_DECODE = 20
SKIP_JIT = 4   # be generous, drop first 4
K = 4


class FuncCacheV1Where(StaticCache):
    """torch.where + expand_as (current)"""
    def __init__(self, **kw):
        super().__init__(**kw)
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


class FuncCacheV2CloneIndexCopy(StaticCache):
    """clone() + index_copy_ — each write to private tensor, no shared XLA mutation"""
    def __init__(self, **kw):
        super().__init__(**kw)
        self._latest_k = [None] * len(self.layers)
        self._latest_v = [None] * len(self.layers)
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[-2] != 1:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        cp = cache_kwargs["cache_position"]
        prior_k = self._latest_k[layer_idx] if self._latest_k[layer_idx] is not None else self.layers[layer_idx].keys
        prior_v = self._latest_v[layer_idx] if self._latest_v[layer_idx] is not None else self.layers[layer_idx].values
        new_k = prior_k.clone()
        new_k.index_copy_(2, cp, key_states)
        new_v = prior_v.clone()
        new_v.index_copy_(2, cp, value_states)
        self._latest_k[layer_idx] = new_k
        self._latest_v[layer_idx] = new_v
        return new_k, new_v


class FuncCacheV3SliceScatter(StaticCache):
    """torch.slice_scatter — functional in-place at a fixed position"""
    def __init__(self, **kw):
        super().__init__(**kw)
        self._latest_k = [None] * len(self.layers)
        self._latest_v = [None] * len(self.layers)
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[-2] != 1:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        cp = cache_kwargs["cache_position"]
        prior_k = self._latest_k[layer_idx] if self._latest_k[layer_idx] is not None else self.layers[layer_idx].keys
        prior_v = self._latest_v[layer_idx] if self._latest_v[layer_idx] is not None else self.layers[layer_idx].values
        # slice_scatter: replace slice at index cp[0] along dim 2 with key_states[:,:,0,:]
        # signature: slice_scatter(input, src, dim, start, end, step)
        start = cp[0]
        end = cp[0] + 1
        new_k = torch.slice_scatter(prior_k, key_states, dim=2, start=start, end=end, step=1)
        new_v = torch.slice_scatter(prior_v, value_states, dim=2, start=start, end=end, step=1)
        self._latest_k[layer_idx] = new_k
        self._latest_v[layer_idx] = new_v
        return new_k, new_v


class MultiDecodeK(nn.Module):
    def __init__(self, m, K): super().__init__(); self.m = m; self.K = K
    def forward(self, t_in, cps, poss, attention_mask, past_key_values):
        toks = []; cur = t_in
        for k in range(self.K):
            out = self.m(input_ids=cur, past_key_values=past_key_values,
                         cache_position=cps[k], use_cache=True,
                         attention_mask=attention_mask, position_ids=poss[k])
            cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(cur)
        return torch.cat(toks, dim=-1)


print(f"[v3] loading {MODEL}", flush=True)
config = AutoConfig.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True
)
base.eval(); base = base.to(device)
nkv = config.num_key_value_heads
hd = config.hidden_size // config.num_attention_heads


def make_cache(cache_cls):
    c = cache_cls(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                  device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device); layer.values = layer.values.to(device)
    return c


def prefill(cache):
    random.seed(42)
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


def bench_k4(cache_cls, label):
    import torch._dynamo
    torch._dynamo.utils.counters.clear()
    torch._dynamo.reset()
    try:
        cache = make_cache(cache_cls)
        first_tok, attn_mask = prefill(cache)
    except Exception as e:
        return None, [], f"prefill: {type(e).__name__}: {e}"
    multi = MultiDecodeK(base, K=K)
    compiled = torch.compile(multi, backend="tt")
    cur = first_tok; cache_pos = INPUT_LEN
    times = []; tokens_out = []
    try:
        for w in range(N_DECODE // K):
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
            tokens_out.extend(out_cpu)
            per_tok = (t1 - t0) * 1000 / K
            for _ in range(K):
                times.append(per_tok)
            cur = out_cpu[-1]; cache_pos += K
    except Exception as e:
        return None, [], f"decode: {type(e).__name__}: {e}"
    return times, tokens_out, None


print(f"\n[v3] === K={K} TPOT comparison across functional-cache variants ===\n", flush=True)
results = {}
for cache_cls, label in [
    (FuncCacheV1Where,          "V1: torch.where+expand_as"),
    (FuncCacheV2CloneIndexCopy, "V2: clone+index_copy_"),
    (FuncCacheV3SliceScatter,   "V3: slice_scatter"),
]:
    print(f"[v3] benching {label}", flush=True)
    times, toks, err = bench_k4(cache_cls, label)
    if err is not None:
        print(f"  FAIL: {err}", flush=True)
        results[label] = {"error": err}
    else:
        steady = times[SKIP_JIT:]
        mn = min(steady); mx = max(steady); mean = sum(steady) / len(steady)
        results[label] = {
            "tpot_mean": round(mean, 1),
            "tpot_min":  round(mn, 1),
            "tpot_max":  round(mx, 1),
            "tok_s":     round(1000 / mean, 1),
            "tokens":    toks[:4],
        }
        print(f"  K=4 TPOT: mean={mean:.1f}ms  min={mn:.1f}  max={mx:.1f}  ({1000/mean:.1f} tok/s)", flush=True)
        print(f"  first 4 toks: {toks[:4]}", flush=True)

print("\n" + "="*80)
print(f"{'Variant':<32} {'TPOT mean':>10} {'min':>7} {'max':>8} {'tok/s':>7}")
print("-"*80)
for label, r in results.items():
    if "error" in r:
        print(f"{label:<32} FAIL: {r['error'][:50]}")
    else:
        print(f"{label:<32} {r['tpot_mean']:>8.1f}ms {r['tpot_min']:>6.1f}  {r['tpot_max']:>6.1f}   {r['tok_s']:>6}")

print(f"\nBaseline K=1 reference: ~58ms/tok (from probe_t2_1_perf.py)")
print(f"Ideal K=4 (PJRT amortized): ~30ms/tok")
