"""Phase 0 probe: Can TTFunctionalCache survive torch.compile on TT?

Tests three things, in order:
  1. K=1 single-decode w/ TTFunctionalCache (22L TinyLlama) — basic compile + execute
  2. K=2 multi-decode wrapper                                 — first multi-call test
  3. K=4 multi-decode wrapper                                 — A1 target

Gates per test:
  - Compile success (no exception)
  - Execute success (no Error 13)
  - Bit-exact tokens vs StaticCache baseline
  - dynamo frames compiled (target: 1 per K-value across the loop)
  - dynamo recompiles in steady state (target: 0)
  - dynamo graph breaks (target: 0)
  - FX graph mutation-op count (target: <= 1 per layer per K-step; ideally 0 for index_copy_)
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
from transformers.cache_utils import Cache, StaticCache

device = torch_xla.device()
N = xr.global_runtime_device_count()
print(f"[probe] device={device}, n_devices={N}", flush=True)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 64
SEED = 42
N_DECODE_REF = 5    # reference tokens
N_DECODE_K1 = 5     # K=1 probe
N_DECODE_K2 = 4     # K=2 probe (2 wrapper calls => 4 tokens)
N_DECODE_K4 = 4     # K=4 probe (1 wrapper call => 4 tokens)


# ====================================================================
# TTFunctionalCache: subclass StaticCache, override update() with torch.where
# ====================================================================
class TTFunctionalCache(StaticCache):
    """Functional KV update — torch.where instead of index_copy_. Decode-only (q_len=1)."""

    def __init__(self, config, max_batch_size, max_cache_len, device, dtype):
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        # Track latest per-layer K/V as Python list (the suspect rebind)
        self._latest_k = [None] * len(self.layers)
        self._latest_v = [None] * len(self.layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Prefill / multi-token: fall through to StaticCache parent (uses index_copy_, proven for q_len > 1)
        if key_states.shape[-2] != 1:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Decode (q_len=1): functional path
        cp = cache_kwargs["cache_position"]            # [1]
        # Read prior K/V either from stored latest (after first decode) or from layer.keys
        prior_k = self._latest_k[layer_idx] if self._latest_k[layer_idx] is not None else self.layers[layer_idx].keys
        prior_v = self._latest_v[layer_idx] if self._latest_v[layer_idx] is not None else self.layers[layer_idx].values

        max_len = prior_k.shape[2]
        pos = torch.arange(max_len, device=cp.device)
        write_mask = (pos == cp[0]).view(1, 1, max_len, 1)

        new_k = torch.where(write_mask, key_states.expand_as(prior_k), prior_k)
        new_v = torch.where(write_mask, value_states.expand_as(prior_v), prior_v)

        # Python rebind — the operation we want to verify dynamo handles
        self._latest_k[layer_idx] = new_k
        self._latest_v[layer_idx] = new_v
        return new_k, new_v


# ====================================================================
# Multi-token decode wrapper (A1 K=2/K=4)
# ====================================================================
class MultiDecodeK(nn.Module):
    def __init__(self, m, K):
        super().__init__()
        self.m = m
        self.K = K

    def forward(self, t_in, cps, poss, attention_mask, past_key_values):
        toks = []
        cur = t_in
        for k in range(self.K):
            out = self.m(
                input_ids=cur,
                past_key_values=past_key_values,
                cache_position=cps[k],
                use_cache=True,
                attention_mask=attention_mask,
                position_ids=poss[k],
            )
            cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(cur)
        return torch.cat(toks, dim=-1)


# ====================================================================
# Dynamo counters helper
# ====================================================================
def reset_counters():
    import torch._dynamo
    torch._dynamo.utils.counters.clear()
    torch._dynamo.reset()

def snapshot_counters():
    import torch._dynamo
    c = torch._dynamo.utils.counters
    return {
        "frames_ok": dict(c.get("frames", {})).get("ok", 0),
        "frames_total": dict(c.get("frames", {})).get("total", 0),
        "graph_break": sum(dict(c.get("graph_break", {})).values()),
        "graph_break_reasons": dict(c.get("graph_break", {})),
        "recompiles": dict(c.get("stats", {})).get("recompiles", 0),
    }

def print_counters(label, snap):
    print(f"[probe] {label}: frames_ok={snap['frames_ok']}  "
          f"frames_total={snap['frames_total']}  "
          f"graph_breaks={snap['graph_break']}  "
          f"recompiles={snap['recompiles']}", flush=True)
    if snap["graph_break_reasons"]:
        print(f"          break reasons: {snap['graph_break_reasons']}", flush=True)


# ====================================================================
# Build model + prefill via StaticCache (canonical, known-working path)
# ====================================================================
print(f"[probe] loading {MODEL}", flush=True)
config = AutoConfig.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True
)
base.eval()
base = base.to(device)

nkv = config.num_key_value_heads
hd = config.hidden_size // config.num_attention_heads
num_layers = config.num_hidden_layers
print(f"[probe] layers={num_layers}, num_kv_heads={nkv}, head_dim={hd}", flush=True)


def make_static_cache():
    c = StaticCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                    device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    return c


def make_functional_cache():
    c = TTFunctionalCache(config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
                          device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=1, num_heads=nkv, head_dim=hd,
                           dtype=torch.bfloat16, device="cpu")
    for layer in c.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    return c


def prefill_into(cache):
    random.seed(SEED)
    rand_ids = torch.tensor([[random.randint(10, config.vocab_size - 1) for _ in range(INPUT_LEN)]])
    attn_mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)
    compiled = torch.compile(base, backend="tt")
    with torch.no_grad():
        out = compiled(
            input_ids=rand_ids.to(device),
            past_key_values=cache,
            cache_position=torch.arange(INPUT_LEN).to(device),
            use_cache=True,
            attention_mask=attn_mask.to(device),
            position_ids=torch.arange(INPUT_LEN, dtype=torch.long).unsqueeze(0).to(device),
        )
    torch_xla.sync()
    return out.logits[:, -1, :].argmax(-1).to("cpu").item(), attn_mask


# ====================================================================
# Test 0: BASELINE — StaticCache, K=1 decode
# ====================================================================
print("\n[probe] ============================================================")
print("[probe] TEST 0: BASELINE — StaticCache K=1")
print("[probe] ============================================================")

reset_counters()
static_cache = make_static_cache()
first_tok, attn_mask = prefill_into(static_cache)
print(f"[probe] prefill first_tok = {first_tok}", flush=True)

compiled_base = torch.compile(base, backend="tt")
ref_tokens = []
cur = first_tok
cache_pos = INPUT_LEN
for i in range(N_DECODE_REF):
    next_ids = torch.tensor([[cur]]).to(device)
    new_cp = torch.tensor([cache_pos]).to(device)
    new_pos = torch.tensor([[cache_pos]], dtype=torch.long).to(device)
    attn_mask[:, cache_pos] = 1
    md = attn_mask.to(device)
    with torch.no_grad():
        out = compiled_base(input_ids=next_ids, past_key_values=static_cache,
                            cache_position=new_cp, use_cache=True,
                            attention_mask=md, position_ids=new_pos)
    torch_xla.sync()
    nxt = out.logits[:, -1, :].argmax(-1).to("cpu").item()
    ref_tokens.append(nxt)
    cur = nxt
    cache_pos += 1

print_counters("BASELINE counters", snapshot_counters())
print(f"[probe] reference tokens: {ref_tokens}", flush=True)


# ====================================================================
# Test 1: TTFunctionalCache K=1
# ====================================================================
print("\n[probe] ============================================================")
print("[probe] TEST 1: TTFunctionalCache K=1")
print("[probe] ============================================================")

reset_counters()

fc_status = "PASS"
fc_error = None
fc_tokens = []
try:
    func_cache = make_functional_cache()
    fc_first_tok, fc_attn_mask = prefill_into(func_cache)
    print(f"[probe] prefill via TTFunctionalCache: first_tok = {fc_first_tok}", flush=True)
except Exception as e:
    fc_status = "FAIL"
    fc_error = f"prefill: {type(e).__name__}: {e}"
    print(f"  K=1 prefill FAIL: {fc_error}", flush=True)
    traceback.print_exc()
    fc_first_tok = None
    fc_attn_mask = None
# After prefill via the inherited path, layer.keys has been updated in-place by index_copy_
# We need _latest to be in sync. Since prefill goes through StaticCache.update path (parent),
# our _latest_k/_v stays None and we read from self.layers[layer_idx].keys — that's correct.

compiled_fc = torch.compile(base, backend="tt")
cur = fc_first_tok
cache_pos = INPUT_LEN

try:
    if fc_first_tok is None:
        raise RuntimeError("prefill failed earlier, skipping decode")
    for i in range(N_DECODE_K1):
        next_ids = torch.tensor([[cur]]).to(device)
        new_cp = torch.tensor([cache_pos]).to(device)
        new_pos = torch.tensor([[cache_pos]], dtype=torch.long).to(device)
        fc_attn_mask[:, cache_pos] = 1
        md = fc_attn_mask.to(device)
        with torch.no_grad():
            out = compiled_fc(input_ids=next_ids, past_key_values=func_cache,
                              cache_position=new_cp, use_cache=True,
                              attention_mask=md, position_ids=new_pos)
        torch_xla.sync()
        nxt = out.logits[:, -1, :].argmax(-1).to("cpu").item()
        fc_tokens.append(nxt)
        cur = nxt
        cache_pos += 1
        print(f"  K=1 iter {i}: tok={nxt}", flush=True)
except Exception as e:
    fc_status = "FAIL"
    fc_error = f"{type(e).__name__}: {e}"
    print(f"  K=1 FAIL: {fc_error}", flush=True)
    traceback.print_exc()

snap_k1 = snapshot_counters()
print_counters("K=1 counters", snap_k1)
print(f"[probe] K=1 tokens: {fc_tokens}", flush=True)
print(f"[probe] K=1 bit-exact vs ref: {fc_tokens == ref_tokens}", flush=True)
print(f"[probe] K=1 STATUS: {fc_status}", flush=True)


# ====================================================================
# Test 2: K=2 wrapper
# ====================================================================
print("\n[probe] ============================================================")
print("[probe] TEST 2: TTFunctionalCache + K=2 wrapper")
print("[probe] ============================================================")

reset_counters()
k2_status = "PASS"
k2_error = None
k2_tokens = []
try:
    func_cache2 = make_functional_cache()
    fc2_first_tok, fc2_attn_mask = prefill_into(func_cache2)
    print(f"[probe] prefill first_tok = {fc2_first_tok}", flush=True)
except Exception as e:
    k2_status = "FAIL"
    k2_error = f"prefill: {type(e).__name__}: {e}"
    print(f"  K=2 prefill FAIL: {k2_error}", flush=True)
    traceback.print_exc()
    fc2_first_tok = None
    fc2_attn_mask = None

multi_k2 = MultiDecodeK(base, K=2)
compiled_k2 = torch.compile(multi_k2, backend="tt")

cur = fc2_first_tok
cache_pos = INPUT_LEN

try:
    if fc2_first_tok is None:
        raise RuntimeError("prefill failed, skipping decode")
    for w in range(N_DECODE_K2 // 2):
        # Pre-extend mask for the next 2 positions
        for k in range(2):
            fc2_attn_mask[:, cache_pos + k] = 1
        md = fc2_attn_mask.to(device)

        t_in = torch.tensor([[cur]]).to(device)
        cps = tuple(torch.tensor([cache_pos + k]).to(device) for k in range(2))
        poss = tuple(torch.tensor([[cache_pos + k]], dtype=torch.long).to(device) for k in range(2))

        with torch.no_grad():
            out_tokens = compiled_k2(t_in, cps, poss, md, past_key_values=func_cache2)
        out_cpu = out_tokens.to("cpu").tolist()[0]
        k2_tokens.extend(out_cpu)
        cur = out_cpu[-1]
        cache_pos += 2
        print(f"  K=2 wrap {w}: toks={out_cpu}", flush=True)
except Exception as e:
    k2_status = "FAIL"
    k2_error = f"{type(e).__name__}: {e}"
    print(f"  K=2 FAIL: {k2_error}", flush=True)
    traceback.print_exc()

snap_k2 = snapshot_counters()
print_counters("K=2 counters", snap_k2)
print(f"[probe] K=2 tokens: {k2_tokens}", flush=True)
print(f"[probe] K=2 bit-exact vs ref[:4]: {k2_tokens == ref_tokens[:4]}", flush=True)
print(f"[probe] K=2 STATUS: {k2_status}", flush=True)


# ====================================================================
# Test 3: K=4 wrapper
# ====================================================================
print("\n[probe] ============================================================")
print("[probe] TEST 3: TTFunctionalCache + K=4 wrapper (A1 target)")
print("[probe] ============================================================")

reset_counters()
k4_status = "PASS"
k4_error = None
k4_tokens = []
try:
    func_cache4 = make_functional_cache()
    fc4_first_tok, fc4_attn_mask = prefill_into(func_cache4)
    print(f"[probe] prefill first_tok = {fc4_first_tok}", flush=True)
except Exception as e:
    k4_status = "FAIL"
    k4_error = f"prefill: {type(e).__name__}: {e}"
    print(f"  K=4 prefill FAIL: {k4_error}", flush=True)
    traceback.print_exc()
    fc4_first_tok = None
    fc4_attn_mask = None

multi_k4 = MultiDecodeK(base, K=4)
compiled_k4 = torch.compile(multi_k4, backend="tt")

cur = fc4_first_tok
cache_pos = INPUT_LEN

try:
    if fc4_first_tok is None:
        raise RuntimeError("prefill failed, skipping decode")
    for w in range(N_DECODE_K4 // 4):
        for k in range(4):
            fc4_attn_mask[:, cache_pos + k] = 1
        md = fc4_attn_mask.to(device)

        t_in = torch.tensor([[cur]]).to(device)
        cps = tuple(torch.tensor([cache_pos + k]).to(device) for k in range(4))
        poss = tuple(torch.tensor([[cache_pos + k]], dtype=torch.long).to(device) for k in range(4))

        with torch.no_grad():
            out_tokens = compiled_k4(t_in, cps, poss, md, past_key_values=func_cache4)
        out_cpu = out_tokens.to("cpu").tolist()[0]
        k4_tokens.extend(out_cpu)
        cur = out_cpu[-1]
        cache_pos += 4
        print(f"  K=4 wrap {w}: toks={out_cpu}", flush=True)
except Exception as e:
    k4_status = "FAIL"
    k4_error = f"{type(e).__name__}: {e}"
    print(f"  K=4 FAIL: {k4_error}", flush=True)
    traceback.print_exc()

snap_k4 = snapshot_counters()
print_counters("K=4 counters", snap_k4)
print(f"[probe] K=4 tokens: {k4_tokens}", flush=True)
print(f"[probe] K=4 bit-exact vs ref[:4]: {k4_tokens == ref_tokens[:4]}", flush=True)
print(f"[probe] K=4 STATUS: {k4_status}", flush=True)


# ====================================================================
# Summary
# ====================================================================
print("\n[probe] ============================================================")
print("[probe] SUMMARY")
print("[probe] ============================================================")
print(f"{'Test':<25} {'Status':<8} {'Bit-exact':<12} {'frames':<8} {'recomp':<8} {'breaks':<8}")
print("-" * 75)
print(f"{'BASELINE StaticCache':<25} {'PASS':<8} {'(ref)':<12} {'':<8} {'':<8} {'':<8}")
print(f"{'K=1 TTFunctionalCache':<25} {fc_status:<8} {str(fc_tokens == ref_tokens):<12} "
      f"{snap_k1['frames_ok']:<8} {snap_k1['recompiles']:<8} {snap_k1['graph_break']:<8}")
print(f"{'K=2 MultiDecodeK':<25} {k2_status:<8} {str(k2_tokens == ref_tokens[:4]):<12} "
      f"{snap_k2['frames_ok']:<8} {snap_k2['recompiles']:<8} {snap_k2['graph_break']:<8}")
print(f"{'K=4 MultiDecodeK':<25} {k4_status:<8} {str(k4_tokens == ref_tokens[:4]):<12} "
      f"{snap_k4['frames_ok']:<8} {snap_k4['recompiles']:<8} {snap_k4['graph_break']:<8}")

print("\n[probe] DECISION GATE:")
if fc_status == "PASS" and fc_tokens == ref_tokens:
    print("  ✓ TTFunctionalCache K=1 works AND is bit-exact")
else:
    print(f"  ✗ TTFunctionalCache K=1 broken: status={fc_status}, bit-exact={fc_tokens == ref_tokens}")
if k4_status == "PASS" and k4_tokens == ref_tokens[:4]:
    print("  ✓ A1 K=4 works AND is bit-exact — DESIGN GOES AHEAD")
elif k2_status == "PASS" and k2_tokens == ref_tokens[:4]:
    print("  ~ K=2 works but K=4 doesn't — partial design (K=2 only)")
else:
    print("  ✗ Multi-decode wrapper BROKEN — fall back to per-arch attention patching")
