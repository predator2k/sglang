"""T2.1 acceptance: TTFunctionalCache bit-exact vs StaticCache on TinyLlama."""
import os, random
import pytest

os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch
import torch_xla, torch_xla.runtime as xr

xr.set_device_type("TT")
xr.use_spmd()

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import StaticCache

from sglang.srt.hardware_backend.tenstorrent.models.tt_functional_cache import (
    TTFunctionalCache,
)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_LEN = 2048
CACHE_SIZE = INPUT_LEN + 64
SEED = 42
N_DECODE = 5


@pytest.fixture(scope="module")
def loaded_model():
    config = AutoConfig.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", use_cache=True
    )
    base.eval()
    base = base.to(torch_xla.device())
    return base, config


def _make_cache(cache_cls, config):
    c = cache_cls(
        config=config, max_batch_size=1, max_cache_len=CACHE_SIZE,
        device="cpu", dtype=torch.bfloat16,
    )
    c.early_initialization(
        batch_size=1,
        num_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=torch.bfloat16,
        device="cpu",
    )
    dev = torch_xla.device()
    for layer in c.layers:
        layer.keys = layer.keys.to(dev)
        layer.values = layer.values.to(dev)
    return c


def _decode_n(base, cache, n_tokens):
    device = torch_xla.device()
    compiled = torch.compile(base, backend="tt")
    random.seed(SEED)
    rand_ids = torch.tensor(
        [[random.randint(10, base.config.vocab_size - 1) for _ in range(INPUT_LEN)]]
    )
    attn_mask = torch.ones((1, CACHE_SIZE), dtype=torch.int32)
    with torch.no_grad():
        out = compiled(
            input_ids=rand_ids.to(device), past_key_values=cache,
            cache_position=torch.arange(INPUT_LEN).to(device),
            use_cache=True, attention_mask=attn_mask.to(device),
            position_ids=torch.arange(INPUT_LEN, dtype=torch.long).unsqueeze(0).to(device),
        )
    torch_xla.sync()
    cur = out.logits[:, -1, :].argmax(-1).to("cpu").item()
    cache_pos = INPUT_LEN
    tokens = []
    for _ in range(n_tokens):
        next_ids = torch.tensor([[cur]]).to(device)
        new_cp = torch.tensor([cache_pos]).to(device)
        new_pos = torch.tensor([[cache_pos]], dtype=torch.long).to(device)
        attn_mask[:, cache_pos] = 1
        with torch.no_grad():
            out = compiled(
                input_ids=next_ids, past_key_values=cache,
                cache_position=new_cp, use_cache=True,
                attention_mask=attn_mask.to(device), position_ids=new_pos,
            )
        torch_xla.sync()
        cur = out.logits[:, -1, :].argmax(-1).to("cpu").item()
        tokens.append(cur)
        cache_pos += 1
    return tokens


def test_functional_cache_bitexact_vs_static(loaded_model):
    base, config = loaded_model
    ref_tokens = _decode_n(base, _make_cache(StaticCache, config), N_DECODE)
    func_tokens = _decode_n(base, _make_cache(TTFunctionalCache, config), N_DECODE)
    assert func_tokens == ref_tokens, (
        f"Bit-exact mismatch: TTFunctionalCache={func_tokens} vs StaticCache={ref_tokens}"
    )
