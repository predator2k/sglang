"""Smoke test for tt-xla backend.

Validates that torch.compile(backend="tt") with StaticCache produces
correct output on Tenstorrent Blackhole hardware.

Usage (from tt-xla-slim container with PYTHONPATH set):
    SGLANG_TT_EXECUTION_BACKEND=tt_xla python test_tt_xla_smoke.py

Requires: tt-xla-slim container with pjrt-plugin-tt + transformers.
Does NOT require full SGLang install -- tests the core compilation
path independently.
"""

from __future__ import annotations

import json
import os
import sys
import time


def _init_spmd():
    """Initialize SPMD mode once for the entire test session."""
    import torch_xla
    import torch_xla.runtime as xr

    os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
    xr.set_device_type("TT")
    xr.use_spmd()
    return torch_xla.device()


def test_basic_matmul(device):
    """Verify basic matrix multiply on TT device via PJRT."""
    import torch

    x = torch.randn(4, 8).to(device)
    y = torch.randn(8, 4).to(device)
    z = x @ y
    result = z.cpu()
    assert result.shape == (4, 4), f"Expected (4, 4), got {result.shape}"
    assert torch.isfinite(result).all(), "Matmul produced non-finite values"
    print(f"  matmul: PASS (shape={result.shape})")


def test_tinyllama_generation(device):
    """Verify TinyLlama generation via torch.compile(backend='tt') + StaticCache."""
    import torch
    import torch_xla.runtime as xr
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import StaticCache
    num_devices = xr.global_runtime_device_count()
    print(f"  devices: {num_devices}")

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare inputs with static cache.
    prompt = "The capital of France is"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )
    max_cache_len = 64
    batch_size = 1

    static_cache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    cache_pos = torch.arange(0, inputs.input_ids.shape[1])
    prompt_len = inputs.input_ids.shape[1]
    full_attn_mask = torch.ones((batch_size, max_cache_len), dtype=inputs.attention_mask.dtype)
    full_attn_mask[:, :prompt_len] = inputs.attention_mask

    # Move to device.
    for layer in static_cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    input_ids = inputs.input_ids.to(device)
    cache_pos_dev = cache_pos.to(device)
    full_attn_mask_dev = full_attn_mask.to(device)
    model = model.to(device)

    # Compile with tt backend.
    compiled_model = torch.compile(model, backend="tt")

    # Prefill.
    t0 = time.perf_counter()
    with torch.no_grad():
        output = compiled_model(
            input_ids=input_ids,
            past_key_values=static_cache,
            cache_position=cache_pos_dev,
            use_cache=True,
            attention_mask=full_attn_mask_dev,
        )
    prefill_time = time.perf_counter() - t0
    logits = output.logits.to("cpu")
    first_token = logits[0, -1].argmax().item()
    first_word = tokenizer.decode(first_token)
    print(f"  prefill: {prefill_time:.2f}s, first_token={first_word!r}")

    # Generate 5 decode tokens.
    generated = [first_token]
    cur_token = first_token
    cur_cache_pos = cache_pos_dev
    for i in range(5):
        next_input = torch.tensor([[cur_token]]).to(device)
        host_cp = cur_cache_pos.to("cpu")
        new_cp = torch.tensor([host_cp[-1:] + 1]).to(device)
        cur_cache_pos = new_cp

        with torch.no_grad():
            output = compiled_model(
                input_ids=next_input,
                past_key_values=static_cache,
                cache_position=cur_cache_pos,
                use_cache=True,
                attention_mask=full_attn_mask_dev,
            )
        logits = output.logits.to("cpu")
        cur_token = logits[0, -1].argmax().item()
        generated.append(cur_token)

    text = tokenizer.decode(generated)
    full_output = prompt + text
    print(f"  generated: {full_output!r}")

    # Validate.
    assert "Paris" in text or "paris" in text.lower(), (
        f"Expected 'Paris' in output, got: {text!r}"
    )
    assert "!!!" not in text, f"Garbage output detected: {text!r}"
    print(f"  generation: PASS")

    return {
        "model": model_name,
        "prompt": prompt,
        "output": full_output,
        "prefill_time_s": round(prefill_time, 2),
        "num_devices": num_devices,
    }


def main():
    print("tt-xla smoke test")
    print("=" * 40)

    # Initialize SPMD once, before any tests.
    device = _init_spmd()

    results = {}

    print("[1/2] Basic matmul...")
    test_basic_matmul(device)
    results["basic_matmul"] = "PASS"

    print("[2/2] TinyLlama generation...")
    gen_result = test_tinyllama_generation(device)
    results["tinyllama_generation"] = gen_result

    print()
    print("ALL TESTS PASSED")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
