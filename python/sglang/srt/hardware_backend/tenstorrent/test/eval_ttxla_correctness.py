"""v143 Correctness evaluation for tt-xla models.

Runs math, factual, and simple GSM8K-style tests on each model
using torch.compile(backend="tt") + StaticCache + Int32 mask pattern.

Usage:
    # Run all models:
    python3 eval_ttxla_correctness.py

    # Run specific models:
    EVAL_MODELS="TinyLlama-1.1B,Qwen3-0.6B" python3 eval_ttxla_correctness.py

Output: Writes JSON to v143_ttxla_correctness_eval.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

MAX_CACHE_LEN = 64
# Generate enough tokens to get the answer (but limited by cache)
NUM_DECODE = 30
FIXTURE_DIR = "/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures"
OUTPUT_FILE = os.path.join(FIXTURE_DIR, "v143_ttxla_correctness_eval.json")

# ---- Model registry ----
# short_name -> (local_path, family, param_count, model_card_gsm8k)
MODEL_REGISTRY = {
    "TinyLlama-1.1B": (
        "/models/TinyLlama-1.1B-Chat-v1.0", "tinyllama", "1.1B", "~10%",
    ),
    "SmolLM2-135M": (
        "/models/SmolLM2-135M-Instruct", "smollm", "0.1B", "N/A (too small)",
    ),
    "Qwen2.5-0.5B": (
        "/models/Qwen2.5-0.5B", "qwen2.5", "0.5B", "~36%",
    ),
    "Qwen2.5-3B": (
        "/models/Qwen2.5-3B", "qwen2.5", "3B", "~65%",
    ),
    "Qwen3-0.6B": (
        "/models/Qwen3-0.6B", "qwen3", "0.6B", "~50%",
    ),
    "Qwen3-1.7B": (
        "/models/Qwen3-1.7B", "qwen3", "1.7B", "~65%",
    ),
    "Qwen3-4B": (
        "/models/Qwen3-4B", "qwen3", "4B", "~75%",
    ),
    "Qwen3-8B": (
        "/models/Qwen3-8B", "qwen3", "8B", "~80%",
    ),
    "Qwen3-14B": (
        "/models/Qwen3-14B", "qwen3", "14B", "~85%",
    ),
    "Llama-3.1-8B": (
        "/models/Llama-3.1-8B-Instruct", "llama", "8B", "~55%",
    ),
    "Mistral-7B": (
        "/models/Mistral-7B-Instruct-v0.3", "mistral", "7B", "~40%",
    ),
    "Nemotron-Mini-4B": (
        "/models/Nemotron-Mini-4B-Instruct", "nemotron", "4B", "~45%",
    ),
    "StableLM-2-1.6B": (
        "/models/stablelm-2-1_6b", "stablelm", "1.6B", "~30%",
    ),
    "Phi-4-mini": (
        "/models/Phi-4-mini-instruct", "phi", "3.8B", "~70%",
    ),
}

# ---- Test sets ----

MATH_TESTS = [
    ("What is 2+3?", ["5"]),
    ("What is 7*8?", ["56"]),
    ("What is 100/4?", ["25"]),
    ("What is 15+27?", ["42"]),
    ("What is 9*9?", ["81"]),
    ("What is 144/12?", ["12"]),
    ("What is 33+67?", ["100"]),
    ("What is 6*7?", ["42"]),
    ("What is 50-17?", ["33"]),
    ("What is 8*12?", ["96"]),
]

FACTUAL_TESTS = [
    ("The capital of France is", ["paris"]),
    ("The capital of Japan is", ["tokyo"]),
    ("The chemical symbol for gold is", ["au"]),
    ("The chemical symbol for water is", ["h2o"]),
    ("The author of Romeo and Juliet is", ["shakespeare"]),
    ("The speed of light is approximately", ["300", "3"]),
    ("The largest planet in our solar system is", ["jupiter"]),
    ("The boiling point of water is", ["100", "212"]),
    ("Pi is approximately", ["3.14"]),
    ("The square root of 64 is", ["8"]),
]

GSM8K_SIMPLE_TESTS = [
    ("Tom has 5 apples. He buys 3 more. How many apples does Tom have? Answer:", ["8"]),
    ("A store sells pencils for $2 each. If you buy 4, how much do you pay? Answer:", ["8", "$8"]),
    ("Sarah has 20 cookies and gives away 7. How many does she have left? Answer:", ["13"]),
    ("A car travels at 60 mph for 2 hours. How far does it go? Answer:", ["120"]),
    ("There are 24 students split into 4 groups. How many per group? Answer:", ["6"]),
]


def check_answer(output_text: str, expected_patterns: list[str]) -> bool:
    """Check if any expected pattern appears in output (case-insensitive)."""
    output_lower = output_text.lower()
    for pattern in expected_patterns:
        if pattern.lower() in output_lower:
            return True
    return False


def generate_text(
    compiled_model, tokenizer, cache_config, device, prompt: str, max_new_tokens: int = NUM_DECODE
) -> str:
    """Generate text using the compiled model with StaticCache.

    NOTE: cache_config is a tuple (config, nkv, hd) used to create a fresh cache each time.
    On tt-xla, in-place zero_() and torch.zeros_like() on device tensors corrupt the
    SPMD graph, so we recreate the cache from scratch for each prompt.
    """
    config, nkv, hd = cache_config

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tok = tokenizer(
        prompt, return_tensors="pt", max_length=32,
        truncation=True,
        padding="max_length", padding_side="left",
        return_attention_mask=True,
    )
    prompt_ids = tok.input_ids
    prompt_len = prompt_ids.shape[1]

    # Create fresh cache for each prompt (avoids in-place ops on device tensors)
    cache = StaticCache(
        config=config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN,
        device="cpu", dtype=torch.bfloat16,
    )
    cache.early_initialization(
        batch_size=1, num_heads=nkv, head_dim=hd,
        dtype=torch.bfloat16, device="cpu",
    )
    for layer in cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)

    # Int32 attention mask
    attn_mask = torch.ones((1, MAX_CACHE_LEN), dtype=torch.int32)
    attn_mask[:, :prompt_len] = tok.attention_mask
    cp = torch.arange(0, prompt_len)

    ids_dev = prompt_ids.to(device)
    cp_dev = cp.to(device)
    mask_dev = attn_mask.to(device)

    # Prefill
    with torch.no_grad():
        out = compiled_model(
            input_ids=ids_dev, past_key_values=cache,
            cache_position=cp_dev, use_cache=True,
            attention_mask=mask_dev,
        )
    logits = out.logits.to("cpu")
    first_tok = logits[0, -1].argmax().item()

    # Decode
    generated = [first_tok]
    cur = first_tok
    cache_pos = prompt_len

    eos_id = tokenizer.eos_token_id

    for i in range(max_new_tokens):
        if cache_pos >= MAX_CACHE_LEN - 1:
            break  # Can't exceed cache
        if cur == eos_id:
            break

        next_ids = torch.tensor([[cur]]).to(device)
        new_cp = torch.tensor([cache_pos]).to(device)

        with torch.no_grad():
            out = compiled_model(
                input_ids=next_ids, past_key_values=cache,
                cache_position=new_cp, use_cache=True,
                attention_mask=mask_dev,
            )
        logits = out.logits.to("cpu")
        cur = logits[0, -1].argmax().item()
        generated.append(cur)
        cache_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True)


def run_test_suite(
    compiled_model, tokenizer, cache_config, device, test_name: str, tests: list
) -> dict:
    """Run a test suite and return results."""
    passed = 0
    total = len(tests)
    failures = []

    for i, (prompt, expected_patterns) in enumerate(tests):
        try:
            output = generate_text(compiled_model, tokenizer, cache_config, device, prompt)
            is_pass = check_answer(output, expected_patterns)
            if is_pass:
                passed += 1
                print(f"    [{test_name} {i+1}/{total}] PASS: {prompt[:40]}... -> {output[:60]}", flush=True)
            else:
                failures.append({
                    "prompt": prompt,
                    "expected": ", ".join(expected_patterns),
                    "got": output[:200],
                })
                print(f"    [{test_name} {i+1}/{total}] FAIL: {prompt[:40]}... -> {output[:60]} (expected: {expected_patterns})", flush=True)
        except Exception as e:
            failures.append({
                "prompt": prompt,
                "expected": ", ".join(expected_patterns),
                "got": f"ERROR: {str(e)[:150]}",
            })
            print(f"    [{test_name} {i+1}/{total}] ERROR: {prompt[:40]}... -> {e}", flush=True)

    return {
        "score": f"{passed}/{total}",
        "passed": passed,
        "total": total,
        "failures": failures,
    }


def eval_one_model(model_name: str, model_path: str, device) -> dict:
    """Evaluate a single model on all test suites."""
    print(f"\n{'='*60}", flush=True)
    print(f"  Evaluating: {model_name} ({model_path})", flush=True)
    print(f"{'='*60}", flush=True)

    result = {
        "model_path": model_path,
        "status": "PENDING",
        "math_score": "",
        "factual_score": "",
        "gsm8k_simple_score": "",
        "total_score": "",
        "accuracy": 0.0,
        "failures": [],
        "error": "",
        "prefill_time_s": None,
    }

    try:
        # Load model
        t0 = time.time()
        print(f"  Loading model...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        attn_impl = "eager"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                use_cache=True, trust_remote_code=True,
                attn_implementation="eager",
            )
        except Exception:
            attn_impl = "default"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                use_cache=True, trust_remote_code=True,
            )
        model.eval()
        config = model.config
        print(f"  Loaded in {time.time()-t0:.1f}s (attn={attn_impl})", flush=True)

        # Move + compile
        model = model.to(device)
        compiled = torch.compile(model, backend="tt")

        # Cache config (cache is recreated per prompt to avoid in-place ops)
        nkv = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", None))
        if nkv is None:
            nkv = config.num_attention_heads
        hd = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        cache_config = (config, nkv, hd)

        # Warmup with a simple prompt
        print(f"  Warming up (JIT compile)...", flush=True)
        t0 = time.perf_counter()
        _ = generate_text(compiled, tokenizer, cache_config, device, "Hello", max_new_tokens=3)
        warmup_time = time.perf_counter() - t0
        result["prefill_time_s"] = round(warmup_time, 2)
        print(f"  Warmup done in {warmup_time:.1f}s", flush=True)

        # Run test suites
        print(f"\n  --- Math Tests (10) ---", flush=True)
        math_results = run_test_suite(compiled, tokenizer, cache_config, device, "math", MATH_TESTS)

        print(f"\n  --- Factual Tests (10) ---", flush=True)
        factual_results = run_test_suite(compiled, tokenizer, cache_config, device, "factual", FACTUAL_TESTS)

        print(f"\n  --- GSM8K Simple Tests (5) ---", flush=True)
        gsm8k_results = run_test_suite(compiled, tokenizer, cache_config, device, "gsm8k", GSM8K_SIMPLE_TESTS)

        # Aggregate
        total_passed = math_results["passed"] + factual_results["passed"] + gsm8k_results["passed"]
        total_questions = math_results["total"] + factual_results["total"] + gsm8k_results["total"]
        all_failures = math_results["failures"] + factual_results["failures"] + gsm8k_results["failures"]

        result["math_score"] = math_results["score"]
        result["factual_score"] = factual_results["score"]
        result["gsm8k_simple_score"] = gsm8k_results["score"]
        result["total_score"] = f"{total_passed}/{total_questions}"
        result["accuracy"] = round(total_passed / total_questions, 4) if total_questions > 0 else 0.0
        result["failures"] = all_failures
        result["status"] = "PASS"

        print(f"\n  TOTAL: {total_passed}/{total_questions} ({result['accuracy']*100:.1f}%)", flush=True)

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)[:500]
        print(f"  FATAL: {e}", flush=True)
        traceback.print_exc()

    finally:
        # Cleanup GPU memory
        try:
            del model, compiled
        except Exception:
            pass
        gc.collect()

    return result


def load_existing_results() -> dict:
    """Load existing results file if it exists (for resume support)."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_results(data: dict):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}", flush=True)


def main():
    # Initialize SPMD
    xr.set_device_type("TT")
    xr.use_spmd()
    device = torch_xla.device()
    num_devices = xr.global_runtime_device_count()
    print(f"[init] device={device}, num_devices={num_devices}", flush=True)

    # Determine which models to run
    env_models = os.environ.get("EVAL_MODELS", "")
    if env_models:
        model_names = [m.strip() for m in env_models.split(",") if m.strip()]
    else:
        model_names = list(MODEL_REGISTRY.keys())

    # Load or create results structure
    existing = load_existing_results()
    if existing and existing.get("version") == "v143":
        results = existing
        print(f"[resume] Loaded existing v143 results", flush=True)
    else:
        results = {
            "version": "v143",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "max_cache_len": MAX_CACHE_LEN,
            "num_decode_tokens": NUM_DECODE,
            "num_devices": num_devices,
            "limitation": "max_cache_len=64 limits output to ~30-50 tokens, preventing full reasoning chains",
            "models": {},
        }

    model_card_gsm8k_map = {
        "TinyLlama-1.1B": "~10%",
        "SmolLM2-135M": "N/A (too small)",
        "Qwen2.5-0.5B": "~36%",
        "Qwen2.5-3B": "~65%",
        "Qwen3-0.6B": "~50%",
        "Qwen3-1.7B": "~65%",
        "Qwen3-4B": "~75%",
        "Qwen3-8B": "~80%",
        "Qwen3-14B": "~85%",
        "Llama-3.1-8B": "~55%",
        "Mistral-7B": "~40%",
        "Nemotron-Mini-4B": "~45%",
        "StableLM-2-1.6B": "~30%",
        "Phi-4-mini": "~70%",
    }

    # Models to skip due to known issues
    SKIP_MODELS = {
        "Phi-4-mini": "rope_scaling long_factor/short_factor causes recompilation every decode step (~14.5s/tok); 25 prompts x 30 tokens = ~3h, impractical",
    }

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            print(f"\n[skip] {model_name} not in registry", flush=True)
            continue

        # Check if model should be skipped
        if model_name in SKIP_MODELS:
            print(f"\n[skip] {model_name}: {SKIP_MODELS[model_name]}", flush=True)
            results["models"][model_name] = {
                "status": "SKIP",
                "error": SKIP_MODELS[model_name],
                "model_card_gsm8k": model_card_gsm8k_map.get(model_name, "N/A"),
                "family": MODEL_REGISTRY[model_name][1],
                "param_count": MODEL_REGISTRY[model_name][2],
            }
            save_results(results)
            continue

        # Check if already done
        if model_name in results["models"] and results["models"][model_name].get("status") == "PASS":
            print(f"\n[skip] {model_name} already evaluated (PASS)", flush=True)
            continue

        model_path, family, param_count, gsm8k_card = MODEL_REGISTRY[model_name]

        model_result = eval_one_model(model_name, model_path, device)
        model_result["model_card_gsm8k"] = model_card_gsm8k_map.get(model_name, "N/A")
        model_result["family"] = family
        model_result["param_count"] = param_count

        # Add contextual note
        if model_result["status"] == "PASS":
            acc = model_result["accuracy"]
            if acc >= 0.8:
                model_result["note"] = "Strong performance across all test categories"
            elif acc >= 0.6:
                model_result["note"] = "Good performance; some failures likely due to short output limit (max_cache_len=64)"
            else:
                model_result["note"] = "Limited performance; small model or output truncation affects results"

        results["models"][model_name] = model_result

        # Save after each model (resume support)
        save_results(results)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    pass_count = sum(1 for m in results["models"].values() if m.get("status") == "PASS")
    fail_count = sum(1 for m in results["models"].values() if m.get("status") == "FAIL")
    print(f"  Models evaluated: {len(results['models'])}", flush=True)
    print(f"  PASS: {pass_count}, FAIL: {fail_count}", flush=True)

    for name, m in results["models"].items():
        if m.get("status") == "PASS":
            print(f"    {name}: {m['total_score']} ({m['accuracy']*100:.1f}%) [card GSM8K: {m.get('model_card_gsm8k', 'N/A')}]", flush=True)
        else:
            print(f"    {name}: {m['status']} - {m.get('error', '')[:80]}", flush=True)

    # Final save
    results["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_results(results)
    print(f"\nDone. Output: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
