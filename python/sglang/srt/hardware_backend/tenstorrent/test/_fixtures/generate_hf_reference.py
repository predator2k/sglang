"""Generate the HF reference fixture for spec §8.2 greedy correctness.

Runs Llama-3.1-8B-Instruct greedy decode on CPU (bf16) for each of the
10 prompts in llama31_prompts.py, captures the 50-token continuation as
both decoded text and raw token ids, and writes JSON.

Run once on a host with the model weights mounted; commit the JSON
alongside the test:

  python generate_hf_reference.py --model /models/Llama-3.1-8B-Instruct

Estimated runtime on CPU bf16: ~3 minutes for 10 × 50 tokens.

Output file: llama31_greedy_50tok.json (same directory).
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the prompt list. The fixture script lives in the same directory.
import sys
sys.path.insert(0, os.path.dirname(__file__))
from llama31_prompts import PROMPTS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path or HF id of Llama-3.1-8B-Instruct")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "llama31_greedy_50tok.json"))
    args = p.parse_args()

    t0 = time.perf_counter()
    print(f"loading from {args.model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    grand_t0 = time.perf_counter()
    for i, prompt in enumerate(PROMPTS, 1):
        t1 = time.perf_counter()
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = inputs["input_ids"][0].tolist()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:].tolist()
        text_out = tok.decode(new_ids)
        elapsed = time.perf_counter() - t1
        tok_s = len(new_ids) / elapsed if elapsed > 0 else 0
        print(
            f"  [{i:2d}/10] {len(prompt_ids):3d} in -> "
            f"{len(new_ids):2d} out  {elapsed:5.1f}s  {tok_s:4.1f} tok/s  "
            f"{text_out[:60]!r}",
            flush=True,
        )
        results.append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "reference_ids": new_ids,
            "reference_text": text_out,
        })

    total = time.perf_counter() - grand_t0
    print(f"\nALL DONE in {total:.1f}s ({sum(len(r['reference_ids']) for r in results)/total:.2f} tok/s overall)", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "max_tokens": args.max_tokens,
            "prompts": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
