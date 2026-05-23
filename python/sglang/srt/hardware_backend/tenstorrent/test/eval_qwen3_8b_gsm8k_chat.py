"""Chat-format GSM8K(10) eval for Qwen3-8B served by SGLang on Tenstorrent.

This script is the canonical correctness gate for any TT-side optimization
work on Qwen3-8B (prefetcher, precision, fusion, etc.).

WHY chat-format + sampling, not raw 5-shot greedy:
    The earlier "raw 5-shot greedy" harness reported 28% on canonical TT for
    Qwen3-8B, which suggested a TT-side accuracy bug. Investigation showed
    that was a methodology mismatch: Qwen3-8B-Instruct is a *reasoning* model
    that requires its chat template (with <|im_start|>/<|im_end|> turn
    markers) and sampling (T=0.6, top_p=0.95, top_k=20). Under those
    conditions canonical TT scores 9/10 = 90% on GSM8K test (matches the
    published HF reference); under raw greedy the model mode-collapses
    independently of any TT noise.

Usage:
    # Local (host network):
    python3 .../test/eval_qwen3_8b_gsm8k_chat.py
    # In container with explicit args:
    podman exec p3a-ngram python3 .../test/eval_qwen3_8b_gsm8k_chat.py \\
        --url http://127.0.0.1:30000 \\
        --model /models/Qwen3-8B \\
        --gsm /tmp/gsm.jsonl \\
        --num 10

Pass criterion for the prefetcher fix:
    >= 7/10 (allows minor noise from sampling vs the 9/10 canonical baseline)
"""

import argparse
import json
import os
import re
import time
import urllib.request


INVALID = -9999999
GSM_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def _download_gsm(path: str) -> None:
    """Download GSM8K test split to ``path`` if not already present."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[gsm8k] downloading test set to {path}", flush=True)
    urllib.request.urlretrieve(GSM_URL, path)


def extract_num(s: str):
    """Extract the predicted answer number.

    Prefer the value after ``#### `` (the GSM8K convention enforced by the
    prompt). Fall back to the last number in the text.
    """
    m = re.findall(r"####\s*([-+]?\d+\.?\d*)", s)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            return INVALID
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    if not nums:
        return INVALID
    try:
        return float(nums[-1])
    except ValueError:
        return INVALID


def build_prompt(question: str, model_path: str) -> str:
    """Render the GSM8K question through the model's chat template."""
    from transformers import AutoTokenizer  # heavy import, kept local

    tok = AutoTokenizer.from_pretrained(model_path)
    msgs = [
        {
            "role": "user",
            "content": question + "\n\nSolve step by step and end with '#### <final_number>'.",
        }
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"))
    ap.add_argument("--model", default=os.environ.get("HF_MODEL_PATH", "/models/Qwen3-8B"))
    ap.add_argument("--gsm", default=os.environ.get("GSM_JSONL", "/tmp/gsm.jsonl"))
    ap.add_argument("--num", type=int, default=10, help="number of questions (default 10)")
    ap.add_argument("--skip", type=int, default=5, help="skip the first N questions (5-shot exemplars)")
    ap.add_argument("--max-new", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=600.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    _download_gsm(args.gsm)

    with open(args.gsm) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    # Pre-load tokenizer once so each prompt build is cheap.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    correct = 0
    total = args.num

    for i in range(total):
        q_idx = i + args.skip
        q = lines[q_idx]["question"]
        gt_raw = lines[q_idx]["answer"].split("####")[-1].strip()
        try:
            gt = float(gt_raw)
        except ValueError:
            gt = INVALID

        msgs = [
            {
                "role": "user",
                "content": q + "\n\nSolve step by step and end with '#### <final_number>'.",
            }
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": args.max_new,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "stop": ["<|im_end|>", "<|endoftext|>"],
            },
        }
        req = urllib.request.Request(
            args.url.rstrip("/") + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=args.timeout).read())
            ans = r["text"]
            pred = extract_num(ans)
            dt = time.time() - t0
        except Exception as e:
            ans = f"<ERROR {type(e).__name__}: {e}>"
            pred = INVALID
            dt = 0.0

        if pred == INVALID:
            status = "INVALID"
        elif abs(pred - gt) < 1e-3:
            status = "CORRECT"
            correct += 1
        else:
            status = "WRONG"

        last_line = ans.strip().split("\n")[-1][:100] if ans else ""
        print(
            f"Q{i + 1:2d}/{total} {status:7s} gt={gt} pred={pred} {dt:5.1f}s  "
            f"last_line={last_line!r}",
            flush=True,
        )

    pct = correct / total * 100.0
    print(f"\nFINAL: {correct}/{total} = {pct:.1f}%")
    # Exit code: 0 if correct >= 7/10 (the documented pass threshold), else 1.
    return 0 if correct >= max(1, int(0.7 * total)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
