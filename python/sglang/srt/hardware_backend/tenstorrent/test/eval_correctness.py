#!/usr/bin/env python3
"""Correctness evaluation for Qwen3-8B on TT P300.

20 problems each for translation, math, and programming.
Checks output quality against expected answers.
"""

import json
import time
import urllib.request
import re
import sys

URL = "http://localhost:30000/v1/chat/completions"
MODEL = "/models/Qwen3-8B"
HEADERS = {"Content-Type": "application/json"}

TRANSLATION_PROBLEMS = [
    {"src": "en", "tgt": "zh", "input": "The quick brown fox jumps over the lazy dog.", "expected_keywords": ["狐狸", "狗"]},
    {"src": "en", "tgt": "fr", "input": "Good morning, how are you today?", "expected_keywords": ["Bonjour", "comment"]},
    {"src": "en", "tgt": "de", "input": "I would like to order a cup of coffee, please.", "expected_keywords": ["Kaffee", "bestellen"]},
    {"src": "en", "tgt": "ja", "input": "The cherry blossoms are beautiful in spring.", "expected_keywords": ["桜", "春"]},
    {"src": "en", "tgt": "ko", "input": "Where is the nearest train station?", "expected_keywords": ["역", "가장"]},
    {"src": "en", "tgt": "es", "input": "The weather is very nice today.", "expected_keywords": ["tiempo", "hoy"]},
    {"src": "en", "tgt": "pt", "input": "Can you help me find my hotel?", "expected_keywords": ["hotel", "ajudar"]},
    {"src": "en", "tgt": "it", "input": "This pasta is delicious.", "expected_keywords": ["pasta", "delizios"]},
    {"src": "en", "tgt": "ru", "input": "Mathematics is the queen of sciences.", "expected_keywords": ["математика", "наук"]},
    {"src": "zh", "tgt": "en", "input": "人工智能正在改变我们的生活方式。", "expected_keywords": ["artificial intelligence", "life", "chang"]},
    {"src": "zh", "tgt": "en", "input": "今天天气很好，适合外出散步。", "expected_keywords": ["weather", "walk"]},
    {"src": "ja", "tgt": "en", "input": "東京は日本の首都です。", "expected_keywords": ["Tokyo", "capital", "Japan"]},
    {"src": "fr", "tgt": "en", "input": "La vie est belle quand on la partage.", "expected_keywords": ["life", "beautiful", "shar"]},
    {"src": "de", "tgt": "en", "input": "Die Wissenschaft macht große Fortschritte.", "expected_keywords": ["science", "progress"]},
    {"src": "en", "tgt": "zh", "input": "Machine learning models require large amounts of training data.", "expected_keywords": ["机器学习", "训练", "数据"]},
    {"src": "en", "tgt": "ja", "input": "Please take a seat and make yourself comfortable.", "expected_keywords": ["座", "どうぞ"]},
    {"src": "en", "tgt": "ko", "input": "The Internet has connected people around the world.", "expected_keywords": ["인터넷", "세계"]},
    {"src": "es", "tgt": "en", "input": "La educación es la base del progreso.", "expected_keywords": ["education", "progress"]},
    {"src": "en", "tgt": "ar", "input": "Knowledge is power.", "expected_keywords": ["المعرفة", "قوة"]},
    {"src": "en", "tgt": "zh", "input": "Quantum computers can solve problems that classical computers cannot.", "expected_keywords": ["量子", "计算机"]},
]

MATH_PROBLEMS = [
    {"q": "What is 17 × 23?", "answer": 391},
    {"q": "What is the square root of 144?", "answer": 12},
    {"q": "If x + 5 = 12, what is x?", "answer": 7},
    {"q": "What is 2^10?", "answer": 1024},
    {"q": "What is 15% of 200?", "answer": 30},
    {"q": "What is the sum of the first 10 natural numbers?", "answer": 55},
    {"q": "If a triangle has sides 3, 4, and 5, what is its area?", "answer": 6},
    {"q": "What is 7! (7 factorial)?", "answer": 5040},
    {"q": "What is the GCD of 48 and 36?", "answer": 12},
    {"q": "What is 3/4 + 2/3? Give the answer as a decimal.", "answer": 1.4167},
    {"q": "A car travels 120 miles in 2 hours. What is its speed in mph?", "answer": 60},
    {"q": "What is the value of pi to 4 decimal places?", "answer": 3.1416},
    {"q": "If f(x) = 2x + 3, what is f(7)?", "answer": 17},
    {"q": "What is the 10th Fibonacci number?", "answer": 55},
    {"q": "What is log base 2 of 64?", "answer": 6},
    {"q": "A rectangle has length 8 and width 5. What is its perimeter?", "answer": 26},
    {"q": "What is 999 + 1?", "answer": 1000},
    {"q": "What is the derivative of x^3?", "answer": "3x^2"},
    {"q": "What is the integral of 2x dx?", "answer": "x^2"},
    {"q": "Solve: 2x - 4 = 10. What is x?", "answer": 7},
]

PROGRAMMING_PROBLEMS = [
    {"q": "Write a Python function to check if a number is prime. Show the function and test it with 17.", "check": ["def ", "prime", "True"]},
    {"q": "Write a Python function to reverse a string. Test with 'hello'.", "check": ["def ", "reverse", "olleh"]},
    {"q": "Write Python code to find the maximum element in a list [3, 7, 2, 9, 1].", "check": ["max", "9"]},
    {"q": "Write a Python function for binary search in a sorted list.", "check": ["def ", "binary", "mid"]},
    {"q": "Write Python code to count the frequency of each character in 'hello world'.", "check": ["l", "3"]},
    {"q": "Write a Python function to calculate factorial recursively.", "check": ["def ", "factorial", "recursive"]},
    {"q": "Write Python code to merge two sorted lists [1,3,5] and [2,4,6].", "check": ["merge", "[1, 2, 3, 4, 5, 6]"]},
    {"q": "Write a Python function to check if a string is a palindrome. Test with 'racecar'.", "check": ["def ", "palindrome", "True"]},
    {"q": "Write Python code to implement FizzBuzz for numbers 1 to 15.", "check": ["Fizz", "Buzz", "FizzBuzz"]},
    {"q": "Write a Python class for a simple stack with push, pop, and peek methods.", "check": ["class ", "push", "pop", "peek"]},
    {"q": "Write Python code to flatten a nested list [[1,2],[3,[4,5]],6].", "check": ["flatten", "[1, 2, 3, 4, 5, 6]"]},
    {"q": "Write a Python function to compute the nth Fibonacci number.", "check": ["def ", "fib", "return"]},
    {"q": "Write Python code to remove duplicates from a list [1,2,2,3,3,3,4].", "check": ["[1, 2, 3, 4]"]},
    {"q": "Write a Python function to convert Celsius to Fahrenheit. Convert 100°C.", "check": ["def ", "212"]},
    {"q": "Write Python code to sort a dictionary by values: {'a':3, 'b':1, 'c':2}.", "check": ["sort", "b"]},
    {"q": "Write a Python function that returns the second largest number in a list.", "check": ["def ", "second"]},
    {"q": "Write Python code to count vowels in 'Hello World'.", "check": ["vowel", "3"]},
    {"q": "Write a Python generator function that yields even numbers up to n.", "check": ["def ", "yield", "even"]},
    {"q": "Write Python code to transpose a 2D matrix [[1,2,3],[4,5,6]].", "check": ["transpose", "[1, 4]"]},
    {"q": "Write a Python function to find all pairs in a list that sum to a target. Find pairs summing to 7 in [1,2,3,4,5,6].", "check": ["def ", "pair", "(1, 6)"]},
]


def chat(prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(URL, data, HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]


def eval_translation(problems):
    results = []
    for i, p in enumerate(problems):
        prompt = f"Translate the following from {p['src']} to {p['tgt']}. Only output the translation, nothing else.\n\n{p['input']}"
        try:
            output = chat(prompt, max_tokens=256)
            output_lower = output.lower()
            found = sum(1 for kw in p["expected_keywords"] if kw.lower() in output_lower)
            passed = found >= len(p["expected_keywords"]) // 2 + 1
            results.append({"idx": i, "pass": passed, "found": found, "total_kw": len(p["expected_keywords"]), "output": output[:200]})
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] #{i+1} {p['src']}→{p['tgt']}: {found}/{len(p['expected_keywords'])} keywords | {output[:80]}")
        except Exception as e:
            results.append({"idx": i, "pass": False, "error": str(e)})
            print(f"  [ERROR] #{i+1}: {e}")
    return results


def eval_math(problems):
    results = []
    for i, p in enumerate(problems):
        prompt = f"Solve this math problem. Give ONLY the final numerical answer (no units, no explanation).\n\n{p['q']}"
        try:
            output = chat(prompt, max_tokens=256)
            clean = output.strip().replace(",", "")
            numbers = re.findall(r'-?\d+\.?\d*', clean)
            expected = str(p["answer"])
            if isinstance(p["answer"], str):
                passed = p["answer"].lower().replace(" ", "") in output.lower().replace(" ", "")
            elif numbers:
                passed = any(abs(float(n) - float(p["answer"])) < 0.01 for n in numbers)
            else:
                passed = expected in clean
            results.append({"idx": i, "pass": passed, "expected": p["answer"], "output": output[:200]})
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] #{i+1}: expected={p['answer']} got={output[:60]}")
        except Exception as e:
            results.append({"idx": i, "pass": False, "error": str(e)})
            print(f"  [ERROR] #{i+1}: {e}")
    return results


def eval_programming(problems):
    results = []
    for i, p in enumerate(problems):
        prompt = f"{p['q']}\n\nProvide the complete Python code with output."
        try:
            output = chat(prompt, max_tokens=1024)
            found = sum(1 for check in p["check"] if check.lower() in output.lower())
            passed = found >= len(p["check"]) // 2 + 1
            results.append({"idx": i, "pass": passed, "found": found, "total": len(p["check"]), "output": output[:300]})
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] #{i+1}: {found}/{len(p['check'])} checks | {output[:60].replace(chr(10), ' ')}")
        except Exception as e:
            results.append({"idx": i, "pass": False, "error": str(e)})
            print(f"  [ERROR] #{i+1}: {e}")
    return results


def main():
    print("=" * 60)
    print("Correctness evaluation — Qwen3-8B on 2×P150a")
    print("=" * 60)

    # Warmup
    print("\nWarmup...")
    chat("Hello", max_tokens=10)

    all_results = {}

    print(f"\n--- Translation (20 problems) ---")
    t0 = time.perf_counter()
    trans = eval_translation(TRANSLATION_PROBLEMS)
    t_trans = time.perf_counter() - t0
    trans_pass = sum(1 for r in trans if r.get("pass"))
    print(f"  Score: {trans_pass}/20 ({trans_pass/20*100:.0f}%) in {t_trans:.1f}s")
    all_results["translation"] = {"pass": trans_pass, "total": 20, "time_s": round(t_trans, 1), "details": trans}

    print(f"\n--- Math (20 problems) ---")
    t0 = time.perf_counter()
    math_r = eval_math(MATH_PROBLEMS)
    t_math = time.perf_counter() - t0
    math_pass = sum(1 for r in math_r if r.get("pass"))
    print(f"  Score: {math_pass}/20 ({math_pass/20*100:.0f}%) in {t_math:.1f}s")
    all_results["math"] = {"pass": math_pass, "total": 20, "time_s": round(t_math, 1), "details": math_r}

    print(f"\n--- Programming (20 problems) ---")
    t0 = time.perf_counter()
    prog = eval_programming(PROGRAMMING_PROBLEMS)
    t_prog = time.perf_counter() - t0
    prog_pass = sum(1 for r in prog if r.get("pass"))
    print(f"  Score: {prog_pass}/20 ({prog_pass/20*100:.0f}%) in {t_prog:.1f}s")
    all_results["programming"] = {"pass": prog_pass, "total": 20, "time_s": round(t_prog, 1), "details": prog}

    total_pass = trans_pass + math_pass + prog_pass
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass}/60 ({total_pass/60*100:.0f}%)")
    print(f"  Translation: {trans_pass}/20")
    print(f"  Math: {math_pass}/20")
    print(f"  Programming: {prog_pass}/20")

    out_path = "/home/mhnie/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/_fixtures/v87_acceptance_breakthrough/v124_correctness_eval.json"
    with open(out_path, "w") as f:
        json.dump({"date": "2026-05-16", "model": "Qwen3-8B", "device": "2xP150a", "total_pass": total_pass, "total": 60, "results": all_results}, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
