"""10 fixed prompts for spec §8.2 greedy correctness — short English,
long English, code, multilingual, chat-formatted.

Covers a range of contexts to catch backend bugs that only manifest in
certain shapes (e.g. multi-byte tokenization for Chinese, leading-
whitespace handling for code).

Each prompt asks for 50 generated tokens at temperature=0. Reference
top-1 token ids are precomputed by `generate_hf_reference.py` using HF
transformers on CPU and saved to llama31_greedy_50tok.json. The TT
backend is compared per-token against that fixture.
"""

PROMPTS = [
    # 1. Short English
    "The quick brown fox",
    # 2. Long English
    "In the year 2050, artificial intelligence had become an integral part of "
    "everyday life. Doctors used AI assistants to diagnose patients, students "
    "relied on AI tutors to understand difficult subjects, and writers worked "
    "alongside AI editors. The transformation was profound, and many people "
    "wondered",
    # 3. Code (Python)
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return ",
    # 4. Code (Bash)
    "#!/bin/bash\n# Print all .py files in the current directory\nfor f in ",
    # 5. Multilingual (Chinese)
    "中国的首都是",
    # 6. Multilingual (French)
    "La capitale de la France est",
    # 7. Multilingual (German)
    "Die Hauptstadt Deutschlands ist",
    # 8. Chat-formatted (Llama 3.1 chat template, no assistant prefix)
    "<|start_header_id|>user<|end_header_id|>\n\nWhat is 17 times 23?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    # 9. Factual Q/A
    "Q: Who wrote the play Hamlet?\nA:",
    # 10. List-ish completion
    "Three primary colors are: 1) red, 2)",
]
