"""In-process wrapper that sets torch_xla custom compile options BEFORE
delegating to ``probe_decode_op_profile.py`` (or any other test entry).

Why a wrapper?
- ``torch_xla.set_custom_compile_options`` modifies per-process state. It
  does NOT survive ``subprocess.run`` boundaries, so it has to be called
  inside the same Python interpreter that invokes the probe.

Usage (Phase 2 IR-dump verification):

    python3 _task02_wrapper.py \\
        --export-path /tmp/phase2_dump \\
        --enable-const-eval 1 \\
        --enable-const-eval-on-cpu 0 \\
        -- \\
        probe_decode_op_profile.py --model Qwen/Qwen3-8B --input-len 1024 --decode 1 \\
                                   --skip-warmup 0 --out-tag phase02

Everything after ``--`` is the inner script path + its argv. The wrapper
sets the compile options, rewrites ``sys.argv``, and runs the inner script
via ``runpy.run_path``.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path


def _truthy(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    # Split argv on `--`. Everything before goes to argparse; everything
    # after is forwarded to the inner script.
    if "--" not in sys.argv:
        raise SystemExit(
            "Usage: _task02_wrapper.py <wrapper-opts> -- <inner.py> <inner-argv>"
        )
    split = sys.argv.index("--")
    wrapper_argv = sys.argv[1:split]
    inner_argv = sys.argv[split + 1 :]

    ap = argparse.ArgumentParser()
    ap.add_argument("--export-path", required=True)
    ap.add_argument("--enable-const-eval", default="1")
    ap.add_argument("--enable-const-eval-on-cpu", default="0")
    args = ap.parse_args(wrapper_argv)

    if not inner_argv:
        raise SystemExit("inner script required after --")

    # Make sure the dump dir exists.
    Path(args.export_path).mkdir(parents=True, exist_ok=True)

    # Set compile options BEFORE the inner script (which probably imports
    # torch_xla and triggers compilation). This must run in-process.
    import torch_xla  # noqa: E402

    opts = {
        "export_path": args.export_path,
        "enable_const_eval": _truthy(args.enable_const_eval),
        "enable_const_eval_on_cpu": _truthy(args.enable_const_eval_on_cpu),
    }
    torch_xla.set_custom_compile_options(opts)
    print(f"[_task02_wrapper] set_custom_compile_options({opts})", flush=True)

    # Hand off to inner script: rewrite sys.argv and exec its __main__.
    inner_path = inner_argv[0]
    if not os.path.isabs(inner_path):
        # Resolve relative to this wrapper's directory (same test dir).
        inner_path = str(Path(__file__).parent / inner_path)
    sys.argv = [inner_path] + inner_argv[1:]
    print(f"[_task02_wrapper] running {inner_path} with argv={sys.argv}", flush=True)
    runpy.run_path(inner_path, run_name="__main__")


if __name__ == "__main__":
    main()
