"""Phase 0.3 - count ttnn.to_layout-pair triples in TTNN MLIR dump.

Usage:
    PYTHONPATH=/opt/tt-mlir-toolchain/python_packages/mlir_core \\
      python3 phase0_count_to_layout_triples.py <ttnn_dump.mlir> [<ttnn_dump2.mlir> ...]

Triple = ttnn.to_layout(unary_allowlist_op(ttnn.to_layout(x))).
Reports the candidate-pair count per input plus a combined total.

Phase 3 gate: >= 50 means A.2.a is worth implementing.

Deviations from the plan source (gated by Task 0.3 "expect bugs" note):
  1. Plan's script assumes ``ir.Module.parse`` accepts the unregistered TTNN /
     TTCore dialects via ``ctx.allow_unregistered_dialects = True``. In
     practice, ``ttcore.device_module`` is written in *custom* op syntax,
     which still requires the dialect to be registered. We therefore
     pre-pass each input through ``ttmlir-opt --mlir-print-op-generic`` so
     the parser sees generic-form ops (parseable with any unregistered
     dialect). The MLIR Python bindings then load cleanly.
  2. Plan's flat 3-level traversal (``module.body`` -> ``regions`` ->
     ``blocks`` -> ``op``) reaches at most ``func.func`` ops nested directly
     under ``module``. The real TTNN dump nests them deeper:
     ``module -> ttcore.device_module -> builtin.module -> func.func ...``.
     We replace the hand-rolled walk with a recursive ``_walk_ops`` that
     visits every op in every region.
  3. Plan's signature took a single positional path. The Step-3 invocation
     uses a shell glob (``ttnn_*.mlir``), which expands to multiple files.
     We accept ``argv[1:]`` and aggregate.
  4. ``operand.owner`` can be a ``Block`` (block argument). We guard for
     ``Operation`` / ``OpView`` only; everything else is treated as a
     non-match.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable

UNARY_ALLOWLIST = {
    "ttnn.relu",
    "ttnn.gelu",
    "ttnn.silu",
    "ttnn.typecast",
}

TTMLIR_OPT_CANDIDATES = [
    "/tt-xla/third_party/tt-mlir/src/tt-mlir/build/bin/ttmlir-opt",
    "/tt-mlir-sglang/build/bin/ttmlir-opt",
]


def _find_ttmlir_opt() -> str:
    """Locate ``ttmlir-opt``. We need it to convert custom-form TTNN IR
    into generic form before the stock MLIR parser can read it."""
    override = os.environ.get("TTMLIR_OPT")
    if override and os.path.exists(override):
        return override
    for candidate in TTMLIR_OPT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    found = shutil.which("ttmlir-opt")
    if found:
        return found
    raise FileNotFoundError(
        "ttmlir-opt not found; set TTMLIR_OPT or install one of: "
        + ", ".join(TTMLIR_OPT_CANDIDATES)
    )


def _to_generic_mlir(input_path: str, ttmlir_opt: str) -> str:
    """Return the IR at ``input_path`` printed in generic op form."""
    result = subprocess.run(
        [ttmlir_opt, "--mlir-print-op-generic", input_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _resolve_op(value) -> object | None:
    """Return the producing ``ir.Operation`` for an SSA value, or ``None``
    if the owner is a block (block argument) or otherwise not an op."""
    from mlir import ir

    owner = value.owner
    if owner is None:
        return None
    if isinstance(owner, ir.OpView):
        return owner.operation
    if isinstance(owner, ir.Operation):
        return owner
    # Block arguments have a Block owner; treat them as non-match.
    return None


def _walk_ops(op_or_block) -> Iterable[object]:
    """Yield every ``ir.Operation`` reachable from ``op_or_block``,
    recursing into nested regions."""
    from mlir import ir

    if isinstance(op_or_block, ir.Block):
        for child in op_or_block.operations:
            yield from _walk_ops(child)
        return

    if isinstance(op_or_block, ir.OpView):
        op = op_or_block.operation
    else:
        op = op_or_block

    yield op
    for region in op.regions:
        for block in region:
            for child in block.operations:
                yield from _walk_ops(child)


def _count_triples(src: str) -> int:
    """Count ttnn.to_layout -> unary_allowlist_op -> ttnn.to_layout triples
    in the given generic-form MLIR source."""
    from mlir import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(src, context=ctx)

    triples = 0
    for op in _walk_ops(module.body):
        if op.name != "ttnn.to_layout":
            continue
        if len(op.operands) == 0:
            continue
        producer_op = _resolve_op(op.operands[0])
        if producer_op is None:
            continue
        if producer_op.name not in UNARY_ALLOWLIST:
            continue
        if len(producer_op.operands) == 0:
            continue
        inner_op = _resolve_op(producer_op.operands[0])
        if inner_op is None:
            continue
        if inner_op.name != "ttnn.to_layout":
            continue
        triples += 1
    return triples


def main(paths: list[str]) -> int:
    ttmlir_opt = _find_ttmlir_opt()
    grand_total = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"SKIP {path}: not found", file=sys.stderr)
            continue
        try:
            generic_src = _to_generic_mlir(path, ttmlir_opt)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "<no stderr>"
            print(
                f"SKIP {path}: ttmlir-opt failed ({stderr})",
                file=sys.stderr,
            )
            continue
        try:
            count = _count_triples(generic_src)
        except Exception as exc:  # pragma: no cover - report and continue
            print(f"SKIP {path}: parse error ({exc})", file=sys.stderr)
            continue
        print(f"{path}: {count} triple(s)")
        grand_total += count

    print(
        f"ttnn.to_layout -> unary_allowlist_op -> ttnn.to_layout triples: "
        f"{grand_total}"
    )
    print(
        f"Phase 3 gate (>= 50): {'PASS' if grand_total >= 50 else 'SKIP'}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
