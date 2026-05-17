"""v145 TPOT regression check.

Compares the most recent bench_ttxla_tpot_all_models.py run against the v145
baseline fixture. PASS gate: every model that PASSED in baseline must PASS now,
and its tpot_ms must be within +10% of baseline.

Exit code 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "_fixtures"
BASELINE = FIXTURES / "v145_ttxla_tpot_all_models.json"
TOLERANCE = 0.10  # 10% slowdown allowed


def _pass_tpot_by_model(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {
        r["model"]: r["tpot_ms"]
        for r in data.get("results", [])
        if r.get("status") == "PASS" and "tpot_ms" in r
    }


def _latest_run() -> Path:
    # Find the most recent v*_ttxla_tpot_all_models.json that isn't the baseline.
    candidates = sorted(
        p for p in FIXTURES.glob("v*_ttxla_tpot_all_models.json") if p != BASELINE
    )
    if not candidates:
        sys.exit(f"[regression] no current run fixture found alongside {BASELINE.name}")
    return candidates[-1]


def main() -> int:
    base = _pass_tpot_by_model(BASELINE)
    cur_path = _latest_run()
    cur = _pass_tpot_by_model(cur_path)
    print(f"[regression] baseline={BASELINE.name}  current={cur_path.name}")

    failures: list[str] = []
    for name, base_tpot in sorted(base.items()):
        cur_tpot = cur.get(name)
        if cur_tpot is None:
            failures.append(f"{name}: MISSING from current (baseline tpot={base_tpot}ms)")
            print(f"  {name:<24} base={base_tpot:>6.1f}ms cur=MISSING")
            continue
        ratio = cur_tpot / base_tpot
        marker = "OK " if ratio <= 1.0 + TOLERANCE else "REGRESS"
        print(f"  {name:<24} base={base_tpot:>6.1f}ms cur={cur_tpot:>6.1f}ms ratio={ratio:.2f} {marker}")
        if ratio > 1.0 + TOLERANCE:
            failures.append(f"{name}: regressed {ratio:.2f}x (base={base_tpot}ms cur={cur_tpot}ms)")

    if failures:
        print(f"\nFAIL: {len(failures)} model(s) regressed >{int(TOLERANCE*100)}%:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"\nPASS: all {len(base)} models within {int(TOLERANCE*100)}% of v145 baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
