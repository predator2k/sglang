"""Aggregate Tracy ops_perf_results CSV by OP CODE.

Note: DEVICE KERNEL DURATION column appears unreliable on this build
(values jump between sensible ns and uninitialized 3.9e12 ns). We use
OP TO OP LATENCY instead, which is the wall time between consecutive op
dispatches — a reliable proxy for per-op device cost.

Usage: python3 tracy_aggregate.py <ops_perf_results.csv>
"""
import csv
import sys
from collections import defaultdict


def _int_or_zero(v):
    try:
        x = int(v)
        # Filter out obviously broken values (>1 second per op is implausible).
        if 0 <= x < 10_000_000_000:
            return x
    except (ValueError, TypeError):
        pass
    return 0


def main():
    path = sys.argv[1]
    stats = defaultdict(lambda: {
        "count": 0, "host_ns": 0, "op2op_ns": 0,
        "kernel_ns_valid": 0, "kernel_ns_count": 0,
    })
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = row["OP CODE"]
            stats[op]["count"] += 1
            stats[op]["host_ns"] += _int_or_zero(row.get("HOST DURATION [ns]"))
            stats[op]["op2op_ns"] += _int_or_zero(row.get("OP TO OP LATENCY [ns]"))
            k = _int_or_zero(row.get("DEVICE KERNEL DURATION [ns]"))
            if 0 < k < 100_000_000:  # < 100 ms per kernel
                stats[op]["kernel_ns_valid"] += k
                stats[op]["kernel_ns_count"] += 1

    total_op2op_ms = sum(s["op2op_ns"] for s in stats.values()) / 1e6
    total_host_ms = sum(s["host_ns"] for s in stats.values()) / 1e6
    total_kernel_ms = sum(s["kernel_ns_valid"] for s in stats.values()) / 1e6
    total_count = sum(s["count"] for s in stats.values())

    print(f"Total op calls: {total_count}, unique op codes: {len(stats)}")
    print(f"Total OP→OP LATENCY: {total_op2op_ms:.1f} ms ({total_op2op_ms/1000:.2f}s)")
    print(f"Total HOST DURATION: {total_host_ms:.1f} ms ({total_host_ms/1000:.2f}s)")
    print(f"Sum of valid (<100ms) device kernel durations: {total_kernel_ms:.1f} ms "
          f"(rows: {sum(s['kernel_ns_count'] for s in stats.values())})")
    print()
    print(f"{'OP CODE':<55} {'count':>6} {'op2op_ms':>10} {'mean_us':>9} {'%':>5}")
    print("-" * 95)
    sorted_ops = sorted(stats.items(), key=lambda kv: -kv[1]["op2op_ns"])
    for op, s in sorted_ops[:30]:
        oms = s["op2op_ns"] / 1e6
        mean_us = s["op2op_ns"] / max(1, s["count"]) / 1000
        pct = oms / total_op2op_ms * 100 if total_op2op_ms else 0
        print(f"{op:<55} {s['count']:>6} {oms:>9.1f}  {mean_us:>8.1f}  {pct:>4.1f}")


if __name__ == "__main__":
    main()
