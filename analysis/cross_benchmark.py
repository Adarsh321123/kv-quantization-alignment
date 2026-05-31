#!/usr/bin/env python3
"""Cross-benchmark comparison (S5).

Compares custom vs AdvBench flip rates for the same model/bits.

Usage:
    python analysis/cross_benchmark.py \\
      --custom results/qwen/sweep_custom.json \\
      --advbench results/qwen/sweep_advbench.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.results import load_results


def main():
    parser = argparse.ArgumentParser(description="Cross-benchmark comparison")
    parser.add_argument("--custom", type=str, required=True,
                        help="Path to custom benchmark sweep results")
    parser.add_argument("--advbench", type=str, required=True,
                        help="Path to AdvBench sweep results")
    args = parser.parse_args()

    custom = load_results(Path(args.custom))
    advbench = load_results(Path(args.advbench))

    model = custom.metadata.get("model_name", "unknown")
    print(f"\n{'='*60}")
    print(f"  CROSS-BENCHMARK COMPARISON: {model}")
    print(f"{'='*60}")

    # Find common bit-widths
    custom_bits = set(custom.summary.keys())
    advbench_bits = set(advbench.summary.keys())
    common_bits = sorted(custom_bits & advbench_bits, key=lambda x: int(x) if x.isdigit() else 0, reverse=True)

    print(f"\n{'Bits':>5} | {'Custom Flip':>12} | {'AdvBench Flip':>14} | {'Delta':>8}")
    print("-" * 50)

    for bits in common_bits:
        c_flip = custom.summary[bits].get("flip_rate", None)
        a_flip = advbench.summary[bits].get("flip_rate", None)

        c_s = f"{c_flip:.1%}" if c_flip is not None else "-"
        a_s = f"{a_flip:.1%}" if a_flip is not None else "-"

        if c_flip is not None and a_flip is not None:
            delta = a_flip - c_flip
            d_s = f"{delta:+.1%}"
        else:
            d_s = "-"

        print(f"{bits:>5} | {c_s:>12} | {a_s:>14} | {d_s:>8}")


if __name__ == "__main__":
    main()
