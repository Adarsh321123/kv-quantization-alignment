#!/usr/bin/env python3
"""Cross-benchmark layer consistency analysis.

Computes Spearman/Kendall rank correlations of per-layer flip rates between
the Custom (N=63) and AdvBench (N=520) benchmarks, measuring how consistent
the identified safety-critical layers are across benchmarks.

Data source: per-layer sensitivity tables (Custom and AdvBench) in the paper appendix.

Usage:
    python -u analysis/cross_benchmark_layers.py \
      --output results/cross_benchmark_layer_consistency
"""

import argparse
import csv
import json
import sys
from collections import OrderedDict

from scipy.stats import spearmanr, kendalltau


# ---------------------------------------------------------------------------
# Per-layer flip rates extracted from the paper appendix.
# Custom benchmark (Qwen, LLaMA).  N_ref = 48 (Qwen), 51 (LLaMA).
# AdvBench (Qwen, LLaMA).  N_ref = 515 (Qwen), 484 (LLaMA).
# Flip rates are percentages.
# ---------------------------------------------------------------------------

LAYER_DATA = {
    "Qwen-2.5-7B": {
        "n_layers": 28,
        "custom": [
            68.8, 8.3, 6.2, 10.4, 4.2, 6.2, 4.2, 6.2,
            10.4, 2.1, 6.2, 2.1, 10.4, 12.5, 2.1, 12.5,
            6.2, 2.1, 10.4, 2.1, 4.2, 6.2, 2.1, 6.2,
            6.2, 4.2, 2.1, 20.8,
        ],
        "advbench": [
            82.9, 0.0, 0.6, 0.4, 0.0, 0.0, 1.2, 0.0,
            0.8, 0.0, 1.0, 0.0, 1.6, 0.8, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6,
            0.0, 0.0, 0.0, 56.3,
        ],
        "custom_n_ref": 48,
        "advbench_n_ref": 515,
    },
    "LLaMA-3.1-8B": {
        "n_layers": 32,
        "custom": [
            2.0, 3.9, 11.8, 19.6, 5.9, 7.8, 3.9, 7.8,
            7.8, 13.7, 3.9, 5.9, 5.9, 3.9, 5.9, 5.9,
            3.9, 9.8, 3.9, 3.9, 5.9, 5.9, 0.0, 5.9,
            5.9, 7.8, 2.0, 3.9, 3.9, 5.9, 2.0, 0.0,
        ],
        "advbench": [
            0.0, 0.8, 2.1, 3.1, 0.8, 1.4, 2.3, 1.4,
            0.0, 0.6, 2.5, 0.2, 0.2, 1.0, 1.2, 1.0,
            0.0, 1.7, 0.8, 0.2, 1.4, 0.4, 0.4, 0.2,
            0.2, 0.4, 0.4, 0.2, 0.4, 0.0, 0.2, 0.2,
        ],
        "custom_n_ref": 51,
        "advbench_n_ref": 484,
    },
}

# Models with AdvBench-only per-layer data (no per-layer Custom breakdown
# published; only the summary critical layer is known).
# AdvBench per-layer data included so we can report critical layer comparison.
ADVBENCH_ONLY = {
    "Mistral-7B": {
        "n_layers": 32,
        "critical_layer_custom": 3,
        "advbench": [
            12.8, 11.9, 17.4, 22.3, 13.1, 18.0, 9.8, 10.1,
            18.0, 9.8, 7.0, 8.5, 10.7, 8.2, 13.7, 7.9,
            7.6, 6.4, 6.7, 5.5, 4.9, 4.9, 4.3, 4.6,
            5.2, 4.0, 3.7, 4.3, 4.6, 5.2, 11.3, 6.1,
        ],
    },
    "DeepSeek-7B": {
        "n_layers": 30,
        "critical_layer_custom": 1,
        "advbench": [
            14.8, 17.6, 9.8, 12.3, 4.7, 6.6, 7.6, 4.3,
            2.3, 5.9, 2.9, 1.0, 0.8, 0.6, 3.3, 2.9,
            1.4, 1.2, 0.4, 0.2, 0.2, 0.8, 0.2, 0.0,
            0.2, 0.0, 0.2, 0.2, 0.0, 0.0,
        ],
    },
    "Yi-1.5-9B": {
        "n_layers": 48,
        "critical_layer_custom": 31,
        "advbench": [
            3.1, 2.1, 1.5, 3.3, 2.7, 2.9, 3.1, 2.1,
            5.4, 4.0, 4.2, 2.9, 5.9, 4.2, 2.5, 3.8,
            3.6, 2.9, 4.8, 2.9, 2.1, 2.9, 1.5, 1.3,
            2.1, 2.1, 2.5, 2.5, 2.9, 3.3, 2.9, 3.6,
            3.3, 4.8, 1.7, 2.7, 2.1, 2.3, 2.1, 1.0,
            2.3, 1.5, 1.7, 1.9, 1.5, 1.5, 1.7, 1.0,
        ],
    },
}


def top_k_layers(flip_rates, k=5):
    indexed = sorted(enumerate(flip_rates), key=lambda x: -x[1])
    return [i for i, _ in indexed[:k]]


def analyse_model(name, data):
    custom = data["custom"]
    advbench = data["advbench"]
    n = data["n_layers"]

    assert len(custom) == n, f"{name}: custom has {len(custom)} layers, expected {n}"
    assert len(advbench) == n, f"{name}: advbench has {len(advbench)} layers, expected {n}"

    rho, rho_p = spearmanr(custom, advbench)
    tau, tau_p = kendalltau(custom, advbench)

    top5_custom = top_k_layers(custom, 5)
    top5_advbench = top_k_layers(advbench, 5)
    overlap = len(set(top5_custom) & set(top5_advbench))

    crit_custom = top5_custom[0]
    crit_advbench = top5_advbench[0]
    crit_match = crit_custom == crit_advbench

    return OrderedDict([
        ("model", name),
        ("n_layers", n),
        ("spearman_rho", round(rho, 4)),
        ("spearman_p", round(rho_p, 6)),
        ("kendall_tau", round(tau, 4)),
        ("kendall_p", round(tau_p, 6)),
        ("crit_layer_custom", f"L{crit_custom}"),
        ("crit_layer_advbench", f"L{crit_advbench}"),
        ("crit_match", "Yes" if crit_match else "No"),
        ("top5_overlap", f"{overlap}/5"),
        ("custom_n_ref", data["custom_n_ref"]),
        ("advbench_n_ref", data["advbench_n_ref"]),
    ])


def print_table(results):
    print()
    print("=" * 120)
    print("  CROSS-BENCHMARK LAYER CONSISTENCY (Custom N=63 vs AdvBench N=520)")
    print("  Per-layer flip rates under 3-bit per-tensor symmetric quantization")
    print("=" * 120)
    print()

    header = (
        f"{'Model':<16} | {'Layers':>6} | {'Spearman ρ':>10} | {'p-value':>9} | "
        f"{'Kendall τ':>10} | {'p-value':>9} | "
        f"{'Crit (Custom)':>13} | {'Crit (AdvB)':>11} | {'Match':>5} | {'Top-5 Overlap':>13}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['model']:<16} | {r['n_layers']:>6} | {r['spearman_rho']:>10.4f} | "
            f"{r['spearman_p']:>9.6f} | {r['kendall_tau']:>10.4f} | {r['kendall_p']:>9.6f} | "
            f"{r['crit_layer_custom']:>13} | {r['crit_layer_advbench']:>11} | "
            f"{r['crit_match']:>5} | {r['top5_overlap']:>13}"
        )

    print()
    print("-" * 80)
    print("NOTES:")
    print("  - Only models with BOTH Custom and AdvBench per-layer breakdowns are included")
    print("    in the correlation table above.")
    print("  - Mistral-7B, DeepSeek-7B, Yi-1.5-9B have AdvBench per-layer data but only")
    print("    summary Custom data (critical layer only, no full per-layer Custom).")
    print("    Critical layer comparison (Custom summary vs AdvBench per-layer peak):")
    for name, info in ADVBENCH_ONLY.items():
        advb = info["advbench"]
        advb_crit = advb.index(max(advb))
        match = "MATCH" if advb_crit == info["critical_layer_custom"] else "DIFFER"
        print(f"    {name}: Custom=L{info['critical_layer_custom']}, "
              f"AdvBench=L{advb_crit} ({max(advb):.1f}%) [{match}]")
    print()

    # Summary statistics
    rhos = [r["spearman_rho"] for r in results]
    taus = [r["kendall_tau"] for r in results]
    overlaps = [int(r["top5_overlap"].split("/")[0]) for r in results]
    matches = sum(1 for r in results if r["crit_match"] == "Yes")

    print("SUMMARY:")
    print(f"  Mean Spearman ρ:       {sum(rhos)/len(rhos):.4f}")
    print(f"  Mean Kendall τ:        {sum(taus)/len(taus):.4f}")
    print(f"  Critical layer match:  {matches}/{len(results)} models")
    print(f"  Mean top-5 overlap:    {sum(overlaps)/len(overlaps):.1f}/5")
    print(f"  All correlations sig:  "
          f"{'Yes' if all(r['spearman_p'] < 0.05 for r in results) else 'No'} (p < 0.05)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-benchmark layer consistency analysis")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path prefix (writes .csv and .json)")
    args = parser.parse_args()

    results = []
    for name, data in LAYER_DATA.items():
        try:
            r = analyse_model(name, data)
            results.append(r)
        except Exception as e:
            print(f"WARNING: Skipping {name}: {e}", file=sys.stderr)

    print_table(results)

    # Build JSON output with full detail
    json_output = {
        "analysis": "cross_benchmark_layer_consistency",
        "description": "Spearman/Kendall rank correlation of per-layer flip rates "
                       "between Custom (N=63) and AdvBench (N=520) benchmarks",
        "quantization": "3-bit per-tensor symmetric",
        "models": [],
    }
    for name, data in LAYER_DATA.items():
        r = next(x for x in results if x["model"] == name)
        json_output["models"].append({
            **dict(r),
            "custom_flip_rates": data["custom"],
            "advbench_flip_rates": data["advbench"],
        })
    advbench_only_out = {}
    for name, info in ADVBENCH_ONLY.items():
        advb = info["advbench"]
        advb_crit = advb.index(max(advb))
        advbench_only_out[name] = {
            "n_layers": info["n_layers"],
            "critical_layer_custom": info["critical_layer_custom"],
            "critical_layer_advbench": advb_crit,
            "critical_layer_match": advb_crit == info["critical_layer_custom"],
            "advbench_flip_rates": advb,
            "note": "Per-layer Custom breakdown not published; "
                    "only critical layer known from summary table",
        }
    json_output["advbench_only_models"] = advbench_only_out

    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV: {csv_path}")

    json_path = out.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
