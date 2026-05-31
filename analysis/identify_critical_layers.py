#!/usr/bin/env python3
"""Identify critical layers from layer ablation results.

Usage:
    python analysis/identify_critical_layers.py results/qwen/layer_ablation.json

Output:
    Critical layers for qwen: [0] (layer 0: 45.2% flip rate)

    Suggested commands:
    python experiments/run_channel_ablation.py --model qwen --bits 3 --layer 0 ...
    python experiments/run_activation_analysis.py --model qwen --layers 0 ...
    python experiments/run_protection_sweep.py --model qwen --base-bits 4 --protect-layers "0" "0,1" ...
"""

import argparse
import json
import sys
from pathlib import Path


def identify_critical_layers(result_path: str, threshold: float = 0.10, top_k: int = 3):
    """Find layers with highest flip rates.

    Args:
        result_path: Path to layer ablation result JSON
        threshold: Minimum flip rate to consider a layer "critical"
        top_k: Number of top layers to report

    Returns:
        List of (layer_index, flip_rate) tuples, sorted by flip rate descending
    """
    with open(result_path) as f:
        data = json.load(f)

    # Extract per-layer flip rates from summary
    layer_flips = []
    for key, stats in data["summary"].items():
        if key.startswith("layer_"):
            layer_idx = int(key.split("_")[1])
            flip_rate = stats.get("flip_rate", 0.0)
            layer_flips.append((layer_idx, flip_rate))

    # Sort by flip rate descending
    layer_flips.sort(key=lambda x: x[1], reverse=True)

    # Filter by threshold and top_k
    critical = [(l, f) for l, f in layer_flips[:top_k] if f >= threshold]

    return critical


def main():
    parser = argparse.ArgumentParser(description="Identify critical layers from ablation results")
    parser.add_argument("result_path", help="Path to layer ablation result JSON")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Min flip rate for critical (default 0.10)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top layers to report (default 3)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON (for pipeline automation)")
    args = parser.parse_args()

    with open(args.result_path) as f:
        data = json.load(f)

    model_name = data["metadata"].get("model_name", "unknown")
    critical = identify_critical_layers(args.result_path, args.threshold, args.top_k)

    if args.json:
        print(json.dumps({
            "model": model_name,
            "critical_layers": [{"layer": l, "flip_rate": f} for l, f in critical],
        }))
        return

    if not critical:
        print(f"No critical layers found above threshold {args.threshold}")
        return

    print(f"\nCritical layers for {model_name}:")
    for layer, flip_rate in critical:
        print(f"  Layer {layer}: {flip_rate:.1%} flip rate")

    top_layer = critical[0][0]
    top_layers_str = ",".join(str(l) for l, _ in critical)

    print(f"\nSuggested commands:")
    print(f"  python experiments/run_channel_ablation.py --model {model_name} "
          f"--bits 3 --layer {top_layer} --prompts custom "
          f"--output results/{model_name}/channel_ablation.json")
    print(f"  python experiments/run_activation_analysis.py --model {model_name} "
          f"--layers {top_layers_str} --prompts custom "
          f"--output results/{model_name}/activations.json")

    # Protection sweep: build progressive layer sets
    protect_args = []
    for i in range(1, len(critical) + 1):
        layers = ",".join(str(l) for l, _ in critical[:i])
        protect_args.append(f'"{layers}"')
    protect_str = " ".join(protect_args)

    print(f"  python experiments/run_protection_sweep.py --model {model_name} "
          f"--base-bits 4 --protect-layers {protect_str} --prompts custom "
          f"--output results/{model_name}/protection.json")


if __name__ == "__main__":
    main()
