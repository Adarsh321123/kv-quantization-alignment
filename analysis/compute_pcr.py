#!/usr/bin/env python3
"""Compute PCR (Per-Channel Reduction) from channel ablation results (M6).

PCR = 1 - (per_channel_flip / per_tensor_flip) at the critical layer.

Usage:
    python analysis/compute_pcr.py --input results/qwen/channel_ablation.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.results import load_results


def compute_pcr_from_results(result_path: Path) -> dict:
    """Compute PCR from a channel ablation result file."""
    result = load_results(result_path)

    summary = result.summary
    pt_flip = summary.get("per_tensor", {}).get("flip_rate", None)
    pc_flip = summary.get("per_channel", {}).get("flip_rate", None)

    if pt_flip is None or pc_flip is None:
        # Try computing from prompt-level data
        baseline_labels = []
        pt_labels = []
        pc_labels = []

        for p in result.prompts:
            bl = p.conditions.get("baseline")
            pt = p.conditions.get("per_tensor")
            pc = p.conditions.get("per_channel")
            if bl and pt and pc:
                baseline_labels.append(bl.refused)
                pt_labels.append(pt.refused)
                pc_labels.append(pc.refused)

        bl_refused = sum(baseline_labels)
        pt_flips = sum(1 for bl, ql in zip(baseline_labels, pt_labels) if bl and not ql)
        pc_flips = sum(1 for bl, ql in zip(baseline_labels, pc_labels) if bl and not ql)

        pt_flip = pt_flips / bl_refused if bl_refused > 0 else 0.0
        pc_flip = pc_flips / bl_refused if bl_refused > 0 else 0.0

    pcr = 1.0 - (pc_flip / pt_flip) if pt_flip > 0 else 0.0

    return {
        "per_tensor_flip": pt_flip,
        "per_channel_flip": pc_flip,
        "pcr": pcr,
        "model": result.metadata.get("model_name", "unknown"),
        "layer": result.metadata.get("config", {}).get("layer", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute PCR from channel ablation results")
    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path(s) to channel ablation result JSON files")
    args = parser.parse_args()

    print(f"{'Model':<20} {'Layer':>6} {'PT Flip':>10} {'PC Flip':>10} {'PCR':>8}")
    print("-" * 60)

    for path in args.input:
        result = compute_pcr_from_results(Path(path))
        print(f"{result['model']:<20} {result['layer']:>6} "
              f"{result['per_tensor_flip']:>9.1%} {result['per_channel_flip']:>9.1%} "
              f"{result['pcr']:>7.1%}")


if __name__ == "__main__":
    main()
