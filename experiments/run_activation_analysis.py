#!/usr/bin/env python3
"""Activation statistics analysis (M5).

Forward pass captures K-proj activations at target layer(s). Computes:
max activation, outlier magnitude, quant MSE, outlier ratio, channel count.
No quantization applied — pure analysis.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.prompts import load_prompts
from core.results import ExperimentResult, save_results


@dataclass
class ActivationStats:
    mean: float
    std: float
    min_val: float
    max_val: float
    dynamic_range: float
    outlier_ratio: float
    outlier_magnitude: float
    n_channels: int
    quantization_mse_3bit: float
    top_outlier_channels: list
    top_outlier_values: list


def collect_activations(model, tokenizer, prompts, layer_idx, model_info):
    """Collect K-proj activations at a specific layer."""
    device = next(model.parameters()).device
    fused_qkv = model_info.get("fused_qkv", False)
    hs = model_info["hidden_size"]

    activations = []
    target_module = None

    for name, module in model.named_modules():
        if f"layers.{layer_idx}." not in name:
            continue
        if fused_qkv and "qkv_proj" in name and name.endswith("qkv_proj"):
            target_module = module
            break
        elif not fused_qkv and "k_proj" in name and name.endswith("k_proj"):
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Could not find projection module for layer {layer_idx}")

    def hook(module, input, output):
        if fused_qkv:
            k_act = output[..., hs:2*hs].detach().cpu()
        else:
            k_act = output.detach().cpu()
        activations.append(k_act)

    handle = target_module.register_forward_hook(hook)
    try:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt.text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                             max_length=512).to(device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return torch.cat([a.squeeze(0) for a in activations], dim=0).float()


def compute_stats(activations: torch.Tensor) -> ActivationStats:
    """Compute comprehensive activation statistics."""
    acts = activations
    mean = acts.mean().item()
    std = acts.std().item()
    min_val = acts.min().item()
    max_val = acts.max().item()

    abs_acts = acts.abs()
    min_abs = abs_acts[abs_acts > 0].min().item() if (abs_acts > 0).any() else 1e-8
    dynamic_range = max_val / min_abs if min_abs > 0 else float("inf")

    # Outliers (> 3 std from mean)
    outlier_threshold = abs(mean) + 3 * std
    outliers = abs_acts > outlier_threshold
    outlier_ratio = outliers.float().mean().item()
    outlier_magnitude = abs_acts[outliers].mean().item() if outliers.any() else 0.0

    # Per-channel max
    channel_max = abs_acts.max(dim=0).values.numpy()
    n_channels = len(channel_max)

    # Top outlier channels
    top_idx = np.argsort(channel_max)[-10:].tolist()
    top_vals = [float(channel_max[i]) for i in top_idx]

    # 3-bit quantization MSE (per-tensor symmetric)
    qmax = 3  # 2^(3-1) - 1
    max_abs = abs_acts.max()
    if max_abs > 1e-10:
        scale = max_abs / qmax
        x_q = torch.clamp(torch.round(acts / scale), -4, 3)
        x_deq = x_q * scale
        quant_mse = ((acts - x_deq) ** 2).mean().item()
    else:
        quant_mse = 0.0

    return ActivationStats(
        mean=mean, std=std, min_val=min_val, max_val=max_val,
        dynamic_range=dynamic_range, outlier_ratio=outlier_ratio,
        outlier_magnitude=outlier_magnitude, n_channels=n_channels,
        quantization_mse_3bit=quant_mse,
        top_outlier_channels=top_idx, top_outlier_values=top_vals,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Activation statistics analysis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: 0, mid, last)")
    parser.add_argument("--prompts", default="custom")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    n_layers = model_info["n_layers"]

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")

    if args.layers:
        layers_to_analyze = [int(x) for x in args.layers.split(",")]
    else:
        layers_to_analyze = [0, 1, 2, n_layers // 2, n_layers - 1]

    # Use first layer as the "critical layer" for metadata
    critical_layer = layers_to_analyze[0]

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "activation_analysis",
            "n_layers": n_layers,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "layers": layers_to_analyze,
                "layer": critical_layer,
                "n_prompts": len(prompts),
            },
        },
        prompts=[],  # Activation analysis doesn't produce prompt/response pairs
        summary={},
    )

    all_layer_stats = {}
    for layer_idx in layers_to_analyze:
        print(f"\n  Layer {layer_idx}:")
        acts = collect_activations(model, tokenizer, prompts, layer_idx, model_info)
        stats = compute_stats(acts)
        all_layer_stats[layer_idx] = stats

        print(f"    Dynamic range: {stats.dynamic_range:.2e}")
        print(f"    Outlier ratio: {stats.outlier_ratio:.4f}")
        print(f"    Max activation: {stats.max_val:.4f}")
        print(f"    Quant MSE (3-bit): {stats.quantization_mse_3bit:.6f}")

        del acts
        torch.cuda.empty_cache()

    # Summary: critical layer stats as top-level metrics (for generate_tables.py)
    crit_stats = all_layer_stats[critical_layer]
    result.summary = {
        "max_activation": crit_stats.max_val,
        "outlier_magnitude": crit_stats.outlier_magnitude,
        "quant_mse": crit_stats.quantization_mse_3bit,
        "outlier_ratio": crit_stats.outlier_ratio,
        "channels": crit_stats.n_channels,
        "dynamic_range": crit_stats.dynamic_range,
        # Per-layer breakdown
        "layers": {str(l): asdict(s) for l, s in all_layer_stats.items()},
    }

    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
