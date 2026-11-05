#!/usr/bin/env python3
"""PCR stability over random calibration subsets.

Checks whether PCR is stable when computed from different random 20-prompt subsets.
Approach A: post-hoc subsampling from existing channel ablation data.
Approach B: run fresh channel ablation on AdvBench if existing data is insufficient.
"""

import argparse
import csv
import gc
import sys
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.results import ExperimentResult, PromptResult, ConditionResult, save_results, load_results
from core.metrics import wilson_ci

CRITICAL_LAYERS = {
    "qwen": 0,
    "mistral": 3,
    "llama": 3,
    "gemma2": 1,
    "deepseek": 1,
}

MODELS = ["qwen", "mistral", "llama", "gemma2", "deepseek"]


def compute_pcr_from_triples(triples, subset_indices=None):
    """Compute PCR from (baseline_refused, pt_refused, pc_refused) triples.

    If subset_indices is given, only use those indices.
    """
    if subset_indices is not None:
        triples = [triples[i] for i in subset_indices]

    baseline_refused_triples = [(bl, pt, pc) for bl, pt, pc in triples if bl]
    if not baseline_refused_triples:
        return None

    n_bl_refused = len(baseline_refused_triples)
    pt_flips = sum(1 for bl, pt, pc in baseline_refused_triples if not pt)
    pc_flips = sum(1 for bl, pt, pc in baseline_refused_triples if not pc)

    pt_flip_rate = pt_flips / n_bl_refused
    pc_flip_rate = pc_flips / n_bl_refused

    if pt_flip_rate == 0:
        return 0.0

    pcr = 1.0 - (pc_flip_rate / pt_flip_rate)
    return pcr


def pcr_bin(pcr_val):
    if pcr_val is None:
        return "N/A"
    if pcr_val < 0.30:
        return "<30%"
    elif pcr_val <= 0.70:
        return "30-70%"
    else:
        return ">70%"


def run_stability_analysis(triples, n_subsets=20, subset_size=20, seed=42):
    """Run PCR stability analysis via subsampling."""
    rng = random.Random(seed)

    baseline_refused_indices = [i for i, (bl, pt, pc) in enumerate(triples) if bl]

    if len(baseline_refused_indices) < subset_size:
        print(f"  Warning: only {len(baseline_refused_indices)} baseline-refused prompts, "
              f"need {subset_size}. Using all.")
        subset_size = len(baseline_refused_indices)

    full_pcr = compute_pcr_from_triples(triples)
    full_bin = pcr_bin(full_pcr)

    pcr_values = []
    for _ in range(n_subsets):
        subset_idx = rng.sample(baseline_refused_indices, min(subset_size, len(baseline_refused_indices)))
        pcr_val = compute_pcr_from_triples(triples, subset_idx)
        if pcr_val is not None:
            pcr_values.append(pcr_val)

    if not pcr_values:
        return None

    mean_pcr = np.mean(pcr_values)
    std_pcr = np.std(pcr_values)
    min_pcr = np.min(pcr_values)
    max_pcr = np.max(pcr_values)

    bins_matching = sum(1 for v in pcr_values if pcr_bin(v) == full_bin)
    bin_stability = bins_matching / len(pcr_values)

    return {
        "full_pcr": full_pcr,
        "mean_pcr": float(mean_pcr),
        "std_pcr": float(std_pcr),
        "min_pcr": float(min_pcr),
        "max_pcr": float(max_pcr),
        "bin_full": full_bin,
        "bin_stability": float(bin_stability),
        "n_subsets": n_subsets,
        "subset_size": subset_size,
        "n_baseline_refused": len(baseline_refused_indices),
        "pcr_values": [float(v) for v in pcr_values],
    }


def try_approach_a(results_dir):
    """Try to extract per-prompt triples from existing channel ablation JSONs."""
    results_dir = Path(results_dir)
    model_triples = {}

    for model_name in MODELS:
        patterns = [
            results_dir / model_name / "channel_ablation.json",
            results_dir / f"{model_name}_channel_ablation.json",
            results_dir / f"{model_name}_channel_ablation_advbench.json",
        ]

        for pattern in patterns:
            if not pattern.exists():
                continue

            try:
                result = load_results(pattern)
                triples = []
                for p in result.prompts:
                    bl = p.conditions.get("baseline")
                    pt = p.conditions.get("per_tensor")
                    pc = p.conditions.get("per_channel")
                    if bl and pt and pc:
                        if bl.classifier != "pending" and pt.classifier != "pending" and pc.classifier != "pending":
                            triples.append((bl.refused, pt.refused, pc.refused))

                if len(triples) >= 20:
                    model_triples[model_name] = triples
                    print(f"  Found {len(triples)} classified triples for {model_name} in {pattern}")
                    break
            except Exception as e:
                print(f"  Error loading {pattern}: {e}")

    return model_triples


def run_approach_b(models, results_dir, max_prompts=None):
    """Run fresh channel ablation on AdvBench for each model."""
    from core.model_loader import load_model
    from core.quantization import KVQuantizer, QuantConfig, PRESET_SECTION5
    from core.generation import generate_responses_enhanced
    from core.classifier import get_classifier, classify_stored_results
    from core.prompts import load_prompts

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts("advbench")
    if max_prompts:
        prompts = prompts[:max_prompts]
    print(f"  Loaded {len(prompts)} AdvBench prompts")

    model_triples = {}

    for model_name in models:
        critical_layer = CRITICAL_LAYERS[model_name]
        output_path = results_dir / f"{model_name}_channel_ablation_advbench.json"

        print(f"\n{'='*60}")
        print(f"  Model: {model_name} | Critical layer: {critical_layer}")
        print(f"{'='*60}")

        print(f"  Loading model: {model_name}")
        model, tokenizer, model_info = load_model(model_name)

        result = ExperimentResult(
            metadata={
                "model": model_info.get("hf_id", model_name),
                "model_name": model_name,
                "experiment": "channel_ablation_pcr_stability",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "bits": 3,
                    "layer": critical_layer,
                    "quantizer": {"symmetric": True, "granularity": "per_tensor"},
                    "classifier": "wildguard",
                    "prompts": "advbench",
                    "prompt_count": len(prompts),
                },
                "pipeline_version": "1.0.0",
            },
        )

        for p in prompts:
            result.prompts.append(PromptResult(
                prompt_idx=p.index, prompt_text=p.text, category=p.category,
            ))

        # Phase 1: Generation
        print(f"\n  [baseline] FP16 generation...")
        baseline_outputs = generate_responses_enhanced(
            model, tokenizer, [p.text for p in prompts],
            max_new_tokens=256, batch_size=4,
        )
        for i, gen_out in enumerate(baseline_outputs):
            result.prompts[i].conditions["baseline"] = ConditionResult(
                response=gen_out.response,
                refused=False,
                classifier="pending",
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
            )
        print(f"    {len(baseline_outputs)} baseline responses generated")

        print(f"  [per_tensor] 3-bit per-tensor at layer {critical_layer}...")
        pt_config = QuantConfig(
            bits=3, symmetric=True, granularity="per_tensor",
            layers=[critical_layer],
        )
        with KVQuantizer(model, pt_config, model_info) as quantizer:
            pt_outputs = generate_responses_enhanced(
                model, tokenizer, [p.text for p in prompts],
                max_new_tokens=256, batch_size=4,
            )
            pt_mse = quantizer.get_mean_mse()
        for i, gen_out in enumerate(pt_outputs):
            result.prompts[i].conditions["per_tensor"] = ConditionResult(
                response=gen_out.response,
                refused=False,
                classifier="pending",
                kv_mse=pt_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
            )
        print(f"    {len(pt_outputs)} per-tensor responses generated")

        print(f"  [per_channel] 3-bit per-channel at layer {critical_layer}...")
        pc_config = QuantConfig(
            bits=3, symmetric=True, granularity="per_channel",
            layers=[critical_layer],
        )
        with KVQuantizer(model, pc_config, model_info) as quantizer:
            pc_outputs = generate_responses_enhanced(
                model, tokenizer, [p.text for p in prompts],
                max_new_tokens=256, batch_size=4,
            )
            pc_mse = quantizer.get_mean_mse()
        for i, gen_out in enumerate(pc_outputs):
            result.prompts[i].conditions["per_channel"] = ConditionResult(
                response=gen_out.response,
                refused=False,
                classifier="pending",
                kv_mse=pc_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
            )
        print(f"    {len(pc_outputs)} per-channel responses generated")

        # Phase 2: Classification
        print(f"\n  [classify] Unloading generation model...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        save_results(result, output_path)
        print(f"    [saved raw] {output_path}")

        try:
            yi_mode = "yi" in model_name.lower()
            classifier = get_classifier("wildguard", yi_mode=yi_mode)
            classify_stored_results(result, classifier)
            classifier.unload()
        except Exception as e:
            print(f"    [WARN] Classification failed: {e}")
            print(f"    Raw results saved to {output_path}")

        result.summary = {}
        save_results(result, output_path)

        triples = []
        for p in result.prompts:
            bl = p.conditions.get("baseline")
            pt = p.conditions.get("per_tensor")
            pc = p.conditions.get("per_channel")
            if bl and pt and pc:
                triples.append((bl.refused, pt.refused, pc.refused))

        model_triples[model_name] = triples
        print(f"    Extracted {len(triples)} triples for {model_name}")

        gc.collect()
        torch.cuda.empty_cache()

    return model_triples


def parse_args():
    parser = argparse.ArgumentParser(description="PCR stability analysis")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path for summary")
    parser.add_argument("--n-subsets", type=int, default=20)
    parser.add_argument("--subset-size", type=int, default=20)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit AdvBench prompts (for Approach B)")
    parser.add_argument("--force-approach-b", action="store_true",
                        help="Force fresh channel ablation even if data exists")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model list (default: all 5)")
    return parser.parse_args()


def main():
    args = parse_args()

    models_to_run = args.models.split(",") if args.models else MODELS

    print(f"PCR Stability Analysis")
    print(f"  Models: {models_to_run}")
    print(f"  Subsets: {args.n_subsets} x {args.subset_size} prompts")

    # Try Approach A first
    model_triples = {}
    if not args.force_approach_b:
        print(f"\n[Approach A] Checking existing results in {args.results_dir}...")
        model_triples = try_approach_a(args.results_dir)
        print(f"  Found data for: {list(model_triples.keys())}")

    # Approach B for missing models
    missing_models = [m for m in models_to_run if m not in model_triples]
    if missing_models:
        print(f"\n[Approach B] Running fresh channel ablation for: {missing_models}")
        new_triples = run_approach_b(missing_models, Path(args.results_dir), max_prompts=args.max_prompts)
        model_triples.update(new_triples)

    # Run stability analysis
    print(f"\n{'='*70}")
    print(f"  PCR STABILITY RESULTS")
    print(f"{'='*70}")

    all_results = {}
    for model_name in models_to_run:
        if model_name not in model_triples:
            print(f"  {model_name}: NO DATA")
            continue

        triples = model_triples[model_name]
        stability = run_stability_analysis(
            triples,
            n_subsets=args.n_subsets,
            subset_size=args.subset_size,
        )

        if stability is None:
            print(f"  {model_name}: insufficient baseline-refused prompts")
            continue

        all_results[model_name] = stability

    # Print summary table
    print(f"\n{'Model':<12} | {'Full PCR':>10} | {'Mean±Std PCR':>15} | {'Min-Max':>12} | {'Bin Stability':>15}")
    print("-" * 75)
    for model_name in models_to_run:
        if model_name not in all_results:
            continue
        s = all_results[model_name]
        full_pcr = f"{s['full_pcr']:.1%}" if s['full_pcr'] is not None else "N/A"
        mean_std = f"{s['mean_pcr']:.1%}±{s['std_pcr']:.1%}"
        min_max = f"{s['min_pcr']:.0%}-{s['max_pcr']:.0%}"
        bin_stab = f"{int(s['bin_stability'] * s['n_subsets'])}/{s['n_subsets']} ({s['bin_full']})"
        print(f"{model_name:<12} | {full_pcr:>10} | {mean_std:>15} | {min_max:>12} | {bin_stab:>15}")

    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "full_pcr", "mean_pcr", "std_pcr", "min_pcr",
                         "max_pcr", "bin_full", "bin_stability", "n_baseline_refused"])
        for model_name in models_to_run:
            if model_name not in all_results:
                continue
            s = all_results[model_name]
            writer.writerow([
                model_name,
                f"{s['full_pcr']:.4f}" if s['full_pcr'] is not None else "",
                f"{s['mean_pcr']:.4f}",
                f"{s['std_pcr']:.4f}",
                f"{s['min_pcr']:.4f}",
                f"{s['max_pcr']:.4f}",
                s['bin_full'],
                f"{s['bin_stability']:.4f}",
                s['n_baseline_refused'],
            ])
    print(f"\nCSV saved to {output_path}")

    # Save JSON with full detail
    json_output = output_path.with_suffix(".json")
    json_result = ExperimentResult(
        metadata={
            "experiment": "pcr_stability",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_subsets": args.n_subsets,
                "subset_size": args.subset_size,
                "models": models_to_run,
            },
            "pipeline_version": "1.0.0",
        },
        summary=all_results,
    )
    save_results(json_result, json_output)
    print(f"JSON saved to {json_output}")


if __name__ == "__main__":
    main()
