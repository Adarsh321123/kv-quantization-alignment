#!/usr/bin/env python3
"""Channel-level ablation (M3).

Measures activation magnitudes to identify outlier/non-outlier channels,
then tests subsets: {all_per_tensor, all_per_channel, outliers_only,
non_outliers_only, low_magnitude, random subsets}.
"""

import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import KVQuantizer, QuantConfig, PRESET_SECTION4, PRESET_SECTION5
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def get_channel_magnitudes(model, tokenizer, prompts, layer_idx, model_info):
    """Capture K-proj activations and compute per-channel magnitudes."""
    device = next(model.parameters()).device
    fused_qkv = model_info.get("fused_qkv", False)
    hs = model_info["hidden_size"]

    activations = []

    # Find target module
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
            # Extract K portion from fused output
            k_act = output[..., hs:2*hs].detach().cpu()
        else:
            k_act = output.detach().cpu()
        activations.append(k_act)

    handle = target_module.register_forward_hook(hook)
    try:
        for prompt in prompts[:20]:  # Use subset for speed
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

    if not activations:
        raise RuntimeError("No activations captured")

    all_acts = torch.cat([a.squeeze(0) for a in activations], dim=0).float()
    channel_max = all_acts.abs().max(dim=0).values.numpy()
    return channel_max


def parse_args():
    parser = argparse.ArgumentParser(description="Channel-level ablation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--layer", type=int, required=True, help="Critical layer index")
    parser.add_argument("--prompts", default="custom")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto",
                        help="Device for classifier model")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--quantizer-preset", default="section5",
                        choices=["section4", "section5"],
                        help="Quantizer preset (section5=per-tensor sym for mechanistic, section4=per-token asym)")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")

    yi_mode = "yi" in args.model.lower()

    # Quantizer preset (default: per-tensor symmetric for mechanistic analysis)
    preset = PRESET_SECTION5 if args.quantizer_preset == "section5" else PRESET_SECTION4
    print(f"  Quantizer: symmetric={preset.symmetric}, granularity={preset.granularity}")

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "channel_ablation",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": args.bits,
                "layer": args.layer,
                "quantizer": {"symmetric": preset.symmetric, "granularity": preset.granularity},
                "classifier": args.classifier,
                "prompts": args.prompts,
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

    # FP16 baseline
    print("\n[baseline] FP16 generation...")
    baseline_outputs = generate_responses_enhanced(
        model, tokenizer, [p.text for p in prompts],
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    for i, gen_out in enumerate(baseline_outputs):
        result.prompts[i].conditions["baseline"] = ConditionResult(
            response=gen_out.response,
            refused=False,  # placeholder, classified in Phase 2
            classifier="pending",
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )

    print(f"  Baseline: {len(baseline_outputs)} responses generated")

    # Measure activation magnitudes
    print(f"\n[activations] Measuring channel magnitudes at layer {args.layer}...")
    channel_max = get_channel_magnitudes(model, tokenizer, prompts, args.layer, model_info)
    n_channels = len(channel_max)
    print(f"  {n_channels} channels, max={channel_max.max():.4f}")

    # Identify outlier channels (top 5% by magnitude)
    threshold = np.percentile(channel_max, 95)
    outlier_channels = np.where(channel_max >= threshold)[0].tolist()
    non_outlier_channels = [c for c in range(n_channels) if c not in outlier_channels]

    # Low-magnitude channels (bottom 50%)
    low_threshold = np.percentile(channel_max, 50)
    low_channels = np.where(channel_max <= low_threshold)[0].tolist()

    print(f"  Outlier channels (top 5%): {len(outlier_channels)}")
    print(f"  Low-magnitude channels (bottom 50%): {len(low_channels)}")

    # Define channel subsets to test
    np.random.seed(42)
    subsets = {
        "per_tensor": None,  # All channels, per-tensor quantization
        "per_channel": "per_channel",  # All channels, per-channel quantization
        "outliers": outlier_channels,
        "non_outliers": non_outlier_channels,
        "low_magnitude": low_channels[:100],
    }
    for pct in [1, 5, 10, 25, 50]:
        n_random = int(n_channels * pct / 100)
        random_ch = np.random.choice(n_channels, n_random, replace=False).tolist()
        subsets[f"random_{pct}pct"] = random_ch

    def _run_condition(cond_key, channels, granularity=None):
        if granularity is None:
            granularity = preset.granularity
        print(f"\n[{cond_key}] Testing...")
        config = QuantConfig(
            bits=args.bits,
            symmetric=preset.symmetric,
            granularity=granularity,
            layers=[args.layer],
            channels=channels if isinstance(channels, list) else None,
        )
        with KVQuantizer(model, config, model_info) as quantizer:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, [p.text for p in prompts],
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
            )
            mean_mse = quantizer.get_mean_mse()
            per_layer_stats = quantizer.get_per_layer_stats()

        for i, gen_out in enumerate(gen_outputs):
            result.prompts[i].conditions[cond_key] = ConditionResult(
                response=gen_out.response,
                refused=False,  # placeholder, classified in Phase 2
                classifier="pending",
                kv_mse=mean_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
                kv_stats_per_layer=per_layer_stats,
            )

        n_ch = len(channels) if isinstance(channels, list) else n_channels
        print(f"  {cond_key}: {len(gen_outputs)} responses generated ({n_ch} channels)")

    # Run each subset
    for cond_key, channels in subsets.items():
        if channels == "per_channel":
            _run_condition(cond_key, None, granularity="per_channel")
        else:
            _run_condition(cond_key, channels)

    # Phase 2: Classification
    print("\n[classify] Unloading generation model...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Save raw results first (safety net)
    save_results(result, Path(args.output))
    print(f"  [saved raw] {args.output}")

    try:
        classifier = get_classifier(args.classifier, yi_mode=yi_mode,
                                    device=args.classifier_device)
        classify_stored_results(result, classifier)
        classifier.unload()
    except Exception as e:
        print(f"  [WARN] Classification failed: {e}")
        print(f"  Raw results (without classification) saved to {args.output}")

    result.summary = compute_summary(result, baseline_condition="baseline")
    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
