#!/usr/bin/env python3
"""Cumulative ablation: first-k / last-k (M2).

For k in [1, n_layers], quantize first-k or last-k layers -> generate ->
store responses. After all conditions, classify all at once.
"""

import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime

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


def parse_args():
    parser = argparse.ArgumentParser(description="Cumulative layer ablation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--direction", default="both",
                        choices=["first", "last", "both"])
    parser.add_argument("--step", type=int, default=1,
                        help="Step size for k values (1=every layer)")
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
    n_layers = model_info["n_layers"]
    print(f"  Layers: {n_layers}")

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
            "experiment": "cumulative",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": args.bits,
                "direction": args.direction,
                "step": args.step,
                "quantizer": {"symmetric": preset.symmetric, "granularity": preset.granularity},
                "classifier": args.classifier,
                "max_new_tokens": args.max_new_tokens,
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

    # k values
    k_values = list(range(1, n_layers + 1, args.step))
    if n_layers not in k_values:
        k_values.append(n_layers)

    directions = []
    if args.direction in ("first", "both"):
        directions.append("first")
    if args.direction in ("last", "both"):
        directions.append("last")

    for direction in directions:
        for k in k_values:
            if direction == "first":
                layers = list(range(k))
                cond_key = f"first_{k}"
            else:
                layers = list(range(n_layers - k, n_layers))
                cond_key = f"last_{k}"

            config = QuantConfig(
                bits=args.bits,
                symmetric=preset.symmetric,
                granularity=preset.granularity,
                layers=layers,
            )

            print(f"\n[{cond_key}] Quantizing {direction} {k} layers...")
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

            print(f"  {cond_key}: {len(gen_outputs)} responses generated")

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
