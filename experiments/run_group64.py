#!/usr/bin/env python3
"""Group-64 validation (M4).

Compare per-tensor all-layers vs group-64 all-layers. Tests the PCR
framework's prediction of group-64 reduction.
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
    parser = argparse.ArgumentParser(description="Group-64 validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=3)
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
            "experiment": "group64",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": args.bits,
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
    print("\n[1/3] FP16 baseline...")
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

    # Per-tensor all layers
    print(f"\n[2/3] {args.bits}-bit per-tensor all layers...")
    config_pt = QuantConfig(bits=args.bits, symmetric=preset.symmetric, granularity="per_tensor")
    with KVQuantizer(model, config_pt, model_info) as q:
        gen_outputs_pt = generate_responses_enhanced(
            model, tokenizer, [p.text for p in prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        mse_pt = q.get_mean_mse()
        stats_pt = q.get_per_layer_stats()

    for i, gen_out in enumerate(gen_outputs_pt):
        result.prompts[i].conditions["per_tensor"] = ConditionResult(
            response=gen_out.response,
            refused=False,  # placeholder, classified in Phase 2
            classifier="pending",
            kv_mse=mse_pt,
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
            kv_stats_per_layer=stats_pt,
        )

    print(f"  Per-tensor: {len(gen_outputs_pt)} responses generated")

    # Group-64 all layers
    print(f"\n[3/3] {args.bits}-bit group-64 all layers...")
    config_g64 = QuantConfig(bits=args.bits, symmetric=preset.symmetric,
                             granularity="per_group", group_size=64)
    with KVQuantizer(model, config_g64, model_info) as q:
        gen_outputs_g64 = generate_responses_enhanced(
            model, tokenizer, [p.text for p in prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        mse_g64 = q.get_mean_mse()
        stats_g64 = q.get_per_layer_stats()

    for i, gen_out in enumerate(gen_outputs_g64):
        result.prompts[i].conditions["group_64"] = ConditionResult(
            response=gen_out.response,
            refused=False,  # placeholder, classified in Phase 2
            classifier="pending",
            kv_mse=mse_g64,
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
            kv_stats_per_layer=stats_g64,
        )

    print(f"  Group-64: {len(gen_outputs_g64)} responses generated")

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

    # Summary
    result.summary = compute_summary(result, baseline_condition="baseline")

    # Compute group-64 reduction from summary
    pt_fr = result.summary.get("per_tensor", {}).get("flip_rate", 0.0)
    g64_fr = result.summary.get("group_64", {}).get("flip_rate", 0.0)
    reduction = 1.0 - (g64_fr / pt_fr) if pt_fr > 0 else 0.0
    print(f"\n  Group-64 reduction: {reduction:.1%}")

    result.summary["group64_reduction"] = reduction
    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
