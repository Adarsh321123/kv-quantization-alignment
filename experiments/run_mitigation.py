#!/usr/bin/env python3
"""Per-channel / mixed-precision mitigation (P2, P3).

For each strategy (per_tensor, per_channel, group_64, fp16_critical, mixed),
configure quantization appropriately -> generate -> store responses.
After all strategies, classify all responses at once.
"""

import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import FlexQuantizer, KVQuantizer, QuantConfig
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


STRATEGIES = {
    "per_tensor": {"granularity": "per_token", "description": "Per-token asymmetric all layers (baseline damage)"},
    "per_channel": {"granularity": "per_channel", "description": "Per-channel all layers"},
    "group_64": {"granularity": "per_group", "group_size": 64, "description": "Group-64 all layers"},
    "fp16_critical": {"description": "FP16 on critical layer, per-tensor rest"},
    "perchannel_critical_pertensor_rest": {"description": "Per-channel critical layer, per-tensor rest"},
    "perchannel_critical_group64_rest": {"description": "Per-channel critical layer, group-64 rest"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Mitigation strategy comparison")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--strategies", type=str,
                        default="per_tensor,per_channel,group_64,fp16_critical",
                        help="Comma-separated strategy names")
    parser.add_argument("--critical-layer", type=int, default=0,
                        help="Critical layer for fp16/per-channel protection")
    parser.add_argument("--prompts", default="custom")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto",
                        help="Device for classifier model")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    strategy_names = [s.strip() for s in args.strategies.split(",")]

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    n_layers = model_info["n_layers"]

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")

    yi_mode = "yi" in args.model.lower()

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "mitigation",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": args.bits,
                "strategies": strategy_names,
                "critical_layer": args.critical_layer,
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

    crit = args.critical_layer

    for strategy in strategy_names:
        print(f"\n[{strategy}] Running...")

        if strategy == "per_tensor":
            config = QuantConfig(bits=args.bits, symmetric=False, granularity="per_token")
            ctx = KVQuantizer(model, config, model_info)
        elif strategy == "per_channel":
            config = QuantConfig(bits=args.bits, symmetric=False, granularity="per_channel")
            ctx = KVQuantizer(model, config, model_info)
        elif strategy == "group_64":
            config = QuantConfig(bits=args.bits, symmetric=False,
                               granularity="per_group", group_size=64)
            ctx = KVQuantizer(model, config, model_info)
        elif strategy == "fp16_critical":
            layer_configs = {crit: QuantConfig(bits=16)}
            default = QuantConfig(bits=args.bits, symmetric=False, granularity="per_tensor")
            ctx = FlexQuantizer(model, layer_configs, default, model_info)
        elif strategy == "perchannel_critical_pertensor_rest":
            layer_configs = {crit: QuantConfig(bits=args.bits, symmetric=False, granularity="per_channel")}
            default = QuantConfig(bits=args.bits, symmetric=False, granularity="per_tensor")
            ctx = FlexQuantizer(model, layer_configs, default, model_info)
        elif strategy == "perchannel_critical_group64_rest":
            layer_configs = {crit: QuantConfig(bits=args.bits, symmetric=False, granularity="per_channel")}
            default = QuantConfig(bits=args.bits, symmetric=False,
                                granularity="per_group", group_size=64)
            ctx = FlexQuantizer(model, layer_configs, default, model_info)
        else:
            print(f"  Unknown strategy: {strategy}, skipping")
            continue

        with ctx as q:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, [p.text for p in prompts],
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
            )
            mean_mse = q.get_mean_mse()

        for i, gen_out in enumerate(gen_outputs):
            result.prompts[i].conditions[strategy] = ConditionResult(
                response=gen_out.response,
                refused=False,  # placeholder, classified in Phase 2
                classifier="pending",
                kv_mse=mean_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
            )

        print(f"  {strategy}: {len(gen_outputs)} responses generated, MSE={mean_mse:.6f}")

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
