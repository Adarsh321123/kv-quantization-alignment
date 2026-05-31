#!/usr/bin/env python3
"""FP16 protection sweep (P1).

Selective FP16: some layers at 16-bit, rest at base_bits. Tests contiguous
and selective layer sets to find minimal protection for alignment recovery.
"""

import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import FlexQuantizer, QuantConfig
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="FP16 protection sweep")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-bits", type=int, default=4)
    parser.add_argument("--protect-layers", type=str, nargs="+", required=True,
                        help="Comma-separated layer lists, e.g. '0' '0,1' '0,1,2'")
    parser.add_argument("--protect-bits", type=int, default=16,
                        help="Bit-width for protected layers (default: 16=FP16, use 8 for 8-bit protection)")
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

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    n_layers = model_info["n_layers"]

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")

    yi_mode = "yi" in args.model.lower()

    # Parse protection configs
    protect_configs = []
    for spec in args.protect_layers:
        layers = [int(x) for x in spec.split(",")]
        protect_configs.append(layers)

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "protection_sweep",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_bits": args.base_bits,
                "protect_configs": [str(c) for c in protect_configs],
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

    # Unprotected baseline (all layers quantized)
    print(f"\n[unprotected] {args.base_bits}-bit all layers...")
    default_config = QuantConfig(bits=args.base_bits, symmetric=False, granularity="per_token")
    with FlexQuantizer(model, {}, default_config, model_info) as fq:
        gen_outputs = generate_responses_enhanced(
            model, tokenizer, [p.text for p in prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )

    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions["unprotected"] = ConditionResult(
            response=gen_out.response,
            refused=False,  # placeholder, classified in Phase 2
            classifier="pending",
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )

    print(f"  Unprotected: {len(gen_outputs)} responses generated")

    # Each protection config
    for protect_layers in protect_configs:
        layer_label = "_".join(f"L{l}" for l in protect_layers)
        cond_key = f"protect_{layer_label}"

        protect_label = "FP16" if args.protect_bits >= 16 else f"{args.protect_bits}-bit"
        print(f"\n[{cond_key}] Protecting layers {protect_layers} at {protect_label}...")

        layer_configs = {}
        for l in protect_layers:
            layer_configs[l] = QuantConfig(bits=args.protect_bits)

        default = QuantConfig(bits=args.base_bits, symmetric=False, granularity="per_token")

        with FlexQuantizer(model, layer_configs, default, model_info) as fq:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, [p.text for p in prompts],
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
            )
            mean_mse = fq.get_mean_mse()

        for i, gen_out in enumerate(gen_outputs):
            result.prompts[i].conditions[cond_key] = ConditionResult(
                response=gen_out.response,
                refused=False,  # placeholder, classified in Phase 2
                classifier="pending",
                kv_mse=mean_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
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
