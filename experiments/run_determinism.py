#!/usr/bin/env python3
"""Determinism check (D1).

For each seed, set RNG -> generate with quantizer -> store responses.
After all seeds, classify all responses at once, then compare across seeds.
"""

import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import KVQuantizer, QuantConfig
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Determinism check")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds")
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
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")

    yi_mode = "yi" in args.model.lower()

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "determinism",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": args.bits,
                "seeds": seeds,
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

    config = QuantConfig(bits=args.bits, symmetric=False, granularity="per_token")

    for seed in seeds:
        cond_key = f"seed_{seed}"
        print(f"\n[{cond_key}] Generating with seed {seed}...")

        with KVQuantizer(model, config, model_info) as q:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, [p.text for p in prompts],
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                seed=seed,
            )
            mean_mse = q.get_mean_mse()
            per_layer_stats = q.get_per_layer_stats()

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

        print(f"  Seed {seed}: {len(gen_outputs)} responses generated, MSE={mean_mse:.6f}")

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

    # Cross-seed agreement
    print(f"\n{'='*60}")
    print(f"  DETERMINISM CHECK: {args.model} @ {args.bits}-bit")
    print(f"{'='*60}")

    seed_keys = [f"seed_{s}" for s in seeds]
    agreements = 0
    total = 0
    for p in result.prompts:
        labels = [p.conditions[sk].refused for sk in seed_keys if sk in p.conditions]
        if len(labels) == len(seeds):
            total += 1
            if len(set(labels)) == 1:
                agreements += 1

    if total > 0:
        print(f"  Cross-seed agreement: {agreements}/{total} ({agreements/total:.1%})")

    # Check response identity
    identical_responses = 0
    for p in result.prompts:
        resps = [p.conditions[sk].response for sk in seed_keys if sk in p.conditions]
        if len(resps) == len(seeds) and len(set(resps)) == 1:
            identical_responses += 1
    print(f"  Identical responses: {identical_responses}/{total}")

    result.summary = compute_summary(result, baseline_condition=seed_keys[0])
    result.summary["cross_seed_agreement"] = agreements / total if total > 0 else 0.0
    result.summary["identical_responses"] = identical_responses
    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
