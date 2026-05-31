#!/usr/bin/env python3
"""Sampling temperature robustness check (Exp 5).

Tests whether alignment collapse persists under stochastic decoding.
Runs at a single bit-width (the model's collapse point) with temperature > 0
across multiple seeds, plus a greedy FP16 baseline.

Sampling robustness sweep across seeds at the collapse bit-width.
"""

import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import KVQuantizer, QuantConfig, PRESET_SECTION4
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Temperature robustness check")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, required=True,
                        help="Bit-width at collapse point (e.g., 4 for Mistral, 6 for Qwen)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated random seeds")
    parser.add_argument("--prompts", default="advbench")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    print(f"  Layers: {model_info['n_layers']}, Hidden: {model_info['hidden_size']}")
    print(f"  Bits: {args.bits}, Temp: {args.temperature}, Top-p: {args.top_p}")
    print(f"  Seeds: {seeds}")

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} {args.prompts} prompts")

    base_config = QuantConfig(
        symmetric=PRESET_SECTION4.symmetric,
        granularity=PRESET_SECTION4.granularity,
    )

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "temperature",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": args.bits,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seeds": seeds,
                "quantizer": {
                    "symmetric": base_config.symmetric,
                    "granularity": base_config.granularity,
                },
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

    prompt_texts = [p.text for p in prompts]

    # Phase 1: Generation

    # FP16 greedy baseline (deterministic reference)
    print("\n[baseline] FP16 greedy generation...")
    gen_outputs = generate_responses_enhanced(
        model, tokenizer, prompt_texts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=0.0, do_sample=False,
    )
    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions["16_greedy"] = ConditionResult(
            response=gen_out.response,
            refused=False,
            classifier="pending",
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )
    print(f"  Baseline: {len(gen_outputs)} responses")

    # Quantized greedy (single seed, for direct comparison)
    print(f"\n[{args.bits}-bit greedy] Generating...")
    config = QuantConfig(
        bits=args.bits,
        symmetric=base_config.symmetric,
        granularity=base_config.granularity,
    )
    with KVQuantizer(model, config, model_info) as quantizer:
        gen_outputs = generate_responses_enhanced(
            model, tokenizer, prompt_texts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            temperature=0.0, do_sample=False,
        )
        greedy_mse = quantizer.get_mean_mse()

    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions[f"{args.bits}_greedy"] = ConditionResult(
            response=gen_out.response,
            refused=False,
            classifier="pending",
            kv_mse=greedy_mse,
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )
    print(f"  Greedy {args.bits}-bit: {len(gen_outputs)} responses")

    # Quantized with temperature, multiple seeds
    for seed in seeds:
        cond_key = f"{args.bits}_t{args.temperature}_seed{seed}"
        print(f"\n[{cond_key}] Generating...")

        with KVQuantizer(model, config, model_info) as quantizer:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, prompt_texts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                seed=seed,
            )
            mean_mse = quantizer.get_mean_mse()

        for i, gen_out in enumerate(gen_outputs):
            result.prompts[i].conditions[cond_key] = ConditionResult(
                response=gen_out.response,
                refused=False,
                classifier="pending",
                kv_mse=mean_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
            )

        print(f"  {cond_key}: {len(gen_outputs)} responses")
        save_results(result, Path(args.output))

    # Phase 2: Classify
    print("\n[classify] Unloading generation model...")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    save_results(result, Path(args.output))

    print(f"[classify] Loading classifier: {args.classifier}")
    try:
        yi_mode = "yi" in args.model.lower()
        classifier = get_classifier(args.classifier, yi_mode=yi_mode,
                                    device=args.classifier_device)
        classify_stored_results(result, classifier)
        classifier.unload()
    except Exception as e:
        print(f"  [WARN] Classification failed: {e}")

    result.summary = compute_summary(result, baseline_condition="16_greedy")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  TEMPERATURE ROBUSTNESS: {args.model} @ {args.bits}-bit")
    print(f"{'='*70}")
    print(f"{'Condition':>30} | {'Refusal':>8} | {'Flip':>8} | {'MSE':>10}")
    print("-" * 65)
    for cond_key in sorted(result.summary.keys()):
        s = result.summary[cond_key]
        rr = s.get("refusal_rate", 0)
        fr = s.get("flip_rate")
        mse = s.get("mean_kv_mse")
        fr_s = f"{fr:.1%}" if fr is not None else "  -"
        mse_s = f"{mse:.6f}" if mse is not None else "  -"
        print(f"{cond_key:>30} | {rr:>7.1%} | {fr_s:>8} | {mse_s:>10}")

    # Cross-seed statistics for sampled conditions
    seed_flips = []
    for seed in seeds:
        cond_key = f"{args.bits}_t{args.temperature}_seed{seed}"
        s = result.summary.get(cond_key, {})
        fr = s.get("flip_rate")
        if fr is not None:
            seed_flips.append(fr)
    if seed_flips:
        import statistics
        mean_fr = statistics.mean(seed_flips)
        std_fr = statistics.stdev(seed_flips) if len(seed_flips) > 1 else 0.0
        print(f"\n  Sampled flip rate: {mean_fr:.1%} ± {std_fr:.1%} "
              f"(n={len(seed_flips)} seeds)")
        greedy_s = result.summary.get(f"{args.bits}_greedy", {})
        greedy_fr = greedy_s.get("flip_rate")
        if greedy_fr is not None:
            print(f"  Greedy flip rate:  {greedy_fr:.1%}")
            print(f"  Delta (sampled - greedy): {mean_fr - greedy_fr:+.1%}")

    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
