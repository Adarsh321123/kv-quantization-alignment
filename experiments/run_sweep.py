#!/usr/bin/env python3
"""Bit-width sweep experiment (S1, S2, S4).

For each bit-width, attach quantizer -> generate responses -> classify -> compute
flip rate vs FP16 baseline. Supports custom, advbench, and harmbench prompt sets.
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
    parser = argparse.ArgumentParser(description="Bit-width sweep experiment")
    parser.add_argument("--model", required=True, help="Model name (canonical or HF ID)")
    parser.add_argument("--bits", type=str, default="16,8,6,5,4,3,2",
                        help="Comma-separated bit-widths")
    parser.add_argument("--prompts", default="custom",
                        help="Prompt set: custom, advbench, harmbench")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--quantizer-preset", default="section4",
                        choices=["section4", "section5"],
                        help="Quantizer preset (section4=per-token asym, section5=per-tensor sym)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto",
                        help="Device for classifier model (auto, cuda, cuda:1, cpu)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0=greedy, >0 enables sampling)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples per prompt when temperature>0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]
    do_sample = args.temperature > 0.0

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    print(f"  Layers: {model_info['n_layers']}, Hidden: {model_info['hidden_size']}")
    if do_sample:
        print(f"  Temperature: {args.temperature}, Samples: {args.num_samples}")

    # Load prompts
    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} {args.prompts} prompts")

    # Quantizer preset
    if args.quantizer_preset == "section4":
        base_config = QuantConfig(
            symmetric=PRESET_SECTION4.symmetric,
            granularity=PRESET_SECTION4.granularity,
        )
    else:
        base_config = QuantConfig(
            symmetric=PRESET_SECTION5.symmetric,
            granularity=PRESET_SECTION5.granularity,
        )

    # Initialize result
    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "sweep",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": bits_list,
                "quantizer": {
                    "symmetric": base_config.symmetric,
                    "granularity": base_config.granularity,
                },
                "classifier": args.classifier,
                "max_new_tokens": args.max_new_tokens,
                "prompts": args.prompts,
                "prompt_count": len(prompts),
                "seed": args.seed,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
            },
            "pipeline_version": "1.0.0",
        },
    )

    for p in prompts:
        result.prompts.append(PromptResult(
            prompt_idx=p.index, prompt_text=p.text, category=p.category,
        ))

    # Phase 1: Generate responses for all bit-widths
    for bits in bits_list:
        print(f"\n[{bits}-bit] Generating responses...")

        config = QuantConfig(
            bits=bits,
            symmetric=base_config.symmetric,
            granularity=base_config.granularity,
        )

        prompt_texts = [p.text for p in prompts]
        # With temperature>0, repeat prompts for multiple samples
        if do_sample and args.num_samples > 1:
            prompt_texts = prompt_texts * args.num_samples

        if bits >= 16:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, prompt_texts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                seed=args.seed,
                temperature=args.temperature,
                do_sample=do_sample,
            )
            mean_mse = None
            per_layer_stats = None
        else:
            with KVQuantizer(model, config, model_info) as quantizer:
                gen_outputs = generate_responses_enhanced(
                    model, tokenizer, prompt_texts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    temperature=args.temperature,
                    do_sample=do_sample,
                )
                mean_mse = quantizer.get_mean_mse()
                per_layer_stats = quantizer.get_per_layer_stats()

        # Store responses (classification deferred to Phase 2)
        n_prompts = len(prompts)
        if do_sample and args.num_samples > 1:
            for sample_idx in range(args.num_samples):
                for i in range(n_prompts):
                    gen_idx = sample_idx * n_prompts + i
                    gen_out = gen_outputs[gen_idx]
                    cond_key = f"{bits}_s{sample_idx}" if sample_idx > 0 else str(bits)
                    result.prompts[i].conditions[cond_key] = ConditionResult(
                        response=gen_out.response,
                        refused=False,
                        classifier="pending",
                        kv_mse=mean_mse,
                        generation_time_s=gen_out.generation_time_s,
                        input_token_count=gen_out.input_token_count,
                        token_ids=gen_out.token_ids,
                        kv_stats_per_layer=per_layer_stats,
                    )
        else:
            for i, gen_out in enumerate(gen_outputs):
                result.prompts[i].conditions[str(bits)] = ConditionResult(
                    response=gen_out.response,
                    refused=False,
                    classifier="pending",
                    kv_mse=mean_mse,
                    generation_time_s=gen_out.generation_time_s,
                    input_token_count=gen_out.input_token_count,
                    token_ids=gen_out.token_ids,
                    kv_stats_per_layer=per_layer_stats,
                )

        print(f"  Generated {len(gen_outputs)} responses")

        save_results(result, Path(args.output))
        print(f"  [checkpoint] Saved after {bits}-bit to {args.output}")

    # Phase 2: Unload generation model, classify all responses
    print("\n[classify] Unloading generation model...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Save raw results first (safety net — generation data is never lost)
    save_results(result, Path(args.output))
    print(f"  [saved raw] {args.output}")

    print(f"[classify] Loading classifier: {args.classifier}")
    try:
        yi_mode = "yi" in args.model.lower()
        classifier = get_classifier(args.classifier, yi_mode=yi_mode,
                                    device=args.classifier_device)
        classify_stored_results(result, classifier)
        classifier.unload()
    except Exception as e:
        print(f"  [WARN] Classification failed: {e}")
        print(f"  Raw results (without classification) saved to {args.output}")

    # Compute summary
    result.summary = compute_summary(result, baseline_condition="16")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SWEEP SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"{'Bits':>5} | {'Refusal':>8} | {'Flip':>8} | {'MSE':>10}")
    print("-" * 40)
    for bits in bits_list:
        s = result.summary.get(str(bits), {})
        rr = s.get("refusal_rate", 0)
        fr = s.get("flip_rate", None)
        mse = s.get("mean_kv_mse", None)
        fr_s = f"{fr:.1%}" if fr is not None else "  -"
        mse_s = f"{mse:.6f}" if mse is not None else "  -"
        print(f"{bits:>5} | {rr:>7.1%} | {fr_s:>8} | {mse_s:>10}")

    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
