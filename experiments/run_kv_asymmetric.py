#!/usr/bin/env python3
"""K vs V asymmetric quantization experiment (Exp 7).

For each model and bit-width, tests three conditions:
- K-only: quantize keys to target bits, values at FP16
- V-only: quantize values to target bits, keys at FP16
- Both:   quantize both K and V (baseline, matches sweep results)

Reports flip rates and MSE separately for K and V projections.
Uses per-token asymmetric quantization (PRESET_SECTION4).
"""

import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model, resolve_hf_id, get_model_info
from core.quantization import KVQuantizer, QuantConfig, PRESET_SECTION4
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="K vs V asymmetric quantization")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=str, default="4,3",
                        help="Comma-separated bit-widths to test")
    parser.add_argument("--prompts", default="advbench")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto",
                        help="Device for classifier model")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit (for large models that don't fit in bf16)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (for very large models)")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]

    print(f"Loading model: {args.model}")
    if args.load_in_8bit or args.load_in_4bit:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_id = resolve_hf_id(args.model)
        model_info = get_model_info(args.model)
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if args.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            print("  Loading in 8-bit mode")
        else:
            load_kwargs["load_in_4bit"] = True
            print("  Loading in 4-bit mode")
        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        model.eval()
    else:
        model, tokenizer, model_info = load_model(args.model)
    n_layers = model_info["n_layers"]
    print(f"  Layers: {n_layers}")

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")
    print(f"  Bit-widths: {bits_list}")

    yi_mode = "yi" in args.model.lower()

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "kv_asymmetric",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": bits_list,
                "quantizer": {"symmetric": False, "granularity": "per_token"},
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
            refused=False,
            classifier="pending",
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )
    print(f"  Baseline: {len(baseline_outputs)} responses generated")

    # For each bit-width, test K-only, V-only, and both
    for bits in bits_list:
        conditions = [
            (f"{bits}bit_k_only", True, False),   # K quantized, V at FP16
            (f"{bits}bit_v_only", False, True),    # V quantized, K at FP16
            (f"{bits}bit_both",   True, True),     # Both quantized (baseline)
        ]

        for cond_key, quant_k, quant_v in conditions:
            print(f"\n[{cond_key}] bits={bits}, K={'quant' if quant_k else 'FP16'}, V={'quant' if quant_v else 'FP16'}...")

            config = QuantConfig(
                bits=bits,
                symmetric=PRESET_SECTION4.symmetric,
                granularity=PRESET_SECTION4.granularity,
                quantize_keys=quant_k,
                quantize_values=quant_v,
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
                    refused=False,
                    classifier="pending",
                    kv_mse=mean_mse,
                    generation_time_s=gen_out.generation_time_s,
                    input_token_count=gen_out.input_token_count,
                    token_ids=gen_out.token_ids,
                    kv_stats_per_layer=per_layer_stats,
                )

            print(f"  {cond_key}: {len(gen_outputs)} responses, MSE={mean_mse:.6f}")

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

    # Print summary
    print(f"\n{'='*60}")
    print(f"  K vs V ASYMMETRIC RESULTS: {args.model}")
    print(f"{'='*60}")
    for bits in bits_list:
        for suffix in ["k_only", "v_only", "both"]:
            key = f"{bits}bit_{suffix}"
            s = result.summary.get(key, {})
            fr = s.get("flip_rate", 0.0)
            mse = s.get("kv_mse", 0.0)
            print(f"  {key:>20}: flip={fr:.1%}  MSE={mse:.6f}")


if __name__ == "__main__":
    main()
