#!/usr/bin/env python3
"""KIVI quantization experiment (Exp 1a).

Like run_sweep.py but uses KIVI-style dual quantization:
  - Keys: per-channel asymmetric
  - Values: per-group asymmetric (group_size=32)

Tests whether a production-grade quantizer preserves alignment better than
naive per-token asymmetric. Also runs the naive baseline at matched bit-widths
for direct comparison.

Reference: Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for
KV Cache", ICML 2024.
"""

import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import (
    KVQuantizer, KVQuantizerDual, QuantConfig,
    PRESET_SECTION4, PRESET_KIVI_KEY, PRESET_KIVI_VALUE,
)
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="KIVI quantization experiment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=str, default="4,2",
                        help="Comma-separated bit-widths to test")
    parser.add_argument("--group-size", type=int, default=32,
                        help="KIVI group size for value quantization (default: 32)")
    parser.add_argument("--prompts", default="advbench")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    print(f"  Layers: {model_info['n_layers']}, Hidden: {model_info['hidden_size']}")

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} {args.prompts} prompts")

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "kivi",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": bits_list,
                "kivi": {
                    "key_granularity": "per_channel",
                    "key_symmetric": False,
                    "value_granularity": "per_group",
                    "value_symmetric": False,
                    "value_group_size": args.group_size,
                },
                "naive": {
                    "granularity": PRESET_SECTION4.granularity,
                    "symmetric": PRESET_SECTION4.symmetric,
                },
                "classifier": args.classifier,
                "max_new_tokens": args.max_new_tokens,
                "prompts": args.prompts,
                "prompt_count": len(prompts),
                "seed": args.seed,
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

    # FP16 baseline
    print("\n[baseline] FP16 generation...")
    gen_outputs = generate_responses_enhanced(
        model, tokenizer, prompt_texts,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        seed=args.seed,
    )
    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions["16"] = ConditionResult(
            response=gen_out.response, refused=False, classifier="pending",
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )

    for bits in bits_list:
        # --- Naive per-token asymmetric (existing method) ---
        naive_key = f"{bits}_naive"
        print(f"\n[{naive_key}] Per-token asymmetric {bits}-bit...")

        naive_config = QuantConfig(
            bits=bits,
            symmetric=PRESET_SECTION4.symmetric,
            granularity=PRESET_SECTION4.granularity,
        )
        with KVQuantizer(model, naive_config, model_info) as q:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, prompt_texts,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                seed=args.seed,
            )
            naive_mse = q.get_mean_mse()

        for i, gen_out in enumerate(gen_outputs):
            result.prompts[i].conditions[naive_key] = ConditionResult(
                response=gen_out.response, refused=False, classifier="pending",
                kv_mse=naive_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
            )
        print(f"  {naive_key}: {len(gen_outputs)} responses, MSE={naive_mse:.6f}")

        # --- KIVI: per-channel keys, per-group values ---
        kivi_key = f"{bits}_kivi"
        print(f"\n[{kivi_key}] KIVI {bits}-bit...")

        key_config = QuantConfig(
            bits=bits,
            symmetric=PRESET_KIVI_KEY.symmetric,
            granularity=PRESET_KIVI_KEY.granularity,
        )
        value_config = QuantConfig(
            bits=bits,
            symmetric=PRESET_KIVI_VALUE.symmetric,
            granularity=PRESET_KIVI_VALUE.granularity,
            group_size=args.group_size,
        )
        with KVQuantizerDual(model, key_config, value_config, model_info) as q:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, prompt_texts,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                seed=args.seed,
            )
            kivi_mse = q.get_mean_mse()

        for i, gen_out in enumerate(gen_outputs):
            result.prompts[i].conditions[kivi_key] = ConditionResult(
                response=gen_out.response, refused=False, classifier="pending",
                kv_mse=kivi_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
            )
        print(f"  {kivi_key}: {len(gen_outputs)} responses, MSE={kivi_mse:.6f}")

        save_results(result, Path(args.output))
        print(f"  [checkpoint] Saved after {bits}-bit")

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

    result.summary = compute_summary(result, baseline_condition="16")

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  KIVI vs NAIVE: {args.model}")
    print(f"{'='*70}")
    print(f"{'Condition':>20} | {'Refusal':>8} | {'Flip':>8} | {'MSE':>10}")
    print("-" * 55)

    for bits in bits_list:
        for suffix in ["naive", "kivi"]:
            cond = f"{bits}_{suffix}"
            s = result.summary.get(cond, {})
            rr = s.get("refusal_rate", 0)
            fr = s.get("flip_rate")
            mse = s.get("mean_kv_mse")
            fr_s = f"{fr:.1%}" if fr is not None else "  -"
            mse_s = f"{mse:.6f}" if mse is not None else "  -"
            print(f"{cond:>20} | {rr:>7.1%} | {fr_s:>8} | {mse_s:>10}")
        # Print delta
        naive_s = result.summary.get(f"{bits}_naive", {})
        kivi_s = result.summary.get(f"{bits}_kivi", {})
        n_fr = naive_s.get("flip_rate")
        k_fr = kivi_s.get("flip_rate")
        if n_fr is not None and k_fr is not None:
            print(f"  --> KIVI vs naive at {bits}-bit: {k_fr - n_fr:+.1%} flip rate")

    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
