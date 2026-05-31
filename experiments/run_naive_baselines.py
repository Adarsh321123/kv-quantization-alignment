#!/usr/bin/env python3
"""Naive baseline comparison for the diagnostic protocol (Exp 9).

Compares three naive mitigation strategies against the protocol-prescribed fix:
  (a) Always protect layers 0-1 at FP16
  (b) Protect top-2 layers by attention mass (computed from calibration prompts)
  (c) Uniform Group-64 all layers

Uses FlexQuantizer for (a) and (b), KVQuantizer with PRESET_GROUP64 for (c).
Naive baseline mitigation strategies for protocol comparison.
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
    FlexQuantizer, KVQuantizer, QuantConfig, PRESET_SECTION4, PRESET_GROUP64,
)
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def compute_attention_layer_ranking(model, tokenizer, prompts, n_calibration=20,
                                    max_new_tokens=32):
    """Compute per-layer attention mass from calibration prompts.

    Does a forward pass with output_attentions=True on n_calibration prompts,
    averages attention weights per layer, returns layers sorted by total mass.

    Returns: list of (layer_idx, mean_attention_mass) sorted descending.
    """
    device = next(model.parameters()).device
    cal_prompts = prompts[:n_calibration]

    print(f"  Computing attention mass from {len(cal_prompts)} calibration prompts...")

    # SDPA doesn't support output_attentions — temporarily switch to eager
    old_attn_impl = None
    if hasattr(model, 'config') and hasattr(model.config, '_attn_implementation'):
        old_attn_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    # Also set per-layer attention implementation for models that cache it
    for module in model.modules():
        if hasattr(module, '_attn_implementation'):
            module._attn_implementation = "eager"

    layer_masses = {}

    for p in cal_prompts:
        messages = [{"role": "user", "content": p.text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                use_cache=False,
            )

        if outputs.attentions is None:
            raise RuntimeError(
                "output_attentions=True returned None. "
                "This model may not support attention output. "
                "Try loading with attn_implementation='eager'."
            )

        # outputs.attentions: tuple of (batch, heads, seq, seq) per layer.
        # NOTE: attn.mean() is useless — each query row sums to 1 by softmax,
        # so mean(attn) = 1/seq_len for every layer (constant, not informative).
        # Instead we use mean peak attention per query: for each (batch, head,
        # query_pos), take the max weight across keys, then average. Higher
        # values mean this layer concentrates attention more sharply — the
        # metric H2O/KVzip-style methods implicitly rank layers on.
        for layer_idx, attn in enumerate(outputs.attentions):
            mass = attn.max(dim=-1).values.mean().item()
            layer_masses.setdefault(layer_idx, []).append(mass)

        del outputs, inputs
        torch.cuda.empty_cache()

    # Restore original attention implementation
    if old_attn_impl is not None:
        model.config._attn_implementation = old_attn_impl
        for module in model.modules():
            if hasattr(module, '_attn_implementation'):
                module._attn_implementation = old_attn_impl

    # Average across calibration prompts
    avg_masses = []
    for layer_idx in sorted(layer_masses.keys()):
        avg = sum(layer_masses[layer_idx]) / len(layer_masses[layer_idx])
        avg_masses.append((layer_idx, avg))

    # Sort by attention mass, descending
    avg_masses.sort(key=lambda x: x[1], reverse=True)
    return avg_masses


def parse_args():
    parser = argparse.ArgumentParser(description="Naive baseline comparison")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, required=True,
                        help="Base bit-width at collapse point")
    parser.add_argument("--prompts", default="advbench")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--n-calibration", type=int, default=20,
                        help="Number of prompts for attention mass calibration")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    n_layers = model_info["n_layers"]
    print(f"  Layers: {n_layers}, Hidden: {model_info['hidden_size']}")

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} {args.prompts} prompts")

    # Step 0: Compute attention-based layer ranking from calibration prompts
    attn_ranking = compute_attention_layer_ranking(
        model, tokenizer, prompts, n_calibration=args.n_calibration,
    )
    top2_attn_layers = [layer_idx for layer_idx, _ in attn_ranking[:2]]
    print(f"  Top-2 attention layers: {top2_attn_layers}")
    print(f"  (masses: {[f'L{l}={m:.4f}' for l, m in attn_ranking[:5]]})")

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "naive_baselines",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_bits": args.bits,
                "attention_ranking_top5": [
                    {"layer": l, "mass": m} for l, m in attn_ranking[:5]
                ],
                "top2_attn_layers": top2_attn_layers,
                "n_calibration": args.n_calibration,
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

    # Phase 1: Generation for all conditions

    # (0) FP16 baseline
    print("\n[baseline] FP16 generation...")
    gen_outputs = generate_responses_enhanced(
        model, tokenizer, prompt_texts,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions["baseline"] = ConditionResult(
            response=gen_out.response, refused=False, classifier="pending",
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )

    # (1) Unprotected (all layers quantized, per-token asymmetric)
    print(f"\n[unprotected] {args.bits}-bit all layers...")
    default_config = QuantConfig(
        bits=args.bits,
        symmetric=PRESET_SECTION4.symmetric,
        granularity=PRESET_SECTION4.granularity,
    )
    with KVQuantizer(model, default_config, model_info) as q:
        gen_outputs = generate_responses_enhanced(
            model, tokenizer, prompt_texts,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        unprotected_mse = q.get_mean_mse()
    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions["unprotected"] = ConditionResult(
            response=gen_out.response, refused=False, classifier="pending",
            kv_mse=unprotected_mse,
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )

    # (a) Naive: always protect layers 0-1 at FP16
    print(f"\n[naive_first2] FP16 layers 0-1, rest {args.bits}-bit...")
    layer_cfgs_a = {0: QuantConfig(bits=16), 1: QuantConfig(bits=16)}
    with FlexQuantizer(model, layer_cfgs_a, default_config, model_info) as fq:
        gen_outputs = generate_responses_enhanced(
            model, tokenizer, prompt_texts,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        mse_a = fq.get_mean_mse()
    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions["naive_first2"] = ConditionResult(
            response=gen_out.response, refused=False, classifier="pending",
            kv_mse=mse_a,
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )

    # (b) Naive: protect top-2 layers by attention mass at FP16
    attn_label = "_".join(f"L{l}" for l in sorted(top2_attn_layers))
    print(f"\n[naive_attn] FP16 layers {top2_attn_layers}, rest {args.bits}-bit...")
    layer_cfgs_b = {l: QuantConfig(bits=16) for l in top2_attn_layers}
    with FlexQuantizer(model, layer_cfgs_b, default_config, model_info) as fq:
        gen_outputs = generate_responses_enhanced(
            model, tokenizer, prompt_texts,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        mse_b = fq.get_mean_mse()
    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions["naive_attn"] = ConditionResult(
            response=gen_out.response, refused=False, classifier="pending",
            kv_mse=mse_b,
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )

    # (c) Naive: uniform Group-64 all layers
    print(f"\n[naive_g64] Group-64 all layers at {args.bits}-bit...")
    g64_config = QuantConfig(
        bits=args.bits,
        symmetric=PRESET_GROUP64.symmetric,
        granularity=PRESET_GROUP64.granularity,
        group_size=PRESET_GROUP64.group_size,
    )
    with KVQuantizer(model, g64_config, model_info) as q:
        gen_outputs = generate_responses_enhanced(
            model, tokenizer, prompt_texts,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        mse_c = q.get_mean_mse()
    for i, gen_out in enumerate(gen_outputs):
        result.prompts[i].conditions["naive_g64"] = ConditionResult(
            response=gen_out.response, refused=False, classifier="pending",
            kv_mse=mse_c,
            generation_time_s=gen_out.generation_time_s,
            input_token_count=gen_out.input_token_count,
            token_ids=gen_out.token_ids,
        )

    save_results(result, Path(args.output))

    # Phase 2: Classify
    print("\n[classify] Unloading generation model...")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    save_results(result, Path(args.output))

    try:
        yi_mode = "yi" in args.model.lower()
        classifier = get_classifier(args.classifier, yi_mode=yi_mode,
                                    device=args.classifier_device)
        classify_stored_results(result, classifier)
        classifier.unload()
    except Exception as e:
        print(f"  [WARN] Classification failed: {e}")

    result.summary = compute_summary(result, baseline_condition="baseline")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  NAIVE BASELINES: {args.model} @ {args.bits}-bit")
    print(f"{'='*70}")
    conditions = ["unprotected", "naive_first2", "naive_attn", "naive_g64"]
    labels = {
        "unprotected": f"Unprotected ({args.bits}-bit all)",
        "naive_first2": "Naive: FP16 L0-1",
        "naive_attn": f"Naive: FP16 attn top-2 ({attn_label})",
        "naive_g64": "Naive: Group-64 all",
    }
    print(f"{'Strategy':>35} | {'Refusal':>8} | {'Flip':>8} | {'Recovery':>10}")
    print("-" * 70)

    unprotected_fr = result.summary.get("unprotected", {}).get("flip_rate", 0)
    for cond in conditions:
        s = result.summary.get(cond, {})
        rr = s.get("refusal_rate", 0)
        fr = s.get("flip_rate")
        fr_s = f"{fr:.1%}" if fr is not None else "  -"
        if fr is not None and unprotected_fr > 0:
            recovery = 1.0 - fr / unprotected_fr
            rec_s = f"{recovery:.1%}"
        else:
            rec_s = "  -"
        label = labels.get(cond, cond)
        print(f"{label:>35} | {rr:>7.1%} | {fr_s:>8} | {rec_s:>10}")

    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
