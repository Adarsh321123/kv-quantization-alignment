#!/usr/bin/env python3
"""PCR Predictive Validation Experiment.

Demonstrates that PCR computed on a small calibration set (20 custom prompts)
accurately predicts mitigation effectiveness on unseen test data (AdvBench).

For each model:
1. Run channel ablation on calibration set → compute calibration PCR
2. Run per-tensor + per-channel + group-64 quantization on test set (AdvBench)
3. Compute actual drift reduction on test set
4. Compare calibration PCR vs actual drift reduction

Validates PCR as a generalizable diagnostic, not a dataset-specific artifact.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model, resolve_hf_id, get_model_info
from core.quantization import KVQuantizer, QuantConfig, PRESET_SECTION5
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


DEFAULT_CRITICAL_LAYERS = {
    "phi35": 2, "qwen7b": 0, "mistral7b": 0, "deepseek7b": 17,
    "llama8b": 2, "yi9b": 16, "gemma9b": 1, "msmall24b": 14,
    "qwen72b": 0, "yi34b": 0,
    "microsoft/Phi-3.5-mini-instruct": 2,
    "Qwen/Qwen2.5-7B-Instruct": 0,
    "mistralai/Mistral-7B-Instruct-v0.2": 0,
    "deepseek-ai/deepseek-llm-7b-chat": 17,
    "meta-llama/Llama-3.1-8B-Instruct": 2,
    "01-ai/Yi-1.5-9B-Chat": 16,
    "google/gemma-2-9b-it": 1,
    "mistralai/Mistral-Small-24B-Instruct-2501": 14,
    "Qwen/Qwen2.5-72B-Instruct": 0,
    "01-ai/Yi-1.5-34B-Chat": 0,
}


def get_channel_magnitudes(model, tokenizer, prompts, layer_idx, model_info):
    """Capture K-proj activations and compute per-channel magnitudes."""
    device = next(model.parameters()).device
    fused_qkv = model_info.get("fused_qkv", False)
    hs = model_info["hidden_size"]

    activations = []
    target_module = None
    for name, module in model.named_modules():
        if f"layers.{layer_idx}." not in name:
            continue
        if fused_qkv and "qkv_proj" in name and name.endswith("qkv_proj"):
            target_module = module
            break
        elif not fused_qkv and "k_proj" in name and name.endswith("k_proj"):
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Could not find projection module for layer {layer_idx}")

    def hook(module, input, output):
        if fused_qkv:
            k_act = output[..., hs:2*hs].detach().cpu()
        else:
            k_act = output.detach().cpu()
        activations.append(k_act)

    handle = target_module.register_forward_hook(hook)
    try:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt.text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                             max_length=512).to(device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    if not activations:
        raise RuntimeError("No activations captured")

    all_acts = torch.cat([a.squeeze(0) for a in activations], dim=0).float()
    channel_max = all_acts.abs().max(dim=0).values.numpy()
    return channel_max


def parse_args():
    parser = argparse.ArgumentParser(description="PCR Predictive Validation")
    parser.add_argument("--model", required=True, help="Model name (canonical or HF ID)")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--critical-layer", type=int, default=None,
                        help="Critical layer (auto-detected if not specified)")
    parser.add_argument("--calibration-prompts", default="custom")
    parser.add_argument("--calibration-size", type=int, default=20)
    parser.add_argument("--test-prompts", default="advbench")
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="family_a")
    parser.add_argument("--classifier-device", default="auto")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit (for large models that don't fit in bf16)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (for very large models)")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

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
    print(f"  Layers: {model_info['n_layers']}, Hidden: {model_info['hidden_size']}")

    cal_prompts = load_prompts(args.calibration_prompts)[:args.calibration_size]
    test_prompts = load_prompts(args.test_prompts)[:args.test_size]
    print(f"  Calibration: {len(cal_prompts)} {args.calibration_prompts} prompts")
    print(f"  Test: {len(test_prompts)} {args.test_prompts} prompts")

    critical_layer = args.critical_layer
    if critical_layer is None:
        critical_layer = DEFAULT_CRITICAL_LAYERS.get(args.model, 0)
    print(f"  Critical layer: {critical_layer}")

    yi_mode = "yi" in args.model.lower()
    preset = PRESET_SECTION5

    # ==================== PHASE 1: Calibration ====================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Calibration ({len(cal_prompts)} prompts)")
    print(f"{'='*60}")

    cal_result = ExperimentResult(metadata={
        "model": model_info.get("hf_id", args.model),
        "experiment": "pcr_prediction_calibration",
        "timestamp": datetime.now().isoformat(),
        "config": {"bits": args.bits, "layer": critical_layer,
                    "calibration_set": args.calibration_prompts,
                    "calibration_size": len(cal_prompts)},
    })
    for p in cal_prompts:
        cal_result.prompts.append(PromptResult(
            prompt_idx=p.index, prompt_text=p.text, category=p.category,
        ))

    # Measure channel magnitudes
    print(f"\n[cal] Measuring channel magnitudes at layer {critical_layer}...")
    channel_max = get_channel_magnitudes(
        model, tokenizer, cal_prompts, critical_layer, model_info
    )
    n_channels = len(channel_max)
    threshold = np.percentile(channel_max, 95)
    outlier_channels = np.where(channel_max >= threshold)[0].tolist()
    print(f"  {n_channels} channels, {len(outlier_channels)} outliers (top 5%)")

    # Random-matched subset for causal comparison
    np.random.seed(42)
    random_channels = np.random.choice(
        n_channels, len(outlier_channels), replace=False
    ).tolist()

    # Calibration: FP16 baseline
    print(f"\n[cal/baseline] FP16 generation...")
    cal_baseline = generate_responses_enhanced(
        model, tokenizer, [p.text for p in cal_prompts],
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    for i, gen in enumerate(cal_baseline):
        cal_result.prompts[i].conditions["baseline"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            generation_time_s=gen.generation_time_s,
        )

    # Calibration: per-tensor (critical layer only)
    print(f"\n[cal/per_tensor] Per-tensor @ layer {critical_layer}...")
    pt_config = QuantConfig(bits=args.bits, symmetric=preset.symmetric,
                            granularity="per_tensor", layers=[critical_layer])
    with KVQuantizer(model, pt_config, model_info) as q:
        cal_pt = generate_responses_enhanced(
            model, tokenizer, [p.text for p in cal_prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        cal_pt_mse = q.get_mean_mse()
    for i, gen in enumerate(cal_pt):
        cal_result.prompts[i].conditions["per_tensor"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            kv_mse=cal_pt_mse, generation_time_s=gen.generation_time_s,
        )

    # Calibration: per-channel (critical layer only)
    print(f"\n[cal/per_channel] Per-channel @ layer {critical_layer}...")
    pc_config = QuantConfig(bits=args.bits, symmetric=preset.symmetric,
                            granularity="per_channel", layers=[critical_layer])
    with KVQuantizer(model, pc_config, model_info) as q:
        cal_pc = generate_responses_enhanced(
            model, tokenizer, [p.text for p in cal_prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        cal_pc_mse = q.get_mean_mse()
    for i, gen in enumerate(cal_pc):
        cal_result.prompts[i].conditions["per_channel"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            kv_mse=cal_pc_mse, generation_time_s=gen.generation_time_s,
        )

    # Calibration: outlier-only
    print(f"\n[cal/outliers] Outlier-only ({len(outlier_channels)} channels)...")
    out_config = QuantConfig(bits=args.bits, symmetric=preset.symmetric,
                             granularity="per_tensor", layers=[critical_layer],
                             channels=outlier_channels)
    with KVQuantizer(model, out_config, model_info) as q:
        cal_outlier = generate_responses_enhanced(
            model, tokenizer, [p.text for p in cal_prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
    for i, gen in enumerate(cal_outlier):
        cal_result.prompts[i].conditions["outliers"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            generation_time_s=gen.generation_time_s,
        )

    # Calibration: random-matched
    print(f"\n[cal/random] Random-matched ({len(random_channels)} channels)...")
    rand_config = QuantConfig(bits=args.bits, symmetric=preset.symmetric,
                              granularity="per_tensor", layers=[critical_layer],
                              channels=random_channels)
    with KVQuantizer(model, rand_config, model_info) as q:
        cal_random = generate_responses_enhanced(
            model, tokenizer, [p.text for p in cal_prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
    for i, gen in enumerate(cal_random):
        cal_result.prompts[i].conditions["random"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            generation_time_s=gen.generation_time_s,
        )

    save_results(cal_result, Path(args.output).parent / "pcr_calibration.json")

    # ==================== PHASE 2: Test Set ====================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Test Set ({len(test_prompts)} {args.test_prompts} prompts)")
    print(f"{'='*60}")

    test_result = ExperimentResult(metadata={
        "model": model_info.get("hf_id", args.model),
        "experiment": "pcr_prediction_test",
        "timestamp": datetime.now().isoformat(),
        "config": {"bits": args.bits, "test_set": args.test_prompts,
                    "test_size": len(test_prompts)},
    })
    for p in test_prompts:
        test_result.prompts.append(PromptResult(
            prompt_idx=p.index, prompt_text=p.text, category=p.category,
        ))

    # Test: FP16 baseline
    print(f"\n[test/baseline] FP16 generation...")
    test_baseline = generate_responses_enhanced(
        model, tokenizer, [p.text for p in test_prompts],
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    for i, gen in enumerate(test_baseline):
        test_result.prompts[i].conditions["baseline"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            generation_time_s=gen.generation_time_s,
        )

    # Test: per-tensor (ALL layers — deployment mode)
    print(f"\n[test/per_tensor] Per-tensor (all layers)...")
    pt_all = QuantConfig(bits=args.bits, symmetric=preset.symmetric,
                         granularity="per_tensor")
    with KVQuantizer(model, pt_all, model_info) as q:
        test_pt = generate_responses_enhanced(
            model, tokenizer, [p.text for p in test_prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        test_pt_mse = q.get_mean_mse()
    for i, gen in enumerate(test_pt):
        test_result.prompts[i].conditions["per_tensor"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            kv_mse=test_pt_mse, generation_time_s=gen.generation_time_s,
        )

    # Test: per-channel (ALL layers)
    print(f"\n[test/per_channel] Per-channel (all layers)...")
    pc_all = QuantConfig(bits=args.bits, symmetric=preset.symmetric,
                         granularity="per_channel")
    with KVQuantizer(model, pc_all, model_info) as q:
        test_pc = generate_responses_enhanced(
            model, tokenizer, [p.text for p in test_prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        test_pc_mse = q.get_mean_mse()
    for i, gen in enumerate(test_pc):
        test_result.prompts[i].conditions["per_channel"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            kv_mse=test_pc_mse, generation_time_s=gen.generation_time_s,
        )

    # Test: group-64 (ALL layers)
    print(f"\n[test/group64] Group-64 (all layers)...")
    g64_all = QuantConfig(bits=args.bits, symmetric=preset.symmetric,
                          granularity="per_group", group_size=64)
    with KVQuantizer(model, g64_all, model_info) as q:
        test_g64 = generate_responses_enhanced(
            model, tokenizer, [p.text for p in test_prompts],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        test_g64_mse = q.get_mean_mse()
    for i, gen in enumerate(test_g64):
        test_result.prompts[i].conditions["group64"] = ConditionResult(
            response=gen.response, refused=False, classifier="pending",
            kv_mse=test_g64_mse, generation_time_s=gen.generation_time_s,
        )

    save_results(test_result, Path(args.output).parent / "pcr_test.json")

    # ==================== PHASE 3: Classification ====================
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Classification")
    print(f"{'='*60}")

    print("[classify] Unloading generation model...")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    try:
        classifier = get_classifier(args.classifier, yi_mode=yi_mode,
                                    device=args.classifier_device)
        print("[classify] Classifying calibration set...")
        classify_stored_results(cal_result, classifier)
        print("[classify] Classifying test set...")
        classify_stored_results(test_result, classifier)
        classifier.unload()
    except Exception as e:
        print(f"  [WARN] Classification failed: {e}")

    cal_result.summary = compute_summary(cal_result, baseline_condition="baseline")
    test_result.summary = compute_summary(test_result, baseline_condition="baseline")

    save_results(cal_result, Path(args.output).parent / "pcr_calibration.json")
    save_results(test_result, Path(args.output).parent / "pcr_test.json")

    # ==================== PHASE 4: Analysis ====================
    print(f"\n{'='*60}")
    print(f"  PHASE 4: PCR Prediction Analysis")
    print(f"{'='*60}")

    cal_pt_flip = cal_result.summary.get("per_tensor", {}).get("flip_rate", 0)
    cal_pc_flip = cal_result.summary.get("per_channel", {}).get("flip_rate", 0)
    cal_outlier_flip = cal_result.summary.get("outliers", {}).get("flip_rate", 0)
    cal_random_flip = cal_result.summary.get("random", {}).get("flip_rate", 0)
    cal_pcr = 1.0 - (cal_pc_flip / cal_pt_flip) if cal_pt_flip > 0 else 0.0

    test_pt_flip = test_result.summary.get("per_tensor", {}).get("flip_rate", 0)
    test_pc_flip = test_result.summary.get("per_channel", {}).get("flip_rate", 0)
    test_g64_flip = test_result.summary.get("group64", {}).get("flip_rate", 0)
    test_pcr = 1.0 - (test_pc_flip / test_pt_flip) if test_pt_flip > 0 else 0.0
    test_g64r = 1.0 - (test_g64_flip / test_pt_flip) if test_pt_flip > 0 else 0.0

    causal_ratio = (cal_outlier_flip / cal_random_flip
                    if cal_random_flip > 0 else float('inf'))

    pcr_error = abs(cal_pcr - test_pcr)
    # Both agree on whether PCR is high (mitigation will help)
    prediction_correct = (cal_pcr > 0.3) == (test_pcr > 0.3)

    output = {
        "metadata": {
            "model": model_info.get("hf_id", args.model),
            "experiment": "pcr_prediction",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": args.bits,
                "critical_layer": critical_layer,
                "calibration_set": args.calibration_prompts,
                "calibration_size": len(cal_prompts),
                "test_set": args.test_prompts,
                "test_size": len(test_prompts),
            },
        },
        "calibration": {
            "per_tensor_flip": cal_pt_flip,
            "per_channel_flip": cal_pc_flip,
            "outlier_flip": cal_outlier_flip,
            "random_flip": cal_random_flip,
            "pcr": cal_pcr,
            "causal_ratio": causal_ratio if causal_ratio != float('inf') else None,
            "n_prompts": len(cal_prompts),
            "n_outlier_channels": len(outlier_channels),
            "n_total_channels": n_channels,
        },
        "test": {
            "per_tensor_flip": test_pt_flip,
            "per_channel_flip": test_pc_flip,
            "group64_flip": test_g64_flip,
            "pcr": test_pcr,
            "group64_reduction": test_g64r,
            "n_prompts": len(test_prompts),
        },
        "prediction": {
            "calibration_pcr": cal_pcr,
            "test_pcr": test_pcr,
            "pcr_absolute_error": pcr_error,
            "prediction_correct": prediction_correct,
            "calibration_predicts_mitigation": cal_pcr > 0.3,
            "actual_mitigation_effective": test_pcr > 0.3,
        },
    }

    print(f"\n  CALIBRATION ({len(cal_prompts)} {args.calibration_prompts} prompts, layer {critical_layer}):")
    print(f"    Per-tensor flip:  {cal_pt_flip:.1%}")
    print(f"    Per-channel flip: {cal_pc_flip:.1%}")
    print(f"    Outlier flip:     {cal_outlier_flip:.1%}")
    print(f"    Random flip:      {cal_random_flip:.1%}")
    print(f"    PCR:              {cal_pcr:.3f}")
    if causal_ratio != float('inf'):
        print(f"    Causal ratio:     {causal_ratio:.1f}x")

    print(f"\n  TEST ({len(test_prompts)} {args.test_prompts} prompts, all layers):")
    print(f"    Per-tensor flip:  {test_pt_flip:.1%}")
    print(f"    Per-channel flip: {test_pc_flip:.1%}")
    print(f"    Group-64 flip:    {test_g64_flip:.1%}")
    print(f"    Test PCR:         {test_pcr:.3f}")
    print(f"    Group-64 red.:    {test_g64r:.3f}")

    print(f"\n  PREDICTION:")
    print(f"    Cal PCR → Test PCR error: {pcr_error:.3f}")
    print(f"    Prediction correct: {prediction_correct}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
