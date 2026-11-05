#!/usr/bin/env python3
"""Direct refusal direction probing.

Extracts the refusal direction (Arditi et al., NeurIPS 2024) via
difference-in-means on residual stream activations, then measures how
much 3-bit per-tensor quantization noise projects onto that direction.

The causal chain:
  1. Extract refusal direction at the critical layer (difference-in-means)
  2. Compute quantization noise: quantized_K - original_K
  3. Project noise onto refusal direction
  4. Show low-PCR models have higher noise projection (safety channels
     overlap with outlier channels)
"""

import argparse
import gc
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model, get_model_info
from core.quantization import quantize_tensor
from core.prompts import load_prompts
from core.results import ExperimentResult, save_results

# Model -> critical layer mapping (from the layer scan)
CRITICAL_LAYERS = {
    "qwen": 0,
    "llama": 3,
    "mistral": 3,
    "gemma2": 1,
}

PCR_VALUES = {
    "qwen": 54.5,
    "llama": 70.0,
    "mistral": 76.9,
    "gemma2": 100.0,
}

HARMLESS_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Write a poem about the ocean.",
    "What are the benefits of regular exercise?",
    "How does a combustion engine work?",
    "Describe the water cycle.",
    "What is the Pythagorean theorem?",
    "List three types of renewable energy.",
    "Who wrote Romeo and Juliet?",
    "What causes rainbows?",
    "Explain how vaccines work.",
    "What is the speed of light?",
    "Describe the process of making bread.",
    "What are the planets in our solar system?",
    "How do birds migrate?",
    "What is the tallest mountain in the world?",
    "Explain the concept of supply and demand.",
    "What is the difference between weather and climate?",
    "How does the internet work?",
    "What are the primary colors?",
    "Describe the life cycle of a butterfly.",
    "What is the boiling point of water?",
    "How do magnets work?",
    "What is the largest ocean on Earth?",
    "Explain the theory of relativity simply.",
    "What are the five senses?",
    "How do plants absorb water?",
    "What is democracy?",
    "Describe how a refrigerator works.",
    "What is the chemical formula for water?",
    "How do earthquakes happen?",
    "What is the Fibonacci sequence?",
    "Explain the greenhouse effect.",
    "What are the stages of human development?",
    "How does a compass work?",
    "What is the periodic table?",
    "Describe the structure of DNA.",
    "What causes tides?",
    "How do airplanes fly?",
    "What is the significance of the Rosetta Stone?",
    "Explain how solar panels work.",
    "What are the four seasons?",
    "How does sound travel?",
    "What is the human genome?",
    "Describe the process of photosynthesis.",
    "What is an ecosystem?",
    "How do telescopes work?",
    "What is gravity?",
    "Explain the concept of inflation in economics.",
    "What are antibiotics?",
]


def collect_residual_stream(model, tokenizer, prompts, layer_idx):
    """Collect residual stream activations at the output of a transformer block.

    Returns tensor of shape (n_prompts, hidden_size) — last token position.
    """
    device = next(model.parameters()).device
    target_layer = model.model.layers[layer_idx]

    activations = []

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        last_token = hidden[:, -1, :].detach().cpu().float()
        activations.append(last_token)

    handle = target_layer.register_forward_hook(hook)
    try:
        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return torch.cat(activations, dim=0)


def collect_kproj_with_noise(model, tokenizer, prompts, layer_idx, model_info, bits=3):
    """Collect original and quantized K-projection activations at a layer.

    Returns (original_K, quantized_K) each of shape (n_prompts, k_dim).
    Both are last-token activations.
    """
    device = next(model.parameters()).device
    fused_qkv = model_info.get("fused_qkv", False)
    hs = model_info["hidden_size"]

    originals = []
    quantized_list = []

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
        raise ValueError(f"Could not find k_proj for layer {layer_idx}")

    def hook(module, input, output):
        if fused_qkv:
            k_act = output[..., hs:2*hs]
        else:
            k_act = output

        last_orig = k_act[:, -1, :].detach().cpu().float()
        originals.append(last_orig)

        k_quant = quantize_tensor(
            k_act,
            bits=bits,
            symmetric=True,
            granularity="per_tensor",
        )
        last_quant = k_quant[:, -1, :].detach().cpu().float()
        quantized_list.append(last_quant)

        return output

    handle = target_module.register_forward_hook(hook)
    try:
        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return torch.cat(originals, dim=0), torch.cat(quantized_list, dim=0)


def extract_refusal_direction(harmful_acts, harmless_acts):
    """Compute refusal direction via difference-in-means (Arditi et al., 2024).

    Returns normalized refusal direction vector.
    """
    mean_harmful = harmful_acts.mean(dim=0)
    mean_harmless = harmless_acts.mean(dim=0)
    refusal_dir = mean_harmful - mean_harmless
    refusal_dir = refusal_dir / refusal_dir.norm()
    return refusal_dir


def validate_refusal_direction(harmful_acts, harmless_acts, refusal_dir):
    """Validate that the refusal direction separates harmful from harmless."""
    cos_harmful = torch.nn.functional.cosine_similarity(
        harmful_acts, refusal_dir.unsqueeze(0), dim=1
    )
    cos_harmless = torch.nn.functional.cosine_similarity(
        harmless_acts, refusal_dir.unsqueeze(0), dim=1
    )
    return {
        "mean_cos_sim_harmful": cos_harmful.mean().item(),
        "std_cos_sim_harmful": cos_harmful.std().item(),
        "mean_cos_sim_harmless": cos_harmless.mean().item(),
        "std_cos_sim_harmless": cos_harmless.std().item(),
        "separation": (cos_harmful.mean() - cos_harmless.mean()).item(),
    }


def compute_noise_projection(noise, refusal_dir):
    """Compute how much quantization noise projects onto the refusal direction.

    Args:
        noise: (n_prompts, hidden_size) quantization noise vectors
        refusal_dir: (hidden_size,) normalized refusal direction

    Returns dict with per-prompt and aggregate statistics.
    """
    noise_dot_refusal = (noise * refusal_dir.unsqueeze(0)).sum(dim=1)
    abs_noise_dot_refusal = noise_dot_refusal.abs()

    noise_norms = noise.norm(dim=1)
    noise_norms_safe = noise_norms.clamp(min=1e-10)

    noise_on_refusal_frac = abs_noise_dot_refusal / noise_norms_safe

    return {
        "mean_noise_on_refusal_frac": noise_on_refusal_frac.mean().item(),
        "std_noise_on_refusal_frac": noise_on_refusal_frac.std().item(),
        "mean_abs_noise_dot_refusal": abs_noise_dot_refusal.mean().item(),
        "std_abs_noise_dot_refusal": abs_noise_dot_refusal.std().item(),
        "mean_noise_magnitude": noise_norms.mean().item(),
        "std_noise_magnitude": noise_norms.std().item(),
        "mean_signed_noise_dot_refusal": noise_dot_refusal.mean().item(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Refusal direction probing")
    parser.add_argument("--model", required=True, choices=list(CRITICAL_LAYERS.keys()))
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local path to model weights (overrides HF download)")
    parser.add_argument("--max-prompts", type=int, default=50,
                        help="Number of harmful/harmless prompts (default 50)")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    critical_layer = CRITICAL_LAYERS[model_name]
    pcr = PCR_VALUES[model_name]
    n_prompts = args.max_prompts

    print(f"=== Refusal Direction Probing ===")
    print(f"Model: {model_name} | Critical layer: {critical_layer} | PCR: {pcr}%")
    print(f"Prompts: {n_prompts} harmful + {n_prompts} harmless")
    print()

    # Load prompts
    print("[1/5] Loading prompts...")
    advbench = load_prompts("advbench")
    harmful_texts = [p.text for p in advbench[:n_prompts]]
    harmless_texts = HARMLESS_PROMPTS[:n_prompts]

    actual_harmful = len(harmful_texts)
    actual_harmless = len(harmless_texts)
    print(f"  Harmful: {actual_harmful}, Harmless: {actual_harmless}")

    # Load model
    print(f"\n[2/5] Loading model: {model_name}...")
    if args.model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_info = get_model_info(model_name)
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        model.eval()
    else:
        model, tokenizer, model_info = load_model(model_name)
    print(f"  Loaded: {model_info['hf_id']}")
    print(f"  Layers: {model_info['n_layers']}, Hidden: {model_info['hidden_size']}")

    # Part A: Extract refusal direction
    print(f"\n[3/5] Extracting refusal direction at layer {critical_layer}...")
    print(f"  Collecting harmful activations ({actual_harmful} prompts)...")
    harmful_acts = collect_residual_stream(
        model, tokenizer, harmful_texts, critical_layer
    )
    print(f"    Shape: {harmful_acts.shape}")

    print(f"  Collecting harmless activations ({actual_harmless} prompts)...")
    harmless_acts = collect_residual_stream(
        model, tokenizer, harmless_texts, critical_layer
    )
    print(f"    Shape: {harmless_acts.shape}")

    refusal_dir = extract_refusal_direction(harmful_acts, harmless_acts)
    print(f"  Refusal direction norm (pre-normalize): "
          f"{(harmful_acts.mean(0) - harmless_acts.mean(0)).norm().item():.4f}")

    # Validate
    validation = validate_refusal_direction(harmful_acts, harmless_acts, refusal_dir)
    print(f"  Validation:")
    print(f"    Mean cos(harmful, refusal_dir):  {validation['mean_cos_sim_harmful']:.4f}")
    print(f"    Mean cos(harmless, refusal_dir): {validation['mean_cos_sim_harmless']:.4f}")
    print(f"    Separation:                      {validation['separation']:.4f}")

    if validation["separation"] < 0.05:
        print(f"  WARNING: Weak separation ({validation['separation']:.4f} < 0.05)")
        print(f"  The refusal direction may not reliably separate harmful/harmless at this layer.")

    # Part B: Compute quantization noise projection
    print(f"\n[4/5] Computing quantization noise at layer {critical_layer} ({args.bits}-bit per-tensor sym)...")
    print(f"  Collecting K-proj activations (original + quantized)...")
    orig_k, quant_k = collect_kproj_with_noise(
        model, tokenizer, harmful_texts, critical_layer, model_info, bits=args.bits
    )
    noise = quant_k - orig_k
    print(f"    K-proj shape: {orig_k.shape}")
    print(f"    Mean noise magnitude: {noise.norm(dim=1).mean().item():.4f}")

    # Project noise onto refusal direction
    # The refusal direction is in residual stream space (hidden_size),
    # while K-proj noise is in k_proj output space (could be different dim
    # for GQA models). We need to handle dimension mismatch.
    k_dim = orig_k.shape[1]
    h_dim = refusal_dir.shape[0]

    if k_dim == h_dim:
        noise_proj = compute_noise_projection(noise, refusal_dir)
    else:
        # For GQA models, k_proj dim != hidden_size.
        # Project the refusal direction into K-space by taking the first k_dim
        # components, then re-normalize. This is an approximation — the true
        # projection would require the k_proj weight matrix.
        print(f"  NOTE: K-proj dim ({k_dim}) != hidden dim ({h_dim}), "
              f"using k_proj weight matrix for projection")
        # Find the k_proj weight matrix
        k_proj_module = None
        for name, module in model.named_modules():
            if f"layers.{critical_layer}." in name and name.endswith("k_proj"):
                k_proj_module = module
                break

        if k_proj_module is not None and hasattr(k_proj_module, "weight"):
            W_k = k_proj_module.weight.detach().cpu().float()  # (k_dim, h_dim)
            refusal_in_k = W_k @ refusal_dir  # (k_dim,)
            refusal_in_k = refusal_in_k / refusal_in_k.norm()
            noise_proj = compute_noise_projection(noise, refusal_in_k)
        else:
            print(f"  WARNING: Could not find k_proj weight, truncating refusal dir")
            trunc_dir = refusal_dir[:k_dim]
            trunc_dir = trunc_dir / trunc_dir.norm()
            noise_proj = compute_noise_projection(noise, trunc_dir)

    print(f"  Noise projection results:")
    print(f"    Mean |noise·refusal|/||noise||: {noise_proj['mean_noise_on_refusal_frac']:.6f}")
    print(f"    Mean |noise·refusal|:           {noise_proj['mean_abs_noise_dot_refusal']:.4f}")
    print(f"    Mean ||noise||:                 {noise_proj['mean_noise_magnitude']:.4f}")

    # Compute random baseline: expected projection onto a random direction
    random_baseline = 1.0 / np.sqrt(k_dim)  # E[|x·r|/||x||] for random unit r
    amplification = noise_proj["mean_noise_on_refusal_frac"] / random_baseline
    print(f"    Random baseline (1/sqrt(d)):    {random_baseline:.6f}")
    print(f"    Amplification vs random:        {amplification:.2f}x")

    # Unload model
    print(f"\n[5/5] Saving results...")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Build result
    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", model_name),
            "model_name": model_name,
            "experiment": "refusal_direction_probing",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "critical_layer": critical_layer,
                "pcr": pcr,
                "bits": args.bits,
                "granularity": "per_tensor",
                "symmetric": True,
                "n_harmful": actual_harmful,
                "n_harmless": actual_harmless,
                "method": "difference_in_means (Arditi et al., 2024)",
            },
        },
        prompts=[],
        summary={
            "refusal_direction": {
                "method": "difference_in_means",
                "reference": "Arditi et al., Refusal in Language Models Is Mediated by a Single Direction, NeurIPS 2024",
                "n_harmful": actual_harmful,
                "n_harmless": actual_harmless,
                "layer": critical_layer,
                "activation_source": "residual_stream_post (transformer block output)",
                "token_position": "last",
                "validation": validation,
            },
            "noise_projection": {
                **noise_proj,
                "random_baseline_frac": random_baseline,
                "amplification_vs_random": amplification,
                "k_proj_dim": k_dim,
                "hidden_dim": h_dim,
            },
            "pcr": pcr,
        },
    )

    save_results(result, Path(args.output))
    print(f"\n=== Done: {model_name} ===")
    print(f"  PCR: {pcr}%")
    print(f"  Refusal dir separation: {validation['separation']:.4f}")
    print(f"  Noise→refusal fraction: {noise_proj['mean_noise_on_refusal_frac']:.6f}")
    print(f"  Amplification vs random: {amplification:.2f}x")


if __name__ == "__main__":
    main()
