#!/usr/bin/env python3
"""QuaRot-style rotation-based KV cache quantization experiment.

Tests whether Hadamard rotation before KV cache quantization changes the
alignment collapse behavior, and whether PCR correctly predicts the outcome.

Method: Apply Walsh-Hadamard rotation per-head before quantize-dequantize,
then inverse rotation after. This redistributes outlier energy uniformly
across channels within each attention head.

Reference: Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in
Rotated LLMs", NeurIPS 2024.
"""

import argparse
import gc
import math
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch import Tensor, nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import (
    KVQuantizer, QuantConfig, quantize_tensor,
    PRESET_SECTION4,
)
from core.generation import generate_responses_enhanced
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


# ---------------------------------------------------------------------------
# Hadamard rotation utilities
# ---------------------------------------------------------------------------

def _walsh_hadamard_unnorm(n: int) -> Tensor:
    """Build unnormalized Walsh-Hadamard matrix (entries ±1) of size n x n."""
    if n == 1:
        return torch.ones(1, 1)
    h = _walsh_hadamard_unnorm(n // 2)
    top = torch.cat([h, h], dim=1)
    bot = torch.cat([h, -h], dim=1)
    return torch.cat([top, bot], dim=0)


def hadamard_matrix(n: int) -> Tensor:
    """Normalized Walsh-Hadamard matrix: H @ H^T = I, H = H^T.

    n must be a power of 2.
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    return _walsh_hadamard_unnorm(n) / math.sqrt(n)


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Rotated KV quantizer
# ---------------------------------------------------------------------------

class KVQuantizerRotated(KVQuantizer):
    """KV quantizer with per-head Hadamard rotation before quantization.

    Flow per hook call:
      1. Reshape output to per-head: [..., num_kv_heads, head_dim]
      2. Pad head_dim to next power of 2 if needed
      3. Rotate per head: x_rot = x @ H
      4. Truncate back to original head_dim
      5. Reshape back to flat: [..., num_kv_heads * head_dim]
      6. Quantize-dequantize (same granularity as naive baseline)
      7. Reshape to per-head, pad, apply inverse rotation, truncate
      8. Reshape back to original shape
    """

    def __init__(self, model: nn.Module, config: QuantConfig, model_info: dict,
                 num_kv_heads: int, head_dim: int):
        super().__init__(model, config, model_info)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._padded_dim = _next_power_of_2(head_dim)
        self._needs_pad = self._padded_dim != head_dim

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self._H = hadamard_matrix(self._padded_dim).to(device=device, dtype=dtype)

    def _make_hook(self, layer_idx: int, proj_type: str):
        config = self.config
        H = self._H
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        padded_dim = self._padded_dim
        needs_pad = self._needs_pad

        def hook(module, input, output):
            if layer_idx not in self._target_layers:
                return output
            if proj_type == "k" and not config.quantize_keys:
                return output
            if proj_type == "v" and not config.quantize_values:
                return output

            original = output.detach()
            orig_shape = output.shape

            # Reshape to per-head: [..., num_kv_heads, head_dim]
            x = output.view(*orig_shape[:-1], num_kv_heads, head_dim)

            # Pad to next power of 2 if head_dim isn't one
            if needs_pad:
                x = F.pad(x, (0, padded_dim - head_dim))

            # Rotate per head: x @ H (broadcasts across batch, seq, heads)
            x_rot = torch.matmul(x, H)

            # Truncate back to original head_dim
            if needs_pad:
                x_rot = x_rot[..., :head_dim]

            # Reshape back to flat for quantization
            x_flat = x_rot.reshape(orig_shape)

            # Quantize-dequantize (same granularity as naive)
            x_quant = quantize_tensor(
                x_flat,
                bits=config.bits,
                symmetric=config.symmetric,
                granularity=config.granularity,
                group_size=config.group_size,
                channels=config.channels,
            )

            # Reshape to per-head for inverse rotation
            x_q_heads = x_quant.view(*orig_shape[:-1], num_kv_heads, head_dim)

            # Pad again for inverse rotation
            if needs_pad:
                x_q_heads = F.pad(x_q_heads, (0, padded_dim - head_dim))

            # Inverse rotation: H^{-1} = H^T = H for normalized Walsh-Hadamard
            x_out = torch.matmul(x_q_heads, H)

            # Truncate and reshape back to original
            if needs_pad:
                x_out = x_out[..., :head_dim]
            result = x_out.reshape(orig_shape)

            mse = F.mse_loss(result.detach(), original).item()
            self._mse_values.append(mse)
            self._per_layer_mse.setdefault(layer_idx, []).append(mse)

            return result

        return hook


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="QuaRot-style rotation-based KV cache quantization experiment"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=str, default="4,2",
                        help="Comma-separated bit-widths to test")
    parser.add_argument("--prompts", default="advbench")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    print(f"  Layers: {model_info['n_layers']}, Hidden: {model_info['hidden_size']}")

    # Get num_kv_heads and head_dim from model config
    num_kv_heads = getattr(model.config, "num_key_value_heads",
                           getattr(model.config, "num_attention_heads", None))
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"  KV heads: {num_kv_heads}, Head dim: {head_dim}")

    padded_dim = _next_power_of_2(head_dim)
    if padded_dim != head_dim:
        print(f"  [note] head_dim {head_dim} is not power of 2, padding to {padded_dim}")

    # Load prompts
    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} {args.prompts} prompts")

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "quarot",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": bits_list,
                "naive": {
                    "granularity": PRESET_SECTION4.granularity,
                    "symmetric": PRESET_SECTION4.symmetric,
                },
                "rotated": {
                    "method": "walsh_hadamard",
                    "scope": "per_head",
                    "head_dim": head_dim,
                    "num_kv_heads": num_kv_heads,
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

    # ===== Phase 1: Generation =====

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
    print(f"  Baseline: {len(gen_outputs)} responses")

    for bits in bits_list:
        # --- Naive per-token asymmetric ---
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

        # --- Rotated per-token asymmetric ---
        rot_key = f"{bits}_rotated"
        print(f"\n[{rot_key}] Hadamard-rotated {bits}-bit...")

        rot_config = QuantConfig(
            bits=bits,
            symmetric=PRESET_SECTION4.symmetric,
            granularity=PRESET_SECTION4.granularity,
        )
        with KVQuantizerRotated(model, rot_config, model_info,
                                num_kv_heads, head_dim) as q:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer, prompt_texts,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                seed=args.seed,
            )
            rot_mse = q.get_mean_mse()

        for i, gen_out in enumerate(gen_outputs):
            result.prompts[i].conditions[rot_key] = ConditionResult(
                response=gen_out.response, refused=False, classifier="pending",
                kv_mse=rot_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
                token_ids=gen_out.token_ids,
            )
        print(f"  {rot_key}: {len(gen_outputs)} responses, MSE={rot_mse:.6f}")

        save_results(result, Path(args.output))
        print(f"  [checkpoint] Saved after {bits}-bit")

    # ===== Phase 2: Classify =====
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

    # Print comparison table
    print(f"\n{'='*75}")
    print(f"  QUAROT vs NAIVE: {args.model}")
    print(f"{'='*75}")
    print(f"{'Condition':>20} | {'Refusal':>8} | {'Flip':>8} | {'MSE':>10}")
    print("-" * 55)

    bl_s = result.summary.get("16", {})
    bl_rr = bl_s.get("refusal_rate", 0)
    print(f"{'FP16':>20} | {bl_rr:>7.1%} |      -   |          -")

    for bits in bits_list:
        for suffix in ["naive", "rotated"]:
            cond = f"{bits}_{suffix}"
            s = result.summary.get(cond, {})
            rr = s.get("refusal_rate", 0)
            fr = s.get("flip_rate")
            mse = s.get("mean_kv_mse")
            fr_s = f"{fr:.1%}" if fr is not None else "  -"
            mse_s = f"{mse:.6f}" if mse is not None else "  -"
            print(f"{cond:>20} | {rr:>7.1%} | {fr_s:>8} | {mse_s:>10}")
        naive_s = result.summary.get(f"{bits}_naive", {})
        rot_s = result.summary.get(f"{bits}_rotated", {})
        n_fr = naive_s.get("flip_rate")
        r_fr = rot_s.get("flip_rate")
        if n_fr is not None and r_fr is not None:
            delta = r_fr - n_fr
            print(f"  --> Rotation vs naive at {bits}-bit: {delta:+.1%} flip rate")

    save_results(result, Path(args.output))
    print(f"\n[done] Results saved to {args.output}")


if __name__ == "__main__":
    main()
