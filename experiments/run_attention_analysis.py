#!/usr/bin/env python3
"""Attention-based layer importance analysis.

Captures attention weights during inference to compute per-layer importance
scores. Used for tab:causal_vs_attention and tab:protocol_vs_attention to
compare attention-based layer selection vs causal (ablation-based) selection.

Two importance metrics:
1. Mean attention entropy: higher entropy = more distributed attention = more important
2. Attention to safety-relevant tokens: measures how much each layer attends to
   the safety-relevant parts of the prompt

Outputs a result JSON with per-layer attention statistics that can be compared
against layer ablation results to evaluate causal vs attention-based protection.
"""

import argparse
import sys
import math
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results,
)


def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """Compute mean entropy of attention distributions.

    Args:
        attn_weights: [batch, heads, seq_q, seq_k] attention probability tensor

    Returns:
        Mean entropy across all heads and query positions.
    """
    # Clamp for numerical stability
    attn_weights = attn_weights.clamp(min=1e-10)
    entropy = -(attn_weights * attn_weights.log()).sum(dim=-1)  # [batch, heads, seq_q]
    return entropy.mean().item()


def compute_attention_concentration(attn_weights: torch.Tensor) -> float:
    """Compute attention concentration (max attention weight averaged).

    Higher concentration = attention focuses on fewer tokens.
    """
    max_attn = attn_weights.max(dim=-1).values  # [batch, heads, seq_q]
    return max_attn.mean().item()


class AttentionCaptureHook:
    """Captures attention weights from self-attention layers.

    Attaches to the self_attn module and captures output_attentions.
    """

    def __init__(self):
        self.per_layer_entropy = {}  # layer_idx -> list of entropy values
        self.per_layer_concentration = {}
        self.handles = []

    def attach(self, model, model_info):
        """Attach hooks to capture attention weights."""
        self.detach()
        n_layers = model_info.get("n_layers", 32)

        for name, module in model.named_modules():
            for layer_idx in range(n_layers):
                # Match self_attn modules
                target = f"layers.{layer_idx}.self_attn"
                if name == f"model.layers.{layer_idx}.self_attn" or name.endswith(target):
                    if name.count(".self_attn") == 1 and name.endswith("self_attn"):
                        handle = module.register_forward_hook(
                            self._make_hook(layer_idx)
                        )
                        self.handles.append(handle)
                        break

    def _make_hook(self, layer_idx):
        def hook(module, inputs, outputs):
            # outputs is typically (hidden_states, attn_weights, past_key_value)
            # attn_weights is only present if output_attentions=True
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                attn_weights = outputs[1]
                if attn_weights is not None:
                    entropy = compute_attention_entropy(attn_weights)
                    concentration = compute_attention_concentration(attn_weights)
                    self.per_layer_entropy.setdefault(layer_idx, []).append(entropy)
                    self.per_layer_concentration.setdefault(layer_idx, []).append(concentration)
        return hook

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.per_layer_entropy.clear()
        self.per_layer_concentration.clear()

    def get_layer_stats(self):
        """Get aggregated per-layer attention statistics."""
        stats = {}
        for layer_idx in sorted(set(self.per_layer_entropy.keys()) | set(self.per_layer_concentration.keys())):
            entropies = self.per_layer_entropy.get(layer_idx, [])
            concentrations = self.per_layer_concentration.get(layer_idx, [])
            stats[layer_idx] = {
                "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
                "mean_concentration": sum(concentrations) / len(concentrations) if concentrations else 0.0,
                "sample_count": len(entropies),
            }
        return stats

    def get_importance_ranking(self):
        """Rank layers by attention entropy (higher = more important).

        Returns list of (layer_idx, entropy) sorted by entropy descending.
        """
        stats = self.get_layer_stats()
        ranking = [(idx, s["mean_entropy"]) for idx, s in stats.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.detach()


def parse_args():
    parser = argparse.ArgumentParser(description="Attention-based layer importance analysis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", default="custom")
    parser.add_argument("--max-new-tokens", type=int, default=1,
                        help="Tokens to generate (1 is enough for attention capture)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (1 recommended for attention capture)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model, attn_implementation="eager")

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")

    n_layers = model_info.get("n_layers", 32)

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "attention_analysis",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "prompts": args.prompts,
                "prompt_count": len(prompts),
                "n_layers": n_layers,
            },
            "pipeline_version": "1.0.0",
        },
    )

    for p in prompts:
        result.prompts.append(PromptResult(
            prompt_idx=p.index, prompt_text=p.text, category=p.category,
        ))

    # Capture attention weights
    hook = AttentionCaptureHook()
    hook.attach(model, model_info)

    print("\nCapturing attention weights...")
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt.text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
            )

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}]")

        # Store minimal info per prompt
        result.prompts[i].conditions["attention"] = ConditionResult(
            response="",
            refused=False,
            classifier="none",
            classifier_detail={"note": "attention capture only"},
            input_token_count=inputs["input_ids"].shape[1],
        )

    # Compute summary statistics (must be before detach, which clears data)
    layer_stats = hook.get_layer_stats()
    ranking = hook.get_importance_ranking()

    hook.detach()

    result.summary = {
        "per_layer_attention": {
            str(idx): stats for idx, stats in layer_stats.items()
        },
        "importance_ranking": [
            {"layer": idx, "entropy": ent} for idx, ent in ranking
        ],
        "top_5_layers": [idx for idx, _ in ranking[:5]],
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ATTENTION IMPORTANCE RANKING: {args.model}")
    print(f"{'='*60}")
    print(f"  {'Layer':>6} {'Entropy':>10} {'Concentration':>15}")
    print(f"  {'-'*6:>6} {'-'*10:>10} {'-'*15:>15}")

    for idx, entropy in ranking[:10]:
        conc = layer_stats[idx]["mean_concentration"]
        marker = " <--" if idx in [r[0] for r in ranking[:3]] else ""
        print(f"  {idx:>6} {entropy:>10.4f} {conc:>15.4f}{marker}")

    print(f"\n  Top-5 layers by attention entropy: {result.summary['top_5_layers']}")

    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
