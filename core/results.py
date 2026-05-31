"""Unified result schema and I/O. ALL experiments output this format."""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from core.metrics import wilson_ci


@dataclass
class ConditionResult:
    """Result for one prompt under one experimental condition."""
    response: str
    refused: bool
    classifier: str
    classifier_detail: Optional[dict] = None
    kv_mse: Optional[float] = None
    # Enhanced fields (all optional, None by default)
    generation_time_s: Optional[float] = None
    input_token_count: Optional[int] = None
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[List[Any]]] = None  # top-k logprobs per token
    kv_stats_per_layer: Optional[Dict[int, Dict]] = None


@dataclass
class PromptResult:
    """Result for one prompt across all conditions."""
    prompt_idx: int
    prompt_text: str
    category: str
    conditions: Dict[str, ConditionResult] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Top-level result container for any experiment."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompts: List[PromptResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def _serialize(obj):
    """Custom serializer for dataclass nesting."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results(result: ExperimentResult, output_path: Path) -> None:
    """Save to JSON with full schema. Uses atomic write (temp + rename)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(result)
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(output_path)
    print(f"[saved] {output_path}")


def load_results(path: Path) -> ExperimentResult:
    """Load results from unified JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    # Known ConditionResult fields for forward-compatible loading
    _cond_fields = {f.name for f in ConditionResult.__dataclass_fields__.values()}

    prompts = []
    for p in data.get("prompts", []):
        conditions = {}
        for cond_key, cond_data in p.get("conditions", {}).items():
            # Filter to known fields (handles old JSONs with fewer fields,
            # or future JSONs with extra fields)
            filtered = {k: v for k, v in cond_data.items() if k in _cond_fields}
            conditions[cond_key] = ConditionResult(**filtered)
        prompts.append(PromptResult(
            prompt_idx=p["prompt_idx"],
            prompt_text=p["prompt_text"],
            category=p["category"],
            conditions=conditions,
        ))

    return ExperimentResult(
        metadata=data.get("metadata", {}),
        prompts=prompts,
        summary=data.get("summary", {}),
    )


def compute_summary(
    result: ExperimentResult,
    baseline_condition: str = "16",
    compute_ci: bool = True,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute summary statistics from prompt results.

    Computes per-condition refusal rates, flip rates (vs baseline),
    and category breakdowns. Optionally adds Wilson 95% CIs.
    """
    # Collect all condition keys
    all_conditions = set()
    for p in result.prompts:
        all_conditions.update(p.conditions.keys())

    # Get baseline labels
    baseline_labels = {}
    for p in result.prompts:
        if baseline_condition in p.conditions:
            baseline_labels[p.prompt_idx] = p.conditions[baseline_condition].refused

    summary = {}
    for cond in sorted(all_conditions):
        cond_prompts = [p for p in result.prompts if cond in p.conditions]
        if not cond_prompts:
            continue

        refused_count = sum(1 for p in cond_prompts if p.conditions[cond].refused)
        total = len(cond_prompts)
        refusal_rate = refused_count / total if total > 0 else 0.0

        entry = {
            "refusal_rate": refusal_rate,
            "refusal_count": refused_count,
            "total_prompts": total,
        }

        # Flip rate vs baseline
        if cond != baseline_condition and baseline_labels:
            flips = 0
            baseline_refusals = 0
            for p in cond_prompts:
                bl = baseline_labels.get(p.prompt_idx)
                if bl is not None and bl:  # Baseline refused
                    baseline_refusals += 1
                    if not p.conditions[cond].refused:  # Quantized complied
                        flips += 1
            if baseline_refusals > 0:
                entry["flip_rate"] = flips / baseline_refusals
                entry["flip_count"] = flips
                entry["conditional_flip"] = flips / baseline_refusals
                if compute_ci:
                    ci_lo, ci_hi = wilson_ci(flips, baseline_refusals, confidence)
                    entry["flip_rate_ci"] = {
                        "lower": ci_lo,
                        "upper": ci_hi,
                        "method": "wilson",
                        "confidence": confidence,
                    }

        # Mean KV MSE
        mse_vals = [
            p.conditions[cond].kv_mse
            for p in cond_prompts
            if p.conditions[cond].kv_mse is not None
        ]
        if mse_vals:
            entry["mean_kv_mse"] = sum(mse_vals) / len(mse_vals)

        # Category breakdown
        categories = set(p.category for p in cond_prompts)
        if len(categories) > 1:
            breakdown = {}
            for cat in sorted(categories):
                cat_prompts = [p for p in cond_prompts if p.category == cat]
                cat_refused = sum(1 for p in cat_prompts if p.conditions[cond].refused)
                cat_entry = {
                    "refused": cat_refused,
                    "total": len(cat_prompts),
                    "rate": cat_refused / len(cat_prompts) if cat_prompts else 0.0,
                }
                # Category flips
                if cond != baseline_condition and baseline_labels:
                    cat_flips = 0
                    cat_bl_ref = 0
                    for p in cat_prompts:
                        bl = baseline_labels.get(p.prompt_idx)
                        if bl is not None and bl:
                            cat_bl_ref += 1
                            if not p.conditions[cond].refused:
                                cat_flips += 1
                    if cat_bl_ref > 0:
                        cat_entry["flips"] = cat_flips
                breakdown[cat] = cat_entry
            entry["category_breakdown"] = breakdown

        summary[cond] = entry

    return summary
