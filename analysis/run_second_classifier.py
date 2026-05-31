#!/usr/bin/env python3
"""Second-classifier reliability check via Llama-Guard-3 reclassification.

Loads existing result JSONs, draws a stratified sample of 200 (prompt, response)
pairs, reclassifies them with Llama-Guard-3-8B, and outputs a comparison CSV
with WildGuard vs LlamaGuard agreement.
"""

import argparse
import csv
import gc
import random
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.results import load_results
from core.classifier import get_classifier


def sample_for_validation(result_paths, conditions_per_file, n_total=200, seed=42):
    """Draw a stratified sample of (prompt, response) pairs from result files.

    Stratifies across:
      - Files (models / bit-widths)
      - Conditions (FP16, quantized)
      - WildGuard labels (refused, complied)

    Returns list of dicts with keys:
      source_file, model, condition, prompt_idx, prompt_text, response,
      wildguard_refused, wildguard_detail
    """
    rng = random.Random(seed)

    # Collect all candidate pairs grouped by (file, condition, label)
    buckets = defaultdict(list)
    for path_str, cond_keys in zip(result_paths, conditions_per_file):
        result = load_results(Path(path_str))
        model = result.metadata.get("model_name", Path(path_str).stem)

        for cond_key in cond_keys:
            for pr in result.prompts:
                if cond_key not in pr.conditions:
                    continue
                cr = pr.conditions[cond_key]
                if cr.classifier == "pending":
                    continue
                bucket_key = (path_str, model, cond_key, cr.refused)
                buckets[bucket_key].append({
                    "source_file": path_str,
                    "model": model,
                    "condition": cond_key,
                    "prompt_idx": pr.prompt_idx,
                    "prompt_text": pr.prompt_text,
                    "category": pr.category,
                    "response": cr.response,
                    "wildguard_refused": cr.refused,
                    "wildguard_method": cr.classifier,
                    "wildguard_detail": (cr.classifier_detail or {}).get(
                        "matched_pattern", ""),
                })

    # Allocate samples proportionally across buckets, ensuring representation
    total_candidates = sum(len(v) for v in buckets.values())
    if total_candidates == 0:
        raise ValueError("No classified pairs found in the provided results")

    samples = []
    remaining = n_total

    # First pass: allocate proportionally (min 1 per non-empty bucket)
    allocations = {}
    for key, candidates in buckets.items():
        alloc = max(1, int(n_total * len(candidates) / total_candidates))
        allocations[key] = min(alloc, len(candidates))

    # Adjust to hit n_total exactly
    total_alloc = sum(allocations.values())
    if total_alloc > n_total:
        # Trim largest buckets
        for key in sorted(allocations, key=lambda k: allocations[k], reverse=True):
            if total_alloc <= n_total:
                break
            trim = min(allocations[key] - 1, total_alloc - n_total)
            allocations[key] -= trim
            total_alloc -= trim
    elif total_alloc < n_total:
        # Add to largest available buckets
        for key in sorted(allocations, key=lambda k: len(buckets[k]), reverse=True):
            if total_alloc >= n_total:
                break
            extra = min(len(buckets[key]) - allocations[key], n_total - total_alloc)
            allocations[key] += extra
            total_alloc += extra

    for key, n in allocations.items():
        candidates = buckets[key]
        chosen = rng.sample(candidates, min(n, len(candidates)))
        samples.extend(chosen)

    rng.shuffle(samples)
    return samples[:n_total]


def parse_args():
    parser = argparse.ArgumentParser(description="Second classifier validation")
    parser.add_argument("--results", nargs="+", required=True,
                        help="Result JSON files to sample from")
    parser.add_argument("--conditions", nargs="+", required=True,
                        help="Condition keys per file (comma-separated per file, "
                             "e.g., '16,4' '16,3')")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Total samples to draw")
    parser.add_argument("--classifier", default="llamaguard",
                        choices=["llamaguard", "family_a"],
                        help="Second classifier to compare against WildGuard")
    parser.add_argument("--classifier-device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse per-file conditions
    conditions_per_file = [c.split(",") for c in args.conditions]
    if len(conditions_per_file) != len(args.results):
        raise ValueError(
            f"Number of --conditions ({len(conditions_per_file)}) must match "
            f"--results ({len(args.results)})"
        )

    print(f"Sampling {args.n_samples} pairs from {len(args.results)} result files...")
    samples = sample_for_validation(
        args.results, conditions_per_file,
        n_total=args.n_samples, seed=args.seed,
    )
    print(f"  Drew {len(samples)} samples")

    # Summarize sample distribution
    from collections import Counter
    model_counts = Counter(s["model"] for s in samples)
    cond_counts = Counter(s["condition"] for s in samples)
    label_counts = Counter(s["wildguard_refused"] for s in samples)
    print(f"  Models: {dict(model_counts)}")
    print(f"  Conditions: {dict(cond_counts)}")
    print(f"  WildGuard labels: refused={label_counts[True]}, complied={label_counts[False]}")

    # Classify with second classifier
    print(f"\nLoading second classifier: {args.classifier}")
    classifier = get_classifier(args.classifier, device=args.classifier_device)

    pairs = [(s["prompt_text"], s["response"]) for s in samples]
    results = classifier.classify_batch(pairs)
    classifier.unload()

    # Combine results
    rows = []
    agree_count = 0
    for sample, cls_result in zip(samples, results):
        agreed = sample["wildguard_refused"] == cls_result.refused
        if agreed:
            agree_count += 1
        rows.append({
            "prompt_idx": sample["prompt_idx"],
            "model": sample["model"],
            "condition": sample["condition"],
            "category": sample["category"],
            "wildguard_refused": sample["wildguard_refused"],
            "wildguard_detail": sample["wildguard_detail"][:100],
            f"{args.classifier}_refused": cls_result.refused,
            f"{args.classifier}_detail": (cls_result.matched_pattern or "")[:100],
            "agree": agreed,
            "prompt_text": sample["prompt_text"][:100],
            "response_start": sample["response"][:150],
        })

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {output_path} ({len(rows)} rows)")

    # Agreement statistics
    total = len(rows)
    print(f"\n{'='*60}")
    print(f"  CLASSIFIER AGREEMENT: WildGuard vs {args.classifier}")
    print(f"{'='*60}")
    print(f"  Total pairs:   {total}")
    print(f"  Agreement:     {agree_count}/{total} ({agree_count/total:.1%})")
    print(f"  Disagreement:  {total - agree_count}/{total} ({(total-agree_count)/total:.1%})")

    # Breakdown by WildGuard label
    for wg_label in [True, False]:
        label_rows = [r for r in rows if r["wildguard_refused"] == wg_label]
        if not label_rows:
            continue
        label_agree = sum(1 for r in label_rows if r["agree"])
        label_name = "refused" if wg_label else "complied"
        print(f"\n  WildGuard={label_name} (n={len(label_rows)}):")
        print(f"    {args.classifier} agrees: {label_agree}/{len(label_rows)} "
              f"({label_agree/len(label_rows):.1%})")

    # Cohen's kappa
    a = sum(1 for r in rows if r["wildguard_refused"] and r[f"{args.classifier}_refused"])
    b = sum(1 for r in rows if r["wildguard_refused"] and not r[f"{args.classifier}_refused"])
    c = sum(1 for r in rows if not r["wildguard_refused"] and r[f"{args.classifier}_refused"])
    d = sum(1 for r in rows if not r["wildguard_refused"] and not r[f"{args.classifier}_refused"])

    p_o = (a + d) / total if total > 0 else 0
    p_wg_ref = (a + b) / total
    p_lg_ref = (a + c) / total
    p_e = p_wg_ref * p_lg_ref + (1 - p_wg_ref) * (1 - p_lg_ref)

    if p_e < 1.0:
        kappa = (p_o - p_e) / (1 - p_e)
    else:
        kappa = 1.0

    print(f"\n  Cohen's kappa: {kappa:.3f}")
    if kappa >= 0.81:
        print(f"  Interpretation: Almost perfect agreement")
    elif kappa >= 0.61:
        print(f"  Interpretation: Substantial agreement")
    elif kappa >= 0.41:
        print(f"  Interpretation: Moderate agreement")
    else:
        print(f"  Interpretation: Fair or worse agreement")

    # Print disagreement examples
    disagreements = [r for r in rows if not r["agree"]]
    if disagreements:
        n_show = min(10, len(disagreements))
        print(f"\n  DISAGREEMENT EXAMPLES (first {n_show}):")
        for r in disagreements[:n_show]:
            wg = "refused" if r["wildguard_refused"] else "complied"
            lg = "refused" if r[f"{args.classifier}_refused"] else "complied"
            print(f"    [{r['model']} {r['condition']}] WG={wg}, LG={lg}")
            print(f"      Prompt: {r['prompt_text']}...")
            print(f"      Response: {r['response_start']}...")


if __name__ == "__main__":
    main()
