#!/usr/bin/env python3
"""First-divergent-token analysis (Exp 6).

Post-hoc script: takes two result JSONs (FP16 and quantized), identifies
flipped prompts, and computes the first token position where outputs diverge.

If token_ids are present in the results, uses them directly.
If not, regenerates token-level outputs for flipped prompts only.

Outputs a CSV summary with divergence positions and categories.
"""

import argparse
import csv
import gc
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.results import load_results


def find_first_divergence(fp16_ids, quant_ids):
    """Find the first token position where two ID sequences differ.

    Returns (position, category) where category is:
      - "token_1": divergence at first token (immediate decision flip)
      - "early": divergence at tokens 2-10
      - "late": divergence at token 11+
      - "length_only": sequences agree but differ in length
    """
    min_len = min(len(fp16_ids), len(quant_ids))
    for i in range(min_len):
        if fp16_ids[i] != quant_ids[i]:
            pos = i + 1  # 1-indexed
            if pos == 1:
                cat = "token_1"
            elif pos <= 10:
                cat = "early"
            else:
                cat = "late"
            return pos, cat

    # Sequences agree up to min_len but differ in length
    if len(fp16_ids) != len(quant_ids):
        return min_len + 1, "length_only"

    return None, "identical"


def identify_flipped_prompts(fp16_result, quant_result, fp16_cond, quant_cond):
    """Identify prompts that flipped from refusal (FP16) to compliance (quantized).

    Returns list of (prompt_idx, prompt_text, category, fp16_token_ids, quant_token_ids,
                     fp16_response, quant_response).
    """
    fp16_map = {p.prompt_idx: p for p in fp16_result.prompts}
    quant_map = {p.prompt_idx: p for p in quant_result.prompts}

    flipped = []
    for idx in fp16_map:
        if idx not in quant_map:
            continue
        fp16_pr = fp16_map[idx]
        quant_pr = quant_map[idx]

        if fp16_cond not in fp16_pr.conditions or quant_cond not in quant_pr.conditions:
            continue

        fp16_cr = fp16_pr.conditions[fp16_cond]
        quant_cr = quant_pr.conditions[quant_cond]

        # Flip: refused in FP16, complied under quantization
        if fp16_cr.refused and not quant_cr.refused:
            fp16_ids = fp16_cr.token_ids or []
            quant_ids = quant_cr.token_ids or []
            flipped.append((
                idx,
                fp16_pr.prompt_text,
                fp16_pr.category,
                fp16_ids,
                quant_ids,
                fp16_cr.response,
                quant_cr.response,
            ))

    return flipped


def regenerate_token_ids(model_name, prompts, bits, max_new_tokens=256):
    """Regenerate token-level outputs for specific prompts under FP16 and quantized.

    Returns dict: {prompt_text: {"fp16": [token_ids], "quant": [token_ids]}}
    """
    import torch
    from core.model_loader import load_model
    from core.quantization import KVQuantizer, QuantConfig, PRESET_SECTION4
    from core.generation import generate_responses_enhanced

    print(f"\n[regenerate] Loading model: {model_name}")
    model, tokenizer, model_info = load_model(model_name)

    result_map = {}

    # FP16 pass
    print(f"[regenerate] FP16 pass on {len(prompts)} prompts...")
    fp16_outputs = generate_responses_enhanced(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens, batch_size=4,
        return_token_ids=True,
    )
    for prompt, gen_out in zip(prompts, fp16_outputs):
        result_map[prompt] = {"fp16": gen_out.token_ids or []}

    # Quantized pass
    print(f"[regenerate] {bits}-bit pass on {len(prompts)} prompts...")
    config = QuantConfig(
        bits=bits,
        symmetric=PRESET_SECTION4.symmetric,
        granularity=PRESET_SECTION4.granularity,
    )
    with KVQuantizer(model, config, model_info) as q:
        quant_outputs = generate_responses_enhanced(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens, batch_size=4,
            return_token_ids=True,
        )
    for prompt, gen_out in zip(prompts, quant_outputs):
        result_map[prompt]["quant"] = gen_out.token_ids or []

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[regenerate] Done, GPU freed")

    return result_map


def parse_args():
    parser = argparse.ArgumentParser(description="First-divergent-token analysis")
    parser.add_argument("--fp16-result", required=True,
                        help="Path to FP16 result JSON")
    parser.add_argument("--quant-result", required=True,
                        help="Path to quantized result JSON (can be same file as fp16)")
    parser.add_argument("--fp16-condition", default="16",
                        help="Condition key for FP16 in the result (default: '16')")
    parser.add_argument("--quant-condition", required=True,
                        help="Condition key for quantized (e.g., '4', '3')")
    parser.add_argument("--model", default=None,
                        help="Model name for regeneration if token_ids missing")
    parser.add_argument("--bits", type=int, default=None,
                        help="Bit-width for quantized regeneration (required with --model)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--examples", type=int, default=20,
                        help="Number of side-by-side examples to print")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading FP16 results: {args.fp16_result}")
    fp16_result = load_results(Path(args.fp16_result))
    print(f"Loading quantized results: {args.quant_result}")
    quant_result = load_results(Path(args.quant_result))

    print(f"FP16 condition: '{args.fp16_condition}', "
          f"Quant condition: '{args.quant_condition}'")

    flipped = identify_flipped_prompts(
        fp16_result, quant_result,
        args.fp16_condition, args.quant_condition,
    )
    print(f"Found {len(flipped)} flipped prompts")

    if not flipped:
        print("No flipped prompts found. Nothing to analyze.")
        return

    # Check if token_ids are available
    has_tokens = sum(1 for _, _, _, fp_ids, q_ids, _, _ in flipped
                     if fp_ids and q_ids)
    print(f"  {has_tokens}/{len(flipped)} have token_ids for both conditions")

    # Regenerate if needed and --model is provided
    if has_tokens < len(flipped) and args.model:
        if args.bits is None:
            print("[ERROR] --bits is required when using --model for regeneration")
            sys.exit(1)
        prompts_needing_regen = [
            text for _, text, _, fp_ids, q_ids, _, _ in flipped
            if not (fp_ids and q_ids)
        ]
        regen_map = regenerate_token_ids(
            args.model, prompts_needing_regen, args.bits,
            max_new_tokens=args.max_new_tokens,
        )
        # Patch token_ids into flipped list
        patched = []
        for idx, text, category, fp_ids, q_ids, fp_resp, q_resp in flipped:
            if not (fp_ids and q_ids) and text in regen_map:
                fp_ids = regen_map[text]["fp16"]
                q_ids = regen_map[text]["quant"]
            patched.append((idx, text, category, fp_ids, q_ids, fp_resp, q_resp))
        flipped = patched
        has_tokens = sum(1 for _, _, _, fp_ids, q_ids, _, _ in flipped
                         if fp_ids and q_ids)
        print(f"  After regeneration: {has_tokens}/{len(flipped)} have token_ids")
    elif has_tokens < len(flipped):
        print("\n[WARNING] Some prompts lack token_ids. Pass --model and --bits to regenerate.")

    # Compute divergence for each flipped prompt
    rows = []
    for idx, text, category, fp_ids, q_ids, fp_resp, q_resp in flipped:
        if fp_ids and q_ids:
            div_pos, div_cat = find_first_divergence(fp_ids, q_ids)
        else:
            div_pos, div_cat = None, "no_token_ids"

        rows.append({
            "prompt_idx": idx,
            "category": category,
            "divergence_position": div_pos,
            "divergence_category": div_cat,
            "fp16_response_len": len(fp_ids) if fp_ids else len(fp_resp.split()),
            "quant_response_len": len(q_ids) if q_ids else len(q_resp.split()),
            "fp16_first_20": " ".join(str(t) for t in fp_ids[:20]) if fp_ids else "",
            "quant_first_20": " ".join(str(t) for t in q_ids[:20]) if q_ids else "",
            "prompt_text": text[:100],
            "fp16_response_start": fp_resp[:150],
            "quant_response_start": q_resp[:150],
        })

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {output_path} ({len(rows)} rows)")

    # Print summary statistics
    cat_counts = Counter(r["divergence_category"] for r in rows)
    total_with_tokens = sum(v for k, v in cat_counts.items() if k != "no_token_ids")
    print(f"\nDivergence category distribution ({total_with_tokens} with token_ids):")
    for cat in ["token_1", "early", "late", "length_only", "identical", "no_token_ids"]:
        count = cat_counts.get(cat, 0)
        pct = f" ({count/len(rows):.0%})" if rows else ""
        print(f"  {cat:>15}: {count}{pct}")

    if total_with_tokens > 0:
        positions = [r["divergence_position"] for r in rows
                     if r["divergence_position"] is not None]
        if positions:
            import statistics
            print(f"\nDivergence position stats (n={len(positions)}):")
            print(f"  Mean:   {statistics.mean(positions):.1f}")
            print(f"  Median: {statistics.median(positions):.1f}")
            print(f"  Min:    {min(positions)}")
            print(f"  Max:    {max(positions)}")

    # Print examples
    n_examples = min(args.examples, len(rows))
    if n_examples > 0:
        print(f"\n{'='*80}")
        print(f"  SIDE-BY-SIDE EXAMPLES (first {n_examples})")
        print(f"{'='*80}")
        for r in rows[:n_examples]:
            print(f"\n--- Prompt {r['prompt_idx']} [{r['category']}] "
                  f"Diverge @ token {r['divergence_position']} ({r['divergence_category']})")
            print(f"  Prompt: {r['prompt_text']}...")
            print(f"  FP16:   {r['fp16_response_start']}...")
            print(f"  Quant:  {r['quant_response_start']}...")


if __name__ == "__main__":
    main()
