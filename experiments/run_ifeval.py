#!/usr/bin/env python3
"""IFEval benchmark (B2).

Load IFEval prompts -> for each bit-width, generate with quantizer ->
score with constraint checkers -> compute ConditionalConstraintFlip.
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import KVQuantizer, QuantConfig
from core.generation import generate_responses_enhanced
from core.prompts import load_ifeval
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def _log(msg):
    """Print to stderr for immediate visibility (stderr is line-buffered)."""
    print(msg, file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="IFEval benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=str, default="16,8,6,5,4,3,2")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]

    _log(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)

    prompts = load_ifeval()
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    _log(f"  Loaded {len(prompts)} IFEval prompts")

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "ifeval",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": bits_list,
                "max_new_tokens": args.max_new_tokens,
                "prompt_count": len(prompts),
            },
            "pipeline_version": "1.0.0",
        },
    )

    for p in prompts:
        result.prompts.append(PromptResult(
            prompt_idx=p.index, prompt_text=p.text, category=p.category,
        ))

    out_path = Path(args.output)
    completed_bits = set()
    if out_path.exists():
        import json
        try:
            with open(out_path) as f:
                prev = json.load(f)
            if prev.get("prompts") and prev["prompts"][0].get("conditions"):
                completed_bits = set(prev["prompts"][0]["conditions"].keys())
                _log(f"  Resuming: found existing results for bits {completed_bits}")
                from core.results import load_results
                result = load_results(out_path)
        except Exception as e:
            _log(f"  Warning: could not load previous results ({e}), starting fresh")

    total_bits = len(bits_list)
    for bit_idx, bits in enumerate(bits_list):
        cond_key = str(bits)
        if cond_key in completed_bits:
            _log(f"\n[{bits}-bit] ({bit_idx+1}/{total_bits}) Already completed, skipping.")
            continue

        _log(f"\n[{bits}-bit] ({bit_idx+1}/{total_bits}) Generating IFEval responses...")
        bit_t0 = time.time()

        if bits >= 16:
            gen_outputs = generate_responses_enhanced(
                model, tokenizer,
                [p.text for p in prompts],
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                return_token_ids=False,
            )
            mean_mse = None
        else:
            config = QuantConfig(bits=bits, symmetric=False, granularity="per_token")
            with KVQuantizer(model, config, model_info) as q:
                gen_outputs = generate_responses_enhanced(
                    model, tokenizer,
                    [p.text for p in prompts],
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    return_token_ids=False,
                )
                mean_mse = q.get_mean_mse()

        for i, (prompt, gen_out) in enumerate(zip(prompts, gen_outputs)):
            result.prompts[i].conditions[cond_key] = ConditionResult(
                response=gen_out.response,
                refused=False,
                classifier="none",
                classifier_detail={"note": "IFEval uses constraint-based scoring"},
                kv_mse=mean_mse,
                generation_time_s=gen_out.generation_time_s,
                input_token_count=gen_out.input_token_count,
            )

        bit_elapsed = time.time() - bit_t0
        _log(f"  Generated {len(gen_outputs)} responses in {bit_elapsed:.0f}s")
        save_results(result, out_path)
        _log(f"  [checkpoint] Saved after {bits}-bit ({bit_idx+1}/{total_bits} done)")

    result.summary = compute_summary(result, baseline_condition=str(bits_list[0]))
    save_results(result, out_path)
    _log("  Note: Run IFEval scoring separately on the generated responses.")


if __name__ == "__main__":
    main()
