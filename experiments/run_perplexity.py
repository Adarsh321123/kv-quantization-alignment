#!/usr/bin/env python3
"""Perplexity measurement (S3).

For each bit-width, attach quantizer -> compute WikiText-103 PPL.
No prompt generation needed.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import KVQuantizer, QuantConfig
from core.metrics import compute_perplexity
from core.results import ExperimentResult, save_results


def _log(msg):
    """Print to stderr for immediate visibility (stderr is line-buffered)."""
    print(msg, file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Perplexity measurement")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=str, default="16,8,6,5,4,3,2",
                        help="Comma-separated bit-widths")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]
    out_path = Path(args.output)

    _log(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.model),
            "model_name": args.model,
            "experiment": "perplexity",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_length": args.max_length,
                "stride": args.stride,
                "bits": bits_list,
            },
        },
        prompts=[],
        summary={},
    )

    completed_bits = set()
    if out_path.exists():
        try:
            with open(out_path) as f:
                prev = json.load(f)
            if prev.get("summary"):
                completed_bits = set(prev["summary"].keys())
                result.summary = prev["summary"]
                _log(f"  Resuming: found existing results for bits {completed_bits}")
        except Exception as e:
            _log(f"  Warning: could not load previous results ({e}), starting fresh")

    total_bits = len(bits_list)
    for bit_idx, bits in enumerate(bits_list):
        cond_key = str(bits)
        if cond_key in completed_bits:
            _log(f"\n[{bits}-bit] ({bit_idx+1}/{total_bits}) Already completed, skipping.")
            continue

        _log(f"\n[{bits}-bit] ({bit_idx+1}/{total_bits}) Computing perplexity...")
        bit_t0 = time.time()

        if bits >= 16:
            ppl = compute_perplexity(
                model, tokenizer,
                max_length=args.max_length,
                stride=args.stride,
            )
            mse = None
        else:
            config = QuantConfig(bits=bits, symmetric=False, granularity="per_token")
            with KVQuantizer(model, config, model_info) as q:
                ppl = compute_perplexity(
                    model, tokenizer,
                    max_length=args.max_length,
                    stride=args.stride,
                )
                mse = q.get_mean_mse()

        result.summary[str(bits)] = {
            "perplexity": ppl,
            "kv_mse": mse,
        }
        bit_elapsed = time.time() - bit_t0
        _log(f"  {bits}-bit PPL: {ppl:.4f}" + (f", MSE: {mse:.6f}" if mse else "")
             + f" ({bit_elapsed:.0f}s)")
        save_results(result, out_path)
        _log(f"  [checkpoint] Saved after {bits}-bit ({bit_idx+1}/{total_bits} done)")

    # Summary table
    _log(f"\n{'='*40}")
    _log(f"{'Bits':>5} | {'PPL':>10} | {'MSE':>10}")
    _log("-" * 30)
    for bits in bits_list:
        r = result.summary[str(bits)]
        mse_s = f"{r['kv_mse']:.6f}" if r["kv_mse"] is not None else "  -"
        _log(f"{bits:>5} | {r['perplexity']:>10.4f} | {mse_s:>10}")

    save_results(result, out_path)


if __name__ == "__main__":
    main()
