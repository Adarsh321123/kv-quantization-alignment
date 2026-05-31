#!/usr/bin/env python3
"""vLLM deployment validation: FP8 KV cache quantization in a production serving framework.

Demonstrates that KV cache quantization-induced alignment collapse occurs
in a production serving framework (vLLM), not just our simulated hooks.
"""

import os

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.prompts import load_prompts
from core.classifier import get_classifier, classify_stored_results
from core.metrics import compute_flip_rate
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    compute_summary, save_results, load_results,
)


CONDITIONS = [
    ("fp16", "auto"),
    ("fp8_e5m2", "fp8_e5m2"),
    ("fp8_e4m3", "fp8_e4m3"),
]


def create_llm(model_id, kv_cache_dtype, max_model_len=512):
    from vllm import LLM
    return LLM(
        model=model_id,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
    )


def generate_responses(llm, prompts, max_tokens=256):
    from vllm import SamplingParams
    sampling = SamplingParams(temperature=0, top_p=1.0, max_tokens=max_tokens)
    messages_list = [[{"role": "user", "content": p.text}] for p in prompts]
    outputs = llm.chat(messages_list, sampling_params=sampling)
    return [out.outputs[0].text for out in outputs]


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def print_results_table(result, prompts, conditions, vllm_version):
    fp16_labels = [pr.conditions["fp16"].refused for pr in result.prompts]
    gpu_name = result.metadata.get("gpu", "GPU")

    print(f"\n\nvLLM DEPLOYMENT VALIDATION ({gpu_name})")
    print("=" * 60)
    print(f"Model: {result.metadata['model']}")
    print(f"Prompts: {len(prompts)} ({result.metadata['prompt_set']})")
    print(f"vLLM version: {vllm_version}")
    print()
    print(f"{'Condition':<18}| {'Refusal Rate':>13} | {'ConditionalFlip':>16} | {'Flips':>8}")
    print(f"{'-'*18}+{'-'*15}+{'-'*18}+{'-'*9}")

    for cond_name, _ in conditions:
        labels = [pr.conditions[cond_name].refused for pr in result.prompts]
        refusal_rate = sum(labels) / len(labels) * 100

        if cond_name == "fp16":
            print(f"{cond_name:<18}| {refusal_rate:>12.1f}% | {'—':>16} | {'—':>8}")
        else:
            fm = compute_flip_rate(fp16_labels, labels, compute_ci=True)
            ci_lo = (fm.flip_rate_ci_lower or 0) * 100
            ci_hi = (fm.flip_rate_ci_upper or 0) * 100
            flip_str = f"{fm.flip_count}/{fm.baseline_refusal_count}"
            print(f"{cond_name:<18}| {refusal_rate:>12.1f}% | {fm.flip_rate*100:>5.1f}% [{ci_lo:.1f}-{ci_hi:.1f}] | {flip_str:>8}")

    return fp16_labels


def print_comparison(result, fp16_labels, sim_path: Optional[Path]):
    """Optionally compare vLLM FP8 results to a simulated-quantization run.

    Pass ``--compare-with PATH`` on the CLI to enable; if no path is given
    or the file is missing, just print the vLLM numbers.
    """
    if sim_path is None or not sim_path.exists():
        if sim_path is not None:
            print(f"\nSimulated comparison file not found at {sim_path}")
        return

    sim_result = load_results(sim_path)
    sim_8bit_flip = None
    if sim_result.summary:
        for cond_key, cond_data in sim_result.summary.items():
            if cond_key == "8" and isinstance(cond_data, dict):
                sim_8bit_flip = cond_data.get("flip_rate")
                break

    print(f"\nComparison with simulated quantization ({sim_path.name}):")
    if sim_8bit_flip is not None:
        print(f"  Simulated 8-bit ConditionalFlip: {sim_8bit_flip*100:.1f}%")
    else:
        print(f"  Simulated 8-bit ConditionalFlip: (not found)")

    for cond_name in ["fp8_e4m3", "fp8_e5m2"]:
        labels = [pr.conditions[cond_name].refused for pr in result.prompts]
        fm = compute_flip_rate(fp16_labels, labels)
        print(f"  vLLM {cond_name} ConditionalFlip:  {fm.flip_rate*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="vLLM FP8 KV cache deployment validation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompts", default="advbench")
    parser.add_argument("--max-prompts", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--output", default="results/vllm_deployment/vllm_qwen.json")
    parser.add_argument("--compare-with", default=None,
                        help="Optional path to a simulated-quantization sweep_advbench.json "
                             "to compare vLLM FP8 numbers against.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    try:
        import vllm
        vllm_version = vllm.__version__
    except AttributeError:
        vllm_version = "unknown"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

    result = ExperimentResult(metadata={
        "experiment": "vllm_deployment",
        "model": args.model,
        "prompt_set": args.prompts,
        "num_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "classifier": args.classifier,
        "vllm_version": vllm_version,
        "gpu": gpu_name,
    })

    for i, p in enumerate(prompts):
        result.prompts.append(PromptResult(
            prompt_idx=i,
            prompt_text=p.text,
            category=p.category,
        ))

    # Phase 1: vLLM generation for each KV cache dtype
    for cond_name, kv_dtype in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (kv_cache_dtype={kv_dtype})")
        print(f"{'='*60}")

        t0 = time.time()
        llm = create_llm(args.model, kv_dtype)
        print(f"Model loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        responses = generate_responses(llm, prompts, args.max_tokens)
        gen_time = time.time() - t0
        print(f"Generated {len(responses)} responses in {gen_time:.1f}s")

        for i, resp in enumerate(responses):
            result.prompts[i].conditions[cond_name] = ConditionResult(
                response=resp,
                refused=False,
                classifier="pending",
                generation_time_s=gen_time / len(prompts),
            )

        del llm
        free_gpu()
        print("GPU memory freed")

    # Phase 2: WildGuard classification
    print(f"\n{'='*60}")
    print("Phase 2: WildGuard classification")
    print(f"{'='*60}")

    yi_mode = "yi" in args.model.lower()
    classifier = get_classifier(args.classifier, yi_mode=yi_mode)
    classify_stored_results(result, classifier)
    classifier.unload()
    free_gpu()

    result.summary = compute_summary(result, baseline_condition="fp16")
    save_results(result, output_path)
    print(f"\nResults saved to {output_path}")

    fp16_labels = print_results_table(result, prompts, CONDITIONS, vllm_version)
    print_comparison(result, fp16_labels, Path(args.compare_with) if args.compare_with else None)
    print("\nDone!")


if __name__ == "__main__":
    main()
