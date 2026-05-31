#!/usr/bin/env python3
"""Speculative decoding experiment (X1).

Load target + draft models. For each bit-width, quantize target KV ->
run speculative decoding -> store responses. After all bit-widths,
delete both models, then classify all responses at once.
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.quantization import KVQuantizer, QuantConfig, PRESET_SECTION4, PRESET_SECTION5
from core.classifier import get_classifier, classify_stored_results
from core.prompts import load_prompts
from core.results import (
    ExperimentResult, PromptResult, ConditionResult,
    save_results, compute_summary,
)


def speculative_decode(
    target_model, target_tokenizer,
    draft_model, draft_tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    draft_length: int = 5,
) -> dict:
    """Simple speculative decoding: draft generates candidates, target verifies."""
    device = next(target_model.parameters()).device

    # Format prompt
    messages = [{"role": "user", "content": prompt}]
    formatted = target_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = target_tokenizer.encode(formatted, return_tensors="pt").to(device)

    generated_tokens = []
    accepted_tokens = 0
    total_draft_tokens = 0
    start_time = time.time()

    current_ids = input_ids
    for _ in range(max_new_tokens // draft_length + 1):
        if len(generated_tokens) >= max_new_tokens:
            break

        # Draft generates candidates
        with torch.no_grad():
            draft_out = draft_model.generate(
                current_ids,
                max_new_tokens=draft_length,
                do_sample=False,
                pad_token_id=draft_tokenizer.eos_token_id,
            )
        draft_tokens = draft_out[0][current_ids.shape[1]:]
        if len(draft_tokens) == 0:
            break
        total_draft_tokens += len(draft_tokens)

        # Target verifies
        candidate_ids = torch.cat([current_ids, draft_tokens.unsqueeze(0)], dim=1)
        with torch.no_grad():
            target_out = target_model(candidate_ids)
            target_logits = target_out.logits

        # Accept matching tokens
        n_accepted = 0
        for j in range(len(draft_tokens)):
            pos = current_ids.shape[1] + j - 1
            target_token = target_logits[0, pos].argmax().item()
            if target_token == draft_tokens[j].item():
                n_accepted += 1
                generated_tokens.append(target_token)
            else:
                generated_tokens.append(target_token)
                break

        accepted_tokens += n_accepted
        current_ids = torch.cat([
            input_ids,
            torch.tensor([generated_tokens], device=device)
        ], dim=1)

        # Check for EOS
        if generated_tokens and generated_tokens[-1] == target_tokenizer.eos_token_id:
            break

    elapsed = time.time() - start_time
    response = target_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    acceptance_rate = accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    tokens_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0.0

    return {
        "response": response,
        "acceptance_rate": acceptance_rate,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": len(generated_tokens),
        "elapsed": elapsed,
        "input_token_count": input_ids.shape[1],
        "token_ids": generated_tokens,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Speculative decoding experiment")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True,
                        help="HuggingFace ID for draft model")
    parser.add_argument("--bits", type=str, default="16,8,4,3")
    parser.add_argument("--draft-length", type=int, default=5)
    parser.add_argument("--quantizer-preset", default="section4",
                        choices=["section4", "section5"],
                        help="Quantizer preset (section4=per-token asym, section5=per-tensor sym)")
    parser.add_argument("--prompts", default="custom")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--classifier", default="wildguard")
    parser.add_argument("--classifier-device", default="auto",
                        help="Device for classifier model")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]

    print(f"Loading target model: {args.target_model}")
    target_model, target_tokenizer, model_info = load_model(args.target_model)

    print(f"Loading draft model: {args.draft_model}")
    draft_tokenizer = target_tokenizer  # Assume same tokenizer family
    from transformers import AutoModelForCausalLM
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    draft_model.eval()

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")

    yi_mode = "yi" in args.target_model.lower()

    result = ExperimentResult(
        metadata={
            "model": model_info.get("hf_id", args.target_model),
            "model_name": args.target_model,
            "draft_model": args.draft_model,
            "experiment": "spec_decoding",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bits": bits_list,
                "draft_length": args.draft_length,
                "classifier": args.classifier,
                "prompts": args.prompts,
                "prompt_count": len(prompts),
            },
            "pipeline_version": "1.0.0",
        },
    )

    for p in prompts:
        result.prompts.append(PromptResult(
            prompt_idx=p.index, prompt_text=p.text, category=p.category,
        ))

    # Resolve quantizer preset
    if args.quantizer_preset == "section4":
        base_config = PRESET_SECTION4
    else:
        base_config = PRESET_SECTION5

    # Phase 1: Generation

    for bits in bits_list:
        cond_key = str(bits)
        print(f"\n[{bits}-bit] Running speculative decoding...")

        if bits < 16:
            config = QuantConfig(
                bits=bits,
                symmetric=base_config.symmetric,
                granularity=base_config.granularity,
            )
            quantizer = KVQuantizer(target_model, config, model_info)
            quantizer.attach()

        for i, prompt in enumerate(prompts):
            sd_result = speculative_decode(
                target_model, target_tokenizer,
                draft_model, draft_tokenizer if draft_tokenizer else target_tokenizer,
                prompt.text,
                max_new_tokens=args.max_new_tokens,
                draft_length=args.draft_length,
            )

            result.prompts[i].conditions[cond_key] = ConditionResult(
                response=sd_result["response"],
                refused=False,  # placeholder, classified in Phase 2
                classifier="pending",
                classifier_detail={
                    "acceptance_rate": sd_result["acceptance_rate"],
                    "tokens_per_sec": sd_result["tokens_per_sec"],
                },
                generation_time_s=sd_result["elapsed"],
                input_token_count=sd_result["input_token_count"],
                token_ids=sd_result["token_ids"],
            )

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(prompts)}]")

        if bits < 16:
            quantizer.detach()

        print(f"  {bits}-bit: {len(prompts)} responses generated")

    # Phase 2: Classification
    print("\n[classify] Unloading generation models...")
    try:
        quantizer.model = None
    except NameError:
        pass
    del target_model
    del draft_model
    del target_tokenizer
    del draft_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Save raw results first (safety net)
    save_results(result, Path(args.output))
    print(f"  [saved raw] {args.output}")

    try:
        classifier = get_classifier(args.classifier, yi_mode=yi_mode,
                                    device=args.classifier_device)
        classify_stored_results(result, classifier)
        classifier.unload()
    except Exception as e:
        print(f"  [WARN] Classification failed: {e}")
        print(f"  Raw results (without classification) saved to {args.output}")

    result.summary = compute_summary(result, baseline_condition="16")
    save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
