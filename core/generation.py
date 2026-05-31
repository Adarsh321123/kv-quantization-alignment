"""Deterministic response generation with chat template support."""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
from torch import nn


@dataclass
class GenerationOutput:
    """Enhanced output from generation with metadata."""
    response: str
    token_ids: List[int] = field(default_factory=list)
    input_token_count: int = 0
    generation_time_s: float = 0.0


def generate_responses_enhanced(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
    apply_chat_template: bool = True,
    temperature: float = 0.0,
    do_sample: bool = False,
    seed: Optional[int] = None,
    return_token_ids: bool = True,
    top_p: Optional[float] = None,
) -> List[GenerationOutput]:
    """Enhanced generation that returns metadata alongside responses.

    Returns GenerationOutput with response text, token IDs, input token count,
    and generation wall-clock time.
    """
    device = next(model.parameters()).device

    # Format prompts
    if apply_chat_template:
        formatted = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            fmt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted.append(fmt)
    else:
        formatted = list(prompts)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    results = []
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i : i + batch_size]
        t0 = time.time()
        try:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            elapsed = time.time() - t0
            input_len = inputs.input_ids.shape[1]
            per_sample_time = elapsed / outputs.shape[0]

            for j in range(outputs.shape[0]):
                gen_tokens = outputs[j][input_len:]
                resp = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                # Count non-padding input tokens for this sample
                input_ids_j = inputs.input_ids[j]
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                input_count = (input_ids_j != pad_id).sum().item()

                token_ids_list = gen_tokens.tolist() if return_token_ids else []

                results.append(GenerationOutput(
                    response=resp,
                    token_ids=token_ids_list,
                    input_token_count=input_count,
                    generation_time_s=per_sample_time,
                ))

            del outputs, inputs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for text in batch:
                t1 = time.time()
                try:
                    inp = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                    ).to(device)
                    with torch.no_grad():
                        out = model.generate(**inp, **gen_kwargs)
                    single_time = time.time() - t1
                    gen_ids = out[0][inp.input_ids.shape[1]:]
                    resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results.append(GenerationOutput(
                        response=resp,
                        token_ids=gen_ids.tolist() if return_token_ids else [],
                        input_token_count=inp.input_ids.shape[1],
                        generation_time_s=single_time,
                    ))
                    del out, inp
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    results.append(GenerationOutput(
                        response="[OOM]",
                        generation_time_s=time.time() - t1,
                    ))

        if (i // batch_size + 1) % 10 == 0:
            torch.cuda.empty_cache()

    tokenizer.padding_side = old_padding_side
    torch.cuda.empty_cache()
    return results


