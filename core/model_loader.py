"""Model and tokenizer loading with model-specific handling."""

from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model registry: canonical name -> HuggingFace ID + config
MODEL_REGISTRY = {
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "hidden_size": 3584,
        "fused_qkv": False,
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "n_layers": 32,
        "hidden_size": 4096,
        "fused_qkv": False,
    },
    "deepseek": {
        "hf_id": "deepseek-ai/deepseek-llm-7b-chat",
        "n_layers": 30,
        "hidden_size": 4096,
        "fused_qkv": False,
    },
    "yi": {
        "hf_id": "01-ai/Yi-1.5-9B-Chat",
        "n_layers": 48,
        "hidden_size": 4096,
        "fused_qkv": False,
    },
    "llama": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "hidden_size": 4096,
        "fused_qkv": False,
    },
    "gemma2": {
        "hf_id": "google/gemma-2-9b-it",
        "n_layers": 42,
        "hidden_size": 3584,
        "fused_qkv": False,
    },
    "mistral-small": {
        "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "n_layers": 40,
        "hidden_size": 6144,
        "fused_qkv": False,
    },
    "phi35": {
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "n_layers": 32,
        "hidden_size": 3072,
        "fused_qkv": True,
    },
    "qwen-72b": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "n_layers": 80,
        "hidden_size": 8192,
        "fused_qkv": False,
    },
    "yi-34b": {
        "hf_id": "01-ai/Yi-1.5-34B-Chat",
        "n_layers": 60,
        "hidden_size": 7168,
        "fused_qkv": False,
    },
    "mixtral": {
        "hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "n_layers": 32,
        "hidden_size": 4096,
        "fused_qkv": False,
    },
    "olmo2": {
        "hf_id": "allenai/OLMo-2-1124-7B-Instruct",
        "n_layers": 32,
        "hidden_size": 4096,
        "fused_qkv": False,
    },
}


def get_model_info(model_name: str) -> dict:
    """Look up model in registry. Accepts canonical name or HF ID."""
    # Direct canonical name lookup
    if model_name in MODEL_REGISTRY:
        return dict(MODEL_REGISTRY[model_name])

    # Try matching by HF ID
    for name, info in MODEL_REGISTRY.items():
        if info["hf_id"] == model_name:
            return dict(info)

    # Try partial match on HF ID
    for name, info in MODEL_REGISTRY.items():
        if model_name.lower() in info["hf_id"].lower():
            return dict(info)

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Known models: {list(MODEL_REGISTRY.keys())}"
    )


def resolve_hf_id(model_name: str) -> str:
    """Resolve a canonical name to a HuggingFace model ID."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]["hf_id"]
    # If it looks like an HF ID already (contains /), use it directly
    if "/" in model_name:
        return model_name
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Known models: {list(MODEL_REGISTRY.keys())}"
    )


def load_model(
    model_name: str,
    dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """Load model and tokenizer. Returns (model, tokenizer, model_info).

    model_info dict contains: hf_id, n_layers, hidden_size, fused_qkv.

    Special handling:
    - Phi-3.5: trust_remote_code=False (built-in Phi-3 impl works)
    - 72B: device_map="auto" for multi-GPU sharding
    - All others: trust_remote_code=True
    """
    hf_id = resolve_hf_id(model_name)
    try:
        model_info = get_model_info(model_name)
    except ValueError:
        if "/" not in model_name:
            raise
        # For unseen HF IDs, populate required fields after model load.
        model_info = {"hf_id": hf_id}

    # Resolve dtype
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Phi-3.5-style fused QKV models can require standard code paths.
    # Unknown HF IDs default to trust_remote_code=True and we infer details later.
    trust_remote_code = not model_info.get("fused_qkv", False)

    # Build model kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    model.eval()

    # Ensure required metadata exists for quantization hooks.
    if "n_layers" not in model_info:
        n_layers = getattr(model.config, "num_hidden_layers", None)
        if n_layers is None and hasattr(model, "model") and hasattr(model.model, "layers"):
            n_layers = len(model.model.layers)
        if n_layers is None:
            raise ValueError(
                f"Could not infer number of layers for model: {hf_id}. "
                "Add this model to MODEL_REGISTRY in core/model_loader.py."
            )
        model_info["n_layers"] = int(n_layers)

    if "hidden_size" not in model_info:
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not infer hidden size for model: {hf_id}. "
                "Add this model to MODEL_REGISTRY in core/model_loader.py."
            )
        model_info["hidden_size"] = int(hidden_size)

    if "fused_qkv" not in model_info:
        model_info["fused_qkv"] = any(
            name.endswith("qkv_proj") for name, _ in model.named_modules()
        )

    model_info.setdefault("hf_id", hf_id)

    return model, tokenizer, model_info
