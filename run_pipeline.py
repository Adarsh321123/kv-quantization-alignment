#!/usr/bin/env python3
"""Unified pipeline runner.

Produces every result needed for the paper in a single invocation.
Supports --all for full 8-model runs, --resume for crash recovery,
--dry-run for execution plan preview, and --gpu for multi-GPU parallelism.

Usage:
    # Full pipeline for all 8 standard models
    python run_pipeline.py --all --dry-run

    # Run everything for a specific model
    python run_pipeline.py --model qwen7b --all --dry-run

    # Run specific experiment groups
    python run_pipeline.py --all --experiments sweep mechanistic --dry-run

    # Resume after crash
    python run_pipeline.py --all --resume

    # Quick test with 5 prompts
    python run_pipeline.py --model qwen7b --all --max-prompts 5 --dry-run

    # Parallel runs on 2 GPUs
    python run_pipeline.py --model phi35 qwen7b mistral7b --gpu 0 --output-dir results/run_final/
    python run_pipeline.py --model deepseek7b llama8b yi9b --gpu 1 --output-dir results/run_final/
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Ignore SIGHUP so pipeline survives terminal closure (belt-and-suspenders with nohup)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

PYTHON = os.environ.get("PIPELINE_PYTHON", sys.executable)

# ---------------------------------------------------------------------------
# Model registry — every model with its experiment config
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "microsoft/Phi-3.5-mini-instruct": {
        "short_name": "phi35",
        "layers": 32,
        "vram_gb": 8,  # bf16 weight; assumes two-phase (generation model and WildGuard never coexist)
        "size_order": 0,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
            "determinism": {"bits": 3},
            "protection_sweep": {"base_bits": 3},
        },
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "short_name": "qwen7b",
        "layers": 28,
        "vram_gb": 14,
        "size_order": 1,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
            "attention_analysis": {},
            "determinism": {"bits": 8},
            "spec_decoding": {"draft_model": "Qwen/Qwen2.5-0.5B-Instruct"},
            "protection_sweep": {"base_bits": 4},
            "mitigation": [
                {"bits": 4, "strategies": "per_tensor,per_channel,group_64,fp16_critical,perchannel_critical_pertensor_rest,perchannel_critical_group64_rest"},
                {"bits": 3, "strategies": "per_tensor,per_channel,group_64,fp16_critical,perchannel_critical_pertensor_rest,perchannel_critical_group64_rest"},
            ],
            "mitigation_8bit": {"base_bits": 4, "protect_bits": 8},
        },
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "short_name": "mistral7b",
        "layers": 32,
        "vram_gb": 14,
        "size_order": 2,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
            "determinism": {"bits": 3},
            "mitigation": [
                {"bits": 3, "strategies": "per_tensor,per_channel,group_64,fp16_critical,perchannel_critical_pertensor_rest,perchannel_critical_group64_rest"},
            ],
            "mitigation_8bit": {"base_bits": 3, "protect_bits": 8},
        },
    },
    "deepseek-ai/deepseek-llm-7b-chat": {
        "short_name": "deepseek7b",
        "layers": 30,
        "vram_gb": 14,
        "size_order": 3,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
            "determinism": {"bits": 4},
            "mitigation": [
                {"bits": 3, "strategies": "per_tensor,per_channel,group_64,fp16_critical,perchannel_critical_pertensor_rest,perchannel_critical_group64_rest"},
            ],
            "mitigation_8bit": {"base_bits": 3, "protect_bits": 8},
        },
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "short_name": "llama8b",
        "layers": 32,
        "vram_gb": 16,
        "size_order": 4,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
            "attention_analysis": {},
            "determinism": {"bits": 4},
            "protection_sweep": {"base_bits": 4},
        },
    },
    "01-ai/Yi-1.5-9B-Chat": {
        "short_name": "yi9b",
        "layers": 48,
        "vram_gb": 18,
        "size_order": 5,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
            "determinism": {"bits": 8},
        },
    },
    "google/gemma-2-9b-it": {
        "short_name": "gemma9b",
        "layers": 42,
        "vram_gb": 18,
        "size_order": 6,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
        },
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "short_name": "msmall24b",
        "layers": 40,
        "vram_gb": 48,
        "size_order": 7,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
            "determinism": {"bits": 3},
            "protection_sweep": {"base_bits": 3},
        },
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "short_name": "qwen72b",
        "layers": 80,
        "vram_gb": 144,
        "device_map": "auto",
        "size_order": 99,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
        },
    },
    "01-ai/Yi-1.5-34B-Chat": {
        "short_name": "yi34b",
        "layers": 60,
        "vram_gb": 68,
        "device_map": "auto",
        "size_order": 98,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
            "determinism": {"bits": 8},
        },
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "short_name": "mixtral",
        "layers": 32,
        "vram_gb": 90,
        "device_map": "auto",
        "size_order": 99,
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "channel_ablation": {},
            "activation_analysis": {},
        },
    },
    "allenai/OLMo-2-1124-7B-Instruct": {
        "short_name": "olmo2",
        "layers": 32,
        "vram_gb": 14,
        "size_order": 97,  # held-out model, excluded from --all
        "experiments": {
            "sweep": {"benchmarks": ["custom", "advbench", "harmbench", "xstest"]},
            "perplexity": {},
            "ifeval": {},
            "layer_ablation": {},
            "cumulative": {},
            "channel_ablation": {},
            "group64": {},
            "activation_analysis": {},
        },
    },
}

# Short name -> HF ID lookup
SHORT_NAMES = {v["short_name"]: k for k, v in MODEL_REGISTRY.items()}

# The 8 standard models (excludes large models like 72B, 34B)
STANDARD_MODELS = [
    hf_id for hf_id, info in MODEL_REGISTRY.items()
    if info["size_order"] < 90
]

# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------

EXPERIMENT_GROUPS = {
    "sweep": ["sweep"],
    "quality": ["perplexity", "ifeval"],
    "mechanistic": ["layer_ablation", "cumulative", "channel_ablation", "group64",
                     "activation_analysis", "attention_analysis"],
    "mitigation_group": ["protection_sweep", "mitigation", "mitigation_8bit"],
    "validation": ["determinism", "spec_decoding"],
    "section4": ["sweep", "perplexity", "ifeval", "determinism", "spec_decoding"],
    "section5": ["layer_ablation", "cumulative", "channel_ablation", "group64",
                 "activation_analysis", "attention_analysis"],
    "section6": ["protection_sweep", "mitigation", "mitigation_8bit"],
}

# All possible experiment names (union of everything)
ALL_EXPERIMENTS = [
    "sweep", "perplexity", "ifeval", "determinism", "spec_decoding",
    "layer_ablation", "cumulative", "channel_ablation", "group64",
    "activation_analysis", "attention_analysis",
    "protection_sweep", "mitigation", "mitigation_8bit",
]

# ---------------------------------------------------------------------------
# Default critical layers (fallback when layer_ablation hasn't run)
# ---------------------------------------------------------------------------

DEFAULT_CRITICAL = {
    "phi35": [2, 11, 14],
    "qwen7b": [0, 27, 1, 8],
    "mistral7b": [0],
    "deepseek7b": [17, 1],
    "llama8b": [2, 3, 0, 1, 9],
    "yi9b": [16],
    "gemma9b": [1],
    "msmall24b": [14, 1, 2],
    "qwen72b": [0],
    "yi34b": [0],
    "mixtral": [0],
    "olmo2": [0],
}

# ---------------------------------------------------------------------------
# Script paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent / "experiments"

SCRIPTS = {
    "sweep": SCRIPT_DIR / "run_sweep.py",
    "perplexity": SCRIPT_DIR / "run_perplexity.py",
    "ifeval": SCRIPT_DIR / "run_ifeval.py",
    "determinism": SCRIPT_DIR / "run_determinism.py",
    "spec_decoding": SCRIPT_DIR / "run_spec_decoding.py",
    "layer_ablation": SCRIPT_DIR / "run_layer_ablation.py",
    "cumulative": SCRIPT_DIR / "run_cumulative.py",
    "channel_ablation": SCRIPT_DIR / "run_channel_ablation.py",
    "group64": SCRIPT_DIR / "run_group64.py",
    "activation_analysis": SCRIPT_DIR / "run_activation_analysis.py",
    "attention_analysis": SCRIPT_DIR / "run_attention_analysis.py",
    "protection_sweep": SCRIPT_DIR / "run_protection_sweep.py",
    "mitigation": SCRIPT_DIR / "run_mitigation.py",
    "mitigation_8bit": SCRIPT_DIR / "run_protection_sweep.py",  # reuses protection_sweep with --protect-bits 8
    # Supplementary scripts (not auto-invoked by --all; documented for
    # standalone reproduction of appendix experiments).
    "kivi": SCRIPT_DIR / "run_kivi.py",
    "system_prompt": SCRIPT_DIR / "run_system_prompt.py",
    "temperature": SCRIPT_DIR / "run_temperature.py",
    "naive_baselines": SCRIPT_DIR / "run_naive_baselines.py",
    "pcr_prediction": SCRIPT_DIR / "run_pcr_prediction.py",
    "vllm_deployment": SCRIPT_DIR / "run_vllm_deployment.py",
}

# ---------------------------------------------------------------------------
# VRAM detection
# ---------------------------------------------------------------------------

def get_vram_info() -> dict:
    """Detect VRAM across all visible GPUs.

    Returns dict with:
        max_single_gpu_gb: Largest single GPU's VRAM (for single-GPU models).
        total_gb: Sum across all GPUs (for device_map="auto" models).
        gpu_count: Number of GPUs detected.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"max_single_gpu_gb": 0.0, "total_gb": 0.0, "gpu_count": 0}
        per_gpu = []
        for i in range(torch.cuda.device_count()):
            gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            per_gpu.append(gb)
        return {
            "max_single_gpu_gb": max(per_gpu) if per_gpu else 0.0,
            "total_gb": sum(per_gpu),
            "gpu_count": len(per_gpu),
        }
    except (RuntimeError, AttributeError) as e:
        warnings.warn(f"VRAM detection failed: {e}")
        return {"max_single_gpu_gb": 0.0, "total_gb": 0.0, "gpu_count": 0}


def filter_models_by_vram(model_ids: list, vram_info: dict, explicit: bool) -> list:
    """Filter models that fit in available VRAM.

    Uses max single GPU VRAM for standard models, total VRAM for models
    with device_map="auto" (multi-GPU).

    Args:
        model_ids: List of HF IDs to check.
        vram_info: Dict from get_vram_info().
        explicit: If True (user passed --model), warn but don't filter.

    Returns:
        Filtered list of HF IDs.
    """
    max_single = vram_info["max_single_gpu_gb"]
    total = vram_info["total_gb"]

    if max_single <= 0:
        return model_ids  # Can't detect — run everything, let it fail naturally

    kept = []
    skipped = []
    for hf_id in model_ids:
        info = MODEL_REGISTRY[hf_id]
        required = info.get("vram_gb", 0)
        # Models with device_map="auto" can spread across GPUs
        available = total if info.get("device_map") == "auto" else max_single
        if required > available:
            if explicit:
                print(f"  WARNING: {info['short_name']} requires ~{required}GB, "
                      f"available {available:.0f}GB — running anyway (explicit --model)")
                kept.append(hf_id)
            else:
                skipped.append((info["short_name"], required))
        else:
            kept.append(hf_id)

    if skipped:
        print(f"\n  VRAM: {max_single:.0f}GB per GPU, {total:.0f}GB total "
              f"({vram_info['gpu_count']} GPU{'s' if vram_info['gpu_count'] != 1 else ''})")
        for name, req in skipped:
            print(f"  Skipping {name} (requires ~{req}GB)")
        print()

    return kept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model(name: str) -> str:
    """Resolve a model name (short or HF ID) to the canonical HF ID."""
    if name in MODEL_REGISTRY:
        return name
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    # Try case-insensitive short name match
    name_lower = name.lower()
    for short, hf_id in SHORT_NAMES.items():
        if short.lower() == name_lower:
            return hf_id
    raise ValueError(f"Unknown model: {name}. Available: {list(SHORT_NAMES.keys())} or HF IDs.")


def get_short_name(hf_id: str) -> str:
    return MODEL_REGISTRY[hf_id]["short_name"]


def get_critical_layers(short_name: str, output_dir: Path, top_k: int = 5):
    """Auto-detect critical layers from layer ablation results, or use defaults."""
    ablation_path = output_dir / short_name / "layer_ablation.json"
    if ablation_path.exists():
        try:
            with open(ablation_path) as f:
                data = json.load(f)
            layer_flips = []
            for key, stats in data.get("summary", {}).items():
                if key.startswith("layer_"):
                    layer_idx = int(key.split("_")[1])
                    flip_rate = stats.get("flip_rate", 0.0)
                    layer_flips.append((layer_idx, flip_rate))
            layer_flips.sort(key=lambda x: x[1], reverse=True)
            critical = [l for l, f in layer_flips[:top_k] if f >= 0.10]
            if critical:
                return critical, True  # (layers, auto_detected)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            warnings.warn(
                f"Could not load ablation results for {short_name}: {e}, using defaults"
            )
    # Fall back to defaults
    return DEFAULT_CRITICAL.get(short_name, [0]), False


def generate_protection_configs(critical_layers: list, n_layers: int) -> list:
    """Generate protection layer configs for protection_sweep.

    Returns list of comma-separated layer specs like ["0", "0,1", "0,1,2", ...].
    """
    if not critical_layers:
        return ["0"]

    configs = []

    # Single critical layer
    top = critical_layers[0]
    configs.append(str(top))

    # If critical layer is at/near layer 0, build contiguous ranges from 0
    if top <= 2:
        for k in range(2, min(9, n_layers)):
            spec = ",".join(str(i) for i in range(k))
            if spec not in configs:
                configs.append(spec)
    else:
        # Build contiguous range including 0 up to critical layer
        for end in range(top + 1, min(top + 4, n_layers)):
            spec = ",".join(str(i) for i in range(end))
            if spec not in configs:
                configs.append(spec)

    # Progressive top-k from ablation results
    for k in range(2, min(len(critical_layers) + 1, 6)):
        top_k_layers = sorted(critical_layers[:k])
        spec = ",".join(str(l) for l in top_k_layers)
        if spec not in configs:
            configs.append(spec)

    # Ensure at least 6 configs
    if len(configs) < 6 and top <= 2:
        # Already covered by contiguous range above
        pass
    elif len(configs) < 6:
        # Add contiguous from 0
        for k in [2, 3, 4, 5, 6]:
            spec = ",".join(str(i) for i in range(k))
            if spec not in configs:
                configs.append(spec)
            if len(configs) >= 8:
                break

    return configs


# ---------------------------------------------------------------------------
# Manifest (resume support)
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {
        "started": datetime.now().isoformat(),
        "completed": {},
        "failed": [],
    }


def save_manifest(manifest: dict, manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(manifest_path)  # atomic on POSIX


# ---------------------------------------------------------------------------
# Experiment command builders
# ---------------------------------------------------------------------------

def build_commands(hf_id: str, output_dir: Path, requested_experiments: list,
                   max_prompts: int = None, batch_size: int = 4,
                   classifier: str = "wildguard"):
    """Build all (run_key, cmd) pairs for a model, in execution order.

    Returns list of (run_key, cmd_list, phase_label) tuples.
    """
    info = MODEL_REGISTRY[hf_id]
    short = info["short_name"]
    exps = info["experiments"]
    n_layers = info["layers"]
    model_dir = output_dir / short
    commands = []

    def _common(extra_flags: list) -> list:
        """Append --max-prompts and --batch-size if applicable."""
        flags = list(extra_flags)
        if max_prompts is not None:
            flags.extend(["--max-prompts", str(max_prompts)])
        return flags

    def _common_gen(extra_flags: list) -> list:
        """Append --max-prompts and --batch-size for generation scripts."""
        flags = list(extra_flags)
        if max_prompts is not None:
            flags.extend(["--max-prompts", str(max_prompts)])
        flags.extend(["--batch-size", str(batch_size)])
        return flags

    # === Phase A — §4 experiments (per-token asymmetric) ===

    # Sweep x 4 benchmarks
    if "sweep" in exps and "sweep" in requested_experiments:
        benchmarks = exps["sweep"].get("benchmarks", ["custom"])
        for bench in benchmarks:
            run_key = f"{short}/sweep_{bench}"
            out = model_dir / f"sweep_{bench}.json"
            cmd = [PYTHON, str(SCRIPTS["sweep"]),
                   "--model", hf_id,
                   "--bits", "16,8,6,5,4,3,2",
                   "--prompts", bench,
                   "--quantizer-preset", "section4",
                   "--max-new-tokens", "256",
                   "--classifier", classifier,
                   "--output", str(out)]
            commands.append((run_key, _common_gen(cmd), "Phase A — §4 Deployment"))

    # Perplexity
    if "perplexity" in exps and "perplexity" in requested_experiments:
        run_key = f"{short}/perplexity"
        out = model_dir / "perplexity.json"
        cmd = [PYTHON, str(SCRIPTS["perplexity"]),
               "--model", hf_id,
               "--bits", "16,8,6,5,4,3,2",
               "--output", str(out)]
        # NO --max-prompts for perplexity (uses WikiText)
        commands.append((run_key, cmd, "Phase A — §4 Deployment"))

    # IFEval
    if "ifeval" in exps and "ifeval" in requested_experiments:
        run_key = f"{short}/ifeval"
        out = model_dir / "ifeval.json"
        cmd = [PYTHON, str(SCRIPTS["ifeval"]),
               "--model", hf_id,
               "--bits", "16,8,6,5,4,3,2",
               "--max-new-tokens", "1024",
               "--batch-size", str(batch_size),
               "--output", str(out)]
        if max_prompts is not None:
            cmd.extend(["--max-prompts", str(max_prompts)])
        commands.append((run_key, cmd, "Phase A — §4 Deployment"))

    # Determinism
    if "determinism" in exps and "determinism" in requested_experiments:
        det_bits = exps["determinism"].get("bits", 4)
        run_key = f"{short}/determinism"
        out = model_dir / "determinism.json"
        cmd = [PYTHON, str(SCRIPTS["determinism"]),
               "--model", hf_id,
               "--bits", str(det_bits),
               "--seeds", "42,123,456",
               "--prompts", "custom",
               "--max-new-tokens", "256",
               "--classifier", classifier,
               "--output", str(out)]
        commands.append((run_key, _common_gen(cmd), "Phase A — §4 Validation"))

    # Speculative decoding
    if "spec_decoding" in exps and "spec_decoding" in requested_experiments:
        draft = exps["spec_decoding"].get("draft_model", "Qwen/Qwen2.5-0.5B-Instruct")
        run_key = f"{short}/spec_decoding"
        out = model_dir / "spec_decoding.json"
        cmd = [PYTHON, str(SCRIPTS["spec_decoding"]),
               "--target-model", hf_id,
               "--draft-model", draft,
               "--bits", "16,8,4,3",
               "--draft-length", "5",
               "--quantizer-preset", "section4",
               "--prompts", "custom",
               "--max-new-tokens", "256",
               "--classifier", classifier,
               "--output", str(out)]
        if max_prompts is not None:
            cmd.extend(["--max-prompts", str(max_prompts)])
        commands.append((run_key, cmd, "Phase A — §4 Validation"))

    # === Phase B — §5 mechanistic experiments (per-tensor symmetric) ===

    # Layer ablation (MUST run before channel_ablation, protection_sweep, mitigation)
    if "layer_ablation" in exps and "layer_ablation" in requested_experiments:
        run_key = f"{short}/layer_ablation"
        out = model_dir / "layer_ablation.json"
        cmd = [PYTHON, str(SCRIPTS["layer_ablation"]),
               "--model", hf_id,
               "--bits", "3",
               "--prompts", "custom",
               "--quantizer-preset", "section5",
               "--max-new-tokens", "256",
               "--classifier", classifier,
               "--output", str(out)]
        commands.append((run_key, _common_gen(cmd), "Phase B — §5 Mechanistic"))

    # [auto-detect critical layers happens at runtime, not at build time]

    # Cumulative
    if "cumulative" in exps and "cumulative" in requested_experiments:
        run_key = f"{short}/cumulative"
        out = model_dir / "cumulative.json"
        cmd = [PYTHON, str(SCRIPTS["cumulative"]),
               "--model", hf_id,
               "--bits", "3",
               "--direction", "both",
               "--step", "1",
               "--prompts", "custom",
               "--quantizer-preset", "section5",
               "--max-new-tokens", "256",
               "--classifier", classifier,
               "--output", str(out)]
        commands.append((run_key, _common_gen(cmd), "Phase B — §5 Mechanistic"))

    # Channel ablation (needs critical layer — resolved at runtime)
    if "channel_ablation" in exps and "channel_ablation" in requested_experiments:
        run_key = f"{short}/channel_ablation"
        out = model_dir / "channel_ablation.json"
        # --layer will be filled at runtime via CRITICAL_LAYER_PLACEHOLDER
        cmd = [PYTHON, str(SCRIPTS["channel_ablation"]),
               "--model", hf_id,
               "--bits", "3",
               "--layer", "CRITICAL_LAYER_PLACEHOLDER",
               "--prompts", "custom",
               "--quantizer-preset", "section5",
               "--max-new-tokens", "256",
               "--classifier", classifier,
               "--output", str(out)]
        commands.append((run_key, _common_gen(cmd), "Phase B — §5 Mechanistic"))

    # Group-64
    if "group64" in exps and "group64" in requested_experiments:
        run_key = f"{short}/group64"
        out = model_dir / "group64.json"
        cmd = [PYTHON, str(SCRIPTS["group64"]),
               "--model", hf_id,
               "--bits", "3",
               "--prompts", "custom",
               "--quantizer-preset", "section5",
               "--max-new-tokens", "256",
               "--classifier", classifier,
               "--output", str(out)]
        commands.append((run_key, _common_gen(cmd), "Phase B — §5 Mechanistic"))

    # Activation analysis (needs critical layers — resolved at runtime)
    if "activation_analysis" in exps and "activation_analysis" in requested_experiments:
        run_key = f"{short}/activation_analysis"
        out = model_dir / "activation_analysis.json"
        cmd = [PYTHON, str(SCRIPTS["activation_analysis"]),
               "--model", hf_id,
               "--layers", "CRITICAL_LAYERS_PLACEHOLDER",
               "--prompts", "custom",
               "--output", str(out)]
        if max_prompts is not None:
            cmd.extend(["--max-prompts", str(max_prompts)])
        commands.append((run_key, cmd, "Phase B — §5 Mechanistic"))

    # Attention analysis
    if "attention_analysis" in exps and "attention_analysis" in requested_experiments:
        run_key = f"{short}/attention_analysis"
        out = model_dir / "attention_analysis.json"
        cmd = [PYTHON, str(SCRIPTS["attention_analysis"]),
               "--model", hf_id,
               "--prompts", "custom",
               "--max-new-tokens", "1",
               "--batch-size", "1",
               "--output", str(out)]
        if max_prompts is not None:
            cmd.extend(["--max-prompts", str(max_prompts)])
        commands.append((run_key, cmd, "Phase B — §5 Mechanistic"))

    # === Phase C — §6 mitigation experiments (per-token asymmetric) ===

    # Protection sweep (needs critical layers — resolved at runtime)
    if "protection_sweep" in exps and "protection_sweep" in requested_experiments:
        base_bits = exps["protection_sweep"].get("base_bits", 4)
        run_key = f"{short}/protection_sweep"
        out = model_dir / "protection_sweep.json"
        cmd = [PYTHON, str(SCRIPTS["protection_sweep"]),
               "--model", hf_id,
               "--base-bits", str(base_bits),
               "--protect-layers", "PROTECTION_CONFIGS_PLACEHOLDER",
               "--prompts", "custom",
               "--max-new-tokens", "256",
               "--classifier", classifier,
               "--output", str(out)]
        commands.append((run_key, _common_gen(cmd), "Phase C — §6 Mitigation"))

    # Mitigation (may be a list of configs)
    if "mitigation" in exps and "mitigation" in requested_experiments:
        mit_config = exps["mitigation"]
        if isinstance(mit_config, list):
            for mc in mit_config:
                bits = mc["bits"]
                strategies = mc["strategies"]
                run_key = f"{short}/mitigation_{bits}bit"
                out = model_dir / f"mitigation_{bits}bit.json"
                cmd = [PYTHON, str(SCRIPTS["mitigation"]),
                       "--model", hf_id,
                       "--bits", str(bits),
                       "--strategies", strategies,
                       "--critical-layer", "CRITICAL_LAYER_PLACEHOLDER",
                       "--prompts", "custom",
                       "--max-new-tokens", "256",
                       "--classifier", classifier,
                       "--output", str(out)]
                commands.append((run_key, _common_gen(cmd), "Phase C — §6 Mitigation"))
        else:
            bits = mit_config.get("bits", 3)
            strategies = mit_config.get("strategies", "per_tensor,per_channel,group_64,fp16_critical")
            run_key = f"{short}/mitigation"
            out = model_dir / "mitigation.json"
            cmd = [PYTHON, str(SCRIPTS["mitigation"]),
                   "--model", hf_id,
                   "--bits", str(bits),
                   "--strategies", strategies,
                   "--critical-layer", "CRITICAL_LAYER_PLACEHOLDER",
                   "--prompts", "custom",
                   "--max-new-tokens", "256",
                   "--classifier", classifier,
                   "--output", str(out)]
            commands.append((run_key, _common_gen(cmd), "Phase C — §6 Mitigation"))

    # Mitigation 8-bit (uses protection_sweep with --protect-bits 8)
    if "mitigation_8bit" in exps and "mitigation_8bit" in requested_experiments:
        m8 = exps["mitigation_8bit"]
        base_bits = m8.get("base_bits", 4)
        protect_bits = m8.get("protect_bits", 8)
        run_key = f"{short}/mitigation_8bit_protection"
        out = model_dir / "mitigation_8bit_protection.json"
        cmd = [PYTHON, str(SCRIPTS["mitigation_8bit"]),
               "--model", hf_id,
               "--base-bits", str(base_bits),
               "--protect-bits", str(protect_bits),
               "--protect-layers", "PROTECTION_CONFIGS_PLACEHOLDER",
               "--prompts", "custom",
               "--max-new-tokens", "256",
               "--classifier", classifier,
               "--output", str(out)]
        commands.append((run_key, _common_gen(cmd), "Phase C — §6 Mitigation"))

    return commands


def resolve_placeholders(cmd: list, critical_layers: list, protection_configs: list) -> list:
    """Replace placeholder tokens in command with actual values."""
    resolved = []
    i = 0
    while i < len(cmd):
        token = cmd[i]
        if token == "CRITICAL_LAYER_PLACEHOLDER":
            resolved.append(str(critical_layers[0] if critical_layers else 0))
        elif token == "CRITICAL_LAYERS_PLACEHOLDER":
            # Comma-separated list of critical layers + standard layers (0, 1, 2, mid, last)
            resolved.append(",".join(str(l) for l in critical_layers))
        elif token == "PROTECTION_CONFIGS_PLACEHOLDER":
            # Protection configs are multiple positional args after --protect-layers
            resolved.extend(protection_configs)
        else:
            resolved.append(token)
        i += 1
    return resolved


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

def run_experiment(cmd: list, run_key: str, manifest: dict, manifest_path: Path,
                   dry_run: bool, env: dict = None) -> bool:
    """Run a single experiment as a subprocess."""
    if dry_run:
        print(f"    {' '.join(cmd)}")
        return True

    print(f"\n{'='*70}")
    print(f"  RUNNING: {run_key}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, env=env,
                            start_new_session=True)
    elapsed = time.time() - t0

    manifest["completed"][run_key] = {
        "finished": datetime.now().isoformat(),
        "exit_code": result.returncode,
        "elapsed_s": round(elapsed, 1),
    }
    save_manifest(manifest, manifest_path)

    if result.returncode != 0:
        if run_key not in manifest["failed"]:
            manifest["failed"].append(run_key)
        save_manifest(manifest, manifest_path)
        print(f"\n  FAILED: {run_key} (exit code {result.returncode}, {elapsed:.0f}s)")
        return False

    print(f"\n  COMPLETED: {run_key} ({elapsed:.0f}s)")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all --dry-run
  python run_pipeline.py --model qwen7b --all --dry-run
  python run_pipeline.py --model phi35 qwen7b mistral7b --gpu 0 --output-dir results/run_final/
  python run_pipeline.py --all --experiments sweep --dry-run
  python run_pipeline.py --all --resume
  python run_pipeline.py --all --max-prompts 5 --dry-run
        """,
    )
    parser.add_argument("--all", action="store_true",
                        help="Run ALL experiments for all 8 standard models (or specified models)")
    parser.add_argument("--model", type=str, nargs="+", dest="models",
                        help="One or more models (HF ID or short name). "
                             "Accepts: qwen7b, mistral7b, deepseek7b, yi9b, llama8b, gemma9b, msmall24b, phi35, qwen72b")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        help="Experiment names or group names: sweep, mechanistic, mitigation_group, "
                             "quality, validation, section4, section5, section6, or individual names")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: results/run_TIMESTAMP)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print full execution plan without running")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed experiments (reads run_manifest.json)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit prompts for testing (forwarded to scripts that support it)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU ID for CUDA_VISIBLE_DEVICES (e.g., 0 or 1). "
                             "Use with --output-dir to run parallel instances on different GPUs")
    parser.add_argument("--classifier", type=str, default="wildguard",
                        help="Classifier for safety labeling: wildguard (default; "
                             "used to produce all paper results, requires HF auth) or "
                             "family_a (keyword-based, no GPU needed; for quick smoke tests only).")
    return parser.parse_args()


def resolve_experiments(experiment_args: list) -> list:
    """Expand experiment group names into individual experiment names."""
    if experiment_args is None:
        return list(ALL_EXPERIMENTS)

    result = []
    for name in experiment_args:
        if name in EXPERIMENT_GROUPS:
            for exp in EXPERIMENT_GROUPS[name]:
                if exp not in result:
                    result.append(exp)
        elif name in ALL_EXPERIMENTS:
            if name not in result:
                result.append(name)
        else:
            print(f"WARNING: Unknown experiment or group: {name}")
            print(f"  Available experiments: {ALL_EXPERIMENTS}")
            print(f"  Available groups: {list(EXPERIMENT_GROUPS.keys())}")
            sys.exit(1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve models
    explicit_models = bool(args.models)
    if args.models:
        model_ids = [resolve_model(m) for m in args.models]
    elif args.all:
        model_ids = sorted(STANDARD_MODELS, key=lambda x: MODEL_REGISTRY[x]["size_order"])
    else:
        print("ERROR: Specify --all or --model MODEL")
        sys.exit(1)

    # VRAM filter: auto-skip models that won't fit
    vram_info = get_vram_info()
    model_ids = filter_models_by_vram(model_ids, vram_info, explicit=explicit_models)

    # GPU assignment for subprocess isolation (works for any CUDA-visible GPU).
    if args.gpu is not None:
        sub_env = os.environ.copy()
        sub_env["CUDA_VISIBLE_DEVICES"] = args.gpu
        sub_env["HIP_VISIBLE_DEVICES"] = args.gpu
        sub_env["ROCR_VISIBLE_DEVICES"] = args.gpu
    else:
        sub_env = None

    # Resolve experiments
    if args.all and args.experiments is None:
        requested = list(ALL_EXPERIMENTS)
    else:
        requested = resolve_experiments(args.experiments)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.resume:
        # Find most recent run directory
        results_base = Path("results")
        if results_base.exists():
            run_dirs = sorted([d for d in results_base.iterdir()
                              if d.is_dir() and d.name.startswith("run_")],
                             key=lambda d: d.name, reverse=True)
            if run_dirs:
                output_dir = run_dirs[0]
                print(f"Resuming from: {output_dir}")
            else:
                print("ERROR: No run directories found for --resume")
                sys.exit(1)
        else:
            print("ERROR: No results/ directory found for --resume")
            sys.exit(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"run_{timestamp}"

    if args.gpu is not None:
        manifest_path = output_dir / f"run_manifest_gpu{args.gpu}.json"
    else:
        manifest_path = output_dir / "run_manifest.json"
    manifest = load_manifest(manifest_path) if args.resume else {
        "started": datetime.now().isoformat(),
        "completed": {},
        "failed": [],
    }

    # Validate scripts exist
    missing_scripts = []
    for exp_name, script_path in SCRIPTS.items():
        if not script_path.exists():
            missing_scripts.append((exp_name, script_path))
    if missing_scripts:
        print("ERROR: Missing experiment scripts:")
        for name, path in missing_scripts:
            print(f"  {name}: {path}")
        sys.exit(1)

    # Build execution plan
    all_commands = []  # (run_key, cmd, phase_label, model_short, model_hf_id)
    for hf_id in model_ids:
        short = get_short_name(hf_id)
        # Filter requested experiments to only those this model has
        model_exps = set(MODEL_REGISTRY[hf_id]["experiments"].keys())
        model_requested = [e for e in requested if e in model_exps]
        skipped_exps = [e for e in requested if e not in model_exps]
        if skipped_exps and args.experiments is not None:
            print(f"  Note: {short} does not have: {', '.join(skipped_exps)}")

        commands = build_commands(hf_id, output_dir, model_requested,
                                 max_prompts=args.max_prompts,
                                 batch_size=args.batch_size,
                                 classifier=args.classifier)
        for run_key, cmd, phase in commands:
            all_commands.append((run_key, cmd, phase, short, hf_id))

    # Count totals
    total_runs = len(all_commands)

    # Print header
    print(f"\n{'='*70}")
    print(f"  PIPELINE EXECUTION PLAN")
    print(f"{'='*70}")
    print(f"  Models: {len(model_ids)} | Total experiment runs: {total_runs}")
    print(f"  Output: {output_dir}")
    if args.gpu is not None:
        print(f"  GPU: {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")
    if args.resume:
        completed = sum(1 for v in manifest["completed"].values()
                       if v.get("exit_code", -1) == 0)
        print(f"  Resume: {completed} already completed, skipping those")
    if args.max_prompts:
        print(f"  Max prompts: {args.max_prompts} (test mode)")
    print(f"{'='*70}")

    # Execute per model
    current_model = None
    model_num = 0
    run_num = 0
    successes = 0
    failures = 0
    skipped = 0

    for run_key, cmd, phase, short, hf_id in all_commands:
        # Model header
        if hf_id != current_model:
            current_model = hf_id
            model_num += 1
            info = MODEL_REGISTRY[hf_id]
            print(f"\nMODEL {model_num}/{len(model_ids)}: {hf_id} "
                  f"({short}, {info['layers']} layers)")

        run_num += 1

        # Resume: skip completed
        if args.resume and run_key in manifest.get("completed", {}):
            prev = manifest["completed"][run_key]
            if prev.get("exit_code", -1) == 0:
                if args.dry_run:
                    print(f"  [{run_num:>3}] [SKIP] {run_key} (completed)")
                skipped += 1
                continue

        # Resolve critical layer placeholders
        # This must happen at runtime (after layer_ablation may have completed)
        needs_critical = any(
            tok in ("CRITICAL_LAYER_PLACEHOLDER", "CRITICAL_LAYERS_PLACEHOLDER",
                    "PROTECTION_CONFIGS_PLACEHOLDER")
            for tok in cmd
        )
        if needs_critical:
            critical_layers, auto = get_critical_layers(short, output_dir)
            protection_configs = generate_protection_configs(
                critical_layers, MODEL_REGISTRY[hf_id]["layers"]
            )
            cmd = resolve_placeholders(cmd, critical_layers, protection_configs)
            source = "auto-detected" if auto else "default"
            if not args.dry_run:
                print(f"  Critical layers ({source}): {critical_layers}")

        # Phase header (for dry-run readability)
        if args.dry_run:
            # Show phase grouping
            print(f"  [{run_num:>3}] {run_key}")
            print(f"        {' '.join(cmd)}")
            successes += 1
            continue

        # Run
        ok = run_experiment(cmd, run_key, manifest, manifest_path, dry_run=False, env=sub_env)
        if ok:
            successes += 1
        else:
            failures += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"  PIPELINE {'PLAN' if args.dry_run else 'COMPLETE'}")
    print(f"{'='*70}")
    print(f"  Total runs: {total_runs}")
    if args.dry_run:
        print(f"  Would execute: {successes}")
        if skipped:
            print(f"  Would skip (already done): {skipped}")
    else:
        print(f"  Succeeded: {successes}")
        print(f"  Failed: {failures}")
        print(f"  Skipped (resume): {skipped}")
        if failures > 0:
            print(f"\n  FAILED EXPERIMENTS:")
            for key in manifest.get("failed", []):
                info = manifest["completed"].get(key, {})
                print(f"    {key} (exit code {info.get('exit_code', '?')})")
        print(f"\n  Results: {output_dir}")
        print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
