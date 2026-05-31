#!/usr/bin/env python3
"""Generate LaTeX tables from unified result JSON files.

Covers all 38 labelled paper tables. Tables are grouped by section:
  - Results (§4): sweep, advbench, ppl, mse, phase thresholds, seeds, etc.
  - Mechanism (§5): layer sensitivity, cumulative, channel ablation, PCR, etc.
  - Protocol (§6): protection sweeps, mitigation, protocol validation, etc.
  - Appendix: real-dtype, 72B

Usage:
    python analysis/generate_tables.py --results-dir results/ --table all
    python analysis/generate_tables.py --results-dir results/ --table sweep --models qwen,mistral
    python analysis/generate_tables.py --results-dir results/ --list
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.results import load_results
from core.model_loader import MODEL_REGISTRY

# Display names for models
DISPLAY_NAMES = {
    "qwen": "Qwen-2.5-7B",
    "mistral": "Mistral-7B",
    "deepseek": "DeepSeek-7B",
    "yi": "Yi-1.5-9B",
    "llama": "LLaMA-3.1-8B",
    "gemma2": "Gemma-2-9B",
    "mistral-small": "Mistral-Small-24B",
    "phi35": "Phi-3.5-mini",
    "qwen-72b": "Qwen2.5-72B",
    "yi-34b": "Yi-34B",
}

N_LAYERS = {
    "qwen": 28, "mistral": 32, "deepseek": 30, "yi": 48,
    "llama": 32, "gemma2": 42, "mistral-small": 40, "phi35": 32,
    "qwen-72b": 80,
    "yi-34b": 60,
}

ALL_MODELS = ["qwen", "mistral", "deepseek", "yi", "llama", "gemma2", "mistral-small", "phi35"]

# Mapping from canonical names to pipeline directory names (run_pipeline.py uses these)
_DIR_ALIASES = {
    "qwen": "qwen7b",
    "mistral": "mistral7b",
    "deepseek": "deepseek7b",
    "yi": "yi9b",
    "llama": "llama8b",
    "gemma2": "gemma9b",
    "mistral-small": "msmall24b",
    "phi35": "phi35",
    "qwen-72b": "qwen72b",
    "yi-34b": "yi34b",
    "mixtral": "mixtral",
    "olmo2": "olmo2",
}
# Reverse: pipeline name -> canonical
_DIR_ALIASES_REV = {v: k for k, v in _DIR_ALIASES.items()}


def _dn(model_name):
    """Get display name for a model."""
    return DISPLAY_NAMES.get(model_name, model_name)


def _pct(val, fmt=".1%"):
    """Format a value as percentage, or return '-' if None."""
    if val is None:
        return "-"
    return f"{val:{fmt}}"


def _cat_flip_rate(summary, bits, category, baseline="16"):
    """Compute per-category flip rate from flips count and baseline refused count.

    The category_breakdown stores 'flips' (count) not 'flip_rate'.
    Flip rate = flips / baseline_refused_in_category.
    """
    cat = summary.get(str(bits), {}).get("category_breakdown", {}).get(category, {})
    bl_cat = summary.get(baseline, {}).get("category_breakdown", {}).get(category, {})
    flips = cat.get("flips")
    bl_refused = bl_cat.get("refused", 0)
    if flips is not None and bl_refused > 0:
        return flips / bl_refused
    return None


def _cat_flip_rate_raw(cat_entry, bl_cat_entry):
    """Compute flip rate from a category breakdown entry and its baseline.

    Works with any key structure (bit-widths, seeds, conditions).
    """
    flips = cat_entry.get("flips")
    bl_refused = bl_cat_entry.get("refused", 0)
    if flips is not None and bl_refused > 0:
        return flips / bl_refused
    return None


def _try_load(results_dir, model_name, filename):
    """Try to load a result file, return None if not found.

    Checks both canonical directory names (qwen/) and pipeline names (qwen7b/).
    """
    path = results_dir / model_name / filename
    if path.exists():
        return load_results(path)
    # Try pipeline directory name alias
    alias = _DIR_ALIASES.get(model_name)
    if alias and alias != model_name:
        path = results_dir / alias / filename
        if path.exists():
            return load_results(path)
    return None


# =============================================================================
# §4 Results Tables
# =============================================================================

def generate_custom_all_models(results_dir, models):
    """tab:custom_all_models — Custom benchmark (63 prompts, all models)."""
    bits_list = [16, 8, 6, 5, 4, 3, 2]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Custom benchmark results (63 prompts) under uniform KV cache quantization.}",
        r"\label{tab:custom_all_models}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Bits & KV MSE & Refusal Flip & Privacy Leak & Jailbreak \\",
        r"\midrule",
    ]

    for model_name in models:
        result = _try_load(results_dir, model_name, "sweep_custom.json")
        if result is None:
            continue

        dn = _dn(model_name)
        first = True
        for bits in bits_list:
            s = result.summary.get(str(bits), {})
            mse = s.get("mean_kv_mse")
            mse_str = f"{mse:.4f}" if mse else "-"

            cat = s.get("category_breakdown", {})
            ref_flip = _cat_flip_rate(result.summary, bits, "refusal")
            priv_flip = _cat_flip_rate(result.summary, bits, "privacy")
            jail_flip = _cat_flip_rate(result.summary, bits, "jailbreak")

            if bits == 16:
                # Baseline: show refusal rates instead of flip rates
                ref_rate = cat.get("refusal", {}).get("rate")
                priv_rate = cat.get("privacy", {}).get("rate")
                jail_rate = cat.get("jailbreak", {}).get("rate")
                mrow = f"\\multirow{{7}}{{*}}{{{dn}}}" if first else ""
                row = f"{mrow} & {bits} & - & {_pct(ref_rate)} & {_pct(priv_rate)} & {_pct(jail_rate)}"
            else:
                row = f" & {bits} & {mse_str} & {_pct(ref_flip)} & {_pct(priv_flip)} & {_pct(jail_flip)}"

            lines.append(row + r" \\")
            first = False

        lines.append(r"\midrule")

    # Remove trailing midrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_advbench_main(results_dir, models):
    """tab:advbench_main — AdvBench results (N=520)."""
    bits_list = [8, 4, 3, 2]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{AdvBench results (N=520) on refusal under quantization.}",
        r"\label{tab:advbench_main}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Bits & Baseline Refusal & Quantized Refusal & Flip Rate & Conditional Flip \\",
        r"\midrule",
    ]

    for model_name in models:
        result = _try_load(results_dir, model_name, "sweep_advbench.json")
        if result is None:
            continue

        dn = _dn(model_name)
        bl = result.summary.get("16", {})
        bl_rate = bl.get("refusal_rate", 0)

        first = True
        for bits in bits_list:
            s = result.summary.get(str(bits), {})
            q_rate = s.get("refusal_rate")
            fr = s.get("flip_rate")
            cf = s.get("conditional_flip", fr)

            mrow = f"\\multirow{{{len(bits_list)}}}{{*}}{{{dn}}}" if first else ""
            lines.append(
                f"{mrow} & {bits} & {_pct(bl_rate)} & {_pct(q_rate)} & {_pct(fr)} & {_pct(cf)}" + r" \\"
            )
            first = False
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_multi_model_ppl(results_dir, models):
    """tab:multi_model_ppl — Perplexity under KV quantization."""
    bits_list = [16, 8, 6, 5, 4, 3, 2]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Perplexity (WikiText-103) under KV cache quantization across all models.}",
        r"\label{tab:multi_model_ppl}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(bits_list) + "}",
        r"\toprule",
        r"Model & " + " & ".join(f"{b}-bit" for b in bits_list) + r" \\",
        r"\midrule",
    ]

    for model_name in models:
        result = _try_load(results_dir, model_name, "perplexity.json")
        if result is None:
            continue

        dn = _dn(model_name)
        cells = [dn]
        for bits in bits_list:
            s = result.summary.get(str(bits), {})
            ppl = s.get("perplexity")
            cells.append(f"{ppl:.2f}" if ppl is not None else "-")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines)


def generate_std_metrics_mistral(results_dir, models):
    """tab:std_metrics_mistral — Standard language modeling metrics for Mistral."""
    bits_list = [16, 4, 3, 2]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Standard language modeling metric under uniform KV quantization for Mistral-7B-Instruct-v0.2.}",
        r"\label{tab:std_metrics_mistral}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"KV Precision & KV Memory vs FP16 & WikiText PPL \\",
        r"\midrule",
    ]

    mem_scale = {16: "$1.00\\times$", 4: "$0.25\\times$", 3: "$0.19\\times$", 2: "$0.13\\times$"}
    result = _try_load(results_dir, "mistral", "perplexity.json")

    for bits in bits_list:
        label = "FP16" if bits == 16 else f"{bits}-bit"
        scale = mem_scale.get(bits, "-")
        ppl = "-"
        if result:
            s = result.summary.get(str(bits), {})
            p = s.get("perplexity")
            if p is not None:
                ppl = f"{p:.2f}"
        lines.append(f"{label} & {scale} & {ppl}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_mistral_mse_phase(results_dir, models):
    """tab:mistral_mse_phase — Mistral KV MSE and phase transition."""
    bits_list = [4, 3, 2]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mistral-7B phase transition: MSE growth vs refusal collapse.}",
        r"\label{tab:mistral_mse_phase}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Bits & KV MSE & Relative MSE & Refusal Flip \\",
        r"\midrule",
    ]

    result = _try_load(results_dir, "mistral", "sweep_custom.json")
    if result:
        mse_4 = result.summary.get("4", {}).get("mean_kv_mse")
        for bits in bits_list:
            s = result.summary.get(str(bits), {})
            mse = s.get("mean_kv_mse")
            flip = s.get("flip_rate")
            rel = f"${mse/mse_4:.1f}\\times$" if mse and mse_4 and mse_4 > 0 else "-"
            mse_str = f"{mse:.4f}" if mse else "-"
            lines.append(
                f"{bits}-bit & {mse_str} & {rel} & {_pct(flip)}" + r" \\"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_phase_thresholds(results_dir, models):
    """tab:phase_thresholds — Phase-transition thresholds by model."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Phase-transition thresholds by model (conditional flip rate).}",
        r"\label{tab:phase_thresholds}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Safe Zone & Collapse Begins & Complete Collapse \\",
        r"\midrule",
    ]

    bits_list = [16, 8, 6, 5, 4, 3, 2]
    for model_name in models:
        result = _try_load(results_dir, model_name, "sweep_custom.json")
        if result is None:
            result = _try_load(results_dir, model_name, "sweep_advbench.json")
        if result is None:
            continue

        dn = _dn(model_name)
        safe = collapse_begin = complete = "-"

        for bits in bits_list:
            s = result.summary.get(str(bits), {})
            fr = s.get("flip_rate")
            if fr is None:
                continue
            if fr < 0.05:
                safe = f"{bits}-bit ({fr:.1%})"
            elif fr >= 0.05 and collapse_begin == "-":
                collapse_begin = f"{bits}-bit ({fr:.1%})"
            if fr >= 0.80:
                complete = f"{bits}-bit ({fr:.1%})"
                break

        lines.append(f"{dn} & {safe} & {collapse_begin} & {complete}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_cross_suite_compare(results_dir, models):
    """tab:cross_suite_compare — Custom vs AdvBench comparison."""
    bits_list = [4, 3, 2]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-suite comparison of refusal drift (custom vs AdvBench).}",
        r"\label{tab:cross_suite_compare}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Bits & Custom Refusal Flip & AdvBench Cond.\ Flip & Custom Regime & AdvBench Regime \\",
        r"\midrule",
    ]

    def _regime(fr):
        if fr is None:
            return "-"
        if fr < 0.05:
            return "Safe"
        elif fr < 0.50:
            return "Partial"
        else:
            return "Collapse"

    for model_name in models:
        custom = _try_load(results_dir, model_name, "sweep_custom.json")
        advbench = _try_load(results_dir, model_name, "sweep_advbench.json")
        if custom is None and advbench is None:
            continue

        dn = _dn(model_name)
        for bits in bits_list:
            c_fr = custom.summary.get(str(bits), {}).get("flip_rate") if custom else None
            a_fr = advbench.summary.get(str(bits), {}).get("flip_rate") if advbench else None
            lines.append(
                f"{dn} & {bits} & {_pct(c_fr)} & {_pct(a_fr)} & {_regime(c_fr)} & {_regime(a_fr)}" + r" \\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_mistral_seeds(results_dir, models):
    """tab:mistral_seeds — Mistral-7B seed-level reproducibility."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mistral-7B at 3-bit: perfect reproducibility across seeds.}",
        r"\label{tab:mistral_seeds}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Seed & Refusal Flip & Privacy Leak & Jailbreak & KV MSE \\",
        r"\midrule",
    ]

    result = _try_load(results_dir, "mistral", "determinism.json")
    if result:
        for key, val in sorted(result.summary.items()):
            if not key.startswith("seed_"):
                continue
            seed = key.split("_")[1]
            cat = val.get("category_breakdown", {})
            # Baseline: try "16" key first, fallback to first seed
            bl_summary = result.summary.get("16", {})
            bl_cat = bl_summary.get("category_breakdown", {})
            ref = _cat_flip_rate_raw(cat.get("refusal", {}), bl_cat.get("refusal", {}))
            priv = _cat_flip_rate_raw(cat.get("privacy", {}), bl_cat.get("privacy", {}))
            jail = _cat_flip_rate_raw(cat.get("jailbreak", {}), bl_cat.get("jailbreak", {}))
            mse = val.get("mean_kv_mse")
            mse_str = f"{mse:.6f}" if mse else "-"
            lines.append(f"{seed} & {_pct(ref)} & {_pct(priv)} & {_pct(jail)} & {mse_str}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_multi_seeds(results_dir, models):
    """tab:multi_seeds — Multi-model seed reproducibility."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Seed-level reproducibility at phase-transition boundaries.}",
        r"\label{tab:multi_seeds}",
        r"\begin{tabular}{llcl}",
        r"\toprule",
        r"Model & Bit-width & Refusal Flip & Result (3 seeds) \\",
        r"\midrule",
    ]

    for model_name in models:
        result = _try_load(results_dir, model_name, "determinism.json")
        if result is None:
            continue

        dn = _dn(model_name)
        bits = result.metadata.get("config", {}).get("bits", "?")

        # Check cross-seed agreement
        agreement = result.summary.get("cross_seed_agreement", 0)
        identical = result.summary.get("identical_responses", 0)
        total = len(result.prompts)

        seed_keys = [k for k in result.summary if k.startswith("seed_")]
        if seed_keys:
            fr = result.summary[seed_keys[0]].get("flip_rate", 0)
            result_str = f"Identical ({identical}/{total})" if agreement == 1.0 else f"Agreement {agreement:.0%}"
            lines.append(f"{dn} & {bits}-bit & {_pct(fr)} & {result_str}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_mitigation_main(results_dir, models):
    """tab:mitigation_main — Alignment-aware mixed precision."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mitigation via alignment-aware mixed precision.}",
        r"\label{tab:mitigation_main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & FP16 Refusal & Uniform Refusal & Mixed ($\rho$=0.10) & Recovery \\",
        r"\midrule",
    ]

    for model_name in models:
        result = _try_load(results_dir, model_name, "mitigation.json")
        if result is None:
            result = _try_load(results_dir, model_name, "mitigation_4bit.json")
        if result is None:
            result = _try_load(results_dir, model_name, "mitigation_3bit.json")
        if result is None:
            continue

        dn = _dn(model_name)
        bl = result.summary.get("baseline", {})
        pt = result.summary.get("per_tensor", {})
        # Look for mixed or fp16_critical
        mixed = result.summary.get("fp16_critical", result.summary.get("mixed", {}))

        bl_rate = bl.get("refusal_rate")
        pt_rate = pt.get("refusal_rate")
        mx_rate = mixed.get("refusal_rate")

        # Recovery = (mixed_refusal - uniform_refusal) / (fp16_refusal - uniform_refusal)
        recovery = "-"
        if bl_rate and pt_rate and mx_rate and (bl_rate - pt_rate) > 0:
            rec = (mx_rate - pt_rate) / (bl_rate - pt_rate)
            recovery = _pct(rec)

        lines.append(
            f"{dn} & {_pct(bl_rate)} & {_pct(pt_rate)} & {_pct(mx_rate)} & {recovery}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_specdecode_flip(results_dir, models):
    """tab:specdecode_flip — Alignment drift under speculative decoding."""
    bits_list = [8, 4, 3]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Alignment drift under speculative decoding (relative to FP16).}",
        r"\label{tab:specdecode_flip}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Config & Refusal Flip & Privacy Flip & Jailbreak Flip & AdvBench Flip & Conditional Flip \\",
        r"\midrule",
    ]

    # Try qwen first, then any model with spec_decoding results
    for model_name in ["qwen"] + list(models):
        result = _try_load(results_dir, model_name, "spec_decoding.json")
        if result:
            break
    else:
        result = None

    if result:
        for bits in bits_list:
            s = result.summary.get(str(bits), {})
            cat = s.get("category_breakdown", {})
            bl_cat = result.summary.get("16", {}).get("category_breakdown", {})
            ref = _cat_flip_rate_raw(cat.get("refusal", {}), bl_cat.get("refusal", {}))
            priv = _cat_flip_rate_raw(cat.get("privacy", {}), bl_cat.get("privacy", {}))
            jail = _cat_flip_rate_raw(cat.get("jailbreak", {}), bl_cat.get("jailbreak", {}))
            advb = _cat_flip_rate_raw(cat.get("advbench", {}), bl_cat.get("advbench", {}))
            cf = s.get("flip_rate")
            lines.append(
                f"{bits}-bit KV & {_pct(ref)} & {_pct(priv)} & {_pct(jail)} & {_pct(advb)} & {_pct(cf)}" + r" \\"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_72b_alignment(results_dir, models):
    """tab:72b_alignment — Qwen-72B alignment degradation."""
    bits_list = [16, 8, 4, 3, 2]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Alignment degradation under KV quantization on Qwen2.5-72B-Instruct.}",
        r"\label{tab:72b_alignment}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"KV Bits & MSE & Refusal Flip & Privacy Drift & Jailbreak Success & Overall Drift \\",
        r"\midrule",
    ]

    result = _try_load(results_dir, "qwen-72b", "sweep_custom.json")
    if result:
        total = len(result.prompts)
        for bits in bits_list:
            s = result.summary.get(str(bits), {})
            mse = s.get("mean_kv_mse")
            mse_str = f"{mse:.4f}" if mse else "-"
            cat = s.get("category_breakdown", {})
            ref = _cat_flip_rate(result.summary, bits, "refusal")
            priv = _cat_flip_rate(result.summary, bits, "privacy")
            jail = _cat_flip_rate(result.summary, bits, "jailbreak")
            fc = s.get("flip_count", 0)
            overall = f"{fc}/{total} ({fc/total:.1%})" if total > 0 else "-"

            if bits == 16:
                lines.append(f"{bits} & - & - & - & - & -" + r" \\")
            else:
                lines.append(
                    f"{bits} & {mse_str} & {_pct(ref)} & {_pct(priv)} & {_pct(jail)} & {overall}" + r" \\"
                )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# =============================================================================
# §5 Mechanistic Tables
# =============================================================================

def generate_individual_layer_sensitivity(results_dir, models):
    """tab:individual_layer_sensitivity — Top damaging layers per model."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Individual layer sensitivity: refusal flip when single layer quantized to 3-bit.}",
        r"\label{tab:individual_layer_sensitivity}",
        r"\begin{tabular}{llccl}",
        r"\toprule",
        r"Model & Total Layers & Critical Layer & Single-Layer Flip & Pattern \\",
        r"\midrule",
    ]

    for model_name in models:
        result = _try_load(results_dir, model_name, "layer_ablation.json")
        if result is None:
            continue

        dn = _dn(model_name)
        n_layers = N_LAYERS.get(model_name, "?")

        # Find critical layer (highest flip)
        layer_flips = []
        for key, val in result.summary.items():
            if key.startswith("layer_"):
                idx = int(key.split("_")[1])
                fr = val.get("flip_rate", 0.0)
                layer_flips.append((idx, fr))
        layer_flips.sort(key=lambda x: x[1], reverse=True)

        if not layer_flips:
            continue

        crit_layer, crit_flip = layer_flips[0]
        n_above_10 = sum(1 for _, fr in layer_flips if fr > 0.10)

        if n_above_10 <= 1:
            pattern = "Concentrated"
        elif n_above_10 <= 4:
            pattern = "Distributed-early"
        else:
            pattern = "Ultra-distributed"

        lines.append(
            f"{dn} & {n_layers} & Layer {crit_layer} & {_pct(crit_flip)} & {pattern}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def _generate_full_layer_table(results_dir, model_name, label, caption):
    """Helper: full layer sensitivity for one model (two-column layout)."""
    result = _try_load(results_dir, model_name, "layer_ablation.json")
    n_layers = N_LAYERS.get(model_name, 32)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{rcclrcl}",
        r"\toprule",
        r"Layer & Flips & Flip Rate & & Layer & Flips & Flip Rate \\",
        r"\midrule",
    ]

    if result:
        half = (n_layers + 1) // 2
        bl_refusals = result.summary.get("baseline", {}).get("refusal_count", 0)

        for i in range(half):
            left_key = f"layer_{i}"
            left_s = result.summary.get(left_key, {})
            left_flips = left_s.get("flip_count", 0)
            left_fr = left_s.get("flip_rate", 0)

            right_idx = i + half
            if right_idx < n_layers:
                right_key = f"layer_{right_idx}"
                right_s = result.summary.get(right_key, {})
                right_flips = right_s.get("flip_count", 0)
                right_fr = right_s.get("flip_rate", 0)
                right_part = f"{right_idx} & {right_flips} & {_pct(right_fr)}"
            else:
                right_part = "& &"

            lines.append(f"{i} & {left_flips} & {_pct(left_fr)} & & {right_part}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_qwen_individual_all(results_dir, models):
    """tab:qwen_individual_all — Qwen complete layer sensitivity."""
    return _generate_full_layer_table(
        results_dir, "qwen", "tab:qwen_individual_all",
        "Qwen-2.5-7B complete individual layer sensitivity (all 28 layers)."
    )


def generate_llama_individual_all(results_dir, models):
    """tab:llama_individual_all — LLaMA complete layer sensitivity."""
    return _generate_full_layer_table(
        results_dir, "llama", "tab:llama_individual_all",
        "LLaMA-3.1-8B complete individual layer sensitivity (all 32 layers)."
    )


def generate_cross_model_layer_summary(results_dir, models):
    """tab:cross_model_layer_summary — Cross-model comparison of layer-level safety."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-model comparison of layer-level safety encoding.}",
        r"\label{tab:cross_model_layer_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
        "Feature & " + " & ".join(f"{_dn(m)} ({N_LAYERS.get(m, '?')}L)" for m in models) + r" \\",
        r"\midrule",
    ]

    features = ["Most critical layer", "Layers >10\\%", "Max single-layer flip"]
    rows = {f: [] for f in features}

    for model_name in models:
        result = _try_load(results_dir, model_name, "layer_ablation.json")

        layer_flips = []
        if result:
            for key, val in result.summary.items():
                if key.startswith("layer_"):
                    idx = int(key.split("_")[1])
                    fr = val.get("flip_rate", 0.0)
                    layer_flips.append((idx, fr))
            layer_flips.sort(key=lambda x: x[1], reverse=True)

        if layer_flips:
            rows["Most critical layer"].append(f"Layer {layer_flips[0][0]}")
            rows["Layers >10\\%"].append(str(sum(1 for _, fr in layer_flips if fr > 0.10)))
            rows["Max single-layer flip"].append(_pct(layer_flips[0][1]))
        else:
            for f in features:
                rows[f].append("-")

    for f in features:
        lines.append(f + " & " + " & ".join(rows[f]) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines)


def _generate_cumulative_table(results_dir, model_name, label, caption):
    """Helper: cumulative ablation for one model."""
    result = _try_load(results_dir, model_name, "cumulative.json")
    n_layers = N_LAYERS.get(model_name, 32)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Layers Quantized & Direction & Flip Rate \\",
        r"\midrule",
    ]

    if result:
        # Collect first-k and last-k entries
        entries = []
        for key, val in result.summary.items():
            if key.startswith("first_") or key.startswith("last_"):
                direction, k = key.split("_", 1)
                fr = val.get("flip_rate", 0.0)
                entries.append((direction, int(k), fr))

        # Show select k values to keep table reasonable
        entries.sort(key=lambda x: (x[0], x[1]))
        for direction, k, fr in entries:
            if direction == "first":
                desc = f"Layers 0--{k-1}"
                dir_label = "First-$k$"
            else:
                desc = f"Layers {n_layers-k}--{n_layers-1}"
                dir_label = "Last-$k$"
            lines.append(f"{desc} & {dir_label} & {_pct(fr)}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_cumulative_qwen(results_dir, models):
    """tab:cumulative_qwen"""
    return _generate_cumulative_table(
        results_dir, "qwen", "tab:cumulative_qwen",
        "Qwen-2.5-7B cumulative ablation (28 layers, 3-bit)."
    )


def generate_cumulative_deepseek(results_dir, models):
    """tab:cumulative_deepseek"""
    return _generate_cumulative_table(
        results_dir, "deepseek", "tab:cumulative_deepseek",
        "DeepSeek-7B cumulative ablation (30 layers, 3-bit)."
    )


def generate_cumulative_mistral(results_dir, models):
    """tab:cumulative_mistral"""
    return _generate_cumulative_table(
        results_dir, "mistral", "tab:cumulative_mistral",
        "Mistral-7B cumulative ablation (32 layers, 3-bit)."
    )


def generate_cumulative_yi(results_dir, models):
    """tab:cumulative_yi"""
    return _generate_cumulative_table(
        results_dir, "yi", "tab:cumulative_yi",
        "Yi-1.5-9B cumulative ablation (48 layers, 3-bit)."
    )


def generate_cumulative_phi35(results_dir, models):
    """tab:cumulative_phi35"""
    return _generate_cumulative_table(
        results_dir, "phi35", "tab:cumulative_phi35",
        "Phi-3.5-mini cumulative ablation (32 layers, 3-bit)."
    )


def generate_deepseek_asymmetry(results_dir, models):
    """tab:deepseek_asymmetry — DeepSeek front-vs-back asymmetry."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{DeepSeek-7B front-vs-back asymmetry in layer sensitivity.}",
        r"\label{tab:deepseek_asymmetry}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$k$ layers & Front (0..$k$$-$1) & Back (30$-$$k$..29) & Ratio \\",
        r"\midrule",
    ]

    result = _try_load(results_dir, "deepseek", "cumulative.json")
    if result:
        k_values = [1, 3, 5, 10]
        for k in k_values:
            front = result.summary.get(f"first_{k}", {}).get("flip_rate")
            back = result.summary.get(f"last_{k}", {}).get("flip_rate")
            if front is not None and back is not None and back > 0:
                ratio = f"${front/back:.1f}\\times$"
            elif front is not None and back is not None and back == 0:
                ratio = "---"
            else:
                ratio = "-"
            lines.append(f"{k} & {_pct(front)} & {_pct(back)} & {ratio}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_causal_vs_attention(results_dir, models):
    """tab:causal_vs_attention — Causal vs attention-based layer importance."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Causal vs.\ attention-based layer importance for Qwen at 4-bit.}",
        r"\label{tab:causal_vs_attention}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Protection Strategy & Flip Rate & Recovery & Selection Basis \\",
        r"\midrule",
    ]

    # This table compares protection sweep configs with different selection bases
    result = _try_load(results_dir, "qwen", "protection_sweep.json")
    if result:
        unprotected = result.summary.get("unprotected", {}).get("flip_rate", 0)
        baseline_rr = result.summary.get("baseline", {}).get("refusal_rate", 1)

        for key, val in sorted(result.summary.items()):
            if not key.startswith("protect_"):
                continue
            fr = val.get("flip_rate")
            if fr is not None and unprotected > 0:
                recovery = 1.0 - (fr / unprotected) if unprotected > 0 else 0
                lines.append(
                    f"{key} & {_pct(fr)} & {_pct(recovery)} & Causal (ablation)" + r" \\"
                )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_channel_ablation_all(results_dir, models):
    """tab:channel_ablation_all — Channel-level ablation at critical layers."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Channel-level ablation at each model's critical safety layer.}",
        r"\label{tab:channel_ablation_all}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Critical Layer & Channels & Outliers & Channel Subset Quantized & Flip Rate \\",
        r"\midrule",
    ]

    subsets = ["per_tensor", "per_channel", "outliers", "non_outliers", "random_5pct"]
    subset_labels = {
        "per_tensor": "All (per-tensor)",
        "per_channel": "All (per-channel)",
        "outliers": "Outlier channels only",
        "non_outliers": "Non-outlier channels only",
        "random_5pct": "Random 5\\%",
    }

    for model_name in models:
        result = _try_load(results_dir, model_name, "channel_ablation.json")
        if result is None:
            continue

        dn = _dn(model_name)
        cfg = result.metadata.get("config", {})
        layer = cfg.get("layer", "?")
        channels = cfg.get("total_channels", "?")
        outliers = cfg.get("outlier_count", "?")

        first = True
        for subset in subsets:
            s = result.summary.get(subset, {})
            fr = s.get("flip_rate")
            if fr is None:
                continue

            mrow = f"\\multirow{{{len(subsets)}}}{{*}}{{{dn}}}" if first else ""
            layer_str = f"Layer {layer}" if first else ""
            ch_str = str(channels) if first else ""
            out_str = str(outliers) if first else ""

            lines.append(
                f"{mrow} & {layer_str} & {ch_str} & {out_str} & {subset_labels.get(subset, subset)} & {_pct(fr)}" + r" \\"
            )
            first = False
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_pcr_framework(results_dir, models):
    """tab:pcr_framework — Per-Channel Reduction framework."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-Channel Reduction (PCR) framework and Group-64 validation.}",
        r"\label{tab:pcr_framework}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Critical Layer & PCR & Per-tensor Flip & Group-64 Flip & G64 Reduction & Prediction \\",
        r"\midrule",
    ]

    for model_name in models:
        ca = _try_load(results_dir, model_name, "channel_ablation.json")
        g64 = _try_load(results_dir, model_name, "group64.json")
        if ca is None:
            continue

        dn = _dn(model_name)
        layer = ca.metadata.get("config", {}).get("layer", "?")

        pt_flip = ca.summary.get("per_tensor", {}).get("flip_rate", 0)
        pc_flip = ca.summary.get("per_channel", {}).get("flip_rate", 0)
        pcr = 1 - (pc_flip / pt_flip) if pt_flip > 0 else 0

        g64_flip = "-"
        g64_red = "-"
        prediction = "-"
        if g64:
            g64_f = g64.summary.get("group_64", {}).get("flip_rate")
            pt_all = g64.summary.get("per_tensor", {}).get("flip_rate")
            if g64_f is not None:
                g64_flip = _pct(g64_f)
            if g64_f is not None and pt_all and pt_all > 0:
                red = 1.0 - (g64_f / pt_all)
                g64_red = _pct(red)
                prediction = "\\checkmark" if (pcr > 0.5 and red > 0.3) or (pcr < 0.3 and red < 0.1) else "\\texttimes"

        lines.append(
            f"{dn} & Layer {layer} & {_pct(pcr)} & {_pct(pt_flip)} & {g64_flip} & {g64_red} & {prediction}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_activation_stats(results_dir, models):
    """tab:activation_stats — Activation statistics at critical layers."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Activation statistics at critical safety layers during inference.}",
        r"\label{tab:activation_stats}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
    ]

    # Header with model names and layers
    header_parts = ["Metric"]
    for m in models:
        result = _try_load(results_dir, m, "activation_analysis.json")
        layer = result.metadata.get("config", {}).get("layer", "?") if result else "?"
        header_parts.append(f"{_dn(m)} L{layer}")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    metrics = ["max_activation", "outlier_magnitude", "quant_mse", "outlier_ratio", "channels"]
    metric_labels = {
        "max_activation": "Max activation",
        "outlier_magnitude": "Outlier magnitude",
        "quant_mse": "Quantization MSE",
        "outlier_ratio": "Outlier ratio",
        "channels": "Channels",
    }

    for metric in metrics:
        cells = [metric_labels.get(metric, metric)]
        for m in models:
            result = _try_load(results_dir, m, "activation_analysis.json")
            if result:
                val = result.summary.get(metric)
                if val is not None:
                    if isinstance(val, float):
                        cells.append(f"{val:.4f}" if val < 1 else f"{val:.1f}")
                    else:
                        cells.append(str(val))
                else:
                    cells.append("-")
            else:
                cells.append("-")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines)


def generate_taxonomy(results_dir, models):
    """tab:taxonomy — Taxonomy of three failure modes."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Taxonomy of three failure modes governing safety vulnerability.}",
        r"\label{tab:taxonomy}",
        r"\begin{tabular}{lp{4.8cm}ccp{4.2cm}}",
        r"\toprule",
        r"Failure Mode & Models (PCR) & PCR Range & G64 Effective? & Prescribed Mitigation \\",
        r"\midrule",
    ]

    # Static taxonomy table — values derived from PCR analysis
    taxonomy = [
        (
            "Outlier-as-safety",
            "Qwen (15\\%), Mistral-7B (33\\%)",
            "$<$30\\%",
            "No (0--36\\%)",
            "FP16 protect critical layer; group-64 insufficient"
        ),
        (
            "Outlier-crushes-safety",
            "Yi (78\\%), DeepSeek (97\\%), Gemma-2 (100\\%)",
            "$>$70\\%",
            "Yes (55--92\\%)",
            "Group-64 all layers; per-channel at critical layer"
        ),
        (
            "Multi-layer dilution",
            "Phi-3.5 (81\\%), LLaMA-3.1 (89\\%), M-Small (50\\%)",
            "30--90\\%",
            "Variable (0--76\\%)",
            "Model-specific: Phi needs full FP16; LLaMA/M-Small need multi-layer protection"
        ),
    ]

    for mode, model_list, pcr_range, g64, mitigation in taxonomy:
        lines.append(
            f"{mode} & {model_list} & {pcr_range} & {g64} & {mitigation}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


# =============================================================================
# §6 Protocol Tables
# =============================================================================

def generate_decision_tree(results_dir, models):
    """tab:decision_tree — PCR-based mitigation decision tree."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{PCR-based mitigation decision tree with threshold-derived recommendations.}",
        r"\label{tab:decision_tree}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"PCR Range & Failure Mode & Recommended Mitigation \\",
        r"\midrule",
        r"$<$30\% & Outlier-as-safety & FP16 protect critical layer \\",
        r"30--70\% & Mixed/transitional & Per-channel critical + group-64 rest \\",
        r"$>$70\% & Outlier-crushes-safety & Group-64 all layers \\",
        r"$>$70\%, multi-layer & Multi-layer dilution & Protect top-$k$ layers at FP16 \\",
        r"All layers >10\% & Hyper-vulnerable & Full FP16 (no safe quantization) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _generate_protection_sweep_table(results_dir, model_name, label, caption, n_cols=5):
    """Helper: protection sweep for one model."""
    result = _try_load(results_dir, model_name, "protection_sweep.json")
    n_layers = N_LAYERS.get(model_name, 32)

    has_flips_col = n_cols == 5
    col_spec = "lcccc" if has_flips_col else "lccc"
    header = r"Layers Protected & Flip Rate & Recovery & Mem.\ Overhead"
    if has_flips_col:
        header += r" & Flips"
    header += r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    if result:
        # Unprotected baseline
        unprot = result.summary.get("unprotected", {})
        unprot_fr = unprot.get("flip_rate", 0)
        bl_refusals = result.summary.get("baseline", {}).get("refusal_count", 0)

        row = f"None (all quantized) & {_pct(unprot_fr)} & - & 0\\%"
        if has_flips_col:
            fc = unprot.get("flip_count", 0)
            row += f" & {fc}/{bl_refusals}"
        lines.append(row + r" \\")

        # Protection configs
        for key in sorted(result.summary.keys()):
            if not key.startswith("protect_"):
                continue
            val = result.summary[key]
            fr = val.get("flip_rate")
            if fr is None:
                continue

            # Parse layer list from key
            layers = key.replace("protect_", "").replace("_", ", ").replace("L", "L")
            n_protected = key.count("L")
            overhead = f"{n_protected/n_layers:.0%}" if n_layers > 0 else "-"

            recovery = "-"
            if unprot_fr > 0:
                rec = 1.0 - (fr / unprot_fr)
                recovery = _pct(rec)

            row = f"{layers} & {_pct(fr)} & {recovery} & {overhead}"
            if has_flips_col:
                fc = val.get("flip_count", 0)
                row += f" & {fc}/{bl_refusals}"
            lines.append(row + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_qwen_protection_sweep(results_dir, models):
    """tab:qwen_protection_sweep"""
    return _generate_protection_sweep_table(
        results_dir, "qwen", "tab:qwen_protection_sweep",
        "Qwen-2.5-7B FP16 protection sweep at 4-bit base quantization.", n_cols=5
    )


def generate_llama_protection_sweep(results_dir, models):
    """tab:llama_protection_sweep"""
    return _generate_protection_sweep_table(
        results_dir, "llama", "tab:llama_protection_sweep",
        "LLaMA-3.1-8B FP16 protection sweep at 4-bit base quantization.", n_cols=4
    )


def generate_phi35_protection_sweep(results_dir, models):
    """tab:phi35_protection_sweep"""
    return _generate_protection_sweep_table(
        results_dir, "phi35", "tab:phi35_protection_sweep",
        "Phi-3.5-mini FP16 protection sweep at 3-bit base quantization.", n_cols=4
    )


def generate_msmall_protection_sweep(results_dir, models):
    """tab:msmall_protection_sweep"""
    return _generate_protection_sweep_table(
        results_dir, "mistral-small", "tab:msmall_protection_sweep",
        "Mistral-Small-24B FP16 protection sweep at 3-bit base quantization.", n_cols=4
    )


def generate_protection_comparison(results_dir, models):
    """tab:protection_comparison — Protection sweep comparison across failure modes."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Protection sweep comparison across failure modes.}",
        r"\label{tab:protection_comparison}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Safety Pattern & Base Bits & Best FP16 Config & Recovery & Overhead & G64 Reduction \\",
        r"\midrule",
    ]

    patterns = {
        "qwen": "Concentrated-L0",
        "llama": "Distributed-early",
        "mistral-small": "Uniformly-diffuse",
        "phi35": "Hyper-vulnerable",
    }
    base_bits = {"qwen": 4, "llama": 4, "mistral-small": 3, "phi35": 3}

    for model_name in ["qwen", "llama", "mistral-small", "phi35"]:
        result = _try_load(results_dir, model_name, "protection_sweep.json")
        g64 = _try_load(results_dir, model_name, "group64.json")

        dn = _dn(model_name)
        pattern = patterns.get(model_name, "?")
        bits = base_bits.get(model_name, "?")

        best_config = best_recovery = overhead = "-"
        if result:
            unprot_fr = result.summary.get("unprotected", {}).get("flip_rate", 0)
            best_rec = -1
            for key, val in result.summary.items():
                if not key.startswith("protect_"):
                    continue
                fr = val.get("flip_rate")
                if fr is not None and unprot_fr > 0:
                    rec = 1.0 - (fr / unprot_fr)
                    if rec > best_rec:
                        best_rec = rec
                        best_config = key.replace("protect_", "").replace("_", ", ")
                        best_recovery = _pct(rec)
                        n_protected = key.count("L")
                        n_layers = N_LAYERS.get(model_name, 32)
                        overhead = f"{n_protected/n_layers:.0%}"

        g64_red = "-"
        if g64:
            val = g64.summary.get("group64_reduction")
            if val is not None:
                g64_red = _pct(val)

        lines.append(
            f"{dn} & {pattern} & {bits} & {best_config} & {best_recovery} & {overhead} & {g64_red}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_perchannel_mitigation(results_dir, models):
    """tab:perchannel_mitigation — Quantization granularity strategies."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantization granularity strategies for Qwen-2.5-7B at 4-bit and 3-bit.}",
        r"\label{tab:perchannel_mitigation}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Bits & Strategy & Flip Rate & vs.\ Baseline \\",
        r"\midrule",
    ]

    # Load mitigation results for qwen
    result = _try_load(results_dir, "qwen", "mitigation.json")
    if result is None:
        result = _try_load(results_dir, "qwen", "mitigation_4bit.json")
    if result is None:
        result = _try_load(results_dir, "qwen", "mitigation_3bit.json")
    if result:
        baseline_fr = result.summary.get("baseline", {}).get("flip_rate", 0)

        for key in ["per_tensor", "per_channel", "group_64", "fp16_critical",
                     "perchannel_critical_pertensor_rest", "perchannel_critical_group64_rest"]:
            val = result.summary.get(key, {})
            fr = val.get("flip_rate")
            if fr is None:
                continue
            bits = result.metadata.get("config", {}).get("bits", "?")
            diff = fr - baseline_fr if baseline_fr is not None and fr is not None else None
            diff_str = f"{diff:+.1%}" if diff is not None else "-"
            strategy_label = key.replace("_", " ").title()
            lines.append(f"{bits} & {strategy_label} & {_pct(fr)} & {diff_str}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_protocol_validation(results_dir, models):
    """tab:protocol_validation — End-to-end validation of diagnostic protocol."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{End-to-end validation of the diagnostic protocol across all eight models.}",
        r"\label{tab:protocol_validation}",
        r"\begin{tabular}{lcclccc}",
        r"\toprule",
        r"Model & PCR & Prescribed Fix & Baseline Flip & Mitigated Flip & Recovery & Mem.\ Overhead \\",
        r"\midrule",
    ]

    pcr_fixes = {
        "qwen": "FP16 protect L0",
        "mistral": "FP16 protect L0",
        "deepseek": "Group-64 all layers",
        "yi": "Group-64 all layers",
        "llama": "FP16 protect top-$k$",
        "gemma2": "Group-64 all layers",
        "mistral-small": "Multi-layer FP16",
        "phi35": "Full FP16",
    }

    for model_name in models:
        dn = _dn(model_name)
        ca = _try_load(results_dir, model_name, "channel_ablation.json")
        mit = _try_load(results_dir, model_name, "mitigation.json")
        if mit is None:
            mit = _try_load(results_dir, model_name, "mitigation_4bit.json")
        if mit is None:
            mit = _try_load(results_dir, model_name, "mitigation_3bit.json")
        prot = _try_load(results_dir, model_name, "protection_sweep.json")
        g64 = _try_load(results_dir, model_name, "group64.json")

        # Compute PCR
        pcr_val = "-"
        if ca:
            pt = ca.summary.get("per_tensor", {}).get("flip_rate", 0)
            pc = ca.summary.get("per_channel", {}).get("flip_rate", 0)
            if pt > 0:
                pcr_val = _pct(1 - pc / pt)

        fix = pcr_fixes.get(model_name, "-")

        # Baseline flip from whichever result is available
        baseline_flip = "-"
        mitigated_flip = "-"
        recovery = "-"
        overhead = "-"

        # Try mitigation results first, then protection sweep, then g64
        for src in [mit, prot, g64]:
            if src is None:
                continue
            bl = src.summary.get("baseline", {}).get("flip_rate")
            if bl is None:
                bl = src.summary.get("unprotected", {}).get("flip_rate")
            if bl is not None:
                baseline_flip = _pct(bl)
                break

        lines.append(
            f"{dn} & {pcr_val} & {fix} & {baseline_flip} & {mitigated_flip} & {recovery} & {overhead}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_protocol_vs_attention(results_dir, models):
    """tab:protocol_vs_attention — Attention-based vs causal layer selection."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Attention-based vs.\ causal layer selection for Qwen-2.5-7B at 4-bit.}",
        r"\label{tab:protocol_vs_attention}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Selection Method & Layers Protected & Flip Rate & Recovery \\",
        r"\midrule",
    ]

    # This table is populated from attention analysis + protection sweep
    result = _try_load(results_dir, "qwen", "protection_sweep.json")
    attn = _try_load(results_dir, "qwen", "attention_analysis.json")

    if result:
        unprot_fr = result.summary.get("unprotected", {}).get("flip_rate", 0)
        for key, val in sorted(result.summary.items()):
            if not key.startswith("protect_"):
                continue
            fr = val.get("flip_rate")
            if fr is None:
                continue
            layers = key.replace("protect_", "").replace("_", ", ")
            recovery = _pct(1.0 - fr / unprot_fr) if unprot_fr > 0 else "-"
            lines.append(f"Causal (ablation) & {layers} & {_pct(fr)} & {recovery}" + r" \\")

    if attn:
        # Attention-based entries
        for key, val in sorted(attn.summary.items()):
            if key.startswith("attention_"):
                fr = val.get("flip_rate")
                layers = key.replace("attention_", "")
                lines.append(f"Attention-based & {layers} & {_pct(fr)} & -" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# =============================================================================
# Appendix Tables
# =============================================================================

# =============================================================================
# Table Registry
# =============================================================================

TABLE_GENERATORS = {
    # §4 Results
    "custom_all_models": generate_custom_all_models,
    "advbench_main": generate_advbench_main,
    "multi_model_ppl": generate_multi_model_ppl,
    "std_metrics_mistral": generate_std_metrics_mistral,
    "mistral_mse_phase": generate_mistral_mse_phase,
    "phase_thresholds": generate_phase_thresholds,
    "cross_suite_compare": generate_cross_suite_compare,
    "mistral_seeds": generate_mistral_seeds,
    "multi_seeds": generate_multi_seeds,
    "mitigation_main": generate_mitigation_main,
    "specdecode_flip": generate_specdecode_flip,
    "72b_alignment": generate_72b_alignment,
    # §5 Mechanism
    "individual_layer_sensitivity": generate_individual_layer_sensitivity,
    "qwen_individual_all": generate_qwen_individual_all,
    "llama_individual_all": generate_llama_individual_all,
    "cross_model_layer_summary": generate_cross_model_layer_summary,
    "cumulative_qwen": generate_cumulative_qwen,
    "cumulative_deepseek": generate_cumulative_deepseek,
    "cumulative_mistral": generate_cumulative_mistral,
    "cumulative_yi": generate_cumulative_yi,
    "cumulative_phi35": generate_cumulative_phi35,
    "deepseek_asymmetry": generate_deepseek_asymmetry,
    "causal_vs_attention": generate_causal_vs_attention,
    "channel_ablation_all": generate_channel_ablation_all,
    "pcr_framework": generate_pcr_framework,
    "activation_stats": generate_activation_stats,
    "taxonomy": generate_taxonomy,
    # §6 Protocol
    "decision_tree": generate_decision_tree,
    "qwen_protection_sweep": generate_qwen_protection_sweep,
    "llama_protection_sweep": generate_llama_protection_sweep,
    "phi35_protection_sweep": generate_phi35_protection_sweep,
    "msmall_protection_sweep": generate_msmall_protection_sweep,
    "protection_comparison": generate_protection_comparison,
    "perchannel_mitigation": generate_perchannel_mitigation,
    "protocol_validation": generate_protocol_validation,
    "protocol_vs_attention": generate_protocol_vs_attention,
}

# Legacy aliases (from old 3-table version)
TABLE_GENERATORS["sweep"] = generate_custom_all_models
TABLE_GENERATORS["layer_sensitivity"] = generate_individual_layer_sensitivity
TABLE_GENERATORS["pcr"] = generate_pcr_framework


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment results")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--table", type=str, default="all",
                        help="Table name or 'all'. Use --list to see available tables.")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: auto-detect)")
    parser.add_argument("--list", action="store_true",
                        help="List all available table generators")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print(f"Available tables ({len(TABLE_GENERATORS)} generators):\n")
        sections = {
            "Results (§4)": ["custom_all_models", "advbench_main", "multi_model_ppl",
                             "std_metrics_mistral", "mistral_mse_phase", "phase_thresholds",
                             "cross_suite_compare", "mistral_seeds", "multi_seeds",
                             "mitigation_main", "specdecode_flip",
                             "72b_alignment"],
            "Mechanism (§5)": ["individual_layer_sensitivity", "qwen_individual_all",
                               "llama_individual_all", "cross_model_layer_summary",
                               "cumulative_qwen", "cumulative_deepseek", "cumulative_mistral",
                               "cumulative_yi", "cumulative_phi35", "deepseek_asymmetry",
                               "causal_vs_attention", "channel_ablation_all", "pcr_framework",
                               "activation_stats", "taxonomy"],
            "Protocol (§6)": ["decision_tree", "qwen_protection_sweep", "llama_protection_sweep",
                              "phi35_protection_sweep", "msmall_protection_sweep",
                              "protection_comparison", "perchannel_mitigation",
                              "protocol_validation", "protocol_vs_attention"],
        }
        for section, tables in sections.items():
            print(f"  {section}:")
            for t in tables:
                gen = TABLE_GENERATORS[t]
                print(f"    {t:40s} {gen.__doc__.strip().split(chr(10))[0] if gen.__doc__ else ''}")
            print()
        return

    results_dir = Path(args.results_dir)

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        # Auto-detect from results directory (accepts both canonical and pipeline names)
        _known_dirs = set(MODEL_REGISTRY.keys()) | set(_DIR_ALIASES.values())
        if results_dir.exists():
            models = []
            for d in sorted(results_dir.iterdir()):
                if not d.is_dir():
                    continue
                if d.name in MODEL_REGISTRY:
                    models.append(d.name)
                elif d.name in _DIR_ALIASES_REV:
                    models.append(_DIR_ALIASES_REV[d.name])  # qwen7b -> qwen
        else:
            models = ALL_MODELS

    if not models:
        models = ALL_MODELS

    # Determine which tables to generate
    if args.table == "all":
        # Exclude legacy aliases
        table_names = [k for k in TABLE_GENERATORS if k not in ("sweep", "layer_sensitivity", "pcr")]
    elif args.table in TABLE_GENERATORS:
        table_names = [args.table]
    else:
        print(f"Unknown table: {args.table}")
        print(f"Use --list to see available tables.")
        return

    output_lines = []
    for table_name in table_names:
        gen = TABLE_GENERATORS[table_name]
        latex = gen(results_dir, models)
        output_lines.append(f"\n% === tab:{table_name} ===")
        output_lines.append(latex)
        output_lines.append("")

    output = "\n".join(output_lines)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
        print(f"Written {len(table_names)} tables to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
