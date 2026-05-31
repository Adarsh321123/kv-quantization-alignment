# Experiment Guide

Each experiment lives in `experiments/` and follows the same two-phase pattern:
1. **Phase 1 (Generation):** Load model, load prompts, run quantized inference, store responses
2. **Phase 2 (Classification):** Unload generation model, load WildGuard classifier, classify all responses, save results

## Quick Reference

| Script | Experiment | Paper Section | GPU Time |
|--------|-----------|---------------|----------|
| `run_sweep.py` | Bit-width sweep | §4 (S1, S2, S4) | ~30min/model |
| `run_layer_ablation.py` | Per-layer sensitivity | §5 (M1) | ~2h/model |
| `run_cumulative.py` | Cumulative ablation | §5 (M2) | ~1h/model |
| `run_channel_ablation.py` | Channel-level ablation | §5 (M3) | ~30min |
| `run_group64.py` | Group-64 validation | §5 (M4) | ~15min |
| `run_activation_analysis.py` | Activation statistics | §5 (M5) | ~10min |
| `run_attention_analysis.py` | Attention capture | §5 | ~20min |
| `run_protection_sweep.py` | FP16 protection sweep | §6 (P1) | ~30min |
| `run_mitigation.py` | Mitigation strategies | §6 (P2, P3) | ~30min |
| `run_perplexity.py` | WikiText-103 PPL | §4 (S3) | ~20min/model |
| `run_determinism.py` | Seed reproducibility | §4 (D1) | ~15min |
| `run_spec_decoding.py` | Speculative decoding | §4 (X1) | ~1h |
| `run_ifeval.py` | IFEval benchmark | Appendix (B2) | ~30min |
| `run_pcr_prediction.py` | PCR predictive validation | §5 | ~45min |

## Common Options

All classification experiments accept:

| Flag | Default | Description |
|------|---------|-------------|
| `--classifier` | `wildguard` | Classifier: `wildguard` or `family_a` |
| `--classifier-device` | `auto` | Device for WildGuard model |
| `--prompts` | `custom` | Prompt set: `custom`, `advbench`, `harmbench`, `xstest` |
| `--max-new-tokens` | `256` | Max tokens to generate per response |
| `--batch-size` | `4` | Generation batch size |

## Models

Use canonical names with `--model`:

| Name | HuggingFace ID | Layers | Special |
|------|---------------|--------|---------|
| `qwen7b` | Qwen/Qwen2.5-7B-Instruct | 28 | |
| `mistral7b` | mistralai/Mistral-7B-Instruct-v0.2 | 32 | |
| `deepseek7b` | deepseek-ai/deepseek-llm-7b-chat | 30 | |
| `yi9b` | 01-ai/Yi-1.5-9B-Chat | 48 | Space-stripped refusal matching |
| `llama8b` | meta-llama/Llama-3.1-8B-Instruct | 32 | |
| `gemma9b` | google/gemma-2-9b-it | 42 | Sliding window attention |
| `msmall24b` | mistralai/Mistral-Small-24B-Instruct-2501 | 40 | |
| `phi35` | microsoft/Phi-3.5-mini-instruct | 32 | Fused QKV (qkv_proj) |
| `qwen72b` | Qwen/Qwen2.5-72B-Instruct | 80 | Multi-GPU |
| `yi34b` | 01-ai/Yi-1.5-34B-Chat | 60 | Multi-GPU |
| `mixtral` | mistralai/Mixtral-8x7B-Instruct-v0.1 | 32 | MoE, Multi-GPU |
| `olmo2` | allenai/OLMo-2-1124-7B-Instruct | 32 | Held-out model |

## Quantization

Experiments use different quantizer presets depending on their paper section:

| Preset | Symmetric | Granularity | Default For | Paper Section |
|--------|-----------|-------------|-------------|---------------|
| `section4` | No (asymmetric) | per-token | §4/§6 scripts: sweep, perplexity, ifeval, determinism, spec_decoding, protection_sweep, mitigation | §4 (deployment), §6 (solutions) |
| `section5` | Yes (symmetric) | per-tensor | §5 scripts: layer_ablation, cumulative, channel_ablation, group64 | §5 (mechanism) |

**Why**: Per-tensor symmetric is the harshest regime, maximizing flip rate signal for mechanistic analysis. Per-token asymmetric matches HuggingFace production deployments for realistic impact assessment.

All mechanistic scripts accept `--quantizer-preset section4` for validation runs. The `run_pipeline.py` orchestrator automatically applies the correct preset.

## Testing with --max-prompts

All scripts that load prompts accept `--max-prompts N` to limit the number of prompts for quick testing:

```bash
# Quick test: 5 prompts instead of 63
python experiments/run_sweep.py --model qwen --bits 16,4 --prompts custom \
  --max-prompts 5 --output /tmp/test.json

# Layer ablation with 3 prompts
python experiments/run_layer_ablation.py --model qwen --bits 3 --prompts custom \
  --max-prompts 3 --output /tmp/test_ablation.json
```

Note: `run_perplexity.py` uses WikiText-103 (not prompts) and does not accept `--max-prompts`.

## Benchmarks

| Name | Prompts | Categories |
|------|---------|------------|
| `custom` | 63 | 19 refusal + 21 privacy + 23 jailbreak |
| `advbench` | 520 | harmful behaviors |
| `harmbench` | 320 | harmful behaviors |
| `xstest` | 450 | 250 safe (over-refusal test) + 200 unsafe |
| `ifeval` | 541 | instruction-following |

## Experiment Details

### Bit-Width Sweep (S1, S2, S4)

```bash
# Custom benchmark (63 prompts)
python experiments/run_sweep.py \
  --model qwen --bits 16,8,6,5,4,3,2 \
  --prompts custom --output results/qwen/sweep_custom.json

# AdvBench (520 prompts)
python experiments/run_sweep.py \
  --model qwen --bits 16,8,4,3 \
  --prompts advbench --output results/qwen/sweep_advbench.json

# XSTest (450 prompts)
python experiments/run_sweep.py \
  --model qwen --bits 16,8,4,3 \
  --prompts xstest --output results/qwen/sweep_xstest.json
```

### Per-Layer Sensitivity (M1)

Quantizes one layer at a time to find critical safety layers.

```bash
python experiments/run_layer_ablation.py \
  --model qwen --bits 3 --prompts custom \
  --output results/qwen/layer_ablation.json
```

### Cumulative Ablation (M2)

Quantizes first-k or last-k layers cumulatively.

```bash
python experiments/run_cumulative.py \
  --model qwen --bits 3 --direction both \
  --prompts custom --output results/qwen/cumulative.json
```

### Channel Ablation (M3)

Tests channel subsets at a specific layer.

```bash
python experiments/run_channel_ablation.py \
  --model qwen --bits 3 --layer 0 \
  --prompts custom --output results/qwen/channel_ablation.json
```

### Group-64 Validation (M4)

Compares per-tensor vs group-64 quantization across all layers.

```bash
python experiments/run_group64.py \
  --model qwen --bits 3 --prompts custom \
  --output results/qwen/group64.json
```

### Activation Analysis (M5)

Captures K-projection activations without quantization.

```bash
python experiments/run_activation_analysis.py \
  --model qwen --layers 0 --prompts custom \
  --output results/qwen/activations.json
```

### FP16 Protection Sweep (P1)

Tests selective FP16 retention at critical layers.

```bash
python experiments/run_protection_sweep.py \
  --model qwen --base-bits 4 \
  --protect-layers "0" "0,1" "0,1,2" \
  --prompts custom --output results/qwen/protection.json
```

### Mitigation Strategies (P2, P3)

Compares per-tensor, per-channel, group-64, and FP16 protection.

```bash
python experiments/run_mitigation.py \
  --model qwen --bits 3 \
  --strategies per_tensor,per_channel,group_64,fp16_critical \
  --prompts custom --output results/qwen/mitigation.json
```

### Perplexity (S3)

WikiText-103 perplexity under quantization.

```bash
python experiments/run_perplexity.py \
  --model qwen --bits 16,8,6,5,4,3,2 \
  --output results/qwen/perplexity.json
```

### Determinism Check (D1)

Verifies identical outputs across seeds under greedy decoding.

```bash
python experiments/run_determinism.py \
  --model qwen --bits 4 --seeds 42,123,456 \
  --prompts custom --output results/qwen/determinism.json
```

### Speculative Decoding (X1)

```bash
python experiments/run_spec_decoding.py \
  --target-model qwen \
  --draft-model Qwen/Qwen2.5-0.5B-Instruct \
  --bits 16,8,4,3 --prompts custom \
  --output results/qwen/spec_decoding.json
```

### IFEval (B2)

Instruction-following evaluation (uses max_new_tokens=1024).

```bash
python experiments/run_ifeval.py \
  --model qwen --bits 16,8,4,3 \
  --output results/qwen/ifeval.json
```

### PCR Predictive Validation

Validates that PCR (Protection Coverage Ratio) computed on a small calibration set
(20 custom prompts) accurately predicts mitigation effectiveness on unseen test data
(200 AdvBench prompts).

```bash
python experiments/run_pcr_prediction.py \
  --model qwen --bits 4 \
  --output results/qwen/pcr_prediction.json
```

Options: `--calibration-size 20`, `--test-size 200`, `--calibration-prompts custom`,
`--test-prompts advbench`, `--critical-layer N` (auto-detected if omitted).

## Two-Phase Pattern

All classification experiments follow this flow to manage GPU memory:

```
Phase 1: Generate all responses
  - Load main model (7-24B)
  - For each condition: quantize → generate → store responses
  - Delete main model, free GPU memory

Phase 2: Classify all responses
  - Load WildGuard classifier (7B)
  - Batch classify all stored responses
  - Unload classifier
  - Save results
```

This ensures the main model and WildGuard never compete for GPU memory.

## Full Pipeline

Run all experiments for a model:

```bash
python run_pipeline.py --model qwen7b --experiments sweep layer_ablation cumulative --output-dir results/
```

Run specific experiments across models:

```bash
python run_pipeline.py \
  --model qwen7b llama8b mistral7b \
  --experiments sweep layer_ablation cumulative \
  --output-dir results/
```

## Output Format

All experiments output unified JSON. See `core/results.py` for the schema. Key fields:

```json
{
  "metadata": {"model": "...", "experiment": "...", "config": {...}},
  "prompts": [
    {
      "prompt_idx": 0,
      "prompt_text": "...",
      "category": "refusal",
      "conditions": {
        "16": {"response": "...", "refused": true, "classifier": "wildguard"},
        "4":  {"response": "...", "refused": false, "classifier": "wildguard", "kv_mse": 0.034}
      }
    }
  ],
  "summary": {
    "16": {"refusal_rate": 0.698, "refusal_count": 44},
    "4":  {"refusal_rate": 0.127, "flip_rate": 0.818, "flip_rate_ci": {"lower": 0.68, "upper": 0.90}}
  }
}
```

## Paper Table-to-Command Mapping

The following table maps key paper tables to `generate_tables.py` labels and
the experiments that produce the underlying data.  Run a table with:

```bash
python analysis/generate_tables.py --results-dir results/ --table <label>
```

Use `--list` to see all available labels (38 generators plus legacy aliases).

### Main text tables

| Paper Label | `generate_tables.py` label | Data produced by | Result file(s) |
|---|---|---|---|
| Table 1: Custom benchmark (all models) | `custom_all_models` | `run_sweep.py --prompts custom` | `sweep_custom.json` |
| Table 2: AdvBench results | `advbench_main` | `run_sweep.py --prompts advbench` | `sweep_advbench.json` |
| Table 3: Perplexity under quantization | `multi_model_ppl` | `run_perplexity.py` | `perplexity.json` |
| Table 4: Phase-transition thresholds | `phase_thresholds` | `run_sweep.py` (custom or advbench) | `sweep_custom.json` / `sweep_advbench.json` |
| Taxonomy of failure modes | `taxonomy` | `run_channel_ablation.py` (PCR analysis) | `channel_ablation.json` |
| Individual layer sensitivity | `individual_layer_sensitivity` | `run_layer_ablation.py` | `layer_ablation.json` |
| Channel ablation (all models) | `channel_ablation_all` | `run_channel_ablation.py` | `channel_ablation.json` |
| PCR framework + Group-64 | `pcr_framework` | `run_channel_ablation.py` + `run_group64.py` | `channel_ablation.json` + `group64.json` |
| Mitigation (mixed precision) | `mitigation_main` | `run_mitigation.py` | `mitigation.json` / `mitigation_Nbit.json` |
| Decision tree | `decision_tree` | (static -- no experiment data needed) | -- |
| Protocol validation | `protocol_validation` | `run_channel_ablation.py` + `run_mitigation.py` + `run_protection_sweep.py` + `run_group64.py` | multiple |

### Key appendix tables

| Paper Label | `generate_tables.py` label | Data produced by | Result file(s) |
|---|---|---|---|
| Mistral seed reproducibility | `mistral_seeds` | `run_determinism.py` | `determinism.json` |
| Multi-model seeds | `multi_seeds` | `run_determinism.py` | `determinism.json` |
| Speculative decoding flip | `specdecode_flip` | `run_spec_decoding.py` | `spec_decoding.json` |
| 72B alignment | `72b_alignment` | `run_sweep.py --model qwen-72b` | `sweep_custom.json` |
| Cross-suite comparison | `cross_suite_compare` | `run_sweep.py` (custom + advbench) | `sweep_custom.json` + `sweep_advbench.json` |
| Cumulative ablation (per model) | `cumulative_qwen`, `cumulative_mistral`, ... | `run_cumulative.py` | `cumulative.json` |
| Protection sweeps (per model) | `qwen_protection_sweep`, `llama_protection_sweep`, ... | `run_protection_sweep.py` | `protection_sweep.json` |

## Real-Dtype Validation (Table 22)

The paper's real-dtype validation table compares simulated quantization
against native low-precision storage formats (FP8, INT8, packed-INT4).
Those runs depended on a legacy runtime that is not part of this release;
the simulation framework used by every other experiment here produces
equivalent behavioral results, as discussed in the paper.

## Scheme Transfer (Section 4 vs Section 5 Quantizer Presets)

To validate that mechanistic findings (section 5, per-tensor symmetric) transfer to
deployment-realistic quantization (section 4, per-token asymmetric), run the layer
ablation experiment with the `section4` preset and compare layer rankings:

```bash
# Section 5 preset (default for mechanistic experiments)
python experiments/run_layer_ablation.py \
  --model qwen --bits 3 --quantizer-preset section5 \
  --prompts advbench --output results/qwen_layer_ablation_section5.json

# Section 4 preset (deployment-realistic)
python experiments/run_layer_ablation.py \
  --model qwen --bits 3 --quantizer-preset section4 \
  --prompts advbench --output results/qwen_layer_ablation_section4.json
```

Compare the layer rankings between the two outputs. The critical layers (highest
flip rate) should be consistent across presets, confirming that the mechanistic
analysis generalizes to deployment settings.

## Loading Legacy Results

Legacy format conversion is no longer shipped with this repository; the
pipeline writes results in the unified schema documented above.
