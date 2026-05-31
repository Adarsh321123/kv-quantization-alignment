# Alignment Collapse Under KV Cache Quantization

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)

Research codebase accompanying *Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation*. The pipeline tests 11 instruction-tuned models (3.8B--72B dense plus an 8x7B Mixture-of-Experts) across bit-width sweeps, mechanistic layer/channel ablation, and mitigation strategies (PCR-guided protection).

## Reproduce All Paper Results

One command produces every result needed for the paper across all standard models:

```bash
python run_pipeline.py --all --dry-run          # preview the plan (~109 experiment runs)
python run_pipeline.py --all                     # run everything
```

This executes the full set of experiments in dependency order (smallest model first), including:
- 4 benchmark sweeps per model (custom, advbench, harmbench, xstest)
- Perplexity, IFEval, determinism checks
- Layer ablation with automatic critical layer detection
- Channel ablation, cumulative ablation, group-64 validation
- Protection sweeps and mitigation strategies (model-specific)
- Speculative decoding (Qwen only)

If the machine crashes mid-run, resume from where it stopped:

```bash
python run_pipeline.py --all --resume
```

The pipeline automatically detects per-GPU VRAM and skips models that won't fit on a single GPU (e.g., Mistral-Small-24B on a 24GB GPU). Use `--model` to override:

```bash
python run_pipeline.py --model msmall24b --all    # warns but runs anyway
```

For a quick smoke test (5 prompts per experiment):

```bash
python run_pipeline.py --all --max-prompts 5
```

### Run a Single Model

```bash
python run_pipeline.py --model qwen7b --all --dry-run
python run_pipeline.py --model qwen7b --all
```

### Multi-GPU

On a multi-GPU machine, run parallel pipeline instances on separate GPUs using `--gpu`:

```bash
# Split models across two GPUs
python run_pipeline.py --model phi35 qwen7b mistral7b deepseek7b --gpu 0 --all &
python run_pipeline.py --model llama8b yi9b gemma9b              --gpu 1 --all &
```

Each `--gpu N` instance sets `CUDA_VISIBLE_DEVICES=N` for all subprocesses and writes a separate manifest file (`run_manifest_gpu0.json`, `run_manifest_gpu1.json`) so `--resume` works independently per GPU. Use `--model` to pass multiple models (space-separated short names or HF IDs).

Short names: `phi35`, `qwen7b`, `mistral7b`, `deepseek7b`, `llama8b`, `yi9b`, `gemma9b`, `msmall24b`, `qwen72b`, `yi34b`, `mixtral`, `olmo2`

### Run Specific Experiment Groups

```bash
python run_pipeline.py --all --experiments sweep              # 32 runs (4 benchmarks x 8 models)
python run_pipeline.py --all --experiments mechanistic        # layer/channel/cumulative/group64/activation/attention
python run_pipeline.py --all --experiments section6           # protection_sweep + mitigation + mitigation_8bit
python run_pipeline.py --all --experiments quality            # perplexity + ifeval
python run_pipeline.py --all --experiments validation         # determinism + spec_decoding
```

### Output Structure

```
results/run_YYYYMMDD_HHMMSS/
  run_manifest.json        # tracks completed/failed experiments for --resume
  phi35/
    sweep_custom.json, sweep_advbench.json, sweep_harmbench.json, sweep_xstest.json
    perplexity.json, ifeval.json, determinism.json
    layer_ablation.json, cumulative.json, channel_ablation.json, group64.json
    activation_analysis.json, protection_sweep.json
  qwen7b/
    (same as above, plus:)
    spec_decoding.json, attention_analysis.json
    mitigation_4bit.json, mitigation_3bit.json, mitigation_8bit_protection.json
  ...
```

## Setup

```bash
# Loose ranges (works with most recent torch/transformers):
pip install -e .

# OR exact pins used to produce the paper numbers (recommended for
# reviewers reproducing reported results):
pip install -r requirements.txt && pip install -e . --no-deps
```

**Hardware.** Models up to ~24B parameters fit on a single NVIDIA RTX 3090 (24 GB) in fp16/bf16 with the default settings; 24B--47B models were run on NVIDIA A100 (80 GB). The 34B / 72B / Mixtral entries in the registry require either a single larger-memory GPU or sharding across multiple GPUs (`device_map="auto"`) and are excluded from `--all` (see below). The optional vLLM
deployment script in
[experiments/run_vllm_deployment.py](experiments/run_vllm_deployment.py)
requires a GPU/runtime combination that supports FP8 KV cache.

**Disk and time budget.** Downloading every model in the registry
requires roughly 150 GB of free space (mostly Mixtral, Qwen-72B, and
Yi-34B). Reproducing every paper number end-to-end is on the order of
several tens of GPU-hours on a recent data-center GPU; the standard
models alone account for the bulk of that time, with the 34B/72B/MoE
models and the appendix experiments adding the remainder.

**Software.** Python 3.10 + a recent CUDA toolchain. Tested with the
exact versions in [requirements.txt](requirements.txt). The optional
vLLM deployment script additionally requires `pip install vllm` (kept
out of the core requirements file because it pulls in a heavy
GPU-runtime build).

**HuggingFace access.** Several models and the WildGuard / Llama-Guard-3
classifiers are gated and require a HuggingFace account with accepted
licenses and `huggingface-cli login` configured before running:

- Gated models: `meta-llama/Llama-3.1-8B-Instruct`,
  `google/gemma-2-9b-it`, `01-ai/Yi-1.5-9B-Chat`,
  `mistralai/Mistral-Small-24B-Instruct-2501`,
  `mistralai/Mixtral-8x7B-Instruct-v0.1`.
- Gated safety classifiers: `allenai/wildguard` (primary),
  `meta-llama/Llama-Guard-3-8B` (used by
  [analysis/run_second_classifier.py](analysis/run_second_classifier.py)
  for inter-rater agreement).

Ungated models (Phi-3.5, Qwen-2.5-7B/72B, Mistral-7B-Instruct-v0.2,
DeepSeek-7B-Chat) work without authentication.

## Models

| Model | Short Name | Layers | Parameters |
|-------|-----------|--------|------------|
| Phi-3.5-mini-instruct | `phi35` | 32 | 3.8B |
| Qwen-2.5-7B-Instruct | `qwen7b` | 28 | 7B |
| Mistral-7B-Instruct-v0.2 | `mistral7b` | 32 | 7B |
| DeepSeek-7B-Chat | `deepseek7b` | 30 | 7B |
| LLaMA-3.1-8B-Instruct | `llama8b` | 32 | 8B |
| Yi-1.5-9B-Chat | `yi9b` | 48 | 9B |
| Gemma-2-9B-IT | `gemma9b` | 42 | 9B |
| Mistral-Small-24B-Instruct | `msmall24b` | 40 | 24B |
| Yi-1.5-34B-Chat | `yi34b` | 60 | 34B (large/multi-GPU) |
| Qwen-2.5-72B-Instruct | `qwen72b` | 80 | 72B (large/multi-GPU) |
| Mixtral-8x7B-Instruct-v0.1 | `mixtral` | 32 | 47B MoE (large/multi-GPU) |
| OLMo-2-1124-7B-Instruct | `olmo2` | 32 | 7B (held-out) |

The 34B/72B/Mixtral/OLMo-2 models are excluded from `--all` (each needs a
single larger-memory GPU or sharding across multiple smaller GPUs).
Run them explicitly:

```bash
python run_pipeline.py --model qwen72b --all
python run_pipeline.py --model yi34b   --all
python run_pipeline.py --model mixtral --all
python run_pipeline.py --model olmo2   --all   # held-out model for PCR prediction
```

## Benchmarks

| Name | Prompts | Source |
|------|---------|--------|
| custom | 63 (19 refusal + 21 privacy + 23 jailbreak) | Hand-crafted |
| advbench | 520 | AdvBench (Zou et al., 2023) |
| harmbench | 320 | HarmBench (Mazeika et al., 2024), direct-request ("standard") subset only |
| xstest | 450 (250 safe + 200 unsafe) | XSTest (Rottger et al., 2024) |
| ifeval | 541 | IFEval (Zhou et al., 2023) |

[core/prompts.py::load_harmbench](core/prompts.py) restricts HarmBench
to rows whose `FunctionalCategory` is `"standard"` (the direct-request
split reported in the paper); copyright/contextual rows are skipped
automatically when the column is present.

## Running Individual Experiments

Each experiment can also be run standalone:

```bash
# Bit-width sweep
python experiments/run_sweep.py \
  --model Qwen/Qwen2.5-7B-Instruct --bits 16,8,6,5,4,3,2 \
  --prompts custom --output results/qwen_sweep.json

# Per-layer sensitivity scan
python experiments/run_layer_ablation.py \
  --model Qwen/Qwen2.5-7B-Instruct --bits 3 --prompts custom \
  --output results/qwen_layers.json

# Protection sweep with 8-bit protection (for tab:mitigation_main)
python experiments/run_protection_sweep.py \
  --model Qwen/Qwen2.5-7B-Instruct --base-bits 4 --protect-bits 8 \
  --protect-layers "0" "0,1" "0,1,2" --output results/qwen_prot8.json
```

For long-running experiments:

```bash
nohup python -u run_pipeline.py --all > pipeline.log 2>&1 &
tail -f pipeline.log

# Multi-GPU: run two instances in parallel
nohup python -u run_pipeline.py --model phi35 qwen7b mistral7b deepseek7b --gpu 0 --all > gpu0.log 2>&1 &
nohup python -u run_pipeline.py --model llama8b yi9b gemma9b              --gpu 1 --all > gpu1.log 2>&1 &
```

See [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md) for all experiments and CLI flags.

### New Experiments (Revision)

Six new scripts for revision experiments. Four are GPU experiments in `experiments/`, two are post-hoc analysis in `analysis/`.

| Script | Exp | What it does |
|--------|-----|--------------|
| `experiments/run_kivi.py` | 1 | KIVI-style quantization (per-channel K, per-group V) vs naive, side-by-side |
| `experiments/run_system_prompt.py` | 4 | Bit-width sweep with and without a safety system prompt |
| `experiments/run_temperature.py` | 5 | Sampling (temp=0.6, top-p=0.9) across seeds at collapse bit-width |
| `experiments/run_naive_baselines.py` | 9 | Three naive mitigation strategies vs protocol-prescribed fix |
| `analysis/run_divergence_analysis.py` | 6 | First-divergent-token position for flipped prompts (post-hoc, CSV output) |
| `analysis/run_second_classifier.py` | 2 | Llama-Guard-3 reclassification of stratified sample, Cohen's kappa |

```bash
# KIVI vs naive quantization (Exp 1)
python experiments/run_kivi.py \
  --model Qwen/Qwen2.5-7B-Instruct --bits 4,2 \
  --prompts advbench --output results/qwen_kivi.json

# System prompt robustness (Exp 4) — use --dry-run to inspect formatted prompts
python experiments/run_system_prompt.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 --bits 16,4,3,2 \
  --prompts advbench --output results/mistral_sysprompt.json

# Temperature robustness (Exp 5)
python experiments/run_temperature.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 --bits 4 \
  --seeds 0,1,2 --output results/mistral_temp.json

# Naive baselines (Exp 9)
python experiments/run_naive_baselines.py \
  --model Qwen/Qwen2.5-7B-Instruct --bits 4 \
  --prompts advbench --output results/qwen_naive.json

# First-divergent-token analysis (Exp 6, post-hoc, no GPU needed)
python analysis/run_divergence_analysis.py \
  --fp16-result results/qwen_sweep.json \
  --quant-result results/qwen_sweep.json \
  --quant-condition 4 --output results/qwen_divergence.csv

# Second classifier comparison (Exp 2, needs Llama-Guard-3 access)
python analysis/run_second_classifier.py \
  --results results/mistral_sweep.json results/qwen_sweep.json \
  --conditions 16,4 16,4 --output results/classifier_comparison.csv
```

## Testing

Run the preflight test suite before any GPU run (~5 seconds, no GPU required):

```bash
python tests/test_preflight.py
```

This catches structural wiring bugs (schema mismatches, filename mismatches, directory aliases, VRAM attribute errors, argparse contract violations) that would cause silent failures after a multi-day GPU run. The argparse-contract test imports each experiment script and is skipped automatically if `transformers` is not installed in the current environment; install runtime dependencies first if you want full coverage.

For a full end-to-end smoke test (~70 min on a single GPU):

```bash
python run_pipeline.py --model qwen7b --max-prompts 2 --output-dir results/smoke_test/
```

Qwen-7B is the only model assigned all 19 experiment runs (15 registry entries), so it exercises every code path. See [docs/VERIFICATION.md](docs/VERIFICATION.md) for the full verification procedure and historical results.

## How the Pipeline Works

The pipeline executes experiments in three phases per model:

| Phase | Paper Section | Quantizer Preset | Experiments |
|-------|--------------|-------------------|-------------|
| A | 4 (deployment) | per-token asymmetric | sweep x4, perplexity, ifeval, determinism, spec_decoding |
| B | 5 (mechanistic) | per-tensor symmetric | layer_ablation, cumulative, channel_ablation, group64, activation_analysis, attention_analysis |
| C | 6 (mitigation) | per-token asymmetric | protection_sweep, mitigation, mitigation_8bit |
| Supplementary (standalone only) | Appendix | per-token asymmetric | kivi, system_prompt, temperature, naive_baselines |

> **Note:** The supplementary experiments (kivi, system_prompt, temperature, naive_baselines) are **not** included in `--all` runs. They must be invoked individually as standalone scripts; see [New Experiments (Revision)](#new-experiments-revision) above for usage.

After `layer_ablation` completes, the pipeline automatically detects critical layers and passes them to downstream experiments (channel ablation, protection sweep, mitigation). If layer ablation hasn't run, hardcoded defaults are used.

Each experiment runs as a **subprocess** for GPU memory isolation and crash safety. The manifest file (`run_manifest.json`, or `run_manifest_gpu{N}.json` with `--gpu`) records every completed experiment so `--resume` can skip past finished work.

## Directory Structure

```
alignment-collapse-kv-quantization/
  core/                          # Shared modules
    model_loader.py              #   Model registry + loading
    quantization.py              #   KVQuantizer, KVQuantizerDual (KIVI), FlexQuantizer
    generation.py                #   Batched response generation
    classifier.py                #   Refusal classification (WildGuard + LlamaGuard)
    metrics.py                   #   Flip rate, PPL, Wilson CI
    prompts.py                   #   Prompt loading (custom, advbench, harmbench, xstest, ifeval)
    results.py                   #   Unified result schema + I/O
  experiments/                   # Experiment scripts (one per experiment type)
  analysis/                      # Post-hoc analysis (no GPU needed)
    identify_critical_layers.py  #   Extract critical layers from ablation results
    compute_pcr.py               #   PCR from channel ablation
    cross_benchmark.py           #   Cross-benchmark comparison
    generate_tables.py           #   LaTeX table generation
    run_divergence_analysis.py   #   First-divergent-token analysis
    run_second_classifier.py     #   Llama-Guard reclassification
  tests/
    test_preflight.py            # Structural tests (no GPU, ~5s)
  data/
    prompts/                     # 63 custom prompts
    benchmarks/                  # advbench.csv, harmbench.csv, xstest.csv
  docs/                          # User-facing experiment guide and verification notes
  run_pipeline.py                # Unified pipeline orchestrator
```

## Citation

```bibtex
@article{xu2026alignment,
  title     = {Alignment Collapse Under {KV} Cache Quantization: Diagnosis and Mitigation},
  author    = {Xu, Bruce Changlong and Kumarappan, Adarsh and Zhou, Mu},
  journal   = {arXiv preprint arXiv:TODO},
  year      = {2026}
}
```

## License

This project is released under the MIT License.

