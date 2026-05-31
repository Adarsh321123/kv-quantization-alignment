# Verification

How to verify the pipeline works before committing to a multi-day GPU run.

## Pre-flight Test Suite

`tests/test_preflight.py` runs 10 structural tests that catch wiring bugs between `run_pipeline.py`, experiment scripts, and `generate_tables.py`. No GPU required. Runs in ~5 seconds (plus ~50 seconds for dry-run caching on first invocation).

```bash
python tests/test_preflight.py           # standalone
pytest tests/test_preflight.py -v        # pytest discovery
```

**Preflight must pass before any GPU run.** The tests guard against these 6 historical bugs:

| Bug | What broke | Test |
|-----|-----------|------|
| Perplexity raw dict | `run_perplexity.py` wrote a raw dict instead of `ExperimentResult` | test 8 (unified schema) |
| Activation raw dict | `run_activation_analysis.py` wrote a raw dict instead of `ExperimentResult` | test 8 (unified schema) |
| Directory name mismatch | Pipeline creates `qwen7b/` but `generate_tables.py` expected `qwen/` | test 5 (directory aliases) |
| VRAM attribute | `torch.cuda.get_device_properties(i).total_mem` doesn't exist (correct: `.total_memory`) | test 6 (VRAM detection) |
| Sweep filename | Pipeline writes `sweep_custom.json` but tables loaded `sweep.json` | test 4 (filename match) |
| Mitigation filename | Pipeline writes `mitigation_4bit.json` but tables loaded `mitigation.json` | test 4 (filename match) |

The full test list:

1. **Dry run succeeds** -- no import errors, crashes, or tracebacks
2. **No unresolved placeholders** -- all `CRITICAL_LAYER_PLACEHOLDER` etc. resolved
3. **Argparse contracts** -- every `--flag` the pipeline passes is accepted by the script
4. **Output filenames match tables** -- pipeline `--output` basenames match `_try_load()` calls
5. **Directory aliases complete** -- every pipeline short name has a `_DIR_ALIASES` entry
6. **VRAM detection** -- correct `.total_memory` attribute, runtime check if GPU available
7. **Quantizer presets** -- section4 experiments get section4, section5 get section5
8. **Unified schema** -- every `run_*.py` imports from `core.results` and calls `save_results()`
9. **Model routing** -- model-specific experiments routed to correct models only

## Smoke Test Procedure

The smoke test runs the full pipeline on a single model with minimal prompts.

**IMPORTANT:** Restrict to a single GPU to match `--gpu N` production usage:

```bash
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --model qwen7b --max-prompts 2 --output-dir results/smoke_test/
```

**Why single GPU?** With both GPUs visible, the classifier may silently load on GPU 1 while the generation model occupies GPU 0, masking OOM bugs that only appear when `--gpu` restricts subprocesses to a single device. The two-phase pattern (generate → unload → classify) must fit sequentially on one GPU.

**Why qwen7b?** It is the only model assigned all 15 experiment registry entries (14 distinct experiment types, with sweep running 4 benchmarks), producing 18 experiment runs. This exercises every code path: model loading, quantization hooks, response generation, WildGuard classification, result serialization, critical layer auto-detection, protection config generation, and mitigation strategies.

**Why `--max-prompts 2`?** Two prompts is the minimum needed to produce non-trivial flip rate statistics while keeping runtime to ~70 minutes. This tests real model loading, quantization, generation, classification, and serialization -- not just argument parsing.

### Verification steps after the smoke test

**1. Check all 18 JSON files exist:**

```bash
ls results/smoke_test/qwen7b/*.json | wc -l
# Expected: 18
```

The 18 files:
- `sweep_custom.json`, `sweep_advbench.json`, `sweep_harmbench.json`, `sweep_xstest.json`
- `perplexity.json`, `ifeval.json`, `determinism.json`
- `spec_decoding.json`
- `layer_ablation.json`, `cumulative.json`, `channel_ablation.json`, `group64.json`
- `activation_analysis.json`, `attention_analysis.json`
- `protection_sweep.json`
- `mitigation_4bit.json`, `mitigation_3bit.json`, `mitigation_8bit_protection.json`

**2. Validate all JSON files have ExperimentResult schema:**

```bash
python -c "
import json
from pathlib import Path
for f in sorted(Path('results/smoke_test/qwen7b').glob('*.json')):
    d = json.loads(f.read_text())
    assert 'metadata' in d and 'summary' in d, f'{f.name}: missing schema keys'
    print(f'{f.name}: OK ({len(d.get(\"prompts\", []))} prompts, {len(d[\"summary\"])} summary keys)')
"
```

**3. Generate tables from smoke test results:**

```bash
python analysis/generate_tables.py --results-dir results/smoke_test/ --table all --output tables/smoke_test/all_tables.tex
```

**4. Verify table output:**

~36 tables generated total. Roughly two-thirds should be populated for qwen (the remainder are model-specific tables for other models like LLaMA protection sweep, Phi-3.5 protection sweep, etc. -- these are expected to be empty in a single-model smoke test).

## Verification Record

Results from initial smoke test run:

- **Date:** during development
- **GPU:** single consumer 24 GB GPU
- **Runtime:** 4170s (69.5 min)
- **Experiments:** 18/18 passed
- **JSON outputs:** 18/18 valid schema
- **Tables:** ~36 generated, majority populated
- **Bug regressions:** 0/5

## When to Re-verify

**Full verification (preflight + smoke test)** should be re-run if:

- Any experiment script in `experiments/` is modified
- `run_pipeline.py` command construction or MODEL_REGISTRY is changed
- `generate_tables.py` loading logic or filename references are changed
- `core/results.py` schema is changed
- New experiments or models are added to MODEL_REGISTRY

**Preflight alone** is sufficient for changes to:

- Analysis scripts (other than `generate_tables.py` load logic)
- Documentation
- Paper LaTeX
- Non-pipeline code
