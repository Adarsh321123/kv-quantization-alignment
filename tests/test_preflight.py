#!/usr/bin/env python3
"""Preflight test suite for the pipeline.

Catches structural wiring bugs (schema mismatches, filename mismatches,
directory aliases, VRAM attribute errors) that would cause silent failures
after a week-long GPU run. No GPU required. Runs in ~5 seconds.

Bug coverage matrix:
  - run_perplexity.py / run_activation_analysis.py writing raw dicts  -> test 8
  - qwen7b/ vs qwen/ directory mismatch                              -> test 5
  - total_mem vs total_memory VRAM attribute                          -> test 6
  - sweep.json vs sweep_custom.json filename                          -> test 4
  - mitigation.json vs mitigation_4bit.json filename                  -> test 4
  - del model without gc.collect()/empty_cache() → OOM on single GPU  -> test 10

Usage:
    python tests/test_preflight.py           # standalone (~5s)
    pytest tests/test_preflight.py -v        # pytest discovery
"""

import re
import subprocess
import sys
import unittest
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_dry_run_output():
    """Run `python run_pipeline.py --all --dry-run` once, cache result."""
    return subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py"), "--all", "--dry-run"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=30,
    )


def _parse_commands(stdout):
    """Extract (run_key, command_string) pairs from dry-run output.

    Parses line pairs like:
      [  1] phi35/sweep_custom
            python /path/to/run_sweep.py --model ... --output ...
    """
    pairs = []
    lines = stdout.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"\s+\[\s*\d+\]\s+(\S+)", lines[i])
        if m:
            run_key = m.group(1)
            if i + 1 < len(lines):
                cmd_line = lines[i + 1].strip()
                # Match any python interpreter (python, python3, /usr/bin/python3,
                # virtualenv paths, etc.) by looking for the script path that
                # follows it.
                if re.match(r"\S*python[\d.]*\s+\S+", cmd_line):
                    pairs.append((run_key, cmd_line))
            i += 2
        else:
            i += 1
    return pairs


def _extract_flags_from_help(script_path):
    """Run `python script --help`, return set of accepted --flag names."""
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    return set(re.findall(r"(--[\w-]+)", result.stdout))


def _extract_try_load_filenames():
    """Extract all filenames from _try_load(..., \"X.json\") in generate_tables.py."""
    source = (ROOT / "analysis" / "generate_tables.py").read_text()
    return set(re.findall(r'_try_load\s*\([^,]+,\s*[^,]+,\s*"([^"]+)"', source))


def _run_key_to_experiment(run_key):
    """Map run_key like 'phi35/sweep_custom' to experiment base name."""
    _, exp_part = run_key.split("/", 1)
    if exp_part.startswith("sweep_"):
        return "sweep"
    if exp_part.startswith("mitigation_8bit"):
        return "mitigation_8bit"
    if re.match(r"mitigation_\d+bit", exp_part):
        return "mitigation"
    return exp_part


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreflight(unittest.TestCase):
    """Pre-flight checks for pipeline wiring."""

    # 1 ---------------------------------------------------------------
    def test_dry_run_succeeds(self):
        """Import errors, broken registry, any crash -> exit 0, no traceback."""
        result = _get_dry_run_output()
        self.assertEqual(
            result.returncode, 0,
            f"Dry-run failed (exit {result.returncode}):\n{result.stderr[-2000:]}",
        )
        self.assertNotIn(
            "Traceback", result.stderr,
            f"Traceback in dry-run stderr:\n{result.stderr[-2000:]}",
        )

    # 2 ---------------------------------------------------------------
    def test_no_unresolved_placeholders(self):
        """Placeholder typos or resolution failures -> zero *_PLACEHOLDER tokens."""
        result = _get_dry_run_output()
        self.assertEqual(result.returncode, 0)
        placeholders = re.findall(r"\w+_PLACEHOLDER\b", result.stdout)
        self.assertEqual(
            placeholders, [],
            f"Unresolved placeholders in dry-run output: {set(placeholders)}",
        )

    # 3 ---------------------------------------------------------------
    def test_argparse_contracts(self):
        """Every --flag in a dry-run command must be accepted by the script."""
        result = _get_dry_run_output()
        self.assertEqual(result.returncode, 0)
        commands = _parse_commands(result.stdout)
        self.assertGreater(len(commands), 0, "No commands parsed from dry-run")

        help_cache = {}
        mismatches = []

        for run_key, cmd_str in commands:
            parts = cmd_str.split()
            script = parts[1]  # parts[0] = python interpreter
            cmd_flags = {p for p in parts if p.startswith("--")}

            if script not in help_cache:
                help_cache[script] = _extract_flags_from_help(script)

            accepted = help_cache[script]
            # If `--help` returned no flags at all, the script could not be
            # imported (e.g. `transformers` not installed in the test
            # environment). Skip rather than report false mismatches.
            if not accepted:
                self.skipTest(
                    f"Cannot import {Path(script).name} (missing runtime "
                    "dependencies). Run `pip install -e .` first."
                )
            for flag in cmd_flags:
                if flag not in accepted:
                    mismatches.append(
                        f"  {run_key}: {flag} not in {Path(script).name}"
                    )

        self.assertEqual(
            mismatches, [],
            "Argparse contract violations:\n" + "\n".join(mismatches),
        )

    # 4 ---------------------------------------------------------------
    def test_output_filenames_match_tables(self):
        """Pipeline --output basenames must match generate_tables.py _try_load() calls."""
        result = _get_dry_run_output()
        self.assertEqual(result.returncode, 0)
        commands = _parse_commands(result.stdout)

        # Collect pipeline output basenames
        pipeline_basenames = set()
        for _, cmd_str in commands:
            parts = cmd_str.split()
            for i, p in enumerate(parts):
                if p == "--output" and i + 1 < len(parts):
                    pipeline_basenames.add(Path(parts[i + 1]).name)

        self.assertGreater(len(pipeline_basenames), 0, "No --output flags parsed")

        # Collect _try_load filenames from generate_tables.py
        table_filenames = _extract_try_load_filenames()
        self.assertGreater(len(table_filenames), 0, "No _try_load calls parsed")

        # Known exceptions: appendix-only files not produced by pipeline
        exceptions: set = set()

        # Every table filename must be in pipeline basenames or have its stem
        # as a substring of some pipeline basename (handles mitigation.json ->
        # mitigation_4bit.json fallback pattern)
        unmatched = []
        for fn in sorted(table_filenames - exceptions):
            if fn in pipeline_basenames:
                continue
            stem = fn.replace(".json", "")
            if any(stem in pb.replace(".json", "") for pb in pipeline_basenames):
                continue
            unmatched.append(fn)

        self.assertEqual(
            unmatched, [],
            f"Table generator references files with no pipeline match: {unmatched}",
        )

    # 5 ---------------------------------------------------------------
    def test_directory_aliases_complete(self):
        """_DIR_ALIASES values must cover all pipeline short names."""
        # Parse short names from run_pipeline.py MODEL_REGISTRY
        pipeline_src = (ROOT / "run_pipeline.py").read_text()
        short_names = set(re.findall(r'"short_name":\s*"(\w+)"', pipeline_src))
        self.assertGreater(len(short_names), 0, "No short_name entries parsed")

        # Parse _DIR_ALIASES from generate_tables.py
        tables_src = (ROOT / "analysis" / "generate_tables.py").read_text()
        alias_block = re.search(
            r"_DIR_ALIASES\s*=\s*\{([^}]+)\}", tables_src, re.DOTALL
        )
        self.assertIsNotNone(alias_block, "_DIR_ALIASES not found in generate_tables.py")

        alias_pairs = re.findall(
            r'"([\w-]+)"\s*:\s*"(\w+)"', alias_block.group(1)
        )
        alias_values = {v for _, v in alias_pairs}

        # Every pipeline short name must appear as an alias value
        uncovered = short_names - alias_values
        self.assertEqual(
            uncovered, set(),
            f"Pipeline short names missing from _DIR_ALIASES values: {uncovered}",
        )

        # Reverse mapping must be injective (no two canonical names -> same dir)
        self.assertEqual(
            len(alias_pairs), len(alias_values),
            "Duplicate values in _DIR_ALIASES — multiple canonical names map to "
            "the same pipeline directory",
        )

    # 6 ---------------------------------------------------------------
    def test_vram_detection(self):
        """Correct PyTorch attribute: .total_memory not .total_mem."""
        pipeline_src = (ROOT / "run_pipeline.py").read_text()

        # Static: must use .total_memory
        self.assertIn(
            ".total_memory", pipeline_src,
            "run_pipeline.py missing .total_memory attribute",
        )

        # Static: must NOT use truncated .total_mem
        bad = re.findall(r"\.total_mem\b(?!ory)", pipeline_src)
        self.assertEqual(
            bad, [],
            f"Found incorrect .total_mem (should be .total_memory): {bad}",
        )

        # Runtime: if CUDA available, function should return > 0
        try:
            import torch

            if torch.cuda.is_available():
                sys.path.insert(0, str(ROOT))
                from run_pipeline import get_vram_info

                info = get_vram_info()
                self.assertGreater(
                    info["max_single_gpu_gb"], 0.0,
                    "CUDA available but get_vram_info() returned 0 for max_single_gpu_gb",
                )
        except ImportError:
            pass  # No torch — static check is sufficient

    # 7 ---------------------------------------------------------------
    def test_quantizer_presets(self):
        """section4 experiments must use section4 preset; section5 must use section5."""
        result = _get_dry_run_output()
        self.assertEqual(result.returncode, 0)
        commands = _parse_commands(result.stdout)

        section4_exps = {"sweep", "spec_decoding"}
        section5_exps = {"layer_ablation", "cumulative", "channel_ablation", "group64"}

        mismatches = []
        for run_key, cmd_str in commands:
            exp = _run_key_to_experiment(run_key)
            parts = cmd_str.split()

            preset = None
            for i, p in enumerate(parts):
                if p == "--quantizer-preset" and i + 1 < len(parts):
                    preset = parts[i + 1]

            if exp in section4_exps:
                if preset != "section4":
                    mismatches.append(
                        f"  {run_key}: expected section4, got {preset}"
                    )
            elif exp in section5_exps:
                if preset != "section5":
                    mismatches.append(
                        f"  {run_key}: expected section5, got {preset}"
                    )

        self.assertEqual(
            mismatches, [],
            "Quantizer preset mismatches:\n" + "\n".join(mismatches),
        )

    # 8 ---------------------------------------------------------------
    def test_scripts_use_unified_schema(self):
        """Every run_*.py must import from core.results and call save_results()."""
        scripts = sorted((ROOT / "experiments").glob("run_*.py"))
        self.assertGreater(len(scripts), 0, "No experiment scripts found")

        missing_import = []
        missing_save = []

        for script in scripts:
            src = script.read_text()
            if "from core.results import" not in src:
                missing_import.append(script.name)
            if "save_results(" not in src:
                missing_save.append(script.name)

        self.assertEqual(
            missing_import, [],
            f"Scripts missing 'from core.results import': {missing_import}",
        )
        self.assertEqual(
            missing_save, [],
            f"Scripts missing save_results() call: {missing_save}",
        )

    # 9 ---------------------------------------------------------------
    def test_model_routing(self):
        """Model-specific experiments routed to correct models only."""
        sys.path.insert(0, str(ROOT))
        from run_pipeline import MODEL_REGISTRY

        # Build experiment -> set of short names
        exp_models = {}
        for hf_id, info in MODEL_REGISTRY.items():
            short = info["short_name"]
            for exp in info["experiments"]:
                exp_models.setdefault(exp, set()).add(short)

        # spec_decoding: only qwen7b
        self.assertEqual(
            exp_models.get("spec_decoding", set()), {"qwen7b"},
            f"spec_decoding routing: {exp_models.get('spec_decoding')}",
        )

        # attention_analysis: only qwen7b + llama8b
        self.assertEqual(
            exp_models.get("attention_analysis", set()), {"qwen7b", "llama8b"},
            f"attention_analysis routing: {exp_models.get('attention_analysis')}",
        )

        # gemma9b: no model-specific experiments
        gemma_exps = set()
        for hf_id, info in MODEL_REGISTRY.items():
            if info["short_name"] == "gemma9b":
                gemma_exps = set(info["experiments"].keys())
        model_specific = {
            "spec_decoding", "attention_analysis",
            "mitigation", "mitigation_8bit", "protection_sweep",
        }
        unexpected = gemma_exps & model_specific
        self.assertEqual(
            unexpected, set(),
            f"gemma9b has unexpected model-specific experiments: {unexpected}",
        )

        # All 8 standard models must have sweep
        standard = {
            info["short_name"]
            for info in MODEL_REGISTRY.values()
            if info["size_order"] < 99
        }
        sweep_models = exp_models.get("sweep", set())
        missing = standard - sweep_models
        self.assertEqual(
            missing, set(),
            f"Standard models missing sweep: {missing}",
        )


    # 10 --------------------------------------------------------------
    def test_model_unload_releases_memory(self):
        """Verify the unload path has gc.collect() + empty_cache() to prevent OOM on single-GPU runs.

        Historical bug: two-phase pattern (generate → unload → classify) OOMed when
        CUDA_VISIBLE_DEVICES restricted to one GPU because del model alone doesn't free
        PyTorch's CUDA cached memory. The unload must do del + gc.collect() + empty_cache().
        """
        # Check experiment scripts with two-phase unload
        experiment_scripts = sorted((ROOT / "experiments").glob("run_*.py"))
        self.assertGreater(len(experiment_scripts), 0, "No experiment scripts found")

        missing_gc = []
        missing_empty_cache = []

        for script in experiment_scripts:
            src = script.read_text()

            # Only check scripts that have an unload site (del model pattern)
            has_unload = ("del model" in src or "del target_model" in src)
            if not has_unload:
                continue

            # After del model, must have gc.collect() and empty_cache()
            if "gc.collect()" not in src:
                missing_gc.append(script.name)
            if "torch.cuda.empty_cache()" not in src:
                missing_empty_cache.append(script.name)

        self.assertEqual(
            missing_gc, [],
            f"Scripts with model unload missing gc.collect(): {missing_gc}",
        )
        self.assertEqual(
            missing_empty_cache, [],
            f"Scripts with model unload missing torch.cuda.empty_cache(): {missing_empty_cache}",
        )

        # Check classifier.py unload method
        classifier_src = (ROOT / "core" / "classifier.py").read_text()
        self.assertIn(
            "gc.collect()", classifier_src,
            "core/classifier.py unload() missing gc.collect()",
        )
        self.assertIn(
            "empty_cache()", classifier_src,
            "core/classifier.py unload() missing torch.cuda.empty_cache()",
        )


if __name__ == "__main__":
    unittest.main()
