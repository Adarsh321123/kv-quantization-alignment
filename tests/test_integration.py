#!/usr/bin/env python3
"""End-to-end integration tests for the unified pipeline.

These tests require GPU access and a loaded model. Run before GPU experiments:
    python tests/test_integration.py

Tests:
1. Sweep: 5 prompts, FP16 + 4-bit, keyword classifier
2. Layer ablation: 3 prompts, per-tensor symmetric, verify metadata
3. Output structure: JSON schema compliance
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_script(cmd, timeout=600):
    """Run a script and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def test_sweep_integration():
    """Test: sweep with 5 prompts, FP16 + 4-bit, keyword classifier."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    rc, stdout, stderr = run_script(
        f"python experiments/run_sweep.py "
        f"--model qwen --bits 16,4 --prompts custom --max-prompts 5 "
        f"--classifier family_a --output {output_path}"
    )

    if rc != 0:
        print(f"[FAIL] Sweep script failed:\n{stderr[-500:]}")
        return False

    with open(output_path) as f:
        data = json.load(f)

    # Structure checks
    assert "metadata" in data, "Missing metadata"
    assert "prompts" in data, "Missing prompts"
    assert "summary" in data, "Missing summary"
    assert len(data["prompts"]) == 5, f"Expected 5 prompts, got {len(data['prompts'])}"

    p0 = data["prompts"][0]
    assert "16" in p0["conditions"], "Missing FP16 condition"
    assert "4" in p0["conditions"], "Missing 4-bit condition"

    c16 = p0["conditions"]["16"]
    assert c16["classifier"] == "family_a", f"Wrong classifier: {c16['classifier']}"
    assert len(c16["response"]) > 0, "Empty FP16 response"
    assert "generation_time_s" in c16, "Missing generation_time_s"
    assert "input_token_count" in c16, "Missing input_token_count"

    # Summary checks
    assert "16" in data["summary"]
    assert "4" in data["summary"]
    s16 = data["summary"]["16"]
    s4 = data["summary"]["4"]
    assert "refusal_rate" in s16
    assert "refusal_rate" in s4

    Path(output_path).unlink(missing_ok=True)
    print(f"[PASS] Sweep integration: FP16 refusal={s16['refusal_rate']:.0%}, "
          f"4-bit flip={s4.get('flip_rate', 'N/A')}")
    return True


def test_layer_ablation_integration():
    """Test: layer ablation with 3 prompts, verify per-tensor symmetric."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    rc, stdout, stderr = run_script(
        f"python experiments/run_layer_ablation.py "
        f"--model qwen --bits 3 --prompts custom --max-prompts 3 "
        f"--classifier family_a --output {output_path}"
    )

    if rc != 0:
        print(f"[FAIL] Layer ablation script failed:\n{stderr[-500:]}")
        return False

    with open(output_path) as f:
        data = json.load(f)

    config = data["metadata"]["config"]
    assert config["quantizer"]["symmetric"] is True, \
        f"Expected symmetric=True, got {config['quantizer']['symmetric']}"
    assert config["quantizer"]["granularity"] == "per_tensor", \
        f"Expected per_tensor, got {config['quantizer']['granularity']}"

    p0 = data["prompts"][0]
    assert "layer_0" in p0["conditions"], "Missing layer_0 condition"
    assert "baseline" in p0["conditions"], "Missing baseline condition"

    Path(output_path).unlink(missing_ok=True)
    print(f"[PASS] Layer ablation integration: symmetric=True, per_tensor, "
          f"layer_0 flip={data['summary']['layer_0'].get('flip_rate', 'N/A')}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("End-to-End Integration Tests")
    print("=" * 60)

    results = []
    results.append(("sweep", test_sweep_integration()))
    results.append(("layer_ablation", test_layer_ablation_integration()))

    print("\n" + "=" * 60)
    all_pass = all(r for _, r in results)
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    if all_pass:
        print("\nAll integration tests passed")
    else:
        print("\nSome tests failed")
        sys.exit(1)
