"""Prompt loading for all benchmarks. Single source of truth."""

import csv
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Prompt:
    text: str
    category: str       # "refusal", "privacy", "jailbreak", "advbench", "harmbench", "xstest_safe", "xstest_unsafe", "ifeval"
    source: str          # "custom", "advbench", "harmbench", "xstest", "ifeval"
    index: int           # Position in original dataset


# Path to consolidated prompt data
DATA_DIR = Path(__file__).parent.parent / "data"


def load_custom_prompts(data_dir: Optional[Path] = None) -> List[Prompt]:
    """Load custom benchmark prompts (refusal + privacy + jailbreak).
    Source: data/prompts/{refusal,privacy,jailbreak}.txt
    """
    if data_dir is None:
        data_dir = DATA_DIR / "prompts"

    prompts = []
    idx = 0
    for category in ["refusal", "privacy", "jailbreak"]:
        path = data_dir / f"{category}.txt"
        if not path.exists():
            print(f"[warning] Prompts file not found: {path}")
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(Prompt(
                    text=line,
                    category=category,
                    source="custom",
                    index=idx,
                ))
                idx += 1
    return prompts


def load_advbench(
    max_samples: int = 520,
    data_dir: Optional[Path] = None,
) -> List[Prompt]:
    """Load AdvBench prompts from CSV. Default 520 (all prompts)."""
    if data_dir is None:
        data_dir = DATA_DIR / "benchmarks"

    path = data_dir / "advbench.csv"
    if not path.exists():
        raise FileNotFoundError(f"AdvBench CSV not found: {path}")

    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            goal = row.get("goal") or row.get("Goal") or ""
            if goal:
                prompts.append(Prompt(
                    text=goal,
                    category="advbench",
                    source="advbench",
                    index=i,
                ))
            if len(prompts) >= max_samples:
                break
    return prompts


def load_harmbench(
    max_samples: int = 1000,
    data_dir: Optional[Path] = None,
) -> List[Prompt]:
    """Load HarmBench direct-request prompts."""
    if data_dir is None:
        data_dir = DATA_DIR / "benchmarks"

    path = data_dir / "harmbench.csv"
    if not path.exists():
        raise FileNotFoundError(f"HarmBench CSV not found: {path}")

    prompts = []
    candidate_cols = ["Behavior", "behavior", "prompt", "Prompt", "text", "Text"]
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return prompts
        col = next((c for c in candidate_cols if c in reader.fieldnames), None)
        if col is None:
            col = reader.fieldnames[0]
        # Restrict to the "standard" (direct-request) split if HarmBench
        # ships a FunctionalCategory column. The paper reports numbers on
        # this subset only; copyright/contextual rows are excluded.
        category_col = next(
            (c for c in ("FunctionalCategory", "functional_category", "category")
             if c in reader.fieldnames),
            None,
        )
        for i, row in enumerate(reader):
            if category_col is not None:
                cat = (row.get(category_col) or "").strip().lower()
                if cat and cat != "standard":
                    continue
            p = (row.get(col) or "").strip()
            if p:
                prompts.append(Prompt(
                    text=p,
                    category="harmbench",
                    source="harmbench",
                    index=i,
                ))
            if len(prompts) >= max_samples:
                break
    return prompts


def load_xstest(
    label: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> List[Prompt]:
    """Load XSTest prompts (Röttger et al., NAACL 2024).

    Args:
        label: Filter by label. None=all 450, "safe"=250 safe prompts,
               "unsafe"=200 unsafe contrast prompts.
        data_dir: Override data directory.

    Returns:
        List of Prompt objects with category "xstest_safe" or "xstest_unsafe".
    """
    if data_dir is None:
        data_dir = DATA_DIR / "benchmarks"

    path = data_dir / "xstest.csv"
    if not path.exists():
        raise FileNotFoundError(f"XSTest CSV not found: {path}")

    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row_label = row.get("label", "").strip()
            if label is not None and row_label != label:
                continue
            text = (row.get("prompt") or "").strip()
            if text:
                prompts.append(Prompt(
                    text=text,
                    category=f"xstest_{row_label}" if row_label else "xstest",
                    source="xstest",
                    index=i,
                ))
    return prompts


def load_ifeval() -> List[Prompt]:
    """Load IFEval prompts from HuggingFace (541 prompts)."""
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = []
    for i, item in enumerate(ds):
        prompts.append(Prompt(
            text=item["prompt"],
            category="ifeval",
            source="ifeval",
            index=i,
        ))
    return prompts


def load_prompts(name: str, **kwargs) -> List[Prompt]:
    """Factory: load prompt set by name.

    Available sets:
        custom   — 63 prompts (19 refusal + 21 privacy + 23 jailbreak)
        advbench — 520 AdvBench prompts (harmful behaviors)
        harmbench — 320 HarmBench direct-request prompts (official test split)
        xstest   — 450 XSTest prompts (250 safe + 200 unsafe), filterable via label kwarg
        ifeval   — 541 IFEval instruction-following prompts
    """
    loaders = {
        "custom": load_custom_prompts,
        "advbench": load_advbench,
        "harmbench": load_harmbench,
        "xstest": load_xstest,
        "ifeval": load_ifeval,
    }
    if name not in loaders:
        raise ValueError(f"Unknown prompt set: {name}. Available: {list(loaders.keys())}")
    return loaders[name](**kwargs)
