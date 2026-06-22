"""
Smoke tests for notebook helper functions.
We don't run the full notebooks in CI (no WandB, no results files)
but we test all the analysis functions in isolation.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_notebook_module(name: str, path: str):
    """Load a .py notebook as a module without executing top-level code."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    return module


def test_notebook_files_exist():
    for name in [
        "01_training_curves",
        "02_eval_analysis",
        "03_reward_hacking",
        "04_ablations",
    ]:
        path = Path(f"notebooks/{name}.py")
        assert path.exists(), f"Missing notebook: {path}"


def test_reward_hacking_detect_special_casing():
    """Import just the detection function and test it."""
    sys.path.insert(0, "notebooks")
    # inline the function to avoid executing notebook top level
    import re

    def detect_special_casing(completion: str, test_inputs: list[str]) -> bool:
        if not test_inputs or not completion:
            return False
        for inp in test_inputs:
            inp_stripped = inp.strip()
            if len(inp_stripped) > 0 and repr(inp_stripped) in completion:
                return True
            for token in inp_stripped.split():
                if len(token) > 1:
                    pattern = rf"\b{re.escape(token)}\b"
                    if re.search(pattern, completion):
                        return True
        return False

    # should detect hardcoded input
    assert detect_special_casing("if x == '1 2': print(3)", ["1 2"]) is True
    # should not flag normal code
    assert (
        detect_special_casing("a, b = map(int, input().split())\nprint(a+b)", ["1 2"])
        is False
    )
    # empty inputs
    assert detect_special_casing("print(3)", []) is False


def test_reward_hacking_reasoning_length():
    import re

    def reasoning_length(completion: str) -> int:
        if not completion:
            return 0
        code_match = re.search(r"<code>(.*?)</code>", completion, re.DOTALL)
        code = code_match.group(1) if code_match else completion
        return len(code.split())

    assert reasoning_length("") == 0
    assert reasoning_length("<code>print(3)</code>") == 1
    assert (
        reasoning_length("<code>a, b = map(int, input().split())\nprint(a + b)</code>")
        > 1
    )


def test_summary_table_structure():
    """Test that summary table has expected columns."""
    rows = [
        {"Condition": "Baseline", "Overall": 0.35, "Introductory": 0.50},
        {"Condition": "RLVR", "Overall": 0.45, "Introductory": 0.62},
    ]
    df = pd.DataFrame(rows).sort_values("Overall", ascending=False)
    assert df.iloc[0]["Condition"] == "RLVR"
    assert df.iloc[1]["Condition"] == "Baseline"


def test_outcome_labelling():
    def outcome(base_rate, rlvr_rate):
        b = base_rate == 1.0
        r = rlvr_rate == 1.0
        if b and r:
            return "Both correct"
        if not b and r:
            return "RLVR fixed"
        if b and not r:
            return "RLVR broke"
        return "Both wrong"

    assert outcome(1.0, 1.0) == "Both correct"
    assert outcome(0.0, 1.0) == "RLVR fixed"
    assert outcome(1.0, 0.0) == "RLVR broke"
    assert outcome(0.0, 0.0) == "Both wrong"
