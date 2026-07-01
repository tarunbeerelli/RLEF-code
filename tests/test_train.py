"""
Unit tests for the GRPO training configuration and trajectory replay logic.
Validates stratification, data loading preparation, and rewards processing.

Run with:
    poetry run pytest tests/test_train.py -v
"""

import pytest

# Core protection block: Skip the entire module if CUDA is missing (Mac environment)
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

if not HAS_CUDA:
    pytest.skip(
        "Skipping GPU-dependent training tests on local non-CUDA environment.",
        allow_module_level=True,
    )
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np

# Guard against cluster-only training dependencies on local/CI environments
if importlib.util.find_spec("trl") is None:
    pytest.skip(
        "Skipping training tests: cluster-only dependencies (trl) are missing or unsupported locally.",
        allow_module_level=True,
    )

# Your existing imports continue below...
from rlef.data import APPSProblem
from rlef.train import make_reward_fn, prepare_dataset, stratified_sample


@pytest.fixture
def mock_apps_problems():
    """Generates a collection of mock problems split across difficulty tiers."""
    p1 = MagicMock(spec=APPSProblem)
    p1.problem_id = 1
    p1.difficulty = "introductory"
    p1.question = "Solve A"
    p1.inputs = ["1"]
    p1.outputs = ["2"]
    p1.fn_name = "solve"

    p2 = MagicMock(spec=APPSProblem)
    p2.problem_id = 2
    p2.difficulty = "interview"
    p2.question = "Solve B"
    p2.inputs = ["3"]
    p2.outputs = ["4"]
    p2.fn_name = "solve_b"

    return [p1, p2]


# ── 1. Data Pipeline Validation Tests ─────────────────────────────────────────


@patch("rlef.train.difficulty_split")
def test_stratified_sampling_balances_buckets(mock_split, mock_apps_problems):
    """Ensures sampling logic splits selections evenly across available difficulties."""
    mock_split.return_value = {
        "introductory": [mock_apps_problems[0]],
        "interview": [mock_apps_problems[1]],
    }

    sampled = stratified_sample(mock_apps_problems, n=2)
    assert len(sampled) == 2
    # Verify both unique problem items are included via balanced sampling
    ids = [p.problem_id for p in sampled]
    assert 1 in ids
    assert 2 in ids


def test_prepare_dataset_serializes_chat_tokens():
    """Validates dataset transformations process raw strings into training strings cleanly."""
    mock_prob = MagicMock(spec=APPSProblem)
    mock_prob.problem_id = 50
    mock_prob.difficulty = "introductory"
    mock_prob.question = "Write code."
    mock_prob.inputs = [1]
    mock_prob.outputs = [2]
    mock_prob.fn_name = "run"

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = (
        "<system>Prompt</system><user>Write code.</user>"
    )
    # Mock encoding length maps safely
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
    mock_tokenizer.decode.return_value = (
        "<system>Prompt</system><user>Write code.</user>"
    )

    dataset = prepare_dataset([mock_prob], mock_tokenizer, max_prompt_length=256)

    assert len(dataset) == 1
    assert dataset[0]["problem_id"] == 50
    assert "prompt" in dataset[0]
    assert dataset[0]["fn_name"] == "run"


# ── 2. Reward & Advantage Clipping Tests ──────────────────────────────────────


@patch("rlef.train.normalize_batch_rewards")
@patch("rlef.train.parse_output")
def test_reward_function_clamps_outlier_advantages(mock_parse, mock_norm):
    """Verifies optimization loops clip outlier adjustments to protect stability bounds."""
    cfg = {"max_turns": 1, "ablation": {"use_normalization": True}}

    # Simulate a raw advantage normalization sequence containing statistical spikes
    mock_norm.return_value = np.array([0.5, -4.2, 3.8])

    mock_parsed = MagicMock()
    mock_parsed.is_valid = False  # Triggers a fallback format exception boundary path
    mock_parse.return_value = mock_parsed

    reward_fn = make_reward_fn(cfg)
    processed_rewards = reward_fn(completions=["out1", "out2", "out3"])

    # Assert output values match the explicit absolute outer clip boundary threshold (+/- 2.0)
    assert processed_rewards[0] == 0.5
    assert processed_rewards[1] == -2.0  # Scaled safely to lower floor cap
    assert processed_rewards[2] == 2.0  # Scaled safely to upper ceiling cap
