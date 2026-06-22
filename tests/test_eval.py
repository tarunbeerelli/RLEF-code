"""
Tests for eval and ablation modules.
Actual model inference is skipped locally — we test the
plumbing: summary computation, config generation, result parsing.
"""

import pytest
from rlef.ablations import ABLATIONS, generate_ablation_configs
from rlef.eval import _compute_summary

# ── Summary tests ─────────────────────────────────────────────────────────────


def test_compute_summary_perfect():
    results = [
        {"difficulty": "introductory", "pass_rate": 1.0},
        {"difficulty": "introductory", "pass_rate": 1.0},
    ]
    s = _compute_summary(results, tag="test", benchmark="apps")
    assert s["pass_at_1"] == pytest.approx(1.0)
    assert s["solved"] == 2
    assert s["total"] == 2


def test_compute_summary_partial():
    results = [
        {"difficulty": "introductory", "pass_rate": 1.0},
        {"difficulty": "interview", "pass_rate": 0.0},
    ]
    s = _compute_summary(results, tag="test", benchmark="apps")
    assert s["pass_at_1"] == pytest.approx(0.5)
    assert s["by_difficulty"]["introductory"]["pass_at_1"] == pytest.approx(1.0)
    assert s["by_difficulty"]["interview"]["pass_at_1"] == pytest.approx(0.0)


def test_compute_summary_empty():
    s = _compute_summary([], tag="test", benchmark="apps")
    assert s["pass_at_1"] == pytest.approx(0.0)
    assert s["total"] == 0


def test_compute_summary_by_difficulty_counts():
    results = [
        {"difficulty": "introductory", "pass_rate": 1.0},
        {"difficulty": "introductory", "pass_rate": 0.0},
        {"difficulty": "interview", "pass_rate": 1.0},
    ]
    s = _compute_summary(results, tag="test", benchmark="apps")
    assert s["by_difficulty"]["introductory"]["total"] == 2
    assert s["by_difficulty"]["introductory"]["solved"] == 1
    assert s["by_difficulty"]["interview"]["total"] == 1


# ── Ablation config tests ─────────────────────────────────────────────────────


def test_generate_ablation_configs(tmp_path):
    base_cfg = {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "reward_type": "continuous",
        "shaped": True,
        "credit_type": "step",
        "max_turns": 1,
        "output_dir": "./checkpoints",
        "wandb_project": "rlef-code",
    }
    paths = generate_ablation_configs(base_cfg, tmp_path)
    # base + one per ablation
    assert len(paths) == len(ABLATIONS) + 1


def test_ablation_configs_have_correct_overrides(tmp_path):
    import yaml

    base_cfg = {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "reward_type": "continuous",
        "shaped": True,
        "credit_type": "step",
        "max_turns": 1,
        "output_dir": "./checkpoints",
        "wandb_project": "rlef-code",
    }
    generate_ablation_configs(base_cfg, tmp_path)

    binary_path = tmp_path / "binary_reward.yaml"
    assert binary_path.exists()
    with open(binary_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["reward_type"] == "binary"
    assert cfg["shaped"] is False


def test_ablation_configs_dont_mutate_base(tmp_path):
    import yaml

    base_cfg = {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "reward_type": "continuous",
        "shaped": True,
        "credit_type": "step",
        "max_turns": 1,
        "output_dir": "./checkpoints",
        "wandb_project": "rlef-code",
    }
    generate_ablation_configs(base_cfg, tmp_path)

    # base config should be unchanged
    base_path = tmp_path / "base.yaml"
    with open(base_path) as f:
        saved = yaml.safe_load(f)
    assert saved["reward_type"] == "continuous"
    assert saved["shaped"] is True


def test_all_ablation_names_unique():
    names = [name for name, _ in ABLATIONS]
    assert len(names) == len(set(names))


def test_compute_summary_livecodebench():
    results = [
        {"difficulty": "easy", "pass_rate": 1.0},
        {"difficulty": "medium", "pass_rate": 0.0},
        {"difficulty": "hard", "pass_rate": 0.0},
    ]
    s = _compute_summary(results, tag="baseline", benchmark="livecodebench")
    assert s["benchmark"] == "livecodebench"
    assert s["pass_at_1"] == pytest.approx(1 / 3)
