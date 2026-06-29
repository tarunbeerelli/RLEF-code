"""
Tests for reward functions.

Execution reward tests are marked integration (need E2B).
Shaping and credit assignment tests run locally.
"""

import pytest
from rlef.reward import (
    assign_step_credit,
    execution_reward,
    shape_reward,
)
from rlef.tools import ToolName, ToolResult

# ── Shaping tests (no external deps) ─────────────────────────────────────────


def test_shape_reward_boundaries():
    assert shape_reward(0.0) == pytest.approx(0.0, abs=1e-9)
    assert shape_reward(1.0) == pytest.approx(1.0, abs=1e-6)


def test_shape_reward_midpoint_above_half():
    # log shaping should give > 0.5 for input 0.5
    assert shape_reward(0.5) > 0.5


def test_shape_reward_monotone():
    values = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
    shaped = [shape_reward(v) for v in values]
    assert shaped == sorted(shaped)


def test_shape_reward_invalid_input():
    with pytest.raises(AssertionError):
        shape_reward(1.1)
    with pytest.raises(AssertionError):
        shape_reward(-0.1)


# ── Step credit tests (no external deps) ─────────────────────────────────────


def _make_result(
    tool: ToolName, success: bool, output: str = "", lint_errors=None
) -> ToolResult:
    return ToolResult(
        tool=tool,
        success=success,
        output=output,
        lint_errors=lint_errors or [],
    )


def test_trajectory_credit_all_equal():
    results = [
        _make_result(ToolName.LINT, True, "No lint errors."),
        _make_result(ToolName.EXECUTE, True, "stdout: 2"),
    ]
    credits = assign_step_credit(results, final_reward=0.8, credit_type="trajectory")
    assert all(c.credit == pytest.approx(0.8) for c in credits)


def test_step_credit_useful_steps_get_reward():
    results = [
        _make_result(ToolName.LINT, True, "2 lint error(s)", lint_errors=[{}, {}]),
        _make_result(ToolName.EXECUTE, True, "stdout: correct"),
    ]
    credits = assign_step_credit(results, final_reward=1.0, credit_type="step")
    assert credits[0].useful is True
    assert credits[0].credit == pytest.approx(1.0 * (0.9**0))
    assert credits[1].credit == pytest.approx(1.0 * (0.9**1))


def test_step_credit_redundant_lint_penalised():
    results = [
        _make_result(ToolName.LINT, True, "No lint errors.", lint_errors=[]),
    ]
    credits = assign_step_credit(results, final_reward=1.0, credit_type="step")
    assert credits[0].useful is False
    assert credits[0].credit == pytest.approx(-0.05)


def test_step_credit_sandbox_crash_penalised():
    results = [
        _make_result(ToolName.EXECUTE, False, "Sandbox error: connection refused"),
    ]
    credits = assign_step_credit(results, final_reward=0.5, credit_type="step")
    assert credits[0].useful is False
    assert credits[0].credit == pytest.approx(-0.05)


def test_step_credit_returns_correct_count():
    results = [_make_result(ToolName.TESTS, True) for _ in range(4)]
    credits = assign_step_credit(results, final_reward=0.6, credit_type="step")
    assert len(credits) == 4


# ── Execution reward tests (need E2B) ─────────────────────────────────────────


@pytest.mark.integration
def test_execution_reward_correct_solution():
    code = "a, b = map(int, input().split())\nprint(a + b)"
    result = execution_reward(
        code=code,
        inputs=["1 2", "3 4"],
        outputs=["3", "7"],
        shaped=False,
    )
    assert result.passed == 2
    assert result.pass_rate == pytest.approx(1.0)
    assert result.raw_reward == pytest.approx(1.0)


@pytest.mark.integration
def test_execution_reward_partial_credit():
    # correct for first input only
    code = "print(3)"
    result = execution_reward(
        code=code,
        inputs=["1 2", "3 4"],
        outputs=["3", "7"],
        shaped=False,
    )
    assert result.passed == 1
    assert result.pass_rate == pytest.approx(0.5)


@pytest.mark.integration
def test_execution_reward_with_ablation_modifications():
    # Verify our custom dense reward ablation tracking handles penalties
    code = "print(3)"
    result = execution_reward(
        code=code,
        inputs=["1 2", "3 4"],
        outputs=["3", "7"],
        current_turn=3,
        shaped=True,
        ablation_cfg={
            "use_lint_bonus": False,
            "use_step_credit": False,
            "use_multi_turn": True,  # Applies turn penalty: (3 - 1) * 0.05 = -0.10
            "use_log_reward": False,
        },
    )
    # raw pass_rate is 0.5, but turn penalty drops final reward below 0.5
    assert result.raw_reward == 0.5
    assert result.final_reward == pytest.approx(0.5 - 0.10)


@pytest.mark.integration
def test_execution_reward_shaped_higher_than_raw():
    code = "print(3)"
    result = execution_reward(
        code=code,
        inputs=["1 2", "3 4"],
        outputs=["3", "7"],
        shaped=True,
        ablation_cfg={
            "use_lint_bonus": False,
            "use_step_credit": False,
            "use_multi_turn": False,
            "use_log_reward": True,  # Force log shaping explicitly
        },
    )
    # Log shaped 0.5 should compress to a value higher than raw 0.5
    assert result.final_reward > result.raw_reward
