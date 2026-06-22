"""Tests for trajectory management."""

import pytest
from rlef.data import APPSProblem
from rlef.prompt import parse_output
from rlef.reward import ExecutionResult
from rlef.tools import ToolName, ToolResult
from rlef.trajectory import EpisodeStatus, Trajectory


def _make_problem() -> APPSProblem:
    return APPSProblem(
        problem_id="0001",
        difficulty="introductory",
        question="Add two numbers.",
        inputs=["1 2"],
        outputs=["3"],
        fn_name=None,
        solutions=[],
        url="",
    )


def _make_exec_result(
    pass_rate: float, passed: int = 1, total: int = 1
) -> ExecutionResult:
    return ExecutionResult(
        passed=passed,
        total=total,
        pass_rate=pass_rate,
        raw_reward=pass_rate,
        final_reward=pass_rate,
        error_types=[],
    )


def _make_tool_result(tool: ToolName, output: str = "ok") -> ToolResult:
    return ToolResult(tool=tool, success=True, output=output)


def _make_parsed(tool: str = "execute", code: str = "print(3)") -> object:
    return parse_output(f"<tool>{tool}</tool>\n<code>\n{code}\n</code>")


def test_initial_state():
    traj = Trajectory(problem=_make_problem(), max_turns=3)
    assert traj.current_turn == 0
    assert traj.status == EpisodeStatus.RUNNING
    assert not traj.is_done


def test_solved_on_full_pass():
    traj = Trajectory(problem=_make_problem(), max_turns=3)
    traj.add_turn(
        _make_parsed("execute"),
        _make_tool_result(ToolName.EXECUTE),
        _make_exec_result(1.0),
    )
    assert traj.status == EpisodeStatus.SOLVED
    assert traj.is_done
    assert traj.final_reward == pytest.approx(1.0)


def test_partial_on_max_turns_with_reward():
    traj = Trajectory(problem=_make_problem(), max_turns=2)
    for _ in range(2):
        traj.add_turn(
            _make_parsed("execute"),
            _make_tool_result(ToolName.EXECUTE),
            _make_exec_result(0.5),
        )
    assert traj.status == EpisodeStatus.PARTIAL
    assert traj.final_reward == pytest.approx(0.5)


def test_failed_on_max_turns_zero_reward():
    traj = Trajectory(problem=_make_problem(), max_turns=2)
    for _ in range(2):
        traj.add_turn(
            _make_parsed("execute"),
            _make_tool_result(ToolName.EXECUTE),
            _make_exec_result(0.0),
        )
    assert traj.status == EpisodeStatus.FAILED
    assert traj.final_reward == pytest.approx(0.0)


def test_format_error_on_invalid_first_output():
    traj = Trajectory(problem=_make_problem(), max_turns=3)
    bad_parsed = parse_output("this is not valid format at all")
    traj.add_turn(
        bad_parsed,
        _make_tool_result(ToolName.EXECUTE, "Sandbox error"),
    )
    assert traj.status == EpisodeStatus.FORMAT_ERROR


def test_tool_usage_tracking():
    traj = Trajectory(problem=_make_problem(), max_turns=5)
    traj.add_turn(
        _make_parsed("lint"), _make_tool_result(ToolName.LINT, "No lint errors.")
    )
    traj.add_turn(
        _make_parsed("execute"),
        _make_tool_result(ToolName.EXECUTE),
        _make_exec_result(1.0),
    )
    assert traj.tool_usage["lint"] == 1
    assert traj.tool_usage["execute"] == 1


def test_history_grows_each_turn():
    traj = Trajectory(problem=_make_problem(), max_turns=3)
    traj.add_turn(
        _make_parsed("lint"), _make_tool_result(ToolName.LINT, "No lint errors.")
    )
    assert len(traj.history) == 2  # assistant + user feedback


def test_step_credits_assigned_on_done():
    traj = Trajectory(problem=_make_problem(), max_turns=3, credit_type="step")
    traj.add_turn(
        _make_parsed("execute"),
        _make_tool_result(ToolName.EXECUTE),
        _make_exec_result(1.0),
    )
    assert len(traj.step_credits) == 1


def test_log_dict_keys():
    traj = Trajectory(problem=_make_problem(), max_turns=3)
    traj.add_turn(
        _make_parsed("execute"),
        _make_tool_result(ToolName.EXECUTE),
        _make_exec_result(1.0),
    )
    log = traj.to_log_dict()
    assert "episode/status" in log
    assert "episode/final_reward" in log
    assert "tools/execute" in log
