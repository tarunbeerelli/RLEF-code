"""
Unit tests for the updated multi-turn evaluation framework.
Ensures tracking payloads serialize correctly to destination files.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest
from rlef.data import APPSProblem
from rlef.eval import eval_apps
from rlef.trajectory import EpisodeStatus, Trajectory


@pytest.fixture
def mock_apps_problems_dataset():
    """Generates a list containing a single mock APPS problem for unit test runtime isolation."""
    prob = MagicMock(spec=APPSProblem)
    prob.problem_id = 101
    prob.difficulty = "introductory"
    prob.inputs = [["1", "2"]]
    prob.outputs = [["3"]]
    prob.prompt = "Write a function to sum two inputs."
    return [prob]


@patch("rlef.eval.run_agent_trajectory")
@patch("rlef.eval.load_apps_split")
def test_eval_apps_runs_trajectory_and_saves_json(
    mock_load_apps, mock_run_trajectory, mock_apps_problems_dataset, tmp_path
):
    """
    Asserts that eval_apps executes the full stateful agent loop
    and writes out summary telemetry dictionaries safely.
    """
    # 1. Setup environmental dataset controls
    mock_load_apps.return_value = mock_apps_problems_dataset

    mock_traj = MagicMock(spec=Trajectory)
    mock_traj.status = EpisodeStatus.SOLVED
    mock_traj.final_reward = 1.0
    mock_traj.current_turn = 2
    mock_traj.tool_usage = {"generate_tests": 1, "execute": 1}
    mock_traj.error_types = []
    mock_run_trajectory.return_value = mock_traj

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # 2. Intercept JSON writing by mocking file operations directly to avoid directory system collisions
    m_open = mock_open()
    with patch("rlef.eval.open", m_open, create=True), patch("rlef.eval.Path.mkdir"):
        summary = eval_apps(
            model=mock_model,
            tokenizer=mock_tokenizer,
            data_dir="mock/dir",
            tag="smoke_test",
            max_examples=1,
            device="cpu",
            max_turns=2,
        )

    # 3. Structural verification assertions
    assert summary["total"] == 1
    assert summary["solved"] == 1
    assert summary["pass_at_1"] == 1.0
    assert "introductory" in summary["by_difficulty"]

    # Confirm the JSON dumper was called to write evaluation output logs
    assert m_open.called
