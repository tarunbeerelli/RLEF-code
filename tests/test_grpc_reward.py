"""
Unit tests for the standalone gRPC Remote Reward Server.
Verifies Protocol Buffer serialization and sandbox execution routing.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from rlef import reward_pb2

from src.rlef.grpc_reward_server import HighSpeedRewardServicer


@pytest.mark.asyncio
async def test_grpc_servicer_returns_bonus_for_valid_xml():
    """Verifies that completions with valid tool brackets receive formatting bonuses."""
    servicer = HighSpeedRewardServicer(use_lint_bonus=True)

    # Mock out execution_reward to avoid launching real system subprocess sandboxes during unit tests
    mock_result = MagicMock()
    mock_result.pass_rate = 1.0

    request = reward_pb2.RewardRequest(
        samples=[
            reward_pb2.TextSample(
                prompt="Write code.",
                completion="<tool>generate_tests</tool> ```python\ndef solution(): pass\n``` </tool>",
            )
        ]
    )

    # Intercept run_in_executor to return our predictable mock payload
    with patch("asyncio.get_running_loop") as mock_loop:
        mock_executor = MagicMock()
        mock_executor.run_in_executor = asyncio.create_task
        mock_loop.return_value = mock_executor

        with patch(
            "src.rlef.grpc_reward_server.execution_reward", return_value=mock_result
        ):
            response = await servicer.EvaluateBatch(request, None)

    assert len(response.rewards) == 1
    # 1.0 (pass_rate) + 0.05 (xml tool lint bonus)
    assert response.rewards[0] == pytest.approx(1.05)


@pytest.mark.asyncio
async def test_grpc_servicer_handles_malformed_input_gracefully():
    """Ensures that code blocks missing markdown python identifiers safely fall back to 0.0 reward."""
    servicer = HighSpeedRewardServicer(use_lint_bonus=True)

    request = reward_pb2.RewardRequest(
        samples=[
            reward_pb2.TextSample(
                prompt="Write code.",
                completion="This text contains zero markdown python format blocks or code blocks.",
            )
        ]
    )

    response = await servicer.EvaluateBatch(request, None)

    assert len(response.rewards) == 1
    assert response.rewards[0] == 0.0


@pytest.mark.asyncio
async def test_grpc_servicer_catches_sandbox_exceptions():
    """Confirms that internal execution exceptions return 0.0 instead of killing the server thread."""
    servicer = HighSpeedRewardServicer(use_lint_bonus=False)

    request = reward_pb2.RewardRequest(
        samples=[
            reward_pb2.TextSample(
                prompt="Write code.", completion="```python\nprint('broken')\n```"
            )
        ]
    )

    # Force execution_reward to raise a system error
    with patch(
        "src.rlef.grpc_reward_server.execution_reward",
        side_effect=RuntimeError("Sandbox crash"),
    ):
        response = await servicer.EvaluateBatch(request, None)

    assert len(response.rewards) == 1
    assert response.rewards[0] == 0.0
