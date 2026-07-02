"""
Unit tests for the standalone gRPC Remote Reward Server.
"""

from unittest.mock import MagicMock, patch

import pytest
from rlef import reward_pb2
from rlef.grpc_reward_server import HighSpeedRewardServicer


@pytest.mark.asyncio
async def test_grpc_servicer_returns_bonus_for_valid_xml():
    """Completions with <tool> tags receive a 0.05 lint bonus on top of pass_rate."""
    servicer = HighSpeedRewardServicer(use_lint_bonus=True)

    mock_result = MagicMock()
    mock_result.pass_rate = 1.0

    request = reward_pb2.RewardRequest(
        samples=[
            reward_pb2.TextSample(
                prompt="Write code.",
                completion="<tool>generate_tests</tool>\n```python\ndef solution(): pass\n```",
                metadata_json='{"inputs": ["1"], "outputs": ["1"], "difficulty": "introductory"}',
            )
        ]
    )

    with patch("rlef.grpc_reward_server.reward_func", return_value=mock_result):
        response = await servicer.EvaluateBatch(request, None)

    assert len(response.rewards) == 1
    assert response.rewards[0] == pytest.approx(1.05)


@pytest.mark.asyncio
async def test_grpc_servicer_handles_malformed_input_gracefully():
    """Completions with no code blocks return 0.0."""
    servicer = HighSpeedRewardServicer(use_lint_bonus=True)

    request = reward_pb2.RewardRequest(
        samples=[
            reward_pb2.TextSample(
                prompt="Write code.",
                completion="This text contains zero markdown python format blocks.",
                metadata_json='{"inputs": [], "outputs": [], "difficulty": "introductory"}',
            )
        ]
    )

    response = await servicer.EvaluateBatch(request, None)

    assert len(response.rewards) == 1
    assert response.rewards[0] == 0.0


@pytest.mark.asyncio
async def test_grpc_servicer_catches_sandbox_exceptions():
    """Execution exceptions return 0.0 when inputs are present."""
    servicer = HighSpeedRewardServicer(use_lint_bonus=False)

    request = reward_pb2.RewardRequest(
        samples=[
            reward_pb2.TextSample(
                prompt="Write code.",
                completion="```python\nprint('broken')\n```",
                metadata_json='{"inputs": ["1"], "outputs": ["1"], "difficulty": "introductory"}',
            )
        ]
    )

    with patch(
        "rlef.grpc_reward_server.reward_func",
        side_effect=RuntimeError("Sandbox crash"),
    ):
        response = await servicer.EvaluateBatch(request, None)

    assert len(response.rewards) == 1
    assert response.rewards[0] == 0.0
