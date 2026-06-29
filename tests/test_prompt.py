"""
Tests for the asymmetric prompt and XML parsing layer.

Validates that Turn 1 delivers the Test Oracle context, subsequent
turns scale down to programmatic sandbox permissions, and XML regex matches
decouple code safely.
"""

from unittest.mock import MagicMock

import pytest
from rlef.data import APPSProblem
from rlef.prompt import (
    SUBSEQUENT_TURNS_PROMPT,
    TURN_1_ORACLE_PROMPT,
    format_prompt,
    parse_output,
)


@pytest.fixture
def mock_apps_problem():
    """Generates an isolated APPS problem asset for contract testing."""
    prob = MagicMock(spec=APPSProblem)
    prob.question = "Write a function to verify if an integer is a palindrome."
    return prob


# ── 1. Forced Asymmetry Routing Tests ─────────────────────────────────────────


def test_format_prompt_initializes_turn_1_as_oracle(mock_apps_problem):
    """Ensures Turn 1 delivers the oracle configuration alongside the live question."""
    messages = format_prompt(problem=mock_apps_problem, history=[])

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == TURN_1_ORACLE_PROMPT

    # The active live question should sit at the very end of the message history stack
    assert messages[-1]["role"] == "user"
    assert "palindrome" in messages[-1]["content"]


def test_format_prompt_transitions_to_sandbox_on_later_turns(mock_apps_problem):
    """Asserts that once conversation history exists, execution tools unlock."""
    mock_history = [
        {"role": "assistant", "content": "<tool>generate_tests</tool><code></code>"},
        {
            "role": "user",
            "content": "Tool: generate_tests\nResult: 3 assert statements added.",
        },
    ]
    messages = format_prompt(problem=mock_apps_problem, history=mock_history)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SUBSEQUENT_TURNS_PROMPT
    # Verify the multi-turn trajectory context is preserved chronologically at the end
    assert messages[-2] == mock_history[0]
    assert messages[-1] == mock_history[1]


# ── 2. Strict XML Parsing Boundary Tests ──────────────────────────────────────


def test_parse_output_extracts_valid_xml_blocks():
    """Validates perfect regex extraction fields from structured XML envelopes."""
    raw_response = "<tool>execute</tool><code>def is_pal(x):\n    return str(x) == str(x)[::-1]</code>"
    parsed = parse_output(raw_response)

    assert parsed.is_valid is True
    assert parsed.tool == "execute"
    assert "return str(x)" in parsed.code


def test_parse_output_ignores_xml_whitespace_and_case():
    """Ensures parser isolates values regardless of structural spacing or character case."""
    raw_response = "<TOOL>\n  lint\n</TOOL>\n<code>\nx = 5\n</code>"
    parsed = parse_output(raw_response)

    assert parsed.is_valid is True
    assert parsed.tool == "lint"
    assert parsed.code == "x = 5"


def test_parse_output_fails_on_unauthorized_tools():
    """Asserts that attempts to issue non-whitelist tools returns invalid schemas."""
    raw_response = "<tool>delete_sandbox</tool><code>import os</code>"
    parsed = parse_output(raw_response)

    assert parsed.is_valid is False
    assert parsed.tool is None


def test_parse_output_fallback_to_legacy_markdown():
    """Validates backwards compatibility with single-turn python markdown style blocks."""
    raw_response = "```python\ndef solve(): pass\n```"
    parsed = parse_output(raw_response)

    assert parsed.is_valid is True
    assert parsed.tool == "execute"
    assert parsed.code == "def solve(): pass"
