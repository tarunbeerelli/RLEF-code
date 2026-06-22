"""Tests for prompt formatting and output parsing."""

from rlef.data import APPSProblem
from rlef.prompt import format_feedback, format_prompt, parse_output
from rlef.tools import ToolName


def _make_problem() -> APPSProblem:
    return APPSProblem(
        problem_id="0001",
        difficulty="introductory",
        question="Write a function that adds two numbers.",
        inputs=["1 2", "3 4"],
        outputs=["3", "7"],
        fn_name=None,
        solutions=[],
        url="",
    )


def test_parse_valid_execute():
    text = "<tool>execute</tool>\n<code>\nprint(1+1)\n</code>"
    result = parse_output(text)
    assert result.is_valid
    assert result.tool == "execute"
    assert result.tool_name == ToolName.EXECUTE
    assert "print(1+1)" in result.code


def test_parse_valid_lint():
    text = "<tool>lint</tool>\n<code>\ndef f():\n    return 1\n</code>"
    result = parse_output(text)
    assert result.is_valid
    assert result.tool == "lint"
    assert result.tool_name == ToolName.LINT


def test_parse_valid_generate_tests():
    text = "<tool>generate_tests</tool>\n<code>\ndef f(x): return x\n</code>"
    result = parse_output(text)
    assert result.is_valid
    assert result.tool == "generate_tests"
    assert result.tool_name == ToolName.TESTS


def test_parse_invalid_tool_name():
    text = "<tool>unknown_tool</tool>\n<code>\nprint(1)\n</code>"
    result = parse_output(text)
    assert not result.is_valid
    assert result.tool is None


def test_parse_missing_code_block():
    text = "<tool>execute</tool>\nsome code without tags"
    result = parse_output(text)
    assert not result.is_valid
    assert result.code is None


def test_parse_missing_tool_tag():
    text = "<code>\nprint(1)\n</code>"
    result = parse_output(text)
    assert not result.is_valid
    assert result.tool is None


def test_parse_preserves_raw():
    text = "<tool>execute</tool>\n<code>\nprint(1)\n</code>"
    result = parse_output(text)
    assert result.raw == text


def test_format_prompt_returns_messages():
    problem = _make_problem()
    messages = format_prompt(problem)
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert problem.question[:20] in messages[1]["content"]


def test_format_prompt_with_history():
    problem = _make_problem()
    history = [
        {
            "role": "assistant",
            "content": "<tool>lint</tool>\n<code>\nprint(1)\n</code>",
        },
        {"role": "user", "content": "Tool: lint\nResult:\nNo lint errors."},
    ]
    messages = format_prompt(problem, history=history)
    assert len(messages) == 4
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"


def test_format_feedback_structure():
    fb = format_feedback("execute", "stdout: 3")
    assert fb["role"] == "user"
    assert "execute" in fb["content"]
    assert "stdout: 3" in fb["content"]
