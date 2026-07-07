"""
Tests for the tool layer.

Execute tests are marked as integration tests since they need
an E2B API key and network access. Run them with:
    poetry run pytest tests/test_tools.py -m integration -v

Lint and router tests run locally with no external dependencies.
"""

import pytest
from rlef.tools import ToolName, call_tool, execute, generate_tests, lint

# ── Lint tests (no external deps) ────────────────────────────────────────────


def test_lint_clean_code():
    code = "def add(a, b):\n    return a + b\n"
    result = lint(code)
    assert result.tool == ToolName.LINT
    assert result.success is True
    assert "No lint errors" in result.output
    assert result.lint_errors == []


def test_lint_catches_undefined_variable():
    code = "def foo():\n    return undefined_var\n"
    result = lint(code)
    assert result.success is True  # tool ran fine
    assert len(result.lint_errors) > 0


def test_lint_catches_syntax_error():
    code = "def foo(\n    return 1\n"
    result = lint(code)
    assert result.success is True
    assert len(result.lint_errors) > 0


def test_lint_empty_code():
    result = lint("")
    assert result.success is True
    assert result.lint_errors == []


def test_lint_output_has_line_numbers():
    code = "import os\nimport sys\nx = undefined\n"
    result = lint(code)
    if result.lint_errors:
        assert "Line" in result.output


# ── Generate tests (no external deps) ────────────────────────────────────────


def test_generate_tests_returns_prompt():
    result = generate_tests(
        problem="Write a function that adds two numbers.",
        code="def add(a, b):\n    return a + b\n",
    )
    assert result.tool == ToolName.TESTS
    assert result.success is True
    assert "assert" in result.output.lower()
    assert len(result.generated_tests) > 0


# ── Router tests ──────────────────────────────────────────────────────────────


def test_router_unknown_tool():
    result = call_tool("nonexistent_tool", code="x = 1")
    assert result.success is False
    assert "Unknown tool" in result.output


def test_router_lint_dispatch():
    result = call_tool("lint", code="def f():\n    return 1\n")
    assert result.tool == ToolName.LINT


def test_router_tests_dispatch():
    result = call_tool(
        "generate_tests",
        code="def f(x): return x",
        problem="Write a function f.",
    )
    assert result.tool == ToolName.TESTS


# ── Execute tests (need E2B key) ──────────────────────────────────────────────


@pytest.mark.integration
def test_execute_simple_print():
    result = execute("print('hello')")
    assert result.tool == ToolName.EXECUTE
    assert result.success is True
    assert "hello" in result.stdout


@pytest.mark.integration
def test_execute_runtime_error():
    result = execute("1 / 0")
    assert result.success is False
    assert result.error != ""


@pytest.mark.integration
def test_execute_timeout():
    result = execute("while True: pass", timeout=3)
    assert result.success is False


@pytest.mark.integration
def test_execute_stdout_captured():
    result = execute("for i in range(3): print(i)")
    assert "0" in result.stdout
    assert "1" in result.stdout
    assert "2" in result.stdout
