"""
tools.py — Tool layer for RLEF-Code

Three tools the agent can use during code generation:

  execute(code)        → runs code in E2B sandbox, returns stdout/stderr/error
  lint(code)           → runs ruff on code, returns structured lint feedback
  generate_tests(...)  → prompts the model to write its own test cases

Each tool returns a ToolResult dataclass so the reward function and
trajectory manager always get a consistent structure regardless of
which tool was called.

Design note: tools are stateless functions, not a class hierarchy.
The router (at the bottom) dispatches by name. This keeps it easy
to add tools later without restructuring.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum

# ── Result types ──────────────────────────────────────────────────────────────


class ToolName(str, Enum):
    EXECUTE = "execute"
    LINT = "lint"
    TESTS = "generate_tests"


@dataclass
class ToolResult:
    tool: ToolName
    success: bool  # did the tool itself run without crashing
    output: str  # human-readable feedback to show the model
    # execution-specific
    stdout: str = ""
    stderr: str = ""
    error: str = ""  # runtime error message if any
    # lint-specific
    lint_errors: list[dict] = field(default_factory=list)
    # test generation specific
    generated_tests: str = ""


# ── Tool 1: Execute ───────────────────────────────────────────────────────────


def execute_local(code: str, timeout: int = 10) -> ToolResult:
    """
    Run code locally in a subprocess — fallback when E2B is unavailable.
    Only use on trusted code (eval/training on known datasets).
    """
    import os
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            ["python3", tmp], capture_output=True, text=True, timeout=timeout
        )
        stdout = result.stdout
        stderr = result.stderr
        error = stderr if result.returncode != 0 else ""

        parts = []
        if stdout:
            parts.append(f"stdout:\n{stdout.strip()}")
        if stderr:
            parts.append(f"stderr:\n{stderr.strip()}")
        if not parts:
            parts.append("(no output)")

        return ToolResult(
            tool=ToolName.EXECUTE,
            success=result.returncode == 0,
            output="\n".join(parts),
            stdout=stdout,
            stderr=stderr,
            error=error,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool=ToolName.EXECUTE, success=False, output="Timeout", error="Timeout"
        )
    except Exception as e:
        return ToolResult(
            tool=ToolName.EXECUTE, success=False, output=str(e), error=str(e)
        )
    finally:
        os.unlink(tmp)


def execute(code: str, timeout: int = 30) -> ToolResult:
    """
    Run code locally in a subprocess.
    E2B bypassed due to ongoing outage — local execution on Vast.ai machine.
    """
    import os
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            ["python3", tmp], capture_output=True, text=True, timeout=timeout
        )
        stdout = result.stdout
        stderr = result.stderr
        error = stderr if result.returncode != 0 else ""

        parts = []
        if stdout:
            parts.append(f"stdout:\n{stdout.strip()}")
        if stderr:
            parts.append(f"stderr:\n{stderr.strip()}")
        if not parts:
            parts.append("(no output)")

        return ToolResult(
            tool=ToolName.EXECUTE,
            success=result.returncode == 0,
            output="\n".join(parts),
            stdout=stdout,
            stderr=stderr,
            error=error,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool=ToolName.EXECUTE, success=False, output="Timeout", error="Timeout"
        )
    except Exception as e:
        return ToolResult(
            tool=ToolName.EXECUTE, success=False, output=str(e), error=str(e)
        )
    finally:
        os.unlink(tmp)


def lint(code: str) -> ToolResult:
    """
    Run ruff on the code and return structured lint feedback.

    Why ruff and not pylint/flake8:
      - Already a dev dependency (we use it for our own code)
      - Fastest linter available, no cold start
      - Output is structured JSON, easy to parse

    The model sees a formatted list of errors with line numbers.
    If there are no errors, it sees "No lint errors." — a positive
    signal that syntax is clean before burning an execution call.
    """
    # write code to a temp file — ruff operates on files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # ruff exits 0 = no errors, 1 = errors found, 2 = internal error
        if result.returncode == 2:
            return ToolResult(
                tool=ToolName.LINT,
                success=False,
                output=f"Lint tool error: {result.stderr}",
            )

        import json as _json

        try:
            errors = _json.loads(result.stdout) if result.stdout.strip() else []
        except _json.JSONDecodeError:
            errors = []

        if not errors:
            return ToolResult(
                tool=ToolName.LINT,
                success=True,
                output="No lint errors.",
                lint_errors=[],
            )

        # Format errors for the model — line number, rule, message
        lines = []
        for e in errors:
            loc = e.get("location", {})
            row = loc.get("row", "?")
            col = loc.get("column", "?")
            code_ = e.get("code", "")
            msg = e.get("message", "")
            lines.append(f"  Line {row}:{col} [{code_}] {msg}")

        output = f"{len(errors)} lint error(s):\n" + "\n".join(lines)

        return ToolResult(
            tool=ToolName.LINT,
            success=True,  # tool ran fine, code just has errors
            output=output,
            lint_errors=errors,
        )

    except subprocess.TimeoutExpired:
        return ToolResult(
            tool=ToolName.LINT,
            success=False,
            output="Lint timed out.",
        )
    finally:
        os.unlink(tmp_path)


# ── Tool 3: Generate tests ────────────────────────────────────────────────────


def generate_tests(problem: str, code: str) -> ToolResult:
    """
    Ask the model to write its own test cases for the problem.

    This is a prompt-based tool — it calls no external service directly.
    The caller (training loop) injects the model response back.
    Here we just format the prompt the model should respond to.

    Why this matters: a model that generates good tests understands
    the problem specification from both sides (implementation + verification).
    We measure whether test quality correlates with final pass rate.
    """
    prompt = (
        "Write 3 Python assert statements to test the following solution.\n"
        "Only output the assert statements, nothing else.\n\n"
        f"Problem:\n{problem.strip()}\n\n"
        f"Solution:\n{code.strip()}\n\n"
        "Assert statements:"
    )

    return ToolResult(
        tool=ToolName.TESTS,
        success=True,
        output=prompt,
        generated_tests=prompt,
    )


# ── Router ────────────────────────────────────────────────────────────────────


def call_tool(
    tool_name: str,
    code: str = "",
    problem: str = "",
    timeout: int = 10,
) -> ToolResult:
    """
    Dispatch a tool call by name.
    This is what the training loop calls — it never imports tools directly.
    """
    name = tool_name.strip().lower()

    if name == ToolName.EXECUTE:
        return execute(code, timeout=timeout)
    elif name == ToolName.LINT:
        return lint(code)
    elif name == ToolName.TESTS:
        return generate_tests(problem, code)
    else:
        return ToolResult(
            tool=ToolName.EXECUTE,
            success=False,
            output=f"Unknown tool: '{tool_name}'. Available: execute, lint, generate_tests",
        )
