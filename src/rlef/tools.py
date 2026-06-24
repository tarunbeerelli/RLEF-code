"""
tools.py — Tool layer for RLEF-Code

Three tools the agent can use during code generation:
  execute(code)        → runs code locally, returns stdout/stderr/error
  lint(code)           → runs ruff on code, returns structured lint feedback
  generate_tests(...)  → prompts the model to write its own test cases

Each tool returns a ToolResult dataclass so the reward function and
trajectory manager always get a consistent structure regardless of
which tool was called.
"""

import json as _json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum

# ── 1. Result types ──────────────────────────────────────────────────────────


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


# ── 2. Tool 1: Execute ───────────────────────────────────────────────────────────


def execute_local(code: str, timeout: int = 10) -> ToolResult:
    """
    Run code locally in a subprocess — fallback when E2B is unavailable.
    Only use on trusted code (eval/training on known datasets).
    """
    return execute(code, timeout=timeout)


def execute(code: str, timeout: int = 10) -> ToolResult:
    """
    Run code locally in a subprocess with cache-busting protections.
    E2B bypassed due to ongoing outage — local execution on Vast.ai machine.
    """
    # Generate a completely unique filename for EVERY single test case run
    # This prevents parallel workers or multi-test loops from collision
    unique_id = uuid.uuid4().hex
    tmp_filename = f"rlef_sandbox_{unique_id}.py"

    with open(tmp_filename, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        # Use python -B to completely disable caching (.pyc generation)
        # This keeps distinct task states completely sandbox-isolated
        result = subprocess.run(
            [sys.executable, "-B", tmp_filename],
            capture_output=True,
            text=True,
            timeout=timeout,
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
            tool=ToolName.EXECUTE,
            success=False,
            output="Timeout",
            error="Timeout: Execution exceeded time limit",
        )
    except Exception as e:
        return ToolResult(
            tool=ToolName.EXECUTE, success=False, output=str(e), error=str(e)
        )
    finally:
        if os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except OSError:
                pass


# ── 3. Tool 2: Lint ───────────────────────────────────────────────────────────


def lint(code: str) -> ToolResult:
    """
    Run ruff on the code and return structured lint feedback.
    """
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

        if result.returncode == 2:
            return ToolResult(
                tool=ToolName.LINT,
                success=False,
                output=f"Lint tool error: {result.stderr}",
            )

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
            success=True,
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
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── 4. Tool 3: Generate tests ─────────────────────────────────────────────────


def generate_tests(problem: str, code: str) -> ToolResult:
    """
    Ask the model to write its own test cases for the problem.
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


# ── 5. Router ─────────────────────────────────────────────────────────────────


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
    try:
        name = ToolName(tool_name)
    except ValueError:
        return ToolResult(
            tool=ToolName.EXECUTE,
            success=False,
            output=f"Unknown tool: '{tool_name}'. Available: execute, lint, generate_tests",
        )

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
