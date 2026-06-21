"""
reward.py — Reward functions for RLEF-Code

Three components:

  1. execution_reward(code, problem)
       Runs code against all APPS test cases via E2B.
       Returns continuous pass rate in [0, 1].

  2. shape_reward(raw)
       Applies log shaping to compress the reward range.
       Makes small improvements in the low range feel larger.

  3. assign_step_credit(trajectory, final_reward)
       Discounts the final reward back through trajectory steps
       weighted by how "useful" each tool call was.
       This is the step-level credit assignment contribution.

Ablation flags:
  - reward_type: "continuous" | "binary"
  - shaped:      True | False
  - credit:      "step" | "trajectory"

All combinations are supported so we can run clean ablations
without touching the training code.
"""

import math
from dataclasses import dataclass
from typing import Literal

from rlef.tools import ToolName, ToolResult

# ── 1. Execution reward ───────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    passed: int
    total: int
    pass_rate: float
    raw_reward: float  # before shaping
    final_reward: float  # after shaping (if enabled)
    error_types: list[str]  # SyntaxError, RuntimeError, etc. for analysis


def _run_against_test_cases(
    code: str,
    inputs: list[str],
    outputs: list[str],
    fn_name: str | None,
    timeout: int = 10,
) -> tuple[int, int, list[str]]:
    """
    Run code against all (input, output) pairs from an APPS problem.
    Returns (passed, total, error_types).

    APPS has two problem formats:
      - stdin/stdout: code reads from input(), writes to print()
      - fn_name:      code defines a function, we call it directly

    We handle both by wrapping the code appropriately.
    """
    from rlef.tools import execute

    passed = 0
    error_types = []

    for inp, expected_out in zip(inputs, outputs):
        if fn_name:
            test_code = (
                f"{code}\n\n"
                f"result = {fn_name}(*{repr(inp)})\n"
                f"expected = {repr(expected_out)}\n"
                f"assert str(result).strip() == str(expected).strip(), "
                f"f'got {{result}}, expected {{expected}}'\n"
                f"print('PASS')\n"
            )
        else:
            # Jupyter kernel blocks sys.stdin redirection.
            # Instead we patch builtins.input to return lines from the
            # input string one at a time, which works in any environment.
            test_code = (
                f"import builtins\n"
                f"_lines = iter({repr(inp)}.splitlines())\n"
                f"builtins.input = lambda *a, **kw: next(_lines)\n"
                f"{code}\n"
            )

        result = execute(test_code, timeout=timeout)

        if fn_name:
            # check explicit PASS marker
            if result.success and "PASS" in result.stdout:
                passed += 1
            else:
                err = _classify_error(result)
                error_types.append(err)
        else:
            # for stdin/stdout, compare actual stdout to expected
            actual = result.stdout.strip()
            expected = expected_out.strip()
            if result.success and actual == expected:
                passed += 1
            else:
                err = _classify_error(result)
                error_types.append(err)

    return passed, len(inputs), error_types


def _classify_error(result: ToolResult) -> str:
    """Classify the error type for analysis notebooks."""
    if not result.error:
        return "WrongOutput"
    error_str = result.error.lower()
    if "syntaxerror" in error_str:
        return "SyntaxError"
    if "timeout" in error_str or "timed out" in error_str:
        return "Timeout"
    if "assertionerror" in error_str:
        return "AssertionError"
    return "RuntimeError"


def execution_reward(
    code: str,
    inputs: list[str],
    outputs: list[str],
    fn_name: str | None = None,
    reward_type: Literal["continuous", "binary"] = "continuous",
    shaped: bool = True,
    timeout: int = 10,
) -> ExecutionResult:
    """
    Main reward function. Runs code against test cases and returns
    a structured result with raw and final rewards.

    Args:
        code:        model-generated Python code
        inputs:      list of test inputs from APPSProblem
        outputs:     list of expected outputs from APPSProblem
        fn_name:     if set, problem uses function-call format
        reward_type: "continuous" = pass_rate, "binary" = 0 or 1
        shaped:      whether to apply log shaping
        timeout:     seconds per test case execution
    """
    if not inputs:
        return ExecutionResult(0, 0, 0.0, 0.0, 0.0, [])

    passed, total, error_types = _run_against_test_cases(
        code, inputs, outputs, fn_name, timeout
    )

    pass_rate = passed / total if total > 0 else 0.0

    # raw reward before shaping
    if reward_type == "binary":
        raw = 1.0 if passed == total else 0.0
    else:
        raw = pass_rate

    # apply log shaping if enabled
    final = shape_reward(raw) if shaped else raw

    return ExecutionResult(
        passed=passed,
        total=total,
        pass_rate=pass_rate,
        raw_reward=raw,
        final_reward=final,
        error_types=error_types,
    )


# ── 2. Reward shaping ─────────────────────────────────────────────────────────


def shape_reward(raw: float) -> float:
    """
    Log-shape a reward value from [0, 1] → [0, 1].

    f(r) = log(1 + r * (e - 1))

    Properties:
      f(0.0) = 0.0   (zero stays zero)
      f(1.0) = 1.0   (perfect stays perfect)
      f(0.5) ≈ 0.62  (halfway gets more than half credit)

    The curve compresses high rewards and expands low ones.
    This makes early training more stable — going from 0.1 to 0.2
    produces a larger gradient update than going from 0.8 to 0.9,
    which is exactly what we want when the model is just starting out.
    """
    assert 0.0 <= raw <= 1.0, f"reward must be in [0,1], got {raw}"
    return math.log1p(raw * (math.e - 1))


# ── 3. Step-level credit assignment ───────────────────────────────────────────


@dataclass
class StepCredit:
    step_idx: int
    tool: ToolName
    credit: float  # assigned credit for this step
    useful: bool  # was this step judged useful?
    reason: str  # human-readable explanation (for analysis)


def _step_utility(tool_result: ToolResult) -> tuple[bool, str]:
    """
    Judge whether a tool call was useful.

    A step is useful if it produced information that could guide
    the next action. The heuristic:

      execute: useful if it ran without sandbox error
               (even wrong output is useful — it tells the model what to fix)
               not useful if it was a pure timeout or sandbox crash

      lint:    useful if it found errors (fixed a real problem)
               not useful if code was already clean (redundant call)

      tests:   always useful — generating tests always adds information
    """
    if tool_result.tool == ToolName.EXECUTE:
        # sandbox crash = not useful; wrong output or runtime error = useful
        if "Sandbox error" in tool_result.output:
            return False, "sandbox crash"
        return True, "execution produced feedback"

    elif tool_result.tool == ToolName.LINT:
        if tool_result.lint_errors:
            return True, f"found {len(tool_result.lint_errors)} lint error(s)"
        return False, "no lint errors found (redundant call)"

    elif tool_result.tool == ToolName.TESTS:
        return True, "test generation always informative"

    return False, "unknown tool"


def assign_step_credit(
    tool_results: list[ToolResult],
    final_reward: float,
    gamma: float = 0.9,
    credit_type: Literal["step", "trajectory"] = "step",
) -> list[StepCredit]:
    """
    Assign credit to each step in a trajectory.

    trajectory mode: every step gets the same final_reward.
                     This is standard GRPO — no step-level signal.

    step mode:       useful steps get discounted final_reward,
                     useless steps get a small penalty.
                     Discount factor gamma means later steps get
                     less credit than earlier ones that enabled them.

    Args:
        tool_results:  ordered list of ToolResults from one episode
        final_reward:  the outcome reward (from execution_reward)
        gamma:         discount factor for temporal credit
        credit_type:   "step" or "trajectory"
    """
    credits = []

    for i, result in enumerate(tool_results):
        if credit_type == "trajectory":
            # baseline: every step gets identical credit
            credit = final_reward
            useful = True
            reason = "trajectory-level (no step signal)"

        else:
            # step-level: discount by position and utility
            useful, reason = _step_utility(result)

            if useful:
                # earlier useful steps get more credit (they enabled later ones)
                # gamma^i means step 0 gets full discount, step 1 gets gamma, etc.
                credit = final_reward * (gamma**i)
            else:
                # useless steps get a small negative signal
                # -0.05 is enough to discourage redundant calls without
                # destabilising training
                credit = -0.05

        credits.append(
            StepCredit(
                step_idx=i,
                tool=result.tool,
                credit=credit,
                useful=useful,
                reason=reason,
            )
        )

    return credits
