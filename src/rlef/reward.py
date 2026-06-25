"""
reward.py — Reward functions for RLEF-Code

Three components:

  1. execution_reward(code, problem)
       Runs code against all APPS test cases via sandbox execution.
       Returns continuous pass rate in.

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

import json
import math
import re
from dataclasses import dataclass
from typing import Literal

from rlef.tools import ToolName, ToolResult, execute

# ── 1. Data Structures ────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    passed: int
    total: int
    pass_rate: float
    raw_reward: float  # before shaping
    final_reward: float  # after shaping (if enabled)
    error_types: list[str]  # SyntaxError, RuntimeError, etc. for analysis


@dataclass
class StepCredit:
    step_idx: int
    tool: ToolName
    credit: float  # assigned credit for this step
    useful: bool  # was this step judged useful?
    reason: str  # human-readable explanation (for analysis)


# ── 2. Execution Reward Internal Pipeline ─────────────────────────────────────
def _run_against_test_cases(
    code: str,
    inputs: list[str] | str,
    outputs: list[str] | str,
    fn_name: str | None,
    timeout: int = 10,
) -> tuple[int, int, list[str]]:
    """
    Subprocess-Isolated APPS Evaluation Engine. Packs execution environments
    entirely inside isolated subprocess runs to honor timeout bounds perfectly
    and prevent distributed master GPU thread hangs.
    """

    # --- DEFENSIVE DATA UNBOXING ---
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except Exception:
            inputs = [inputs]

    if isinstance(outputs, str):
        try:
            outputs = json.loads(outputs)
        except Exception:
            outputs = [outputs]

    inputs = [str(x) for x in inputs]
    outputs = [str(x) for x in outputs]

    total_cases = len(inputs)
    if total_cases == 0:
        return 0, 0, []

    # Dynamic regex to detect custom top-level functions in competitor code
    found_fns = re.findall(r"def\s+([a-zA-Z_0-9]+)\s*\(", code)
    discovered_fns = [
        f
        for f in found_fns
        if f not in ("main", "solve", "_safe_readline", "run_all_tests")
    ]

    # Inspect the first test case string item to see if it's a structural JSON list/dict
    is_json_list = False
    if len(inputs) > 0:
        try:
            parsed_first = json.loads(inputs[0])
            if isinstance(parsed_first, (list, dict)):
                is_json_list = True
        except Exception:
            pass

    use_fn_routing = bool(fn_name) or (is_json_list and len(discovered_fns) > 0)
    active_fn_name = (
        fn_name if fn_name else (discovered_fns[0] if discovered_fns else None)
    )

    # --- BRANCH A: BUNDLED FUNCTION-CALL EVALUATION ENGINE ---
    if use_fn_routing and active_fn_name:
        # Building the string using clean array lists entirely rules out syntax breaks
        lines = [
            "import json",
            "import sys",
            "sys.setrecursionlimit(200000)\n",
            "user_code = " + repr(code),
            "sandbox_globals = {'json': json, 'sys': sys, 'globals': globals, 'locals': locals, 'isinstance': isinstance, 'str': str, 'getattr': getattr, 'print': print}\n",
            "try:",
            "    exec(user_code, sandbox_globals)",
            "except Exception as exec_err:",
            "    print(f'EXEC_CRASH: {exec_err}')",
            "    sys.exit(0)\n",
            "def run_all_tests():",
            "    all_inputs = " + repr(inputs),
            "    all_outputs = " + repr(outputs),
            "    passed = 0",
            "    ",
            "    for inp, expected_out in zip(all_inputs, all_outputs):",
            "        try:",
            "            raw_inp = json.loads(inp)",
            "        except Exception:",
            "            raw_inp = inp",
            "        try:",
            "            expected = json.loads(expected_out)",
            "        except Exception:",
            "            expected = expected_out",
            "            ",
            "        args = raw_inp if isinstance(raw_inp, list) else [raw_inp]",
            "        res = None",
            "        try:",
            "            if 'Solution' in sandbox_globals:",
            "                try:",
            "                    obj = sandbox_globals['Solution']()",
            "                    res = getattr(obj, '"
            + str(active_fn_name)
            + "')(*args)",
            "                except Exception:",
            "                    try:",
            "                        res = sandbox_globals['"
            + str(active_fn_name)
            + "'](*args)",
            "                    except Exception:",
            "                        res = getattr(sandbox_globals['Solution'], '"
            + str(active_fn_name)
            + "')(sandbox_globals['Solution'](), *args)",
            "            else:",
            "                res = sandbox_globals['"
            + str(active_fn_name)
            + "'](*args)",
            "                ",
            "            str_res = str(res).strip().lower()",
            "            str_exp = str(expected).strip().lower()",
            "            if res == expected or str_res == str_exp:",
            "                passed += 1",
            "            elif (str_res == 'true' and str_exp == '1') or (str_res == '1' and str_exp == 'true'):",
            "                passed += 1",
            "            elif (str_res == 'false' and str_exp == '0') or (str_res == '0' and str_exp == 'false'):",
            "                passed += 1",
            "        except Exception:",
            "            continue",
            "    print(f'RLEF_PASSED: {passed}')",
            "run_all_tests()",
        ]
        test_code = "\n".join(lines) + "\n"

        result = execute(test_code, timeout=timeout)
        if not result.success:
            return 0, total_cases, [_classify_error(result)] * total_cases

        match = re.search(r"RLEF_PASSED:\s*(\d+)", result.stdout or "")
        if match:
            passed = int(match.group(1))
            failed_count = total_cases - passed
            error_types = ["WrongOutput"] * failed_count if failed_count > 0 else []
            return passed, total_cases, error_types
        else:
            if "EXEC_CRASH" in (result.stdout or ""):
                return 0, total_cases, ["SyntaxError"] * total_cases
            return 0, total_cases, [_classify_error(result)] * total_cases

    # --- BRANCH B: BUNDLED STDIN/STDOUT BATCH ENGINE WITH SUBPROCESS ISOLATION ---
    else:
        lines = [
            "import builtins",
            "import sys",
            "import io",
            "sys.setrecursionlimit(200000)\n",
            "user_code = " + repr(code),
            "all_inputs = " + repr(inputs),
            "all_outputs = " + repr(outputs),
            "passed_count = 0\n",
            "for inp, expected_out in zip(all_inputs, all_outputs):",
            "    sys.stdin = io.StringIO(str(inp) + '\\n')",
            "    _real_readline = sys.stdin.readline",
            "    sys.stdin.readline = lambda *a, **kw: _real_readline(*a, **kw) or '\\n'",
            "    builtins.input = lambda *a, **kw: sys.stdin.readline().rstrip()",
            "    ",
            "    old_stdout = sys.stdout",
            "    sys.stdout = captured = io.StringIO()",
            "    ",
            "    sandbox_globals = {'sys': sys, 'io': io, 'builtins': builtins, 'print': print}",
            "    try:",
            "        exec(user_code, sandbox_globals)",
            "        if 'main' in sandbox_globals: sandbox_globals['main']()",
            "        elif 'solve' in sandbox_globals: sandbox_globals['solve']()",
            "    except Exception:",
            "        sys.stdout = old_stdout",
            "        continue",
            "        ",
            "    sys.stdout = old_stdout",
            "    actual_tokens = captured.getvalue().strip().split()",
            "    expected_tokens = expected_out.strip().split()",
            "    ",
            "    if actual_tokens == expected_tokens:",
            "        passed_count += 1",
            "    else:",
            "        try:",
            "            act_floats = [round(float(x), 3) for x in actual_tokens]",
            "            exp_floats = [round(float(x), 3) for x in expected_tokens]",
            "            if act_floats == exp_floats and len(act_floats) > 0:",
            "                passed_count += 1",
            "        except Exception:",
            "            pass",
            "print(f'RLEF_PASSED: {passed_count}')",
        ]
        test_code = "\n".join(lines) + "\n"

        result = execute(test_code, timeout=timeout)
        if not result.success:
            return 0, total_cases, [_classify_error(result)] * total_cases

        match = re.search(r"RLEF_PASSED:\s*(\d+)", result.stdout or "")
        if match:
            passed = int(match.group(1))
            failed_count = total_cases - passed
            error_types = ["WrongOutput"] * failed_count if failed_count > 0 else []
            return passed, total_cases, error_types
        else:
            return 0, total_cases, [_classify_error(result)] * total_cases

        match = re.search(r"RLEF_PASSED:\s*(\d+)", result.stdout or "")
        if match:
            passed = int(match.group(1))
            failed_count = total_cases - passed
            error_types = ["WrongOutput"] * failed_count if failed_count > 0 else []
            return passed, total_cases, error_types
        else:
            return 0, total_cases, [_classify_error(result)] * total_cases


def _classify_error(result: ToolResult) -> str:
    """Classify the error type for analysis notebooks."""
    if not result.error:
        # If there is no stderr but success=False, it was killed by a system timeout signal
        return "Timeout"

    error_str = result.error.lower()
    if "syntaxerror" in error_str:
        # Code structure cannot be compiled
        return "SyntaxError"
    if "timeout" in error_str or "timed out" in error_str:
        return "Timeout"
    if "assertionerror" in error_str:
        return "WrongOutput"
    if "recursionerror" in error_str:
        return "RecursionError"
    if "indexerror" in error_str:
        return "IndexError"
    if "valueerror" in error_str:
        return "ValueError"

    # Catch-all for other unhandled exceptions
    return "RuntimeError"


# ── 3. Main Reward Evaluation Entrypoint ──────────────────────────────────────

MAX_TESTS_BY_DIFFICULTY = {
    "introductory": 10,
    "interview": 20,
    "competition": 30,
}
DEFAULT_MAX_TESTS = 10


def execution_reward(
    code: str | list[str],
    inputs: list[str],
    outputs: list[str],
    fn_name: str | None = None,
    reward_type: Literal["continuous", "binary"] = "continuous",
    shaped: bool = True,
    timeout: int = 10,
    difficulty: str = "introductory",
) -> ExecutionResult:
    """
    Main reward function. Runs code against ALL test cases present
    in the dataset without slicing truncation to maintain true data alignment.
    """
    # Defensive guard rail for lists passed as code blocks
    if isinstance(code, list):
        code = code[0] if len(code) > 0 else ""

    if not inputs or not code:
        return ExecutionResult(0, 0, 0.0, 0.0, 0.0, [])

    # Evaluate against ALL test cases natively
    passed, total, error_types = _run_against_test_cases(
        code, inputs, outputs, fn_name, timeout
    )

    pass_rate = passed / total if total > 0 else 0.0

    # Raw reward calculation
    if reward_type == "binary":
        raw = 1.0 if passed == total else 0.0
    else:
        raw = pass_rate

    # Apply log shaping if enabled
    final = shape_reward(raw) if shaped else raw

    return ExecutionResult(
        passed=passed,
        total=total,
        pass_rate=pass_rate,
        raw_reward=raw,
        final_reward=final,
        error_types=error_types,
    )


# ── 4. Reward shaping ─────────────────────────────────────────────────────────


def shape_reward(raw: float) -> float:
    """
    Log-shape a reward value from [0, 1] →.

    f(r) = log(1 + r * (e - 1))

    Properties:
      f(0.0) = 0.0   (zero stays zero)
      f(1.0) = 1.0   (perfect stays perfect)
    The curve compresses high rewards and expands low ones.
    This makes early training more stable — going from 0.1 to 0.2
    produces a larger gradient update than going from 0.8 to 0.9,
    which is exactly what we want when the model is just starting out.
    """
    assert 0.0 <= raw <= 1.0, f"reward must be in, got {raw}"
    return math.log1p(raw * (math.e - 1))


# ── 5. Step-level credit assignment ───────────────────────────────────────────


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
        if tool_result.output and "Sandbox error" in tool_result.output:
            return False, "sandbox crash"
        return True, "execution produced feedback"

    elif tool_result.tool == ToolName.LINT:
        if getattr(tool_result, "lint_errors", None):
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
