"""
reward.py — Reward functions for RLEF-Code

Three components:
  1. execution_reward(code, problem, ablation_cfg)
       Runs code against all APPS test cases via sandbox execution.
       Returns continuous pass rate and applies ablation-controlled dense rewards.
  2. shape_reward(raw)
       Applies log shaping to compress the reward range.
  3. assign_step_credit(trajectory, final_reward)
       Discounts the final reward back through trajectory steps.
  4. normalize_batch_rewards(rewards)
       Applies Z-score normalization across a batch of generations.
  5. format_reward_fn(prompts, completions)
       Dense shaping reward to score XML tag compliance.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from rlef.tools import ToolName, ToolResult, execute

# ── 1. Data Structures ────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    passed: int
    total: int
    pass_rate: float
    raw_reward: float  # before shaping
    final_reward: float  # after shaping/normalization prep
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


def _classify_error(result: ToolResult) -> str:
    """Classify the error type for analysis notebooks."""
    if not result.error:
        return "Timeout"

    error_str = result.error.lower()
    if "syntaxerror" in error_str:
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

    return "RuntimeError"


# ── 3. Main Reward Evaluation Entrypoint ──────────────────────────────────────


def execution_reward(
    code: str | list[str],
    inputs: list[str],
    outputs: list[str],
    fn_name: str | None = None,
    timeout: int = 2,
    difficulty: str | list[str] = "introductory",
    current_turn: int = 1,
    ablation_cfg: dict | None = None,
    shaped: bool = True,
) -> ExecutionResult:
    """
    Calculates granular code rewards using explicit research ablation toggles.
    """
    if ablation_cfg is None:
        ablation_cfg = {
            "use_lint_bonus": True,
            "use_step_credit": True,
            "use_multi_turn": True,
            "use_log_reward": False,
        }

    # Handle batched GRPO kwargs lists
    if isinstance(difficulty, list):
        difficulty = difficulty[0] if len(difficulty) > 0 else "introductory"

    if isinstance(code, list):
        code = code[0]
    if not inputs or not code:
        return ExecutionResult(0, 0, 0.0, 0.0, 0.0, [])

    passed, total, error_types = _run_against_test_cases(
        code, inputs, outputs, fn_name, timeout
    )

    pass_rate = passed / total if total > 0 else 0.0
    base_reward = pass_rate

    # If shaped is disabled (eval mode), return pure pass rate metric immediately
    if not shaped:
        return ExecutionResult(
            passed=passed,
            total=total,
            pass_rate=pass_rate,
            raw_reward=base_reward,
            final_reward=pass_rate,
            error_types=error_types,
        )

    # Toggle: Lint / Syntax Scaffolding
    compile_bonus = 0.0
    error_penalty = 0.0
    if ablation_cfg.get("use_lint_bonus", True):
        compile_bonus = 0.05 if "SyntaxError" not in error_types else 0.0
        error_penalty = -0.01 if "SyntaxError" in error_types else 0.0

    # Toggle: Step-wise Partial Progress
    progress_bonus = 0.0
    if ablation_cfg.get("use_step_credit", True):
        progress_bonus = 0.15 if 0.0 < pass_rate < 1.0 else 0.0

    # Toggle: Multi-Turn Turn Efficiency Penalty
    turn_penalty = 0.0
    if ablation_cfg.get("use_multi_turn", True):
        turn_penalty = (current_turn - 1) * 0.05

    final_reward = (
        base_reward + progress_bonus + compile_bonus + error_penalty - turn_penalty
    )

    difficulty_scales = {"introductory": 1.0, "interview": 1.2, "competition": 1.5}
    final_reward *= difficulty_scales.get(difficulty, 1.0)

    # Ensure non-negative before applying log shapes
    final_reward = max(0.0, final_reward)

    # Toggle: Log Shaping
    if ablation_cfg.get("use_log_reward", False):
        capped_reward = min(1.0, final_reward)
        final_reward = shape_reward(capped_reward)

    return ExecutionResult(
        passed=passed,
        total=total,
        pass_rate=pass_rate,
        raw_reward=base_reward,
        final_reward=final_reward,
        error_types=error_types,
    )


# ── 4. Reward shaping ─────────────────────────────────────────────────────────


def shape_reward(raw: float) -> float:
    """Log-shape a reward value from [0, 1]."""
    assert 0.0 <= raw <= 1.0, f"reward must be in [0, 1], got {raw}"
    return math.log1p(raw * (math.e - 1))


# ── 5. Step-level credit assignment ───────────────────────────────────────────


def _step_utility(tool_result: ToolResult) -> tuple[bool, str]:
    if tool_result.tool == ToolName.EXECUTE:
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
    credits = []

    for i, result in enumerate(tool_results):
        if credit_type == "trajectory":
            credit = final_reward
            useful = True
            reason = "trajectory-level (no step signal)"
        else:
            useful, reason = _step_utility(result)
            if useful:
                credit = final_reward * (gamma**i)
            else:
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


# ── 6. Batch Normalization ────────────────────────────────────────────────────


def normalize_batch_rewards(
    rewards: list[float], use_normalization: bool = True
) -> list[float]:
    """
    Toggle: Z-Score Batch Normalization.
    """
    if not use_normalization or len(rewards) <= 1:
        return rewards

    arr = np.array(rewards, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr)

    if std < 1e-8:
        return [0.0] * len(rewards)

    normalized = (arr - mean) / std
    return normalized.tolist()


# ── 7. Structural Schema Formatting Reward ────────────────────────────────────


def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """
    Dense structural compliance reward function for GRPO.
    Awards partial fractional metrics for matching valid XML tool block structures.
    """
    rewards = []
    for completion in completions:
        # Extract content text string safely from TRL token structures
        text = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0

        if not text:
            rewards.append(0.0)
            continue

        # Rule A: Open and Close structural tags detected
        if "<tool>" in text and "</tool>" in text:
            score += 0.25
        if "<code>" in text and ("Code>" in text or "</code>" in text):
            score += 0.25

        # Rule B: Valid operational targeted tool words selected
        if any(t in text.lower() for t in ["lint", "execute", "generate_tests"]):
            score += 0.25

        # Rule C: Perfect bounding encapsulations
        stripped = text.strip()
        if stripped.startswith("<tool>") and (
            stripped.endswith("</code>") or stripped.endswith("</tool>")
        ):
            score += 0.25

        rewards.append(score)
    return rewards
