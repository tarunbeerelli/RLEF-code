"""
reward.py — Advanced Native Reward functions for RLEF-Code
Completely decoupled from tools.py. Uses native subprocess isolation.
"""

import json
import re
import subprocess
from dataclasses import dataclass

# ── 1. Data Structures ────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    passed: int
    total: int
    pass_rate: float
    final_reward: float
    error_types: list[str]


# ── 2. Native Subprocess Isolation ────────────────────────────────────────────


def _native_execute(code_str: str, timeout: int = 2) -> tuple[bool, str, str]:
    """Runs code in an isolated subprocess to prevent master thread crashes."""
    try:
        res = subprocess.run(
            ["python3", "-c", code_str], capture_output=True, text=True, timeout=timeout
        )
        return res.returncode == 0, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"


def _classify_error(stderr: str) -> str:
    stderr = stderr.lower()
    if "syntaxerror" in stderr:
        return "SyntaxError"
    if "timeout" in stderr:
        return "Timeout"
    if "assertionerror" in stderr:
        return "AssertionError"
    if "indexerror" in stderr:
        return "IndexError"
    return "RuntimeError"


# ── 3. Test Verification (The True/False Anti-Hallucination Check) ────────────


def verify_generated_tests(
    test_code: str, model_code: str, fn_name: str = None
) -> float:
    """
    Rewards the model for writing tests that actually test logic.
    1. Tests must PASS when run alongside the actual code.
    2. Tests must FAIL when run alongside a dummy function.
    """
    if not test_code or "assert " not in test_code:
        return 0.0

    real_env_code = f"{model_code}\n\n{test_code}"
    real_success, _, _ = _native_execute(real_env_code, timeout=2)

    if not real_success:
        return -0.1  # The tests failed on their own code

    target_fn = fn_name if fn_name else "solve"
    dummy_code = f"def {target_fn}(*args, **kwargs): return False\n\n{test_code}"
    dummy_success, _, _ = _native_execute(dummy_code, timeout=2)

    if dummy_success:
        return -0.5  # Heavy penalty for hallucinating fake tests (e.g. assert True)

    return 0.5  # Passed real, failed dummy = mathematically sound tests!


# ── 4. Main Execution & Feedback Utilization Reward ───────────────────────────


def execution_reward(
    code: str,
    inputs: list[str],
    outputs: list[str],
    previous_pass_rate: float = 0.0,
    current_turn: int = 1,
    shaped: bool = True,
) -> ExecutionResult:
    """Calculates linear code rewards, step credit, turn penalties, and feedback utilization."""

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

    if total_cases == 0 or not code.strip():
        return ExecutionResult(0, 0, 0.0, 0.0, ["SyntaxError"])

    lines = [
        "import sys, io, builtins",
        "sys.setrecursionlimit(200000)",
        "user_code = " + repr(code),
        "all_inputs = " + repr(inputs),
        "all_outputs = " + repr(outputs),
        "passed_count = 0",
        "for inp, expected in zip(all_inputs, all_outputs):",
        "    sys.stdin = io.StringIO(str(inp) + '\\n')",
        "    builtins.input = lambda *a, **kw: sys.stdin.readline().rstrip()",
        "    old_stdout = sys.stdout",
        "    sys.stdout = captured = io.StringIO()",
        "    sandbox_globals = {'sys': sys, 'io': io, 'builtins': builtins}",
        "    try:",
        "        exec(user_code, sandbox_globals)",
        "        if 'main' in sandbox_globals: sandbox_globals['main']()",
        "        elif 'solve' in sandbox_globals: sandbox_globals['solve']()",
        "    except Exception as e:",
        "        sys.stdout = old_stdout",
        "        continue",
        "    sys.stdout = old_stdout",
        "    if captured.getvalue().strip().split() == expected.strip().split():",
        "        passed_count += 1",
        "print(f'RLEF_PASSED: {passed_count}')",
    ]

    test_env_code = "\n".join(lines)
    success, stdout, stderr = _native_execute(test_env_code, timeout=5)

    passed = 0
    error_types = []

    match = re.search(r"RLEF_PASSED:\s*(\d+)", stdout)
    if match:
        passed = int(match.group(1))
        failed = total_cases - passed
        if failed > 0:
            error_types = ["WrongOutput"] * failed
    else:
        error_types = [_classify_error(stderr)] * total_cases

    pass_rate = passed / total_cases if total_cases > 0 else 0.0

    if not shaped:
        return ExecutionResult(passed, total_cases, pass_rate, pass_rate, error_types)

    # --- THE REWARD SHAPING LOGIC (NO LINTING) ---
    final_reward = pass_rate  # 1. Linear Execution Pass Rate

    if 0.0 < pass_rate < 1.0:
        final_reward += 0.15  # 2. Step Credit (Partial success)

    final_reward -= (current_turn - 1) * 0.05  # 3. Multi-Turn Efficiency Penalty

    if pass_rate > previous_pass_rate and current_turn > 1:
        final_reward += 0.20  # 4. Feedback Utilization Bonus!

    return ExecutionResult(
        passed=passed,
        total=total_cases,
        pass_rate=pass_rate,
        final_reward=max(0.0, final_reward),
        error_types=error_types,
    )


# ── 5. XML Format Breadcrumbs ─────────────────────────────────────────────────


def format_reward_fn(prompts, completions) -> list[float]:
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0
        if not text:
            rewards.append(0.0)
            continue

        if "<tool>" in text and "</tool>" in text:
            score += 0.25
        if "<code>" in text and "</code>" in text:
            score += 0.25
        if any(t in text.lower() for t in ["execute", "generate_tests"]):
            score += 0.25
        if text.strip().endswith("</tool>"):
            score += 0.25

        rewards.append(score)
    return rewards
