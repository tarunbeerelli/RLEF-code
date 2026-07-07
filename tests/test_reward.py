# tests/test_reward.py
import pytest
from rlef.reward import execution_reward, verify_generated_tests

@pytest.mark.unit
def test_execution_reward_empty_code():
    # Test a structural sanity gate that returns instantly without executing a subprocess
    res = execution_reward(code="", inputs=["1"], outputs=["2"])
    assert res.passed == 0
    assert res.pass_rate == 0.0
    assert "SyntaxError" in res.error_types

@pytest.mark.integration
def test_execution_reward_perfect_pass():
    # Requires a local Python installation with functional execution environment
    code = "x = int(input())\nprint(x * 2)"
    res = execution_reward(code=code, inputs=["5", "10"], outputs=["10", "20"])
    assert res.passed == 2
    assert res.pass_rate == 1.0

@pytest.mark.integration
def test_verify_generated_tests_valid():
    model_code = "def solve(x): return x * 2"
    test_code = "assert solve(2) == 4\nassert solve(0) == 0"
    bonus = verify_generated_tests(test_code, model_code, fn_name="solve")
    assert bonus == 0.15