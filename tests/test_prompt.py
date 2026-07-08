# tests/test_prompt.py
import pytest
from rlef.prompt import build_system_prompt, parse_output

pytestmark = pytest.mark.unit  # Enforces that every test in this file is marked 'unit'


def test_build_system_prompt_toggles():
    cfg_exec = {"use_edge_cases": False}
    prompt_exec = build_system_prompt(cfg_exec)
    assert "IMPLEMENTATION" in prompt_exec
    assert "TEST VALIDATION" not in prompt_exec

    cfg_tdd = {"use_edge_cases": True}
    prompt_tdd = build_system_prompt(cfg_tdd)
    assert "<edge_cases>" in prompt_tdd
    assert "TEST VALIDATION" in prompt_tdd


def test_parse_output_clean_blocks():
    raw = "<reasoning>Logic analysis</reasoning><code>def solve(): pass</code>"
    parsed = parse_output(raw)
    assert parsed["is_valid"] is True
    assert parsed["code"] == "def solve(): pass"
    assert parsed["edge_cases"] is None


def test_parse_output_with_edge_cases():
    raw = "<edge_cases>assert solve(1) == 2</edge_cases><code>def solve(x): return x+1</code>"
    parsed = parse_output(raw)
    assert parsed["is_valid"] is True
    assert parsed["code"] == "def solve(x): return x+1"
    assert parsed["edge_cases"] == "assert solve(1) == 2"


def test_parse_output_invalid_format():
    raw = "def solve(): return True"
    parsed = parse_output(raw)
    assert parsed["is_valid"] is False
    assert parsed["code"] is None
