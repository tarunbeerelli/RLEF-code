"""
Unit tests validating multi-turn agent execution routing contracts.
"""

from rlef.prompt import parse_output


def test_xml_tag_extraction_boundaries():
    """Validates absolute extraction limits of the updated XML prompt strategy."""
    raw_response = "<tool>generate_tests</tool><code>def test_stub():\n    pass</code>"
    parsed = parse_output(raw_response)

    assert parsed.is_valid is True
    assert parsed.tool == "generate_tests"
    assert "test_stub" in parsed.code


def test_malformed_xml_block_handling():
    """Ensures raw text strings break out safely into non-valid parsing records."""
    raw_response = "I am unsure how to resolve this array indexing issue."
    parsed = parse_output(raw_response)

    assert parsed.is_valid is False
    assert parsed.tool is None
    assert parsed.code is None
