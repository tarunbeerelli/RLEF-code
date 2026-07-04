"""
prompt.py — Prompt formatting and model output parsing with Forced Asymmetry.

Manages asymmetric instruction distributions and XML tag validation rules
across separate phases of a multi-turn reasoning episode.
"""

import re
from dataclasses import dataclass

from rlef.data import APPSProblem
from rlef.tools import ToolName

# ── Forced Asymmetry System Prompts ──────────────────────────────────────────

TURN_1_ORACLE_PROMPT = """You are an expert Python algorithmic scientist.
Analyze the user's coding problem description and map out its semantic constraints.

CRITICAL INITIAL PROTOCOL:
You are strictly forbidden from writing a final solution code block or running tools other than 'generate_tests' on this first turn.
You MUST invoke the test generation tool to isolate edge cases, boundary parameters, and valid inputs.

Limit your generation to exactly 3 to 5 high-value test cases. Focus only on critical edge cases rather than exhaustive standard inputs.

You must wrap your selection in explicit XML block structures precisely like this:
<tool>generate_tests</tool>
<code>
# Provide your tentative test verification logic here
</code>
"""

SUBSEQUENT_TURNS_PROMPT = """You are an expert Python programmer operating in a multi-turn execution sandbox.
Review the user's test-case criteria, runtime errors, or execution feedback.

Iterate on your implementation using our operational tool layer.
Available tools:
- <tool>execute</tool> : Evaluates your functional code against the provided test criteria.

CRITICAL INSTRUCTION:
Review any execution errors or failed test cases provided in the previous turn. Use that feedback to debug and refine your solution.

You MUST wrap your tool calls in explicit XML block structures precisely like this:
<tool>execute</tool>
<code>
# Your Python code to run goes here
</code>
"""

PROBLEM_TEMPLATE = "{question}\n"

FEEDBACK_TEMPLATE = """Tool: {tool}
Result:
{output}

Continue solving. Use <tool>execute</tool> when your code adjustments are ready.
"""


@dataclass
class ParsedOutput:
    tool: str | None  # "execute", "generate_tests", or None if parse failed
    code: str | None  # extracted code string, or None if parse failed
    raw: str  # original unmutated model token generation string

    @property
    def is_valid(self) -> bool:
        return self.tool is not None and self.code is not None

    @property
    def tool_name(self) -> ToolName | None:
        if self.tool == "execute":
            return ToolName.EXECUTE
        if self.tool == "generate_tests":
            return ToolName.TESTS
        return None


# ── In-Context Few-Shot Realignment Trajectory ─────────────────────────────────

FEW_SHOT_ALIGNMENT_HISTORY = [
    {
        "role": "user",
        "content": "Write a function `double_it(x)` that returns twice the integer input.",
    },
    {
        "role": "assistant",
        "content": "<tool>generate_tests</tool>\n<code>\ndef test_double():\n    assert double_it(2) == 4\n    assert double_it(-1) == -2\n    assert double_it(0) == 0\n</code>",
    },
    {
        "role": "user",
        "content": "Tool: generate_tests\nResult:\n3 custom assertions successfully compiled into local sandbox execution parameters.",
    },
    {
        "role": "assistant",
        "content": "<tool>execute</tool>\n<code>\ndef double_it(x):\n    return x * 2\n</code>",
    },
    {
        "role": "user",
        "content": "Tool: execute\nResult:\nExecution Pass Rate: 100.0%. All custom testing vectors evaluated successfully.",
    },
]

# ── Trajectory Phase Formatting ───────────────────────────────────────────────


def format_prompt(
    problem: APPSProblem,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Format a problem description into a chat message history sequence.
    Appends strict few-shot examples across turns to align the coder to our XML schema.
    """
    if not history or len(history) == 0:
        system_content = TURN_1_ORACLE_PROMPT
        messages = [
            {"role": "system", "content": system_content},
            *FEW_SHOT_ALIGNMENT_HISTORY,
            {
                "role": "user",
                "content": PROBLEM_TEMPLATE.format(question=problem.question.strip()),
            },
        ]
    else:
        system_content = SUBSEQUENT_TURNS_PROMPT
        messages = [
            {"role": "system", "content": system_content},
            *FEW_SHOT_ALIGNMENT_HISTORY,
            {
                "role": "user",
                "content": PROBLEM_TEMPLATE.format(question=problem.question.strip()),
            },
        ]
        messages.extend(history)

    return messages


def format_feedback(tool: str, output: str) -> dict:
    """Format execution output logs back into message sequences."""
    return {
        "role": "user",
        "content": FEEDBACK_TEMPLATE.format(tool=tool, output=output.strip()),
    }


# ── Strict Extraction Parser ──────────────────────────────────────────────────


def parse_output(text: str) -> ParsedOutput:
    """
    Parse model output to extract tools and targeted logic sequences.
    Strictly parses XML tag declarations to eliminate ambiguous code generation states.
    """
    # Parse explicit XML structures
    tool_match = re.search(r"<tool>\s*(\w+)\s*</tool>", text, re.IGNORECASE)
    xml_tool = tool_match.group(1).lower().strip() if tool_match else None
    valid_tools = {"execute", "generate_tests"}

    code_match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL)
    xml_code = code_match.group(1).strip() if code_match else None

    if tool_match or code_match:
        tool = xml_tool if xml_tool in valid_tools else None
        return ParsedOutput(tool=tool, code=xml_code, raw=text)

    # Legacy markdown fallback for backwards compatibility or single-shot evaluations
    md_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if md_match:
        code = md_match.group(1).strip()
        return ParsedOutput(tool="execute", code=code, raw=text)

    return ParsedOutput(tool=None, code=None, raw=text)
