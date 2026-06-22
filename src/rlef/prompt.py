"""
prompt.py — Prompt formatting and model output parsing

Two responsibilities:

  1. format_prompt(problem, history)
       Converts an APPS problem + conversation history into a
       prompt string the model can respond to.

  2. parse_output(text)
       Extracts the tool call and code block from model output.
       Returns a ParsedOutput with tool name and code.

The format uses XML-style tags because they are:
  - Easy for the model to learn (seen in pretraining)
  - Unambiguous to parse with regex
  - Human readable for debugging

Format the model must follow:
  <tool>execute|lint|generate_tests</tool>
  <code>
  ... python code here ...
  </code>

If the model doesn't follow the format we return a ParsedOutput
with tool=None and code=None — the trajectory manager handles
this as a format error and ends the episode.
"""

import re
from dataclasses import dataclass

from rlef.data import APPSProblem
from rlef.tools import ToolName

SYSTEM_PROMPT = """\
You are an expert Python programmer solving competitive programming problems.

For each problem you must:
1. Think through the solution
2. Use tools to write, test and refine your code
3. Submit your final solution using the execute tool

Available tools:
  execute      — run your code and see the output
  lint         — check your code for syntax and style errors
  generate_tests — write test cases to verify your solution

Always respond in this exact format:
<tool>TOOL_NAME</tool>
<code>
YOUR_PYTHON_CODE
</code>

Rules:
- Replace TOOL_NAME with: execute, lint, or generate_tests
- Put only Python code between the <code> tags
- Use execute as your final submission
- Do not add any text outside the tags
"""

PROBLEM_TEMPLATE = """\
Solve the following programming problem in Python:

{question}

Write a complete Python solution.\
"""

FEEDBACK_TEMPLATE = """\
Tool: {tool}
Result:
{output}

Continue solving. Use execute when your solution is ready.\
"""


@dataclass
class ParsedOutput:
    tool: str | None  # "execute", "lint", "generate_tests", or None if parse failed
    code: str | None  # extracted code, or None if parse failed
    raw: str  # original model output, always preserved

    @property
    def is_valid(self) -> bool:
        return self.tool is not None and self.code is not None

    @property
    def tool_name(self) -> ToolName | None:
        if self.tool == "execute":
            return ToolName.EXECUTE
        if self.tool == "lint":
            return ToolName.LINT
        if self.tool == "generate_tests":
            return ToolName.TESTS
        return None


def format_prompt(
    problem: APPSProblem,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Format a problem into a chat message list.

    Returns a list of dicts in the format:
      [{"role": "system", "content": "..."},
       {"role": "user",   "content": "..."},
       {"role": "assistant", "content": "..."},  # previous turns
       ...]

    This is the standard HuggingFace chat format that
    apply_chat_template() expects.

    Args:
        problem: APPSProblem dataclass
        history: list of previous (assistant, user) message dicts
                 from earlier turns in the same episode
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROBLEM_TEMPLATE.format(question=problem.question.strip()),
        },
    ]

    # append conversation history from previous turns
    if history:
        messages.extend(history)

    return messages


def format_feedback(tool: str, output: str) -> dict:
    """
    Format tool feedback as a user message to append to history.
    This is what the model sees after each tool call.
    """
    return {
        "role": "user",
        "content": FEEDBACK_TEMPLATE.format(tool=tool, output=output.strip()),
    }


def parse_output(text: str) -> ParsedOutput:
    """
    Parse model output to extract tool name and code.

    Expected format:
      <tool>execute</tool>
      <code>
      def solution():
          ...
      </code>

    Returns ParsedOutput with tool=None/code=None if parsing fails.
    """
    # extract tool name
    tool_match = re.search(r"<tool>\s*(\w+)\s*</tool>", text, re.IGNORECASE)
    tool = tool_match.group(1).lower().strip() if tool_match else None

    # validate tool name
    valid_tools = {"execute", "lint", "generate_tests"}
    if tool and tool not in valid_tools:
        tool = None

    # extract code block
    code_match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else None

    return ParsedOutput(tool=tool, code=code, raw=text)
