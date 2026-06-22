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

SYSTEM_PROMPT = """You are an expert Python programmer. Solve the given programming problem by writing a complete Python solution.
Output ONLY a Python code block like this:

```python
# your solution here
```

No explanations before or after the code block.
"""

PROBLEM_TEMPLATE = """{question}
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
    Parse model output to extract code.

    Handles two formats:
      1. XML tags:   <tool>execute</tool><code>...</code>
      2. Markdown:   ```python ... ```

    XML format: tool name must be valid, otherwise tool=None (invalid).
    Markdown format: tool defaults to "execute" since model answered naturally.
    """
    # try XML format first
    tool_match = re.search(r"<tool>\s*(\w+)\s*</tool>", text, re.IGNORECASE)
    xml_tool = tool_match.group(1).lower().strip() if tool_match else None
    valid_tools = {"execute", "lint", "generate_tests"}

    code_match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL)
    xml_code = code_match.group(1).strip() if code_match else None

    # if XML tags found, validate strictly
    if tool_match or code_match:
        tool = xml_tool if xml_tool in valid_tools else None
        return ParsedOutput(tool=tool, code=xml_code, raw=text)

    # fallback: try markdown code block — default tool to execute
    md_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if md_match:
        code = md_match.group(1).strip()
        return ParsedOutput(tool="execute", code=code, raw=text)

    return ParsedOutput(tool=None, code=None, raw=text)
