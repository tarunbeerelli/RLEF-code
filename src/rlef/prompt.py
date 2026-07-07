"""
prompt.py — Dynamic Context Engine for RLHF Rollouts

Generates asymmetric prompts and few-shot alignments based on ablation toggles.
Enforces `<reasoning>`, `<code>`, and conditional `<edge_cases>` protocols.
"""

from rlef.data import APPSProblem
import re

# ─── 1. DYNAMIC SYSTEM PROMPT BUILDER ────────────────────────────────────────


def build_system_prompt(ablation_cfg: dict) -> str:
    """Builds the system prompt dynamically based on the current ablation run."""
    use_edge_cases = ablation_cfg.get("use_edge_cases", False)

    base_prompt = (
        "You are an expert Python algorithmic scientist operating in a multi-turn execution sandbox.\n"
        "Analyze the coding problem and implement a highly optimized solution.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. FORCED CHAIN-OF-THOUGHT: Before writing any code, you MUST 'think out loud'. "
        "Analyze constraints and edge cases, and wrap your logic explicitly inside a <reasoning>...</reasoning> block.\n"
    )

    if use_edge_cases:
        # Runs 6 & 7: Test-Driven Development
        base_prompt += (
            "2. ANCHOR & EXTEND: Extract one sample test case provided in the problem description to act as your ground-truth anchor. "
            "Generate up to 3 ADDITIONAL high-value edge cases (e.g., empty structures, bounds). "
            "Wrap your test logic inside an <edge_cases>...</edge_cases> block using Python assert statements.\n"
            "3. THE DUMMY FUNCTION THREAT: WARNING - Your generated edge cases will be evaluated against a malicious dummy function. "
            "If you write generic, tautological tests (like `assert True` or `assert type(x) == int`), they will fail the sandbox validation "
            "and you will be heavily penalized. Your tests must be mathematically rigorous.\n"
            "4. IMPLEMENTATION: Once you have reasoned and written your edge cases, wrap your functional solution inside a <code>...</code> block.\n"
        )
    else:
        # Runs 1-5: Direct Execution
        base_prompt += (
            "2. IMPLEMENTATION: Implement your final solution and wrap it explicitly inside a <code>...</code> block.\n"
            "3. ITERATION: If your code fails, review the execution errors or failed test cases provided in the next turn and refine your solution.\n"
        )

    return base_prompt


# ─── 2. FEW-SHOT ALIGNMENT GENERATORS ────────────────────────────────────────


def get_few_shot_history(use_edge_cases: bool) -> list[dict]:
    """Provides perfectly formatted XML examples to prevent format hallucinations."""

    if use_edge_cases:
        return [
            {
                "role": "user",
                "content": "Write a function `double_it(x)` that returns twice the integer input.\nSample: double_it(5) -> 10",
            },
            {
                "role": "assistant",
                "content": (
                    "<reasoning>\n"
                    "The problem requires multiplying an integer by 2. The sample provides an anchor of 5->10. "
                    "I need to test negative numbers and zero as edge cases.\n"
                    "</reasoning>\n"
                    "<edge_cases>\n"
                    "assert double_it(5) == 10\n"
                    "assert double_it(0) == 0\n"
                    "assert double_it(-3) == -6\n"
                    "</edge_cases>\n"
                    "<code>\n"
                    "def double_it(x):\n"
                    "    return x * 2\n"
                    "</code>"
                ),
            },
            {
                "role": "user",
                "content": "System Result:\nExecution Pass Rate: 100.0%.",
            },
        ]
    else:
        return [
            {
                "role": "user",
                "content": "Write a function `double_it(x)` that returns twice the integer input.",
            },
            {
                "role": "assistant",
                "content": (
                    "<reasoning>\n"
                    "The problem requires a simple multiplication by 2. I will implement a function that takes `x` and returns `x * 2`.\n"
                    "</reasoning>\n"
                    "<code>\n"
                    "def double_it(x):\n"
                    "    return x * 2\n"
                    "</code>"
                ),
            },
            {
                "role": "user",
                "content": "System Result:\nExecution Pass Rate: 100.0%.",
            },
        ]


# ─── 3. MAIN FORMATTER ───────────────────────────────────────────────────────


def format_prompt(
    problem: APPSProblem,
    ablation_cfg: dict,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Constructs the full context window for the vLLM generation request.
    Injects system prompts, few-shot alignment, and trajectory history.
    """
    use_edge_cases = ablation_cfg.get("use_edge_cases", False)
    system_content = build_system_prompt(ablation_cfg)
    few_shot = get_few_shot_history(use_edge_cases)

    # Base Context
    messages = [
        {"role": "system", "content": system_content},
        *few_shot,
        {"role": "user", "content": f"{problem.question.strip()}\n"},
    ]

    # Append multi-turn execution logs if they exist
    if history and len(history) > 0:
        messages.extend(history)

    return messages


# ─── 4. INLINE EXTRACTION PARSER ─────────────────────────────────────────────
def parse_output(text: str) -> dict:
    code_match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL | re.IGNORECASE)
    edge_match = re.search(
        r"<edge_cases>\s*(.*?)\s*</edge_cases>", text, re.DOTALL | re.IGNORECASE
    )

    return {
        "code": code_match.group(1).strip() if code_match else None,
        "edge_cases": edge_match.group(1).strip() if edge_match else None,
        "is_valid": bool(code_match),  # Code is strictly required to proceed
    }
