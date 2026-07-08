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
    max_turns = ablation_cfg.get("max_turns", 5)
    feedback_type = ablation_cfg.get("feedback_type", "last_failed")

    base_prompt = (
        "You are an expert Python algorithmic scientist operating in an execution sandbox.\n"
        "Analyze the coding problem and implement a highly optimized solution.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. STRICT CONCISENESS: You operate under a severe token limit. Keep your reasoning brief and direct. Write efficient code with zero bloated comments.\n"
        "2. FORCED CHAIN-OF-THOUGHT: Before writing any code, you MUST 'think out loud'. "
        "Analyze constraints and edge cases, and wrap your logic explicitly inside a <reasoning>...</reasoning> block.\n"
    )

    if use_edge_cases:
        # Runs 6 & 7 (Phase 2): Test-Driven Development
        base_prompt += (
            "3. ANCHOR & EXTEND: Extract the ground-truth anchor provided in the prompt. "
            "Generate up to 3 ADDITIONAL high-value edge cases (e.g., empty structures, bounds). "
            "Wrap your test logic inside an <edge_cases>...</edge_cases> block using Python assert statements.\n"
            "4. TEST VALIDATION: Generic or unrelated tests (e.g., `assert True`) will be penalized. Your tests must be mathematically rigorous.\n"
            "5. IMPLEMENTATION: Once you have reasoned and written your edge cases, wrap your functional solution inside a <code>...</code> block.\n"
        )
        step_num = 6
    else:
        # Runs 1-5, 7 (Phase 1): Direct Execution
        base_prompt += "3. IMPLEMENTATION: Implement your final solution and wrap it explicitly inside a <code>...</code> block.\n"
        step_num = 4

    # Dynamic Iteration Instruction based on physical sandbox toggles
    if max_turns > 1:
        if feedback_type == "none":
            feedback_desc = "the overall execution pass rate"
        elif feedback_type == "consolidated":
            feedback_desc = "the execution pass rate and a summary of error types"
        elif feedback_type == "last_failed":
            feedback_desc = "the execution pass rate and the specific input/output of a failed test case"
        else:
            feedback_desc = "the execution pass rate"

        base_prompt += f"{step_num}. ITERATION: If your code fails, review {feedback_desc} provided in the next turn and refine your solution.\n"
    else:
        base_prompt += (
            f"{step_num}. ONE-SHOT EXECUTION: You only have ONE attempt to solve this problem. "
            "Ensure your logic is flawless before outputting the <code> block, as there is no second turn.\n"
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
    reason_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL | re.IGNORECASE
    )
    code_match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL | re.IGNORECASE)
    edge_match = re.search(
        r"<edge_cases>\s*(.*?)\s*</edge_cases>", text, re.DOTALL | re.IGNORECASE
    )

    return {
        "has_reasoning": bool(reason_match),
        "code": code_match.group(1).strip() if code_match else None,
        "edge_cases": edge_match.group(1).strip() if edge_match else None,
        "is_valid": bool(code_match),
    }
