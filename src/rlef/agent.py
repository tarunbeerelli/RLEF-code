"""
agent.py — Multi-turn agent loop for RLEF-Code
Orchestrates the active conversation loop, handles stop-token constraints,
and drives changes through the Trajectory state manager.
"""

from rlef.data import APPSProblem
from rlef.prompt import format_prompt, parse_output
from rlef.tools import call_tool
from rlef.trajectory import Trajectory


def run_agent_trajectory(
    model,
    tokenizer,
    problem: APPSProblem,
    device: str,
    max_turns: int = 5,
    credit_type: str = "step",
) -> Trajectory:
    """
    Executes a complete multi-turn trajectory episode for an APPS problem.
    """
    # Initialize your native trajectory state tracker
    trajectory = Trajectory(
        problem=problem, max_turns=max_turns, credit_type=credit_type
    )

    while not trajectory.is_done:
        # 1. Format the asymmetric conversation history
        messages = format_prompt(problem=problem, history=trajectory.history)

        # 2. Build template token strings
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        # 3. Prevent hallucination by configuring generation to stop on the closing XML block tag
        outputs = model.generate(
            **inputs, max_new_tokens=512, stop_strings=["</code>"], tokenizer=tokenizer
        )

        # Extract only the freshly generated tokens
        prompt_len = inputs.input_ids.shape[1]
        raw_generation = tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        )

        # 4. Parse the action block via our XML rules
        parsed = parse_output(raw_generation)

        # 5. Handle Repetition Guard / Thought Guidance Intervention
        if len(trajectory.turns) >= 2:
            last_turn = trajectory.turns[-1]
            if (
                last_turn.parsed.tool == parsed.tool
                and last_turn.parsed.code == parsed.code
            ):
                # Intercept execution and inject an operational correction notification
                tool_output = "SYSTEM CRITIQUE: You repeated the exact same tool call and code parameters. You must change your algorithmic strategy or adjust your formatting structure."
                trajectory.add_turn(
                    parsed=parsed,
                    tool_result=type("Result", (object,), {"output": tool_output})(),
                    exec_result=None,
                )
                continue

        # 6. Dispatch to environment sandbox if valid tool call detected
        if parsed.is_valid:
            # Dispatch directly to tools.py router
            tool_result = call_tool(
                tool_name=parsed.tool, code=parsed.code, problem=problem.prompt
            )

            # Extract execution metadata if the action hit the sandbox runner
            exec_metadata = getattr(tool_result, "exec_result", None)

            # Step progress recorded inside the tracking database block
            trajectory.add_turn(
                parsed=parsed, tool_result=tool_result, exec_result=exec_metadata
            )
        else:
            # If the model fails to follow the schema formatting rules, handle format tracking termination
            trajectory.add_turn(
                parsed=parsed,
                tool_result=type(
                    "ErrorResult",
                    (object,),
                    {
                        "output": "Formatting Exception. Unrecognized XML configuration structure."
                    },
                )(),
                exec_result=None,
            )

    return trajectory
