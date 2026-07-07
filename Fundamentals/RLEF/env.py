"""
env.py — Multi-Turn Trajectory Environment for OpenRLHF
Implements the step-by-step game loop and the granular reward architecture.
"""

import torch
from rlef.data import APPSProblem
from rlef.prompt import parse_output
from rlef.tools import call_tool


class CodingEnvironment:
    def __init__(self, problem: APPSProblem, max_turns: int = 5):
        self.problem = problem
        self.max_turns = max_turns
        self.current_turn = 0

        # State tracking for delta bonuses and repetition penalties
        self.previous_code = None
        self.previous_error = None
        self.is_done = False

    def reset(self) -> dict:
        """Initialize the trajectory state."""
        self.current_turn = 0
        self.previous_code = None
        self.previous_error = None
        self.is_done = False
        return {"observation": self.problem.question}

    def step(self, action_text: str) -> dict:
        """
        Executes a single trajectory turn:
        1. Parses the model's output
        2. Calculates dense formatting and penalty rewards
        3. Executes the sandbox tool
        4. Calculates functional/delta rewards
        """
        self.current_turn += 1
        step_reward = 0.0
        feedback_text = ""

        # ── 1. The Time & Length Penalties ────────────────────────────────────
        # Time Decay: -0.05 per turn taken
        step_reward -= 0.05

        # Length Penalty: Tax excessive token bloat (budget = 300 words roughly)
        token_estimate = len(action_text.split())
        if token_estimate > 300:
            overage = token_estimate - 300
            step_reward -= 0.001 * overage

        # ── 2. The Format Parsing ─────────────────────────────────────────────
        parsed = parse_output(action_text)

        # Dense Format Rewards (Max +0.20)
        if "<tool>" in action_text or "<code>" in action_text:
            step_reward += 0.05
        if parsed.tool in ["execute", "lint", "generate_tests"]:
            step_reward += 0.05
        if parsed.is_valid:
            step_reward += 0.10

        # ── 3. Rule Enforcement & Repetition ──────────────────────────────────
        if not parsed.is_valid:
            feedback_text = (
                "Format Error: Could not parse XML. Use <tool> and <code> blocks."
            )
            return self._package_step(feedback_text, step_reward, done=False)

        if parsed.code == self.previous_code:
            # The model is stuck in a loop. Terminate the episode.
            step_reward -= 0.50
            return self._package_step(
                "Repetition Error: Code unchanged.", step_reward, done=True
            )

        if self.current_turn == 1 and parsed.tool == "execute":
            # Turn 1 Execute Blockade
            step_reward -= 0.30
            feedback_text = "Rule Violation: You must use 'generate_tests' on Turn 1 before executing."
            return self._package_step(feedback_text, step_reward, done=False)

        # ── 4. Tool Execution & Dynamic Rewards ───────────────────────────────
        if parsed.tool == "generate_tests":
            feedback_text, test_reward = self._evaluate_generated_tests(parsed.code)
            step_reward += test_reward
            self.previous_code = parsed.code
            return self._package_step(feedback_text, step_reward, done=False)

        elif parsed.tool == "execute":
            # Dispatch to your gRPC sandbox
            result = call_tool(
                tool_name="execute", code=parsed.code, problem=self.problem
            )

            # Extract errors for Delta checking
            current_error = (
                result.error
                if result.error
                else ("SyntaxError" if "SyntaxError" in result.stdout else None)
            )

            # The Delta Bonus: Fixed a previous error!
            if self.previous_error and not current_error:
                step_reward += 0.25

            self.previous_error = current_error
            self.previous_code = parsed.code

            # The Final Execution Objective (Terminal State)
            if result.success and not current_error:
                # Assuming result has a pass_rate from your execution script
                pass_rate = getattr(result, "pass_rate", 1.0)
                step_reward += 1.0 * pass_rate
                feedback_text = f"Execution Success! Pass Rate: {pass_rate}"
                return self._package_step(feedback_text, step_reward, done=True)
            else:
                # Code failed, append trace for next turn
                feedback_text = f"Execution Failed:\n{result.stderr or result.stdout}"

                # End episode if max turns reached
                done = self.current_turn >= self.max_turns
                return self._package_step(feedback_text, step_reward, done=done)

        # Fallback
        return self._package_step(
            "System Error: Unhandled tool.", step_reward, done=True
        )

    # ── Helper Methods ────────────────────────────────────────────────────────

    def _package_step(self, feedback: str, reward: float, done: bool) -> dict:
        """Packages the step return variables for OpenRLHF."""
        self.is_done = done
        return {
            "observation": feedback,
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": done,
        }

    def _evaluate_generated_tests(self, generated_tests: str) -> tuple[str, float]:
        """
        Two-Sided Oracle Testing to prevent 'assert True == True' exploits.
        Returns: (feedback_string, reward_float)
        """
        # 1. Check Gold Solution
        gold_result = call_tool(
            "execute",
            code=self.problem.solutions[0] + "\n" + generated_tests,
            problem=self.problem,
        )
        gold_passed = gold_result.success and not gold_result.error

        # 2. Check Poison Solution
        poison_code = f"def {self.problem.fn_name or 'solve'}(*args, **kwargs):\n    return None\n"
        poison_result = call_tool(
            "execute", code=poison_code + "\n" + generated_tests, problem=self.problem
        )
        poison_failed = bool(poison_result.error or not poison_result.success)

        if gold_passed and poison_failed:
            return (
                "Tests valid. Gold solution passed, broken solution failed. Proceed to implementation.",
                0.20,
            )
        else:
            return (
                "Tests invalid. They either fail the gold solution or allow broken code to pass (e.g., assert True == True).",
                0.0,
            )
