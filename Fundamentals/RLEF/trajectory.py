"""
trajectory.py — Multi-turn episode manager

A Trajectory represents one full episode:
  - one APPS problem
  - up to max_turns attempts
  - each turn: model output → tool call → tool result
  - ends when: model calls execute and code passes, OR max_turns reached

The trajectory accumulates everything needed to:
  1. Pass context back to the model each turn (conversation history)
  2. Compute the final reward (last execute result)
  3. Assign step credits (list of tool results)
  4. Log to WandB (turn count, tool usage, error types)

Single-turn mode: max_turns=1, no history passed back.
Multi-turn mode:  max_turns=N, history grows each turn.
"""

from dataclasses import dataclass, field
from enum import Enum

from rlef.data import APPSProblem
from rlef.prompt import ParsedOutput, format_feedback
from rlef.reward import ExecutionResult, StepCredit, assign_step_credit
from rlef.tools import ToolResult


class EpisodeStatus(str, Enum):
    RUNNING = "running"
    SOLVED = "solved"  # model called execute and passed all tests
    PARTIAL = "partial"  # max turns reached, partial credit
    FAILED = "failed"  # max turns reached, zero reward
    FORMAT_ERROR = "format_error"  # model never produced valid output


@dataclass
class Turn:
    turn_idx: int
    parsed: ParsedOutput  # what the model said
    tool_result: ToolResult  # what the tool returned
    exec_result: ExecutionResult | None  # only set for execute calls


@dataclass
class Trajectory:
    problem: APPSProblem
    max_turns: int
    reward_type: str = "continuous"
    shaped: bool = True
    credit_type: str = "step"

    turns: list[Turn] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    status: EpisodeStatus = EpisodeStatus.RUNNING
    final_reward: float = 0.0
    step_credits: list[StepCredit] = field(default_factory=list)

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def current_turn(self) -> int:
        return len(self.turns)

    @property
    def is_done(self) -> bool:
        return self.status != EpisodeStatus.RUNNING

    @property
    def tool_usage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.turns:
            name = t.parsed.tool or "invalid"
            counts[name] = counts.get(name, 0) + 1
        return counts

    @property
    def error_types(self) -> list[str]:
        types = []
        for t in self.turns:
            if t.exec_result:
                types.extend(t.exec_result.error_types)
        return types

    # ── Turn management ───────────────────────────────────────────────────────

    def add_turn(
        self,
        parsed: ParsedOutput,
        tool_result: ToolResult,
        exec_result: ExecutionResult | None = None,
    ) -> None:
        """
        Record a completed turn and update trajectory state.
        Called by the training loop after each model generation + tool call.
        """
        turn = Turn(
            turn_idx=self.current_turn,
            parsed=parsed,
            tool_result=tool_result,
            exec_result=exec_result,
        )
        self.turns.append(turn)

        # append assistant response to history
        self.history.append(
            {
                "role": "assistant",
                "content": parsed.raw,
            }
        )

        # append tool feedback as next user message
        self.history.append(
            format_feedback(parsed.tool or "unknown", tool_result.output)
        )

        # check termination conditions
        self._update_status(exec_result)

    def _update_status(self, exec_result: ExecutionResult | None) -> None:
        """Update status and compute reward if episode is done."""
        # format error on first turn — end immediately
        if not self.turns[-1].parsed.is_valid:
            if self.current_turn == 1:
                self.status = EpisodeStatus.FORMAT_ERROR
                self._finalise(0.0)
                return

        # executed and got a result
        if exec_result is not None:
            if exec_result.pass_rate == 1.0:
                self.status = EpisodeStatus.SOLVED
                self._finalise(exec_result.final_reward)
                return

        # max turns reached
        if self.current_turn >= self.max_turns:
            last_exec = self._last_exec_result()
            if last_exec and last_exec.final_reward > 0:
                self.status = EpisodeStatus.PARTIAL
                self._finalise(last_exec.final_reward)
            else:
                self.status = EpisodeStatus.FAILED
                self._finalise(0.0)

    def _last_exec_result(self) -> ExecutionResult | None:
        """Return the most recent execution result, if any."""
        for turn in reversed(self.turns):
            if turn.exec_result is not None:
                return turn.exec_result
        return None

    def _finalise(self, reward: float) -> None:
        """Compute final reward and step credits."""
        self.final_reward = reward
        tool_results = [t.tool_result for t in self.turns]
        self.step_credits = assign_step_credit(
            tool_results,
            final_reward=reward,
            credit_type=self.credit_type,
        )

    # ── Logging helper ────────────────────────────────────────────────────────

    def to_log_dict(self) -> dict:
        """Flat dict of metrics for WandB logging."""
        return {
            "episode/status": self.status.value,
            "episode/turns": self.current_turn,
            "episode/final_reward": self.final_reward,
            "episode/pass_rate": self._last_exec_result().pass_rate
            if self._last_exec_result()
            else 0.0,
            "tools/execute": self.tool_usage.get("execute", 0),
            "tools/lint": self.tool_usage.get("lint", 0),
            "tools/generate_tests": self.tool_usage.get("generate_tests", 0),
            "tools/invalid": self.tool_usage.get("invalid", 0),
        }
