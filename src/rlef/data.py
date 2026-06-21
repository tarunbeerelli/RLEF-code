"""
data.py — APPS dataset loader

Each problem lives in data/raw/APPS/{split}/{id}/ with four files:
  question.txt       — problem description
  input_output.json  — {"inputs": [...], "outputs": [...], "fn_name"?: "..."}
  solutions.json     — list of correct Python solutions
  metadata.json      — {"difficulty": "introductory|interview|competition", "url": "..."}

We load these into a flat list of dicts, one per problem.
The input_output.json is the reward signal — we run the model's code
against these inputs and check the outputs match.
"""

import json
import pathlib
from dataclasses import dataclass

DIFFICULTY_ORDER = ["introductory", "interview", "competition"]


@dataclass
class APPSProblem:
    problem_id: str
    difficulty: str
    question: str
    # inputs/outputs for the executor reward
    inputs: list[str]
    outputs: list[str]
    fn_name: str | None  # if set, problem expects a function; else stdin/stdout
    solutions: list[str]  # reference solutions (never used as training targets)
    url: str


def load_apps_split(
    data_dir: str | pathlib.Path,
    split: str = "train",
    difficulties: list[str] | None = None,
) -> list[APPSProblem]:
    """
    Load all problems from one split of APPS.

    Args:
        data_dir:     path to data/raw/APPS/
        split:        "train" or "test"
        difficulties: filter to subset e.g. ["introductory", "interview"]
                      None means all difficulties

    Returns:
        list of APPSProblem, sorted by problem_id
    """
    root = pathlib.Path(data_dir) / split
    if not root.exists():
        raise FileNotFoundError(f"APPS {split} split not found at {root}")

    problems = []
    skipped = 0

    for problem_dir in sorted(root.iterdir()):
        if not problem_dir.is_dir():
            continue

        # --- load each file ---
        try:
            question = (
                (problem_dir / "question.txt").read_text(errors="replace").strip()
            )
            metadata = json.loads((problem_dir / "metadata.json").read_text())
            difficulty = metadata.get("difficulty", "unknown")
            url = metadata.get("url", "")
        except Exception:
            skipped += 1
            continue

        # filter by difficulty before loading heavier files
        if difficulties and difficulty not in difficulties:
            continue

        # input_output.json can be missing or empty for some problems
        try:
            io_data = json.loads((problem_dir / "input_output.json").read_text())
            inputs = io_data.get("inputs", [])
            outputs = io_data.get("outputs", [])
            fn_name = io_data.get("fn_name", None)
        except Exception:
            # no test cases — not useful for execution-based reward
            skipped += 1
            continue

        # skip problems with no test cases — no reward signal
        if not inputs or not outputs:
            skipped += 1
            continue

        # solutions.json can be missing (some test problems have no solutions)
        try:
            solutions = json.loads((problem_dir / "solutions.json").read_text())
        except Exception:
            solutions = []

        problems.append(
            APPSProblem(
                problem_id=problem_dir.name,
                difficulty=difficulty,
                question=question,
                inputs=inputs,
                outputs=outputs,
                fn_name=fn_name,
                solutions=solutions,
                url=url,
            )
        )

    print(
        f"Loaded {len(problems)} problems from APPS/{split} "
        f"(skipped {skipped} with missing/empty test cases)"
    )
    return problems


def difficulty_split(
    problems: list[APPSProblem],
) -> dict[str, list[APPSProblem]]:
    """Bucket problems by difficulty."""
    buckets: dict[str, list[APPSProblem]] = {d: [] for d in DIFFICULTY_ORDER}
    for p in problems:
        if p.difficulty in buckets:
            buckets[p.difficulty].append(p)
    for d, ps in buckets.items():
        print(f"  {d}: {len(ps)}")
    return buckets
