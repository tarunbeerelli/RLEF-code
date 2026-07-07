"""
eval.py — Focused Multi-Turn Evaluation on the APPS Test Split.

Executes the stateful multi-turn TDD agent loop on a stratified sample of the
APPS test dataset, allowing the model to use test-time compute to self-correct.
Reports pass@1 broken down by difficulty and exports full trajectory logs.

Run execution:
  python -m rlef.eval --checkpoint none        --tag baseline
  python -m rlef.eval --checkpoint ./checkpoints/final --tag trained
"""

import argparse
import json
import random
from pathlib import Path

import torch
from dotenv import load_dotenv
from rlef.agent import run_agent_trajectory
from rlef.data import difficulty_split, load_apps_split
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# ── Stratified Evaluation Sampling ───────────────────────────────────────────


def stratified_sample_eval(
    problems: list,
    n: int,
) -> list:
    """
    Sample n evaluation problems with equal representation from each difficulty tier.
    Ensures a balanced, predictable benchmarking footprint.
    """
    if len(problems) <= n:
        return problems

    buckets = difficulty_split(problems)
    per_bucket = n // len(buckets)
    sampled = []

    for difficulty, bucket_problems in buckets.items():
        # Fallback handle if a tier has fewer items available than the split target
        k = min(per_bucket, len(bucket_problems))
        # Fixed seed sampling to maintain stable comparative baselines across runs
        rng = random.Random(42)
        sampled.extend(rng.sample(bucket_problems, k))

    return sampled


# ── APPS Multi-Turn Evaluation Engine ─────────────────────────────────────────


def eval_apps(
    model,
    tokenizer,
    data_dir: str,
    tag: str,
    max_examples: int | None = None,
    device: str = "cuda",
    difficulties: list[str] | None = None,
    max_turns: int = 5,
) -> dict:
    """
    Evaluate on APPS test split using the stateful multi-turn TDD agent loop.
    """
    problems = load_apps_split(data_dir, split="test", difficulties=difficulties)

    # Apply stratified balance sampling if limiting the validation test set size
    if max_examples:
        print(
            f"📊 Applying stratified sampling to extract {max_examples} balanced problems from test pool..."
        )
        problems = stratified_sample_eval(problems, max_examples)

    print(f"🔬 Total active evaluation instances queued: {len(problems)}")
    results = []

    for i, problem in enumerate(problems):
        # Fire the complete multi-turn rollout state machine
        trajectory = run_agent_trajectory(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            device=device,
            max_turns=max_turns,
        )

        results.append(
            {
                "problem_id": problem.problem_id,
                "difficulty": problem.difficulty,
                "status": trajectory.status.value,
                "pass_rate": trajectory.final_reward,
                "turns_taken": trajectory.current_turn,
                "tool_usage": trajectory.tool_usage,
                "error_types": trajectory.error_types,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            solved = sum(1 for r in results if r["pass_rate"] == 1.0)
            print(
                f"  [{i+1}/{len(problems)}] Multi-Turn pass@1 so far: {solved/(i+1):.1%}"
            )

    # Compute summary breakdown across difficulty tiers
    summary = _compute_summary(results, tag, benchmark="apps")

    # Save full telemetry results for diagnostic analysis
    out_path = Path(f"results/apps_eval_{tag}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved trajectory logs to {out_path}")

    return summary


# ── Summary Aggregator ────────────────────────────────────────────────────────


def _compute_summary(results: list[dict], tag: str, benchmark: str) -> dict:
    """Compute overall and per-difficulty pass@1 metrics."""
    total = len(results)
    solved = sum(1 for r in results if r.get("pass_rate", 0) == 1.0)

    by_difficulty: dict[str, dict] = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        if d not in by_difficulty:
            by_difficulty[d] = {"solved": 0, "total": 0}
        by_difficulty[d]["total"] += 1
        by_difficulty[d]["solved"] += 1 if r.get("pass_rate", 0) == 1.0 else 0

    summary = {
        "tag": tag,
        "benchmark": benchmark,
        "total": total,
        "solved": solved,
        "pass_at_1": solved / total if total > 0 else 0.0,
        "by_difficulty": {
            d: {
                "pass_at_1": v["solved"] / v["total"] if v["total"] > 0 else 0.0,
                "solved": v["solved"],
                "total": v["total"],
            }
            for d, v in by_difficulty.items()
        },
    }

    print(f"\n=== {benchmark.upper()} eval [{tag}] ===")
    print(f"Overall pass@1: {summary['pass_at_1']:.1%} ({solved}/{total})")
    for d, v in summary["by_difficulty"].items():
        print(f"  {d}: {v['pass_at_1']:.1%} ({v['solved']}/{v['total']})")

    return summary


# ── Model & Tokenizer Loader ──────────────────────────────────────────────────


def load_model(checkpoint: str, device: str = "cuda"):
    """Load model architecture and verify left-padding constraints."""
    model_path = (
        checkpoint if checkpoint != "none" else "Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    print(f"Loading model architecture from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── Entrypoint CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-Turn Evaluation Suite for RLEF")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="none",
        help="Path to checkpoint directory or 'none'",
    )
    parser.add_argument(
        "--tag", type=str, required=True, help="Run tag identifier (e.g., 'baseline')"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/raw/APPS", help="Data directory path"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=600,
        help="Cap evaluation subset slice bounds via stratified sample",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target execution architecture accelerator device",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Turn rollout threshold limits per problem allocation",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="+",
        default=None,
        help="Filter specifically by difficulty tier names",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.device)

    eval_apps(
        model=model,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        tag=args.tag,
        max_examples=args.max_examples,
        device=args.device,
        difficulties=args.difficulties,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
