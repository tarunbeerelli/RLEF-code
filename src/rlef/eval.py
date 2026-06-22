"""
eval.py — Evaluation on APPS test split and HumanEval+

Two evaluation modes:

  eval_apps(model, tokenizer, cfg)
      Runs greedy decoding on the APPS test split.
      Reports pass@1 broken down by difficulty.
      Saves full results to results/apps_eval_{tag}.json

  eval_humaneval(model, tokenizer, cfg)
      Runs greedy decoding on HumanEval+ via evalplus.
      Reports pass@1 overall.
      Saves full results to results/humaneval_eval_{tag}.json

Why two benchmarks:
  APPS test split — same distribution as training, measures in-distribution
  HumanEval+     — different distribution, measures generalisation

Run before training (baseline) and after training (trained):
  python -m rlef.eval --checkpoint none        --tag baseline
  python -m rlef.eval --checkpoint ./checkpoints/final --tag trained
"""

import argparse
import json
from pathlib import Path

import torch
from rlef.data import load_apps_split
from rlef.prompt import format_prompt, parse_output
from rlef.reward import execution_reward
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Greedy decode ─────────────────────────────────────────────────────────────


def generate_solution(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    device: str = "cuda",
) -> str:
    """
    Greedy decode a solution from the model given a prompt.
    Greedy (temperature=0, no sampling) is standard for eval —
    we want the model's single best answer, not a distribution.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── APPS evaluation ───────────────────────────────────────────────────────────


def eval_apps(
    model,
    tokenizer,
    data_dir: str,
    tag: str,
    max_examples: int | None = None,
    device: str = "cuda",
) -> dict:
    """
    Evaluate on APPS test split.
    Returns summary dict with overall and per-difficulty pass@1.
    """
    problems = load_apps_split(data_dir, split="test")
    if max_examples:
        problems = problems[:max_examples]

    results = []

    for i, problem in enumerate(problems):
        messages = format_prompt(problem)
        completion = generate_solution(model, tokenizer, messages, device=device)
        parsed = parse_output(completion)

        if not parsed.is_valid or parsed.tool != "execute" or not parsed.code:
            results.append(
                {
                    "problem_id": problem.problem_id,
                    "difficulty": problem.difficulty,
                    "status": "format_error",
                    "pass_rate": 0.0,
                    "passed": 0,
                    "total": len(problem.inputs),
                }
            )
            continue

        exec_result = execution_reward(
            code=parsed.code,
            inputs=problem.inputs,
            outputs=problem.outputs,
            fn_name=problem.fn_name,
            reward_type="continuous",
            shaped=False,  # raw pass rate for eval — no shaping
        )

        results.append(
            {
                "problem_id": problem.problem_id,
                "difficulty": problem.difficulty,
                "status": "solved" if exec_result.pass_rate == 1.0 else "partial",
                "pass_rate": exec_result.pass_rate,
                "passed": exec_result.passed,
                "total": exec_result.total,
                "error_types": exec_result.error_types,
            }
        )

        if (i + 1) % 50 == 0:
            solved = sum(1 for r in results if r["pass_rate"] == 1.0)
            print(f"  [{i+1}/{len(problems)}] pass@1 so far: {solved/(i+1):.1%}")

    # compute summary
    summary = _compute_summary(results, tag, benchmark="apps")

    # save full results
    out_path = Path(f"results/apps_eval_{tag}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved to {out_path}")

    return summary


# ── HumanEval+ evaluation ─────────────────────────────────────────────────────


def eval_humaneval(
    model,
    tokenizer,
    tag: str,
    max_examples: int | None = None,
    device: str = "cuda",
) -> dict:
    """
    Evaluate on HumanEval+ via evalplus.
    Uses evalplus's own test runner for correctness.
    """
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()
    items = list(problems.items())
    if max_examples:
        items = items[:max_examples]

    results = []
    generations = {}  # task_id -> [completion] for evalplus scorer

    for task_id, problem in items:
        # wrap evalplus problem as a minimal prompt
        messages = [
            {"role": "system", "content": "You are an expert Python programmer."},
            {
                "role": "user",
                "content": (
                    f"Complete the following Python function:\n\n"
                    f"{problem['prompt']}\n\n"
                    f"Respond in this format:\n"
                    f"<tool>execute</tool>\n<code>\nYOUR CODE HERE\n</code>"
                ),
            },
        ]

        completion = generate_solution(model, tokenizer, messages, device=device)
        parsed = parse_output(completion)

        code = parsed.code if parsed.is_valid else ""
        # evalplus expects the full function including the prompt
        full_code = problem["prompt"] + "\n" + code if code else ""
        generations[task_id] = [full_code]

        results.append(
            {
                "task_id": task_id,
                "difficulty": "unknown",
                "has_code": bool(code),
            }
        )

    # use evalplus to score
    try:
        from evalplus.evaluate import evaluate as evalplus_evaluate

        eval_results = evalplus_evaluate(
            dataset="humaneval",
            samples=generations,
        )
        # merge pass rates back into results
        for r in results:
            tid = r["task_id"]
            r["pass_rate"] = (
                1.0 if eval_results.get(tid, {}).get("base_status") == "pass" else 0.0
            )
    except Exception as e:
        print(f"evalplus scorer failed: {e} — using format check only")
        for r in results:
            r["pass_rate"] = 0.0

    summary = _compute_summary(results, tag, benchmark="humaneval")

    out_path = Path(f"results/humaneval_eval_{tag}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved to {out_path}")

    return summary


# LiveCodeBench eval
def eval_livecodebench(
    model,
    tokenizer,
    tag: str,
    max_examples: int | None = None,
    device: str = "cuda",
) -> dict:
    """
    Evaluate on LiveCodeBench — contamination-proof eval.

    LiveCodeBench pulls problems from Codeforces, LeetCode, and AtCoder
    with a release date AFTER most model training cutoffs.
    This is our out-of-distribution generalisation check.

    Uses the livecodebench package for problem loading and scoring.
    Install: poetry add livecodebench
    """
    try:
        from datasets import load_dataset

        # LiveCodeBench is available on HuggingFace
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"LiveCodeBench load failed: {e}")
        print("Skipping LiveCodeBench eval.")
        return {}

    items = list(ds)
    if max_examples:
        items = items[:max_examples]

    results = []

    for item in items:
        messages = [
            {"role": "system", "content": "You are an expert Python programmer."},
            {
                "role": "user",
                "content": (
                    f"Solve this programming problem in Python:\n\n"
                    f"{item['question_content']}\n\n"
                    f"Respond in this format:\n"
                    f"<tool>execute</tool>\n<code>\nYOUR SOLUTION\n</code>"
                ),
            },
        ]

        completion = generate_solution(model, tokenizer, messages, device=device)
        parsed = parse_output(completion)

        if not parsed.is_valid or not parsed.code:
            results.append(
                {
                    "task_id": item.get("question_id", "unknown"),
                    "difficulty": item.get("difficulty", "unknown"),
                    "pass_rate": 0.0,
                }
            )
            continue

        # LiveCodeBench provides public test cases
        test_inputs = item.get("public_test_cases", [])
        inputs = [t["input"] for t in test_inputs] if test_inputs else []
        outputs = [t["output"] for t in test_inputs] if test_inputs else []

        if not inputs:
            results.append(
                {
                    "task_id": item.get("question_id", "unknown"),
                    "difficulty": item.get("difficulty", "unknown"),
                    "pass_rate": 0.0,
                }
            )
            continue

        exec_result = execution_reward(
            code=parsed.code,
            inputs=inputs,
            outputs=outputs,
            fn_name=None,
            reward_type="continuous",
            shaped=False,
        )

        results.append(
            {
                "task_id": item.get("question_id", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "pass_rate": exec_result.pass_rate,
            }
        )

    summary = _compute_summary(results, tag=tag, benchmark="livecodebench")

    out_path = Path(f"results/livecodebench_eval_{tag}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved to {out_path}")

    return summary


# ── Summary ───────────────────────────────────────────────────────────────────


def _compute_summary(results: list[dict], tag: str, benchmark: str) -> dict:
    """Compute overall and per-difficulty pass@1."""
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


# ── CLI ───────────────────────────────────────────────────────────────────────


def load_model(checkpoint: str, device: str = "cuda"):
    """Load model and tokenizer from checkpoint or HuggingFace hub."""
    from transformers import BitsAndBytesConfig

    model_path = (
        checkpoint if checkpoint != "none" else "Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # use 4-bit for eval too — same memory constraints as training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="none",
        help="Path to checkpoint or 'none' for base model",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Label for this eval run e.g. 'baseline' or 'trained'",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="both",
        choices=["apps", "humaneval", "livecodebench", "both", "all"],
    )
    parser.add_argument("--data_dir", type=str, default="data/raw/APPS")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.device)

    if args.benchmark in ("apps", "both"):
        eval_apps(
            model, tokenizer, args.data_dir, args.tag, args.max_examples, args.device
        )

    if args.benchmark in ("humaneval", "both"):
        eval_humaneval(model, tokenizer, args.tag, args.max_examples, args.device)

    if args.benchmark in ("livecodebench", "all"):
        eval_livecodebench(model, tokenizer, args.tag, args.max_examples, args.device)


if __name__ == "__main__":
    main()
