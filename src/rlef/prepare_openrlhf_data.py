"""
prepare_openrlhf_data.py
Converts the APPS dataset into the flat JSONL format required by the custom agent loop.
Dynamically injects anchors and prompt structures based on the train.yaml ablation config.
Generates a static evaluation set if missing, reading paths directly from the config.
"""

import json
from pathlib import Path

import yaml
import random
from transformers import AutoTokenizer

from rlef.data import load_apps_split
from rlef.prompt import format_prompt


def main():
    # 1. Load the central configuration
    print("Reading ablation config from train.yaml...")
    try:
        with open("configs/train.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("CRITICAL: configs/train.yaml not found. Please ensure the file exists.")
        return

    ablation_cfg = cfg.get("ablation", {})
    evaluation_cfg = cfg.get("evaluation", {})
    use_edge_cases = ablation_cfg.get("use_edge_cases", False)

    # Target output paths defined in yaml
    output_path_str = ablation_cfg.get(
        "dataset_path", cfg.get("dataset_path", "data/openrlhf_apps_train.jsonl")
    )
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    eval_path_str = evaluation_cfg.get("dataset_path", "data/apps_eval.jsonl")
    eval_file = Path(eval_path_str)
    eval_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading raw APPS train split...")
    problems = load_apps_split("data/raw/APPS", split="train")

    # Deterministic shuffle for identical splits every time
    random.seed(42)
    random.shuffle(problems)

    # ─── STRATIFIED CAP ACROSS DIFFICULTY ──────────────────────────────────
    # Cap total training problems while preserving the difficulty distribution.
    # `train_cap` and `stratify_mode` come from train.yaml so run_pipeline controls them.
    train_cap = cfg.get("train_cap", ablation_cfg.get("train_cap", 1200))
    stratify_mode = cfg.get("stratify_mode", "proportional")  # or "balanced"

    from collections import defaultdict

    by_diff = defaultdict(list)
    for p in problems:
        by_diff[p.difficulty].append(p)

    diff_order = ["introductory", "interview", "competition"]
    available = {d: len(by_diff.get(d, [])) for d in diff_order}
    total_available = sum(available.values())
    print(
        f"Available verified train problems by difficulty: {available} (total {total_available})"
    )

    if train_cap >= total_available:
        # Use everything, but still report the distribution.
        selected = []
        for d in diff_order:
            selected.extend(by_diff.get(d, []))
        print(
            f"train_cap ({train_cap}) >= available ({total_available}); using all problems."
        )
    elif stratify_mode == "balanced":
        # Equal count per difficulty, clamped by availability.
        per = train_cap // len(diff_order)
        selected = []
        for d in diff_order:
            pool = by_diff.get(d, [])
            random.shuffle(pool)
            take = pool[:per]
            selected.extend(take)
            print(f"  balanced {d}: took {len(take)} (requested {per})")
    else:
        # Proportional: preserve the natural difficulty ratio.
        selected = []
        for d in diff_order:
            pool = by_diff.get(d, [])
            random.shuffle(pool)
            share = int(round(train_cap * (available[d] / total_available)))
            take = pool[:share]
            selected.extend(take)
            print(
                f"  proportional {d}: took {len(take)} "
                f"({available[d]/total_available:.1%} of cap {train_cap})"
            )

    random.shuffle(selected)  # mix difficulties so batches are heterogeneous
    problems = selected
    print(
        f"Stratified training set locked: {len(problems)} problems "
        f"(mode={stratify_mode}, cap={train_cap})"
    )

    # Automatically slice the dataset based on the target filename (curriculum).
    # NOTE: phase slicing now operates on the already-stratified set, so each
    # curriculum half retains a difficulty mix rather than being ordered by it.
    if "phase1" in output_path_str:
        problems = problems[: len(problems) // 2]
        print(f"Curriculum Phase 1 Detected: Subsetting first {len(problems)} rows.")
    elif "phase2" in output_path_str:
        mid = len(problems) // 2
        split_b = problems[mid:]
        replay = random.sample(problems[:mid], max(1, int(mid * 0.10)))
        problems = split_b + replay
        random.shuffle(problems)
        print(
            f"Curriculum Phase 2 Detected: Subsetting second half + 10% replay ({len(problems)} rows)."
        )

    print("Initializing Qwen Tokenizer for chat template formatting...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    # --- 1. GENERATE TRAINING DATA ---
    with open(output_file, "w", encoding="utf-8") as f:
        for p in problems:
            # 2. THE ANCHOR INJECTION (For Runs 6 & 7)
            if use_edge_cases and p.inputs and p.outputs and p.fn_name:
                anchor_text = (
                    "\n\n=== GROUND TRUTH ANCHOR ===\n"
                    "Use this exact syntax for your <edge_cases> block:\n"
                    f"assert {p.fn_name}({repr(p.inputs[0])}) == {repr(p.outputs[0])}\n"
                    "===========================\n"
                )
                p.question = p.question.strip() + anchor_text

            # 3. Build the dynamic prompt based on the ablation config
            messages = format_prompt(p, ablation_cfg)

            # 4. Apply Qwen's exact chat template tokens
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 5. Pack the row (Keep hidden tests for train_agent.py to grade locally)
            row = {
                "problem_id": p.problem_id,
                "difficulty": p.difficulty,
                "prompt": prompt_text,
                "inputs": p.inputs,
                "outputs": p.outputs,
                "fn_name": p.fn_name,
            }
            f.write(json.dumps(row) + "\n")

    print(
        f"\nSuccessfully exported {len(problems)} trajectory prompts to {output_file}"
    )
    if use_edge_cases:
        print(
            "Notice: Explicit 'Anchor & Extend' Ground Truth test cases were injected into the prompts."
        )
    else:
        print("Notice: Standard execution prompts generated (No edge case injection).")

    # --- 2. GENERATE STATIC EVAL DATA (IF MISSING) ---
    if not eval_file.exists():
        print(
            f"\nEvaluation dataset missing. Generating static eval split at {eval_file}..."
        )
        try:
            eval_problems = load_apps_split("data/raw/APPS", split="test")
            random.seed(42)
            # Sample a KNOWN count per difficulty so eval denominators are real.
            # evaluate.py takes up to 250 per bucket; provide that many when available.
            PER_BUCKET = 250
            from collections import defaultdict

            by_diff = defaultdict(list)
            for p in eval_problems:
                by_diff[p.difficulty].append(p)
            balanced = []
            for diff in ["introductory", "interview", "competition"]:
                pool = by_diff.get(diff, [])
                random.shuffle(pool)
                take = pool[:PER_BUCKET]
                balanced.extend(take)
                print(f"  eval {diff}: {len(take)} problems")
            eval_problems = balanced

            with open(eval_file, "w", encoding="utf-8") as f:
                for p in eval_problems:
                    # Baseline zero-shot prompt for evaluation consistency
                    eval_messages = format_prompt(p, {"max_turns": 1})
                    eval_prompt = tokenizer.apply_chat_template(
                        eval_messages, tokenize=False, add_generation_prompt=True
                    )
                    row = {
                        "problem_id": p.problem_id,
                        "difficulty": p.difficulty,
                        "prompt": eval_prompt,
                        "inputs": p.inputs,
                        "outputs": p.outputs,
                        "fn_name": p.fn_name,
                    }
                    f.write(json.dumps(row) + "\n")
            print(
                f"Successfully generated {len(eval_problems)} evaluation prompts to {eval_file}"
            )
        except Exception as e:
            print(
                f"Warning: Could not generate eval split. Ensure 'test' split exists. Error: {e}"
            )


if __name__ == "__main__":
    main()
