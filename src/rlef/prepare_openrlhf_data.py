"""
prepare_openrlhf_data.py
Converts the APPS dataset into the flat JSONL format required by the custom agent loop.
Dynamically injects anchors and prompt structures based on the train.yaml ablation config.
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
    use_edge_cases = ablation_cfg.get("use_edge_cases", False)

    # Target output path defined in yaml (safeguards your different phase datasets)
    output_path_str = ablation_cfg.get(
        "dataset_path", cfg.get("dataset_path", "data/openrlhf_apps_train.jsonl")
    )
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading raw APPS train split...")
    problems = load_apps_split("data/raw/APPS", split="train")

    # Deterministic shuffle for identical splits every time
    random.seed(42)
    random.shuffle(problems)

    # Automatically slice the dataset based on the target filename
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


if __name__ == "__main__":
    main()
