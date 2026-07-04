"""
prepare_openrlhf_data.py
Converts the APPS dataset into the flat JSONL format required by the custom agent loop.
"""

import json
from pathlib import Path

from rlef.data import load_apps_split
from rlef.prompt import format_prompt
from transformers import AutoTokenizer


def main():
    print("Loading raw APPS train split...")
    problems = load_apps_split("data/raw/APPS", split="train")

    print("Initializing Qwen Tokenizer for chat template formatting...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    output_file = Path("data/openrlhf_apps_train.jsonl")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        for p in problems:
            # 1. Use the new robust formatting logic (includes few-shot examples)
            messages = format_prompt(p)

            # 2. Apply Qwen's exact chat template tokens
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 3. CRITICAL: Include the test cases so train_agent.py can locally grade!
            row = {
                "problem_id": p.problem_id,
                "difficulty": p.difficulty,
                "prompt": prompt_text,
                "inputs": p.inputs,
                "outputs": p.outputs,
                "fn_name": p.fn_name,
            }
            f.write(json.dumps(row) + "\n")

    print(f"Successfully exported {len(problems)} trajectory prompts to {output_file}")


if __name__ == "__main__":
    main()
