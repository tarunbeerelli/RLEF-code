"""
prepare_openrlhf_data.py
Converts the APPS dataset into the flat JSONL format required by OpenRLHF.
"""

import json
from pathlib import Path

from rlef.data import load_apps_split
from rlef.prompt import TURN_1_ORACLE_PROMPT


def main():
    # Load your existing dataset
    problems = load_apps_split("data/raw/APPS", split="train")

    output_file = Path("data/openrlhf_apps_train.jsonl")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        for p in problems:
            # We seed the initial trajectory state with the Oracle Prompt
            prompt_text = f"{TURN_1_ORACLE_PROMPT}\n\nProblem:\n{p.question}"

            # OpenRLHF expects a dictionary with a 'prompt' key
            row = {"prompt": prompt_text, "problem_id": p.problem_id}
            f.write(json.dumps(row) + "\n")

    print(f"Successfully exported {len(problems)} trajectory prompts to {output_file}")


if __name__ == "__main__":
    main()
