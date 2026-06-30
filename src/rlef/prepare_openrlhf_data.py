import json
from pathlib import Path

from rlef.data import load_apps_split


def export_prompts_for_openrlhf(data_dir: str, output_path: str):
    problems = load_apps_split(data_dir, split="train")

    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "w") as f:
        for prob in problems:
            # Package test cases alongside the prompt to feed OpenRLHF's label tracking
            payload = {
                "prompt": f"System: You are an expert programming agent. Implement the solution inside a code block.\nHuman: {prob.prompt}\nAssistant:",
                "label": json.dumps(
                    {
                        "inputs": prob.inputs,
                        "outputs": prob.outputs,
                        "fn_name": prob.fn_name,
                        "difficulty": prob.difficulty,
                    }
                ),
            }
            f.write(json.dumps(payload) + "\n")

    print(f"Exported {len(problems)} prompts to {output_path}")


if __name__ == "__main__":
    export_prompts_for_openrlhf("data/processed", "data/openrlhf_prompts.jsonl")
