"""
evaluate.py — High-Throughput Stratified Evaluation Script
Ingests raw APPS test folders, formats them into OpenRLHF style prompts,
and uses vLLM batched inference to evaluate the model.
"""

import json
import os
import random
import re
from collections import defaultdict

from rlef.data import APPSProblem
from rlef.prompt import format_prompt

# Import your native execution sandbox and formatting tools
from rlef.reward import execution_reward
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ─── 1. RAW DIRECTORY PARSING & STRATIFICATION ───────────────────────────


def load_raw_apps_and_stratify(test_dir_path: str, samples_per_difficulty: int = 250):
    """Parses raw APPS folders, extracts metadata/tests, and stratifies."""
    difficulties = defaultdict(list)

    # Iterate through problem directories (e.g., 0000, 0001)
    for problem_id in os.listdir(test_dir_path):
        prob_path = os.path.join(test_dir_path, problem_id)
        if not os.path.isdir(prob_path):
            continue

        try:
            # 1. Read Metadata
            with open(os.path.join(prob_path, "metadata.json"), "r") as f:
                meta = json.load(f)
                diff = meta.get("difficulty", "unknown").lower()

            # 2. Read Question
            with open(os.path.join(prob_path, "question.txt"), "r") as f:
                question = f.read()

            # 3. Read Hidden Test Cases
            io_path = os.path.join(prob_path, "input_output.json")
            if os.path.exists(io_path):
                with open(io_path, "r") as f:
                    io_data = json.load(f)
                    inputs = io_data.get("inputs", [])
                    outputs = io_data.get("outputs", [])
                    fn_name = io_data.get("fn_name", None)
            else:
                inputs, outputs, fn_name = [], [], None

            problem_data = {
                "problem_id": problem_id,
                "difficulty": diff,
                "question": question,
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": fn_name,
            }

            if diff in ["introductory", "interview", "competition"]:
                difficulties[diff].append(problem_data)

        except Exception:
            # Silently skip malformed problem folders
            continue

    sampled_dataset = []
    print("\n--- Stratified Sampling from Raw APPS ---")
    for diff in ["introductory", "interview", "competition"]:
        available = len(difficulties[diff])
        take = min(samples_per_difficulty, available)
        sampled = random.sample(difficulties[diff], take)
        sampled_dataset.extend(sampled)
        print(f"{diff.capitalize()}: {take} samples (from {available} available)")

    random.shuffle(sampled_dataset)
    return sampled_dataset


# ─── 2. OPENRLHF STRING FORMATTING ───────────────────────────────────────


def format_to_openrlhf_strings(sampled_dataset, tokenizer):
    """Converts raw problem dicts into token-ready OpenRLHF-style strings."""
    print("\nFormatting prompts via Qwen Chat Template...")

    formatted_dataset = []
    for data in sampled_dataset:
        # Create the dataclass object expected by your prompt.py
        problem_obj = APPSProblem(
            problem_id=data["problem_id"],
            question=data["question"],
            difficulty=data["difficulty"],
            inputs=data["inputs"],
            outputs=data["outputs"],
        )

        # Get the asymmetric role dictionary from your prompt.py
        messages = format_prompt(problem_obj)

        # Apply Qwen's specific chat template tokens
        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Inject the compiled string back into the dict
        data["prompt"] = prompt_string
        formatted_dataset.append(data)

    return formatted_dataset


# ─── 3. EXTRACTION LOGIC ─────────────────────────────────────────────────


def extract_final_submission(model_output_string: str) -> str:
    """Ignores generated tests and grabs the final execution block for grading."""
    execute_blocks = re.findall(
        r"<tool>\s*execute\s*</tool>\s*<code>\s*(.*?)\s*</code>",
        model_output_string,
        re.IGNORECASE | re.DOTALL,
    )

    if execute_blocks:
        return execute_blocks[-1].strip()
    return None


# ─── 4. THE MASTER EVALUATION LOOP ───────────────────────────────────────


def main():
    test_dir_path = "data/raw/APPS/test"  # Ensure this points to the unpacked folder
    lora_checkpoint_path = "./checkpoints/epoch_120"
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # 1. Load and stratify raw folders
    raw_dataset = load_raw_apps_and_stratify(test_dir_path, samples_per_difficulty=250)

    # 2. Format to OpenRLHF strings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = format_to_openrlhf_strings(raw_dataset, tokenizer)

    print("\nInitializing vLLM Batched Engine...")
    llm = LLM(
        model=model_name,
        enable_lora=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["</tool>"],
        include_stop_str_in_output=True,
    )

    prompts = [data["prompt"] for data in dataset]
    active_lora = LoRARequest("eval_policy", 1, lora_checkpoint_path)

    print(f"\nGenerating responses for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params, lora_request=active_lora)

    print("\nStarting local sandbox execution...")
    results_by_difficulty = {
        "introductory": {"pass": 0, "fail": 0, "format_error": 0},
        "interview": {"pass": 0, "fail": 0, "format_error": 0},
        "competition": {"pass": 0, "fail": 0, "format_error": 0},
    }

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        problem_data = dataset[i]
        diff = problem_data.get("difficulty", "unknown").lower()

        extracted_code = extract_final_submission(generated_text)

        if not extracted_code:
            results_by_difficulty[diff]["format_error"] += 1
            continue

        exec_result = execution_reward(
            code=extracted_code,
            inputs=problem_data.get("inputs", []),
            outputs=problem_data.get("outputs", []),
            fn_name=problem_data.get("fn_name"),
        )

        if exec_result.pass_rate == 1.0:
            results_by_difficulty[diff]["pass"] += 1
        else:
            results_by_difficulty[diff]["fail"] += 1

    # ─── 5. REPORTING ────────────────────────────────────────────────────────

    print("\n=== FINAL APPS PASS@1 EVALUATION ===")
    total_pass = 0
    total_valid = 0

    for diff, stats in results_by_difficulty.items():
        attempted = stats["pass"] + stats["fail"] + stats["format_error"]
        if attempted == 0:
            continue

        pass_rate = (stats["pass"] / attempted) * 100
        format_fail_rate = (stats["format_error"] / attempted) * 100

        total_pass += stats["pass"]
        total_valid += attempted

        print(f"\n{diff.upper()}:")
        print(f"  Pass@1: {pass_rate:.2f}% ({stats['pass']}/{attempted})")
        print(
            f"  Format Failures: {format_fail_rate:.2f}% ({stats['format_error']}/{attempted})"
        )

    print(f"\nOVERALL PASS@1: {(total_pass / total_valid) * 100:.2f}%")


if __name__ == "__main__":
    main()
