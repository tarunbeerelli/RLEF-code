"""
evaluate.py — Asynchronous Rollout Evaluator
Calculates pass@1 and pass@N using the locked environment physics.
Logs final macro evaluation suites directly to Weights & Biases, categorized by difficulty.
"""

import asyncio
import json
import random
import yaml
import wandb
from collections import Counter
from rlef.reward import execution_reward
from rlef.prompt import parse_output
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest


# ─── 1. EVALUATION EPISODE ───────────────────────────────────────────────────
async def evaluate_single_episode(
    prompt_text: str,
    problem_data: dict,
    vllm_engine,
    lora_req,
    sampling_params,
    ablation_cfg,
):
    current_context = prompt_text
    max_turns = ablation_cfg.get("max_turns", 5)
    difficulty = problem_data.get("difficulty", "unknown")
    feedback_type = ablation_cfg.get("feedback_type", "last_failed")

    inputs = problem_data.get("inputs", [])
    outputs = problem_data.get("outputs", [])

    pass_at_1 = 0
    final_pass_rate = 0.0
    episode_errors = []

    for turn in range(max_turns):
        request_generator = vllm_engine.generate(
            prompt=current_context,
            sampling_params=sampling_params,
            request_id=f"eval_{problem_data['problem_id']}_{turn}",
            lora_request=lora_req,
        )

        final_result = None
        async for res in request_generator:
            final_result = res

        completion = final_result.outputs[0].text
        parsed = parse_output(completion)

        if not parsed["is_valid"]:
            feedback_str = "CRITICAL ERROR: Invalid format. Use <code>...</code>."
            current_context += f"{completion}<|im_end|>\n<|im_start|>user\nSystem Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"
            continue

        # Execute Code
        exec_result = await asyncio.to_thread(
            execution_reward,
            code=parsed["code"],
            inputs=inputs,
            outputs=outputs,
            shaped=False,
        )

        final_pass_rate = exec_result.pass_rate
        if hasattr(exec_result, "error_types"):
            episode_errors.extend(exec_result.error_types)

        # Track Zero-Shot pass@1
        if turn == 0 and final_pass_rate == 1.0:
            pass_at_1 = 1

        if final_pass_rate == 1.0:
            break
        else:
            if feedback_type == "last_failed":
                feedback_str = f"Execution Pass Rate: {final_pass_rate * 100}%\n"
                if len(inputs) > 0 and final_pass_rate < 1.0:
                    feedback_str += f"Failed on test case:\n- Input: {inputs[0]}\n- Expected Output: {outputs[0]}"
            else:
                feedback_str = (
                    f"Execution Pass Rate: {final_pass_rate * 100}%. Please revise."
                )

            current_context += f"{completion}<|im_end|>\n<|im_start|>user\nSystem Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"

    return {
        "difficulty": difficulty,
        "pass_at_1": pass_at_1,
        "pass_at_N": 1 if final_pass_rate == 1.0 else 0,
        "final_pass_rate": final_pass_rate,
        "turns_taken": turn + 1,
        "errors": episode_errors,
    }


# ─── 2. MASTER EVAL LOOP ─────────────────────────────────────────────────────
async def main():
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {})
    ablation_cfg = cfg.get("ablation", {})
    max_turns = ablation_cfg.get("max_turns", 5)

    dataset_path = eval_cfg.get("dataset_path", "./data/apps_eval.jsonl")
    lora_path = eval_cfg.get("lora_path", "./checkpoint/active_lora")

    raw_dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            raw_dataset.append(json.loads(line))

    # --- THE EVALUATION CAP ---
    random.seed(42)  # Keep it deterministic for 1:1 baseline comparisons
    dataset = []
    for diff in ["introductory", "interview", "competition"]:
        diff_problems = [d for d in raw_dataset if d.get("difficulty") == diff]
        random.shuffle(diff_problems)
        dataset.extend(diff_problems[:250])

    print(
        f"Loaded {len(dataset)} evaluation problems from {dataset_path} (capped at 250 per difficulty)"
    )
    print(f"Targeting LoRA checkpoint: {lora_path}")

    # Initialize W&B with evaluation metadata
    eval_tags = cfg.get("tags", []).copy()
    if "eval" not in eval_tags:
        eval_tags.append("eval")

    wandb.init(
        project=cfg.get("wandb_project", "rlef-code"),
        entity=cfg.get("wandb_entity", "tarunbeerelli-northeastern-university"),
        config=cfg,
        tags=eval_tags,
        job_type="evaluation",
    )

    print("Initializing vLLM Engine (Read-Only Mode)...")
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.60,  # Generous read-only footprint
        max_model_len=8192,  # Raised ceiling for competition descriptions
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for reliable benchmark execution
        max_tokens=800,  # Generous generation runway
        stop=cfg.get("stop_tokens", ["</code>"]),
        include_stop_str_in_output=True,
    )

    try:
        active_lora = LoRARequest("eval_policy", 1, lora_path=lora_path)
    except Exception as e:
        print(
            f"Warning: Could not load LoRA at {lora_path}. Evaluating Base Model. Error: {e}"
        )
        active_lora = None

    semaphore = asyncio.Semaphore(50)

    async def bounded_eval(data):
        async with semaphore:
            return await evaluate_single_episode(
                data["prompt"],
                data,
                vllm_engine,
                active_lora,
                sampling_params,
                ablation_cfg,
            )

    print("\n=== Commencing Evaluation Sweep ===")
    tasks = [bounded_eval(data) for data in dataset]
    results = await asyncio.gather(*tasks)

    total = len(results)

    metrics = {"eval/total_problems": total}

    # Aggregate and log all evaluation errors dynamically
    sweep_errors = Counter()
    for r in results:
        sweep_errors.update(r["errors"])

    for err_type, count in sweep_errors.items():
        metrics[f"eval_errors/{err_type}"] = count

    # Calculate overall metrics
    metrics["eval/pass_at_1_overall"] = (
        sum(r["pass_at_1"] for r in results) / total
    ) * 100
    metrics[f"eval/pass_at_{max_turns}_overall"] = (
        sum(r["pass_at_N"] for r in results) / total
    ) * 100
    metrics["eval/avg_partial_pass_rate_overall"] = (
        sum(r["final_pass_rate"] for r in results) / total * 100
    )
    metrics["eval/avg_turns_taken"] = sum(r["turns_taken"] for r in results) / total

    # Calculate by category
    for diff in ["introductory", "interview", "competition"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            d_total = len(diff_results)
            metrics[f"eval/pass_at_1_{diff}"] = (
                sum(r["pass_at_1"] for r in diff_results) / d_total
            ) * 100
            metrics[f"eval/pass_at_{max_turns}_{diff}"] = (
                sum(r["pass_at_N"] for r in diff_results) / d_total
            ) * 100

    print("\n=== EVALUATION RESULTS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    wandb.log(metrics)


if __name__ == "__main__":
    asyncio.run(main())
    wandb.finish()
