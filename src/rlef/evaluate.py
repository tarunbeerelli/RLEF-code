"""
evaluate.py — Asynchronous Rollout Evaluator
Calculates pass@1 and pass@N using the locked environment physics.
Logs final macro evaluation suites directly to Weights & Biases.
"""

import asyncio
import json
import re
import yaml
import wandb
from rlef.reward import execution_reward
from rlef.prompt import parse_output
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

"""
# ─── 1. INLINE PARSER ────────────────────────────────────────────────────────
def parse_output(text: str) -> dict:
    code_match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL | re.IGNORECASE)
    return {
        "code": code_match.group(1).strip() if code_match else None,
        "is_valid": bool(code_match)
    }
"""

# ─── 2. EVALUATION EPISODE ───────────────────────────────────────────────────
async def evaluate_single_episode(prompt_text: str, problem_data: dict, vllm_engine, lora_req, sampling_params, ablation_cfg):
    current_context = prompt_text
    max_turns = ablation_cfg.get("max_turns", 5)
    feedback_type = ablation_cfg.get("feedback_type", "last_failed")
    
    inputs = problem_data.get("inputs", [])
    outputs = problem_data.get("outputs", [])
    
    pass_at_1 = 0
    final_pass_rate = 0.0

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
            current_context += f"{completion}\nUser: System Result:\nCRITICAL ERROR: Invalid format. Use <code>...</code>.\nAssistant:\n"
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
        
        # Track Zero-Shot pass@1
        if turn == 0 and final_pass_rate == 1.0:
            pass_at_1 = 1
            
        if final_pass_rate == 1.0:
            break
        else:
            if feedback_type == "last_failed":
                feedback_str = f"Execution Pass Rate: {final_pass_rate * 100}%\n"
                if len(inputs) > 0:
                    feedback_str += f"Failed on test case:\n- Input: {inputs[0]}\n- Expected Output: {outputs[0]}"
            else:
                feedback_str = f"Execution Pass Rate: {final_pass_rate * 100}%. Please revise."

            current_context += f"{completion}\nUser: System Result:\n{feedback_str}\nAssistant:\n"

    return {
        "pass_at_1": pass_at_1,
        "pass_at_N": 1 if final_pass_rate == 1.0 else 0,
        "final_pass_rate": final_pass_rate,
        "turns_taken": turn + 1
    }

# ─── 3. MASTER EVAL LOOP ─────────────────────────────────────────────────────
async def main():
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {})
    ablation_cfg = cfg.get("ablation", {})
    max_turns = ablation_cfg.get("max_turns", 5)
    
    dataset_path = eval_cfg.get("dataset_path", "./data/apps_eval.jsonl")
    lora_path = eval_cfg.get("lora_path", "./checkpoint/active_lora")
    
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    print(f"Loaded {len(dataset)} evaluation problems from {dataset_path}")
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
        job_type="evaluation"
    )

    print("Initializing vLLM Engine (Read-Only Mode)...")
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.60,
        max_model_len=4096,
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.0, # Greedy decoding for reliable benchmark execution
        max_tokens=512,
        stop=cfg.get("stop_tokens", ["</code>"]), 
        include_stop_str_in_output=True,
    )

    try:
        active_lora = LoRARequest("eval_policy", 1, lora_path=lora_path)
    except Exception as e:
        print(f"Warning: Could not load LoRA at {lora_path}. Evaluating Base Model. Error: {e}")
        active_lora = None

    semaphore = asyncio.Semaphore(50)
    async def bounded_eval(data):
        async with semaphore:
            return await evaluate_single_episode(data["prompt"], data, vllm_engine, active_lora, sampling_params, ablation_cfg)

    print("\n=== Commencing Evaluation Sweep ===")
    tasks = [bounded_eval(data) for data in dataset]
    results = await asyncio.gather(*tasks)

    total = len(results)
    total_pass_1 = sum(r["pass_at_1"] for r in results)
    total_pass_n = sum(r["pass_at_N"] for r in results)
    avg_pass_rate = sum(r["final_pass_rate"] for r in results) / total
    avg_turns = sum(r["turns_taken"] for r in results) / total

    pass_1_metric = (total_pass_1 / total) * 100
    pass_n_metric = (total_pass_n / total) * 100

    print("\n=== EVALUATION RESULTS ===")
    print(f"Total Problems: {total}")
    print(f"pass@1 (Zero-Shot): {pass_1_metric:.2f}%")
    print(f"pass@{max_turns} (Multi-Turn): {pass_n_metric:.2f}%")
    print(f"Average Partial Pass Rate: {avg_pass_rate * 100:.2f}%")
    print(f"Average Turns Taken: {avg_turns:.2f}")

    # Push structured evaluation metrics to the dashboard
    wandb.log({
        "eval/total_problems": total,
        "eval/pass_at_1": pass_1_metric,
        f"eval/pass_at_{max_turns}": pass_n_metric,
        "eval/avg_partial_pass_rate": avg_pass_rate * 100,
        "eval/avg_turns_taken": avg_turns,
    })

if __name__ == "__main__":
    asyncio.run(main())
    wandb.finish()