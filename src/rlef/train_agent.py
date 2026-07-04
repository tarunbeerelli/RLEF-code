"""
train_agent.py — Custom Asynchronous RLHF Agent Loop
Executes true interactive multi-turn rollouts using vLLM Prefix Caching
and updates weights using a single-GPU PyTorch LoRA-toggling strategy.
"""

import asyncio
import json

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from rlef.prompt import parse_output
from rlef.reward import execution_reward

# Import your existing gRPC bridge to grade the code
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

# ─── 1. INFRASTRUCTURE & MEMORY PARTITIONING ──────────────────────────────

print("Initializing vLLM Engine (Allocating 50% of H200 VRAM)...")
engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    enable_prefix_caching=True,  # The secret weapon for multi-turn speed
    enable_lora=True,
    max_lora_rank=16,
    gpu_memory_utilization=0.50,  # Strictly fence vLLM so PyTorch doesn't OOM
    max_model_len=4096,
)
vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

print("Initializing PyTorch Policy Model (Allocating remaining VRAM)...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Grabs whatever memory vLLM left behind
)

# Attach Trainable Adapters
lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
policy_model = get_peft_model(base_model, lora_config)
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)

# Force an initial save so vLLM has a file to load from
policy_model.save_pretrained("./checkpoint/active_lora")


# ─── 2. THE INTERACTIVE MULTI-TURN ROLLOUT ───────────────────────────────

# We want vLLM to physically halt generation the moment it closes a tool
sampling_params = SamplingParams(
    temperature=0.9, max_tokens=512, stop=["</tool>"], include_stop_str_in_output=True
)


async def run_single_episode(
    prompt_text: str, problem_data: dict, lora_req: LoRARequest
):
    """Handles a single multi-turn episode with Context-Rich Test Case Feedback."""
    current_context = prompt_text
    trajectory_tokens = []
    final_reward = 0.0

    # Extract the hidden test cases from the dataset
    inputs = problem_data.get("inputs", [])
    outputs = problem_data.get("outputs", [])
    fn_name = problem_data.get("fn_name", None)

    for turn in range(5):
        # 1. Generate via vLLM
        results = await vllm_engine.generate(
            current_context,
            sampling_params,
            request_id=f"req_{problem_data['problem_id']}_{turn}",
            lora_request=lora_req,
        )

        completion = results.outputs[0].text
        trajectory_tokens.append(completion)

        parsed = parse_output(completion)

        if not parsed.is_valid:
            final_reward = 0.0  # Format failure
            break

        # 2. Native Local Execution Sandbox (Bypassing gRPC!)
        if parsed.tool == "execute":
            exec_result = execution_reward(
                code=parsed.code,
                inputs=inputs,
                outputs=outputs,
                fn_name=fn_name,
                shaped=False,  # We want the raw pass rate for the logic check
            )

            step_reward = exec_result.pass_rate

            # 3. Did it pass?
            if step_reward == 1.0:
                final_reward = 1.0
                break
            else:
                # 4. CONTEXT-RICH FEEDBACK ARCHITECTURE
                # Give the model exactly what it needs to debug
                feedback_str = f"Execution Pass Rate: {step_reward * 100}%\n"

                if "SyntaxError" in exec_result.error_types:
                    feedback_str += "Error: Your code threw a SyntaxError or crashed. Check formatting and syntax."
                elif len(inputs) > 0:
                    # Show them the very first test case as a hint!
                    feedback_str += (
                        f"Hint: Your code failed on this test case:\n"
                        f"- Input: {inputs[0]}\n"
                        f"- Expected Output: {outputs[0]}\n"
                    )

                current_context += (
                    f"{completion}\nUser: Tool Result:\n{feedback_str}\nAssistant:\n"
                )

                if turn == 4:
                    final_reward = step_reward  # Partial credit on the final turn

    return {
        "context": prompt_text,
        "completions": trajectory_tokens,
        "reward": final_reward,
    }


# ─── 3. THE GRPO MATH & LORA-TOGGLE UPDATE ───────────────────────────────


def grpo_update_step(batch_trajectories, beta=0.04):
    """Calculates advantages and KL penalty using the memory-saving toggle trick."""
    policy_model.train()
    optimizer.zero_grad()

    rewards = torch.tensor(
        [traj["reward"] for traj in batch_trajectories], device=policy_model.device
    )
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = 0

    for traj, adv in zip(batch_trajectories, advantages):
        # Flatten the multi-turn transcript into one string for PyTorch
        full_text = traj["context"] + "".join(traj["completions"])
        input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(
            policy_model.device
        )

        # PASS 1: Get Policy Log-Probs (WITH LoRA active)
        policy_logits = policy_model(input_ids).logits
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)

        # PASS 2: Get Reference Log-Probs (WITHOUT LoRA)
        # This context manager is why we don't need a second 7B model in memory!
        with policy_model.disable_adapter():
            with torch.no_grad():
                ref_logits = policy_model(input_ids).logits
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)

        # Calculate PPO Surrogate Loss + KL Penalty
        kl_div = (
            torch.exp(ref_logprobs - policy_logprobs)
            - (ref_logprobs - policy_logprobs)
            - 1
        )
        ratio = torch.exp(policy_logprobs - ref_logprobs.detach())
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)
        final_loss = (policy_loss + beta * kl_div).mean()

        final_loss.backward()
        total_loss += final_loss.item()

    optimizer.step()

    # Save the updated weights so vLLM can load them on the next rollout
    policy_model.save_pretrained("./checkpoint/active_lora")
    return total_loss / len(batch_trajectories)


# ─── 4. THE MASTER EXECUTION LOOP ────────────────────────────────────────


async def main():
    # Load the YAML config dynamically
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg.get("dataset_path", "./data/openrlhf_apps_train.jsonl")
    epochs = cfg.get("num_epochs", 3)

    # Load your specific 3-run APPS subset dataset
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    # Restrict to your target runs to keep iteration fast and focused
    dataset = dataset[:3]

    print(f"Loaded {len(dataset)} evaluation problems.")

    epochs = 3
    for epoch in range(epochs):
        print(f"\n=== Starting Epoch {epoch+1} ===")

        # Tell vLLM to use the latest saved LoRA weights for this batch
        active_lora = LoRARequest("active_policy", 1, "./checkpoint/active_lora")

        # 1. Run all episodes asynchronously (Max throughput!)
        tasks = [
            run_single_episode(data["prompt"], data, active_lora) for data in dataset
        ]
        trajectories = await asyncio.gather(*tasks)

        avg_reward = sum(t["reward"] for t in trajectories) / len(trajectories)
        print(f"Rollouts Complete. Average Reward: {avg_reward:.2f}")

        # 2. Hand the trajectories over to PyTorch for the GRPO math
        loss = grpo_update_step(trajectories)
        print(f"Update Step Complete. Loss: {loss:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
