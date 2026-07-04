"""
train_agent.py — Custom Asynchronous RLHF Agent Loop
Executes true interactive multi-turn rollouts using vLLM Prefix Caching
and updates weights using a single-GPU PyTorch LoRA-toggling strategy.
"""

import asyncio
import json
import os
import shutil

import torch
import torch.nn.functional as F
import wandb
import yaml
from peft import LoraConfig, get_peft_model
from rlef.prompt import parse_output
from rlef.reward import execution_reward
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
# sampling_params = SamplingParams(
#    temperature=0.9, max_tokens=512, stop=["</tool>"], include_stop_str_in_output=True
# )


async def run_single_episode(
    prompt_text: str,
    problem_data: dict,
    lora_req: LoRARequest,
    sampling_params: SamplingParams,
):
    """Handles a single multi-turn episode with Context-Rich Test Case Feedback."""
    current_context = prompt_text
    trajectory_tokens = []
    final_reward = 0.0

    # 1. Initialize the tracker for this episode
    episode_stats = {"execute": 0, "generate_tests": 0, "invalid_format": 0}

    # Extract the hidden test cases from the dataset
    inputs = problem_data.get("inputs", [])
    outputs = problem_data.get("outputs", [])
    fn_name = problem_data.get("fn_name", None)

    for turn in range(5):
        # Generate via vLLM
        results = await vllm_engine.generate(
            current_context,
            sampling_params,
            request_id=f"req_{problem_data['problem_id']}_{turn}",
            lora_request=lora_req,
            sampling_params=SamplingParams,
        )

        completion = results.outputs[0].text
        trajectory_tokens.append(completion)

        parsed = parse_output(completion)

        if not parsed.is_valid:
            episode_stats["invalid_format"] += 1
            final_reward = 0.0
            break

        # 1.2. THE TURN 1 ENFORCER (Close the loophole!)
        if turn == 0 and parsed.tool != "generate_tests":
            episode_stats["invalid_format"] += 1
            feedback_str = "CRITICAL PROTOCOL VIOLATION: You MUST use <tool>generate_tests</tool> on the first turn before executing any code."
            current_context += (
                f"{completion}\nUser: Tool Result:\n{feedback_str}\nAssistant:\n"
            )
            continue  # Force it to loop to Turn 1 and try again

        # 2. Route the Tool Calls
        if parsed.tool == "generate_tests":
            episode_stats["generate_tests"] += 1
            # Feed back a generic success so the model moves on to execution
            feedback_str = "Tests noted. Use <tool>execute</tool> when ready."
            current_context += (
                f"{completion}\nUser: Tool Result:\n{feedback_str}\nAssistant:\n"
            )
            continue

        elif parsed.tool == "execute":
            episode_stats["execute"] += 1

            # Native Local Execution Sandbox (Bypassing gRPC!)
            exec_result = execution_reward(
                code=parsed.code,
                inputs=inputs,
                outputs=outputs,
                fn_name=fn_name,
                shaped=False,  # We want the raw pass rate for the logic check
            )

            step_reward = exec_result.pass_rate

            if step_reward == 1.0:
                final_reward = 1.0
                break
            else:
                # CONTEXT-RICH FEEDBACK ARCHITECTURE
                feedback_str = f"Execution Pass Rate: {step_reward * 100}%\n"

                if "SyntaxError" in exec_result.error_types:
                    feedback_str += "Error: Your code threw a SyntaxError or crashed. Check formatting and syntax."
                elif len(inputs) > 0:
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

    # Return the stats alongside the trajectories
    return {
        "context": prompt_text,
        "completions": trajectory_tokens,
        "reward": final_reward,
        "stats": episode_stats,
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
    total_kl = 0

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
        total_kl += kl_div.mean().item()

        ratio = torch.exp(policy_logprobs - ref_logprobs.detach())
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)
        final_loss = (policy_loss + beta * kl_div).mean()

        final_loss.backward()
        total_loss += final_loss.item()

    optimizer.step()

    # Save the updated weights so vLLM can load them on the next rollout
    policy_model.save_pretrained("./checkpoint/active_lora")

    return total_loss / len(batch_trajectories), total_kl / len(batch_trajectories)


# ─── 4. THE MASTER EXECUTION LOOP ────────────────────────────────────────


async def main():
    # Load the YAML config dynamically
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg.get("dataset_path", "./data/openrlhf_apps_train.jsonl")
    epochs = cfg.get("num_epochs", 3)

    # Load your specific APPS subset dataset
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    # Start the W&B Run
    wandb.init(
        project=cfg.get("wandb_project", "rlef-code"),
        entity=cfg.get("wandb_entity", "tarunbeerelli-northeastern-university"),
        config=cfg,
    )

    # Restrict to your target runs to keep iteration fast and focused
    dataset = dataset[:3]

    print(f"Loaded {len(dataset)} evaluation problems.")

    for epoch in range(epochs):
        print(f"\n=== Starting Epoch {epoch+1} ===")
        # --- TEMPERATURE ANNEALING ---
        # Starts at 0.9 (high exploration), decays to 0.2 (high exploitation)
        start_temp = 0.9
        end_temp = 0.2
        # Prevent division by zero if running only 1 epoch
        decay_factor = epoch / max(1, (epochs - 1))
        current_temp = start_temp - decay_factor * (start_temp - end_temp)

        print(f"Sampling Temperature set to: {current_temp:.2f}")

        sampling_params = SamplingParams(
            temperature=current_temp,
            max_tokens=512,
            stop=["</tool>"],
            include_stop_str_in_output=True,
        )

        # Tell vLLM to use the latest saved LoRA weights for this batch
        active_lora = LoRARequest("active_policy", 1, "./checkpoint/active_lora")

        # 1. Run all episodes asynchronously (Max throughput!)
        tasks = [
            run_single_episode(data["prompt"], data, active_lora, sampling_params)
            for data in dataset
        ]
        trajectories = await asyncio.gather(*tasks)

        avg_reward = sum(t["reward"] for t in trajectories) / len(trajectories)
        print(f"Rollouts Complete. Average Reward: {avg_reward:.2f}")

        # 2. Hand the trajectories over to PyTorch for the GRPO math
        loss, kl_divergence = grpo_update_step(trajectories)
        print(f"Update Step Complete. Loss: {loss:.4f}, KL: {kl_divergence:.4f}")

        # 3. Aggregate Tool Stats for Dashboard
        total_execute = sum(t["stats"]["execute"] for t in trajectories)
        total_tests = sum(t["stats"]["generate_tests"] for t in trajectories)
        total_invalid = sum(t["stats"]["invalid_format"] for t in trajectories)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "train/loss": loss,
                "train/kl_divergence": kl_divergence,
                "train/avg_reward": avg_reward,
                "train/learning_rate": current_lr,
                "tools/execute_calls": total_execute,
                "tools/generate_tests_calls": total_tests,
                "tools/invalid_format_errors": total_invalid,
                "train/temperature": current_temp,
                "epoch": epoch + 1,
            }
        )

        # 4. Save a permanent Historical Checkpoint for this Epoch
        epoch_dir = f"./checkpoints/epoch_{epoch + 1}"
        os.makedirs(epoch_dir, exist_ok=True)

        # Copy the latest active weights into the permanent epoch folder
        active_path = "./checkpoint/active_lora"
        if os.path.exists(active_path):
            shutil.copytree(active_path, epoch_dir, dirs_exist_ok=True)
            print(f"Saved historical checkpoint to {epoch_dir}")


if __name__ == "__main__":
    asyncio.run(main())
