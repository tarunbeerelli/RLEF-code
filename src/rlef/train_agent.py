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
from rlef.reward import execution_reward, format_reward_fn, verify_generated_tests
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

# ─── 1. INFRASTRUCTURE & MEMORY PARTITIONING ──────────────────────────────

print("Initializing vLLM Engine (Allocating 50% of H200 VRAM)...")
engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    enable_prefix_caching=True,
    enable_lora=True,
    max_lora_rank=16,
    gpu_memory_utilization=0.50,
    max_model_len=4096,
)
vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

print("Initializing PyTorch Policy Model (Allocating remaining VRAM)...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
policy_model = get_peft_model(base_model, lora_config)
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)

policy_model.save_pretrained("./checkpoint/active_lora")


# ─── 2. THE INTERACTIVE MULTI-TURN ROLLOUT ───────────────────────────────


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

    episode_stats = {
        "execute": 0,
        "generate_tests": 0,
        "invalid_format": 0,
        "syntax_errors": 0,
    }

    inputs = problem_data.get("inputs", [])
    outputs = problem_data.get("outputs", [])
    fn_name = problem_data.get("fn_name", None)

    # State tracking across the 5 turns
    previous_pass_rate = 0.0
    stored_tests = None

    for turn in range(5):
        request_generator = vllm_engine.generate(
            prompt=current_context,
            sampling_params=sampling_params,
            request_id=f"req_{problem_data['problem_id']}_{turn}",
            lora_request=lora_req,
        )

        final_result = None
        async for res in request_generator:
            final_result = res

        completion = final_result.outputs[0].text
        trajectory_tokens.append(completion)

        # 1. XML BREADCRUMB REWARD (Calculated every turn)
        format_score = format_reward_fn(
            prompts=[current_context], completions=[completion]
        )[0]
        final_reward += format_score * 0.1

        parsed = parse_output(completion)

        if not parsed.is_valid:
            episode_stats["invalid_format"] += 1
            feedback_str = "CRITICAL ERROR: Invalid tool format. You must wrap your command in <tool>...</tool> and your target inside <code>...</code>."
            current_context += (
                f"{completion}\nUser: Tool Result:\n{feedback_str}\nAssistant:\n"
            )
            continue

        if turn == 0 and parsed.tool != "generate_tests":
            episode_stats["invalid_format"] += 1
            feedback_str = "CRITICAL PROTOCOL VIOLATION: You MUST use <tool>generate_tests</tool> on the first turn before executing any code."
            current_context += (
                f"{completion}\nUser: Tool Result:\n{feedback_str}\nAssistant:\n"
            )
            continue

        if parsed.tool == "generate_tests":
            episode_stats["generate_tests"] += 1
            stored_tests = parsed.code  # Cache the tests for verification later
            feedback_str = "Tests noted. Use <tool>execute</tool> when ready."
            current_context += (
                f"{completion}\nUser: Tool Result:\n{feedback_str}\nAssistant:\n"
            )
            continue

        elif parsed.tool == "execute":
            episode_stats["execute"] += 1

            # 2. TEST VERIFICATION REWARD (Threaded to prevent blocking)
            if stored_tests:
                test_bonus = await asyncio.to_thread(
                    verify_generated_tests, stored_tests, parsed.code, fn_name
                )
                final_reward += test_bonus
                stored_tests = None  # Clear cache so we don't double dip

            # 3. SHAPED EXECUTION REWARD (Threaded to prevent vLLM timeout)
            exec_result = await asyncio.to_thread(
                execution_reward,
                code=parsed.code,
                inputs=inputs,
                outputs=outputs,
                previous_pass_rate=previous_pass_rate,
                current_turn=turn + 1,
                shaped=True,
            )

            previous_pass_rate = exec_result.pass_rate

            if exec_result.pass_rate == 1.0:
                final_reward += exec_result.final_reward
                break
            else:
                feedback_str = f"Execution Pass Rate: {exec_result.pass_rate * 100}%\n"

                if "SyntaxError" in exec_result.error_types:
                    episode_stats["syntax_errors"] += 1
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
                    final_reward += exec_result.final_reward

    return {
        "context": prompt_text,
        "completions": trajectory_tokens,
        "reward": final_reward,
        "stats": episode_stats,
        "turns_taken": turn + 1,
        "final_context_length": len(current_context),
        "success": 1 if final_reward >= 1.0 else 0,
    }


# ─── 3. THE GRPO MATH & LORA-TOGGLE UPDATE ───────────────────────────────


def grpo_update_step(batch_trajectories, beta=0.04):
    policy_model.train()
    optimizer.zero_grad()

    rewards = torch.tensor(
        [traj["reward"] for traj in batch_trajectories], device=policy_model.device
    )
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = 0
    total_kl = 0

    for traj, adv in zip(batch_trajectories, advantages):
        full_text = traj["context"] + "".join(traj["completions"])
        input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(
            policy_model.device
        )

        policy_logits = policy_model(input_ids).logits
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)

        with policy_model.disable_adapter():
            with torch.no_grad():
                ref_logits = policy_model(input_ids).logits
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)

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
    policy_model.save_pretrained("./checkpoint/active_lora")

    return total_loss / len(batch_trajectories), total_kl / len(batch_trajectories)


# ─── 4. THE MASTER EXECUTION LOOP ────────────────────────────────────────


async def main():
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg.get("dataset_path", "./data/openrlhf_apps_train.jsonl")
    epochs = cfg.get("num_epochs", 3)

    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    wandb.init(
        project=cfg.get("wandb_project", "rlef-code"),
        entity=cfg.get("wandb_entity", "tarunbeerelli-northeastern-university"),
        config=cfg,
    )

    print(f"Loaded {len(dataset)} evaluation problems.")

    # THE BOUNCER: Throttle concurrent episodes to protect event loop
    semaphore = asyncio.Semaphore(50)

    async def bounded_run_episode(data, active_lora, sampling_params):
        async with semaphore:
            return await run_single_episode(
                data["prompt"], data, active_lora, sampling_params
            )

    for epoch in range(epochs):
        print(f"\n=== Starting Epoch {epoch+1} ===")

        start_temp = 0.9
        end_temp = 0.2
        decay_factor = epoch / max(1, (epochs - 1))
        current_temp = start_temp - decay_factor * (start_temp - end_temp)

        print(f"Sampling Temperature set to: {current_temp:.2f}")

        sampling_params = SamplingParams(
            temperature=current_temp,
            max_tokens=512,
            stop=["</tool>"],
            include_stop_str_in_output=True,
        )

        active_lora = LoRARequest(
            "active_policy", 1, lora_path="./checkpoint/active_lora"
        )

        # Launch safely with the Semaphore
        tasks = [
            bounded_run_episode(data, active_lora, sampling_params) for data in dataset
        ]
        trajectories = await asyncio.gather(*tasks)

        avg_reward = sum(t["reward"] for t in trajectories) / len(trajectories)
        print(f"Rollouts Complete. Average Reward: {avg_reward:.2f}")

        loss, kl_divergence = grpo_update_step(trajectories)
        print(f"Update Step Complete. Loss: {loss:.4f}, KL: {kl_divergence:.4f}")

        total_execute = sum(t["stats"]["execute"] for t in trajectories)
        total_tests = sum(t["stats"]["generate_tests"] for t in trajectories)
        total_invalid = sum(t["stats"]["invalid_format"] for t in trajectories)
        total_syntax_errs = sum(t["stats"]["syntax_errors"] for t in trajectories)

        avg_turns = sum(t["turns_taken"] for t in trajectories) / len(trajectories)
        avg_ctx_len = sum(t["final_context_length"] for t in trajectories) / len(
            trajectories
        )
        success_rate = sum(t["success"] for t in trajectories) / len(trajectories)

        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "train/loss": loss,
                "train/kl_divergence": kl_divergence,
                "train/avg_reward": avg_reward,
                "train/learning_rate": current_lr,
                "train/temperature": current_temp,
                "tools/execute_calls": total_execute,
                "tools/generate_tests_calls": total_tests,
                "tools/invalid_format_errors": total_invalid,
                "tools/syntax_errors": total_syntax_errs,
                "metrics/avg_turns": avg_turns,
                "metrics/avg_context_length": avg_ctx_len,
                "metrics/success_rate": success_rate,
                "epoch": epoch + 1,
            }
        )

        epoch_dir = f"./checkpoints/epoch_{epoch + 1}"
        os.makedirs(epoch_dir, exist_ok=True)

        active_path = "./checkpoint/active_lora"
        if os.path.exists(active_path):
            shutil.copytree(active_path, epoch_dir, dirs_exist_ok=True)
            print(f"Saved historical checkpoint to {epoch_dir}")


if __name__ == "__main__":
    asyncio.run(main())
