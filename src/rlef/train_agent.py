"""
train_agent.py — Configurable Asynchronous RLHF Agent Loop
Executes multi-turn rollouts using vLLM Prefix Caching.
Fully parameterized for rigorous ablation studies.
"""

import asyncio
import json
import os
import shutil
from collections import Counter

import torch
import torch.nn.functional as F
import wandb
import yaml
from peft import LoraConfig, get_peft_model
from rlef.reward import execution_reward, verify_generated_tests
from rlef.prompt import parse_output
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

# Load the config globally so the models know if they are in Phase 2
with open("configs/train.yaml", "r") as f:
    cfg = yaml.safe_load(f)

vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

print("Initializing PyTorch Policy Model (Allocating remaining VRAM)...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_resume = cfg.get("lora_resume_path", None)
if lora_resume and os.path.exists(lora_resume):
    print(f"Resuming Curriculum: Loading weights from {lora_resume}")
    from peft import PeftModel

    policy_model = PeftModel.from_pretrained(base_model, lora_resume, is_trainable=True)
else:
    print("Initializing fresh LoRA weights...")
    lora_config = LoraConfig(
        r=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    policy_model = get_peft_model(base_model, lora_config)
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)
policy_model.save_pretrained("./checkpoint/active_lora")

"""
# ─── 2. INLINE EXTRACTION PARSER ─────────────────────────────────────────────

def parse_output(text: str) -> dict:
    code_match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL | re.IGNORECASE)
    edge_match = re.search(r"<edge_cases>\s*(.*?)\s*</edge_cases>", text, re.DOTALL | re.IGNORECASE)

    return {
        "code": code_match.group(1).strip() if code_match else None,
        "edge_cases": edge_match.group(1).strip() if edge_match else None,
        "is_valid": bool(code_match) # Code is strictly required to proceed
    }
"""

# ─── 3. THE INTERACTIVE MULTI-TURN ROLLOUT ───────────────────────────────


async def run_single_episode(
    prompt_text: str,
    problem_data: dict,
    lora_req: LoRARequest,
    sampling_params: SamplingParams,
    ablation_cfg: dict,
):
    current_context = prompt_text
    trajectory_tokens = []
    final_reward = 0.0
    episode_stats = {"execute": 0, "generate_tests": 0, "invalid_format": 0}
    episode_errors = []

    inputs = problem_data.get("inputs", [])
    outputs = problem_data.get("outputs", [])
    fn_name = problem_data.get("fn_name", None)

    # Pull Ablation Toggles
    max_turns = ablation_cfg.get("max_turns", 5)
    use_linear_pass_rate = ablation_cfg.get("use_linear_pass_rate", True)
    use_step_credit = ablation_cfg.get("use_step_credit", True)
    use_turn_penalty = ablation_cfg.get("use_turn_penalty", True)
    use_feedback_bonus = ablation_cfg.get("use_feedback_bonus", True)
    use_edge_cases = ablation_cfg.get("use_edge_cases", False)
    feedback_type = ablation_cfg.get("feedback_type", "last_failed")

    previous_pass_rate = 0.0

    for turn in range(max_turns):
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

        parsed = parse_output(completion)

        if use_edge_cases and not parsed["edge_cases"]:
            episode_stats["invalid_format"] += 1
            feedback_str = "CRITICAL ERROR: Missing <edge_cases> block. You must write test cases before writing code."
            current_context += (
                f"{completion}\nUser: System Result:\n{feedback_str}\nAssistant:\n"
            )
            continue

        if not parsed["is_valid"]:
            episode_stats["invalid_format"] += 1
            feedback_str = "CRITICAL ERROR: Invalid format. You must wrap your executable logic inside a <code>...</code> block. Please try again."
            current_context += f"{completion}<|im_end|>\n<|im_start|>user\nSystem Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"
            continue

        episode_stats["execute"] += 1
        step_reward = 0.0

        # --- EDGE CASE VERIFICATION (RUNS 6 & 7) ---
        if use_edge_cases and parsed["edge_cases"]:
            episode_stats["generate_tests"] += 1
            test_bonus = await asyncio.to_thread(
                verify_generated_tests, parsed["edge_cases"], parsed["code"], fn_name
            )
            step_reward += test_bonus

        # --- EXECUTE MAIN LOGIC ---
        exec_result = await asyncio.to_thread(
            execution_reward,
            code=parsed["code"],
            inputs=inputs,
            outputs=outputs,
            previous_pass_rate=previous_pass_rate,
            current_turn=turn + 1,
            shaped=False,
        )

        pass_rate = exec_result.pass_rate
        if hasattr(exec_result, "error_types"):
            episode_errors.extend(exec_result.error_types)

        # --- CENTRALIZED ABLATION REWARD MATH ---
        step_reward += (
            pass_rate if use_linear_pass_rate else (1.0 if pass_rate == 1.0 else 0.0)
        )

        if parsed["has_reasoning"]:
            step_reward += (
                0.02  # Micro-reward for following chain-of-thought instructions
            )

        if use_step_credit and 0.0 < pass_rate < 1.0:
            step_reward += 0.10

        if use_turn_penalty:
            step_reward -= 0.05  # FIXED: Constant linear penalty per turn

        if use_feedback_bonus and pass_rate > previous_pass_rate and turn > 0:
            step_reward += 0.10

        previous_pass_rate = pass_rate

        if pass_rate == 1.0:
            final_reward += step_reward
            break
        else:
            # --- CENTRALIZED ABLATION FEEDBACK ROUTING ---
            if feedback_type == "none":
                feedback_str = (
                    "Execution failed. Please try a different algorithmic approach."
                )

            elif feedback_type == "consolidated":
                err_counts = Counter(exec_result.error_types)
                err_str = ", ".join([f"{v} {k}" for k, v in err_counts.items()])
                feedback_str = (
                    f"Execution Pass Rate: {pass_rate * 100}%. Errors logged: {err_str}"
                )

            elif feedback_type == "last_failed":
                feedback_str = f"Execution Pass Rate: {pass_rate * 100}%\n"
                if len(inputs) > 0 and pass_rate < 1.0:
                    feedback_str += f"Failed on test case:\n- Input: {inputs[0]}\n- Expected Output: {outputs[0]}"

            else:  # standard
                feedback_str = (
                    f"Execution Pass Rate: {pass_rate * 100}%. Please revise."
                )

            current_context += f"{completion}<|im_end|>\n<|im_start|>user\nSystem Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"

            if turn == max_turns - 1:
                final_reward += step_reward

    return {
        "context": prompt_text,
        "completions": trajectory_tokens,
        "reward": max(0.0, final_reward),
        "stats": episode_stats,
        "errors": episode_errors,
        "turns_taken": turn + 1,
        "final_context_length": len(current_context),
        "success": 1 if pass_rate == 1.0 else 0,
    }


# ─── 4. THE GRPO MATH & LORA-TOGGLE UPDATE ───────────────────────────────


def grpo_update_step(batch_trajectories, beta=0.04):
    if not batch_trajectories:
        return 0.0, 0.0

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


# ─── 5. THE MASTER EXECUTION LOOP ────────────────────────────────────────


async def main():
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ablation_cfg = cfg.get("ablation", {})
    # Data path toggle prioritized in ablation config for Phase 1 vs Phase 2 curriculum
    dataset_path = ablation_cfg.get(
        "dataset_path", cfg.get("dataset_path", "./data/openrlhf_apps_train.jsonl")
    )
    epochs = cfg.get("num_epochs", 3)

    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    wandb.init(
        project=cfg.get("wandb_project", "rlef-code"),
        entity=cfg.get("wandb_entity", "tarunbeerelli-northeastern-university"),
        config=cfg,
        tags=cfg.get("tags", []),
    )

    print(f"Loaded {len(dataset)} evaluation problems from {dataset_path}.")

    start_temp = cfg.get("start_temp", 0.7)
    end_temp = cfg.get("end_temp", 0.2)
    batch_size = cfg.get("batch_size", 50)

    total_batches = (
        len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
    ) * epochs
    global_batch = 0

    for epoch in range(epochs):
        print(f"\n=== Starting Epoch {epoch+1} ===")

        # Step through the dataset in batches to allow continuous temp decay
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i : i + batch_size]

            # Smooth Linear Continuous Decay per batch step
            current_temp = start_temp - (start_temp - end_temp) * (
                global_batch / max(1, total_batches - 1)
            )

            sampling_params = SamplingParams(
                temperature=current_temp,
                max_tokens=512,
                stop=cfg.get("stop_tokens", ["</code>"]),
                include_stop_str_in_output=True,
            )

            active_lora = LoRARequest(
                "active_policy", 1, lora_path="./checkpoint/active_lora"
            )

            tasks = [
                run_single_episode(
                    data["prompt"], data, active_lora, sampling_params, ablation_cfg
                )
                for data in batch_data
            ]
            trajectories = await asyncio.gather(*tasks)

            avg_reward = sum(t["reward"] for t in trajectories) / max(
                1, len(trajectories)
            )

            loss, kl_divergence = grpo_update_step(trajectories)

            total_execute = sum(t["stats"]["execute"] for t in trajectories)
            total_tests = sum(t["stats"]["generate_tests"] for t in trajectories)
            total_invalid = sum(t["stats"]["invalid_format"] for t in trajectories)

            avg_turns = sum(t["turns_taken"] for t in trajectories) / max(
                1, len(trajectories)
            )
            avg_ctx_len = sum(t["final_context_length"] for t in trajectories) / max(
                1, len(trajectories)
            )
            success_rate = sum(t["success"] for t in trajectories) / max(
                1, len(trajectories)
            )

            current_lr = optimizer.param_groups[0]["lr"]

            batch_errors = Counter()
            for t in trajectories:
                batch_errors.update(t["errors"])

            log_metrics = {
                "train/loss": loss,
                "train/kl_divergence": kl_divergence,
                "train/avg_reward": avg_reward,
                "train/learning_rate": current_lr,
                "train/temperature": current_temp,
                "tools/execute_calls": total_execute,
                "tools/generate_tests_calls": total_tests,
                "tools/invalid_format_errors": total_invalid,
                "metrics/avg_turns": avg_turns,
                "metrics/avg_context_length": avg_ctx_len,
                "metrics/success_rate": success_rate,
                "epoch": epoch + 1,
                "global_step": global_batch,
            }

            # Dynamically add all error types to W&B
            for err_type, count in batch_errors.items():
                log_metrics[f"errors/{err_type}"] = count

            wandb.log(log_metrics)

            print(
                f"Batch {global_batch+1}/{total_batches} | Temp: {current_temp:.3f} | Reward: {avg_reward:.2f} | Loss: {loss:.4f}"
            )
            global_batch += 1

        epoch_dir = f"./checkpoints/epoch_{epoch + 1}"
        os.makedirs(epoch_dir, exist_ok=True)

        active_path = "./checkpoint/active_lora"
        if os.path.exists(active_path):
            shutil.copytree(active_path, epoch_dir, dirs_exist_ok=True)
            print(f"Saved historical checkpoint to {epoch_dir}")


if __name__ == "__main__":
    asyncio.run(main())
    wandb.finish()
