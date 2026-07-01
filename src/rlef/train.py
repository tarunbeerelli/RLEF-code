"""
train.py — GRPO training loop for RLEF-Code

Single-turn mode  (configs/train.yaml: max_turns=1):
  For each problem in the batch:
    1. Format prompt
    2. Model generates code + tool call
    3. Parse output
    4. If execute: run against APPS test cases, get reward
    5. GRPO updates weights

Multi-turn mode (max_turns=N):
  Same but wrapped in a Trajectory that handles N turns,
  passing execution feedback back to the model each turn.

DDP: accelerate handles multi-GPU. Run with:
  accelerate launch --config_file accelerate_config.yaml src/rlef/train.py

Ablations: set in configs/train.yaml under the `ablation:` dictionary
"""

import argparse
import os
import random
import re

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
import wandb
import yaml
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, TaskType
from rlef.data import APPSProblem, difficulty_split, load_apps_split
from rlef.prompt import format_prompt, parse_output
from rlef.reward import execution_reward, normalize_batch_rewards
from rlef.tools import call_tool
from rlef.trajectory import Trajectory
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str = "configs/train.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataset preparation ───────────────────────────────────────────────────────


def prepare_dataset(
    problems: list[APPSProblem],
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
) -> Dataset:
    """
    Convert APPSProblem list into a HuggingFace Dataset.
    """
    rows = []
    for p in problems:
        messages = format_prompt(p)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # truncate if needed
        tokens = tokenizer(prompt, truncation=True, max_length=max_prompt_length)
        prompt = tokenizer.decode(tokens["input_ids"], skip_special_tokens=False)

        rows.append(
            {
                "prompt": prompt,
                "problem_id": p.problem_id,
                "difficulty": p.difficulty,
                "inputs": [str(x) for x in p.inputs],
                "outputs": [str(x) for x in p.outputs],
                "fn_name": p.fn_name or "",
            }
        )

    return Dataset.from_list(rows)


# ── Stratified sampling ───────────────────────────────────────────────────────


def stratified_sample(
    problems: list[APPSProblem],
    n: int,
) -> list[APPSProblem]:
    """
    Sample n problems with equal representation from each difficulty tier.
    """
    buckets = difficulty_split(problems)
    per_bucket = n // len(buckets)
    sampled = []
    for difficulty, bucket_problems in buckets.items():
        k = min(per_bucket, len(bucket_problems))
        sampled.extend(random.sample(bucket_problems, k))
    random.shuffle(sampled)
    return sampled


# ── Single-turn reward function ───────────────────────────────────────────────


def make_reward_fn(cfg: dict):
    """
    Returns the reward function GRPOTrainer calls after each generation.
    """
    # Extract the ablation dictionary
    ablation_cfg = cfg.get("ablation", {})
    max_turns = cfg.get("max_turns", 1)
    credit_type = cfg.get("credit_type", "step")

    def reward_fn(
        completions: list[str],
        prompts: list[str] | None = None,
        inputs: list[list[str]] | None = None,
        outputs: list[list[str]] | None = None,
        fn_name: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards = []

        for i, completion in enumerate(completions):
            inp = inputs[i] if inputs else []
            out = outputs[i] if outputs else []
            fn = fn_name[i] if fn_name and fn_name[i] else None

            if max_turns == 1:
                parsed = parse_output(completion)

                if not parsed.is_valid or parsed.tool != "execute":
                    rewards.append(0.0)
                    if wandb.run:
                        wandb.log(
                            {
                                "episode/status": "format_error",
                                "episode/final_reward": 0.0,
                            }
                        )
                    continue

                result = execution_reward(
                    code=parsed.code,
                    inputs=inp,
                    outputs=out,
                    fn_name=fn,
                    current_turn=1,
                    difficulty=kwargs.get("difficulty", ["introductory"])[i]
                    if kwargs.get("difficulty")
                    else "introductory",
                    ablation_cfg=ablation_cfg,  # Pass ablations here
                )
                rewards.append(result.final_reward)

                if wandb.run:
                    wandb.log(
                        {
                            "episode/status": "solved"
                            if result.pass_rate == 1.0
                            else "partial",
                            "episode/final_reward": result.final_reward,
                            "episode/pass_rate": result.pass_rate,
                            "episode/passes": result.passed,
                            "episode/total_tests": result.total,
                            "tools/execute": 1,
                        }
                    )

            else:
                # ── Multi-turn: run trajectory ────────────────────────────
                traj = _replay_trajectory(
                    completion=completion,
                    inputs=inp,
                    outputs=out,
                    fn_name=fn,
                    max_turns=max_turns,
                    credit_type=credit_type,
                    ablation_cfg=ablation_cfg,  # Pass ablations here
                )
                rewards.append(traj.final_reward)

                if wandb.run:
                    wandb.log(traj.to_log_dict())

        # ── Ablation: Normalization ───────────────────────────────────────────
        use_norm = ablation_cfg.get("use_normalization", True)
        normalized_rewards = normalize_batch_rewards(
            rewards, use_normalization=use_norm
        )

        # FINAL SAFETY CLAMP:
        # Clip the advantage to +/- 2.0 standard deviations
        # This prevents "runaway" updates while keeping the relative signal.
        clipped_rewards = np.clip(normalized_rewards, -2.0, 2.0)
        return clipped_rewards.tolist()

    return reward_fn


def _replay_trajectory(
    completion: str,
    inputs: list[str],
    outputs: list[str],
    fn_name: str | None,
    max_turns: int,
    credit_type: str,
    ablation_cfg: dict,
) -> Trajectory:
    """
    Replay a multi-turn completion to compute the reward.
    """
    from rlef.data import APPSProblem as _P

    # dummy problem for trajectory — real data comes from inputs/outputs
    dummy = _P(
        problem_id="replay",
        difficulty="unknown",
        question="",
        inputs=inputs,
        outputs=outputs,
        fn_name=fn_name,
        solutions=[],
        url="",
    )

    traj = Trajectory(
        problem=dummy,
        max_turns=max_turns,
        reward_type="continuous",  # Fallbacks for trajectory.py backwards compatibility
        shaped=False,  # Fallbacks for trajectory.py backwards compatibility
        credit_type=credit_type,
    )

    turn_texts = re.split(r"(?=<tool>)", completion)
    turn_texts = [t.strip() for t in turn_texts if t.strip()]
    turn_counter = 1

    for turn_text in turn_texts[:max_turns]:
        if traj.is_done:
            break

        parsed = parse_output(turn_text)
        tool_result = call_tool(
            tool_name=parsed.tool or "execute",
            code=parsed.code or "",
        )

        exec_result = None
        if parsed.tool == "execute" and parsed.code:
            exec_result = execution_reward(
                code=parsed.code,
                inputs=inputs,
                outputs=outputs,
                fn_name=fn_name,
                current_turn=turn_counter,
                ablation_cfg=ablation_cfg,  # Apply ablations per turn
            )

        traj.add_turn(parsed, tool_result, exec_result)
        turn_counter += 1

    return traj


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        # Added entity support so it maps correctly to your tarunbeerelli workspace
        wandb.init(
            project=cfg["wandb_project"], entity=cfg.get("wandb_entity"), config=cfg
        )
        print(f"Training on {accelerator.num_processes} GPUs")
        print(f"Model: {cfg['model_name']}")
        print(f"Mode: {'multi-turn' if cfg['max_turns'] > 1 else 'single-turn'}")

        # Log active ablations to console for sanity checking
        ablation_cfg = cfg.get("ablation", {})
        print(
            f"Active Ablations: {ablation_cfg if ablation_cfg else 'None (Using Defaults)'}"
        )

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Load model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA — only train adapter weights (~0.5% of params)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    # ── Load and prepare data ─────────────────────────────────────────────────
    if is_main:
        print("Loading APPS dataset...")

    problems = load_apps_split(
        cfg["data_dir"],
        split=cfg["train_split"],
        difficulties=cfg.get("difficulties"),
    )
    problems = stratified_sample(problems, n=len(problems))
    dataset = prepare_dataset(problems, tokenizer, cfg["max_prompt_length"])

    if is_main:
        print(f"Training examples: {len(dataset)}")

    # ── GRPO config ───────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        num_generations=cfg.get("num_generations", 8),
        temperature=cfg.get("temperature", 0.9),
        max_completion_length=cfg.get("max_new_tokens", 512),
        output_dir=cfg.get("output_dir", "./results"),
        num_train_epochs=cfg.get("num_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_batch", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.get("learning_rate", 5e-6),
        beta=cfg.get("kl_coef", 0.05),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        report_to="wandb" if is_main else "none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    reward_fn = make_reward_fn(cfg)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=lora_config,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if is_main:
        print("Starting training...")
        print("Watch reward/mean in WandB — should rise over time.")

    trainer.train()

    if is_main:
        trainer.save_model(os.path.join(cfg["output_dir"], "final"))
        tokenizer.save_pretrained(os.path.join(cfg["output_dir"], "final"))
        wandb.finish()
        print("Done.")


if __name__ == "__main__":
    main()
