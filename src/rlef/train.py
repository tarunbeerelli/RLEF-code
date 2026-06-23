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

Ablations: set in configs/train.yaml
  reward_type: continuous | binary
  shaped:      true | false
  credit_type: step | trajectory
  max_turns:   1 | 3
"""

import argparse
import os
import random
import re

import torch
import wandb
import yaml
from accelerate import Accelerator
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from rlef.data import APPSProblem, difficulty_split, load_apps_split
from rlef.prompt import format_prompt, parse_output
from rlef.reward import execution_reward
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

    GRPOTrainer expects each example to have a "prompt" field
    containing the tokenized prompt. We also store problem metadata
    as extra columns so the reward function can access test cases.

    The reward function receives these columns as kwargs — this is
    how GRPO passes context from the dataset to the reward fn.
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
    This prevents the model from exploiting easy problems for cheap reward.
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

    GRPOTrainer signature:
      reward_fn(completions, prompts, **dataset_columns) -> list[float]

    completions:     list of model-generated strings (one per rollout)
    dataset_columns: anything stored in the dataset is passed as kwargs
                     — this is how we get inputs/outputs/fn_name per problem
    """
    reward_type = cfg.get("reward_type", "continuous")
    shaped = cfg.get("shaped", True)
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
                    reward_type=reward_type,
                    shaped=shaped,
                    difficulty=kwargs.get("difficulty", ["introductory"])[i]
                    if kwargs.get("difficulty")
                    else "introductory",
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
                # Note: in GRPO the full trajectory is one "completion"
                # We replay it here to compute the reward
                # The model already generated all turns — we just score them
                traj = _replay_trajectory(
                    completion=completion,
                    inputs=inp,
                    outputs=out,
                    fn_name=fn,
                    max_turns=max_turns,
                    reward_type=reward_type,
                    shaped=shaped,
                    credit_type=credit_type,
                )
                rewards.append(traj.final_reward)

                if wandb.run:
                    wandb.log(traj.to_log_dict())

        return rewards

    return reward_fn


def _replay_trajectory(
    completion: str,
    inputs: list[str],
    outputs: list[str],
    fn_name: str | None,
    max_turns: int,
    reward_type: str,
    shaped: bool,
    credit_type: str,
) -> Trajectory:
    """
    Replay a multi-turn completion to compute the reward.

    In multi-turn GRPO the model generates the full trajectory as
    one completion string. We split it by turns, execute each tool
    call, and return the final trajectory with reward assigned.
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
        reward_type=reward_type,
        shaped=shaped,
        credit_type=credit_type,
    )

    # split completion into turns by the <tool> tag
    # each turn starts with a <tool> tag
    turn_texts = re.split(r"(?=<tool>)", completion)
    turn_texts = [t.strip() for t in turn_texts if t.strip()]

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
                reward_type=reward_type,
                shaped=shaped,
            )

        traj.add_turn(parsed, tool_result, exec_result)

    return traj


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        wandb.init(project=cfg["wandb_project"], config=cfg)
        print(f"Training on {accelerator.num_processes} GPUs")
        print(f"Model: {cfg['model_name']}")
        print(f"Mode: {'multi-turn' if cfg['max_turns'] > 1 else 'single-turn'}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Load model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA — only train adapter weights (~0.5% of params)
    # keeps optimizer states tiny, fits in 48GB with DDP
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        num_generations=cfg["num_generations"],
        temperature=cfg["temperature"],
        max_completion_length=cfg["max_new_tokens"],
        beta=cfg["kl_coef"],
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
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
