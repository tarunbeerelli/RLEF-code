"""
run_pipeline.py — Sequential Ablation Orchestrator
Dynamically rewrites train.yaml and executes data prep, training, and evaluation.
"""

import subprocess
import yaml
import sys
import os

# ─── 1. STAGE-1 FEEDBACK SCREEN ──────────────────────────────────────────────
# Cheap diagnostic to resolve the feedback_type categorical (standard vs
# consolidated vs last_failed). Everything held fixed EXCEPT feedback_type:
#   - linear pass-rate reward ON (dense info is the settled win)
#   - multi-turn ON at 3 turns (reduced from 5 for the screen)
#   - all shaping bonuses ON, use_edge_cases OFF (that's Stage 2)
#   - train_cap 300 (stratified ~190/78/32), 1 epoch, 10x10 = 100 concurrent
#   - max_tokens 1200 SAME across all three arms (do NOT vary — would confound)
#   - full 702 eval (keep competition visibility)
# Winner's feedback_type carries into Stage 2 (example-injection, edge-cases)
# and the final best model.

_SCREEN_COMMON = {
    "max_turns": 3,
    "use_linear_pass_rate": True,
    "use_step_credit": True,
    "use_turn_penalty": True,
    "use_feedback_bonus": True,
    "use_edge_cases": False,
    "train_cap": 300,
    "num_epochs": 1,
    "batch_size": 10,
    "num_generations": 10,
    "learning_rate": 2.0e-5,
    "lora_rank": 32,
    "lora_alpha": 64,
    "kl_beta": 0.1,
    "max_kl_stop": 0.5,
    "max_model_len": 16384,
    "max_tokens": 1200,
    "gpu_memory_utilization": 0.42,
    "start_temp": 0.7,
    "end_temp": 0.7,
}

RUNS = [
    {
        **_SCREEN_COMMON,
        "name": "screen_standard",
        "tags": ["screen", "feedback_standard"],
        "feedback_type": "standard",
    },
    {
        **_SCREEN_COMMON,
        "name": "screen_consolidated",
        "tags": ["screen", "feedback_consolidated"],
        "feedback_type": "consolidated",
    },
    {
        **_SCREEN_COMMON,
        "name": "screen_last_failed",
        "tags": ["screen", "feedback_last_failed"],
        "feedback_type": "last_failed",
    },
]

# ─── ARCHIVED: full ablation suite (Stage 2 + curriculum, restore when needed) ─
# Stage 2 (run on Stage-1 winner W): W+example_injection, W+example+edge_cases.
# Curriculum: high-variance bet on COMPETITION bucket, built on best config,
# judged on competition pass@1 — run as its own experiment, not in this screen.
ARCHIVE_RUNS = [
    {
        "name": "run_1_sparse_baseline",
        "tags": ["run_1", "sparse_baseline", "zero_shot"],
        "max_turns": 1,
        "use_linear_pass_rate": False,
        "use_step_credit": False,
        "use_turn_penalty": False,
        "use_feedback_bonus": False,
        "use_edge_cases": False,
        "feedback_type": "none",
    },
    {
        "name": "run_2_dense_baseline",
        "tags": ["run_2", "dense_baseline", "zero_shot"],
        "max_turns": 1,
        "use_linear_pass_rate": True,
        "use_step_credit": False,
        "use_turn_penalty": False,
        "use_feedback_bonus": False,
        "use_edge_cases": False,
        "feedback_type": "none",
    },
    {
        "name": "run_6_anchored_tdd",
        "tags": ["run_6", "anchored_tdd", "test_driven"],
        "max_turns": 3,
        "use_linear_pass_rate": True,
        "use_step_credit": True,
        "use_turn_penalty": True,
        "use_feedback_bonus": True,
        "use_edge_cases": True,
        "feedback_type": "last_failed",
    },
    {
        "name": "run_7_curriculum_phase_1",
        "tags": ["run_7", "curriculum", "phase_1_execution"],
        "max_turns": 3,
        "use_linear_pass_rate": True,
        "use_step_credit": True,
        "use_turn_penalty": True,
        "use_feedback_bonus": True,
        "use_edge_cases": False,
        "feedback_type": "last_failed",
        "custom_data_path": "./data/apps_run7_phase1.jsonl",
        "num_epochs": 2,
    },
    {
        "name": "run_7_curriculum_phase_2",
        "tags": ["run_7", "curriculum", "phase_2_tdd"],
        "max_turns": 3,
        "use_linear_pass_rate": True,
        "use_step_credit": True,
        "use_turn_penalty": True,
        "use_feedback_bonus": True,
        "use_edge_cases": True,
        "feedback_type": "last_failed",
        "custom_data_path": "./data/apps_run7_phase2.jsonl",
        "base_model_override": "./checkpoints/run_7_curriculum_phase_1_final",
        "num_epochs": 2,
    },
]

# ─── 2. PIPELINE EXECUTION ENGINE ────────────────────────────────────────────


def run_command(cmd: str, step_name: str):
    """Executes a shell command and halts the pipeline on failure."""
    print(f"\n[{step_name}] Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n CRITICAL FAILURE in {step_name}. Halting pipeline.")
        sys.exit(1)
    print(f"{step_name} completed successfully.")


def update_yaml_config(run_config: dict):
    """Overwrites configs/train.yaml with the exact physics for the current run."""
    dataset_target = run_config.get(
        "custom_data_path", f"./data/apps_{run_config['name']}.jsonl"
    )

    resolved_epochs = run_config.get("num_epochs", 1)
    yaml_structure = {
        "wandb_project": run_config.get("wandb_project", "rlef-code2"),
        "wandb_entity": "tarunbeerelli-northeastern-university",
        "run_name": run_config[
            "name"
        ],  # descriptive W&B run name (e.g. run_1_sparse_baseline)
        "tags": run_config["tags"],
        "num_epochs": resolved_epochs,
        "start_temp": run_config.get("start_temp", 0.7),
        "end_temp": run_config.get("end_temp", 0.7),  # fixed temp: no across-run anneal
        "batch_size": run_config.get(
            "batch_size", 10
        ),  # bs x num_generations = concurrent load
        "num_generations": run_config.get(
            "num_generations", 12
        ),  # rollouts per problem for GRPO groups
        "learning_rate": run_config.get(
            "learning_rate", 2.0e-5
        ),  # stable LR (1e-4 diverged)
        "lora_rank": run_config.get("lora_rank", 32),
        "lora_alpha": run_config.get("lora_alpha", 64),
        "train_cap": run_config.get(
            "train_cap", 1200
        ),  # stratified total training problems
        "stratify_mode": run_config.get("stratify_mode", "proportional"),
        "kl_beta": run_config.get("kl_beta", 0.1),  # tighter KL leash
        "max_kl_stop": run_config.get(
            "max_kl_stop", 0.5
        ),  # hard early-stop on divergence
        # --- length + memory (config-driven; screen uses 16384 window @ util 0.42) ---
        "max_model_len": run_config.get(
            "max_model_len", 16384
        ),  # full multi-turn KV window
        "max_tokens": run_config.get(
            "max_tokens", 1200
        ),  # per-completion cap; SAME across screen arms
        "gpu_memory_utilization": run_config.get("gpu_memory_utilization", 0.42),
        "eval_gpu_memory_utilization": run_config.get(
            "eval_gpu_memory_utilization", 0.60
        ),
        "stop_tokens": ["</code>"],
        "ablation": {
            "dataset_path": dataset_target,
            "max_turns": run_config["max_turns"],
            "use_linear_pass_rate": run_config["use_linear_pass_rate"],
            "use_step_credit": run_config["use_step_credit"],
            "use_turn_penalty": run_config["use_turn_penalty"],
            "use_feedback_bonus": run_config["use_feedback_bonus"],
            "use_edge_cases": run_config["use_edge_cases"],
            "feedback_type": run_config["feedback_type"],
        },
        "evaluation": {
            "dataset_path": "./data/apps_eval.jsonl",
            # Eval loads the checkpoint from the LAST trained epoch. Always derived
            # from resolved_epochs so it can never point at a non-existent epoch.
            "lora_path": f"./checkpoints/epoch_{resolved_epochs}",
            "output_filename": f"apps_eval_{run_config['name']}.json",
        },
    }

    if "base_model_override" in run_config:
        yaml_structure["lora_resume_path"] = run_config["base_model_override"]

    os.makedirs("configs", exist_ok=True)
    with open("configs/train.yaml", "w") as f:
        yaml.dump(yaml_structure, f, sort_keys=False)

    print(f"⚙️ Config locked for {run_config['name']}")


# ─── 3. MASTER LOOP ──────────────────────────────────────────────────────────


def main():
    print("🚀 Commencing Automated Multi-Run Sequence...")

    for idx, run in enumerate(RUNS, 1):
        print("\n=======================================================")
        print(f"   STARTING PHASE {idx}/{len(RUNS)}: {run['name'].upper()}")
        print("=======================================================")

        update_yaml_config(run)

        run_command(
            "PYTHONPATH=src python3 src/rlef/prepare_openrlhf_data.py",
            "Data Preparation",
        )

        run_command("ray stop --force", "vLLM Memory Flush")

        run_command(
            "PYTHONPATH=src python3 src/rlef/train_agent.py", "RLHF Training Loop"
        )

        run_command(
            "PYTHONPATH=src python3 src/rlef/evaluate.py", "Checkpoint Evaluation"
        )

        # Safely archive the final epoch checkpoint so the next run doesn't overwrite it.
        # Default 1 matches update_yaml_config's resolved_epochs default.
        target_epoch = run.get("num_epochs", 1)
        run_dir_name = f"./checkpoints/{run['name']}_final"
        run_command(
            f"mv ./checkpoints/epoch_{target_epoch} {run_dir_name}",
            "Checkpoint Archival",
        )

    print("\n🎉 ALL SCHEDULED RUNS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
