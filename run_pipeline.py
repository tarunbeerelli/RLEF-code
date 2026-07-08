"""
run_pipeline.py — Sequential Ablation Orchestrator
Dynamically rewrites train.yaml and executes data prep, training, and evaluation.
"""

import subprocess
import yaml
import sys
import os

# ─── 1. FULL 7-RUN ABLATION SEQUENCE ─────────────────────────────────────────

RUNS = [
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
        "name": "run_3_multi_turn_baseline",
        "tags": ["run_3", "multi_turn_baseline", "standard_feedback"],
        "max_turns": 5,
        "use_linear_pass_rate": True,
        "use_step_credit": True,
        "use_turn_penalty": True,
        "use_feedback_bonus": True,
        "use_edge_cases": False,
        "feedback_type": "standard",
    },
    {
        "name": "run_4_macro_heuristic",
        "tags": ["run_4", "macro_heuristic", "consolidated_feedback"],
        "max_turns": 5,
        "use_linear_pass_rate": True,
        "use_step_credit": True,
        "use_turn_penalty": True,
        "use_feedback_bonus": True,
        "use_edge_cases": False,
        "feedback_type": "consolidated",
    },
    {
        "name": "run_5_multi_turn_targeted",
        "tags": ["run_5", "multi_turn", "targeted_feedback"],
        "max_turns": 5,
        "use_linear_pass_rate": True,
        "use_step_credit": True,
        "use_turn_penalty": True,
        "use_feedback_bonus": True,
        "use_edge_cases": False,
        "feedback_type": "last_failed",
    },
    {
        "name": "run_6_anchored_tdd",
        "tags": ["run_6", "anchored_tdd", "test_driven"],
        "max_turns": 5,
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
        "max_turns": 5,
        "use_linear_pass_rate": True,
        "use_step_credit": True,
        "use_turn_penalty": True,
        "use_feedback_bonus": True,
        "use_edge_cases": False,
        "feedback_type": "last_failed",
        "custom_data_path": "./data/apps_run7_phase1.jsonl",
    },
    {
        "name": "run_7_curriculum_phase_2",
        "tags": ["run_7", "curriculum", "phase_2_tdd"],
        "max_turns": 5,
        "use_linear_pass_rate": True,
        "use_step_credit": True,
        "use_turn_penalty": True,
        "use_feedback_bonus": True,
        "use_edge_cases": True,
        "feedback_type": "last_failed",
        "custom_data_path": "./data/apps_run7_phase2.jsonl",
        "base_model_override": "./checkpoints/run_7_curriculum_phase_1_final",
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

    yaml_structure = {
        "wandb_project": "rlef-code",
        "wandb_entity": "tarunbeerelli-northeastern-university",
        "tags": run_config["tags"],
        "num_epochs": 3,
        "start_temp": 0.7,
        "end_temp": 0.2,
        "batch_size": 50,
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
            "lora_path": "./checkpoints/epoch_3",
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

        # Skip standard data prep for Run 7 since split_dataset.py handles it
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

        # Safely archive the final epoch checkpoint so the next run doesn't overwrite it
        run_dir_name = f"./checkpoints/{run['name']}_final"
        run_command(f"mv ./checkpoints/epoch_3 {run_dir_name}", "Checkpoint Archival")

    print("\n🎉 ALL SCHEDULED RUNS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
