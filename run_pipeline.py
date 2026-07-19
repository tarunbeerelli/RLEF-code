"""
run_pipeline.py — Sequential Ablation Orchestrator
Dynamically rewrites train.yaml and executes data prep, training, and evaluation.
"""

import subprocess
import yaml
import sys
import os

# Propagate the CUDA fragmentation setting to all training/eval subprocesses this
# orchestrator spawns. run_pipeline never imports torch itself, so this can sit
# after imports (no E402); children read it from the inherited environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Use the local HF cache instead of contacting the Hub at model-load time. The 7B
# is already cached from training, so eval/train need no network — this avoids the
# HTTP 429 (rate limit) that can hit engine startup on fresh/shared instances.
# Subprocesses (train_agent, evaluate) inherit these from the environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ─── 1. RUN SEQUENCE ─────────────────────────────────────────────────────────
# Part A: RE-EVAL the three screen checkpoints (3/4/5) under the CORRECTED
#         per-arm eval prompt (right max_turns AND right feedback_type, plus
#         eval feedback strings now mirror training exactly). No retraining —
#         just a clean feedback comparison on checkpoints we already have.
# Part B: BUILD sequence, concluding last_failed is the feedback winner:
#   - run_5_proper: last_failed @ train_cap 1200 (doubles as curriculum phase 1)
#   - run_6_edge_cases: winner + edge_cases + turn-0 anchor @ train_cap 1500
#   - phase_2: run_6 specs on top of phase_1 checkpoint @ train_cap 1200
# All build runs: 1 epoch, bs 10 x 12 gen = 120 concurrent, stable optimizer.

# Shared physics for the multi-turn build runs.
_BUILD_COMMON = {
    "max_turns": 3,
    "use_linear_pass_rate": True,
    "use_step_credit": True,
    "use_turn_penalty": True,
    "use_feedback_bonus": True,
    "num_epochs": 1,
    "batch_size": 10,
    "num_generations": 12,  # 10 x 12 = 120 concurrent (VRAM-checked below)
    "learning_rate": 2.0e-5,
    "lora_rank": 32,
    "lora_alpha": 64,
    "kl_beta": 0.1,
    "max_kl_stop": 0.5,
    "max_model_len": 16384,
    "max_tokens": 1200,
    "gpu_memory_utilization": 0.38,  # vLLM gets ~53GB; leaves ~87GB for training-side backward at 16384. (0.48 starved training -> OOM)
    "start_temp": 0.7,
    "end_temp": 0.7,
}

# Physics shared by the three eval-only re-evals (must match how they were TRAINED:
# screen arms were max_turns 3, so eval rebuilds the prompt at max_turns 3).
_REEVAL_COMMON = {
    "eval_only": True,
    "max_turns": 3,
    "use_linear_pass_rate": True,
    "use_step_credit": True,
    "use_turn_penalty": True,
    "use_feedback_bonus": True,
    "use_edge_cases": False,
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
    "gpu_memory_utilization": 0.60,  # eval-only: no training memory, can use more
    "start_temp": 0.7,
    "end_temp": 0.7,
    # NOTE: eval scores the full held-out set (up to 250/bucket -> ~702 total:
    # 250 intro + 250 interview + 202 competition). It does NOT read train_cap;
    # that field is training-only, so it's omitted here to avoid confusion.
}

RUNS = [
    # ── RUN A2: adaptation-REACH test ────────────────────────────────────────
    # run_5 established weak-but-valid self-correction signal with the MINIMAL
    # adaptation surface (rank-32 LoRA on the 4 attention projections only). A2 holds
    # rank fixed at 32 and widens the surface to all 7 linear layers — adding the MLP
    # (gate/up/down_proj), where most of the transformer's computation lives — to test
    # whether adaptation REACH, not rank/capacity, is what converts the weak signal to
    # strong. Everything else is identical to run_5 (full distribution, last_failed,
    # 3-turn, no edge cases) so REACH is the only variable. All-7 is 4x the trainable
    # params (20M -> 81M) but only +0.7GB state — activations dominate and are
    # unchanged. Writes its own manifest so a future curriculum phase can reuse it.
    {
        **_BUILD_COMMON,
        "name": "run_A2_all_layers",
        "tags": ["run_A2", "last_failed", "all_linear_layers", "reach_test"],
        "feedback_type": "last_failed",
        "use_edge_cases": False,
        "train_cap": 1200,
        "curriculum_mode": "full",
        "max_turns": 3,
        "batch_size": 12,  # 12 x 12 = 144 concurrent, ~100 steps/epoch
        "num_generations": 12,
        "gpu_memory_utilization": 0.60,  # KV room +13G, PyTorch 56G vs ~38G peak
        "lora_rank": 32,  # UNCHANGED from run_5 — reach is the variable, not rank
        "lora_alpha": 64,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # all 7 (adds MLP)
        "checkpoint_every": 40,
        "write_manifest": True,
        "manifest_path": "./data/runA2_trained_ids.json",
    },
    # ── phase_2 checkpoint evals — COMPLETE (done 2026-07-18). Commented out. ──
    # best-KL: intro 7.6 / interview 3.6 / comp 0.0 (pass@5 11.6/3.6/0.0)
    # epoch_1: intro 7.6 / interview 2.8 / comp 0.5 (pass@5 11.2/3.2/0.5)
    # epoch_2 (final, BEST): intro 8.8 / interview 3.2 / comp 0.0 (pass@5 14.8/4.0/0.5)
    # {**_REEVAL_COMMON,
    #     "name": "reeval_phase2_best_kl",
    #     "tags": ["reeval", "run_7", "curriculum", "best_kl_ckpt"],
    #     "feedback_type": "last_failed",
    #     "use_edge_cases": True,
    #     "max_turns": 5,
    #     "eval_checkpoint": "./checkpoint/last_good_lora"},
    # end-of-EPOCH-1 checkpoint. The archival step mv'd epoch_2 -> _final and left
    # epoch_1 in place, so this is the policy at the epoch boundary — a full first
    # pass over the hard set, BEFORE epoch-2's seen-replay drift (rising KL, faster
    # commits, more wrong outputs). Three-way compare: best-KL vs epoch-1 vs epoch-2
    # (end, already eval'd). If epoch-1 >= epoch-2 on hard-bucket pass@k, epoch 2 was
    # net-negative for generalization — a clean train-vs-eval divergence result.
    # NOTE: verify ./checkpoints/epoch_1 exists before running (ls it); if the run
    # early-stopped or the dir was cleaned, this will fail.
    # {**_REEVAL_COMMON,
    #     "name": "reeval_phase2_epoch1",
    #     "tags": ["reeval", "run_7", "curriculum", "epoch_1_ckpt"],
    #     "feedback_type": "last_failed",
    #     "use_edge_cases": True,
    #     "max_turns": 5,
    #     "eval_checkpoint": "./checkpoints/epoch_1"},
    # {**_REEVAL_COMMON,
    #     "name": "reeval_phase2_epoch2_final",
    #     "tags": ["reeval", "run_7", "curriculum", "epoch_2_final_ckpt"],
    #     "feedback_type": "last_failed",
    #     "use_edge_cases": True,
    #     "max_turns": 5,
    #     "eval_checkpoint": "./checkpoints/run_7_curriculum_phase2_final"},
    # ── Part A: re-evals COMPLETE (done 2026-07-14). Kept for reference; commented
    #    out so the pipeline starts at run_5. Re-enable if re-evaluation is needed. ──
    # {**_REEVAL_COMMON,
    #     "name": "reeval_screen_standard",
    #     "tags": ["reeval", "feedback_standard"],
    #     "feedback_type": "standard",
    #     "eval_checkpoint": "./checkpoints/screen_standard_final"},
    # {**_REEVAL_COMMON,
    #     "name": "reeval_screen_consolidated",
    #     "tags": ["reeval", "feedback_consolidated"],
    #     "feedback_type": "consolidated",
    #     "eval_checkpoint": "./checkpoints/screen_consolidated_final"},
    # {**_REEVAL_COMMON,
    #     "name": "reeval_screen_last_failed",
    #     "tags": ["reeval", "feedback_last_failed"],
    #     "feedback_type": "last_failed",
    #     "eval_checkpoint": "./checkpoints/screen_last_failed_final"},
    # ── Part B: build sequence (last_failed = feedback winner) ──
    # PHASE 1 (run_5) — checkpoint + manifest RECOVERED with the instance, so this is
    # commented out and phase_2 resumes from them directly. Re-enable ONLY if
    # ./checkpoints/run_5_proper_phase1_final or ./data/run5_trained_ids.json is
    # missing (verify with: ls the checkpoint dir && ls the manifest).
    # {**_BUILD_COMMON,
    #     "name": "run_5_proper_phase1",
    #     "tags": ["run_5", "last_failed", "phase_1_base"],
    #     "feedback_type": "last_failed",
    #     "use_edge_cases": False,
    #     "train_cap": 1200,
    #     "curriculum_mode": "full",
    #     "max_turns": 3,
    #     "batch_size": 12,
    #     "num_generations": 12,
    #     "gpu_memory_utilization": 0.60,
    #     "checkpoint_every": 40,
    #     "write_manifest": True,
    #     "manifest_path": "./data/run5_trained_ids.json"},
    # run_6 COMPLETE (trained + drifted; end-checkpoint eval = 3.6%). We do NOT
    # retrain it (diverged at both 2e-5 and 1e-5 — edge-case objective is unstable).
    # Instead, EVAL-ONLY on its saved last_good (least-drifted) checkpoint to test
    # drift-vs-capacity: if last_good scores near run_5, drift caused the collapse;
    # if it's also ~4-6%, edge cases genuinely don't transfer. use_edge_cases=True
    # so the eval prompt matches how run_6 was trained.
    # run_6 last_good eval COMPLETE (done 2026-07-17; intro 2.4% — matches the end
    # checkpoint, confirming the test-driven objective doesn't transfer at 7B, and
    # that the collapse is the OBJECTIVE, not a checkpoint-selection artifact).
    # Commented out so the pipeline starts at phase_2. Re-enable only to redo it.
    # {**_REEVAL_COMMON,
    #     "name": "reeval_run_6_last_good",
    #     "tags": ["reeval", "run_6", "edge_cases", "last_good_ckpt"],
    #     "feedback_type": "last_failed",
    #     "use_edge_cases": True,
    #     "eval_checkpoint": "./checkpoint/last_good_lora"},
    # --- run_6 TRAINING block, disabled (kept for reference; do NOT retrain) ---
    # {**_BUILD_COMMON,
    #     "name": "run_6_edge_cases",
    #     "tags": ["run_6", "last_failed", "edge_cases", "tdd"],
    #     "feedback_type": "last_failed",
    #     "use_edge_cases": True,
    #     "train_cap": 1500,
    #     "curriculum_mode": "full",
    #     "batch_size": 8,
    #     "gpu_memory_utilization": 0.45,
    #     "learning_rate": 1.0e-5,
    #     "kl_beta": 0.15,
    #     "max_kl_stop": 0.3,
    #     "checkpoint_every": 40},
    # curriculum PHASE 2 — resume run_5 checkpoint. Dataset = ALL unseen hard
    # (556) + 15% stratified seen-replay (~83) ≈ 639, ~87% hard / ~87% fresh.
    # 2 epochs to recover step count on the smaller set; KL early-stop guards the
    # 2nd-epoch overfitting/reward-hacking risk. Edge cases ON.
    # max_turns 5 (was 3): hard problems are exactly where a 4th/5th correction turn
    # can convert near-misses; phase_2 rolling_success was flat at 3 turns.
    # MEMORY: 5-turn sequences (~9500 tok peak) need MORE KV than run_6's 3-turn,
    # so phase_2 needs LOWER concurrency than run_6 (can't reuse the same numbers).
    # bs 8 x gen 12 = 96 concurrent @ util 0.60 — the standard config, now feasible
    # at 5 turns because FlashAttention-2 + gradient checkpointing cut the training
    # peak to ~36GB (from ~54GB), freeing enough of the 141GB for vLLM to hold the
    # 96-concurrent 5-turn KV cache. vLLM 84GB (KV room 69 vs ~52 need, +17 margin),
    # PyTorch 56GB vs ~36 peak. Full-depth GRPO groups (gen 12) on the hard buckets.
    # phase_2 TRAINING COMPLETE (done 2026-07-18). Commented out so only the best-KL
    # eval above runs. Re-enable only to retrain. End checkpoint already eval'd.
    # {**_BUILD_COMMON,
    #     "name": "run_7_curriculum_phase2",
    #     "tags": ["run_7", "curriculum", "phase_2_hard_specialize"],
    #     "feedback_type": "last_failed",
    #     "use_edge_cases": True,
    #     "max_turns": 5,
    #     "batch_size": 8,
    #     "num_generations": 12,
    #     "gpu_memory_utilization": 0.60,
    #     "train_cap": 2000,
    #     "num_epochs": 2,
    #     "curriculum_mode": "hard_specialize",
    #     "replay_frac": 0.15,
    #     "checkpoint_every": 40,
    #     "manifest_path": "./data/run5_trained_ids.json",
    #     "base_model_override": "./checkpoints/run_5_proper_phase1_final"},
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
        "lora_target_modules": run_config.get(
            "lora_target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
        "train_cap": run_config.get(
            "train_cap", 1200
        ),  # stratified total training problems
        "stratify_mode": run_config.get("stratify_mode", "proportional"),
        "kl_beta": run_config.get("kl_beta", 0.1),  # tighter KL leash
        "max_kl_stop": run_config.get(
            "max_kl_stop", 0.5
        ),  # hard early-stop on divergence
        "checkpoint_every": run_config.get(
            "checkpoint_every", 40
        ),  # periodic good-ckpt cadence
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
            "curriculum_mode": run_config.get("curriculum_mode", "full"),
            "write_manifest": run_config.get("write_manifest", False),
            "manifest_path": run_config.get(
                "manifest_path", "./data/run5_trained_ids.json"
            ),
            "replay_frac": run_config.get("replay_frac", 0.15),
        },
        "evaluation": {
            "dataset_path": "./data/apps_eval.jsonl",
            # Eval loads the checkpoint from the LAST trained epoch. Always derived
            # from resolved_epochs so it can never point at a non-existent epoch.
            # eval_only: point at the already-archived checkpoint instead.
            "lora_path": (
                run_config.get(
                    "eval_checkpoint", f"./checkpoints/{run_config['name']}_final"
                )
                if run_config.get("eval_only")
                # Training runs: eval reads the stable _final dir, which the loop
                # populates from the best-existing epoch AFTER training and BEFORE
                # eval. Robust to 2-epoch runs and KL early-stops.
                else f"./checkpoints/{run_config['name']}_final"
            ),
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

        # Eval-only: re-evaluate an EXISTING checkpoint under the (now corrected)
        # per-arm eval prompt. No data-prep-train, no archival. Used to get a clean
        # feedback comparison on checkpoints 3/4/5 that were already trained.
        if run.get("eval_only"):
            # Still refresh eval data so the arm-agnostic v2 eval file exists.
            run_command(
                "PYTHONPATH=src python3 src/rlef/prepare_openrlhf_data.py",
                "Eval Data Refresh (eval_only)",
            )
            run_command("ray stop --force", "vLLM Memory Flush")
            run_command(
                "PYTHONPATH=src python3 src/rlef/evaluate.py",
                f"Re-Eval Existing Checkpoint ({run['name']})",
            )
            continue

        run_command(
            "PYTHONPATH=src python3 src/rlef/prepare_openrlhf_data.py",
            "Data Preparation",
        )

        run_command("ray stop --force", "vLLM Memory Flush")

        run_command(
            "PYTHONPATH=src python3 src/rlef/train_agent.py", "RLHF Training Loop"
        )

        # Archive the best-existing epoch to a stable _final dir BEFORE eval.
        # For 2-epoch runs we prefer the highest epoch that actually exists; if
        # epoch 2 never completed (crash) or early-stopped, fall back to the
        # highest epoch present. This also makes eval's lora_path (_final) stable.
        target_epoch = run.get("num_epochs", 1)
        run_dir_name = f"./checkpoints/{run['name']}_final"
        archive_cmd = (
            f"rm -rf {run_dir_name}; "
            f"for e in $(seq {target_epoch} -1 1); do "
            f"  if [ -d ./checkpoints/epoch_$e ]; then "
            f"    mv ./checkpoints/epoch_$e {run_dir_name}; "
            f"    echo archived epoch_$e '->' {run_dir_name}; break; "
            f"  fi; done"
        )
        run_command(archive_cmd, "Checkpoint Archival (best-existing epoch)")

        run_command(
            "PYTHONPATH=src python3 src/rlef/evaluate.py", "Checkpoint Evaluation"
        )

    print("\n🎉 ALL SCHEDULED RUNS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
