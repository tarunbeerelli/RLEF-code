#!/bin/bash
# Final Research Suite — Factorial Ablation Study
# Upgraded with crash-handling and corrected eval paths.

# We remove set -e so a single OOM crash doesn't kill the whole night's queue
set +e
cd /workspace/RLEF-code
mkdir -p logs checkpoints results configs/runs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a logs/overnight.log; }

run_train() {
    local tag=$1 multi=$2 lint=$3 step=$4
    log "=== START TRAIN: ${tag} ==="
    cp configs/train.yaml "configs/runs/${tag}.yaml"

    # Dynamically inject the ablation parameters
    cat << CONFIG_APPEND >> "configs/runs/${tag}.yaml"
num_epochs: 3
output_dir: "./checkpoints/${tag}"
max_turns: $( [[ "$multi" == "true" ]] && echo 3 || echo 1 )
wandb_run_name: "${tag}"
ablation:
  use_multi_turn: ${multi}
  use_lint_bonus: ${lint}
  use_step_credit: ${step}
  use_normalization: true
CONFIG_APPEND

    # Echo config for post-run forensic analysis
    log "Config for ${tag}: Multi=${multi}, Lint=${lint}, Step=${step}"

    poetry run accelerate launch --config_file accelerate_config.yaml \
        src/rlef/train.py --config "configs/runs/${tag}.yaml" > "logs/train_${tag}.log" 2>&1

    # Check if training crashed
    local status=${PIPESTATUS[0]}
    if [ $status -ne 0 ]; then
        log "❌ ERROR: Training failed for ${tag} (Exit Code: $status)."
        return 1 # Return error so we can skip its evaluation
    fi

    log "=== END TRAIN: ${tag} ==="
    return 0
}

run_eval() {
    local checkpoint=$1 tag=$2 diffs=$3
    log "=== EVAL: ${tag} (${diffs}) ==="

    # FIXED: using src/rlef/eval.py instead of -m module, keeping your specific args
    poetry run python src/rlef/eval.py \
        --checkpoint "${checkpoint}" --tag "${tag}" --benchmark apps \
        --max_examples 200 --difficulties "${diffs}" --data_dir data/raw/APPS \
        > "logs/eval_${tag}_${diffs}.log" 2>&1

    local status=${PIPESTATUS[0]}
    if [ $status -ne 0 ]; then
        log "⚠️ WARNING: Evaluation failed for ${tag} (Exit Code: $status)."
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# RESEARCH MATRIX
# ──────────────────────────────────────────────────────────────────────────────

log "=== STARTING OVERNIGHT RUNS ==="

# 1. Baseline (Control) - TRAINING ALREADY DONE!
# We just run the eval for it since we fixed the eval script today.
# (Note: Removed "/final" from checkpoint path, verify if TRL adds this folder)
run_eval "./checkpoints/r1_baseline/final" "baseline_eval" "introductory"

# 2. Add Lint Bonus
if run_train "r2_lint_only" false true false; then
    run_eval "./checkpoints/r2_lint_only/final" "lint_eval" "introductory"
fi

# 3. Add Step Credit
if run_train "r3_step_credit" false true true; then
    run_eval "./checkpoints/r3_step_credit/final" "step_eval" "introductory"
fi

# 4. Full Multi-Turn Stack
if run_train "r4_full_stack" true true true; then
    run_eval "./checkpoints/r4_full_stack/final" "full_eval" "introductory"

    # 5. Generalization Test (Run 4 on hard problems)
    run_eval "./checkpoints/r4_full_stack/final" "gen_eval" "competition"
fi

log "=== ALL RUNS COMPLETE ==="
