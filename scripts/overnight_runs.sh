#!/bin/bash
# Final Research Suite — Factorial Ablation Study
set -e
cd /workspace/RLEF-code
mkdir -p logs checkpoints results configs/runs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a logs/overnight.log; }

run_train() {
    local tag=$1 multi=$2 lint=$3 step=$4
    log "=== START TRAIN: ${tag} ==="
    cp configs/train.yaml "configs/runs/${tag}.yaml"

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
    log "=== END TRAIN: ${tag} ==="
}

run_eval() {
    local checkpoint=$1 tag=$2 diffs=$3
    log "=== EVAL: ${tag} (${diffs}) ==="
    poetry run python -m rlef.eval \
        --checkpoint "${checkpoint}" --tag "${tag}" --benchmark apps \
        --max_examples 200 --difficulties "${diffs}" --data_dir data/raw/APPS \
        > "logs/eval_${tag}_${diffs}.log" 2>&1
}

# ──────────────────────────────────────────────────────────────────────────────
# RESEARCH MATRIX
# ──────────────────────────────────────────────────────────────────────────────

# 1. Baseline (Control)
run_train "r1_baseline" false false false
run_eval "checkpoints/r1_baseline/final" "baseline_eval" "introductory"

# 2. Add Lint Bonus
run_train "r2_lint_only" false true false
run_eval "checkpoints/r2_lint_only/final" "lint_eval" "introductory"

# 3. Add Step Credit
run_train "r3_step_credit" false true true
run_eval "checkpoints/r3_step_credit/final" "step_eval" "introductory"

# 4. Full Multi-Turn Stack
run_train "r4_full_stack" true true true
run_eval "checkpoints/r4_full_stack/final" "full_eval" "introductory"

# 5. Generalization Test (Run 4 on hard problems)
run_eval "checkpoints/r4_full_stack/final" "gen_eval" "competition"

log "=== ALL RUNS COMPLETE ==="
