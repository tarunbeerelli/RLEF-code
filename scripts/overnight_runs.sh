#!/bin/bash
# Overnight training queue — Safe Baseline Dictionary Appending Strategy
# Usage: tmux attach -t rlef_training -> bash scripts/overnight_runs.sh
set -e
cd /workspace/RLEF-code
mkdir -p logs checkpoints results configs/runs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a logs/overnight.log; }

run_train() {
    local epochs=$1 turns=$2 tag=$3
    log "=== START TRAIN: ${tag} ==="

    # 1. Start with your known-good baseline configuration file
    cp configs/train.yaml "configs/runs/${tag}.yaml"

    # 2. Append explicit value overrides to the bottom of the target YAML file safely.
    # Python parsers read the last declared instance of a duplicate key, updating it cleanly.
    cat << CONFIG_APPEND >> "configs/runs/${tag}.yaml"
num_epochs: ${epochs}
max_turns: ${turns}
wandb_project: "rlef-code-optimization"
wandb_run_name: "${tag}"
wandb_mode: "online"
CONFIG_APPEND

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_SILENT=true \
    poetry run accelerate launch \
        --config_file accelerate_config.yaml \
        src/rlef/train.py \
        --config "configs/runs/${tag}.yaml" \
        > "logs/train_${tag}.log" 2>&1
    log "=== END TRAIN: ${tag} ==="
}

run_eval() {
    local checkpoint=$1 tag=$2 diffs=$3
    log "=== START EVAL: ${tag} (Benchmark Tiers: ${diffs}) ==="
    poetry run python -m rlef.eval \
        --checkpoint "${checkpoint}" \
        --tag "${tag}" \
        --benchmark apps \
        --max_examples 200 \
        --difficulties "${diffs}" \
        --data_dir data/raw/APPS \
        --device cuda \
        > "logs/eval_${tag}_${diffs}.log" 2>&1

    local score_line=$(grep "Overall pass@1" "logs/eval_${tag}_${diffs}.log" || echo "Pass rate extraction missing")
    log "=== END EVAL: ${tag} (${diffs}) -> ${score_line} ==="
}

# ──────────────────────────────────────────────────────────────────────────────
# CORE RESEARCH QUEUE EXECUTION MATRIX
# ──────────────────────────────────────────────────────────────────────────────

# ── Run 1: Single-Turn, 1 Epoch Baseline Redo (Q1 Baseline) ──
run_train 1 1 "run1_single_1ep"
cp -r checkpoints/final checkpoints/run1_single_1ep || true
run_eval "checkpoints/run1_single_1ep" "run1_run1_single_1ep" "introductory"

# ── Run 2: Single-Turn, 3 Epochs Main Baseline (Q1 Convergence) ──
run_train 3 1 "run2_single_3ep"
cp -r checkpoints/final checkpoints/run2_single_3ep || true
run_eval "checkpoints/run2_single_3ep" "run2_single_3ep" "introductory"

# ── Run 5: Zero-Shot Out-of-Distribution Generalization (Answers Q4) ──
# Evaluates Run 2's converged checkpoint on unseen Competition metrics
run_eval "checkpoints/run2_single_3ep" "run2_single_3ep_generalization" "competition"

# ── Run 3: Multi-Turn Revision, 2 Epochs (Answers Headline Q2 & Tool Strategy Q5) ──
run_train 2 3 "run3_multiturn"
cp -r checkpoints/final checkpoints/run3_multiturn || true
run_eval "checkpoints/run3_multiturn" "run3_multiturn" "introductory"

# ── Run 4: Trajectory Credit Ablation, 2 Epochs (Answers Algorithmic Q3) ──
# Note: Since 'credit_type' is not parsed on the CLI by train.py, toggle
# this directly in src/rlef/reward.py before executing or during execution.
run_train 2 1 "run4_traj_credit"
cp -r checkpoints/final checkpoints/run4_traj_credit || true
run_eval "checkpoints/run4_traj_credit" "run4_traj_credit" "introductory"

# ──────────────────────────────────────────────────────────────────────────────
# POST-QUEUE SUMMARY LOGGER
# ──────────────────────────────────────────────────────────────────────────────
log "=== ALL COMPONENT RUNS COMPLETED SUCCESSFULLY ==="
log "Summarizing global evaluation scores:"
poetry run python -c "
import glob, json
for path in glob.glob('results/apps_eval_*.json'):
    try:
        with open(path) as f: d = json.load(f)
        s = d.get('summary', {})
        print(f\"  {s.get('tag', 'unknown')}: {s.get('pass_at_1', 0.0):.1%} ({s.get('solved', 0)}/{s.get('total', 0)})\")
    except Exception as e: print(f'Parsing mismatch on {path}: {e}')
" >> logs/overnight.log

tail -n 25 logs/overnight.log
