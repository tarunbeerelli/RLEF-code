#!/bin/bash
# Overnight training queue — runs sequentially, evals after each
# Usage: nohup bash scripts/overnight_runs.sh > logs/overnight.log 2>&1 &
set -e
cd /workspace/RLEF-code
mkdir -p logs checkpoints results configs/runs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a logs/overnight.log; }

run_train() {
    local config=$1 tag=$2
    log "=== START TRAIN: $tag ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    poetry run accelerate launch \
        --config_file accelerate_config.yaml \
        src/rlef/train.py \
        --config "$config" \
        > "logs/train_${tag}.log" 2>&1
    log "=== END TRAIN: $tag ==="
}

run_eval() {
    local checkpoint=$1 tag=$2 diffs=$3
    log "=== START EVAL: $tag (Diff: $diffs) ==="
    poetry run python -m rlef.eval \
        --checkpoint "$checkpoint" \
        --tag "$tag" \
        --benchmark apps \
        --max_examples 200 \
        --difficulties "$diffs" \
        --data_dir data/raw/APPS \
        --device cuda \
        > "logs/eval_${tag}_${diffs}.log" 2>&1
    log "=== END EVAL: $tag ($diffs) ($(grep 'Overall pass@1' logs/eval_${tag}_${diffs}.log || echo 'N/A')) ==="
}

# ── Run 1: single-turn, 1 epoch (Baseline Redo) ────────────────────────────────
cat << 'CONFIG' > configs/runs/run1_single_1ep.yaml
num_epochs: 1
max_turns: 1
credit_type: "step"
difficulties: ["introductory"]
reward_type: "continuous"
shaped: false
CONFIG
run_train "configs/runs/run1_single_1ep.yaml" "run1_single_1ep"
cp -r checkpoints/final checkpoints/run1_single_1ep || true
run_eval "checkpoints/run1_single_1ep" "run1_single_1ep" "introductory"

# ── Run 2: single-turn, 3 epochs (Main Converged Baseline for Q1) ──────────────
cat << 'CONFIG' > configs/runs/run2_single_3ep.yaml
num_epochs: 3
max_turns: 1
credit_type: "step"
difficulties: ["introductory"]
reward_type: "continuous"
shaped: false
CONFIG
run_train "configs/runs/run2_single_3ep.yaml" "run2_single_3ep"
cp -r checkpoints/final checkpoints/run2_single_3ep || true
run_eval "checkpoints/run2_single_3ep" "run2_single_3ep" "introductory"

# ── Run 5: Zero-Shot Generalization Checkpoint Evaluation (Answers Q4) ────────
# Evaluates Run 2's converged model on completely unseen INTERVIEW problems
run_eval "checkpoints/run2_single_3ep" "run2_single_3ep_generalization" "interview"

# ── Run 3: multi-turn 3 turns, 2 epochs (Headline Finding for Q2 & Q5) ─────────
cat << 'CONFIG' > configs/runs/run3_multiturn.yaml
num_epochs: 2
max_turns: 3
credit_type: "step"
difficulties: ["introductory"]
reward_type: "continuous"
shaped: false
CONFIG
run_train "configs/runs/run3_multiturn.yaml" "run3_multiturn"
cp -r checkpoints/final checkpoints/run3_multiturn || true
run_eval "checkpoints/run3_multiturn" "run3_multiturn" "introductory"

# ── Run 4: trajectory credit ablation, 2 epochs (Algorithmic Q3) ───────────────
cat << 'CONFIG' > configs/runs/run4_traj_credit.yaml
num_epochs: 2
max_turns: 1
credit_type: "trajectory"
difficulties: ["introductory"]
reward_type: "continuous"
shaped: false
CONFIG
run_train "configs/runs/run4_traj_credit.yaml" "run4_traj_credit"
cp -r checkpoints/final checkpoints/run4_traj_credit || true
run_eval "checkpoints/run4_traj_credit" "run4_traj_credit" "introductory"

# ── Run 6: Mixed-Tier Training, 2 epochs (Stronger Generalization Baseline) ───
cat << 'CONFIG' > configs/runs/run6_mixed_tier.yaml
num_epochs: 2
max_turns: 1
credit_type: "step"
difficulties: ["introductory", "interview"]
reward_type: "continuous"
shaped: false
CONFIG
run_train "configs/runs/run6_mixed_tier.yaml" "run6_mixed_tier"
cp -r checkpoints/final checkpoints/run6_mixed_tier || true
run_eval "checkpoints/run6_mixed_tier" "run6_mixed_tier" "introductory"
run_eval "checkpoints/run6_mixed_tier" "run6_mixed_tier" "interview"

log "=== ALL RUNS COMPLETE ==="
log "Parsing final results..."
poetry run python -c "
import glob, json
for path in glob.glob('results/apps_eval_*.json'):
    try:
        with open(path) as f:
            d = json.load(f)
        s = d.get('summary', {})
        print(f\"  {s.get('tag', 'unknown')}: {s.get('pass_at_1', 0.0):.1%} ({s.get('solved', 0)}/{s.get('total', 0)})\")
    except Exception as e:
        print(f'Error reading {path}: {e}')
" >> logs/overnight.log
