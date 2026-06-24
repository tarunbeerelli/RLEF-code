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
    local checkpoint=$1 tag=$2
    log "=== START EVAL: $tag ==="
    poetry run python -m rlef.eval \
        --checkpoint "$checkpoint" \
        --tag "$tag" \
        --benchmark apps \
        --max_examples 200 \
        --difficulties introductory \
        --data_dir data/raw/APPS \
        --device cuda \
        > "logs/eval_${tag}.log" 2>&1
    log "=== END EVAL: $tag ($(grep 'Overall pass@1' logs/eval_${tag}.log)) ==="
}

# ── Run 2: single-turn, 3 epochs (main result) ────────────────────────────────
cp configs/train.yaml configs/runs/run2_single_3ep.yaml
run_train "configs/runs/run2_single_3ep.yaml" "run2_single_3ep"
cp -r checkpoints/final checkpoints/run2_single_3ep
run_eval "checkpoints/run2_single_3ep" "run2_single_3ep"

# ── Run 3: multi-turn 3 turns, 2 epochs (headline finding) ───────────────────
cp configs/train.yaml configs/runs/run3_multiturn.yaml
sed -i 's/max_turns: 1/max_turns: 3/' configs/runs/run3_multiturn.yaml
sed -i 's/num_epochs: 3/num_epochs: 2/' configs/runs/run3_multiturn.yaml
run_train "configs/runs/run3_multiturn.yaml" "run3_multiturn"
cp -r checkpoints/final checkpoints/run3_multiturn
run_eval "checkpoints/run3_multiturn" "run3_multiturn"

# ── Run 4: trajectory credit ablation, 2 epochs ───────────────────────────────
cp configs/train.yaml configs/runs/run4_traj_credit.yaml
sed -i 's/credit_type: "step"/credit_type: "trajectory"/' configs/runs/run4_traj_credit.yaml
sed -i 's/num_epochs: 3/num_epochs: 2/' configs/runs/run4_traj_credit.yaml
run_train "configs/runs/run4_traj_credit.yaml" "run4_traj_credit"
cp -r checkpoints/final checkpoints/run4_traj_credit
run_eval "checkpoints/run4_traj_credit" "run4_traj_credit"

# ── Run 5: introductory + interview, 2 epochs (generalization) ───────────────
cp configs/train.yaml configs/runs/run5_interview.yaml
sed -i "s/difficulties: \[\"introductory\"\]/difficulties: [\"introductory\", \"interview\"]/" configs/runs/run5_interview.yaml
sed -i 's/num_epochs: 3/num_epochs: 2/' configs/runs/run5_interview.yaml
run_train "configs/runs/run5_interview.yaml" "run5_interview"
cp -r checkpoints/final checkpoints/run5_interview
run_eval "checkpoints/run5_interview" "run5_interview"

log "=== ALL RUNS COMPLETE ==="
log "Results summary:"
for f in results/apps_eval_*.json; do
    python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
s = d['summary']
print(f\"  {s['tag']}: {s['pass_at_1']:.1%} ({s['solved']}/{s['total']})\")
" 2>/dev/null
done
