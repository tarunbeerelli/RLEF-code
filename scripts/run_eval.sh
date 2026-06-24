#!/bin/bash
# Run evaluation
# Usage: bash scripts/run_eval.sh baseline
#        bash scripts/run_eval.sh trained
set -e

TAG=${1:-baseline}
CHECKPOINT=${2:-none}

echo "Running eval: tag=$TAG checkpoint=$CHECKPOINT"

nohup poetry run python -m rlef.eval \
  --checkpoint "$CHECKPOINT" \
  --tag "$TAG" \
  --benchmark apps \
  --max_examples 200 \
  --difficulties introductory interview \
  --data_dir data/raw/APPS \
  --device cuda \
  > "logs/eval_${TAG}.log" 2>&1 &

echo "PID: $! — tail -f logs/eval_${TAG}.log"
