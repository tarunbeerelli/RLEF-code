#!/bin/bash
# Launch GRPO training with DDP across 2 GPUs
# Usage: bash scripts/run_train.sh [config]
set -e

CONFIG=${1:-configs/train.yaml}
mkdir -p logs checkpoints

echo "Starting training with config: $CONFIG"

nohup poetry run accelerate launch \
  --config_file accelerate_config.yaml \
  src/rlef/train.py \
  --config "$CONFIG" \
  > logs/train.log 2>&1 &

echo "PID: $! — tail -f logs/train.log"
