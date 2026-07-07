#!/usr/bin/env bash
# ==============================================================================
# scripts/run_train.sh
# Dual-GPU GRPO Training Loop Automation Driver for RLEF-Code
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "======================================================================"
echo "🚀 Initiating GRPO Training Loop across 2x RTX 4090 GPUs"
echo "   Config Target: configs/train.yaml"
echo "======================================================================"

# Clean up any lingering bytecode files to isolate runtime state
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

if [ -f "accelerate_config.yaml" ]; then
    echo "⚙️ Found accelerate_config.yaml. Launching distributed 2x GPU environment..."
    poetry run accelerate launch \
        --config_file accelerate_config.yaml \
        src/rlef/train.py \
        --config configs/train.yaml
else
    echo "ℹ️ No explicit accelerate config found. Launching via default multi-device setup..."
    poetry run python -m rlef.train --config configs/train.yaml
fi

echo "======================================================================"
echo "🏁 Training Run Complete."
echo "======================================================================"
