#!/usr/bin/env bash
# ==============================================================================
# scripts/run_eval.sh
# Dual-4090 Isolated Device Evaluation Driver for RLEF-Code
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Runtime Configuration Parameters
MAX_EXAMPLES=600        # Balanced stratified test slice target
MAX_TURNS=5            # Multi-turn test-time compute interaction budget
DEVICE="cuda:0"         # Isolate completely to the first RTX 4090 device

echo "======================================================================"
echo "📊 Launching Multi-Turn Evaluation Suite on Isolated RTX 4090"
echo "   Target Dataset Slice: Balanced APPS Test Split"
echo "   Target Execution Device: ${DEVICE}"
echo "   Budget Ceiling: ${MAX_TURNS} Turns per problem instance"
echo "======================================================================"

# ── Phase 1: Establish Base Reference Baseline ──────────────────────────────
echo -e "\n[Phase 1/2] Evaluating Untrained Base Reference Model..."
poetry run python -m rlef.eval \
    --checkpoint none \
    --tag baseline \
    --max_examples "${MAX_EXAMPLES}" \
    --max_turns "${MAX_TURNS}" \
    --device "${DEVICE}"

# ── Phase 2: Evaluate Reinforced Checkpoint ─────────────────────────────────
TRAINED_CHECKPOINT="./checkpoints/final"

if [ -d "${TRAINED_CHECKPOINT}" ]; then
    echo -e "\n[Phase 2/2] Evaluating Reinforcement-Trained Checkpoint..."
    poetry run python -m rlef.eval \
        --checkpoint "${TRAINED_CHECKPOINT}" \
        --tag trained \
        --max_examples "${MAX_EXAMPLES}" \
        --max_turns "${MAX_TURNS}" \
        --device "${DEVICE}"
else
    echo -e "\n dryness-check: Checkpoint missing at: ${TRAINED_CHECKPOINT}. Skipping Phase 2."
fi

echo "======================================================================"
echo "🏁 Evaluation Sweeps Complete. Trajectory metrics saved to results/"
echo "======================================================================"
