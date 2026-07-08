#!/bin/bash
set -e

echo "=== RLEF-Code Cloud Initialization ==="

echo "[1/5] Hardware Verification..."
# Verify GPU allocation and VRAM before spinning up vLLM
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[2/5] Environment & Dependency Setup..."
# Install and configure Poetry strictly to the local project
pip install poetry
poetry config virtualenvs.in-project true

# Clean up standard cloud image conflicts if they exist (prevents silent CUDA clashes)
sudo pip uninstall xgboost transformer_engine flash_attn pynvml opencv-python-headless -y || true

# Install the locked environment exactly as defined in pyproject.toml
echo "Installing locked dependencies..."
poetry install

echo "[3/5] Fetching & Preparing Datasets..."
# Download the raw APPS dataset
bash scripts/download_apps.sh

# Pre-bake the generic dataset and physically carve the Curriculum splits for Run 7
echo "Generating curriculum splits..."
poetry run python src/rlef/prepare_openrlhf_data.py
poetry run python src/rlef/split_dataset.py

echo "[4/5] Authentication..."
# Authenticate telemetry and model hub
echo "Weights & Biases Login:"
poetry run wandb login

echo "HuggingFace Hub Login:"
HF_HUB_ENABLE_HF_TRANSFER=1 poetry run huggingface-cli login
#hf auth login

echo "[5/5] Running Pre-Flight Validation Suite..."
# Run the pure logic unit tests to verify prompt formats and parsers
poetry run pytest tests/ -m "unit"

echo "======================================================="
echo "✅ SETUP COMPLETE. THE ENVIRONMENT IS PRIMED."
echo "======================================================="
echo ""
echo "To launch the master ablation pipeline securely in the background, run:"
echo "tmux new-session -d -s rlef 'poetry run python run_pipeline.py' && tmux attach -t rlef"
