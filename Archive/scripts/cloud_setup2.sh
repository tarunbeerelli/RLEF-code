#!/bin/bash
set -e

echo "=== RLEF-Code cloud setup ==="

echo "[1/6] GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[2/6] Installing Poetry..."
pip install poetry
poetry config virtualenvs.in-project true

echo "[3/6] Cleaning conflicting system packages & setting up base framework..."
sudo pip uninstall xgboost transformer_engine flash_attn pynvml opencv-python-headless -y

# Force the specific openrlhf/vllm combo into the virtualenv scope
poetry run pip install "openrlhf[vllm]>=0.6.3,<0.7.0"

echo "[4/6] Installing dependencies..."
poetry install --without prod
export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH

echo "[5/6] Ensuring exact runtime layer constraint..."
poetry run pip install "numpy<2.0.0"

echo "[6/6] Compiling Protobuf Services for gRPC inside virtualenv..."
poetry run python3 -m grpc_tools.protoc \
  -I=src/rlef \
  --python_out=src/rlef \
  --grpc_python_out=src/rlef \
  src/rlef/reward.proto

echo "=== STEP 0.0: RUNNING VERIFICATION SWEEP & DATA PREP ==="
bash scripts/download_apps.sh
poetry run python3 clean_and_optimize_dataset.py
poetry run python3 src/rlef/prepare_openrlhf_data.py

poetry run pytest tests/

echo "=== LAUNCHING BACKGROUND SERVICES ==="
# Launch reward server safely in background so the script doesn't hang
poetry run python3 src/rlef/grpc_reward_server.py > reward_server.log 2>&1 &
echo "Waiting for gRPC server to bind to port 50051..."
sleep 3

# Logins
poetry run wandb login
# If hf requires interaction, ensure your token is ready or exported as an ENV var
hf_transfer_enabled=1 poetry run huggingface-cli login

# Run the final launcher (Ensure this file uses poetry run python3 inside it!)
bash scripts/run_grpo_docker.sh
