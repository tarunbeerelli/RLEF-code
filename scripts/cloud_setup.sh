#!/bin/bash
# Full setup from scratch on a cloud GPU machine
# Usage: bash scripts/cloud_setup.sh
set -e

echo "=== RLEF-Code cloud setup ==="

echo "[1/6] GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[2/6] Installing Poetry..."
pip install --upgrade pip
pip install poetry
poetry config virtualenvs.in-project true

echo "[3/6] Pre-installing CUDA 12.4 Torch binaries inside virtualenv..."
# This ensures Poetry doesn't try to pull a mismatched CPU variant
poetry run pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "[4/6] Installing dependencies..."
# Installs remaining locks from pyproject.toml without breaking our torch base
poetry install

echo "[5/6] Overlaying optimized LLM stack & NumPy fix..."
# Force pinning the heavy runtimes to guarantee architectural compatibility
poetry run pip install transformers==4.46.0 peft==0.13.2 trl==0.14.0 vllm==0.6.3
poetry run pip install "numpy<2.0.0" soxr --force-reinstall

echo "[6/6] Compiling Protobuf Services for gRPC..."
poetry run python3 -m grpc_tools.protoc -I. --python_out=src/rlef --grpc_python_out=src/rlef ./reward.proto

echo "=== STEP 0.0: RUNNING VERIFICATION SWEEP & DIRECTORY PURGE ==="
poetry run python clean_and_optimize_dataset.py

echo "=== SMOKE TEST ==="
poetry run python -c "
import torch
from dotenv import load_dotenv
load_dotenv()
print('CUDA:', torch.cuda.is_available())
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  {i}:', torch.cuda.get_device_name(i))
try:
    from rlef import reward_pb2
    print('gRPC Protobuf modules: Loaded Successfully')
except Exception as e:
    print('gRPC Protobuf modules: Failed to load ->', e)
from rlef.data import load_apps_split
problems = load_apps_split('data/raw/APPS', split='train', difficulties=['introductory'])
print(f'APPS train introductory: {len(problems)} problems')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  poetry run wandb login"
echo "  poetry run huggingface-cli login"
echo "  echo 'E2B_API_KEY=your_key' > .env"
echo "  python3 src/rlef/grpc_reward_server.py  (Run this in a separate window first)"
echo "  bash scripts/run_train.sh"
