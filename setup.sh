#!/bin/bash
set -e

echo "=== RLEF-Code cloud setup ==="

echo "[1/4] GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[2/4] Installing Poetry..."
pip install poetry
poetry config virtualenvs.in-project true

echo "[3/4] Installing dependencies..."
poetry install

echo "[4/4] Smoke test..."
poetry run python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  {i}:', torch.cuda.get_device_name(i))
"

echo "=== Done. Run wandb login and huggingface-cli login next. ==="
