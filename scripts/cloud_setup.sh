#!/bin/bash
# Full setup from scratch on a cloud GPU machine
# Usage: bash scripts/cloud_setup.sh
set -e

echo "=== RLEF-Code cloud setup ==="

echo "[1/5] GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[2/5] Installing Poetry..."
pip install poetry
poetry config virtualenvs.in-project true

echo "[3/5] Installing dependencies..."
poetry install

echo "[4/5] Downloading APPS dataset..."
bash scripts/download_apps.sh

echo "[5/5] Smoke test..."
poetry run python -c "
import torch
from dotenv import load_dotenv
load_dotenv()
print('CUDA:', torch.cuda.is_available())
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  {i}:', torch.cuda.get_device_name(i))
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
echo "  bash scripts/run_eval.sh baseline"
echo "  bash scripts/run_train.sh"
