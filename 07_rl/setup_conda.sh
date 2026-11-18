#!/usr/bin/env bash
# Create or update the conda environment for this project, then verify it.
# Works on Linux and macOS.
#
# Usage: bash setup_conda.sh          default PyTorch build (CUDA build on Linux)
#        bash setup_conda.sh --cpu    CPU-only PyTorch (smaller download, no GPU needed)
set -euo pipefail

cd "$(dirname "$0")"
ENV_NAME="rl-guidance"
CPU_ONLY=0
if [ "${1:-}" = "--cpu" ]; then
    CPU_ONLY=1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda를 찾을 수 없습니다. Miniconda 또는 Miniforge를 먼저 설치하세요." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[1/4] 기존 환경 '$ENV_NAME'을 갱신합니다."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "[1/4] 환경 '$ENV_NAME'을 만듭니다."
    conda env create -f environment.yml
fi

echo "[2/4] PyTorch와 강화학습 패키지를 설치합니다."
if [ "$CPU_ONLY" = "1" ]; then
    conda run -n "$ENV_NAME" python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
conda run -n "$ENV_NAME" python -m pip install gymnasium stable-baselines3

echo "[3/4] guidance 패키지를 설치합니다."
conda run -n "$ENV_NAME" python -m pip install -e .

echo "[4/4] 설치를 확인합니다."
conda run -n "$ENV_NAME" python -c "
import numpy, matplotlib, gymnasium, stable_baselines3, torch, guidance
print('numpy', numpy.__version__)
print('gymnasium', gymnasium.__version__)
print('stable-baselines3', stable_baselines3.__version__)
print('torch', torch.__version__, '| CUDA 사용 가능:', torch.cuda.is_available())
"
conda run -n "$ENV_NAME" python -m pytest -q

echo
echo "완료. 사용하려면: conda activate $ENV_NAME"
