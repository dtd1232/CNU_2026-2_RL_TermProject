#!/usr/bin/env bash
# Create or update the conda environment for this project, then verify it.
# Usage: bash setup_conda.sh
set -euo pipefail

cd "$(dirname "$0")"
ENV_NAME="rl-guidance"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda를 찾을 수 없습니다. Miniconda 또는 Miniforge를 먼저 설치하세요." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[1/3] 기존 환경 '$ENV_NAME'을 갱신합니다."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "[1/3] 환경 '$ENV_NAME'을 만듭니다."
    conda env create -f environment.yml
fi

echo "[2/3] guidance 패키지를 설치합니다."
conda run -n "$ENV_NAME" python -m pip install -e .

echo "[3/3] 설치를 확인합니다."
conda run -n "$ENV_NAME" python -c "
import numpy, matplotlib, gymnasium, stable_baselines3, torch, guidance
print('numpy', numpy.__version__)
print('gymnasium', gymnasium.__version__)
print('stable-baselines3', stable_baselines3.__version__)
print('torch', torch.__version__)
"
conda run -n "$ENV_NAME" python -m pytest -q

echo
echo "완료. 사용하려면: conda activate $ENV_NAME"
