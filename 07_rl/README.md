# 07_rl

강화학습 개발 폴더. 교전 환경, PN 베이스라인, Q-learning, PPO 코드와 실험 결과를 둔다.

## 환경 설정

```bash
bash setup_conda.sh
```

```bash
conda activate rl-guidance
```

자세한 내용은 `02_spec/01_dev_environment.md` 참고.

## 실행

모든 명령은 이 폴더(`07_rl/`)에서 실행한다.

| 명령 | 내용 |
|---|---|
| `python -m pytest -q` | 환경 검증 (E0) |
| `python scripts/calibrate_pn.py` | PN 난이도 보정 (E1) |
| `python scripts/plot_pn_trajectories.py` | PN 궤적 그림 |

## 구조

| 경로 | 내용 |
|---|---|
| `guidance/dynamics.py` | 교전 시뮬레이터, 설정 프리셋 |
| `guidance/scenarios.py` | 초기 조건, 표적 기동 |
| `guidance/pn.py` | PN, APN |
| `guidance/env.py` | Gymnasium 환경 |
| `guidance/rewards.py` | 보상 4가지 |
| `guidance/evaluate.py` | 몬테카를로 평가 |
| `scripts/` | 실험 스크립트 |
| `tests/` | 검증 테스트 |
| `results/` | 그림, 표, 모델 |
