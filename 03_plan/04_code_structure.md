# 04. 코드 구조

코드는 모두 `07_rl/` 아래에 둔다. `08_unreal/`은 건드리지 않는다.

## 1. 폴더

```
07_rl/
├── requirements.txt
├── guidance/              패키지
│   ├── dynamics.py        운동 방정식, RK4, 최근접점 계산
│   ├── scenarios.py       초기 조건 추출, 표적 기동
│   ├── env.py             Gymnasium 환경 (관측, 행동, 보상 선택)
│   ├── rewards.py         보상 4가지
│   ├── pn.py              PN, APN
│   ├── discretize.py      Q-learning용 상태·행동 이산화
│   ├── qlearning.py       표 형태 Q-learning
│   ├── evaluate.py        몬테카를로 평가, 지표
│   └── export.py          궤적 CSV 내보내기
├── scripts/
│   ├── calibrate_pn.py    E1
│   ├── train_q.py         E2, E5
│   ├── train_ppo.py       E4, E5
│   ├── evaluate_all.py    E6
│   └── plot_*.py          그림
├── configs/               실험 설정 (YAML)
├── tests/                 E0 검증 항목
└── results/
    ├── figures/           그림 (커밋)
    ├── tables/            지표 표 CSV (커밋)
    ├── models/            Q-table, PPO 모델 (최종본만 커밋)
    └── runs/              학습 로그 (커밋 안 함)
```

## 2. 의존성

- Python 3.11 이상
- numpy, matplotlib, pyyaml
- gymnasium, stable-baselines3 (PyTorch 포함)
- pytest

Q-learning과 PN은 numpy만으로 돈다. PPO 단계에서만 SB3가 필요하다.

## 3. 설계 원칙

- **환경 하나, 설정으로 전환.** 관측 종류, 보상 종류, 시나리오, $a_{max}$, $\tau$를 생성자 인자로 받는다. Q-learning과 PPO가 같은 환경을 쓴다.
- **정책 인터페이스 통일.** PN, Q-table, PPO 모두 "관측을 받아 행동을 내는 함수"로 감싸서 같은 평가 코드에 넣는다.
- **동역학과 Gymnasium 분리.** `dynamics.py`는 Gymnasium을 모른다. 테스트와 궤적 내보내기에 그대로 쓴다.
- **결과 재현.** 각 실행은 설정 사본과 시드를 결과 폴더에 함께 남긴다.

## 4. 궤적 CSV 형식

언리얼 쪽에서 읽을 파일이다.

| 열 | 단위 | 내용 |
|---|---|---|
| `t` | s | 시각 |
| `mx`, `my` | m | 요격탄 위치 |
| `tx`, `ty` | m | 표적 위치 |
| `m_heading`, `t_heading` | deg | 진행 방향 |
| `m_accel` | m/s² | 요격탄 횡가속도 |

- 고정 시간 간격(0.05초)으로 저장한다.
- 파일 첫 줄에 정책 이름, 시나리오, 빗나감 거리를 주석으로 적는다.
- 언리얼의 단위(cm)와 좌표계로 바꾸는 것은 언리얼 쪽에서 한다. 형식을 바꿔야 하면 implementation 단계에서 조정한다.
