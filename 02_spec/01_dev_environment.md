# 01. 개발 환경

## 1. 실행 환경

| 구분 | 내용 |
|---|---|
| 학습과 실험 | 클라우드 리눅스 컨테이너. CPU 4코어, 메모리 15 GB, GPU 없음 |
| 로컬 확인 | macOS, conda 환경 |
| 언리얼 재생 | 로컬에서 직접 관리 (`08_unreal/`) |

문제 크기가 작아서 GPU 없이 진행한다. 관측은 2~6차원이고 신경망은 은닉층 64 × 2 수준이다.

## 2. 소프트웨어

| 패키지 | 용도 | 클라우드에서 쓴 버전 |
|---|---|---|
| Python | | 3.13 (conda 환경은 3.11로 지정) |
| numpy | 시뮬레이터, Q-learning | 2.5 |
| matplotlib | 그림 | 3.11 |
| gymnasium | 환경 인터페이스 | 1.4 |
| pytest | 검증 테스트 | 9.2 |
| stable-baselines3 | PPO | PPO 단계에서 설치 예정 (pip 해석 결과 2.9) |
| PyTorch | SB3가 사용 | PPO 단계에서 설치 예정 (pip 해석 결과 2.15) |

conda 환경 파일은 버전을 고정하지 않고 최신을 받는다. 클라우드와 버전이 달라 문제가 생기면 위 표의 버전으로 고정한다.

## 3. conda 환경 설정

환경 이름은 `rl-guidance`다. 저장소의 `07_rl/` 폴더에서 실행한다.

```bash
bash setup_conda.sh
```

스크립트가 하는 일은 세 가지다.

1. `environment.yml`로 환경을 만든다. 이미 있으면 갱신한다.
2. `guidance` 패키지를 편집 가능 모드로 설치한다. 코드를 고치면 바로 반영된다.
3. 패키지 버전을 출력하고 검증 테스트를 돌린다.

끝나면 환경을 켠다.

```bash
conda activate rl-guidance
```

**확인 상태.** 클라우드 컨테이너에는 conda가 없어서 이 스크립트를 실제로 돌려 보지 못했다. 문법 검사만 했다. 맥에서 처음 실행할 때 오류가 나면 내용을 알려 주면 고친다.

**PyTorch 설치 시간.** stable-baselines3가 PyTorch를 함께 받는다. 용량이 커서 몇 분 걸린다.

## 4. 실행 방법

`07_rl/` 폴더에서 실행한다.

```bash
python -m pytest -q
```

```bash
python scripts/calibrate_pn.py
```

스크립트 목록은 `07_rl/README.md`에 있다.
