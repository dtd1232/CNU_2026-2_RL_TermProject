# 04. RL 방법: Q-learning과 PPO

## 1. MDP 구성

| 요소 | 내용 |
|---|---|
| 상태 | 교전 기하. 관측 후보는 02 문서 6절 |
| 행동 | 요격탄 횡가속도 명령 |
| 전이 | 02 문서의 운동 방정식. 결정적이다. 무작위성은 초기 조건과 표적 기동에서 온다 |
| 보상 | 05 문서 |
| 종료 | 최근접점 통과 또는 시간 초과 |

에피소드 길이는 유도 주기와 교전 시간으로 정해진다. 10초 교전에 20 Hz면 200스텝이다.

## 2. 표 형태 Q-learning

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

**상태 이산화.**

- 1차 후보는 $\dot\lambda$ 하나, 또는 $V_c \dot\lambda$ 하나다. PN이 바로 이 값의 함수이므로, 표가 1차원이어도 PN과 같은 정책을 표현할 수 있다.
- 2차 후보는 여기에 $t_{go}$ 또는 $R$ 구간을 더한 2차원이다.
- $\dot\lambda$는 교전 대부분에서 0 근처에 몰려 있다가 종말에 커진다. 균등 구간보다 0 근처를 촘촘히 나누는 비균등 구간(로그 간격 등)이 맞다.
- 구간 밖의 값은 양 끝 구간으로 잘라 넣는다.

**행동 이산화.** $\{-a_{max}, \dots, 0, \dots, +a_{max}\}$를 5~9단계로 나눈다. 3단계(bang-bang)도 비교 대상이 된다.

**표 크기.** $\dot\lambda$ 21구간 × 행동 7개면 147칸이다. 차원을 하나 더해 10구간을 곱해도 1470칸이라 학습이 빠르다.

**주의할 점.**

- 이산화하면 같은 구간 안의 서로 다른 상태가 구분되지 않아 마르코프 성질이 깨진다. 수렴이 보장되지 않고 정책이 떨릴 수 있다.
- 탐색은 $\epsilon$-greedy에 $\epsilon$ 감소를 쓴다. 희소 보상에서는 낙관적 초기값도 도움이 된다.
- 종단 보상이 200스텝 앞까지 전파되어야 하므로 $\gamma$는 1에 가깝게 잡는다(0.99 이상).
- 비교용으로 SARSA나 n-step 변형을 붙이기 쉽다.

## 3. PPO

정책 경사 계열의 on-policy 알고리즘이다. 정책이 한 번에 너무 크게 바뀌지 않도록 목적 함수를 자른다.

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat A_t, \; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t \right) \right], \qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

- 연속 행동은 가우시안 정책으로 다룬다. 평균은 신경망이 내고 표준편차는 학습되는 매개변수다.
- 이점 추정에는 GAE를 쓴다.
- 평가할 때는 평균 행동(결정적 정책)을 쓴다.

**Stable-Baselines3에서 쓸 때.**

- 환경은 Gymnasium API로 만든다. `reset()`은 `(obs, info)`를, `step()`은 `(obs, reward, terminated, truncated, info)`를 돌려준다. 최근접점 통과는 `terminated`, 시간 초과는 `truncated`로 구분한다.
- 행동 공간은 `Box(-1, 1)`로 두고 환경 안에서 $a_{max}$를 곱한다.
- 관측은 크기 차이가 크다($R$은 수천 m, $\dot\lambda$는 0.01 rad/s 수준). 직접 정규화하거나 `VecNormalize`를 쓴다. 정규화 통계는 모델과 함께 저장해야 평가 때 같은 값을 쓸 수 있다.
- `check_env`로 환경 규격을 검증한다.
- 병렬 환경(`make_vec_env`)으로 표본 수집을 빠르게 한다.
- 순환 정책이 필요하면 `sb3-contrib`의 `RecurrentPPO`가 있다. 관측에 이력을 넣는 방식이 더 단순하다.

**Q-learning과 PPO 사이의 다리.** 이산 행동을 그대로 두고 신경망으로 Q를 근사하는 DQN이 중간 단계가 된다. 일정에 여유가 있을 때의 선택지다.

## 4. 학습된 정책이 PN 구조인지 확인하는 방법

1. **정책 그림.** Q-table에서는 구간별 greedy 행동을 $\dot\lambda$에 대해 그린다. PN과 같다면 0을 지나며 단조 증가하는 계단 모양이 되고 양 끝에서 포화된다. PPO에서는 $V_c \dot\lambda$에 대한 행동 산점도를 그린다.
2. **유효 항법 상수.** 교전 중 수집한 $(V_c \dot\lambda, \; a)$ 쌍에 원점을 지나는 직선 $a = \hat N \cdot V_c \dot\lambda$를 맞춘다. 포화 구간은 뺀다. $\hat N$이 3~5 근처이고 결정계수 $R^2$가 높으면 PN 구조라고 말할 수 있다.
3. **다른 입력에 대한 민감도.** $V_c \dot\lambda$를 고정하고 $R$이나 리드각을 바꿨을 때 행동이 얼마나 달라지는지 본다. 달라진다면 PN에 없는 구조를 배운 것이다. $t_{go}$에 따라 $\hat N$이 변하는지 보는 것이 대표적이다.
4. **궤적 비교.** 같은 초기 조건에서 PN과 RL의 궤적, 가속도 이력, $\dot\lambda$ 이력을 겹쳐 그린다.

## 5. 평가 방법

- 학습과 분리된 고정 초기 조건 집합으로 몬테카를로 평가를 한다(수백 회 이상).
- 지표는 빗나감 거리의 평균·중앙값·95 백분위, 명중률(빗나감 거리가 기준 이하인 비율), 제어 노력 $\int a^2 dt$, 포화 시간 비율이다.
- RL은 시드에 따라 결과가 크게 달라진다. 시드 3~5개의 평균과 편차를 함께 보고한다.
- PN은 $N'$ = 3, 4, 5를 모두 돌려 가장 좋은 값과 비교해야 공정하다.
- 학습 범위 밖의 초기 조건과 학습 때 없던 표적 기동으로 일반화를 본다.

## 참고 문헌

- R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2판, MIT Press, 2018.
- C. J. C. H. Watkins, P. Dayan, "Q-learning," *Machine Learning*, 8, 1992.
- J. Schulman 외, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
- J. Schulman 외, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," arXiv:1506.02438, 2015.
- A. Raffin 외, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *JMLR*, 22(268), 2021.
