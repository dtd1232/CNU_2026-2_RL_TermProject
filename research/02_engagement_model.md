# 02. 2D 교전 모델

교과서 수준의 평면 질점 모델이다. 실제 체계의 제원은 쓰지 않는다.

## 1. 운동 방정식

요격탄(M)과 표적(T) 모두 속력이 일정하고, 속도에 수직인 횡가속도로 방향만 바꾼다.

$$\dot x_i = V_i \cos\gamma_i, \qquad \dot y_i = V_i \sin\gamma_i, \qquad \dot\gamma_i = \frac{a_i}{V_i} \qquad (i = M, T)$$

- $\gamma_i$: 비행 경로각
- $a_M$: 요격탄 횡가속도. 정책이 정하는 값이고 $|a_M| \le a_{max}$로 제한한다
- $a_T$: 표적 횡가속도. 시나리오가 정한다

## 2. 상대 운동

$$R = \sqrt{(x_T - x_M)^2 + (y_T - y_M)^2}, \qquad \lambda = \operatorname{atan2}(y_T - y_M, \; x_T - x_M)$$

$$\dot R = V_T \cos(\gamma_T - \lambda) - V_M \cos(\gamma_M - \lambda)$$

$$R \, \dot\lambda = V_T \sin(\gamma_T - \lambda) - V_M \sin(\gamma_M - \lambda)$$

접근 속도는 $V_c = -\dot R$이다. $\dot\lambda$는 수치 미분하지 않고 위 식으로 계산하면 잡음이 없다. 직교 좌표로 쓰면 상대 위치 $\mathbf r$, 상대 속도 $\mathbf v$에 대해 $\dot\lambda = (r_x v_y - r_y v_x) / R^2$이다.

## 3. 선택 요소: 1차 유도 지연

$$\dot a_M = \frac{a_c - a_M}{\tau}$$

명령 $a_c$와 실제 가속 $a_M$ 사이에 시정수 $\tau$의 지연을 둔다. PN의 한계를 드러내는 대표적인 방법이다(01 문서 4절). 넣을지는 plan 단계에서 정한다.

## 4. 표적 기동 유형

| 유형 | $a_T$ | 용도 |
|---|---|---|
| 비기동 | 0 | PN이 최적인 기준 사례. 구현 검증용 |
| 일정 기동 | $\pm n_T$ | PN 요구 가속도가 끝에서 커지는 사례 |
| bang-bang | $\pm n_T$, 임의 시각에 부호 전환 | 선행 연구(Gaudet 2020)에서 쓴 회피 기동 |
| 위빙 | $n_T \sin(\omega t + \phi)$ | 지연이 있을 때 빗나감 거리가 커진다 |

## 5. 수치 적분과 빗나감 거리

- **적분.** 고정 스텝 RK4나 오일러. 종말 구간은 상대 속도가 커서 스텝당 이동 거리가 크다. 접근 속도 1000 m/s에 스텝 0.01초면 한 스텝에 10 m를 지나므로, 스텝 끝의 거리만 보면 빗나감 거리가 부정확하다.
- **최근접점 보간.** 한 스텝 안에서 상대 운동을 직선으로 보고 최근접 시각을 구한다.

$$t^* = \operatorname{clip}\left(-\frac{\mathbf r \cdot \mathbf v}{\lVert \mathbf v \rVert^2}, \; 0, \; \Delta t\right), \qquad \text{miss} = \lVert \mathbf r + \mathbf v \, t^* \rVert$$

- **종료 조건.** $\mathbf r \cdot \mathbf v \ge 0$이 되면(거리가 다시 늘어나기 시작하면) 종료한다. 시간 초과도 종료로 처리한다.
- **유도 주기와 적분 주기 분리.** 정책은 10~50 Hz로 명령을 내고, 그 사이를 더 작은 스텝으로 적분하는 방식이 흔하다.

## 6. 관측 후보

| 후보 | 구성 | 성격 |
|---|---|---|
| A. 탐색기 수준 | $\dot\lambda$, $V_c$ | PN과 같은 정보. "PN 구조를 찾는가" 질문에 가장 깨끗하다 |
| B. 확장 | A + $R$, 리드각 $\gamma_M - \lambda$ | 종말 구간과 포화 상황을 구분할 수 있다 |
| C. 이력 포함 | B + 직전 관측, 직전 행동 | $\dot\lambda$의 변화로 표적 가속을 간접 추정할 수 있다 |

기억이 없는 정책이 $\dot\lambda$와 $V_c$만 보면 표적 가속도를 알 수 없다. 그 정보만으로는 PN과 비슷한 것 이상을 하기 어렵다. PN을 넘어서려면 C처럼 이력을 주거나 순환 신경망을 써야 한다. Gaudet(2020)이 순환 정책을 쓴 이유가 이것이다.

## 7. 수치 예시

공개 교재 예제 수준의 값이다. 확정은 plan 단계에서 한다.

| 항목 | 예시 값 | 근거 |
|---|---|---|
| 요격탄 속력 | 약 900 m/s | Zarchan 교재 예제 계열(3000 ft/s) |
| 표적 속력 | 약 300 m/s | 같은 계열(1000 ft/s) |
| 초기 거리 | 약 12 km | 같은 계열(40000 ft) |
| 표적 기동 | 3 g | 같은 계열 |
| 초기 조준 오차 | ±20° | 같은 계열 |
| 요격탄 가속도 한계 | 표적 기동의 2~5배 범위에서 변화 | 3배 근처가 PN이 포화되는 경계 |

## 참고 문헌

- P. Zarchan, *Tactical and Strategic Missile Guidance*, AIAA. 2D 교전 시뮬레이션, 유도 지연, 기동 표적 예제.
