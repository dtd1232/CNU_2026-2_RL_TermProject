# 03. RL 유도 선행 연구

원문 확인 수준을 표시했다. "원문"은 본문을 읽은 것이고, "초록"은 초록과 서지 정보만 확인한 것이다.

## 1. 주요 연구

| 연구 | 알고리즘 | 관측과 행동 | 비교 대상 | 결과 요지 | 확인 |
|---|---|---|---|---|---|
| Gaudet & Furfaro, 2012 | RL (세부 미확인) | 호밍 단계 유도 | PN, 개선형 PN | 기체 동역학, 센서·구동기 잡음과 지연이 있을 때 PN보다 낫다고 주장 | 초록 |
| Gaudet, Furfaro, Linares, 2020 | PPO + GRU 순환 정책 (메타 RL) | 시선각과 그 변화율만 관측 | 표적 가속도를 완벽히 아는 augmented ZEM | 거리 정보 없이 동등 이상의 성능 | 원문 |
| Hong, Kim, Park, 2020 | DDPG | 요격탄 속도와 시선각 변화율 | PN | 제어기 모델 불확실성에 PN보다 강건 | 초록 |
| He, Shin, Tsourdos, 2021 | DDPG | 명령을 직접 학습하는 방식과, PN 형태를 고정하고 이득만 학습하는 방식 | PN | 높은 요격 정확도. 보상은 ZEM, 에너지, 요격 시간을 조합 | 초록 |
| Tang 외, 2022 | DQN | 이산 행동 | TPN | 시선각 변화율과 ZEM 기반 보상. TPN보다 낫다고 보고 | 검색 요약 |
| Porter, 2024 (석사 논문) | PPO + 보상 셰이핑 | 저충실도 모델 | 전통 유도 법칙 | 회피 표적에는 빗나감 거리 감소, 비기동 표적에는 격추율 저하, 횡가속도 명령은 더 큼 | 초록 |

## 2. Gaudet 2020 세부 (원문 확인)

이 프로젝트와 가장 가까운 연구다.

- **시나리오.** 대기권 밖 종말 요격. 3D.
- **표적 기동.** 속도에 수직인 bang-bang 기동. 최대 5 g, 시작 시각·지속 시간·전환 시각을 무작위로 정한다.
- **정책.** PPO. 정책과 가치 함수 모두 GRU 순환 층을 쓴다. 순환 층의 은닉 상태가 표적 기동에 따라 다르게 변하면서 기동에 적응한다. 가속도를 추정하는 필터 없이 적응한다는 점이 핵심 주장이다.
- **보상.** 종료 시 빗나감 거리가 50 cm 미만이면 종단 보상을 준다. 무작위 탐색으로는 이 보상을 거의 받지 못하므로 매 스텝 셰이핑 보상을 더한다. 셰이핑은 시선각 오차와 시선각 변화율이 작을수록 큰 가우시안 형태다.
- **할인율.** 셰이핑 보상에는 0.90, 종단 보상에는 0.995를 따로 적용한다.
- **저자들의 관찰.** 성공적인 명중에 주는 종단 보상이 좋은 성능에 결정적이었고, 변화율 항을 넣는 것도 중요했다.
- **일반화.** 학습 때 보지 못한 배럴 롤 기동에서는 RL과 비교 대상 모두 성능이 떨어졌다.

## 3. 선행 연구에서 얻는 것

1. **PN과 같은 정보만 줘도 학습이 된다.** 시선각 변화율 중심의 관측으로 충분하다는 사례가 여럿 있다(Gaudet 2020, Hong 2020).
2. **RL의 장점은 이상 조건이 아닌 곳에서 나온다.** 표적 기동, 지연, 잡음, 모델 불확실성이 있을 때 PN보다 낫다고 보고한다. 이상 조건에서는 PN과 비슷해지는 것이 기대 결과다.
3. **비용도 보고해야 한다.** Porter(2024)는 RL이 더 큰 가속도 명령을 쓰고 비기동 표적에서는 오히려 못했다고 보고했다. 빗나감 거리만이 아니라 제어 노력($\int a^2 dt$)도 함께 비교해야 공정하다.
4. **희소 보상만으로는 어렵다.** 거의 모든 연구가 셰이핑을 쓴다. 희소 대 셰이핑 ablation은 이 분야에서 실제로 의미 있는 질문이다.
5. **중간 형태가 있다.** PN 형태는 고정하고 이득 $N$만 RL로 정하는 방식(He 2021, 시변 PN 이득 연구)이다. 명령을 직접 학습하는 것이 실패할 때의 대안이 된다.
6. **대부분 연속 행동의 심층 RL을 쓴다.** 표 형태 Q-learning으로 푼 사례는 찾지 못했다. 수업 내용으로 푸는 것 자체가 이 프로젝트의 차별점이 될 수 있다. 다만 못 찾았다는 것이지 없다는 뜻은 아니다.

## 4. 이 프로젝트의 위치

선행 연구는 "RL이 PN보다 낫다"를 보이는 데 집중한다. 학습된 정책이 PN과 같은 구조인지를 직접 분석한 사례는 조사 범위에서 찾지 못했다. 2D 저차원 문제라서 정책을 그림으로 그려 볼 수 있다는 점을 활용하면 발표의 중심 내용이 된다.

## 참고 문헌

- B. Gaudet, R. Furfaro, "Missile Homing-Phase Guidance Law Design Using Reinforcement Learning," AIAA GNC Conference, 2012. doi:10.2514/6.2012-4470
- B. Gaudet, R. Furfaro, R. Linares, "Reinforcement Learning for Angle-Only Intercept Guidance of Maneuvering Targets," *Aerospace Science and Technology*, vol. 99, 105746, 2020. doi:10.1016/j.ast.2020.105746. 사전 공개본: https://arxiv.org/abs/1906.02113
- D. Hong, M. Kim, S. Park, "Study on Reinforcement Learning-Based Missile Guidance Law," *Applied Sciences*, 10(18), 6567, 2020. doi:10.3390/app10186567
- S. He, H.-S. Shin, A. Tsourdos, "Computational Missile Guidance: A Deep Reinforcement Learning Approach," *Journal of Aerospace Information Systems*, 18(8), pp. 571–582, 2021. doi:10.2514/1.I010970
- Tang 외, DQN 기반 대기권 밖 호밍 유도, *International Journal of Aerospace Engineering*, 2022. 서지 정보는 검색 요약으로만 확인했다. 인용 전에 원문 확인이 필요하다.
- D. M. Porter, "The Application of Reinforcement Learning for Interceptor Guidance," 석사 논문, Virginia Tech, 2024. https://vtechworks.lib.vt.edu/items/909cd86b-37d9-418f-a700-dd72026a146b
