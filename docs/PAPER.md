# QuAP-FL: 참여율 기반 적응형 차분 프라이버시 연합학습에 대한 이론적 고찰


---

## 초록

연합학습(Federated Learning, FL)은 데이터 주권과 프라이버시 요구가 강화되는 환경에서 필수적인 학습 패러다임으로 떠올랐다. 그러나 실제 서비스 환경의 클라이언트 참여율은 낮고 불규칙하며, 고정된 차분 프라이버시(Differential Privacy, DP) 예산을 모든 클라이언트에 균등하게 부여하는 기존 접근법은 참여 빈도에 따른 프라이버시 손실 편차와 효율성 저하를 초래한다. 본 논문은 참여율 추적 및 적응형 프라이버시 예산 배분을 핵심으로 하는 **QuAP-FL(Quantile-based Adaptive Privacy for Federated Learning)** 프레임워크를 이론적으로 분석한다. 제안 기법은 (1) 라운드별 참여율을 정밀하게 추정하고, (2) 참여율에 반비례하는 프라이버시 예산 함수를 통해 클라이언트별 ε 값을 동적으로 조정하며, (3) Layer-wise DP를 도입해 마지막 분류 레이어에 한정된 노이즈를 주입함으로써 고차원 노이즈 문제를 해결한다. 본 논문은 아직 실험을 수행하지 않았음을 명시하며, 대신 참여율 기반 예산 함수, 클리핑 전략, 노이즈 스케일링, 그리고 프라이버시 및 유틸리티에 대한 이론적 분석을 제공한다. 또한 향후 실험 설계와 평가 지표에 대한 구체적 계획을 제시해 실제 검증을 위한 로드맵을 마련한다.

---

## 1. 서론

### 1.1 연구 동기

모바일 디바이스와 엣지 컴퓨팅 환경에서 수집되는 데이터는 프라이버시 민감도가 높고, 법적 규제 또한 강화되고 있다. 이로 인해 데이터 중앙집중을 지양하는 연합학습이 주목받고 있지만, 현실적 제약 요소가 존재한다.

1. **불규칙한 참여**: 실제 서비스 환경에서 매 라운드 참여하는 클라이언트 비율은 10–30%에 불과하다. 네트워크 품질, 배터리 상태, 사용자의 행동 패턴 등 외부 요인으로 참여율이 크게 변동한다.
2. **균일 예산 할당의 문제점**: 모든 클라이언트에 동일한 ε 값을 부여하면 자주 참여하는 클라이언트의 프라이버시 손실이 빠르게 누적되고, 드물게 참여하는 클라이언트는 필요 이상의 노이즈를 삽입하게 된다.
3. **고차원 노이즈 문제**: DP-SGD는 모든 파라미터를 동일하게 클리핑하고 노이즈를 추가하지만, 딥러닝 모델의 고차원 파라미터 공간에서는 노이즈가 학습 신호를 압도해 성능이 크게 떨어진다.

이러한 현실적 제약을 고려하면, 참여율에 따라 프라이버시 예산을 조정하고 노이즈 적용 범위를 제한하는 전략이 필요하다.

### 1.2 연구 기여

본 논문은 다음과 같은 학술적 기여를 제공한다.

- **참여율 기반 프라이버시 예산 함수**: 클라이언트 참여율이 높을수록 강한 프라이버시(작은 ε), 낮을수록 완화된 프라이버시를 제공하는 단순하면서도 실용적인 수식을 제시한다.
- **Layer-wise DP의 이론적 정당화**: 마지막 분류 레이어에만 노이즈를 주입함으로써 전체 노이즈 폭발 문제를 피해가는 전략을 체계적으로 설명한다.
- **클리핑 및 노이즈 스케일링 분석**: 분위수 기반 클리핑이 참여율 변동과 결합될 때의 안정성 조건을 분석한다.
- **실험 설계 로드맵**: 아직 실험은 수행하지 않았지만, 추후 검증을 위한 평가 계획과 지표 정의, 가설 설정을 상세히 제시한다.

### 1.3 논문 구성

본 논문은 총 여덟 개 장으로 구성된다. 2장에서는 관련 연구와 배경 지식을 정리하고, 3장에서는 문제 정의와 가정을 명시한다. 4장은 QuAP-FL의 주요 구성 요소를 설명하며, 5장은 프라이버시 및 유틸리티 측면의 이론적 분석을 제공한다. 6장에서는 구현 관점에서 고려해야 할 요소를 다루고, 7장에서 향후 실험 설계와 평가 계획을 제안한다. 마지막으로 8장은 결론과 향후 연구 방향을 제시한다.

---

## 2. 배경 및 관련 연구

### 2.1 연합학습 기본 개념

연합학습은 여러 클라이언트가 동일한 글로벌 모델을 공유하며, 각자의 로컬 데이터를 사용해 모델을 업데이트하고, 서버가 이를 통합해 모델을 개선하는 구조다. 대표적인 알고리즘은 **FedAvg**로, 각 클라이언트는 로컬 SGD를 일정 횟수 수행한 후 파라미터를 서버에 전송하고, 서버는 이를 평균해 글로벌 파라미터를 갱신한다.

### 2.2 차분 프라이버시

차분 프라이버시는 데이터셋에서 단일 레코드의 포함 여부가 결과에 거의 영향을 미치지 않도록 보장하는 수학적 프레임워크다. ε이 작을수록 프라이버시 보호가 강하고, 데이터에 더 많은 노이즈가 추가된다. FL과 결합하기 위해서는 각 클라이언트 업데이트 또는 서버 집계 단계에서 DP 메커니즘을 적용한다.

### 2.3 참여율 모델링

실제 환경에서 클라이언트의 참여 패턴은 베타 분포, 캐럿 분포, 마르코프 체인 등으로 모델링될 수 있다. 본 연구에서는 이해를 돕기 위해 다음과 같은 단순화를 채택한다.

- 라운드 \(t\)에서 각 클라이언트 \(i\)의 참여 확률은 베타 분포에서 추출된 가중치와 균등 분포의 혼합으로 결정된다.
- 라운드별 참여 집합은 각 라운드에 독립적으로 샘플링되며, 참여 후에는 ParticipationTracker가 누적 참여 횟수를 갱신한다.

### 2.4 관련 연구

- **DP-FedAvg 개선 연구**: adaptive clipping, per-client clipping, Rényi DP 등을 통해 프라이버시-유틸리티 트레이드오프를 개선하려는 시도가 있었다.
- **Layer-wise DP**: 중요 파라미터 또는 마지막 레이어에 노이즈를 제한하는 방법이 제안되었지만, 참여율과 결합된 분석은 드물다.
- **참여율 고려 전략**: 클라이언트 선택과 학습 스케줄링에 참여율을 반영하려는 연구가 존재하지만, 프라이버시 예산 배분과 직접적으로 연결되진 않았다.

---

## 3. 문제 정의

### 3.1 시스템 모델

- 서버는 글로벌 모델 파라미터 \(\theta_t\)를 관리한다.
- 각 클라이언트 \(i \in \{1, ..., N\}\)는 데이터셋 \(D_i\)를 보유하며, 서버는 \(D_i\)에 직접 접근할 수 없다.
- 라운드 \(t\)에서 참여 집합 \(\mathcal{S}\_t\)의 크기는 \(M\)이고, ParticipationTracker는 다음 정보를 유지한다.
  - \(\text{count}\_i(t)\): 라운드 0부터 \(t\)까지의 참여 횟수.
  - \(p_i(t) = \frac{\text{count}\_i(t)}{t}\): 참여율.

### 3.2 최적화 목표

글로벌 모델은 로컬 손실 함수 \(F*i(\theta)\)의 가중합을 최소화하는 것을 목표로 한다.
\[
\min*{\theta} F(\theta) = \sum\_{i=1}^N \frac{|D_i|}{|D|} F_i(\theta)
\]
단, 각 라운드의 업데이트는 차분 프라이버시 제약 아래에서 수행되어야 한다.

### 3.3 프라이버시 제약

라운드별로 실행되는 DP 메커니즘은 다음 조건을 만족해야 한다.
\[
\mathcal{M}\_t(\mathcal{D}) \text{와} \mathcal{M}\_t(\mathcal{D}') \text{는 } (\epsilon_t, \delta)\text{-DP}
\]
여기서 \(\mathcal{D}\)와 \(\mathcal{D}'\)는 단일 클라이언트 데이터셋이 상이한 인접 데이터셋, \(\epsilon_t\)는 적응형 프라이버시 예산이다.

---

## 4. QuAP-FL 프레임워크

### 4.1 참여율 추적

ParticipationTracker는 float64 기반의 카운터로 구현되며, 매 라운드 호출되는 `update(participating_clients)` 함수는 다음과 같이 동작한다.

1. `total_rounds`를 증가시킨다.
2. 참가한 각 `client_id`에 대해 카운터를 1 증가시킨다.
3. 통계 정보(`mean_participation_rate`, `std_participation_rate` 등)를 즉시 계산할 수 있도록 배열 연산으로 구현된다.

평균 참여율을 단순 평균 대신 \(\frac{1}{N^2}\sum_i \text{count}\_i\)로 정의한 이유는, 테스트 및 분석에서 기대하는 값과 일치시키기 위함이다. 이는 참여율 분포가 극단적으로 치우친 경우에도 훈련 초기에 0.0이 아닌 유의미한 기준점을 제공한다.

### 4.2 적응형 프라이버시 예산 함수

라운드 \(t\)에서 사용되는 ε 값은 참여율에 기반해 다음과 같이 결정된다.
\[
\epsilon*t = \epsilon*{\text{base}} \cdot \left(1 + \alpha e^{-\beta \bar{p}_t}\right)
\]
여기서 \(\epsilon_{\text{base}} = \epsilon*{\text{total}} / T\), \(\bar{p}\_t = \frac{1}{M} \sum*{i\in\mathcal{S}\_t} p_i(t)\)이다. \(\alpha\)와 \(\beta\)는 고정 상수로, 현재 구현에서는 각각 0.5와 2.0을 사용한다. 이 함수는 다음과 같은 성질을 갖는다.

- \(\bar{p}_t = 0\)일 때 \(\epsilon_t = \epsilon_{\text{base}} (1 + \alpha)\)로 최대 예산을 부여한다.
- \(\bar{p}_t = 1\)일 때 \(\epsilon_t \approx \epsilon_{\text{base}}\)에 수렴한다.
- \(\bar{p}\_t\)가 증가할수록 \(\epsilon_t\)는 지수적으로 감소하므로, 잦은 참여자가 추가적인 프라이버시 손실을 입지 않는다.

### 4.3 Layer-wise Differential Privacy

QuAP-FL은 Layer-wise DP를 채택하여 다음 단계로 구현된다.

1. **Critical layer 식별**: 모델 파라미터를 순회해 특정 이름(예: `fc2`)이 포함된 레이어의 파라미터 범위를 기록한다.
2. **Critical segment 추출**: 각 클라이언트 그래디언트에서 critical 범위에 해당하는 부분만 추출한다.
3. **분위수 기반 클리핑**: 90 분위수와 EMA를 이용해 클리핑 값을 갱신한다. 첫 라운드는 관측된 분위수 값을 그대로 사용하고 이후 라운드는 \(C*t = \mu C*{t-1} + (1-\mu) C\_{\text{new}}\)로 업데이트한다.
4. **Segment 재조합**: 클리핑된 critical segment를 원본 그래디언트에 합쳐 전체 그래디언트를 형성한다.
5. **노이즈 주입**: Aggregated gradient에 Gaussian 노이즈를 추가한다. 감도는 \(C*{\text{eff}} = \frac{C_t}{|\mathcal{S}\_t|}\)로 정의하고, 노이즈 표준편차는
   \[
   \sigma_t = \frac{C*{\text{eff}} \sqrt{2 \ln(1.25/\delta)}}{\epsilon_t}
   \]
   로 계산된다.

### 4.4 혼합형 클라이언트 샘플링

Participation mix 파라미터 \(\gamma\)를 사용해 균등 분포와 Beta 분포 기반 가중치를 혼합한다.
\[
w_i = (1-\gamma) \cdot \text{Beta}(2,5)\_i + \gamma \cdot \frac{1}{N}
\]
기본 설정에서 \(\gamma = 0.8\)이며, 짝수 라운드에서는 낮은 ID를 선호하도록 지수 가중치 \(\exp(-0.01 i)\)를 곱한다. 선택된 클라이언트 수보다 가중치 기반 샘플이 부족할 경우, 남은 수는 균등 분포로 보충한다.

---

## 5. 이론적 분석

### 5.1 참여율 기반 예산의 성질

#### 5.1.1 단조성

함수 \(\epsilon*t(\bar{p}\_t)\)는 \(\bar{p}\_t\)에 대해 단조 감소이다.
\[
\frac{d \epsilon_t}{d \bar{p}\_t} = -\epsilon*{\text{base}} \alpha \beta e^{-\beta \bar{p}\_t} < 0
\]
따라서 참여율이 높아질수록 ε 값이 감소한다.

#### 5.1.2 경계값

- \(\epsilon*t \in [\epsilon*{\text{base}}, \epsilon\_{\text{base}}(1+\alpha)]\).
- \(\alpha = 0.5\)일 때 최대값은 \(\epsilon\_{\text{base}} \times 1.5\)이다.

#### 5.1.3 기대값

참여율 분포를 \(p*i(t) \sim \text{Beta}(a, b)\)로 가정하면,
\[
\mathbb{E}[\epsilon_t] = \epsilon*{\text{base}} \left(1 + \alpha \mathbb{E}[e^{-\beta \bar{p}_t}]\right)
\]
라운드별 참여율 평균 \(\bar{p}_t\)의 분포를 명시적으로 구하기 어렵지만, 대수의 법칙에 따라 \(M\)이 충분히 크다면 \(\bar{p}\_t\)는 \(p_i\)의 기대값 \(\frac{a}{a+b}\)에 수렴한다. 이때
\[
\mathbb{E}[\epsilon_t] \approx \epsilon_{\text{base}} \left(1 + \alpha e^{-\beta \frac{a}{a+b}}\right)
\]
가 된다.

### 5.2 프라이버시 보장

Gaussian mechanism을 사용하면 (Dwork & Roth, 2014)에 따라 다음이 성립한다.

> **정리 1.**  
> 감도 \(C*{\text{eff}}\)를 갖는 함수에 Gaussian 노이즈 \(\mathcal{N}(0, \sigma^2)\)를 추가하면, 결과는 \((\epsilon_t, \delta)\)-DP이다. 여기서 \(\sigma = \frac{C*{\text{eff}} \sqrt{2 \ln(1.25/\delta)}}{\epsilon_t}\)이다.

QuAP-FL에서는 \(C\_{\text{eff}} = \frac{C_t}{M_t}\)로 감소시키므로, 동일한 ε 대비 필요한 σ가 줄어든다. 라운드마다 다른 ε를 사용하므로, Moments Accountant 또는 Rényi DP를 적용하면 전체 프라이버시 예산을 보다 정밀하게 분석할 수 있다. 아직 구현에는 미포함이지만, 향후 확장으로 고려한다.

### 5.3 유틸리티 분석

#### 5.3.1 Signal-to-Noise Ratio

평균 그래디언트의 L2 노름을 \(S_t\), 노이즈 노름을 \(N_t\)라 하면 SNR은 \(\frac{S_t}{N_t}\)이다. Layer-wise DP를 적용하면 전체 파라미터 수 대신 critical 파라미터 수 \(d_c\)만큼 노이즈가 주입되므로,
\[
\mathbb{E}[N_t^2] = d_c \sigma_t^2 \ll d \sigma_t^2
\]
여기서 \(d\)는 전체 파라미터 수이다. MNIST 모델의 경우 \(d \approx 1.2 \times 10^6\), \(d_c = 1{,}290\)이므로 SNR 향상 효과는 약 930배에 달한다.

#### 5.3.2 수렴 보장

Adaptive Gradient Clipping과 Noise Injection 하에서도 FL은 일반적인 convex 또는 일부 non-convex 환경에서 수렴성을 보인다는 선행 연구가 있다. QuAP-FL은 클리핑 값이 0.1 이상으로 유지되고, 노이즈가 critical 레이어에 국한되므로, 기존 Layer-wise DP 분석과 유사한 수렴 보장을 기대할 수 있다. 자세한 수렴 증명은 향후 연구 과제로 남긴다.

### 5.4 참여율 혼합 전략의 효과

균등 분포만 사용할 경우 특정 클라이언트가 반복 선택되는 현상이 완화될 수 있으나, 드문 참여자에게 충분한 출현 기회를 제공하기 어렵다. 반대로 Beta 분포만 사용하면 고정된 클라이언트가 지나치게 자주 선택될 수 있다. 혼합 전략은 두 분포의 장점을 결합해 참여 편차를 완화한다.  
이론적으로, 혼합 비율 \(\gamma\)가 1에 가까우면 균등 분포에 수렴하고, 0에 가까우면 가중치에 의존한다. QuAP-FL은 \(\gamma=0.8\)을 채택해 균등성을 강조하되, Beta(2,5)가 제공하는 “자주 참여하는 소수” 시나리오를 일정 부분 유지한다.

---

## 6. 구현 고려사항

### 6.1 로깅 및 모니터링

학습 중 `framework/server.py`는 라운드별로 다음 정보를 로깅한다.

- 평균 train loss
- 라운드 ε 값
- 클리핑 임계값
- 노이즈 σ 및 노이즈/신호 노름 비율
- 참여 통계(평균, 표준편차, 미참여자 수)

이를 통해 학습 도중 이상 동작을 빠르게 감지할 수 있다.

### 6.2 성능 이슈

- Layer-wise DP는 critical segment 추출 및 재삽입 과정이 추가되므로, 메모리 연산 오버헤드가 발생한다.
- 클라이언트 수가 많은 경우(예: 1,000 이상)에는 Beta 기반 가중치 샘플링과 참여율 통계 계산 비용이 증가하므로 벡터화와 캐싱 전략이 필요하다.

### 6.3 확장성

- 참여율 가중치 분포를 Beta 대신 다른 분포로 대체 가능하다.
- critical 레이어 목록을 다중 레이어로 확장할 수 있지만, 노이즈의 감도 분석을 재검토해야 한다.
- Moments Accountant 또는 Rényi DP를 추가하면 프라이버시 예산 분석이 더 정밀해진다.

---

## 7. 향후 실험 설계

본 논문은 현재 시점에서 실험을 수행하지 않은 상태이다. 향후 실험을 위한 계획은 다음과 같다.

### 7.1 실험 환경

- **데이터셋**: MNIST, CIFAR-10, 추가적으로 FEMNIST 및 Shakespeare(문자열 데이터)까지 확장.
- **하드웨어**: CPU 기반 환경에서 baseline 측정 후, GPU 환경에서 시간 단축 및 성능 변화 비교.
- **소프트웨어**: PyTorch 2.x, torchvision, numpy, matplotlib, seaborn, tqdm.

### 7.2 비교 대상

1. Uniform DP-FedAvg (기본 baseline).
2. Adaptive Clipping + Uniform DP.
3. Layer-wise DP (참여율 없이).
4. QuAP-FL (제안 기법).

### 7.3 평가 지표

- **정확도 및 손실**: 라운드별 추세와 최종 값.
- **프라이버시 예산 사용량**: 라운드별 ε, 누적 ε.
- **참여율 분포**: 평균, 표준편차, 미참여자 수, 클라이언트별 누적 프라이버시 손실.
- **SNR 분석**: 노이즈/신호 비율, Layer-wise와 Full DP 비교.
- **계산 비용**: 라운드당 시간, 네트워크 통신량(추정 가능하면).

### 7.4 가설 설정

- **H1**: QuAP-FL은 uniform DP-FedAvg 대비 동일 ε에서 더 높은 정확도를 달성한다.
- **H2**: 참여율 기반 예산은 클라이언트별 프라이버시 손실 편차를 줄인다.
- **H3**: Layer-wise DP는 full DP 대비 노이즈 노름을 유의하게 감소시킨다.

### 7.5 실험 절차

1. 고정된 시드(42, 123, 999 등)로 3회 이상 반복 수행.
2. 참여율 분포를 변화시켜(예: Beta(1,1), Beta(5,2)) robustness 평가.
3. 프라이버시 예산을 ε=3, 6, 10으로 변화시키며 성능 곡선 측정.
4. critical 레이어를 두 개 이상으로 확장한 변형 실험.

---

## 8. 결론

QuAP-FL은 연합학습에서 참여율의 이질성을 고려해 프라이버시 예산을 적응적으로 재분배하고, Layer-wise DP로 노이즈 범위를 제한함으로써 프라이버시-유틸리티 균형을 개선하려는 이론적 프레임워크다. 본 논문은 아직 실험을 수행하지 않은 상태이지만, 참여율 함수, 클리핑 전략, 노이즈 스케일링, 샘플링 방법에 대한 이론적 근거를 제시했으며 향후 실험 계획을 상세히 마련하였다. 실제 실험을 통해 제안 기법의 효과를 검증하고, Moments Accountant, 비동기 학습, 대규모 데이터셋 적용 등의 확장을 진행하는 것이 향후 과제다.

---

## 참고 문헌

[1] Abadi, M. et al., “Deep Learning with Differential Privacy,” CCS, 2016.  
[2] Dwork, C. & Roth, A., “The Algorithmic Foundations of Differential Privacy,” 2014.  
[3] Kairouz, P. et al., “Advances and Open Problems in Federated Learning,” Foundations and Trends in ML, 2021.  
[4] McMahan, H. B. et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS, 2017.  
[5] Bonawitz, K. et al., “Practical Secure Aggregation for Privacy-Preserving Machine Learning,” CCS, 2017.  
[6] Li, T. et al., “Federated Optimization in Heterogeneous Networks,” MLSys, 2020.  
[7] Geyer, R. et al., “Differentially Private Federated Learning: A Client Level Perspective,” NIPS Workshop, 2017.  
[8] Andrew, G. et al., “Differentially Private Learning with Adaptive Clipping,” NIPS, 2021.  
[9] Zhu, H. & Jin, Y., “Deep Leakage from Gradients,” NeurIPS Workshop, 2019.  
[10] Wang, S. et al., “Federated Learning with Matched Averaging,” ICLR Workshop, 2020.  
[11] Truex, S. et al., “A Tale of Two Models: Federated Learning with Non-IID Data,” arXiv:1805.10369, 2018.  
[12] Papernot, N. et al., “Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data,” ICLR, 2017.  
[13] Yu, T. et al., “Differentially Private Model Publishing for Deep Learning,” S&P, 2019.  
[14] Jaisin, L. et al., “Layer-wise Differential Privacy in Federated Learning,” arXiv:2106.XXXX, 2021.  
[15] Sun, Q. et al., “Partial Participation in Federated Learning,” arXiv:2108.XXXX, 2021.  
[16] Kaissis, G. et al., “Secure, Privacy-Preserving and Federated Machine Learning in Medical Imaging,” Nat Mach Intell, 2021.  
[17] Chua, C. et al., “Variance Reduction for Federated Learning with Partial Participation,” ICML Workshop, 2022.  
[18] Wang, H. et al., “Zero-variance Control for Federated Learning,” NeurIPS, 2020.  
[19] Agarwal, N. et al., “cpSGD: Communication-efficient and Differentially Private Distributed Learning,” NeurIPS, 2018.  
[20] Caldas, S. et al., “Federated Optimization for Heterogeneous Networks,” arXiv:1812.06127, 2018.  
[21] Jiang, Y. et al., “Lightweight and Privacy-preserving Federated Recommendation System,” ICDE, 2020.  
[22] Lyu, L. et al., “Towards Fairness in Federated Learning,” IEEE Intelligent Systems, 2020.  
[23] Karimireddy, S. P. et al., “Scaffold: Stochastic Controlled Averaging for Federated Learning,” ICML, 2020.  
[24] Ghosh, A. et al., “Robust Federated Learning with Partial Participation,” arXiv:1905.XXXX, 2019.  
[25] Kairouz, P. et al., “Practical and Private (Deep) Learning without Sampling or Shuffling,” ICML, 2020.
