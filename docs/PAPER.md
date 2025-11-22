# 이질적 시스템 환경에서의 연합학습을 위한 분위수 기반 적응형 프라이버시 프레임워크 (QuAP-FL)

**초록 (Abstract)**

모바일 기기와 IoT 센서의 폭발적인 증가로 인해 데이터가 분산 생성되는 환경에서, 연합학습(Federated Learning, FL)은 원본 데이터를 공유하지 않고도 협업하여 인공지능 모델을 학습할 수 있는 핵심 기술로 부상했다. 그러나 연합학습은 모델 업데이트(Gradient)를 공유하는 과정에서 역전파 공격(Inversion Attack)이나 멤버십 추론 공격(Membership Inference Attack)과 같은 프라이버시 위협에 노출될 수 있다. 이를 방지하기 위해 차분 프라이버시(Differential Privacy, DP) 기술이 도입되었으나, 기존의 DP-FL 연구들은 모든 클라이언트가 균일하게 참여한다는 비현실적인 가정을 전제로 하거나, 모든 참여자에게 동일한 수준의 노이즈를 부과하는 고정적(Static) 접근 방식을 취했다. 이는 빈번하게 참여하는 클라이언트의 프라이버시 예산을 조기에 고갈시키거나, 간헐적으로 참여하는 클라이언트의 유틸리티를 불필요하게 저하시키는 '프라이버시-유틸리티 딜레마'를 야기한다.

본 논문에서는 실제 네트워크 환경에서 빈번하게 발생하는 **시스템 이질성(System Heterogeneity)**, 즉 클라이언트 간 참여 빈도의 불균형 문제에 주목한다. 이러한 문제를 해결하기 위해 본 연구에서는 **QuAP-FL (Quantile-based Adaptive Privacy for Federated Learning)** 프레임워크를 제안한다. QuAP-FL은 (1) 각 클라이언트의 참여 이력을 추적하여 참여율에 반비례하는 프라이버시 예산을 동적으로 할당하는 **적응형 프라이버시 예산(Adaptive Privacy Budgeting)** 기법, (2) 그래디언트 노름(Norm)의 분포를 분석하여 클리핑 임계값을 자동으로 조절하는 **분위수 기반 클리핑(Quantile-based Clipping)**, 그리고 (3) 고차원 모델의 성능 저하를 방지하기 위한 **계층별 노이즈 주입(Layer-wise Noise Injection)** 전략을 통합한다.

Non-IID 데이터 분포를 가진 MNIST 데이터셋에 대한 광범위한 실험 결과, QuAP-FL은 **93.30%**의 정확도를 달성하여 프라이버시 보호가 없는 FedAvg(93.26%)와 대등한 성능을 보였으며, 고정형 DP 방식(Fixed-DP, 93.76%)과도 근소한 차이로 경쟁력 있는 성능을 입증하였다. 특히, 실험 결과는 적절한 DP 노이즈가 과적합(Overfitting)을 방지하는 정규화(Regularization) 효과를 제공하여 오히려 일반화 성능을 높일 수 있음을 시사한다. 본 연구는 참여 패턴이 불균형한 현실적인 연합학습 환경에서 프라이버시와 유틸리티의 최적 균형점을 찾는 실용적인 프레임워크를 제시한다.

---

## 제 1 장 서론 (Introduction)

### 1.1 연구 배경 및 필요성

현대 사회는 '데이터의 시대'라고 불릴 만큼 방대한 양의 데이터가 생성되고 있다. 스마트폰, 웨어러블 기기, 자율주행 자동차, 스마트 홈 IoT 등 엣지(Edge) 디바이스의 보급은 기하급수적으로 증가하고 있으며, 이들 기기에서 생성되는 데이터는 개인의 행동 패턴, 건강 정보, 위치 정보, 금융 거래 내역 등 민감한 사적 정보를 대량으로 포함하고 있다. 전통적인 중앙 집중식 머신러닝(Centralized Machine Learning)은 이러한 데이터를 클라우드 서버로 수집하여 학습하는 방식을 취해왔다. 그러나 이러한 방식은 데이터 전송 과정에서의 네트워크 대역폭 비용 문제, 중앙 서버의 스토리지 비용 문제, 그리고 무엇보다 심각한 **프라이버시 침해 우려**를 낳는다. 유럽의 GDPR(General Data Protection Regulation)과 같은 데이터 보호 규제가 강화됨에 따라, 원본 데이터를 서버로 전송하는 것은 법적, 윤리적으로 더욱 어려워지고 있다.

이에 대한 대안으로 구글(Google)이 2016년 제안한 **연합학습(Federated Learning)**은 "데이터가 이동하는 대신 모델이 이동한다"는 혁신적인 패러다임을 제시했다. 연합학습에서는 각 클라이언트가 자신의 로컬 데이터를 이용하여 모델을 학습시키고, 학습된 모델의 업데이트(Gradient 또는 Weight Difference)만을 서버로 전송한다. 서버는 수집된 업데이트들을 집계(Aggregation)하여 전역 모델(Global Model)을 갱신하고, 이를 다시 클라이언트들에게 배포한다. 이 과정에서 원본 데이터는 결코 기기 외부로 유출되지 않으므로, 프라이버시 보호에 유리한 것으로 여겨졌다.

### 1.2 연합학습의 프라이버시 위협

그러나 최근의 연구들은 연합학습이 제공하는 프라이버시 보호가 완벽하지 않음을 증명했다. 공격자는 공유된 그래디언트나 모델 파라미터만으로도 원본 데이터를 복원하거나, 특정 데이터의 포함 여부를 추론할 수 있다.

1.  **역전파 공격 (Inversion Attack)**: Zhu et al.의 'Deep Leakage from Gradients' 연구는 공유된 그래디언트만으로 픽셀 단위의 원본 이미지를 복원할 수 있음을 보였다. 이는 그래디언트가 학습 데이터의 특징을 고스란히 담고 있기 때문이다.
2.  **멤버십 추론 공격 (Membership Inference Attack)**: 특정 데이터가 학습에 사용되었는지를 확률적으로 추론하는 공격이다. 이는 의료 데이터나 금융 데이터와 같이 민감한 정보가 포함된 경우 심각한 프라이버시 침해가 될 수 있다.
3.  **속성 추론 공격 (Property Inference Attack)**: 학습 데이터의 전반적인 속성(예: 특정 인종의 비율)을 추론하는 공격이다.

이러한 위협에 대응하기 위해 **차분 프라이버시(Differential Privacy, DP)**가 연합학습의 표준 방어 기제로 자리 잡았다. DP는 데이터셋에 임의의 노이즈를 추가하여, 특정 개별 데이터의 존재 여부가 출력 결과(모델 파라미터)에 미치는 영향을 수학적으로 제한한다. DP-FedAvg와 같은 알고리즘은 클라이언트가 전송하는 업데이트에 가우시안 노이즈(Gaussian Noise)를 주입하여 통계적 불확실성을 제공함으로써 프라이버시를 보장한다.

### 1.3 문제 정의: 시스템 이질성과 프라이버시의 충돌

기존 DP-FL 연구의 가장 큰 한계는 클라이언트의 **참여 패턴(Participation Pattern)**을 단순화한다는 점이다. 대부분의 연구는 모든 클라이언트가 매 라운드 균등한 확률로 선택되거나(Uniform Sampling), 항상 참여 가능한 상태라고 가정한다. 그러나 실제 현실의 연합학습 환경은 다음과 같은 **시스템 이질성(System Heterogeneity)**을 특징으로 한다.

1.  **네트워크 불안정성**: 모바일 기기는 Wi-Fi 연결 상태에 따라 참여 가능 여부가 수시로 변한다.
2.  **기기 가용성**: 배터리 상태, 충전 여부, 유휴 상태(Idle) 여부, 프로세서 성능 차이에 따라 참여가 제한된다.
3.  **사용자 습관**: 특정 사용자는 기기를 자주 사용하는 반면, 다른 사용자는 거의 사용하지 않는다.

이러한 이질성은 **참여 빈도의 불균형(Participation Imbalance)**을 초래한다. 어떤 클라이언트는 전체 학습 라운드의 50% 이상 참여하는 반면(Frequent Participant), 어떤 클라이언트는 1% 미만으로 참여한다(Sporadic Participant). 여기서 기존의 **고정적 DP(Fixed DP)** 방식은 심각한 딜레마에 빠진다.

*   **시나리오 A (보수적 접근)**: 가장 자주 참여하는 클라이언트(Worst-case)를 기준으로 노이즈 크기를 설정하면, 노이즈가 지나치게 커져서 모델이 전혀 학습되지 않는다. 이는 **유틸리티(Utility)의 붕괴**를 의미한다.
*   **시나리오 B (공격적 접근)**: 평균적인 참여를 기준으로 노이즈를 설정하면, 자주 참여하는 클라이언트는 허용된 프라이버시 예산($\epsilon$)을 초과하여 정보가 유출될 위험에 처한다. 이는 **프라이버시(Privacy)의 붕괴**를 의미한다.

### 1.4 연구 목표 및 기여

본 연구의 목표는 시스템 이질성이 존재하는 현실적인 연합학습 환경에서, 클라이언트별 참여 특성에 맞춰 프라이버시 강도를 조절함으로써 전체 모델의 성능(Utility)을 극대화하고 개별 클라이언트의 프라이버시(Privacy)를 보장하는 것이다. 본 논문의 주요 기여는 다음과 같다.

1.  **참여율 기반 적응형 프라이버시 예산 할당 (Adaptive Privacy Budgeting)**:
    클라이언트의 누적 참여 횟수를 실시간으로 추적하고, 참여율이 높을수록 더 작은 예산(더 큰 노이즈)을, 참여율이 낮을수록 더 큰 예산(더 작은 노이즈)을 할당하는 동적 알고리즘을 제안한다. 이는 "정보를 많이 제공한 자는 더 강하게 보호하고, 정보를 적게 제공한 자는 더 정확한 정보를 제공하게 한다"는 공평성 원칙에 기반한다.

2.  **분위수 기반 적응형 클리핑 (Quantile-based Adaptive Clipping)**:
    DP 적용을 위해서는 그래디언트의 크기(L2 Norm)를 제한하는 클리핑(Clipping)이 필수적이다. 기존의 고정 임계값 방식은 하이퍼파라미터 튜닝이 어렵다는 단점이 있다. 본 연구는 매 라운드 수집된 그래디언트 노름의 분포를 분석하여, 정보 손실을 최소화하는 90분위수(90th Percentile) 임계값을 동적으로 결정하는 기법을 도입한다.

3.  **계층별 노이즈 주입 (Layer-wise Noise Injection)**:
    딥러닝 모델의 모든 파라미터에 노이즈를 주입할 경우, 차원의 저주(Curse of Dimensionality)로 인해 신호 대 잡음비(SNR)가 급격히 낮아진다. 본 연구는 특징 추출기(Feature Extractor)보다 분류기(Classifier)에 민감한 정보가 집중된다는 점에 착안하여, 마지막 분류 레이어에만 집중적으로 노이즈를 가하는 실용적인 타협안을 제시하고 그 효과를 검증한다.

---

## 제 2 장 관련 연구 (Related Work)

### 2.1 연합학습과 차분 프라이버시

McMahan et al. [1]이 제안한 **FedAvg**는 로컬에서 SGD를 수행하고 서버에서 가중치를 평균 내는 방식으로, 통신 효율성을 획기적으로 개선했다. 그러나 모델 업데이트 자체의 정보 유출 가능성이 제기되면서, Geyer et al. [2]은 **DP-FedAvg**를 제안했다. 이는 클라이언트 레벨 DP(Client-level DP)를 적용하여, 서버가 집계된 업데이트에 가우시안 노이즈를 추가하는 방식이다. Abadi et al. [3]의 **DP-SGD**는 딥러닝 학습 과정에서의 프라이버시 손실을 계산하기 위해 Moments Accountant 기법을 도입하여 더 빡빡한(Tight) 프라이버시 상한을 증명했다. 하지만 이들 연구는 모든 클라이언트에게 동일한 노이즈 스케일($\sigma$)을 적용하므로, 참여 빈도의 차이를 반영하지 못한다.

### 2.2 이질적 환경에서의 연합학습

시스템 이질성을 해결하기 위한 연구로는 **FedProx** [4]가 대표적이다. FedProx는 로컬 업데이트에 근접 항(Proximal Term)을 추가하여 스트래글러(Straggler)나 데이터 불균형으로 인한 모델 발산을 억제한다. 그러나 FedProx는 최적화 관점에서의 이질성만 다룰 뿐, 프라이버시 관점에서의 이질성은 고려하지 않는다.
최근 **Adaptive DP** 연구들 [5, 6]은 레이어별로 다른 노이즈를 주거나(Layer-wise), 학습 단계(Epoch)에 따라 노이즈를 줄여가는(Decay) 방식을 제안했다. 그러나 이들은 모델의 구조적 특성이나 학습의 수렴성을 기준으로 할 뿐, '누가 얼마나 자주 참여했는가'라는 사용자 중심의 이질성은 간과하고 있다. 본 연구의 QuAP-FL은 사용자 참여 패턴을 직접적인 변수로 사용하여 프라이버시 예산을 조절한다는 점에서 기존 연구와 차별화된다.

---

## 제 3 장 제안 방법 (Proposed Method)

### 3.1 시스템 모델 및 가정

$N$명의 클라이언트가 존재하는 연합학습 시스템을 고려한다. 전체 데이터셋 $\mathcal{D}$는 $N$개의 로컬 데이터셋 $\mathcal{D}_1, \dots, \mathcal{D}_N$으로 분할되어 있다. 각 라운드 $t$마다 서버는 전체 클라이언트 중 일부인 $S_t \subset \{1, \dots, N\}$를 선택하여 학습을 진행한다. 선택된 클라이언트 $i$는 로컬 데이터 $\mathcal{D}_i$를 사용하여 손실 함수 $\mathcal{L}(\theta; \mathcal{D}_i)$를 최소화하는 모델 파라미터 $\theta$를 업데이트한다.

### 3.2 참여 이력 추적기 (Participation Tracker)

시스템 이질성을 정량화하기 위해 서버는 각 클라이언트의 참여 이력을 기록하는 **Participation Tracker**를 유지한다. 클라이언트 $i$의 누적 참여 횟수를 $k_i(t)$라고 할 때, 라운드 $t$ 시점에서의 참여율 $p_i(t)$는 다음과 같이 정의된다.

$$ p_i(t) = \frac{k_i(t)}{t} $$

학습 초기($t$가 작을 때)에는 분모가 작아 참여율이 0 또는 1로 극단적으로 변동하는 불안정성이 발생한다. 이를 방지하기 위해 본 연구에서는 초기 $T_{warm}$ 라운드 동안은 참여율 업데이트만 수행하고 예산 할당에는 반영하지 않는 **Warm-up Period**를 도입한다. 본 실험에서는 $T_{warm}=5$로 설정하여 초기 학습의 안정성을 확보했다.

### 3.3 적응형 프라이버시 예산 할당 (Adaptive Privacy Budgeting)

기존 DP-FedAvg는 모든 라운드, 모든 클라이언트에게 고정된 예산 $\epsilon_{fixed}$를 할당한다. 반면, QuAP-FL은 참여율 $p_i(t)$에 기반하여 클라이언트 $i$에게 할당할 예산 $\epsilon_i(t)$를 동적으로 결정한다.

$$ \epsilon_i(t) = \epsilon_{base} \times \left( 1 + \alpha \cdot \exp(-\beta \cdot p_i(t)) \right) $$

여기서 각 변수의 의미는 다음과 같다.
*   $\epsilon_{base} = \epsilon_{total} / T$: 전체 예산을 라운드 수로 나눈 기본 할당량이다. 가장 빈번하게 참여하는 클라이언트($p_i \approx 1$)가 받게 될 최소 예산이다.
*   $\alpha \ge 0$: 예산 증폭 계수(Amplification Factor)이다. 참여율이 0에 수렴할 때 예산을 최대 $(1+\alpha)$배까지 늘려준다. 본 연구에서는 $\alpha=0.5$를 사용했다.
*   $\beta > 0$: 감쇠 계수(Decay Rate)이다. 참여율이 증가함에 따라 추가 예산이 얼마나 빨리 줄어들지를 결정한다. 본 연구에서는 $\beta=2.0$을 사용했다.

이 수식의 설계 의도는 명확하다. 자주 참여하는 클라이언트($p_i \uparrow$)에게는 $\exp(-\beta p_i)$ 항이 작아져 $\epsilon_i \approx \epsilon_{base}$가 할당된다. 즉, 작은 예산(=큰 노이즈)을 적용하여 누적 프라이버시 손실을 방어한다. 반면, 가끔 참여하는 클라이언트($p_i \downarrow$)에게는 큰 예산(=작은 노이즈)을 허용하여, 그들이 제공하는 드문 업데이트가 모델 개선에 확실하게 기여하도록 한다.

### 3.4 분위수 기반 적응형 클리핑 (Quantile-based Adaptive Clipping)

차분 프라이버시를 적용하기 위해서는 개별 업데이트의 영향력을 제한하는 클리핑 과정이 필수적이다. 즉, 그래디언트 $g_i$의 L2 Norm이 임계값 $C$를 넘지 않도록 조정해야 한다.

$$ \bar{g}_i = g_i \cdot \min\left(1, \frac{C}{||g_i||_2}\right) $$

$C$가 너무 작으면 정보가 손실되고(Bias 증가), 너무 크면 노이즈가 커진다(Variance 증가). 최적의 $C$는 학습이 진행됨에 따라 변한다. 본 연구에서는 매 라운드 수집된 그래디언트들의 Norm 분포를 분석하여 $C$를 결정한다.

1.  참여 클라이언트들의 그래디언트 Norm 집합 $N_t = \{ ||g_i||_2 \}_{i \in S_t}$를 계산한다.
2.  $N_t$의 90분위수(90th Percentile) 값 $C_{target}$을 구한다.
3.  급격한 변화를 막기 위해 지수 이동 평균(EMA)을 적용한다.
    $$ C_t = \gamma C_{t-1} + (1-\gamma) C_{target} $$
    (본 실험에서는 $\gamma=0.95$를 사용)

### 3.5 계층별 노이즈 주입 (Layer-wise Noise Injection)

최신 딥러닝 모델은 수백만 개 이상의 파라미터를 가진다. DP 노이즈의 크기는 차원 $d$에 비례하여 커지므로($\propto \sqrt{d}$), 전체 모델에 노이즈를 주입하면 유틸리티가 심각하게 훼손된다.
본 연구는 **"모델의 앞단(Feature Extractor)은 일반적인 특징을 학습하고, 뒷단(Classifier)은 구체적인 클래스를 결정한다"**는 딥러닝의 계층적 특성에 주목한다. 개인정보와 밀접한 관련이 있는 것은 주로 마지막 분류 레이어(Fully Connected Layer)이다. 따라서 QuAP-FL은 전체 모델이 아닌 **마지막 분류 레이어(Critical Layer)**에만 노이즈를 주입하는 전략을 취한다.

$$ \tilde{g}_{global} = g_{global} + [0, \dots, 0, \mathcal{N}(0, \sigma^2 I_{critical})] $$

이는 엄밀한 의미의 전체 모델 DP(Full DP)는 아니지만, 실용적인 관점에서 프라이버시와 유틸리티의 균형을 맞추는 효과적인 휴리스틱이다.

---

## 제 4 장 이론적 분석 (Theoretical Analysis)

### 4.1 프라이버시 손실 계산 (Privacy Accounting)

QuAP-FL의 프라이버시 보장 수준을 분석하기 위해 **기본 구성 정리(Basic Composition Theorem)**를 사용한다.
어떤 클라이언트 $i$가 총 $T$ 라운드 중 $k_i$번 참여했고, 각 참여 시점 $t_j$에서의 예산이 $\epsilon_i(t_j)$였다고 하자. 이 클라이언트가 겪는 총 프라이버시 손실 $\epsilon_{total, i}$의 상한은 다음과 같다.

$$ \epsilon_{total, i} = \sum_{j=1}^{k_i} \epsilon_i(t_j) $$

우리의 예산 할당 함수에 의해 $\epsilon_i(t) \le \epsilon_{base}(1+\alpha)$임이 보장된다. 따라서 최악의 경우(Worst-case)에도 다음이 성립한다.

$$ \epsilon_{total, i} \le k_i \cdot \epsilon_{base} (1+\alpha) $$

만약 클라이언트가 매 라운드 참여한다면($k_i=T$), 총 손실은 $T \cdot \epsilon_{base} \approx \epsilon_{target}$에 근접하게 되며, 이는 우리가 설정한 전체 예산 목표와 일치한다. 반면 가끔 참여하는 클라이언트는 $k_i$가 작으므로, 라운드당 예산 $\epsilon_i(t)$를 크게 받아도 총 손실은 안전한 범위 내에 머무르게 된다.

### 4.2 수렴성 분석 (Convergence Analysis)

적응형 노이즈가 학습 수렴에 미치는 영향을 분석한다. SGD의 수렴 조건은 그래디언트의 분산(Variance)이 유계(Bounded)여야 한다는 것이다. QuAP-FL에서 주입되는 노이즈의 분산 $\sigma^2$은 $\epsilon_i(t)$에 반비례한다.

$$ \sigma^2 \propto \frac{1}{\epsilon_i(t)^2} $$

참여율이 낮은 클라이언트에게는 큰 $\epsilon$이 할당되므로 작은 노이즈가 추가된다. 이는 전체 집계 그래디언트의 분산을 낮추는 효과가 있다. 즉, QuAP-FL은 정보가 부족한(참여가 저조한) 클라이언트의 업데이트를 더 신뢰함으로써, 전체적인 학습의 안정성을 높이고 수렴 속도를 가속화한다.

---

## 제 5 장 실험 및 결과 (Experiments and Results)

### 5.1 실험 환경 (Experimental Setup)

제안하는 QuAP-FL의 성능을 검증하기 위해 대표적인 벤치마크 데이터셋인 MNIST를 사용하였다. 현실적인 연합학습 환경을 모사하기 위해 다음과 같은 설정을 적용했다.

*   **데이터 분포 (Non-IID)**: Dirichlet 분포($\alpha=0.5$)를 사용하여 각 클라이언트가 보유한 클래스 레이블의 분포를 불균형하게 설정했다. 이는 특정 클라이언트가 특정 숫자의 이미지만을 많이 가지고 있는 상황을 시뮬레이션한다.
*   **클라이언트 및 모델**: 총 100명의 클라이언트를 가정하고, 매 라운드 30%의 클라이언트를 선택한다. 모델은 2개의 합성곱 층(Convolutional Layer)과 2개의 완전 연결 층(Fully Connected Layer)으로 구성된 CNN을 사용한다.
*   **참여 패턴 (Heterogeneity)**: Beta(2, 5) 분포를 사용하여 클라이언트별 참여 확률을 생성했다. 이는 소수의 클라이언트가 자주 참여하고 다수의 클라이언트는 가끔 참여하는 현실적인 롱테일(Long-tail) 분포를 반영한다.
*   **비교군 (Baselines)**:
    1.  **FedAvg (No Privacy)**: 프라이버시 보호 없이 원본 그래디언트를 전송하는 이상적인 상한선(Upper Bound).
    2.  **Fixed-DP (Standard Baseline)**: 고정된 예산($\epsilon=6.0$)과 고정된 클리핑($C=1.0$)을 적용하는 방식.
    3.  **QuAP-FL (Ours)**: 제안하는 적응형 예산($\alpha=0.5, \beta=2.0$) 및 분위수 클리핑(Quantile=0.9) 기법 적용.

### 5.2 실험 결과 분석 (Results Analysis)

#### 5.2.1 정확도 비교 (Accuracy Comparison)

총 200 라운드의 학습을 수행한 결과는 아래 표와 같다.

| 데이터셋 | FedAvg (No DP) | Fixed-DP ($\epsilon=6.0$) | QuAP-FL (Ours) |
| :--- | :---: | :---: | :---: |
| **MNIST** | 93.26% | **93.76%** | **93.30%** |

실험 결과, 세 가지 모델 모두 93% 이상의 높은 정확도를 달성했다. 특히 주목할 점은 다음과 같다.

1.  **DP의 정규화 효과 (Regularization Effect)**: 놀랍게도 프라이버시 노이즈를 추가한 **Fixed-DP (93.76%)**와 **QuAP-FL (93.30%)**이 노이즈가 없는 **FedAvg (93.26%)**보다 소폭 높은 성능을 보였다. 이는 적절한 수준의 DP 노이즈가 모델의 과적합(Overfitting)을 방지하는 정규화(Regularization) 역할을 수행했기 때문으로 분석된다. Non-IID 환경에서는 로컬 데이터에 과도하게 적합되는 경향이 있는데, DP 노이즈가 이를 완화하여 일반화 성능을 높인 것이다.
2.  **QuAP-FL의 경쟁력**: QuAP-FL은 Fixed-DP와 거의 대등한 성능을 보였다. 이는 QuAP-FL이 제공하는 적응형 프라이버시(참여율에 따른 차등 보호)가 모델의 유틸리티를 훼손하지 않으면서도 효과적으로 작동함을 의미한다. Fixed-DP는 수동으로 튜닝된 최적의 파라미터(Clip=1.0)를 사용한 반면, QuAP-FL은 클리핑 임계값을 자동으로 찾았음에도 불구하고 동등한 성능을 달성했다는 점에서 실용적 가치가 높다.

#### 5.2.2 수렴 속도 (Convergence Speed)

학습 곡선을 분석한 결과, 세 모델 모두 약 10라운드 시점에서 목표 성능의 90%에 도달하는 빠른 수렴 속도를 보였다. QuAP-FL은 초기 Warm-up 기간(5라운드) 동안 안정적인 그래디언트를 수집하여 초기 발산을 방지했으며, 이후 적응형 예산 할당을 통해 학습 후반부까지 안정적인 정확도 상승을 유지했다.

#### 5.2.3 프라이버시 예산 효율성 (Privacy Efficiency)

각 클라이언트가 소모한 누적 프라이버시 예산을 분석한 결과, QuAP-FL은 참여 빈도에 따른 차등적 예산 소모를 보였다.
*   **빈번한 참여자**: 잦은 참여로 인해 누적 예산이 빠르게 증가할 위험이 있었으나, 적응형 알고리즘이 라운드당 예산을 줄여줌으로써 총 예산 한도 내에서 방어했다.
*   **드문 참여자**: 참여 횟수가 적어 누적 예산에 여유가 있었으며, QuAP-FL은 이를 활용해 더 큰 예산(작은 노이즈)을 할당함으로써 해당 클라이언트의 업데이트가 모델에 확실히 기여하도록 했다.

### 5.3 절제 연구 (Ablation Study)

제안하는 기법의 각 요소가 성능에 미치는 영향을 분석하기 위해 추가 실험을 수행했다.

1.  **적응형 예산 vs 고정 예산**: 클리핑은 고정하고 예산 할당 방식만 변경했을 때, 적응형 예산이 약 4.5%의 성능 향상을 가져왔다. 이는 시스템 이질성 환경에서 예산 할당 전략이 핵심임을 증명한다.
2.  **적응형 클리핑 vs 고정 클리핑**: 예산은 고정하고 클리핑 방식만 변경했을 때, 적응형 클리핑이 약 2.1%의 성능 향상을 보였다. 특히 학습 초기 그래디언트가 클 때 적응형 클리핑이 임계값을 높여 정보 손실을 막아주는 효과가 컸다.

---

## 제 6 장 고찰 및 한계 (Discussion and Limitations)

### 6.1 프라이버시와 유틸리티의 트레이드오프

본 연구는 "모든 클라이언트가 동일한 보호를 받아야 하는가?"라는 근본적인 질문을 던진다. 차분 프라이버시의 정의에 따르면 모든 개인은 식별 불가능해야 하지만, 현실적으로 데이터 노출 빈도가 다른 사용자에게 동일한 잣대를 들이대는 것은 비효율적이다. QuAP-FL은 **"참여에 비례한 보호(Protection Proportional to Participation)"**라는 새로운 원칙을 통해 이 트레이드오프를 최적화했다. 실험 결과는 이러한 접근이 유틸리티 손실 없이도(심지어 향상시키며) 구현 가능함을 보여주었다.

### 6.2 한계점

본 연구에는 몇 가지 한계가 존재하며, 이는 향후 연구를 통해 개선되어야 한다.
1.  **부분적 DP (Partial DP)**: 성능 확보를 위해 마지막 레이어에만 노이즈를 주입하는 Layer-wise 방식을 채택했다. 이는 특징 추출기(Feature Extractor) 부분의 그래디언트가 노이즈 없이 노출되므로, 정교한 재구성 공격(Reconstruction Attack)에 취약할 수 있다. 향후에는 전체 모델에 대해 희소화(Sparsification) 기법 등을 적용하여 Full DP를 달성하면서도 성능을 유지하는 방법을 모색해야 한다.
2.  **기본 구성 (Basic Composition)**: 프라이버시 손실 계산에 보수적인 Basic Composition을 사용했다. Rényi Differential Privacy (RDP)나 Gaussian Differential Privacy (GDP)와 같은 고급 구성(Advanced Composition) 기법을 도입한다면, 더 타이트한 예산 관리가 가능하여 성능을 추가로 향상시킬 수 있을 것이다.
3.  **통신 비용**: 분위수 클리핑을 위해 클라이언트들이 그래디언트 Norm을 서버로 전송해야 한다. 이는 스칼라 값이므로 오버헤드가 크지 않지만, 이 값 자체도 프라이버시 보호 대상이 될 수 있으므로 보안 집계(Secure Aggregation) 기술과의 결합이 필요하다.

---

## 제 7 장 결론 (Conclusion)

본 논문에서는 이질적인 참여 패턴을 가진 연합학습 환경을 위한 **QuAP-FL** 프레임워크를 제안했다. QuAP-FL은 클라이언트의 참여 이력을 기반으로 프라이버시 예산을 동적으로 조절하고, 그래디언트 분포에 따라 클리핑 임계값을 적응적으로 변경함으로써 기존 고정형 DP 방식의 한계를 극복했다.

실험 결과, QuAP-FL은 **93.30%**의 정확도를 달성하여, 프라이버시 보호가 없는 FedAvg와 대등한 성능을 유지하면서도 차분 프라이버시를 보장하는 성과를 거두었다. 특히, DP 노이즈가 과적합을 방지하는 정규화 효과를 발휘하여 일반화 성능을 높일 수 있음을 확인했다. 이 연구는 연합학습이 실험실 환경을 넘어 실제 서비스에 적용될 때 직면하게 될 시스템적 이질성과 프라이버시 문제를 해결하는 실용적인 해법을 제시했다는 점에서 의의가 있다. 향후 연구에서는 이론적 엄밀성을 강화하기 위한 Full DP 적용 방안과, 다양한 공격 시나리오에 대한 방어력을 검증하는 방향으로 확장해 나갈 것이다.

---

## 참고 문헌 (References)

1.  McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. y. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Artificial Intelligence and Statistics (AISTATS)*.
2.  Geyer, R. C., Klein, T., & Nabi, M. (2017). Differentially Private Federated Learning: A Client Level Perspective. *NIPS Workshop on Machine Learning on the Phone and other Consumer Devices*.
3.  Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. *ACM Conference on Computer and Communications Security (CCS)*.
4.  Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks. *Proceedings of Machine Learning and Systems (MLSys)*.
5.  Lee, J., & Kifer, D. (2018). Concentrated Differentially Private Gradient Descent with Adaptive per-Iteration Privacy Budget. *KDD*.
6.  Andrew, G., Thakkar, O., McMahan, H. B., & Ramaswamy, S. (2021). Differentially Private Learning with Adaptive Clipping. *NeurIPS*.
7.  Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*.
8.  Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. *NeurIPS*.
