# 아키텍처 문서

QuAP-FL의 시스템 아키텍처와 설계 원칙을 설명한다.

## 목차

- [전체 구조](#전체-구조)
- [핵심 컴포넌트](#핵심-컴포넌트)
- [데이터 흐름](#데이터-흐름)
- [설계 원칙](#설계-원칙)
- [확장 포인트](#확장-포인트)

---

## 전체 구조

QuAP-FL은 모듈화된 아키텍처를 채택하여 각 컴포넌트가 독립적으로 동작하고 테스트 가능하도록 설계되었다.

```
┌─────────────────────────────────────────────────────────┐
│                      Main Script                         │
│                       (main.py)                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   QuAPFLServer                           │
│              (framework/server.py)                       │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  9-Step Training Loop                             │  │
│  │  1. Client Selection                              │  │
│  │  2. Participation Update                          │  │
│  │  3. Local Training                                │  │
│  │  4. Clip Value Update                             │  │
│  │  5. Gradient Clipping                             │  │
│  │  6. Adaptive Noise Addition                       │  │
│  │  7. Federated Averaging                           │  │
│  │  8. Model Update                                  │  │
│  │  9. Evaluation                                    │  │
│  └───────────────────────────────────────────────────┘  │
└──┬──────────────┬──────────────┬─────────────┬──────────┘
   │              │              │             │
   ▼              ▼              ▼             ▼
┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│Particip.│  │Adaptive │  │Quantile  │  │  Models  │
│Tracker  │  │Privacy  │  │Clipper   │  │ (MNIST,  │
│         │  │Allocator│  │          │  │ CIFAR10) │
└─────────┘  └─────────┘  └──────────┘  └──────────┘
```

---

## 핵심 컴포넌트

### 1. ParticipationTracker

**역할**: 클라이언트 참여 이력 추적

**책임**:
- 각 클라이언트의 참여 횟수 기록
- 참여율 계산
- 참여 통계 제공

**설계 특징**:
- 단순 카운터 기반 (O(1) 업데이트)
- float64 정밀도로 정확성 보장
- 상태 초기화 지원

**인터페이스**:
```python
class ParticipationTracker:
    def update(self, participating_clients: List[int]) -> None
    def get_participation_rate(self, client_id: int) -> float
    def get_statistics(self) -> dict
    def reset(self) -> None
```

### 2. AdaptivePrivacyAllocator

**역할**: 참여율 기반 적응형 프라이버시 예산 할당

**책임**:
- 참여율에 따른 프라이버시 예산 계산
- 가우시안 노이즈 스케일 계산
- 누적 프라이버시 손실 추적

**핵심 알고리즘**:
```
ε_i(t) = ε_base * (1 + α * exp(-β * p_i(t)))

where:
- ε_base: 기본 프라이버시 예산
- α: 적응 강도 (0.5 고정)
- β: 감소율 (2.0 고정)
- p_i(t): 클라이언트 i의 참여율
```

**설계 특징**:
- 수학적 정확성 보장
- 범위 검증으로 안정성 확보
- Gaussian mechanism 표준 구현

**인터페이스**:
```python
class AdaptivePrivacyAllocator:
    def compute_privacy_budget(self, participation_rate: float) -> float
    def compute_noise_multiplier(self, epsilon: float, clip_norm: float) -> float
    def add_gaussian_noise(self, data: np.ndarray, epsilon: float, clip_norm: float) -> np.ndarray
```

### 3. QuantileClipper

**역할**: 분위수 기반 적응형 그래디언트 클리핑

**책임**:
- 그래디언트 norm 계산
- 90th percentile 기반 클리핑 값 결정
- EMA를 통한 안정화

**핵심 알고리즘**:
```
clip_t = momentum * clip_{t-1} + (1 - momentum) * quantile_90(norms_t)

where:
- momentum: 0.95 (EMA 계수)
- quantile_90: 90번째 백분위수
```

**설계 특징**:
- 극단값 제한 (min_clip, max_clip)
- EMA로 급격한 변화 방지
- 이력 추적으로 디버깅 지원

**인터페이스**:
```python
class QuantileClipper:
    def update_clip_value(self, gradients_list: List[np.ndarray]) -> float
    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray
    def clip_gradients(self, gradients_list: List[np.ndarray]) -> List[np.ndarray]
```

### 4. QuAPFLServer

**역할**: 연합학습 메인 조율자

**책임**:
- 클라이언트 선택
- 로컬 학습 관리
- 그래디언트 집계
- 모델 업데이트
- 평가 및 모니터링

**9단계 학습 루프**:
```python
for round_t in range(num_rounds):
    # 1. Client Selection
    selected = self.select_clients(round_t)

    # 2. Participation Update (BEFORE noise!)
    self.tracker.update(selected)

    # 3. Local Training
    gradients = [self.local_training(cid, round_t) for cid in selected]

    # 4. Clip Value Update
    clip_val = self.clipper.update_clip_value(gradients)

    # 5. Gradient Clipping
    clipped = self.clipper.clip_gradients(gradients)

    # 6. Adaptive Noise Addition
    noisy = []
    for i, cid in enumerate(selected):
        p_rate = self.tracker.get_participation_rate(cid)
        eps = self.privacy_allocator.compute_privacy_budget(p_rate)
        noisy_grad = self.privacy_allocator.add_gaussian_noise(
            clipped[i], eps, clip_val
        )
        noisy.append(noisy_grad)

    # 7. Federated Averaging
    global_grad = np.mean(noisy, axis=0)

    # 8. Model Update
    self.update_global_model(global_grad)

    # 9. Evaluation
    if round_t % 10 == 0:
        self.evaluate()
```

**설계 특징**:
- 순서 엄격 준수 (특히 2번과 6번)
- 상태 추적 및 로깅
- 조기 종료 지원

---

## 데이터 흐름

### 학습 데이터 흐름

```
1. Data Preparation
   └─> prepare_non_iid_data()
       └─> Dirichlet(α=0.5) split
           └─> client_indices

2. Client Selection
   └─> QuAPFLServer.select_clients()
       └─> selected_clients

3. Local Training
   └─> QuAPFLServer.local_training()
       └─> client_dataloader
           └─> local_model.train()
               └─> local_gradient

4. Gradient Processing
   └─> QuantileClipper.update_clip_value()
   └─> QuantileClipper.clip_gradients()
   └─> AdaptivePrivacyAllocator.add_gaussian_noise()
       └─> noisy_gradient

5. Aggregation
   └─> np.mean(noisy_gradients)
       └─> global_gradient

6. Model Update
   └─> QuAPFLServer.update_global_model()
       └─> model.parameters() += global_gradient  # gradient는 이미 (local - global) 델타
```

### 프라이버시 예산 흐름

```
Round t
   │
   ├─> ParticipationTracker.update()
   │   └─> participation_count[i] += 1
   │
   ├─> ParticipationTracker.get_participation_rate(i)
   │   └─> p_i = count[i] / total_rounds
   │
   ├─> AdaptivePrivacyAllocator.compute_privacy_budget(p_i)
   │   └─> ε_i = ε_base * (1 + α * exp(-β * p_i))
   │
   └─> AdaptivePrivacyAllocator.compute_noise_multiplier(ε_i)
       └─> σ = C * sqrt(2 * ln(1.25/δ)) / ε_i
```

---

## 설계 원칙

### 1. 모듈화 (Modularity)

각 컴포넌트는 단일 책임을 가지며 독립적으로 테스트 가능하다.

**예시**:
- `ParticipationTracker`: 오직 참여 추적만
- `AdaptivePrivacyAllocator`: 오직 프라이버시 계산만
- `QuantileClipper`: 오직 클리핑만

### 2. 명시성 (Explicitness)

암묵적 동작보다 명시적 동작을 선호한다.

**예시**:
```python
# 나쁜 예: 암묵적 상태 변경
tracker.process(clients)  # 무엇을 하는지 불명확

# 좋은 예: 명시적 동작
tracker.update(clients)  # 참여 기록 업데이트
rate = tracker.get_participation_rate(client_id)  # 참여율 조회
```

### 3. 불변성 (Immutability)

가능한 경우 불변 데이터 구조를 사용한다.

**예시**:
```python
# 설정은 읽기 전용
HYPERPARAMETERS = {
    'alpha': 0.5,  # 변경 금지
    'beta': 2.0,   # 변경 금지
}
```

### 4. 실패 빠름 (Fail Fast)

오류는 가능한 빨리 발견되어야 한다.

**예시**:
```python
def compute_privacy_budget(self, participation_rate: float) -> float:
    if not (0 <= participation_rate <= 1):
        raise ValueError(f"참여율은 [0, 1] 범위여야 함: {participation_rate}")
    # ...
```

### 5. 테스트 가능성 (Testability)

모든 컴포넌트는 단위 테스트 가능해야 한다.

**예시**:
```python
# 의존성 주입으로 테스트 용이성 확보
def __init__(self, tracker, privacy_allocator, clipper):
    self.tracker = tracker  # 테스트 시 mock 가능
    self.privacy_allocator = privacy_allocator
    self.clipper = clipper
```

---

## 확장 포인트

### 1. 새로운 프라이버시 메커니즘

`AdaptivePrivacyAllocator`를 상속하여 새로운 프라이버시 메커니즘 구현:

```python
class RenyiDPAllocator(AdaptivePrivacyAllocator):
    def compute_privacy_budget(self, participation_rate: float) -> float:
        # Rényi DP 구현
        pass
```

### 2. 새로운 클리핑 전략

`QuantileClipper`를 상속하여 새로운 클리핑 전략 구현:

```python
class AdaptiveQuantileClipper(QuantileClipper):
    def update_clip_value(self, gradients_list: List[np.ndarray]) -> float:
        # 동적 분위수 조정
        pass
```

### 3. 새로운 클라이언트 선택 전략

`QuAPFLServer.select_clients()`를 오버라이드:

```python
class CustomQuAPFLServer(QuAPFLServer):
    def select_clients(self, round_t: int) -> List[int]:
        # 손실 기반 선택
        # 데이터 품질 기반 선택
        pass
```

### 4. 새로운 데이터셋

`data/data_utils.py`에 새로운 데이터셋 추가:

```python
def prepare_non_iid_data(dataset_name, ...):
    if dataset_name == 'femnist':
        # FEMNIST 로드
        pass
    elif dataset_name == 'shakespeare':
        # Shakespeare 로드
        pass
```

### 5. 새로운 모델

`models/`에 새로운 모델 추가:

```python
# models/resnet_model.py
class ResNetModel(nn.Module):
    def __init__(self):
        # ResNet 구현
        pass
```

---

## 성능 고려사항

### 1. 메모리 효율성

- 그래디언트는 flatten된 1D 배열로 처리
- 배치 단위 처리로 메모리 사용 최소화
- 불필요한 복사 방지

### 2. 계산 효율성

- NumPy vectorization 활용
- 불필요한 재계산 방지 (클리핑 값 캐싱)
- EMA로 평활화 (매 라운드 완전 재계산 방지)

### 3. 통신 효율성

- 모델 파라미터가 아닌 그래디언트 전송
- 압축 가능성 (향후 확장)

---

## 보안 고려사항

### 1. 프라이버시

- Gaussian mechanism으로 차분 프라이버시 보장
- 참여율 기반 적응형 예산으로 공정성 확보
- 누적 프라이버시 손실 추적

### 2. 안정성

- 입력 검증으로 잘못된 값 방지
- 범위 제한으로 극단값 처리
- 상태 일관성 검증

### 3. 재현성

- 시드 고정 지원
- 결정적 알고리즘 사용
- 중간 상태 로깅

---

## 향후 개선 방향

### 단기 (1-3개월)

1. **Byzantine 방어**: 악의적 클라이언트 탐지
2. **압축**: 통신 효율성 개선
3. **비동기**: 클라이언트 비동기 업데이트

### 중기 (3-6개월)

1. **Rényi DP**: 더 정밀한 프라이버시 분석
2. **개인화**: 클라이언트별 맞춤 모델
3. **연합 최적화**: 서버 측 모멘텀, Adam 등

### 장기 (6-12개월)

1. **크로스 디바이스**: 대규모 모바일 배포
2. **크로스 사일로**: 기관 간 협력 학습

---

## 버전 1.1.1 주요 수정사항

### 그래디언트 계산 수정

**문제**: 로컬 학습 후 그래디언트 방향이 반대로 계산됨
```python
# 잘못된 구현 (v1.1.0 이전)
diff = global_param.data - local_param.data  # 최적화 반대 방향!

# 수정된 구현 (v1.1.1)
diff = local_param.data - global_param.data  # 올바른 방향
```

### 모델 업데이트 로직 수정

**문제**: Learning rate 중복 적용
```python
# 잘못된 구현 (v1.1.0 이전)
param.data -= grad_tensor * learning_rate * decay

# 수정된 구현 (v1.1.1)
param.data += grad_tensor  # gradient는 이미 (local - global) 델타
```

### Loss 함수 호환성

**문제**: MNIST 모델이 log_softmax를 출력하는데 CrossEntropyLoss 사용
```python
# 잘못된 구현 (v1.1.0 이전)
criterion = nn.CrossEntropyLoss()  # 이중 log 적용!

# 수정된 구현 (v1.1.1)
criterion = nn.NLLLoss()  # log_softmax와 호환
```

### 수치적 안정성 개선

- **QuantileClipper**: NaN/Inf gradient norm 필터링 추가
- **QuAPFLServer**: Aggregated gradient norm 제한 추가
  - v1.1.1: 하드코딩된 값 100 사용
  - v1.1.2: `max_agg_norm` 파라미터로 config 기반 관리 (기본값 10000)
- **결과**: Loss 폭발 (5.6e11 → 정상 범위) 및 NaN 전파 해결 (v1.1.1)

### v1.1.2 아키텍처 개선

**1. 하이퍼파라미터 중앙화**

모든 하이퍼파라미터가 `config/hyperparameters.py`에서 관리되도록 개선:
```python
HYPERPARAMETERS = {
    # 기존 파라미터들...

    # v1.1.2에서 추가
    'max_agg_norm': 10000,  # Aggregated gradient norm 제한
}
```

**2. 하드코딩 제거**

`framework/server.py:update_global_model()`:
```python
# Before (v1.1.1)
if grad_norm > 100:  # 하드코딩
    aggregated_gradient = aggregated_gradient * (100 / grad_norm)

# After (v1.1.2)
max_agg_norm = self.config.get('max_agg_norm', 100)
if grad_norm > max_agg_norm:
    aggregated_gradient = aggregated_gradient * (max_agg_norm / grad_norm)
```

**3. 하이퍼파라미터 검증 로깅**

`QuAPFLServer.__init__()` 시점에 핵심 파라미터 출력:
```python
self._log("=" * 60)
self._log("핵심 하이퍼파라미터 검증")
self._log("=" * 60)
self._log(f"  learning_rate: {self.config['learning_rate']}")
self._log(f"  max_clip: {self.config['max_clip']}")
self._log(f"  epsilon_total: {self.config['epsilon_total']}")
self._log(f"  epsilon_base: {epsilon_base:.6f}")
self._log(f"  max_agg_norm: {self.config.get('max_agg_norm', 100)}")
self._log("=" * 60)
```

이를 통해 실험 시작 시 config 값이 올바르게 전달되었는지 즉시 확인 가능하다.