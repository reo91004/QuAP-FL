# API 문서

QuAP-FL의 주요 API 문서다. 각 모듈의 핵심 클래스와 함수를 설명한다.

## 목차

- [Framework](#framework)
  - [ParticipationTracker](#participationtracker)
  - [AdaptivePrivacyAllocator](#adaptiveprivacyallocator)
  - [QuantileClipper](#quantileclipper)
  - [QuAPFLServer](#quapflserver)
- [Models](#models)
- [Data](#data)
- [Config](#config)

---

## Framework

### ParticipationTracker

클라이언트 참여 이력을 추적하는 클래스.

#### 초기화

```python
from framework import ParticipationTracker

tracker = ParticipationTracker(num_clients=100)
```

**파라미터:**
- `num_clients` (int): 전체 클라이언트 수

#### 메소드

##### `update(participating_clients: List[int])`

참여 기록 업데이트.

```python
tracker.update([0, 1, 2])  # 클라이언트 0, 1, 2가 참여
```

**파라미터:**
- `participating_clients` (List[int]): 이번 라운드 참여 클라이언트 ID 리스트

**예외:**
- `ValueError`: 잘못된 클라이언트 ID

##### `get_participation_rate(client_id: int) -> float`

특정 클라이언트의 참여율 반환.

```python
rate = tracker.get_participation_rate(0)
print(f"클라이언트 0의 참여율: {rate:.2%}")
```

**파라미터:**
- `client_id` (int): 클라이언트 ID

**반환:**
- `float`: 참여율 (0.0 ~ 1.0)

##### `get_all_participation_rates() -> np.ndarray`

모든 클라이언트의 참여율 반환.

```python
rates = tracker.get_all_participation_rates()
print(f"평균 참여율: {rates.mean():.2%}")
```

**반환:**
- `np.ndarray`: 참여율 배열 (shape: num_clients)

##### `get_statistics() -> dict`

참여 통계 정보 반환.

```python
stats = tracker.get_statistics()
print(f"총 라운드: {stats['total_rounds']}")
print(f"평균 참여율: {stats['mean_participation_rate']:.3f}")
```

**반환:**
- `dict`: 통계 딕셔너리
  - `total_rounds`: 총 라운드 수
  - `mean_participation_rate`: 평균 참여율
  - `std_participation_rate`: 참여율 표준편차
  - `participating_clients`: 참여한 클라이언트 수
  - `never_participated`: 한 번도 참여하지 않은 클라이언트 수

---

### AdaptivePrivacyAllocator

참여율 기반 적응형 프라이버시 예산 할당.

#### 초기화

```python
from framework import AdaptivePrivacyAllocator

allocator = AdaptivePrivacyAllocator(
    epsilon_base=0.015,
    alpha=0.5,
    beta=2.0,
    delta=1e-5
)
```

**파라미터:**
- `epsilon_base` (float): 기본 프라이버시 예산 (ε_total / num_rounds)
- `alpha` (float): 적응 강도 (0 ~ 1), 기본값: 0.5
- `beta` (float): 감소율 파라미터 (> 0), 기본값: 2.0
- `delta` (float): 차분 프라이버시 파라미터, 기본값: 1e-5

#### 메소드

##### `compute_privacy_budget(participation_rate: float) -> float`

참여율에 따른 프라이버시 예산 계산.

**수식:** `ε_i = ε_base * (1 + α * exp(-β * p_i))`

```python
# 자주 참여하는 클라이언트 (작은 예산)
eps_high = allocator.compute_privacy_budget(0.9)

# 드물게 참여하는 클라이언트 (큰 예산)
eps_low = allocator.compute_privacy_budget(0.1)

print(f"높은 참여율: ε = {eps_high:.6f}")
print(f"낮은 참여율: ε = {eps_low:.6f}")
```

**파라미터:**
- `participation_rate` (float): 참여율 (0.0 ~ 1.0)

**반환:**
- `float`: 적응형 프라이버시 예산

##### `compute_noise_multiplier(epsilon: float, clip_norm: float, delta: Optional[float] = None) -> float`

Gaussian mechanism 노이즈 승수 계산.

**수식:** `σ = C * sqrt(2 * ln(1.25/δ)) / ε`

```python
noise_std = allocator.compute_noise_multiplier(
    epsilon=0.015,
    clip_norm=1.0
)
print(f"노이즈 표준편차: {noise_std:.2f}")
```

**파라미터:**
- `epsilon` (float): 프라이버시 예산
- `clip_norm` (float): 클리핑 norm 값
- `delta` (Optional[float]): 차분 프라이버시 파라미터 (None이면 초기화 값 사용)

**반환:**
- `float`: 노이즈 표준편차

##### `add_gaussian_noise(data: np.ndarray, epsilon: float, clip_norm: float) -> np.ndarray`

데이터에 가우시안 노이즈 추가.

```python
gradient = np.random.randn(100)
noisy_gradient = allocator.add_gaussian_noise(
    gradient,
    epsilon=0.015,
    clip_norm=1.0
)
```

**파라미터:**
- `data` (np.ndarray): 입력 데이터 (이미 클리핑된 그래디언트)
- `epsilon` (float): 프라이버시 예산
- `clip_norm` (float): 클리핑 norm 값

**반환:**
- `np.ndarray`: 노이즈가 추가된 데이터

---

### QuantileClipper

90th percentile 기반 적응형 그래디언트 클리핑.

#### 초기화

```python
from framework import QuantileClipper

clipper = QuantileClipper(
    quantile=0.9,
    momentum=0.95,
    min_clip=0.1,
    max_clip=10.0
)
```

**파라미터:**
- `quantile` (float): 클리핑에 사용할 분위수, 기본값: 0.9
- `momentum` (float): EMA momentum, 기본값: 0.95
- `min_clip` (float): 최소 클리핑 값, 기본값: 0.1
- `max_clip` (float): 최대 클리핑 값, 기본값: 10.0

#### 메소드

##### `update_clip_value(gradients_list: List[np.ndarray]) -> float`

그래디언트 norm의 분위수 기반 클리핑 값 업데이트.

```python
gradients = [np.random.randn(100) for _ in range(10)]
clip_value = clipper.update_clip_value(gradients)
print(f"현재 클리핑 값: {clip_value:.4f}")
```

**파라미터:**
- `gradients_list` (List[np.ndarray]): 이번 라운드 모든 클라이언트의 그래디언트

**반환:**
- `float`: 업데이트된 클리핑 값

##### `clip_gradient(gradient: np.ndarray) -> np.ndarray`

단일 그래디언트 클리핑.

```python
gradient = np.random.randn(100) * 10
clipped = clipper.clip_gradient(gradient)

print(f"원본 norm: {np.linalg.norm(gradient):.4f}")
print(f"클리핑 후 norm: {np.linalg.norm(clipped):.4f}")
```

**파라미터:**
- `gradient` (np.ndarray): 입력 그래디언트

**반환:**
- `np.ndarray`: 클리핑된 그래디언트

##### `clip_gradients(gradients_list: List[np.ndarray]) -> List[np.ndarray]`

여러 그래디언트를 동시에 클리핑.

```python
gradients = [np.random.randn(100) * i for i in range(1, 11)]
clipped_gradients = clipper.clip_gradients(gradients)
```

**파라미터:**
- `gradients_list` (List[np.ndarray]): 그래디언트 리스트

**반환:**
- `List[np.ndarray]`: 클리핑된 그래디언트 리스트

---

### QuAPFLServer

QuAP-FL 연합학습 서버.

#### 초기화

```python
from framework import QuAPFLServer
from models import MNISTModel
from data import prepare_non_iid_data

# 데이터 준비
client_indices, train_dataset, test_dataset = prepare_non_iid_data(
    'mnist', num_clients=100, alpha=0.5
)

# 모델 초기화
model = MNISTModel()

# 서버 초기화
server = QuAPFLServer(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    client_indices=client_indices,
    dataset_name='mnist',
    config=None,  # 기본 설정 사용
    device=None   # 자동 선택
)
```

**파라미터:**
- `model` (nn.Module): 전역 모델
- `train_dataset`: 학습 데이터셋
- `test_dataset`: 테스트 데이터셋
- `client_indices` (List[List[int]]): 각 클라이언트의 데이터 인덱스
- `dataset_name` (str): 'mnist' or 'cifar10'
- `config` (Optional[Dict]): 하이퍼파라미터 (None이면 기본값 사용)
- `device` (Optional[torch.device]): 디바이스 (None이면 자동 선택)

#### 메소드

##### `train() -> dict`

메인 학습 루프 실행.

```python
history = server.train()

print(f"최종 정확도: {history['test_accuracy'][-1]:.4f}")
print(f"최종 손실: {history['test_loss'][-1]:.4f}")
```

**반환:**
- `dict`: 학습 이력
  - `test_accuracy`: 테스트 정확도 리스트
  - `test_loss`: 테스트 손실 리스트
  - `clip_values`: 클리핑 값 이력
  - `noise_levels`: 노이즈 레벨 이력
  - `participation_stats`: 참여 통계 이력
  - `privacy_budgets`: 프라이버시 예산 이력

##### `evaluate() -> tuple`

전역 모델 평가.

```python
accuracy, loss = server.evaluate()
print(f"정확도: {accuracy:.4f}, 손실: {loss:.4f}")
```

**반환:**
- `tuple`: (accuracy, loss)

---

## Models

### MNISTModel

MNIST 분류를 위한 CNN 모델.

```python
from models import MNISTModel

model = MNISTModel()
print(f"파라미터 수: {model.get_num_parameters():,}")
```

**아키텍처:**
- Conv2d(1, 32, 3, 1)
- Conv2d(32, 64, 3, 1)
- Dropout2d(0.25)
- FC(9216, 128)
- Dropout(0.5)
- FC(128, 10)

### CIFAR10Model

CIFAR-10 분류를 위한 CNN 모델.

```python
from models import CIFAR10Model

model = CIFAR10Model()
print(f"파라미터 수: {model.get_num_parameters():,}")
```

**아키텍처:**
- Conv2d(3, 64, 3, padding=1)
- MaxPool2d(2, 2)
- Conv2d(64, 128, 3, padding=1)
- MaxPool2d(2, 2)
- Conv2d(128, 256, 3, padding=1)
- MaxPool2d(2, 2)
- FC(256*4*4, 256)
- Dropout(0.5)
- FC(256, 128)
- FC(128, 10)

---

## Data

### prepare_non_iid_data

Dirichlet 분포 기반 Non-IID 데이터 분할.

```python
from data import prepare_non_iid_data

client_indices, train_dataset, test_dataset = prepare_non_iid_data(
    dataset_name='mnist',
    num_clients=100,
    alpha=0.5,
    data_dir='./data',
    seed=42
)
```

**파라미터:**
- `dataset_name` (str): 'mnist' or 'cifar10'
- `num_clients` (int): 전체 클라이언트 수
- `alpha` (float): Dirichlet 분포 파라미터, 기본값: 0.5
- `data_dir` (str): 데이터 저장 경로, 기본값: './data'
- `seed` (Optional[int]): 랜덤 시드

**반환:**
- `tuple`: (client_indices, train_dataset, test_dataset)

---

## Config

### HYPERPARAMETERS

모든 하이퍼파라미터 설정.

```python
from config import HYPERPARAMETERS

print(f"클라이언트 수: {HYPERPARAMETERS['num_clients']}")
print(f"라운드 수: {HYPERPARAMETERS['num_rounds']}")
print(f"프라이버시 예산: {HYPERPARAMETERS['epsilon_total']}")
```

**주요 파라미터:**
- `num_clients`: 100
- `num_rounds`: 200
- `clients_per_round`: 30
- `local_epochs`: 5
- `local_batch_size`: 32
- `learning_rate`: 0.01
- `epsilon_total`: 3.0
- `alpha`: 0.5 (적응 강도)
- `beta`: 2.0 (감소율)

### TARGET_ACCURACY

목표 정확도 설정.

```python
from config import TARGET_ACCURACY

print(f"MNIST 목표: {TARGET_ACCURACY['mnist']:.1%}")
print(f"CIFAR-10 목표: {TARGET_ACCURACY['cifar10']:.1%}")
```

---

## 예제

### 전체 학습 파이프라인

```python
import torch
from models import MNISTModel
from data import prepare_non_iid_data
from framework import QuAPFLServer

# 시드 고정
torch.manual_seed(42)

# 데이터 준비
client_indices, train_dataset, test_dataset = prepare_non_iid_data(
    'mnist', num_clients=100, alpha=0.5, seed=42
)

# 모델 초기화
model = MNISTModel()

# 서버 초기화
server = QuAPFLServer(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    client_indices=client_indices,
    dataset_name='mnist'
)

# 학습
history = server.train()

# 결과 출력
final_acc = history['test_accuracy'][-1]
print(f"최종 정확도: {final_acc:.4f}")
```