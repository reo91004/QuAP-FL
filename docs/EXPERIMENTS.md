# 실험 가이드

QuAP-FL 논문 결과를 재현하기 위한 상세 실험 가이드다.

## 목차

- [빠른 시작](#빠른-시작)
- [환경 설정](#환경-설정)
- [기본 실험](#기본-실험)
- [고급 실험](#고급-실험)
- [결과 분석](#결과-분석)
- [트러블슈팅](#트러블슈팅)

---

## 빠른 시작

### 5분 튜토리얼

```bash
# 1. 구현 검증
python main.py --validate_only

# 2. MNIST 실험 (약 30분)
python main.py --dataset mnist --seed 42

# 3. 결과 확인
ls results/
```

### 예상 출력

```
============================================================
QuAP-FL Training Start: MNIST
Target Accuracy: 0.971
Clients: 100, Rounds: 200
============================================================

Training: 100%|████████████████████| 200/200

Round 0: Acc=0.4230, Loss=1.9234, Clip=1.2340
✓ Round 10: 0.678 (정상 범위: 0.650-0.700)
...
Round 200: Acc=0.9710, Loss=0.0921, Clip=0.5210
✓ Round 200: 0.971 (정상 범위: 0.965-0.976)

목표 성능 달성! Round 200
```

---

## 환경 설정

### 하드웨어 요구사항

**최소 사양**:
- CPU: 4 cores
- RAM: 8GB
- Disk: 10GB

**권장 사양**:
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with CUDA (선택적)
- Disk: 20GB+

### 소프트웨어 요구사항

```bash
# Python 버전
python --version  # 3.8+ 필요

# 의존성 설치
pip install -r requirements.txt

# GPU 지원 (선택적)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 데이터셋 다운로드

첫 실행 시 자동으로 다운로드된다:

```bash
# MNIST: ~11MB
# CIFAR-10: ~170MB

# 수동 다운로드 (선택적)
python -c "from torchvision import datasets; \
    datasets.MNIST('./data', download=True); \
    datasets.CIFAR10('./data', download=True)"
```

---

## 기본 실험

### 실험 1: MNIST 기본 설정

**목표**: MNIST에서 97.1% 정확도 달성

```bash
# 단일 시드
python main.py --dataset mnist --seed 42

# 예상 시간: CPU 30분, GPU 10분
# 예상 결과: 97.1% ± 0.5%
```

**중간 체크포인트**:
- Round 10: 65-70%
- Round 50: 88-92%
- Round 100: 94-96%
- Round 200: 96.5-97.6%

### 실험 2: CIFAR-10 기본 설정

**목표**: CIFAR-10에서 81.2% 정확도 달성

```bash
# 단일 시드
python main.py --dataset cifar10 --seed 42

# 예상 시간: CPU 90분, GPU 30분
# 예상 결과: 81.2% ± 0.9%
```

**중간 체크포인트**:
- Round 10: 35-40%
- Round 50: 65-70%
- Round 100: 74-78%
- Round 200: 80.3-82.1%

### 실험 3: 다중 시드 (재현성 확인)

**목표**: 결과의 일관성 검증

```bash
# 3개 시드로 실험
python main.py --dataset mnist --seed 42
python main.py --dataset mnist --seed 123
python main.py --dataset mnist --seed 999

# 결과 집계
python aggregate_results.py --dataset mnist
```

**예상 출력**:
```
============================================================
결과 집계: MNIST
============================================================
시드: [42, 123, 999]
실험 수: 3

정확도:
  - 평균: 0.9710
  - 표준편차: 0.0042
  - 최소: 0.9678
  - 최대: 0.9752
  - 95% 신뢰구간: [0.9628, 0.9792]

목표 달성률:
  - 목표: 0.971
  - 달성률: 100.0% (3/3)
```

---

## 고급 실험

### 실험 4: 프라이버시 예산 변화

**목표**: 프라이버시-유틸리티 트레이드오프 분석

```bash
# ε=1 (강한 프라이버시)
python main.py --dataset mnist --seed 42 \
    --num_rounds 300  # 더 많은 라운드 필요

# ε=5 (약한 프라이버시)
python main.py --dataset mnist --seed 42 \
    --num_rounds 150  # 더 적은 라운드 가능

# ε=10 (매우 약한 프라이버시)
python main.py --dataset mnist --seed 42 \
    --num_rounds 100
```

**프라이버시 예산 수정**:
```python
# config/hyperparameters.py
HYPERPARAMETERS = {
    'epsilon_total': 1.0,  # 기본값 3.0에서 변경
    # ...
}
```

### 실험 5: 클라이언트 수 변화

**목표**: 확장성 분석

```bash
# 50 클라이언트
python main.py --dataset mnist --seed 42 --num_clients 50

# 200 클라이언트
python main.py --dataset mnist --seed 42 --num_clients 200

# 500 클라이언트
python main.py --dataset mnist --seed 42 --num_clients 500
```

### 실험 6: Non-IID 정도 변화

**목표**: 데이터 이질성 영향 분석

```python
# data/data_utils.py에서 alpha 값 변경
# alpha=0.1: 매우 Non-IID
# alpha=0.5: 중간 (기본값)
# alpha=1.0: 약한 Non-IID
# alpha=10.0: 거의 IID

client_indices, train_dataset, test_dataset = prepare_non_iid_data(
    'mnist', num_clients=100, alpha=0.1  # 여기 변경
)
```

```bash
# 실험 실행
python experiments/non_iid_experiments.py
```

### 실험 7: 적응형 vs 균일 프라이버시

**목표**: QuAP-FL의 효과 검증

**균일 프라이버시 (Baseline)**:
```python
# framework/adaptive_privacy.py 수정
def compute_privacy_budget(self, participation_rate: float) -> float:
    # 적응형 비활성화
    return self.epsilon_base  # 항상 동일한 예산
```

**비교 실험**:
```bash
# Baseline
python main.py --dataset mnist --seed 42

# QuAP-FL (원래 설정 복원)
python main.py --dataset mnist --seed 42
```

---

## 결과 분석

### 자동 시각화 (권장)

기본적으로 `main.py`는 학습 완료 후 자동으로 시각화를 생성한다.

```bash
# 실험 실행 (시각화 자동 생성)
python main.py --dataset mnist --seed 42

# 생성되는 파일들:
# - results/mnist_seed42_TIMESTAMP.log        (로그)
# - results/mnist_seed42_TIMESTAMP.json       (결과)
# - results/mnist_seed42_TIMESTAMP.png        (시각화)
```

**생성되는 4-subplot 시각화:**
1. **Test Accuracy over Rounds**: 정확도 변화 + 목표선
2. **Test Loss over Rounds**: 손실 변화 (log scale)
3. **Privacy Budget Consumption**: 누적 프라이버시 예산 + 총 예산선
4. **Adaptive Clipping Values**: 클리핑 값 변화 + 평균선

### 다중 시드 비교 시각화

```bash
# 여러 시드로 실험
python main.py --dataset mnist --seed 42
python main.py --dataset mnist --seed 123
python main.py --dataset mnist --seed 999

# 비교 시각화 자동 생성
python aggregate_results.py --dataset mnist

# 생성되는 파일:
# - results/mnist_multi_seed_comparison.png
```

**생성되는 4-subplot 비교 시각화:**
1. **Test Accuracy**: Mean ± Std 범위 + 개별 궤적
2. **Test Loss**: Mean 궤적 + 개별 궤적 (log scale)
3. **Privacy Budget Distribution**: 전체 예산 분포 히스토그램
4. **Final Accuracy Distribution**: 최종 정확도 분포 + 목표선

### 결과 테이블

학습 완료 후 자동으로 출력되는 결과 테이블:

```
======================================================================
QuAP-FL Experiment Results - MNIST (seed=42)
======================================================================
╒═════════════════════════╤═════════════════════╤════════════════════╕
│ Metric                  │ Value               │ Status             │
╞═════════════════════════╪═════════════════════╪════════════════════╡
│ Final Accuracy          │ 0.9710 (97.10%)     │ Achieved           │
│ Best Accuracy           │ 0.9752 (97.52%)     │ Round 190          │
│ Final Loss              │ 0.0921              │ -                  │
│ Total Privacy Budget    │ ε = 3.0             │ -                  │
│ Average Clipping        │ 0.5210              │ -                  │
│ Mean Participation Rate │ 0.300               │ 30 clients         │
╘═════════════════════════╧═════════════════════╧════════════════════╛
======================================================================
```

### 시각화 설정 커스터마이징

`config/hyperparameters.py`에서 시각화 옵션 변경:

```python
VISUALIZATION_CONFIG = {
    'enabled': True,           # 시각화 생성 (False로 설정하면 비활성화)
    'show_plot': False,        # 화면 표시 (True로 설정하면 창으로 표시)
    'dpi': 300,                # 해상도 (논문용: 300, 일반: 150)
    'format': 'pdf',           # 형식 (png, pdf, svg)
    'style': 'seaborn-v0_8',   # matplotlib 스타일
    'figsize': (14, 12),       # Figure 크기
    'include_table': True,     # 결과 테이블 출력
}
```

### 수동 시각화 (고급 사용자)

필요시 직접 시각화를 생성할 수 있다:

```python
import json
from utils import plot_training_history, plot_multi_seed_comparison
from config import VISUALIZATION_CONFIG

# 결과 로드
with open('results/mnist_seed42_TIMESTAMP.json', 'r') as f:
    data = json.load(f)
    history = data['history']

# 시각화 생성
plot_path = plot_training_history(
    history=history,
    dataset_name='mnist',
    seed=42,
    config=VISUALIZATION_CONFIG,
    save_path='custom_plot.png'
)
print(f"시각화 저장: {plot_path}")
```

### 참여율 분석

```python
# 참여 통계 확인
final_stats = history['participation_stats'][-1]

print(f"평균 참여율: {final_stats['mean_participation_rate']:.3f}")
print(f"참여율 표준편차: {final_stats['std_participation_rate']:.3f}")
print(f"참여 클라이언트: {final_stats['participating_clients']}")
print(f"한 번도 참여 안 함: {final_stats['never_participated']}")
```

---

## 트러블슈팅

### 문제 1: 정확도가 목표에 못 미침

**증상**:
```
Round 200: Acc=0.9350, Loss=0.1821
✗ 목표 미달성 (목표: 0.971)
```

**진단**:
```python
# 클리핑 값 확인
print(f"평균 클리핑 값: {np.mean(history['clip_values']):.4f}")
# 예상: 0.5-0.8

# 노이즈 레벨 확인
print(f"평균 노이즈: {np.mean(history['noise_levels']):.2f}")
# 예상: 300-350
```

**해결책**:

1. **클리핑 값이 너무 작음 (<0.3)**:
```python
# config/hyperparameters.py
HYPERPARAMETERS = {
    'min_clip': 0.5,  # 0.1에서 증가
}
```

2. **노이즈가 너무 큼 (>500)**:
```python
HYPERPARAMETERS = {
    'epsilon_total': 5.0,  # 3.0에서 증가 (프라이버시 완화)
}
```

3. **학습률 decay 확인**:
```python
# framework/server.py
lr = 0.01 * (0.99 ** round_t)  # 확인
```

### 문제 2: 메모리 부족

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결책**:

1. **배치 크기 감소**:
```python
HYPERPARAMETERS = {
    'local_batch_size': 16,  # 32에서 감소
}
```

2. **CPU 사용**:
```bash
python main.py --dataset mnist --seed 42 --gpu -1  # CPU 강제
```

3. **그래디언트 체크포인팅** (고급):
```python
# 모델에 gradient checkpointing 추가
torch.utils.checkpoint.checkpoint(model, input)
```

### 문제 3: 수렴이 느림

**증상**:
```
Round 200: Acc=0.8500, Loss=0.4321
```

**진단**:
```python
# 초기 라운드 확인
print(f"Round 10 정확도: {history['test_accuracy'][1]:.4f}")
# 예상: 0.65-0.70
```

**해결책**:

1. **학습률 증가**:
```python
HYPERPARAMETERS = {
    'learning_rate': 0.02,  # 0.01에서 증가
}
```

2. **로컬 에폭 증가**:
```python
HYPERPARAMETERS = {
    'local_epochs': 7,  # 5에서 증가
}
```

3. **참여 클라이언트 증가**:
```python
HYPERPARAMETERS = {
    'clients_per_round': 50,  # 30에서 증가
}
```

### 문제 4: 검증 테스트 실패

**증상**:
```
✗ Test 2: 적응형 프라이버시 예산 실패
```

**해결책**:

1. **의존성 버전 확인**:
```bash
pip list | grep numpy
pip list | grep torch
```

### 문제 5: Loss 폭발 및 NaN (v1.1.1에서 해결)

**증상**:
```
Round 0: Acc=0.098, Loss=559684661647, Clip=0.2779
Round 10: Acc=0.098, Loss=nan, Clip=nan
```

**원인 분석**:
1. 그래디언트 방향이 반대로 계산됨 (`global - local` 대신 `local - global`)
2. MNIST 모델의 log_softmax 출력과 CrossEntropyLoss 간 불일치
3. NaN gradient norm 필터링 부재

**해결책 (v1.1.1 적용)**:

1. **그래디언트 방향 수정**:
```python
# framework/server.py:local_training()
diff = local_param.data - global_param.data  # 올바른 방향
```

2. **Loss 함수 호환성**:
```python
# framework/server.py:local_training(), evaluate()
criterion = nn.NLLLoss()  # log_softmax와 호환
```

3. **NaN/Inf 체크 추가**:
```python
# framework/quantile_clipping.py:update_clip_value()
grad_norms = grad_norms[~np.isnan(grad_norms) & ~np.isinf(grad_norms)]

# framework/server.py:update_global_model()
if np.any(np.isnan(aggregated_gradient)):
    print("Warning: NaN detected, skipping update")
    return
```

4. **Gradient norm 제한**:
```python
# framework/server.py:update_global_model()
if grad_norm > 100:
    aggregated_gradient = aggregated_gradient * (100 / grad_norm)
```

### 문제 6: 학습이 진행되지 않음 (v1.1.1에서 해결)

**증상**:
```
200 라운드 동안 정확도 0.098 (랜덤 추측 수준) 고정
```

**원인**:
모델 업데이트시 learning rate 중복 적용 및 델타 방향 오류

**해결책**:
```python
# framework/server.py:update_global_model()
# gradient는 이미 (local - global) 델타이므로 직접 더함
param.data += grad_tensor  # -= 대신 +=
```

2. **클린 재설치**:
```bash
pip uninstall -y torch torchvision numpy
pip install -r requirements.txt
```

3. **Python 버전 확인**:
```bash
python --version  # 3.8+ 필요
```

---

## 배치 실험

### 실험 스크립트 작성

```bash
# experiments/run_all.sh
#!/bin/bash

# MNIST 실험
for seed in 42 123 999 777 888; do
    echo "Running MNIST with seed $seed"
    python main.py --dataset mnist --seed $seed
done

# CIFAR-10 실험
for seed in 42 123 999; do
    echo "Running CIFAR-10 with seed $seed"
    python main.py --dataset cifar10 --seed $seed
done

# 결과 집계
python aggregate_results.py --dataset mnist
python aggregate_results.py --dataset cifar10

echo "All experiments completed!"
```

```bash
# 실행
chmod +x experiments/run_all.sh
./experiments/run_all.sh
```

### 병렬 실행 (고급)

```bash
# experiments/run_parallel.sh
#!/bin/bash

# GPU 0에서 MNIST
CUDA_VISIBLE_DEVICES=0 python main.py --dataset mnist --seed 42 &

# GPU 1에서 CIFAR-10
CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --seed 42 &

wait
echo "Parallel experiments completed!"
```

---

## 결과 보고

### 표 형식 결과

```python
# experiments/generate_table.py
import numpy as np

datasets = ['mnist', 'cifar10']
seeds = [42, 123, 999]

for dataset in datasets:
    accuracies = []
    for seed in seeds:
        result = np.load(f'results/results_{dataset}_seed{seed}.npy',
                        allow_pickle=True).item()
        accuracies.append(result['test_accuracy'][-1])

    mean = np.mean(accuracies)
    std = np.std(accuracies)

    print(f"{dataset.upper()}: {mean:.4f} ± {std:.4f}")
```

### LaTeX 표 생성

```python
# LaTeX 표 출력
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"Dataset & Accuracy & Privacy Budget \\")
print(r"\midrule")
for dataset in datasets:
    print(f"    {dataset.upper()} & {mean:.1%} $\\pm$ {std:.1%} & $\\epsilon=3.0$ \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{QuAP-FL Performance}")
print(r"\end{table}")
```

---

## 추가 리소스

- **논문**: [링크]
- **이슈 트래커**: https://github.com/your-username/quap-fl/issues
- **토론**: https://github.com/your-username/quap-fl/discussions

## 질문

실험 관련 질문은 GitHub Issues에 올려주세요.