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

### 학습 곡선 시각화

```python
# experiments/visualize_results.py
import numpy as np
import matplotlib.pyplot as plt

# 결과 로드
history = np.load('results/results_mnist_seed42.npy', allow_pickle=True).item()

# 정확도 곡선
plt.figure(figsize=(10, 6))
plt.plot(history['test_accuracy'], label='Test Accuracy')
plt.xlabel('Evaluation Step (every 10 rounds)')
plt.ylabel('Accuracy')
plt.title('QuAP-FL Learning Curve (MNIST)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('learning_curve.png', dpi=300)
```

### 프라이버시 예산 분포

```python
# 프라이버시 예산 히스토그램
budgets = history['privacy_budgets']

plt.figure(figsize=(10, 6))
plt.hist(budgets, bins=50, alpha=0.7)
plt.xlabel('Privacy Budget (ε)')
plt.ylabel('Frequency')
plt.title('Privacy Budget Distribution')
plt.savefig('privacy_distribution.png', dpi=300)
```

### 참여율 분석

```python
# 참여율 통계
final_stats = history['participation_stats'][-1]

print(f"평균 참여율: {final_stats['mean_participation_rate']:.3f}")
print(f"참여율 표준편차: {final_stats['std_participation_rate']:.3f}")
print(f"참여 클라이언트: {final_stats['participating_clients']}")
```

### 클리핑 값 추이

```python
# 클리핑 값 변화
clip_values = history['clip_values']

plt.figure(figsize=(10, 6))
plt.plot(clip_values, label='Clip Value')
plt.xlabel('Round')
plt.ylabel('Clip Value')
plt.title('Adaptive Clipping Value Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('clip_values.png', dpi=300)
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