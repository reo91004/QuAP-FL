# QuAP-FL: Quantile-based Adaptive Privacy for Federated Learning

클라이언트 참여 패턴 기반 적응형 프라이버시 예산 할당을 통한 연합학습 최적화

## 개요

QuAP-FL은 연합학습에서 클라이언트의 불규칙한 참여 패턴을 고려한 적응형 차분 프라이버시 프레임워크다. 핵심 혁신은 클라이언트 참여 이력을 추적하여 **참여 빈도 기반 차등 프라이버시 예산 할당**을 수행하는 것이다.

### 주요 특징

- **적응형 프라이버시 예산**: 참여율에 반비례하는 동적 예산 할당
- **분위수 기반 클리핑**: 90th percentile 기반 적응형 그래디언트 클리핑
- **자동 시각화**: 학습 곡선, 프라이버시 예산, 클리핑 값 등 4-subplot 자동 생성
- **단순한 구현**: 2,500줄 이하 코드, 학부생 수준에서도 재현 가능

### 성능 목표

| 데이터셋 | 목표 정확도 | 프라이버시 예산 |
|---------|-----------|---------------|
| MNIST | 97.1% ± 0.5% | ε = 3.0 |
| CIFAR-10 | 81.2% ± 0.9% | ε = 3.0 |

## 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/quap-fl.git
cd quap-fl

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 1. 구현 검증

먼저 구현이 올바른지 검증한다:

```bash
python main.py --validate_only
```

예상 출력:
```
✓ Test 1: 참여율 계산 통과
✓ Test 2: 적응형 프라이버시 예산 통과
✓ Test 3: 분위수 클리핑 통과
✓ Test 4: 노이즈 계산 통과

✓ 모든 검증 테스트 통과!
```

### 2. MNIST 실험

```bash
# 단일 시드
python main.py --dataset mnist --seed 42

# 여러 시드로 실험 (재현성 확인)
python main.py --dataset mnist --seed 42
python main.py --dataset mnist --seed 123
python main.py --dataset mnist --seed 999

# 결과 집계
python aggregate_results.py --dataset mnist
```

### 3. CIFAR-10 실험

```bash
# 단일 시드
python main.py --dataset cifar10 --seed 42

# 여러 시드로 실험
python main.py --dataset cifar10 --seed 42
python main.py --dataset cifar10 --seed 123
python main.py --dataset cifar10 --seed 999

# 결과 집계
python aggregate_results.py --dataset cifar10
```

### 4. 단위 테스트

```bash
# 모든 테스트 실행
python -m pytest tests/

# 개별 테스트
python tests/test_tracker.py
python tests/test_privacy.py
python tests/test_clipping.py
```

## 프로젝트 구조

```
quap-fl/
├── framework/              # 핵심 알고리즘
│   ├── participation_tracker.py    # 참여 이력 추적
│   ├── adaptive_privacy.py         # 적응형 프라이버시 예산
│   ├── quantile_clipping.py        # 분위수 클리핑
│   └── server.py                   # QuAPFLServer 메인 로직
├── models/                 # 신경망 모델
│   ├── mnist_model.py             # MNIST CNN
│   └── cifar_model.py             # CIFAR-10 CNN
├── data/                  # 데이터 처리
│   └── data_utils.py              # Non-IID 분할 (Dirichlet α=0.5)
├── config/                # 설정
│   └── hyperparameters.py         # 모든 고정값, 시각화 설정
├── utils/                 # 유틸리티
│   ├── validation.py              # 검증 테스트
│   └── visualization.py           # 자동 시각화 및 결과 테이블
├── tests/                 # 단위 테스트
│   ├── test_tracker.py
│   ├── test_privacy.py
│   └── test_clipping.py
├── main.py                        # 메인 실행 스크립트
├── aggregate_results.py           # 결과 집계
└── requirements.txt
```

## 핵심 알고리즘

### 적응형 프라이버시 예산 함수

```
ε_i(t) = ε_base * (1 + α * exp(-β * p_i(t)))
```

- `ε_base`: 기본 프라이버시 예산 (ε_total / num_rounds)
- `α = 0.5`: 적응 강도 (고정값)
- `β = 2.0`: 감소율 파라미터 (고정값)
- `p_i(t)`: 클라이언트 i의 참여율

### 학습 루프 (9단계)

1. 클라이언트 선택
2. 참여 이력 업데이트 (노이즈 전에!)
3. 로컬 학습
4. 클리핑 값 업데이트 (90th percentile)
5. 클리핑 적용
6. 적응형 노이즈 추가
7. 연합 평균
8. 모델 업데이트
9. 평가

## 하이퍼파라미터 (고정값)

이 값들을 변경하면 논문 결과를 재현할 수 없다:

```python
HYPERPARAMETERS = {
    'num_clients': 100,
    'num_rounds': 200,
    'clients_per_round': 30,  # 30% 참여
    'local_epochs': 5,
    'local_batch_size': 32,
    'learning_rate': 0.01,
    'lr_decay': 0.99,

    'epsilon_total': 3.0,
    'delta': 1e-5,

    'alpha': 0.5,  # 적응 강도 (절대 변경 금지)
    'beta': 2.0,   # 감소율 (절대 변경 금지)
    'clip_quantile': 0.9,
    'clip_momentum': 0.95,
}
```

## 중간 체크포인트

구현이 올바른지 확인하기 위한 중간 정확도 범위:

### MNIST

| 라운드 | 예상 정확도 범위 |
|-------|----------------|
| 10 | 65-70% |
| 50 | 88-92% |
| 100 | 94-96% |
| 200 | 96.5-97.6% |

### CIFAR-10

| 라운드 | 예상 정확도 범위 |
|-------|----------------|
| 10 | 35-40% |
| 50 | 65-70% |
| 100 | 74-78% |
| 200 | 80.3-82.1% |

## 예상 출력

```
============================================================
QuAP-FL Training Start: MNIST
Target Accuracy: 0.971
Clients: 100, Rounds: 200
============================================================

Training: 100%|████████████████████| 200/200

Round 0: Acc=0.4230, Loss=1.9234, Clip=1.2340
✓ Round 10: 0.678 (정상 범위: 0.650-0.700)
Round 20: Acc=0.8120, Loss=0.5432, Clip=0.9870
...
Round 190: Acc=0.9690, Loss=0.0987, Clip=0.5430
Round 200: Acc=0.9710, Loss=0.0921, Clip=0.5210
✓ Round 200: 0.971 (정상 범위: 0.965-0.976)

목표 성능 달성! Round 200

============================================================
최종 결과: Accuracy=0.9710, Loss=0.0921
============================================================
```

## 시각화

### 자동 생성되는 결과물

실험 실행 후 자동으로 생성되는 파일들:

```
results/
├── mnist_seed42_TIMESTAMP.log      # 학습 로그
├── mnist_seed42_TIMESTAMP.json     # 결과 데이터
└── mnist_seed42_TIMESTAMP.png      # 4-subplot 시각화
```

### 4-Subplot 시각화

자동 생성되는 시각화는 다음을 포함한다:

1. **Test Accuracy over Rounds**
   - 라운드별 정확도 변화
   - 목표 정확도선 표시

2. **Test Loss over Rounds**
   - 라운드별 손실 변화 (log scale)
   - 학습 안정성 확인

3. **Privacy Budget Consumption**
   - 누적 프라이버시 예산 (ε)
   - 총 예산선 표시

4. **Adaptive Clipping Values**
   - 라운드별 클리핑 임계값 변화
   - 평균 클리핑 값 표시

### 결과 테이블

학습 완료 후 자동으로 출력되는 요약 테이블:

```
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
```

### 다중 시드 비교

여러 시드 실험 후 결과 집계 시 비교 시각화도 자동 생성된다:

```bash
python aggregate_results.py --dataset mnist
# → results/mnist_multi_seed_comparison.png
```

비교 시각화는 다음을 포함:
- Accuracy Mean ± Std 범위
- Loss 비교 (log scale)
- Privacy Budget 분포 히스토그램
- Final Accuracy 분포 히스토그램

### 시각화 비활성화

필요시 `config/hyperparameters.py`에서 비활성화 가능:

```python
VISUALIZATION_CONFIG = {
    'enabled': False,  # 시각화 끄기
    # ...
}
```

## 트러블슈팅

### 정확도가 목표에 못 미침

1. 클리핑 값 확인 (0.5 이상이어야 함)
2. 노이즈 스케일 확인 (너무 크면 학습 안됨)
3. 학습률 decay 적용 확인

### 수렴이 너무 느림

- 학습률이 너무 작을 수 있음
- 로컬 에폭 수 확인 (5 에폭)
- 배치 크기 확인 (32)

## 라이선스

MIT License

## 인용

```bibtex
@article{quap-fl-2025,
  title={QuAP-FL: Quantile-based Adaptive Privacy for Federated Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 연락처

- Email: your.email@example.com
- GitHub Issues: https://github.com/your-username/quap-fl/issues