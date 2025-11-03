# QuAP-FL: Quantile-based Adaptive Privacy for Federated Learning

클라이언트 참여 패턴이 불규칙한 현실 환경에서 차분 프라이버시와 모델 성능을 동시에 만족시키기 위한 연합학습 프레임워크입니다. QuAP-FL은 참여 이력을 기반으로 프라이버시 예산을 동적으로 조절하고, 마지막 분류 레이어에만 노이즈를 부여해 고차원 노이즈 문제를 해결합니다.

---

## 핵심 아이디어

- **참여 이력 추적**  
  `ParticipationTracker`는 각 클라이언트의 참여 횟수를 기록하고 참여율을 계산합니다.

- **적응형 프라이버시 예산**  
  ```text
  ε_i(t) = ε_base × (1 + α · exp(-β · p_i(t)))
  ```
  자주 참여할수록 강한 프라이버시(작은 ε), 드물게 참여할수록 완화된 프라이버시(큰 ε)를 부여합니다.

- **Layer-wise Differential Privacy**  
  마지막 분류 레이어(critical layer)에만 노이즈를 주입해 유효한 신호를 보존합니다. 클리핑 또한 critical 구간에 한정해 적용합니다.

- **90th Percentile 분위수 클리핑**  
  매 라운드 그래디언트의 90 분위수 기반으로 클리핑 임계값을 업데이트하고 EMA(momentum=0.95)로 안정화합니다.

- **혼합형 클라이언트 샘플링**  
  Beta(2,5) 분포로 생성한 가중치와 균등 분포를 혼합(`participation_mix=0.8`)해 다양한 참여 패턴을 시뮬레이션합니다.

---

## 기본 환경 (config/hyperparameters.py)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `num_clients` | 100 | 총 클라이언트 수 |
| `num_rounds` | 200 | 연합 라운드 수 |
| `clients_per_round` | 30 | 라운드당 참여 클라이언트 (30%) |
| `local_epochs` | 3 | 로컬 학습 에폭 |
| `local_batch_size` | 32 | 배치 크기 |
| `learning_rate` | 0.005 | 로컬 학습률 |
| `lr_decay` | 0.995 | 라운드별 학습률 감쇠 |
| `epsilon_total` | 6.0 | 전체 프라이버시 예산 |
| `delta` | 1e-5 | 차분 프라이버시 δ |
| `clip_quantile` | 0.9 | 분위수 클리핑 |
| `clip_momentum` | 0.95 | 클리핑 EMA |
| `noise_strategy` | `layer_wise` | 마지막 레이어에만 노이즈 추가 |
| `participation_mix` | 0.8 | 가중 샘플링 vs 균등 샘플링 비율 |

---

## 설치

```bash
git clone https://github.com/your-username/quap-fl.git
cd quap-fl
pip install -r requirements.txt
```

---

## 빠른 시작

1. **구현 검증**
   ```bash
   python main.py --validate_only
   ```
   핵심 컴포넌트 단위 테스트(참여율, 예산, 클리핑, 노이즈 계산)를 수행합니다.

2. **MNIST 학습 예시**
   ```bash
   python main.py --dataset mnist --seed 42
   ```
   학습 중 주요 로그는 콘솔/파일에 기록되며, 종료 후 `results/`에 JSON·PNG·LOG 파일이 생성됩니다.

3. **결과 집계**
   ```bash
   python aggregate_results.py --dataset mnist
   ```
   여러 시드 결과를 `tabulate` 표와 비교 그래프로 요약합니다.

4. **테스트**
   ```bash
   python -m pytest tests/test_tracker.py tests/test_privacy.py tests/test_clipping.py
   python -m pytest tests/test_layer_wise_dp.py
   python -m pytest tests/test_integration.py   # 약 6~7분 소요 (MNIST 20라운드)
   ```

---

## 기록되는 히스토리

`main.py`는 학습 종료 시 다음 필드를 포함한 JSON을 저장합니다.

- `test_accuracy`, `test_loss`: 평가 라운드별 지표
- `train_loss`: 라운드 평균 로컬 손실
- `clip_values`: 분위수 클리핑 값
- `privacy_budgets`: 라운드별 평균 ε
- `noise_levels`: 라운드별 노이즈 σ
- `noise_stats`: 노이즈/신호 노름 통계
- `participation_stats`: 참여율 평균·표준편차·미참여자 수 등
- `evaluation_rounds`: 평가 시점 라운드 인덱스

`utils/visualization.plot_training_history`는 위 정보를 이용해 4분할 그래프(Accuracy, Loss, Cumulative ε, Clip Value)를 렌더링합니다.

---

## 프로젝트 구조

```
quap-fl/
├── aggregate_results.py          # 다중 시드 집계 스크립트
├── config/
│   └── hyperparameters.py        # 기본 설정
├── data/
│   └── data_utils.py             # Dirichlet 기반 데이터 분할
├── framework/
│   ├── adaptive_privacy.py       # 적응형 프라이버시 예산
│   ├── participation_tracker.py  # 참여 이력 추적
│   ├── quantile_clipping.py      # 분위수 클리핑
│   └── server.py                 # QuAPFLServer (학습 루프)
├── docs/
│   └── PAPER.md                  # 연구 노트 / 논문 초안
├── main.py                       # CLI 실행 스크립트
├── models/
│   ├── mnist_model.py            # MNIST CNN
│   └── cifar_model.py            # CIFAR-10 CNN
├── tests/
│   ├── test_clipping.py
│   ├── test_integration.py
│   ├── test_layer_wise_dp.py
│   ├── test_privacy.py
│   └── test_tracker.py
├── utils/
│   ├── validation.py             # validate_only 진입점
│   └── visualization.py          # 시각화 & 결과 테이블
└── requirements.txt
```

---

## 기대 성능

통합 테스트(`tests/test_integration.py`)는 CPU 기준 약 6분 내외로 수행되며, 20라운드 만에 다음 조건을 검증합니다.

- 정확도 ≥ 70%
- 손실 ≤ 1.0
- critical layer 파라미터 수 = 1,290

풀 스케일 실험(200라운드, ε=6.0)은 아래 수준을 목표로 합니다.

| 데이터셋 | 목표 정확도 | 프라이버시 예산 |
|----------|-------------|------------------|
| MNIST    | 93.2% ± 0.6% | ε = 6.0 |
| CIFAR-10 | 76.8% ± 1.2% | ε = 6.0 |

---

## 자주 묻는 질문

**Q. 하이퍼파라미터를 바꿔도 되나요?**  
A. 연구 목적이라면 가능합니다. 다만 논문 재현 시에는 기본값을 유지하세요.

**Q. 시각화를 끄고 싶어요.**  
A. `config.hyperparameters.VISUALIZATION_CONFIG['enabled'] = False`로 설정하면 됩니다.

**Q. 학습 도중 로그가 부족해요.**  
A. 각 라운드마다 평균 train loss, ε, clip 값이 출력되며, 10라운드마다 평가 결과가 기록됩니다. 필요한 경우 `framework/server.py`에서 로깅을 확장하세요.

---

## 라이선스 및 문의

- 라이선스: MIT
- 문의: GitHub Issues 또는 your.email@example.com

---

QuAP-FL은 “참여율 기반 적응형 프라이버시 + Layer-wise DP”라는 단순하지만 강력한 조합을 실현한 연구용 레퍼런스 구현입니다. 코드 기반 실험이나 후속 연구를 위한 출발점으로 활용해 주세요.
