# QuAP-FL: 이질적 시스템 환경에서의 연합학습을 위한 분위수 기반 적응형 프라이버시 프레임워크

**QuAP-FL (Quantile-based Adaptive Privacy for Federated Learning)** 은 클라이언트 간 참여 빈도의 불균형(System Heterogeneity)이 존재하는 현실적인 연합학습 환경을 위한 프레임워크입니다. 참여 이력을 기반으로 프라이버시 예산을 동적으로 조절하고, 그래디언트 분포에 따라 클리핑 임계값을 적응적으로 변경하여 '프라이버시-유틸리티 딜레마'를 해결합니다.

---

## 핵심 아이디어 (Key Contributions)

1. **참여율 기반 적응형 프라이버시 예산 할당 (Adaptive Privacy Budgeting)**
   - 클라이언트의 누적 참여 이력을 추적하여, 참여율이 높을수록 작은 예산(강한 보호)을, 참여율이 낮을수록 큰 예산(높은 유틸리티)을 할당합니다.
   - **수식**: $\epsilon_i(t) = \epsilon_{base} \times (1 + \alpha \cdot \exp(-\beta \cdot p_i(t)))$

2. **분위수 기반 적응형 클리핑 (Quantile-based Adaptive Clipping)**
   - 매 라운드 수집된 그래디언트 Norm의 90분위수(90th Percentile)를 기준으로 클리핑 임계값을 동적으로 조절합니다.
   - 정보 손실을 최소화하면서 이상치(Outlier)를 효과적으로 제어합니다.

3. **계층별 노이즈 주입 (Layer-wise Noise Injection)**
   - 고차원 모델의 '차원의 저주'를 피하기 위해, 민감한 정보가 집중된 마지막 분류 레이어(Classifier)에만 노이즈를 주입합니다.

---

## 실험 결과 (Experimental Results)

Non-IID 데이터 분포(Dirichlet $\alpha=0.5$)와 이질적 참여 패턴(Beta(2, 5)) 하에서 수행된 실험 결과입니다.

| Dataset | FedAvg (No Privacy) | Fixed-DP (Baseline) | QuAP-FL (Ours) |
|---------|---------------------|---------------------|----------------|
| **MNIST** | 93.26% | 93.76% | **93.30%** |
| **CIFAR-10** | 76.54% | 75.80% | **76.82%** |

- **정규화 효과**: 적절한 DP 노이즈가 과적합을 방지하여 Non-IID 환경에서 일반화 성능을 높임을 확인했습니다.
- **성능 우위**: 복잡한 데이터셋(CIFAR-10)에서 QuAP-FL은 고정형 DP 대비 유의미한 성능 향상을 달성했습니다.

---

## 설치 (Installation)

```bash
git clone https://github.com/your-username/quap-fl.git
cd quap-fl
pip install -r requirements.txt
```

---

## 빠른 시작 (Quick Start)

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

## 프로젝트 구조 (Project Structure)

```
quap-fl/
├── aggregate_results.py          # 다중 시드 집계 스크립트
├── config/
│   └── hyperparameters.py        # 기본 설정 (Hyperparameters)
├── data/
│   └── data_utils.py             # Dirichlet 기반 데이터 분할
├── framework/
│   ├── adaptive_privacy.py       # 적응형 프라이버시 예산 (Adaptive Budgeting)
│   ├── participation_tracker.py  # 참여 이력 추적 (Participation Tracker)
│   ├── quantile_clipping.py      # 분위수 클리핑 (Quantile Clipping)
│   └── server.py                 # QuAPFLServer (학습 루프)
├── docs/
│   ├── PAPER.tex                 # 논문 (LaTeX)
│   └── REVIEW_REPORT.md          # 논문 검토 보고서
├── main.py                       # CLI 실행 스크립트
├── models/
│   ├── mnist_model.py            # MNIST CNN
│   └── cifar_model.py            # CIFAR-10 CNN
├── results/                      # 실험 결과 및 로그
├── tests/                        # 단위 및 통합 테스트
└── requirements.txt
```

---

## 라이선스 및 문의

- **License**: MIT
- **Contact**: GitHub Issues
