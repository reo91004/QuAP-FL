# Changelog

이 파일은 QuAP-FL 프로젝트의 모든 주목할만한 변경사항을 기록한다.

형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 기반으로 하며,
이 프로젝트는 [Semantic Versioning](https://semver.org/lang/ko/)을 준수한다.

## [Unreleased]

### 계획 중
- 비동기 클라이언트 업데이트 지원
- Byzantine 공격 방어 메커니즘
- Rényi DP 지원
- 추가 데이터셋 (FEMNIST, Shakespeare)

## [1.1.2] - 2025-10-03 (진행 중)

### Changed
- **하이퍼파라미터 재조정 시도**
  - `learning_rate`: 0.005 → 0.0005 (Client drift 완화 목적)
  - `epsilon_total`: 6.0 → 3.0 (논문 목표값 복원)
  - `max_clip`: 5.0 → 1.0 (DP-SGD 표준 범위 적용)

### Added
- **Aggregated gradient norm 제한 파라미터**
  - `config/hyperparameters.py`: `max_agg_norm` 파라미터 추가 (10000)
  - `framework/server.py`: 하드코딩된 값 (100)을 config 기반으로 변경
  - 노이즈 추가 후 gradient의 큰 norm을 고려한 설정

- **하이퍼파라미터 검증 로깅**
  - `framework/server.py`: 초기화 시 핵심 파라미터 출력
  - learning_rate, max_clip, epsilon_total, epsilon_base, max_agg_norm 값 확인 가능

### Issues
- **현재 남은 문제**: Learning rate와 privacy budget 간 불균형
  - Learning rate 0.0005는 epsilon=3.0 대비 부족 (Noise/Signal ratio = 44:1)
  - Round 0에서 Loss 폭발 (3366), Round 4부터 NaN 재발생
  - 근본 원인: Gradient signal (0.07)에 비해 DP noise (3.08)가 과다
  - 해결 방안 검토 중: learning rate 증가 vs epsilon 증가 vs local epochs 증가

### Notes
이 버전의 하이퍼파라미터는 아직 검증되지 않았다. 프라이버시 예산과 학습 효율성 간의 균형을 찾기 위한 실험이 진행 중이다.

## [1.1.1] - 2025-10-01

### Fixed
- **치명적 버그 수정**: 그래디언트 방향 계산 오류 수정
  - `framework/server.py`: `global - local` → `local - global`로 수정
  - 이전 구현은 최적화의 반대 방향으로 업데이트되어 학습이 실패하거나 발산하는 문제 발생

- **Loss 함수 호환성 문제 해결**
  - `framework/server.py`: MNIST 모델이 `log_softmax`를 출력하는데 `CrossEntropyLoss` 사용하여 이중 log 적용 문제
  - `nn.CrossEntropyLoss()` → `nn.NLLLoss()`로 변경

- **수치적 안정성 개선**
  - `framework/quantile_clipping.py`: NaN/Inf gradient norm 필터링 추가
  - `framework/server.py`: aggregated gradient의 NaN/Inf 체크 및 norm 제한 (100) 추가
  - 그래디언트 폭발 방지를 위한 안전장치 추가

### Changed
- **모델 업데이트 로직 간소화**
  - `framework/server.py`: learning rate가 로컬 학습에서 이미 적용되므로 전역 업데이트에서 중복 적용 제거
  - `param.data -= grad_tensor` → `param.data += grad_tensor`로 변경 (그래디언트가 이미 델타이므로)

### Impact
- Loss 폭발 문제 해결 (559,684,661,647 → 정상 범위)
- NaN 전파 문제 해결
- 정확도 개선 (0.098 고정 → 정상 학습)

## [1.1.0] - 2025-01-20

### Added
- **시각화 시스템**: 학습 결과 자동 시각화 기능 추가
  - `utils/visualization.py` 모듈 생성
  - `plot_training_history()`: 단일 실험 4-subplot 시각화
  - `plot_multi_seed_comparison()`: 다중 시드 비교 시각화
  - `generate_summary_table()`: tabulate 기반 결과 테이블
- **설정 확장**: `config/hyperparameters.py`에 시각화 관련 설정 추가
  - `VISUALIZATION_CONFIG`: 시각화 옵션 (enabled, dpi, format, style 등)
  - `OUTPUT_CONFIG`: 출력 디렉토리 및 저장 옵션
- **의존성 추가**: `requirements.txt`에 tabulate, seaborn 추가

### Changed
- `main.py`: 학습 완료 후 자동으로 시각화 생성 및 결과 테이블 출력
- `aggregate_results.py`: JSON 형식 지원 및 다중 시드 비교 시각화 추가
- 모든 시각화 옵션이 config 기반으로 통합됨 (명령줄 인자 불필요)

### Features
- **4-subplot 시각화**:
  - Test Accuracy over Rounds (목표선 포함)
  - Test Loss over Rounds (log scale)
  - Privacy Budget Consumption (총 예산선 포함)
  - Adaptive Clipping Values (평균선 포함)
- **다중 시드 비교**:
  - Accuracy Mean ± Std 범위 표시
  - Loss 비교 (log scale)
  - Privacy Budget 분포 히스토그램
  - Final Accuracy 분포 히스토그램
- **결과 테이블**:
  - Final/Best Accuracy, Loss
  - Privacy metrics
  - 목표 달성 여부 표시
  - Participation statistics

### Documentation
- `docs/API.md`: visualization 모듈 API 문서화
- `docs/EXPERIMENTS.md`: 시각화 사용법 섹션 추가
- `README.md`: 시각화 기능 소개 추가

## [1.0.0] - 2025-01-15

### Added
- QuAP-FL 초기 릴리스
- 참여 이력 기반 적응형 프라이버시 예산 할당
- 90th percentile 기반 분위수 클리핑
- Dirichlet(α=0.5) 기반 Non-IID 데이터 분할
- MNIST 및 CIFAR-10 벤치마크 지원
- 완전한 단위 테스트 및 검증 프레임워크
- 다중 시드 실험 결과 집계 도구

### Features
- **ParticipationTracker**: 클라이언트 참여 이력 추적
- **AdaptivePrivacyAllocator**: 참여율 기반 동적 프라이버시 예산
- **QuantileClipper**: EMA 기반 적응형 그래디언트 클리핑
- **QuAPFLServer**: 9단계 연합학습 메인 루프

### Performance
- MNIST: 97.1% ± 0.5% accuracy (ε=3.0)
- CIFAR-10: 81.2% ± 0.9% accuracy (ε=3.0)
- Vanilla DP-FL 대비 35-50% 빠른 수렴

### Documentation
- 완전한 README.md
- API 문서
- 아키텍처 가이드
- 실험 재현 가이드

### Testing
- 4개 핵심 검증 테스트
- 3개 모듈별 단위 테스트 스위트
- 중간 체크포인트 검증

## [0.2.0] - 2025-01-10

### Added
- CIFAR-10 지원
- 분위수 클리핑 EMA 안정화
- 다중 시드 실험 결과 집계

### Changed
- 프라이버시 예산 함수 α=0.5, β=2.0로 고정
- 학습률 decay 0.99로 조정

### Fixed
- 클리핑 값 초기화 버그 수정
- 참여율 계산 정밀도 개선

## [0.1.0] - 2025-01-05

### Added
- 프로젝트 초기 구조
- MNIST 기본 구현
- 참여 추적 메커니즘
- 기본 프라이버시 예산 할당

---

## 버전 규칙

### Major (x.0.0)
- 호환성이 깨지는 API 변경
- 알고리즘 핵심 로직 변경

### Minor (0.x.0)
- 하위 호환성을 유지하는 기능 추가
- 새로운 데이터셋 지원
- 성능 개선

### Patch (0.0.x)
- 버그 수정
- 문서 업데이트
- 작은 성능 최적화

---

## 링크

[Unreleased]: https://github.com/your-username/quap-fl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-username/quap-fl/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/your-username/quap-fl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/your-username/quap-fl/releases/tag/v0.1.0