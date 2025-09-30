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