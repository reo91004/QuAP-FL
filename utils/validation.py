# utils/validation.py

"""
구현 검증 및 중간 체크포인트 확인

논문 재현을 위한 필수 검증 테스트
"""

import numpy as np
import logging
import sys
import os
from typing import Optional

# 상위 디렉토리 import를 위한 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework.participation_tracker import ParticipationTracker
from framework.adaptive_privacy import AdaptivePrivacyAllocator
from framework.quantile_clipping import QuantileClipper
from config.hyperparameters import EXPECTED_MILESTONES


def validate_implementation() -> bool:
    """
    구현이 올바른지 검증하는 테스트
    모든 테스트를 통과해야 논문 재현 가능

    Returns:
        True if all tests passed, False otherwise
    """
    tests_passed = []
    tests_failed = []

    print("=" * 60)
    print("QuAP-FL 구현 검증 시작")
    print("=" * 60)

    # Test 1: 참여율 계산
    try:
        tracker = ParticipationTracker(10)
        tracker.update([0, 1, 2])
        tracker.update([1, 2, 3])

        rate_1 = tracker.get_participation_rate(1)
        rate_0 = tracker.get_participation_rate(0)

        assert abs(rate_1 - 1.0) < 1e-6, f"Expected 1.0, got {rate_1}"
        assert abs(rate_0 - 0.5) < 1e-6, f"Expected 0.5, got {rate_0}"

        tests_passed.append("Test 1: 참여율 계산")
        print("✓ Test 1: 참여율 계산 통과")
    except Exception as e:
        tests_failed.append(f"Test 1: 참여율 계산 - {str(e)}")
        print(f"✗ Test 1: 참여율 계산 실패 - {str(e)}")

    # Test 2: 적응형 프라이버시 예산
    try:
        allocator = AdaptivePrivacyAllocator(epsilon_base=0.015, alpha=0.5, beta=2.0)

        eps_high_part = allocator.compute_privacy_budget(0.9)  # 자주 참여
        eps_low_part = allocator.compute_privacy_budget(0.1)   # 드물게 참여

        assert eps_low_part > eps_high_part, \
            f"낮은 참여율이 더 많은 예산을 받아야 함: {eps_low_part} > {eps_high_part}"

        # 예상 범위 확인
        expected_high = 0.015 * (1 + 0.5 * np.exp(-2 * 0.9))
        expected_low = 0.015 * (1 + 0.5 * np.exp(-2 * 0.1))

        assert abs(eps_high_part - expected_high) < 1e-6, \
            f"Expected {expected_high}, got {eps_high_part}"
        assert abs(eps_low_part - expected_low) < 1e-6, \
            f"Expected {expected_low}, got {eps_low_part}"

        tests_passed.append("Test 2: 적응형 프라이버시 예산")
        print("✓ Test 2: 적응형 프라이버시 예산 통과")
        print(f"  - 높은 참여율 (0.9): ε = {eps_high_part:.6f}")
        print(f"  - 낮은 참여율 (0.1): ε = {eps_low_part:.6f}")
    except Exception as e:
        tests_failed.append(f"Test 2: 적응형 프라이버시 예산 - {str(e)}")
        print(f"✗ Test 2: 적응형 프라이버시 예산 실패 - {str(e)}")

    # Test 3: 분위수 클리핑
    try:
        clipper = QuantileClipper(quantile=0.9, momentum=0.95, min_clip=0.1, max_clip=10.0)

        # 그래디언트 생성 (norm이 1~10)
        grads = [np.random.randn(100) * i for i in range(1, 11)]

        clip_val = clipper.update_clip_value(grads)

        assert 0.1 <= clip_val <= 10.0, f"클리핑 값이 범위를 벗어남: {clip_val}"

        # 클리핑 적용 테스트
        large_grad = np.random.randn(100) * 100  # 큰 그래디언트
        clipped_grad = clipper.clip_gradient(large_grad)
        clipped_norm = np.linalg.norm(clipped_grad.flatten())

        assert clipped_norm <= clip_val + 1e-6, \
            f"클리핑 후 norm ({clipped_norm})이 클리핑 값 ({clip_val})을 초과"

        tests_passed.append("Test 3: 분위수 클리핑")
        print("✓ Test 3: 분위수 클리핑 통과")
        print(f"  - 클리핑 값: {clip_val:.4f}")
        print(f"  - 클리핑 후 norm: {clipped_norm:.4f}")
    except Exception as e:
        tests_failed.append(f"Test 3: 분위수 클리핑 - {str(e)}")
        print(f"✗ Test 3: 분위수 클리핑 실패 - {str(e)}")

    # Test 4: 노이즈 계산
    try:
        allocator = AdaptivePrivacyAllocator(epsilon_base=0.015, alpha=0.5, beta=2.0, delta=1e-5)

        epsilon = 0.015
        clip_norm = 1.0
        noise_mult = allocator.compute_noise_multiplier(epsilon, clip_norm)

        assert noise_mult > 0, f"노이즈 승수는 양수여야 함: {noise_mult}"

        # 예상 값 계산
        expected = clip_norm * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
        assert abs(noise_mult - expected) < 1e-6, \
            f"Expected {expected}, got {noise_mult}"

        tests_passed.append("Test 4: 노이즈 계산")
        print("✓ Test 4: 노이즈 계산 통과")
        print(f"  - 노이즈 승수: {noise_mult:.4f}")
    except Exception as e:
        tests_failed.append(f"Test 4: 노이즈 계산 - {str(e)}")
        print(f"✗ Test 4: 노이즈 계산 실패 - {str(e)}")

    # 결과 요약
    print("=" * 60)
    print(f"검증 결과: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)} 통과")
    print("=" * 60)

    if len(tests_failed) > 0:
        print("\n실패한 테스트:")
        for test in tests_failed:
            print(f"  ✗ {test}")
        return False
    else:
        print("\n✓ 모든 검증 테스트 통과!")
        return True


def check_milestone(dataset: str, round_num: int, accuracy: float,
                    logger: Optional[logging.Logger] = None) -> bool:
    """
    중간 점검 - 이 범위를 벗어나면 구현에 문제가 있음

    Args:
        dataset: 'mnist' or 'cifar10'
        round_num: 현재 라운드 번호
        accuracy: 현재 정확도
        logger: 로거 (None이면 print 사용)

    Returns:
        True if accuracy is in expected range, False otherwise
    """
    def _log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if dataset not in EXPECTED_MILESTONES:
        _log(f"Warning: Unknown dataset '{dataset}'")
        return True

    key = f'round_{round_num}'

    if key not in EXPECTED_MILESTONES[dataset]:
        # 중간 체크포인트가 아니면 통과
        return True

    min_acc, max_acc = EXPECTED_MILESTONES[dataset][key]

    if min_acc <= accuracy <= max_acc:
        _log(f"✓ Round {round_num}: {accuracy:.3f} (정상 범위: {min_acc:.3f}-{max_acc:.3f})")
        return True
    else:
        _log(f"✗ Round {round_num}: {accuracy:.3f} (예상 범위: {min_acc:.3f}-{max_acc:.3f})")
        _log(f"  경고: 구현에 문제가 있을 수 있습니다.")
        return False


if __name__ == "__main__":
    # 직접 실행 시 검증 수행
    success = validate_implementation()
    sys.exit(0 if success else 1)