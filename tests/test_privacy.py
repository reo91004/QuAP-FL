# tests/test_privacy.py

"""
AdaptivePrivacyAllocator 단위 테스트
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework.adaptive_privacy import AdaptivePrivacyAllocator


class TestAdaptivePrivacyAllocator(unittest.TestCase):
    """AdaptivePrivacyAllocator 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        allocator = AdaptivePrivacyAllocator(
            epsilon_base=0.015,
            alpha=0.5,
            beta=2.0,
            delta=1e-5
        )
        self.assertEqual(allocator.epsilon_base, 0.015)
        self.assertEqual(allocator.alpha, 0.5)
        self.assertEqual(allocator.beta, 2.0)
        self.assertEqual(allocator.delta, 1e-5)

    def test_invalid_parameters(self):
        """잘못된 파라미터 테스트"""
        with self.assertRaises(ValueError):
            AdaptivePrivacyAllocator(epsilon_base=-0.1)

        with self.assertRaises(ValueError):
            AdaptivePrivacyAllocator(alpha=1.5)

        with self.assertRaises(ValueError):
            AdaptivePrivacyAllocator(beta=-1.0)

        with self.assertRaises(ValueError):
            AdaptivePrivacyAllocator(delta=0)

    def test_privacy_budget(self):
        """프라이버시 예산 계산 테스트"""
        allocator = AdaptivePrivacyAllocator(
            epsilon_base=0.015,
            alpha=0.5,
            beta=2.0
        )

        # 낮은 참여율 → 높은 예산
        eps_low = allocator.compute_privacy_budget(0.1)
        # 높은 참여율 → 낮은 예산
        eps_high = allocator.compute_privacy_budget(0.9)

        self.assertGreater(eps_low, eps_high)

        # 예상 값 확인
        expected_low = 0.015 * (1 + 0.5 * np.exp(-2 * 0.1))
        expected_high = 0.015 * (1 + 0.5 * np.exp(-2 * 0.9))

        self.assertAlmostEqual(eps_low, expected_low, places=6)
        self.assertAlmostEqual(eps_high, expected_high, places=6)

    def test_privacy_budget_range(self):
        """프라이버시 예산 범위 테스트"""
        allocator = AdaptivePrivacyAllocator(
            epsilon_base=0.015,
            alpha=0.5,
            beta=2.0
        )

        # 참여율 0 → 최대 예산
        eps_min = allocator.compute_privacy_budget(0.0)
        expected_max = 0.015 * (1 + 0.5)
        self.assertAlmostEqual(eps_min, expected_max, places=6)

        # 참여율 1 → 최소 예산 (거의 ε_base)
        eps_max = allocator.compute_privacy_budget(1.0)
        self.assertGreater(eps_max, 0.015)
        self.assertLess(eps_max, 0.015 * 1.1)

    def test_noise_multiplier(self):
        """노이즈 승수 계산 테스트"""
        allocator = AdaptivePrivacyAllocator(
            epsilon_base=0.015,
            delta=1e-5
        )

        epsilon = 0.015
        clip_norm = 1.0

        noise_mult = allocator.compute_noise_multiplier(epsilon, clip_norm)

        # 예상 값
        expected = clip_norm * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon

        self.assertAlmostEqual(noise_mult, expected, places=6)
        self.assertGreater(noise_mult, 0)

    def test_add_gaussian_noise(self):
        """가우시안 노이즈 추가 테스트"""
        allocator = AdaptivePrivacyAllocator(
            epsilon_base=0.015,
            delta=1e-5
        )

        data = np.ones(100)
        epsilon = 0.015
        clip_norm = 1.0

        noisy_data = allocator.add_gaussian_noise(data, epsilon, clip_norm)

        # 형태 확인
        self.assertEqual(noisy_data.shape, data.shape)

        # 노이즈가 추가되었는지 확인
        self.assertFalse(np.allclose(noisy_data, data))

    def test_cumulative_privacy_loss(self):
        """누적 프라이버시 손실 계산 테스트"""
        allocator = AdaptivePrivacyAllocator(
            epsilon_base=0.015,
            alpha=0.5,
            beta=2.0
        )

        participation_rates = np.array([0.5, 0.8, 0.2])
        num_participations = np.array([10, 16, 4])

        cumulative = allocator.compute_cumulative_privacy_loss(
            participation_rates, num_participations
        )

        # 각 클라이언트의 누적 손실 검증
        for i in range(3):
            eps_i = allocator.compute_privacy_budget(participation_rates[i])
            expected = eps_i * num_participations[i]
            self.assertAlmostEqual(cumulative[i], expected, places=6)


if __name__ == '__main__':
    unittest.main()