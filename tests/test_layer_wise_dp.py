# tests/test_layer_wise_dp.py
"""
QuAP-FL Layer-wise DP 단위 테스트

핵심 기능:
1. Critical layer 식별
2. Layer-wise 노이즈
3. 노이즈 감소율
4. 평균 참여율 기반 예산
"""

import unittest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import MNISTModel
from data import prepare_non_iid_data
from framework import QuAPFLServer
from config import HYPERPARAMETERS


class TestLayerWiseDP(unittest.TestCase):
    """QuAP-FL Layer-wise DP 핵심 기능 테스트"""

    @classmethod
    def setUpClass(cls):
        """테스트 환경 설정"""
        print("\nQuAP-FL Layer-wise DP 단위 테스트 시작")
        print("=" * 60)

        cls.model = MNISTModel()
        cls.client_indices, cls.train_dataset, cls.test_dataset = prepare_non_iid_data(
            'mnist', num_clients=10, alpha=0.5, seed=42
        )

        cls.config = HYPERPARAMETERS.copy()
        cls.config['num_clients'] = 10
        cls.config['num_rounds'] = 2
        cls.config['noise_strategy'] = 'layer_wise'

        cls.server = QuAPFLServer(
            model=cls.model,
            train_dataset=cls.train_dataset,
            test_dataset=cls.test_dataset,
            client_indices=cls.client_indices,
            dataset_name='mnist',
            config=cls.config,
            device=torch.device('cpu')
        )

    def test_critical_layer_identification(self):
        """Critical layer 식별 테스트"""
        print("\n1. Critical layer 식별 테스트")

        indices = self.server.critical_layer_indices

        # FC2: 128*10 + 10 = 1290 params
        total_critical = sum(end - start for start, end in indices)
        self.assertEqual(total_critical, 1290,
                        f"FC2 should have 1290 params, got {total_critical}")

        # 마지막 부분이어야 함
        total_params = sum(p.numel() for p in self.model.parameters())
        last_start, last_end = indices[-1]
        self.assertEqual(last_end, total_params,
                        "Critical layer should be at the end")

        print(f"   ✓ Critical layer 식별 성공: {total_critical} params")

    def test_layer_wise_noise(self):
        """Layer-wise 노이즈 테스트"""
        print("\n2. Layer-wise 노이즈 테스트")

        gradient = np.random.randn(sum(p.numel() for p in self.model.parameters()))
        epsilon = 0.03

        noisy = self.server._add_layer_wise_noise(gradient, epsilon)

        # 전체 shape 동일
        self.assertEqual(noisy.shape, gradient.shape)

        # Critical 외 부분은 동일
        critical_start = self.server.critical_layer_indices[0][0]
        np.testing.assert_array_equal(
            noisy[:critical_start],
            gradient[:critical_start],
            err_msg="Non-critical params should be unchanged"
        )

        # Critical 부분은 변경됨
        self.assertFalse(
            np.allclose(noisy[critical_start:], gradient[critical_start:]),
            "Critical params should have noise"
        )

        # 노이즈 norm 확인
        noise = noisy - gradient
        noise_norm = np.linalg.norm(noise)
        critical_params = self.server._count_critical_params()

        # 예상 노이즈 norm: σ * sqrt(d_c)
        clip_norm = 1.0
        sigma = clip_norm * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
        expected_noise_norm = sigma * np.sqrt(critical_params)

        # 오차 범위 30% 이내
        relative_error = abs(noise_norm - expected_noise_norm) / expected_noise_norm
        self.assertLess(
            relative_error,
            0.3,
            f"Noise norm {noise_norm:.2f} should be close to {expected_noise_norm:.2f}"
        )

        print(f"   ✓ Layer-wise 노이즈 적용 성공: ||noise||={noise_norm:.2f}")

    def test_noise_reduction(self):
        """노이즈 감소율 테스트"""
        print("\n3. 노이즈 감소율 테스트")

        gradient = np.random.randn(sum(p.numel() for p in self.model.parameters()))
        epsilon = 0.03

        # Layer-wise
        noisy_layer = self.server._add_layer_wise_noise(gradient, epsilon)
        noise_layer = noisy_layer - gradient
        norm_layer = np.linalg.norm(noise_layer)

        # Full (비교용)
        noisy_full = self.server._add_full_noise(gradient, epsilon)
        noise_full = noisy_full - gradient
        norm_full = np.linalg.norm(noise_full)

        # Layer-wise가 더 작아야 함
        self.assertLess(norm_layer, norm_full,
                       "Layer-wise noise should be smaller than full")

        # 이론적 감소율
        total_params = len(gradient)
        critical_params = self.server._count_critical_params()
        theoretical_ratio = np.sqrt(critical_params / total_params)
        actual_ratio = norm_layer / norm_full

        # 오차 범위 20% 이내
        relative_error = abs(actual_ratio - theoretical_ratio) / theoretical_ratio
        self.assertLess(
            relative_error,
            0.2,
            f"Reduction ratio {actual_ratio:.4f} should be close to {theoretical_ratio:.4f}"
        )

        reduction = norm_full / norm_layer
        print(f"   ✓ 노이즈 감소율 확인: {reduction:.1f}x")

    def test_average_participation_budget(self):
        """평균 참여율 기반 예산 테스트"""
        print("\n4. 평균 참여율 기반 예산 테스트")

        # 시뮬레이션: 3개 클라이언트 선택
        selected = [0, 1, 2]

        # 참여 이력 설정
        self.server.tracker.participation_count = np.array([5, 10, 2, 0, 0, 0, 0, 0, 0, 0])
        self.server.tracker.total_rounds = 10

        # 평균 참여율
        rates = [self.server.tracker.get_participation_rate(i) for i in selected]
        avg_rate = np.mean(rates)  # (0.5 + 1.0 + 0.2) / 3 = 0.567

        # 예산 계산
        epsilon = self.server.privacy_allocator.compute_privacy_budget(avg_rate)

        # 예상값
        epsilon_base = self.config['epsilon_total'] / self.config['num_rounds']
        expected = epsilon_base * (1 + 0.5 * np.exp(-2.0 * avg_rate))

        self.assertAlmostEqual(epsilon, expected, places=6)
        print(f"   ✓ 평균 참여율 기반 예산 계산 성공: ε={epsilon:.6f}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("QuAP-FL Layer-wise DP 단위 테스트")
    print("=" * 60)
    unittest.main(verbosity=2)
