# tests/test_clipping.py

"""
QuantileClipper 단위 테스트
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework.quantile_clipping import QuantileClipper


class TestQuantileClipper(unittest.TestCase):
    """QuantileClipper 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        clipper = QuantileClipper(
            quantile=0.9,
            momentum=0.95,
            min_clip=0.1,
            max_clip=10.0
        )
        self.assertEqual(clipper.quantile, 0.9)
        self.assertEqual(clipper.momentum, 0.95)
        self.assertEqual(clipper.min_clip, 0.1)
        self.assertEqual(clipper.max_clip, 10.0)
        self.assertIsNone(clipper.clip_value)

    def test_invalid_parameters(self):
        """잘못된 파라미터 테스트"""
        with self.assertRaises(ValueError):
            QuantileClipper(quantile=1.5)

        with self.assertRaises(ValueError):
            QuantileClipper(momentum=1.5)

        with self.assertRaises(ValueError):
            QuantileClipper(min_clip=-0.1)

        with self.assertRaises(ValueError):
            QuantileClipper(min_clip=10.0, max_clip=5.0)

    def test_compute_gradient_norms(self):
        """그래디언트 norm 계산 테스트"""
        clipper = QuantileClipper()

        grads = [
            np.ones(100) * 1,
            np.ones(100) * 2,
            np.ones(100) * 3
        ]

        norms = clipper.compute_gradient_norms(grads)

        expected = [
            np.linalg.norm(np.ones(100) * 1),
            np.linalg.norm(np.ones(100) * 2),
            np.linalg.norm(np.ones(100) * 3)
        ]

        np.testing.assert_array_almost_equal(norms, expected)

    def test_update_clip_value(self):
        """클리핑 값 업데이트 테스트"""
        clipper = QuantileClipper(quantile=0.9, momentum=0.95)

        # 그래디언트 생성 (norm이 1~10)
        grads = [np.random.randn(100) * i for i in range(1, 11)]

        # 첫 업데이트
        clip1 = clipper.update_clip_value(grads)
        self.assertIsNotNone(clip1)
        self.assertGreater(clip1, 0)

        # 두 번째 업데이트 (EMA 적용)
        clip2 = clipper.update_clip_value(grads)
        self.assertIsNotNone(clip2)

        # EMA가 적용되어 급격한 변화 없음
        self.assertLess(abs(clip2 - clip1) / clip1, 0.5)

    def test_clip_value_range(self):
        """클리핑 값 범위 제한 테스트"""
        clipper = QuantileClipper(min_clip=0.5, max_clip=5.0)

        # 매우 작은 그래디언트
        small_grads = [np.random.randn(100) * 0.01 for _ in range(10)]
        clip_small = clipper.update_clip_value(small_grads)
        self.assertGreaterEqual(clip_small, 0.5)

        # 새로운 clipper로 매우 큰 그래디언트 테스트
        clipper2 = QuantileClipper(min_clip=0.5, max_clip=5.0)
        large_grads = [np.random.randn(100) * 100 for _ in range(10)]
        clip_large = clipper2.update_clip_value(large_grads)
        self.assertLessEqual(clip_large, 5.0)

    def test_clip_gradient(self):
        """단일 그래디언트 클리핑 테스트"""
        clipper = QuantileClipper()

        grads = [np.random.randn(100) for _ in range(10)]
        clipper.update_clip_value(grads)

        # 큰 그래디언트
        large_grad = np.random.randn(100) * 100
        clipped = clipper.clip_gradient(large_grad)

        # Norm이 clip_value 이하인지 확인
        clipped_norm = np.linalg.norm(clipped)
        self.assertLessEqual(clipped_norm, clipper.clip_value + 1e-6)

        # 작은 그래디언트는 그대로
        small_grad = np.random.randn(100) * 0.001
        clipped_small = clipper.clip_gradient(small_grad)
        np.testing.assert_array_almost_equal(small_grad, clipped_small)

    def test_clip_gradients(self):
        """여러 그래디언트 동시 클리핑 테스트"""
        clipper = QuantileClipper()

        grads = [np.random.randn(100) * i for i in range(1, 11)]
        clipper.update_clip_value(grads)

        clipped_grads = clipper.clip_gradients(grads)

        self.assertEqual(len(clipped_grads), len(grads))

        for clipped in clipped_grads:
            norm = np.linalg.norm(clipped)
            self.assertLessEqual(norm, clipper.clip_value + 1e-6)

    def test_uninitialized_clip(self):
        """초기화되지 않은 상태에서 클리핑 시도 테스트"""
        clipper = QuantileClipper()

        grad = np.random.randn(100)

        with self.assertRaises(RuntimeError):
            clipper.clip_gradient(grad)

        with self.assertRaises(RuntimeError):
            clipper.clip_gradients([grad])

    def test_statistics(self):
        """통계 정보 테스트"""
        clipper = QuantileClipper()

        # 초기 상태
        stats = clipper.get_statistics()
        self.assertIsNone(stats['current_clip'])
        self.assertEqual(stats['num_updates'], 0)

        # 업데이트 후
        grads = [np.random.randn(100) for _ in range(10)]
        clipper.update_clip_value(grads)

        stats = clipper.get_statistics()
        self.assertIsNotNone(stats['current_clip'])
        self.assertEqual(stats['num_updates'], 1)

    def test_reset(self):
        """리셋 테스트"""
        clipper = QuantileClipper()

        grads = [np.random.randn(100) for _ in range(10)]
        clipper.update_clip_value(grads)

        clipper.reset()

        self.assertIsNone(clipper.clip_value)
        self.assertEqual(len(clipper.clip_history), 0)


if __name__ == '__main__':
    unittest.main()