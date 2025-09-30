# tests/test_tracker.py

"""
ParticipationTracker 단위 테스트
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework.participation_tracker import ParticipationTracker


class TestParticipationTracker(unittest.TestCase):
    """ParticipationTracker 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        tracker = ParticipationTracker(10)
        self.assertEqual(tracker.num_clients, 10)
        self.assertEqual(tracker.total_rounds, 0)
        self.assertEqual(len(tracker.participation_count), 10)
        np.testing.assert_array_equal(tracker.participation_count, np.zeros(10))

    def test_update(self):
        """참여 기록 업데이트 테스트"""
        tracker = ParticipationTracker(10)

        # 첫 라운드
        tracker.update([0, 1, 2])
        self.assertEqual(tracker.total_rounds, 1)
        self.assertEqual(tracker.participation_count[0], 1)
        self.assertEqual(tracker.participation_count[1], 1)
        self.assertEqual(tracker.participation_count[2], 1)
        self.assertEqual(tracker.participation_count[3], 0)

        # 두 번째 라운드
        tracker.update([1, 2, 3])
        self.assertEqual(tracker.total_rounds, 2)
        self.assertEqual(tracker.participation_count[0], 1)
        self.assertEqual(tracker.participation_count[1], 2)
        self.assertEqual(tracker.participation_count[2], 2)
        self.assertEqual(tracker.participation_count[3], 1)

    def test_participation_rate(self):
        """참여율 계산 테스트"""
        tracker = ParticipationTracker(10)

        # 초기 상태
        self.assertEqual(tracker.get_participation_rate(0), 0.0)

        # 2라운드 중 2번 참여
        tracker.update([0, 1, 2])
        tracker.update([0, 2, 3])
        self.assertAlmostEqual(tracker.get_participation_rate(0), 1.0)
        self.assertAlmostEqual(tracker.get_participation_rate(1), 0.5)
        self.assertAlmostEqual(tracker.get_participation_rate(2), 1.0)
        self.assertAlmostEqual(tracker.get_participation_rate(3), 0.5)
        self.assertAlmostEqual(tracker.get_participation_rate(4), 0.0)

    def test_invalid_client_id(self):
        """잘못된 클라이언트 ID 테스트"""
        tracker = ParticipationTracker(10)

        with self.assertRaises(ValueError):
            tracker.update([10])  # 범위 초과

        with self.assertRaises(ValueError):
            tracker.update([-1])  # 음수

        with self.assertRaises(ValueError):
            tracker.get_participation_rate(10)

    def test_statistics(self):
        """통계 계산 테스트"""
        tracker = ParticipationTracker(10)

        tracker.update([0, 1, 2])
        tracker.update([0, 1])
        tracker.update([0])

        stats = tracker.get_statistics()

        self.assertEqual(stats['total_rounds'], 3)
        self.assertEqual(stats['participating_clients'], 3)
        self.assertEqual(stats['never_participated'], 7)
        self.assertAlmostEqual(stats['mean_participation_rate'], 0.6 / 10)

    def test_reset(self):
        """리셋 테스트"""
        tracker = ParticipationTracker(10)

        tracker.update([0, 1, 2])
        tracker.update([1, 2, 3])

        tracker.reset()

        self.assertEqual(tracker.total_rounds, 0)
        np.testing.assert_array_equal(tracker.participation_count, np.zeros(10))


if __name__ == '__main__':
    unittest.main()