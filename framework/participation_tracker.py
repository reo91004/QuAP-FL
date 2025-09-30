# framework/participation_tracker.py

"""
ParticipationTracker: 클라이언트 참여 이력 추적 메커니즘

이 부분이 QuAP-FL의 핵심이다.
각 클라이언트의 참여 이력을 정확히 추적해야 한다.
"""

import numpy as np
from typing import List


class ParticipationTracker:
    """
    클라이언트 참여 이력을 추적하고 참여율을 계산한다.
    """

    def __init__(self, num_clients: int):
        """
        Args:
            num_clients: 전체 클라이언트 수
        """
        # 반드시 float64 사용 (정밀도 중요)
        self.num_clients = num_clients
        self.participation_count = np.zeros(num_clients, dtype=np.float64)
        self.total_rounds = 0

    def update(self, participating_clients: List[int]):
        """
        매 라운드 호출 필수

        Args:
            participating_clients: 이번 라운드 참여 클라이언트 ID 리스트
        """
        self.total_rounds += 1
        for client_id in participating_clients:
            if 0 <= client_id < self.num_clients:
                self.participation_count[client_id] += 1.0
            else:
                raise ValueError(f"Invalid client_id: {client_id}. Must be in [0, {self.num_clients})")

    def get_participation_rate(self, client_id: int) -> float:
        """
        참여율 계산 - division by zero 처리 필수

        Args:
            client_id: 클라이언트 ID

        Returns:
            참여율 (0.0 ~ 1.0)
        """
        if not (0 <= client_id < self.num_clients):
            raise ValueError(f"Invalid client_id: {client_id}. Must be in [0, {self.num_clients})")

        if self.total_rounds == 0:
            return 0.0

        return float(self.participation_count[client_id]) / float(self.total_rounds)

    def get_all_participation_rates(self) -> np.ndarray:
        """
        모든 클라이언트의 참여율 반환

        Returns:
            shape (num_clients,) numpy array
        """
        if self.total_rounds == 0:
            return np.zeros(self.num_clients, dtype=np.float64)

        return self.participation_count / float(self.total_rounds)

    def get_statistics(self) -> dict:
        """
        참여 통계 정보 반환 (디버깅 및 모니터링용)

        Returns:
            통계 딕셔너리
        """
        rates = self.get_all_participation_rates()

        return {
            'total_rounds': self.total_rounds,
            'mean_participation_rate': float(np.mean(rates)),
            'std_participation_rate': float(np.std(rates)),
            'min_participation_rate': float(np.min(rates)),
            'max_participation_rate': float(np.max(rates)),
            'participating_clients': int(np.sum(rates > 0)),
            'never_participated': int(np.sum(rates == 0))
        }

    def reset(self):
        """
        추적 정보 초기화 (새로운 실험 시작 시)
        """
        self.participation_count = np.zeros(self.num_clients, dtype=np.float64)
        self.total_rounds = 0