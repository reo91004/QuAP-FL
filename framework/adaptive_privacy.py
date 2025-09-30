# framework/adaptive_privacy.py

"""
AdaptivePrivacyAllocator: 참여율 기반 적응형 프라이버시 예산 할당

핵심 수식: ε_i(t) = ε_base * (1 + α * exp(-β * p_i(t)))

직관:
- 자주 참여하는 클라이언트 (높은 p_i) → 작은 ε_i → 강한 프라이버시 보호 (많은 노이즈)
- 간헐적 참여 클라이언트 (낮은 p_i) → 큰 ε_i → 높은 유틸리티 (적은 노이즈)
"""

import numpy as np
from typing import Optional


class AdaptivePrivacyAllocator:
    """
    참여율 기반 적응형 프라이버시 예산 할당
    """

    def __init__(
        self,
        epsilon_base: float = 0.015,
        alpha: float = 0.5,
        beta: float = 2.0,
        delta: float = 1e-5
    ):
        """
        Args:
            epsilon_base: 기본 프라이버시 예산 (ε_total / num_rounds)
            alpha: 적응 강도 (0 ~ 1), 고정값
            beta: 감소율 파라미터 (> 0), 고정값
            delta: 차분 프라이버시 파라미터
        """
        self.epsilon_base = epsilon_base
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        # 입력 검증
        if epsilon_base <= 0:
            raise ValueError(f"epsilon_base must be positive, got {epsilon_base}")
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

    def compute_privacy_budget(self, participation_rate: float) -> float:
        """
        참여율에 따른 프라이버시 예산 계산

        수식: ε_i = ε_base * (1 + α * exp(-β * p_i))

        Args:
            participation_rate: 클라이언트 참여율 (0.0 ~ 1.0)

        Returns:
            적응형 프라이버시 예산
        """
        if not (0 <= participation_rate <= 1):
            raise ValueError(f"participation_rate must be in [0, 1], got {participation_rate}")

        # 적응 팩터 계산
        adaptive_factor = 1.0 + self.alpha * np.exp(-self.beta * participation_rate)

        # 최종 예산
        epsilon_i = self.epsilon_base * adaptive_factor

        return float(epsilon_i)

    def compute_noise_multiplier(
        self,
        epsilon: float,
        clip_norm: float,
        delta: Optional[float] = None
    ) -> float:
        """
        Gaussian mechanism 노이즈 승수 계산

        공식: σ = C * sqrt(2 * ln(1.25/δ)) / ε

        여기서:
        - C: 클리핑 norm
        - δ: 차분 프라이버시 파라미터
        - ε: 프라이버시 예산

        Args:
            epsilon: 프라이버시 예산
            clip_norm: 클리핑 norm 값
            delta: 차분 프라이버시 파라미터 (None이면 초기화 값 사용)

        Returns:
            노이즈 표준편차
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")

        delta_val = delta if delta is not None else self.delta

        # Gaussian mechanism 노이즈 스케일
        noise_multiplier = clip_norm * np.sqrt(2 * np.log(1.25 / delta_val)) / epsilon

        return float(noise_multiplier)

    def add_gaussian_noise(
        self,
        data: np.ndarray,
        epsilon: float,
        clip_norm: float
    ) -> np.ndarray:
        """
        데이터에 가우시안 노이즈 추가

        Args:
            data: 입력 데이터 (이미 클리핑된 그래디언트)
            epsilon: 프라이버시 예산
            clip_norm: 클리핑 norm 값

        Returns:
            노이즈가 추가된 데이터
        """
        noise_std = self.compute_noise_multiplier(epsilon, clip_norm)

        # 가우시안 노이즈 생성
        noise = np.random.normal(0, noise_std, data.shape)

        return data + noise

    def compute_cumulative_privacy_loss(
        self,
        participation_rates: np.ndarray,
        num_participations: np.ndarray
    ) -> np.ndarray:
        """
        각 클라이언트의 누적 프라이버시 손실 계산

        정리 1 (누적 프라이버시 손실):
        클라이언트 i가 T 라운드 중 k_i번 참여할 때,
        ε_total,i = Σ ε_i(t) ≤ k_i * ε_base * (1 + α)

        Args:
            participation_rates: 각 클라이언트의 참여율 (shape: num_clients)
            num_participations: 각 클라이언트의 참여 횟수 (shape: num_clients)

        Returns:
            누적 프라이버시 손실 (shape: num_clients)
        """
        # 각 클라이언트의 평균 예산
        avg_budgets = np.array([
            self.compute_privacy_budget(rate) for rate in participation_rates
        ])

        # 누적 손실 = 평균 예산 * 참여 횟수
        cumulative_loss = avg_budgets * num_participations

        return cumulative_loss

    def get_privacy_analysis(
        self,
        participation_rates: np.ndarray,
        num_participations: np.ndarray
    ) -> dict:
        """
        프라이버시 분석 정보 반환

        Args:
            participation_rates: 각 클라이언트의 참여율
            num_participations: 각 클라이언트의 참여 횟수

        Returns:
            분석 딕셔너리
        """
        cumulative_loss = self.compute_cumulative_privacy_loss(
            participation_rates, num_participations
        )

        # 상한 계산 (정리 1)
        upper_bound = num_participations * self.epsilon_base * (1.0 + self.alpha)

        return {
            'mean_cumulative_loss': float(np.mean(cumulative_loss)),
            'max_cumulative_loss': float(np.max(cumulative_loss)),
            'min_cumulative_loss': float(np.min(cumulative_loss)),
            'theoretical_upper_bound': float(np.max(upper_bound)),
            'privacy_efficiency': float(np.mean(cumulative_loss) / np.mean(upper_bound)) if np.mean(upper_bound) > 0 else 0.0
        }