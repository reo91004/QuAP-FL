# framework/quantile_clipping.py

"""
QuantileClipper: 90th percentile 기반 적응형 그래디언트 클리핑

핵심:
- 매 라운드 그래디언트 norm의 90th percentile을 계산
- EMA (Exponential Moving Average)로 안정화
- 극단값 제한으로 안전성 확보
"""

import numpy as np
from typing import List, Union, Optional


class QuantileClipper:
    """
    분위수 기반 적응형 그래디언트 클리핑
    """

    def __init__(
        self,
        quantile: float = 0.9,
        momentum: float = 0.95,
        min_clip: float = 0.1,
        max_clip: float = 10.0,
        initial_clip: Optional[float] = None
    ):
        """
        Args:
            quantile: 클리핑에 사용할 분위수 (0.9 = 90th percentile, 고정값)
            momentum: EMA momentum (0.95 고정값)
            min_clip: 최소 클리핑 값 (안정성)
            max_clip: 최대 클리핑 값 (안정성)
            initial_clip: 초기 클리핑 값 (None이면 첫 라운드에서 자동 설정)
        """
        self.quantile = quantile
        self.momentum = momentum
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.clip_value = initial_clip

        # 입력 검증
        if not (0 < quantile < 1):
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        if not (0 <= momentum < 1):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if min_clip <= 0:
            raise ValueError(f"min_clip must be positive, got {min_clip}")
        if max_clip <= min_clip:
            raise ValueError(f"max_clip ({max_clip}) must be greater than min_clip ({min_clip})")

        # 통계 추적 (디버깅용)
        self.clip_history = []

    def compute_gradient_norms(self, gradients_list: List[np.ndarray]) -> np.ndarray:
        """
        그래디언트 L2 norm 계산

        Args:
            gradients_list: 그래디언트 리스트 (각 클라이언트)

        Returns:
            norm 배열 (shape: len(gradients_list))
        """
        norms = []
        for grad in gradients_list:
            # Flatten 후 L2 norm
            norm = np.linalg.norm(grad.flatten(), ord=2)
            norms.append(norm)

        return np.array(norms)

    def update_clip_value(self, gradients_list: List[np.ndarray]) -> float:
        """
        그래디언트 norm의 분위수 기반 클리핑 값 업데이트

        Args:
            gradients_list: 이번 라운드 모든 클라이언트의 그래디언트

        Returns:
            업데이트된 클리핑 값
        """
        if len(gradients_list) == 0:
            raise ValueError("gradients_list cannot be empty")

        # L2 norm 계산
        grad_norms = self.compute_gradient_norms(gradients_list)

        # NaN/Inf 필터링
        grad_norms = grad_norms[~np.isnan(grad_norms) & ~np.isinf(grad_norms)]

        # 유효한 norm이 없으면 이전 값 유지
        if len(grad_norms) == 0:
            return self.clip_value if self.clip_value is not None else self.min_clip

        # 90th percentile 계산
        current_clip = float(np.percentile(grad_norms, self.quantile * 100))

        # NaN/Inf 체크
        if np.isnan(current_clip) or np.isinf(current_clip):
            return self.clip_value if self.clip_value is not None else self.min_clip

        # 범위 제한
        current_clip = np.clip(current_clip, self.min_clip, self.max_clip)

        # EMA 업데이트
        if self.clip_value is None:
            # 첫 라운드: 현재 값으로 초기화
            self.clip_value = current_clip
        else:
            # Exponential Moving Average
            self.clip_value = (
                self.momentum * self.clip_value +
                (1 - self.momentum) * current_clip
            )

        # 통계 기록
        self.clip_history.append(self.clip_value)

        return self.clip_value

    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """
        단일 그래디언트 클리핑

        Args:
            gradient: 입력 그래디언트

        Returns:
            클리핑된 그래디언트
        """
        if self.clip_value is None:
            raise RuntimeError("clip_value not initialized. Call update_clip_value first.")

        # NaN/Inf 체크
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            return np.zeros_like(gradient)

        # L2 norm 계산
        norm = np.linalg.norm(gradient.flatten(), ord=2)

        # NaN norm 체크
        if np.isnan(norm) or np.isinf(norm):
            return np.zeros_like(gradient)

        # 클리핑 필요 여부 확인
        if norm > self.clip_value:
            # 클리핑 적용
            clipped = gradient * (self.clip_value / norm)
            return clipped
        else:
            # 클리핑 불필요
            return gradient

    def clip_gradients(self, gradients_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        여러 그래디언트를 동시에 클리핑

        Args:
            gradients_list: 그래디언트 리스트

        Returns:
            클리핑된 그래디언트 리스트
        """
        if self.clip_value is None:
            raise RuntimeError("clip_value not initialized. Call update_clip_value first.")

        clipped_list = []
        for grad in gradients_list:
            clipped = self.clip_gradient(grad)
            clipped_list.append(clipped)

        return clipped_list

    def get_current_clip_value(self) -> Optional[float]:
        """
        현재 클리핑 값 반환

        Returns:
            현재 클리핑 값 (초기화 전이면 None)
        """
        return self.clip_value

    def get_statistics(self) -> dict:
        """
        클리핑 통계 정보 반환

        Returns:
            통계 딕셔너리
        """
        if len(self.clip_history) == 0:
            return {
                'current_clip': self.clip_value,
                'num_updates': 0
            }

        return {
            'current_clip': self.clip_value,
            'num_updates': len(self.clip_history),
            'mean_clip': float(np.mean(self.clip_history)),
            'std_clip': float(np.std(self.clip_history)),
            'min_clip_observed': float(np.min(self.clip_history)),
            'max_clip_observed': float(np.max(self.clip_history))
        }

    def reset(self):
        """
        클리핑 상태 초기화
        """
        self.clip_value = None
        self.clip_history = []