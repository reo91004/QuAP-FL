# framework/server.py

"""
QuAPFLServer: QuAP-FL 메인 학습 서버

핵심 9단계 순서 (절대 변경 금지):
1. 클라이언트 선택
2. 참여 이력 업데이트 (노이즈 전에!)
3. 로컬 학습
4. 클리핑 값 업데이트
5. 클리핑 적용
6. 적응형 노이즈 추가
7. 연합 평균
8. 모델 업데이트
9. 평가
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Optional, Dict
from tqdm import tqdm
import logging
import sys
import os

# 상위 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework.participation_tracker import ParticipationTracker
from framework.adaptive_privacy import AdaptivePrivacyAllocator
from framework.quantile_clipping import QuantileClipper
from config.hyperparameters import HYPERPARAMETERS, TARGET_ACCURACY
from data.data_utils import create_client_dataloaders
from utils.validation import check_milestone


class QuAPFLServer:
    """
    QuAP-FL 연합학습 서버
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        test_dataset,
        client_indices: List[List[int]],
        dataset_name: str,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model: 전역 모델
            train_dataset: 학습 데이터셋
            test_dataset: 테스트 데이터셋
            client_indices: 각 클라이언트의 데이터 인덱스
            dataset_name: 'mnist' or 'cifar10'
            config: 하이퍼파라미터 (None이면 기본값 사용)
            device: 디바이스 (None이면 자동 선택)
            logger: 로거 (None이면 print 사용)
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_indices = client_indices
        self.dataset_name = dataset_name
        self.logger = logger

        # 설정
        self.config = config if config is not None else HYPERPARAMETERS
        self.device = device if device is not None else (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 모델을 디바이스로 이동
        self.model.to(self.device)

        # 클라이언트 수
        self.num_clients = len(client_indices)
        self.num_rounds = self.config['num_rounds']

        # 핵심 컴포넌트 초기화
        self.tracker = ParticipationTracker(self.num_clients)

        epsilon_base = self.config['epsilon_total'] / self.num_rounds
        self.privacy_allocator = AdaptivePrivacyAllocator(
            epsilon_base=epsilon_base,
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            delta=self.config['delta']
        )

        self.clipper = QuantileClipper(
            quantile=self.config['clip_quantile'],
            momentum=self.config['clip_momentum'],
            min_clip=self.config['min_clip'],
            max_clip=self.config['max_clip']
        )

        # 통계 추적
        self.history = {
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': [],
            'clip_values': [],
            'noise_levels': [],
            'participation_stats': [],
            'privacy_budgets': []
        }

        # 현재 라운드
        self.current_round = 0

    def _log(self, message: str):
        """
        로거가 있으면 logger.info()로, 없으면 print()로 출력

        Args:
            message: 출력할 메시지
        """
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def select_clients(self, round_t: int) -> List[int]:
        """
        클라이언트 선택 (30% 참여율 시뮬레이션)

        Args:
            round_t: 현재 라운드

        Returns:
            선택된 클라이언트 ID 리스트
        """
        num_selected = max(1, self.config['clients_per_round'])

        if num_selected > self.num_clients:
            num_selected = self.num_clients

        # 참여 패턴 변화 시뮬레이션
        if round_t % 2 == 0:
            # 짝수 라운드: 낮은 ID 선호 (이질적 참여 시뮬레이션)
            weights = np.exp(-np.arange(self.num_clients) * 0.01)
        else:
            # 홀수 라운드: 균등
            weights = np.ones(self.num_clients)

        weights = weights / weights.sum()

        selected_clients = np.random.choice(
            self.num_clients,
            size=num_selected,
            replace=False,
            p=weights
        )

        return selected_clients.tolist()

    def local_training(
        self,
        client_id: int,
        round_t: int
    ) -> np.ndarray:
        """
        클라이언트 로컬 학습

        Args:
            client_id: 클라이언트 ID
            round_t: 현재 라운드

        Returns:
            그래디언트 (1D numpy array)
        """
        # 클라이언트 데이터 로더 생성
        dataloader = create_client_dataloaders(
            self.train_dataset,
            self.client_indices,
            self.config['local_batch_size'],
            client_id
        )

        # 로컬 모델 생성 (전역 모델 복사)
        local_model = type(self.model)()
        local_model.load_state_dict(self.model.state_dict())
        local_model.to(self.device)
        local_model.train()

        # 학습률 decay 적용
        lr = self.config['learning_rate'] * (self.config['lr_decay'] ** round_t)

        # Optimizer
        optimizer = optim.SGD(local_model.parameters(), lr=lr)
        # NLLLoss 사용 (모델이 log_softmax를 출력하므로)
        criterion = nn.NLLLoss()

        # 로컬 에폭 학습
        for epoch in range(self.config['local_epochs']):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 그래디언트 계산 (전역 모델과의 차이)
        gradient = []
        for local_param, global_param in zip(
            local_model.parameters(), self.model.parameters()
        ):
            # 올바른 그래디언트 방향: local - global
            diff = local_param.data - global_param.data
            gradient.append(diff.view(-1).cpu().numpy())

        gradient = np.concatenate(gradient)

        return gradient

    def update_global_model(self, aggregated_gradient: np.ndarray):
        """
        전역 모델 업데이트

        Args:
            aggregated_gradient: 집계된 그래디언트 (1D numpy array)
        """
        # NaN/Inf 체크
        if np.any(np.isnan(aggregated_gradient)) or np.any(np.isinf(aggregated_gradient)):
            print("Warning: NaN or Inf detected in aggregated gradient, skipping update")
            return

        # 그래디언트 norm 제한 (추가 안전장치)
        grad_norm = np.linalg.norm(aggregated_gradient)
        if grad_norm > 100:  # 너무 큰 업데이트 방지
            aggregated_gradient = aggregated_gradient * (100 / grad_norm)
            print(f"Warning: Large gradient norm {grad_norm:.2f}, scaling to 100")

        idx = 0
        for param in self.model.parameters():
            numel = param.numel()
            shape = param.shape

            # 그래디언트를 파라미터 형태로 복원
            grad = aggregated_gradient[idx:idx+numel].reshape(shape)
            grad_tensor = torch.from_numpy(grad).to(self.device).float()

            # 파라미터 업데이트
            # gradient는 이미 (local - global) 델타이므로 직접 더해줌
            # learning rate는 로컬 학습에서 이미 적용됨
            param.data += grad_tensor

            idx += numel

    def evaluate(self) -> tuple:
        """
        전역 모델 평가

        Returns:
            (accuracy, loss)
        """
        self.model.eval()
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=256,
            shuffle=False
        )

        # NLLLoss 사용 (모델이 log_softmax를 출력하므로)
        criterion = nn.NLLLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * data.size(0)

                # MNIST는 log_softmax를 사용하므로 처리 방식이 다름
                if self.dataset_name == 'mnist':
                    pred = output.argmax(dim=1, keepdim=True)
                else:
                    pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        accuracy = correct / total
        avg_loss = total_loss / total

        self.model.train()

        return accuracy, avg_loss

    def check_convergence(self, accuracy: float) -> bool:
        """
        목표 성능 달성 체크

        Args:
            accuracy: 현재 정확도

        Returns:
            True if converged, False otherwise
        """
        if self.dataset_name in TARGET_ACCURACY:
            target = TARGET_ACCURACY[self.dataset_name]
            return accuracy >= target
        return False

    def train(self):
        """
        메인 학습 루프

        핵심 9단계를 정확히 따른다 (순서 변경 금지)
        """
        self._log("=" * 60)
        self._log(f"QuAP-FL Training Start: {self.dataset_name.upper()}")
        self._log(f"Target Accuracy: {TARGET_ACCURACY.get(self.dataset_name, 'N/A')}")
        self._log(f"Clients: {self.num_clients}, Rounds: {self.num_rounds}")
        self._log("=" * 60)

        # 초기 모델 평가
        initial_acc, initial_loss = self.evaluate()
        self._log(f"\nInitial model: Acc={initial_acc:.4f}, Loss={initial_loss:.4f}")

        for round_t in tqdm(range(self.num_rounds), desc="Training"):
            self.current_round = round_t

            # 1. 클라이언트 선택
            selected_clients = self.select_clients(round_t)

            # 2. 참여 이력 업데이트 (노이즈 추가 전에!)
            self.tracker.update(selected_clients)

            # 3. 로컬 학습 및 그래디언트 수집
            local_gradients = []
            for client_id in selected_clients:
                gradient = self.local_training(client_id, round_t)
                local_gradients.append(gradient)

            # 4. 클리핑 값 업데이트 (90th percentile)
            current_clip = self.clipper.update_clip_value(local_gradients)
            self.history['clip_values'].append(current_clip)

            # 5. 클리핑 적용
            clipped_gradients = self.clipper.clip_gradients(local_gradients)

            # 6. 적응형 노이즈 추가
            noisy_gradients = []
            for i, client_id in enumerate(selected_clients):
                # 참여율 기반 프라이버시 예산
                p_rate = self.tracker.get_participation_rate(client_id)
                epsilon_i = self.privacy_allocator.compute_privacy_budget(p_rate)

                # 가우시안 노이즈 추가
                noisy_grad = self.privacy_allocator.add_gaussian_noise(
                    clipped_gradients[i],
                    epsilon_i,
                    self.clipper.clip_value
                )

                noisy_gradients.append(noisy_grad)

                # 통계 기록
                noise_mult = self.privacy_allocator.compute_noise_multiplier(
                    epsilon_i, self.clipper.clip_value
                )
                self.history['noise_levels'].append(noise_mult)
                self.history['privacy_budgets'].append(epsilon_i)

            # 디버깅 정보 출력
            if round_t % 1 == 0:  # 매 라운드마다
                grad_norms_before = [np.linalg.norm(g) for g in local_gradients]
                grad_norms_after = [np.linalg.norm(g) for g in clipped_gradients]
                self._log(f"\n[Debug Round {round_t}]")
                self._log(f"  Selected clients: {len(selected_clients)}")
                self._log(f"  Epsilon base: {self.privacy_allocator.epsilon_base:.4f}")
                self._log(f"  Current clip value: {self.clipper.clip_value:.4f}")
                self._log(f"  Avg gradient norm before clip: {np.mean(grad_norms_before):.2f}")
                self._log(f"  Max gradient norm before clip: {np.max(grad_norms_before):.2f}")
                self._log(f"  Avg gradient norm after clip: {np.mean(grad_norms_after):.2f}")
                if self.history['noise_levels']:
                    recent_noise = self.history['noise_levels'][-len(selected_clients):]
                    self._log(f"  Avg noise level: {np.mean(recent_noise):.2f}")

            # 7. 연합 평균
            aggregated_gradient = np.mean(noisy_gradients, axis=0)

            # 8. 모델 업데이트
            self.update_global_model(aggregated_gradient)

            # 9. 평가 (10 라운드마다)
            if round_t % 10 == 0 or round_t == self.num_rounds - 1:
                accuracy, loss = self.evaluate()
                self.history['test_accuracy'].append(accuracy)
                self.history['test_loss'].append(loss)

                # 통계 기록
                stats = self.tracker.get_statistics()
                self.history['participation_stats'].append(stats)

                # 중간 체크포인트 검증
                check_milestone(self.dataset_name, round_t, accuracy, logger=self.logger)

                self._log(f"\nRound {round_t}: Acc={accuracy:.4f}, Loss={loss:.4f}, "
                          f"Clip={current_clip:.4f}")

                # 조기 종료 체크
                if self.check_convergence(accuracy):
                    self._log(f"\n목표 성능 달성! Round {round_t}")
                    break

        # 최종 평가
        final_accuracy, final_loss = self.evaluate()
        self._log("\n" + "=" * 60)
        self._log(f"최종 결과: Accuracy={final_accuracy:.4f}, Loss={final_loss:.4f}")
        self._log("=" * 60)

        return self.history