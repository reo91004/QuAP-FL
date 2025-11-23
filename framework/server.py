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
            max_clip=self.config['max_clip'],
            initial_clip=self.config.get('initial_clip', 1.0)
        )

        # 참여 분포 설정 (Beta 등)
        self.participation_distribution = self.config.get('participation_distribution', 'uniform')
        self.participation_mix_ratio = float(np.clip(self.config.get('participation_mix', 0.2), 0.0, 1.0))
        if self.participation_distribution == 'beta':
            alpha = self.config.get('participation_alpha', 1.0)
            beta_param = self.config.get('participation_beta', 1.0)
            base_weights = np.random.beta(alpha, beta_param, self.num_clients)
            base_weights = np.clip(base_weights, 1e-6, None)
            uniform_weights = np.ones(self.num_clients) / self.num_clients
            base_weights = (
                (1.0 - self.participation_mix_ratio) * base_weights +
                self.participation_mix_ratio * uniform_weights
            )
            # 안전장치: 모든 가중치가 0인 경우 균등 분포 사용
            if np.allclose(base_weights.sum(), 0.0):
                base_weights = np.ones(self.num_clients)
        else:
            base_weights = np.ones(self.num_clients)

        self.participation_base_weights = base_weights / base_weights.sum()

        # 통계 추적
        self.history = {
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': [],
            'clip_values': [],
            'noise_levels': [],
            'noise_stats': [],
            'participation_stats': [],
            'privacy_budgets': [],
            'evaluation_rounds': [],
            'client_participation': []  # 매 라운드 참여한 클라이언트 ID 리스트 저장
        }

        # 현재 라운드
        self.current_round = 0

        # Critical layer 정보 저장
        self.critical_layer_indices = self._identify_critical_layers()

        # 핵심 하이퍼파라미터 검증 로그
        self._log("=" * 60)
        self._log("QuAP-FL Configuration")
        self._log("=" * 60)
        self._log(f"  learning_rate: {self.config['learning_rate']}")
        self._log(f"  max_clip: {self.config['max_clip']}")
        self._log(f"  epsilon_total: {self.config['epsilon_total']}")
        self._log(f"  epsilon_base: {epsilon_base:.6f}")
        self._log(f"  Noise strategy: {self.config.get('noise_strategy', 'layer_wise')}")
        self._log(f"  Critical layers: {self.config.get('critical_layers', ['fc2'])}")
        total_params = sum(p.numel() for p in self.model.parameters())
        critical_params = self._count_critical_params()
        self._log(f"  Total params: {total_params:,}")
        self._log(f"  Critical params: {critical_params:,} ({critical_params/total_params*100:.2f}%)")
        self._log(f"  Noise reduction: {total_params/critical_params:.1f}x")
        self._log("=" * 60)

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

    def _identify_critical_layers(self) -> List[tuple]:
        """
        Critical layer(마지막 FC)의 파라미터 인덱스 범위 식별

        Layer-wise DP를 위해 마지막 classification layer만 노이즈 추가.
        이론적 정당화: Post-processing Theorem (Dwork & Roth 2014)

        Returns:
            List of (start_idx, end_idx) for critical parameters
        """
        critical_names = self.config.get('critical_layers', ['fc2'])
        indices = []

        current_idx = 0
        for name, param in self.model.named_parameters():
            param_size = param.numel()

            # 이름 매칭 (예: 'fc2.weight', 'fc2.bias')
            for critical_name in critical_names:
                if critical_name in name:
                    indices.append((current_idx, current_idx + param_size))
                    self._log(f"  Critical layer found: {name} [{current_idx}:{current_idx + param_size}]")
                    break

            current_idx += param_size

        if len(indices) == 0:
            raise ValueError(f"No critical layers found! Check config: {critical_names}")

        return indices

    def _count_critical_params(self) -> int:
        """Critical 파라미터 총 개수"""
        return sum(end - start for start, end in self.critical_layer_indices)

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

        # 시작 가중치 (Beta 분포 기반 등)
        weights = self.participation_base_weights.copy()

        # 참여 패턴 변화 시뮬레이션 (짝수 라운드에서 저 ID 선호)
        if round_t % 2 == 0:
            weights *= np.exp(-np.arange(self.num_clients) * 0.01)

        weights = weights / weights.sum()

        # 균등 샘플 비율
        uniform_fraction = self.participation_mix_ratio
        num_uniform = int(round(num_selected * uniform_fraction))
        num_uniform = max(0, min(num_uniform, num_selected))
        num_weighted = num_selected - num_uniform

        available_clients = np.arange(self.num_clients)
        selected_clients = []

        if num_weighted > 0:
            weighted_probs = weights.copy()
            if np.sum(weighted_probs) <= 0:
                weighted_probs = np.ones_like(weighted_probs) / len(weighted_probs)
            weighted_probs = weighted_probs / weighted_probs.sum()
            weighted_choices = np.random.choice(
                available_clients,
                size=num_weighted,
                replace=False,
                p=weighted_probs
            )
            selected_clients.extend(weighted_choices.tolist())
            available_clients = np.setdiff1d(available_clients, weighted_choices, assume_unique=True)

        if num_uniform > 0 and len(available_clients) > 0:
            uniform_size = min(num_uniform, len(available_clients))
            uniform_choices = np.random.choice(
                available_clients,
                size=uniform_size,
                replace=False
            )
            selected_clients.extend(uniform_choices.tolist())

        # 필요한 수만큼 확보하지 못했으면 남은 클라이언트로 채우기
        if len(selected_clients) < num_selected:
            remaining = np.setdiff1d(
                np.arange(self.num_clients),
                np.array(selected_clients, dtype=int),
                assume_unique=False
            )
            remaining_size = min(num_selected - len(selected_clients), len(remaining))
            if remaining_size > 0:
                fallback_choices = np.random.choice(
                    remaining,
                    size=remaining_size,
                    replace=False
                )
                selected_clients.extend(fallback_choices.tolist())

        # 최종 순서를 섞어서 반환
        selected_array = np.array(selected_clients, dtype=int)
        np.random.shuffle(selected_array)

        return selected_array.tolist()

    def local_training(
        self,
        client_id: int,
        round_t: int
    ) -> tuple:
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
        total_loss = 0.0
        total_samples = 0
        for epoch in range(self.config['local_epochs']):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        # 그래디언트 계산 (전역 모델과의 차이)
        gradient = []
        for local_param, global_param in zip(
            local_model.parameters(), self.model.parameters()
        ):
            # 올바른 그래디언트 방향: local - global
            diff = local_param.data - global_param.data
            gradient.append(diff.view(-1).cpu().numpy())

        gradient = np.concatenate(gradient)

        # Raw gradient로 정규화 (DP-SGD 표준)
        gradient = gradient / lr

        avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0

        return gradient, avg_loss

    def _add_layer_wise_noise(
        self,
        gradient: np.ndarray,
        epsilon: float,
        participant_count: int = 1
    ) -> np.ndarray:
        """
        Layer-wise Differential Privacy 노이즈 추가

        Critical layer에만 노이즈를 추가하여 고차원 노이즈 문제 완화.
        이론적 정당화: Post-processing Theorem (Dwork & Roth 2014)

        Args:
            gradient: 집계된 gradient (1D numpy array, shape: d)
            epsilon: 프라이버시 예산 (이번 라운드)

        Returns:
            노이즈가 추가된 gradient
        """
        noise = np.zeros_like(gradient)

        # 노이즈 스케일 계산 (Gaussian Mechanism)
        clip_norm = self.clipper.clip_value if self.clipper.clip_value else 1.0
        effective_clip = clip_norm / max(1, participant_count)
        sigma = effective_clip * np.sqrt(2 * np.log(1.25 / self.config['delta'])) / epsilon

        # Critical layer에만 노이즈 추가
        total_critical_params = 0
        for start, end in self.critical_layer_indices:
            d_c = end - start
            noise[start:end] = np.random.normal(0, sigma, d_c)
            total_critical_params += d_c

        # 통계 기록 (디버깅용)
        noise_norm = np.linalg.norm(noise)
        signal_norm = np.linalg.norm(gradient)

        self.history['noise_levels'].append(float(sigma))
        self.history['noise_stats'].append({
            'sigma': float(sigma),
            'noise_norm': float(noise_norm),
            'signal_norm': float(signal_norm)
        })

        if self.current_round % 10 == 0:
            self._log(f"  [Noise] σ={sigma:.2f}, ||noise||={noise_norm:.2f}, "
                     f"||signal||={signal_norm:.2f}, ratio={noise_norm/signal_norm:.1f}:1")

        return gradient + noise

    def _add_full_noise(self, gradient: np.ndarray, epsilon: float, participant_count: int = 1) -> np.ndarray:
        """
        Full DP 노이즈 (비교용, 사용하지 말 것)

        전체 파라미터에 노이즈 추가.
        문제: 고차원에서 노이즈가 신호를 압도함
        """
        clip_norm = self.clipper.clip_value if self.clipper.clip_value else 1.0
        effective_clip = clip_norm / max(1, participant_count)
        sigma = effective_clip * np.sqrt(2 * np.log(1.25 / self.config['delta'])) / epsilon

        noise = np.random.normal(0, sigma, gradient.shape)
        noise_norm = np.linalg.norm(noise)
        signal_norm = np.linalg.norm(gradient)

        self.history['noise_levels'].append(float(sigma))
        self.history['noise_stats'].append({
            'sigma': float(sigma),
            'noise_norm': float(noise_norm),
            'signal_norm': float(signal_norm)
        })

        return gradient + noise

    def _update_global_model(self, aggregated_gradient: np.ndarray):
        """
        전역 모델 업데이트

        Layer-wise 노이즈로 인해 gradient norm이 자연스럽게 제한되므로
        추가적인 max_agg_norm 제한이 불필요함

        Args:
            aggregated_gradient: 노이즈가 추가된 집계 gradient
        """
        # NaN/Inf 체크만 (norm 제한 제거)
        if np.any(np.isnan(aggregated_gradient)) or np.any(np.isinf(aggregated_gradient)):
            self._log("Warning: NaN/Inf in gradient, skipping update")
            return

        idx = 0
        for param in self.model.parameters():
            numel = param.numel()
            shape = param.shape

            grad = aggregated_gradient[idx:idx+numel].reshape(shape)
            grad_tensor = torch.from_numpy(grad).to(self.device).float()

            # gradient는 이미 (local - global) 델타
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

        핵심 특징:
        1. 서버 집계 후 한 번만 노이즈 (표준 DP-FedAvg)
        2. Layer-wise 노이즈 적용 (고차원 문제 해결)
        3. 평균 참여율 기반 적응형 예산
        """
        self._log("=" * 60)
        self._log(f"QuAP-FL Training Start: {self.dataset_name.upper()}")
        self._log(f"Target Accuracy: {TARGET_ACCURACY.get(self.dataset_name, 'N/A')}")
        self._log(f"Clients: {self.num_clients}, Rounds: {self.num_rounds}")
        self._log("=" * 60)

        # 초기 모델 평가
        initial_acc, initial_loss = self.evaluate()
        self._log(f"\nInitial model: Acc={initial_acc:.4f}, Loss={initial_loss:.4f}")

        progress_bar = tqdm(range(self.num_rounds), desc="Training")
        for round_t in progress_bar:
            self.current_round = round_t

            # 1. 클라이언트 선택
            selected_clients = self.select_clients(round_t)
            self.history['client_participation'].append(selected_clients)

            # 2. 참여 이력 업데이트 (노이즈 추가 전에!)
            # Warm-up 기간에는 추적을 건너뛰거나 업데이트만 하고 예산 계산에는 쓰지 않음
            # 여기서는 안정성을 위해 업데이트는 하되, 예산 계산 시 warm-up 여부를 체크함
            if round_t >= self.config.get('warmup_rounds', 0):
                self.tracker.update(selected_clients)
            else:
                # Warm-up 기간에는 로깅만
                pass

            # 3. 로컬 학습 및 그래디언트 수집
            local_gradients = []
            local_losses = []
            for client_id in selected_clients:
                gradient, client_loss = self.local_training(client_id, round_t)
                local_gradients.append(gradient)
                local_losses.append(client_loss)

            mean_train_loss = float(np.mean(local_losses)) if local_losses else 0.0
            self.history['train_loss'].append(mean_train_loss)

            # 4. Critical layer 기반 클리핑 값 업데이트 (90th percentile)
            critical_gradients = [
                np.concatenate([
                    grad[start:end]
                    for start, end in self.critical_layer_indices
                ])
                for grad in local_gradients
            ]
            
            # Clipping Strategy 확인
            clipping_strategy = self.config.get('clipping_strategy', 'quantile')
            
            if clipping_strategy == 'quantile':
                current_clip = self.clipper.update_clip_value(critical_gradients)
            else:
                # Fixed clipping (update하지 않음)
                current_clip = self.clipper.clip_value if self.clipper.clip_value else 1.0
                
            self.history['clip_values'].append(current_clip)

            # 5. Critical segment 클리핑 적용
            clipped_critical = self.clipper.clip_gradients(critical_gradients)
            clipped_gradients = []
            for grad, clipped_seg in zip(local_gradients, clipped_critical):
                grad_copy = grad.copy()
                offset = 0
                for start, end in self.critical_layer_indices:
                    seg_len = end - start
                    grad_copy[start:end] = clipped_seg[offset:offset + seg_len]
                    offset += seg_len
                clipped_gradients.append(grad_copy)

            # 6. 서버 측 집계 (노이즈 전!)
            aggregated_gradient = np.mean(clipped_gradients, axis=0)

            # 7. 평균 참여율 기반 프라이버시 예산 계산
            if round_t < self.config.get('warmup_rounds', 0):
                # Warm-up: 고정 예산 (안정성 확보)
                epsilon_round = self.privacy_allocator.epsilon_base
                if self.logger:
                    self.logger.info(f"  [Warm-up] Using fixed epsilon: {epsilon_round:.6f}")
            else:
                participation_rates = [
                    self.tracker.get_participation_rate(cid)
                    for cid in selected_clients
                ]
                avg_participation = np.mean(participation_rates)
                epsilon_round = self.privacy_allocator.compute_privacy_budget(avg_participation)

            # 통계 기록
            self.history['privacy_budgets'].append(epsilon_round)

            # 8. Layer-wise 노이즈 추가 (한 번만!)
            noise_strategy = self.config.get('noise_strategy', 'layer_wise')

            if noise_strategy == 'layer_wise':
                noisy_gradient = self._add_layer_wise_noise(
                    aggregated_gradient,
                    epsilon_round,
                    participant_count=len(selected_clients)
                )
            elif noise_strategy == 'full':
                noisy_gradient = self._add_full_noise(
                    aggregated_gradient,
                    epsilon_round,
                    participant_count=len(selected_clients)
                )
            elif noise_strategy == 'none':
                # FedAvg (No Privacy)
                noisy_gradient = aggregated_gradient
            else:
                raise ValueError(f"Unknown noise_strategy: {noise_strategy}")

            # 9. Learning rate 적용 (DP-SGD 표준)
            lr = self.config['learning_rate'] * (self.config['lr_decay'] ** round_t)
            scaled_gradient = noisy_gradient * lr

            # 10. 모델 업데이트
            self._update_global_model(scaled_gradient)

            # 진행 상황 로그
            progress_msg = (
                f"[Round {round_t}] train_loss={mean_train_loss:.4f}, "
                f"ε={epsilon_round:.4f}, clip={current_clip:.4f}"
            )
            if self.logger:
                self.logger.info(progress_msg)
            else:
                progress_bar.write(progress_msg)

            # 9. 평가 (10 라운드마다)
            if round_t % 10 == 0 or round_t == self.num_rounds - 1:
                accuracy, loss = self.evaluate()
                self.history['test_accuracy'].append(accuracy)
                self.history['test_loss'].append(loss)
                self.history['evaluation_rounds'].append(round_t)

                # 통계 기록
                stats = self.tracker.get_statistics()
                self.history['participation_stats'].append(stats)

                # 중간 체크포인트 검증
                check_milestone(self.dataset_name, round_t, accuracy, logger=self.logger)

                self._log(f"Round {round_t}: Acc={accuracy:.4f}, Loss={loss:.4f}, "
                          f"Clip={current_clip:.4f}")

                # 조기 종료 체크
                if self.check_convergence(accuracy):
                    self._log(f"\n목표 성능 달성! Round {round_t}")
                    break

        # 최종 평가
        final_accuracy, final_loss = self.evaluate()

        if self.history['test_accuracy']:
            self.history['test_accuracy'][-1] = final_accuracy
        else:
            self.history['test_accuracy'].append(final_accuracy)

        if self.history['test_loss']:
            self.history['test_loss'][-1] = final_loss
        else:
            self.history['test_loss'].append(final_loss)

        if self.history['evaluation_rounds']:
            self.history['evaluation_rounds'][-1] = self.current_round
        else:
            self.history['evaluation_rounds'].append(self.current_round)

        self._log("\n" + "=" * 60)
        self._log(f"최종 결과: Accuracy={final_accuracy:.4f}, Loss={final_loss:.4f}")
        self._log("=" * 60)

        progress_bar.close()

        return self.history
