# tests/test_integration.py
"""
QuAP-FL 통합 테스트

전체 학습 파이프라인 검증 (20 rounds)
- Critical layer 식별
- Layer-wise 노이즈
- 학습 수렴 확인
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models import MNISTModel
from data import prepare_non_iid_data
from framework import QuAPFLServer
from config import HYPERPARAMETERS


class TestIntegration(unittest.TestCase):
    """전체 파이프라인 통합 테스트"""

    @classmethod
    def setUpClass(cls):
        """테스트 환경 설정"""
        print("\nQuAP-FL 통합 테스트 시작 (20 rounds)")
        print("=" * 60)

        # 시드 고정
        torch.manual_seed(42)
        np.random.seed(42)

        # 데이터 준비
        print("1. 데이터 준비 중...")
        cls.client_indices, cls.train_dataset, cls.test_dataset = prepare_non_iid_data(
            'mnist', num_clients=20, alpha=0.5, seed=42
        )
        print(f"   ✓ 20 clients, {len(cls.train_dataset)} train samples")

        # 모델
        print("2. 모델 초기화 중...")
        cls.model = MNISTModel()
        cls.total_params = sum(p.numel() for p in cls.model.parameters())
        print(f"   ✓ MNIST CNN, {cls.total_params:,} parameters")

        # 설정
        cls.config = HYPERPARAMETERS.copy()
        cls.config['num_clients'] = 20
        cls.config['num_rounds'] = 20  # 빠른 검증
        cls.config['clients_per_round'] = 6
        cls.config['noise_strategy'] = 'layer_wise'

        print("3. 서버 생성 중...")
        cls.server = QuAPFLServer(
            model=cls.model,
            train_dataset=cls.train_dataset,
            test_dataset=cls.test_dataset,
            client_indices=cls.client_indices,
            dataset_name='mnist',
            config=cls.config,
            device=torch.device('cpu')
        )

        # Critical layer 확인
        critical_params = cls.server._count_critical_params()
        print(f"   ✓ Critical params: {critical_params} ({critical_params/cls.total_params*100:.2f}%)")
        print(f"   ✓ Noise reduction: {cls.total_params/critical_params:.1f}x")

        # 학습
        print("4. 학습 시작 (20 rounds)...")
        print("-" * 60)
        cls.history = cls.server.train()
        print("-" * 60)

        cls.final_acc = cls.history['test_accuracy'][-1]
        cls.final_loss = cls.history['test_loss'][-1]

    def test_accuracy_threshold(self):
        """정확도 > 70% 확인"""
        print(f"\n정확도 테스트: {self.final_acc:.1%}")
        self.assertGreater(
            self.final_acc,
            0.70,
            f"20 rounds 후 정확도가 70%보다 낮음: {self.final_acc:.1%}"
        )
        print(f"   ✓ Accuracy > 70%: {self.final_acc:.1%}")

    def test_loss_threshold(self):
        """Loss < 1.0 확인"""
        print(f"\nLoss 테스트: {self.final_loss:.4f}")
        self.assertLess(
            self.final_loss,
            1.0,
            f"Loss가 1.0보다 큼: {self.final_loss:.4f}"
        )
        print(f"   ✓ Loss < 1.0: {self.final_loss:.4f}")

    def test_no_nan(self):
        """NaN 발생 없음 확인"""
        print(f"\nNaN 테스트")
        self.assertFalse(
            np.isnan(self.final_acc),
            "Accuracy에 NaN 발생"
        )
        self.assertFalse(
            np.isnan(self.final_loss),
            "Loss에 NaN 발생"
        )
        print(f"   ✓ No NaN values")

    def test_critical_layer_setup(self):
        """Critical layer 설정 확인"""
        print(f"\nCritical layer 설정 테스트")
        self.assertGreater(
            len(self.server.critical_layer_indices),
            0,
            "Critical layer가 식별되지 않음"
        )
        critical_params = self.server._count_critical_params()
        self.assertEqual(
            critical_params,
            1290,
            f"FC2 파라미터 수 불일치: {critical_params} (예상: 1290)"
        )
        print(f"   ✓ Critical layer: {critical_params} params")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("QuAP-FL 통합 테스트")
    print("=" * 60)
    unittest.main(verbosity=2)
