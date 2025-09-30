# main.py

"""
QuAP-FL 메인 실행 스크립트

사용법:
    python main.py --dataset mnist --seed 42
    python main.py --dataset cifar10 --seed 42
    python main.py --validate_only
"""

import argparse
import numpy as np
import torch
import random
import sys
import os
import json
import logging
from datetime import datetime

# 현재 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models import MNISTModel, CIFAR10Model
from data import prepare_non_iid_data
from framework import QuAPFLServer
from config import HYPERPARAMETERS, TARGET_ACCURACY
from utils import validate_implementation


def set_seed(seed: int):
    """
    재현성을 위한 시드 고정

    Args:
        seed: 랜덤 시드
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_path: str):
    """
    로깅 시스템 설정 (터미널 + 파일)

    Args:
        log_path: 로그 파일 경로
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 포맷 설정
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 파일 핸들러
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 터미널 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description='QuAP-FL: Quantile-based Adaptive Privacy FL')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only run validation tests')
    parser.add_argument('--num_clients', type=int, default=None,
                        help='Number of clients (default: from config)')
    parser.add_argument('--num_rounds', type=int, default=None,
                        help='Number of training rounds (default: from config)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')

    args = parser.parse_args()

    # 검증 모드
    if args.validate_only:
        print("\n구현 검증 모드 실행...")
        success = validate_implementation()
        if success:
            print("\n✓ 모든 검증 테스트 통과!")
            sys.exit(0)
        else:
            print("\n✗ 검증 실패 - 구현을 확인하세요")
            sys.exit(1)

    # 데이터셋 필수
    if args.dataset is None:
        parser.error("--dataset is required when not using --validate_only")

    # 시드 설정
    set_seed(args.seed)

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 설정 오버라이드
    config = HYPERPARAMETERS.copy()
    if args.num_clients is not None:
        config['num_clients'] = args.num_clients
    if args.num_rounds is not None:
        config['num_rounds'] = args.num_rounds

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 타임스탬프 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 로깅 설정
    log_path = os.path.join(args.output_dir, f'{args.dataset}_seed{args.seed}_{timestamp}.log')
    logger = setup_logging(log_path)
    logger.info(f"로그 파일 생성: {log_path}")

    # 데이터 준비
    logger.info(f"데이터셋 준비: {args.dataset.upper()}")
    print(f"Non-IID 분할 (Dirichlet α=0.5) with {config['num_clients']} clients")

    client_indices, train_dataset, test_dataset = prepare_non_iid_data(
        args.dataset,
        config['num_clients'],
        alpha=0.5,
        data_dir='./data',
        seed=args.seed
    )

    # 데이터 분포 확인
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Samples per client: {len(train_dataset) // config['num_clients']} (avg)")

    # 모델 초기화
    if args.dataset == 'mnist':
        model = MNISTModel()
    elif args.dataset == 'cifar10':
        model = CIFAR10Model()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Model parameters: {model.get_num_parameters():,}")

    # QuAP-FL 서버 초기화
    server = QuAPFLServer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_indices=client_indices,
        dataset_name=args.dataset,
        config=config,
        device=device
    )

    # 학습 시작
    print("\n" + "=" * 60)
    print(f"QuAP-FL 학습 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Target: {TARGET_ACCURACY.get(args.dataset, 'N/A')}")
    print(f"Privacy budget (ε): {config['epsilon_total']}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # 학습 실행
    history = server.train()

    # 최종 결과
    final_acc = history['test_accuracy'][-1] if history['test_accuracy'] else 0
    final_loss = history['test_loss'][-1] if history['test_loss'] else 0

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"최종 정확도: {final_acc:.4f}")
    print(f"최종 손실: {final_loss:.4f}")

    # 목표 달성 확인
    if args.dataset in TARGET_ACCURACY:
        target = TARGET_ACCURACY[args.dataset]
        if abs(final_acc - target) <= 0.01:
            print(f"✓ 목표 성능 달성! (목표: {target:.3f})")
        else:
            diff = (final_acc - target) * 100
            if diff > 0:
                print(f"✓ 목표 초과 달성! (+{diff:.1f}%)")
            else:
                print(f"✗ 목표 미달성 (목표: {target:.3f}, 차이: {diff:.1f}%)")

    # 결과 저장 (JSON)
    result_data = {
        'metadata': {
            'dataset': args.dataset,
            'seed': args.seed,
            'timestamp': timestamp,
            'target_accuracy': TARGET_ACCURACY.get(args.dataset),
            'final_accuracy': float(final_acc),
            'final_loss': float(final_loss),
            'hyperparameters': config
        },
        'history': {
            'test_accuracy': [float(x) for x in history.get('test_accuracy', [])],
            'test_loss': [float(x) for x in history.get('test_loss', [])],
            'clip_values': [float(x) for x in history.get('clip_values', [])],
            'noise_levels': [float(x) for x in history.get('noise_levels', [])],
            'privacy_budgets': [float(x) for x in history.get('privacy_budgets', [])],
            'participation_stats': history.get('participation_stats', [])
        }
    }

    result_path = os.path.join(
        args.output_dir,
        f'{args.dataset}_seed{args.seed}_{timestamp}.json'
    )

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    logger.info(f"결과 저장: {result_path}")
    print(f"결과 저장: {result_path}")

    # 참여 통계
    if history['participation_stats']:
        final_stats = history['participation_stats'][-1]
        print("\n참여 통계:")
        print(f"  - 평균 참여율: {final_stats['mean_participation_rate']:.3f}")
        print(f"  - 참여 클라이언트: {final_stats['participating_clients']}/{config['num_clients']}")

    # 프라이버시 통계
    if history['privacy_budgets']:
        mean_budget = np.mean(history['privacy_budgets'])
        print(f"\n프라이버시 예산:")
        print(f"  - 평균 라운드당 예산: {mean_budget:.6f}")
        print(f"  - 총 예산: {config['epsilon_total']:.1f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()