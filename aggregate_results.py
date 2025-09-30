# aggregate_results.py

"""
다중 시드 실험 결과 집계

사용법:
    python aggregate_results.py --dataset mnist
    python aggregate_results.py --dataset cifar10
"""

import argparse
import numpy as np
import os
from pathlib import Path


def aggregate_results(dataset: str, result_dir: str = './results', seeds: list = None):
    """
    여러 시드의 실험 결과를 집계

    Args:
        dataset: 'mnist' or 'cifar10'
        result_dir: 결과 파일이 저장된 디렉토리
        seeds: 시드 리스트 (None이면 자동 탐지)
    """
    if seeds is None:
        # 결과 파일 자동 탐지
        pattern = f'results_{dataset}_seed*.npy'
        result_files = list(Path(result_dir).glob(pattern))

        if len(result_files) == 0:
            print(f"No result files found for {dataset}")
            return

        seeds = []
        for f in result_files:
            # 파일명에서 시드 추출
            name = f.stem
            seed_str = name.split('seed')[-1]
            try:
                seed = int(seed_str)
                seeds.append(seed)
            except ValueError:
                continue

        seeds = sorted(seeds)
    else:
        result_files = [
            Path(result_dir) / f'results_{dataset}_seed{seed}.npy'
            for seed in seeds
        ]

    print("=" * 60)
    print(f"결과 집계: {dataset.upper()}")
    print("=" * 60)
    print(f"시드: {seeds}")
    print(f"실험 수: {len(seeds)}")

    # 결과 로드
    all_results = []
    for seed in seeds:
        result_path = Path(result_dir) / f'results_{dataset}_seed{seed}.npy'
        if not result_path.exists():
            print(f"Warning: {result_path} not found, skipping...")
            continue

        result = np.load(result_path, allow_pickle=True).item()
        all_results.append(result)

    if len(all_results) == 0:
        print("No valid results found")
        return

    print(f"유효한 결과: {len(all_results)}개")

    # 최종 정확도 수집
    final_accuracies = []
    final_losses = []

    for result in all_results:
        if 'test_accuracy' in result and len(result['test_accuracy']) > 0:
            final_accuracies.append(result['test_accuracy'][-1])
        if 'test_loss' in result and len(result['test_loss']) > 0:
            final_losses.append(result['test_loss'][-1])

    # 통계 계산
    print("\n" + "=" * 60)
    print("최종 결과 통계")
    print("=" * 60)

    if len(final_accuracies) > 0:
        mean_acc = np.mean(final_accuracies)
        std_acc = np.std(final_accuracies)
        min_acc = np.min(final_accuracies)
        max_acc = np.max(final_accuracies)

        print(f"정확도:")
        print(f"  - 평균: {mean_acc:.4f}")
        print(f"  - 표준편차: {std_acc:.4f}")
        print(f"  - 최소: {min_acc:.4f}")
        print(f"  - 최대: {max_acc:.4f}")
        print(f"  - 95% 신뢰구간: [{mean_acc - 1.96*std_acc:.4f}, {mean_acc + 1.96*std_acc:.4f}]")

    if len(final_losses) > 0:
        mean_loss = np.mean(final_losses)
        std_loss = np.std(final_losses)

        print(f"\n손실:")
        print(f"  - 평균: {mean_loss:.4f}")
        print(f"  - 표준편차: {std_loss:.4f}")

    # 목표 달성 여부
    from config.hyperparameters import TARGET_ACCURACY
    if dataset in TARGET_ACCURACY:
        target = TARGET_ACCURACY[dataset]
        success_rate = sum(acc >= target for acc in final_accuracies) / len(final_accuracies)

        print(f"\n목표 달성률:")
        print(f"  - 목표: {target:.3f}")
        print(f"  - 달성률: {success_rate:.1%} ({sum(acc >= target for acc in final_accuracies)}/{len(final_accuracies)})")

    # 수렴 속도 분석
    convergence_rounds = []
    for result in all_results:
        if 'test_accuracy' not in result or len(result['test_accuracy']) == 0:
            continue

        # 목표의 90%에 도달하는 라운드 찾기
        target_90 = TARGET_ACCURACY.get(dataset, 0.9) * 0.9
        accuracies = result['test_accuracy']

        for i, acc in enumerate(accuracies):
            if acc >= target_90:
                convergence_rounds.append(i * 10)  # 10 라운드마다 평가
                break

    if len(convergence_rounds) > 0:
        print(f"\n수렴 속도 (목표의 90% 달성):")
        print(f"  - 평균 라운드: {np.mean(convergence_rounds):.1f}")
        print(f"  - 표준편차: {np.std(convergence_rounds):.1f}")

    # 참여 통계
    print("\n" + "=" * 60)
    print("참여 통계")
    print("=" * 60)

    for i, result in enumerate(all_results):
        if 'participation_stats' in result and len(result['participation_stats']) > 0:
            final_stats = result['participation_stats'][-1]
            print(f"\n시드 {seeds[i]}:")
            print(f"  - 평균 참여율: {final_stats['mean_participation_rate']:.3f}")
            print(f"  - 참여 클라이언트: {final_stats['participating_clients']}")

    # 프라이버시 통계
    print("\n" + "=" * 60)
    print("프라이버시 통계")
    print("=" * 60)

    all_budgets = []
    for result in all_results:
        if 'privacy_budgets' in result:
            all_budgets.extend(result['privacy_budgets'])

    if len(all_budgets) > 0:
        print(f"라운드당 프라이버시 예산:")
        print(f"  - 평균: {np.mean(all_budgets):.6f}")
        print(f"  - 표준편차: {np.std(all_budgets):.6f}")
        print(f"  - 최소: {np.min(all_budgets):.6f}")
        print(f"  - 최대: {np.max(all_budgets):.6f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Aggregate QuAP-FL results')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar10'],
                        help='Dataset name')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Result directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to aggregate (default: auto-detect)')

    args = parser.parse_args()

    aggregate_results(args.dataset, args.result_dir, args.seeds)


if __name__ == '__main__':
    main()