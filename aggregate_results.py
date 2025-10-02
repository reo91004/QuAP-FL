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
import json
from pathlib import Path
from tabulate import tabulate

from config import TARGET_ACCURACY, VISUALIZATION_CONFIG, OUTPUT_CONFIG
from utils import plot_multi_seed_comparison


def aggregate_results(dataset: str, result_dir: str = './results', seeds: list = None):
    """
    여러 시드의 실험 결과를 집계

    Args:
        dataset: 'mnist' or 'cifar10'
        result_dir: 결과 파일이 저장된 디렉토리
        seeds: 시드 리스트 (None이면 자동 탐지)
    """
    if seeds is None:
        # 결과 파일 자동 탐지 (JSON 형식)
        pattern = f'{dataset}_seed*.json'
        result_files = list(Path(result_dir).glob(pattern))

        if len(result_files) == 0:
            print(f"No result files found for {dataset} in {result_dir}")
            print("Expected format: {dataset}_seed{number}_{timestamp}.json")
            return

        seeds = []
        for f in result_files:
            # 파일명에서 시드 추출
            name = f.stem  # 파일명 (확장자 제외)
            parts = name.split('_')
            for part in parts:
                if part.startswith('seed'):
                    try:
                        seed = int(part.replace('seed', ''))
                        if seed not in seeds:
                            seeds.append(seed)
                    except ValueError:
                        continue

        seeds = sorted(seeds)

    print("=" * 60)
    print(f"결과 집계: {dataset.upper()}")
    print("=" * 60)
    print(f"시드: {seeds}")
    print(f"실험 수: {len(seeds)}")

    # 결과 로드
    all_results = []
    all_histories = []

    for seed in seeds:
        # 해당 시드의 가장 최근 파일 찾기
        pattern = f'{dataset}_seed{seed}_*.json'
        matching_files = list(Path(result_dir).glob(pattern))

        if len(matching_files) == 0:
            print(f"Warning: No results found for seed {seed}, skipping...")
            continue

        # 가장 최근 파일 선택 (타임스탬프 기준)
        latest_file = sorted(matching_files, key=lambda x: x.stem.split('_')[-1])[-1]

        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # history 추출
        if 'history' in data:
            history = data['history']
            all_results.append(data)
            all_histories.append(history)
        else:
            print(f"Warning: {latest_file} has no history, skipping...")
            continue

    if len(all_results) == 0:
        print("No valid results found")
        return

    print(f"유효한 결과: {len(all_histories)}개")

    # 최종 정확도 수집
    final_accuracies = []
    final_losses = []

    for history in all_histories:
        if 'test_accuracy' in history and len(history['test_accuracy']) > 0:
            final_accuracies.append(history['test_accuracy'][-1])
        if 'test_loss' in history and len(history['test_loss']) > 0:
            final_losses.append(history['test_loss'][-1])

    # 통계 계산
    print("\n" + "=" * 60)
    print("최종 결과 통계")
    print("=" * 60)

    table_data = []

    if len(final_accuracies) > 0:
        mean_acc = np.mean(final_accuracies)
        std_acc = np.std(final_accuracies)
        min_acc = np.min(final_accuracies)
        max_acc = np.max(final_accuracies)

        table_data.append(["Final Accuracy (Mean)", f"{mean_acc:.4f} ({mean_acc*100:.2f}%)"])
        table_data.append(["Final Accuracy (Std)", f"{std_acc:.4f}"])
        table_data.append(["Final Accuracy (Min)", f"{min_acc:.4f} ({min_acc*100:.2f}%)"])
        table_data.append(["Final Accuracy (Max)", f"{max_acc:.4f} ({max_acc*100:.2f}%)"])
        table_data.append(["95% CI", f"[{mean_acc - 1.96*std_acc:.4f}, {mean_acc + 1.96*std_acc:.4f}]"])

    if len(final_losses) > 0:
        mean_loss = np.mean(final_losses)
        std_loss = np.std(final_losses)

        table_data.append(["Final Loss (Mean)", f"{mean_loss:.4f}"])
        table_data.append(["Final Loss (Std)", f"{std_loss:.4f}"])

    # 목표 달성 여부
    if dataset in TARGET_ACCURACY and len(final_accuracies) > 0:
        target = TARGET_ACCURACY[dataset]
        success_count = sum(acc >= target for acc in final_accuracies)
        success_rate = success_count / len(final_accuracies)

        table_data.append(["Target Accuracy", f"{target:.3f} ({target*100:.1f}%)"])
        table_data.append(["Achievement Rate", f"{success_rate:.1%} ({success_count}/{len(final_accuracies)})"])

    # 테이블 출력
    print("\n" + tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    print()

    # 수렴 속도 분석
    convergence_rounds = []
    for history in all_histories:
        if 'test_accuracy' not in history or len(history['test_accuracy']) == 0:
            continue

        # 목표의 90%에 도달하는 라운드 찾기
        target_90 = TARGET_ACCURACY.get(dataset, 0.9) * 0.9
        accuracies = history['test_accuracy']

        for i, acc in enumerate(accuracies):
            if acc >= target_90:
                convergence_rounds.append(i * 10)  # 10 라운드마다 평가
                break

    if len(convergence_rounds) > 0:
        print("=" * 60)
        print("수렴 속도 (목표의 90% 달성)")
        print("=" * 60)
        conv_data = [
            ["평균 라운드", f"{np.mean(convergence_rounds):.1f}"],
            ["표준편차", f"{np.std(convergence_rounds):.1f}"],
        ]
        print(tabulate(conv_data, tablefmt="simple"))
        print()

    # 시각화 생성
    if VISUALIZATION_CONFIG.get('enabled', True) and OUTPUT_CONFIG.get('save_plots', True):
        print("=" * 60)
        print("다중 시드 비교 시각화 생성 중...")
        print("=" * 60)
        try:
            plot_path = plot_multi_seed_comparison(
                results=all_histories,
                seeds=seeds,
                dataset_name=dataset,
                config=VISUALIZATION_CONFIG,
                save_path=None  # 자동 경로 생성
            )
            print(f"시각화 저장: {plot_path}")
        except Exception as e:
            print(f"시각화 생성 실패: {e}")

    print("\n" + "=" * 60)
    print("집계 완료")
    print("=" * 60)


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