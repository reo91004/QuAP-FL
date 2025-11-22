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
        # 1. Check if result_dir is a timestamped folder or root
        if os.path.basename(result_dir).isdigit() and '_' in os.path.basename(result_dir):
             # It's likely a timestamped folder
             search_dir = result_dir
        else:
             # It's likely root, try to find latest timestamped folder
             subdirs = [d for d in Path(result_dir).iterdir() if d.is_dir() and d.name[0].isdigit()]
             if subdirs:
                 latest_subdir = sorted(subdirs, key=lambda x: x.name)[-1]
                 print(f"Auto-detected latest session: {latest_subdir}")
                 search_dir = str(latest_subdir)
             else:
                 search_dir = result_dir

        pattern = f'{dataset}_*.json'
        result_files = list(Path(search_dir).glob(pattern))

        if len(result_files) == 0:
            print(f"No result files found for {dataset} in {search_dir}")
            return

        seeds = []
        for f in result_files:
            # 파일명에서 시드 추출
            # format: {dataset}_{mode}_seed{seed}_{timestamp}.json
            name = f.stem
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
        result_dir = search_dir # Update result_dir for later use

    print("=" * 60)
    print(f"결과 집계: {dataset.upper()}")
    print("=" * 60)
    print(f"시드: {seeds}")
    print(f"실험 수: {len(seeds)}")

    # 결과 로드
    all_results = []
    all_histories = []

    for seed in seeds:
        # 해당 시드의 모든 결과 파일 수집 (모드별, 타임스탬프별)
        pattern = f'{dataset}_*_seed{seed}_*.json'
        matching_files = list(Path(result_dir).glob(pattern))

        if len(matching_files) == 0:
            print(f"Warning: No results found for seed {seed}, skipping...")
            continue

        # 모든 파일 로드
        for file_path in matching_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'history' in data:
                    all_results.append(data)
                    all_histories.append(data['history'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if len(all_results) == 0:
        print("No valid results found")
        return

    print(f"유효한 결과: {len(all_histories)}개")

    # 모드별로 결과 분류
    results_by_mode = {}
    
    for data in all_results:
        # 파일명에서 모드 추출 (예: mnist_fedavg_seed42_...)
        # 메타데이터에 모드 정보가 없으면 파일명 추론
        mode = 'unknown'
        if 'hyperparameters' in data['metadata']:
            if data['metadata']['hyperparameters'].get('noise_strategy') == 'none':
                mode = 'FedAvg'
            elif data['metadata']['hyperparameters'].get('alpha') == 0.0:
                mode = 'Fixed-DP'
            else:
                mode = 'QuAP-FL'
        
        if mode not in results_by_mode:
            results_by_mode[mode] = {'accuracies': [], 'losses': [], 'histories': []}
            
        history = data['history']
        if 'test_accuracy' in history and len(history['test_accuracy']) > 0:
            results_by_mode[mode]['accuracies'].append(history['test_accuracy'][-1])
            results_by_mode[mode]['histories'].append(history)
        if 'test_loss' in history and len(history['test_loss']) > 0:
            results_by_mode[mode]['losses'].append(history['test_loss'][-1])

    # 통계 계산 및 출력
    print("\n" + "=" * 80)
    print(f"실험 결과 요약 ({dataset.upper()})")
    print("=" * 80)

    headers = ["Mode", "Exp Count", "Final Acc (Mean)", "Final Acc (Max)", "Final Loss", "Target Reached"]
    table_data = []

    target = TARGET_ACCURACY.get(dataset, 0.0)

    for mode, stats in results_by_mode.items():
        accs = stats['accuracies']
        losses = stats['losses']
        
        if not accs:
            continue
            
        mean_acc = np.mean(accs)
        max_acc = np.max(accs)
        mean_loss = np.mean(losses)
        success_count = sum(a >= target for a in accs)
        success_rate = f"{success_count}/{len(accs)}"
        
        table_data.append([
            mode,
            len(accs),
            f"{mean_acc:.4f} ({mean_acc*100:.2f}%)",
            f"{max_acc:.4f}",
            f"{mean_loss:.4f}",
            success_rate
        ])

    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print()

    # 상세 통계 (QuAP-FL 기준)
    if 'QuAP-FL' in results_by_mode:
        print("QuAP-FL 상세 분석:")
        accs = results_by_mode['QuAP-FL']['accuracies']
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"  Mean: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  95% CI: [{mean_acc - 1.96*std_acc:.4f}, {mean_acc + 1.96*std_acc:.4f}]")
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