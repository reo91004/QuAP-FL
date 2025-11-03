# utils/visualization.py

"""
QuAP-FL 시각화 유틸리티

주요 기능:
- plot_training_history: 단일 실험 결과 시각화 (4-subplot)
- plot_multi_seed_comparison: 다중 시드 결과 비교 시각화
- generate_summary_table: 결과 테이블 생성
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from tabulate import tabulate


def plot_training_history(
    history: Dict,
    dataset_name: str,
    seed: int,
    config: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    단일 실험의 학습 히스토리 시각화 (4-subplot)

    Args:
        history: 학습 히스토리 딕셔너리
        dataset_name: 데이터셋 이름 ('mnist' or 'cifar10')
        seed: 랜덤 시드
        config: 시각화 설정 (VISUALIZATION_CONFIG)
        save_path: 저장 경로 (None이면 자동 생성)

    Returns:
        저장된 파일 경로
    """
    # 스타일 설정
    plt.style.use(config.get('style', 'seaborn-v0_8'))

    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=config.get('figsize', (12, 10)))

    # 평가 라운드 (history에 존재하면 사용, 없으면 10라운드 간격 가정)
    evaluation_rounds = history.get('evaluation_rounds', [])
    if evaluation_rounds:
        rounds = evaluation_rounds
    else:
        eval_interval = 10
        rounds = [i * eval_interval for i in range(len(history.get('test_accuracy', [])))]

    # 1. Test Accuracy
    ax = axes[0, 0]
    if 'test_accuracy' in history and len(history['test_accuracy']) > 0:
        ax.plot(rounds, history['test_accuracy'], marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title('Test Accuracy over Rounds', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 목표선 추가
        from config import TARGET_ACCURACY
        if dataset_name in TARGET_ACCURACY:
            target = TARGET_ACCURACY[dataset_name]
            ax.axhline(y=target, color='red', linestyle='--', linewidth=2,
                      label=f'Target: {target:.3f}')
            ax.legend(fontsize=10)

    # 2. Test Loss (log scale)
    ax = axes[0, 1]
    if 'test_loss' in history and len(history['test_loss']) > 0:
        ax.semilogy(rounds, history['test_loss'], marker='s', linewidth=2,
                   markersize=6, color='orange')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Test Loss (log scale)', fontsize=11)
        ax.set_title('Test Loss over Rounds', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 3. Privacy Budget Consumption
    ax = axes[1, 0]
    if 'privacy_budgets' in history and len(history['privacy_budgets']) > 0:
        # 누적 프라이버시 예산 계산
        cumulative_epsilon = np.cumsum(history['privacy_budgets'])
        if rounds:
            privacy_at_eval = [
                float(cumulative_epsilon[min(round_idx, len(cumulative_epsilon) - 1)])
                for round_idx in rounds
            ]
        else:
            privacy_at_eval = cumulative_epsilon.tolist()
            rounds = list(range(len(privacy_at_eval)))

        ax.plot(rounds, privacy_at_eval, marker='^', linewidth=2,
               markersize=6, color='green')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Cumulative ε', fontsize=11)
        ax.set_title('Privacy Budget Consumption', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 총 예산선 추가
        from config import HYPERPARAMETERS
        total_epsilon = HYPERPARAMETERS.get('epsilon_total', 3.0)
        ax.axhline(y=total_epsilon, color='red', linestyle='--', linewidth=2,
                  label=f'Total Budget: {total_epsilon}')
        ax.legend(fontsize=10)

    # 4. Adaptive Clipping Values
    ax = axes[1, 1]
    if 'clip_values' in history and len(history['clip_values']) > 0:
        # clip_values는 모든 라운드에 대해 기록되므로 샘플링 필요
        if rounds:
            clip_at_eval = [
                history['clip_values'][round_idx]
                for round_idx in rounds
                if round_idx < len(history['clip_values'])
            ]
            rounds_clip = [
                round_idx for round_idx in rounds if round_idx < len(history['clip_values'])
            ]
        else:
            clip_at_eval = history['clip_values']
            rounds_clip = list(range(len(clip_at_eval)))

        ax.plot(rounds_clip, clip_at_eval, marker='D', linewidth=2,
               markersize=6, color='purple')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Clipping Threshold', fontsize=11)
        ax.set_title('Adaptive Clipping Values', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 평균값 표시
        mean_clip = np.mean(clip_at_eval)
        ax.axhline(y=mean_clip, color='gray', linestyle=':', linewidth=2,
                  label=f'Mean: {mean_clip:.3f}')
        ax.legend(fontsize=10)

    # 전체 제목
    plt.suptitle(
        f'QuAP-FL Training History - {dataset_name.upper()} (seed={seed})',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()

    # 저장
    if save_path is None:
        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(
            output_dir,
            f'{dataset_name}_seed{seed}_training.{config.get("format", "png")}'
        )

    plt.savefig(save_path, dpi=config.get('dpi', 150), bbox_inches='tight')

    if config.get('show_plot', False):
        plt.show()
    else:
        plt.close()

    return save_path


def plot_multi_seed_comparison(
    results: List[Dict],
    seeds: List[int],
    dataset_name: str,
    config: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    다중 시드 실험 결과 비교 시각화

    Args:
        results: 각 시드의 결과 딕셔너리 리스트
        seeds: 시드 리스트
        dataset_name: 데이터셋 이름
        config: 시각화 설정
        save_path: 저장 경로

    Returns:
        저장된 파일 경로
    """
    plt.style.use(config.get('style', 'seaborn-v0_8'))

    fig, axes = plt.subplots(2, 2, figsize=config.get('figsize', (14, 10)))

    # 평가 간격
    eval_interval = 10

    # 1. Accuracy Comparison
    ax = axes[0, 0]
    all_accuracies = []
    max_len = 0

    for i, result in enumerate(results):
        if 'test_accuracy' in result:
            acc = result['test_accuracy']
            all_accuracies.append(acc)
            max_len = max(max_len, len(acc))
            rounds = [j * eval_interval for j in range(len(acc))]
            ax.plot(rounds, acc, alpha=0.3, linewidth=1)

    # Mean ± Std 계산
    if len(all_accuracies) > 0:
        # 패딩 (길이가 다른 경우)
        padded = []
        for acc in all_accuracies:
            padded_acc = list(acc) + [acc[-1]] * (max_len - len(acc))
            padded.append(padded_acc)

        padded = np.array(padded)
        mean_acc = np.mean(padded, axis=0)
        std_acc = np.std(padded, axis=0)
        rounds = [i * eval_interval for i in range(max_len)]

        ax.plot(rounds, mean_acc, linewidth=3, color='blue', label='Mean')
        ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.2, color='blue', label='±1 Std')

    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title(f'Test Accuracy (n={len(results)} seeds)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Loss Comparison
    ax = axes[0, 1]
    all_losses = []

    for result in results:
        if 'test_loss' in result:
            loss = result['test_loss']
            all_losses.append(loss)
            rounds = [j * eval_interval for j in range(len(loss))]
            ax.semilogy(rounds, loss, alpha=0.3, linewidth=1)

    if len(all_losses) > 0:
        padded = []
        for loss in all_losses:
            padded_loss = list(loss) + [loss[-1]] * (max_len - len(loss))
            padded.append(padded_loss)

        padded = np.array(padded)
        mean_loss = np.mean(padded, axis=0)

        ax.semilogy(rounds, mean_loss, linewidth=3, color='orange', label='Mean')

    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Test Loss (log scale)', fontsize=11)
    ax.set_title(f'Test Loss (n={len(results)} seeds)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Privacy Budget Distribution
    ax = axes[1, 0]
    all_budgets = []

    for result in results:
        if 'privacy_budgets' in result and len(result['privacy_budgets']) > 0:
            all_budgets.extend(result['privacy_budgets'])

    if len(all_budgets) > 0:
        ax.hist(all_budgets, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(all_budgets), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(all_budgets):.6f}')
        ax.set_xlabel('Privacy Budget per Client (ε)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Privacy Budget Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # 4. Final Accuracy Distribution
    ax = axes[1, 1]
    final_accs = [result['test_accuracy'][-1] for result in results
                  if 'test_accuracy' in result and len(result['test_accuracy']) > 0]

    if len(final_accs) > 0:
        ax.hist(final_accs, bins=min(20, len(final_accs)), alpha=0.7,
               edgecolor='black', color='green')
        ax.axvline(np.mean(final_accs), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(final_accs):.4f}')

        from config import TARGET_ACCURACY
        if dataset_name in TARGET_ACCURACY:
            target = TARGET_ACCURACY[dataset_name]
            ax.axvline(target, color='blue', linestyle='--', linewidth=2,
                      label=f'Target: {target:.3f}')

        ax.set_xlabel('Final Accuracy', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Final Accuracy Distribution (n={len(final_accs)})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'QuAP-FL Multi-Seed Comparison - {dataset_name.upper()}',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()

    # 저장
    if save_path is None:
        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(
            output_dir,
            f'{dataset_name}_multi_seed_comparison.{config.get("format", "png")}'
        )

    plt.savefig(save_path, dpi=config.get('dpi', 150), bbox_inches='tight')

    if config.get('show_plot', False):
        plt.show()
    else:
        plt.close()

    return save_path


def generate_summary_table(
    final_accuracy: float,
    final_loss: float,
    dataset_name: str,
    seed: int,
    history: Dict,
    config: Dict
) -> str:
    """
    실험 결과 요약 테이블 생성

    Args:
        final_accuracy: 최종 정확도
        final_loss: 최종 손실
        dataset_name: 데이터셋 이름
        seed: 랜덤 시드
        history: 학습 히스토리
        config: 하이퍼파라미터 설정

    Returns:
        테이블 문자열
    """
    from config import TARGET_ACCURACY, HYPERPARAMETERS

    # 테이블 데이터 준비
    headers = ["Metric", "Value", "Status"]
    data = []

    # 최종 정확도
    target_acc = TARGET_ACCURACY.get(dataset_name, None)
    if target_acc:
        acc_diff = (final_accuracy - target_acc) * 100
        if final_accuracy >= target_acc:
            status = f"Achieved (+{acc_diff:.1f}%)" if acc_diff > 0 else "Achieved"
        else:
            status = f"Below ({acc_diff:.1f}%)"
    else:
        status = "N/A"

    data.append([
        "Final Accuracy",
        f"{final_accuracy:.4f} ({final_accuracy*100:.2f}%)",
        status
    ])

    # 최고 정확도
    if 'test_accuracy' in history and len(history['test_accuracy']) > 0:
        best_acc = max(history['test_accuracy'])
        data.append([
            "Best Accuracy",
            f"{best_acc:.4f} ({best_acc*100:.2f}%)",
            f"Round {history['test_accuracy'].index(best_acc) * 10}"
        ])

    # 최종 손실
    data.append([
        "Final Loss",
        f"{final_loss:.4f}",
        "-"
    ])

    # 최저 손실
    if 'test_loss' in history and len(history['test_loss']) > 0:
        best_loss = min(history['test_loss'])
        data.append([
            "Best Loss",
            f"{best_loss:.4f}",
            f"Round {history['test_loss'].index(best_loss) * 10}"
        ])

    # 프라이버시 예산
    epsilon_total = HYPERPARAMETERS.get('epsilon_total', 3.0)
    data.append([
        "Total Privacy Budget",
        f"ε = {epsilon_total}",
        "-"
    ])

    # 평균 클리핑 값
    if 'clip_values' in history and len(history['clip_values']) > 0:
        avg_clip = np.mean(history['clip_values'])
        data.append([
            "Average Clipping",
            f"{avg_clip:.4f}",
            "-"
        ])

    # 참여 통계
    if 'participation_stats' in history and len(history['participation_stats']) > 0:
        final_stats = history['participation_stats'][-1]
        data.append([
            "Mean Participation Rate",
            f"{final_stats['mean_participation_rate']:.3f}",
            f"{final_stats['participating_clients']} clients"
        ])

    # 테이블 생성
    table = tabulate(data, headers=headers, tablefmt="fancy_grid")

    # 헤더 추가
    header_str = "=" * 70 + "\n"
    header_str += f"QuAP-FL Experiment Results - {dataset_name.upper()} (seed={seed})\n"
    header_str += "=" * 70 + "\n"

    footer_str = "=" * 70

    return header_str + table + "\n" + footer_str
