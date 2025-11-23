# scripts/generate_non_iid_figure.py

"""
Non-IID 영향도 시각화 생성 스크립트

Dirichlet 파라미터 α를 변화시키며 QuAP-FL, Fixed-DP, FedAvg의 성능을 비교하는 그래프를 생성한다.
짧은 라운드(50-100)로 실험하여 경향성을 파악한다.
"""

import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List

from config.hyperparameters import HYPERPARAMETERS
from data.data_utils import prepare_non_iid_data, create_client_dataloaders
from framework.server import FedAvgServer, FixedDPServer, QuAPFLServer
from models.simple_cnn import SimpleCNN


def run_quick_experiment(
    alpha_value: float,
    method: str,
    num_rounds: int = 50,
    seed: int = 42
) -> float:
    """
    특정 α 값으로 짧은 실험을 실행하여 최종 정확도를 반환한다.
    
    Args:
        alpha_value: Dirichlet 파라미터 α
        method: 'fedavg', 'fixed_dp', 'quap_fl'
        num_rounds: 실행할 라운드 수
        seed: 랜덤 시드
        
    Returns:
        최종 테스트 정확도
    """
    print(f"  Running {method} with α={alpha_value}...")
    
    # 시드 설정
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 설정 복사 및 수정
    config = HYPERPARAMETERS.copy()
    config['dirichlet_alpha'] = alpha_value
    config['num_rounds'] = num_rounds
    config['eval_interval'] = 10
    
    # 데이터 로드 및 분할
    client_indices, train_dataset, test_dataset = prepare_non_iid_data(
        dataset_name='mnist',
        num_clients=config['num_clients'],
        alpha=alpha_value,
        seed=seed
    )
    
    # 각 클라이언트용 DataLoader 생성
    train_loaders = []
    for cid in range(config['num_clients']):
        loader = create_client_dataloaders(
            dataset=train_dataset,
            client_indices=client_indices,
            batch_size=config['batch_size'],
            client_id=cid
        )
        train_loaders.append(loader)
    
    # 테스트 DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.get('test_batch_size', 128),
        shuffle=False
    )
    
    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=10).to(device)
    
    # 서버 선택
    if method == 'fedavg':
        server = FedAvgServer(
            model=model,
            train_loaders=train_loaders,
            test_loader=test_loader,
            config=config,
            device=device
        )
    elif method == 'fixed_dp':
        server = FixedDPServer(
            model=model,
            train_loaders=train_loaders,
            test_loader=test_loader,
            config=config,
            device=device
        )
    elif method == 'quap_fl':
        server = QuAPFLServer(
            model=model,
            train_loaders=train_loaders,
            test_loader=test_loader,
            config=config,
            device=device
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 학습 실행 (verbose=False for cleaner output)
    history = server.train(verbose=False)
    
    # 최종 정확도 반환
    final_acc = history['test_accuracy'][-1] if history['test_accuracy'] else 0.0
    print(f"    → Final accuracy: {final_acc:.4f}")
    
    return final_acc


def generate_non_iid_impact_figure(
    alpha_values: List[float],
    methods: List[str],
    num_rounds: int = 50,
    seed: int = 42,
    save_path: str = None
):
    """
    Non-IID 영향도 그래프 생성
    
    Args:
        alpha_values: 테스트할 Dirichlet α 값 리스트
        methods: 비교할 방법 리스트
        num_rounds: 각 실험의 라운드 수
        seed: 랜덤 시드
        save_path: 저장 경로
    """
    print("Generating Non-IID Impact Figure...")
    print(f"Alpha values: {alpha_values}")
    print(f"Methods: {methods}")
    print(f"Rounds per experiment: {num_rounds}")
    print()
    
    # 결과 저장
    results = {method: [] for method in methods}
    
    # 각 α 값에 대해 실험 실행
    for alpha in alpha_values:
        print(f"Testing α = {alpha}:")
        for method in methods:
            acc = run_quick_experiment(alpha, method, num_rounds, seed)
            results[method].append(acc)
        print()
    
    # 시각화
    plt.figure(figsize=(10, 6))
    
    # 라벨 매핑
    method_labels = {
        'fedavg': 'FedAvg (No Privacy)',
        'fixed_dp': 'Fixed-DP',
        'quap_fl': 'QuAP-FL (Ours)'
    }
    
    # 색상 및 마커 설정
    method_styles = {
        'fedavg': {'color': 'black', 'marker': 'o', 'linestyle': '--'},
        'fixed_dp': {'color': 'orange', 'marker': 's', 'linestyle': '-'},
        'quap_fl': {'color': 'blue', 'marker': '^', 'linestyle': '-'}
    }
    
    # 플롯
    for method in methods:
        style = method_styles.get(method, {})
        plt.plot(
            alpha_values,
            [acc * 100 for acc in results[method]],  # 백분율로 변환
            label=method_labels.get(method, method),
            linewidth=2.5,
            markersize=8,
            **style
        )
    
    # 축 및 제목 설정
    plt.xlabel('Dirichlet Parameter α (Data Heterogeneity)', fontsize=13)
    plt.ylabel('Test Accuracy (%)', fontsize=13)
    plt.title('Impact of Data Heterogeneity (Non-IID) on Performance', 
              fontsize=15, fontweight='bold', pad=15)
    
    # x축을 로그 스케일로 (α가 넓은 범위를 가지므로)
    plt.xscale('log')
    
    # 그리드 및 레전드
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
    
    # y축 범위 조정 (필요시)
    plt.ylim([min(min(results[m]) for m in methods) * 98, 
              max(max(results[m]) for m in methods) * 101])
    
    # 주석 추가
    plt.text(
        0.02, 0.98,
        'Lower α → More heterogeneous\nHigher α → More uniform',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    
    # 저장
    if save_path is None:
        save_path = './results/20251122_183833/non_iid_impact.png'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.close()
    
    # 결과 출력
    print("\nResults Summary:")
    print("-" * 60)
    for method in methods:
        print(f"{method_labels.get(method, method)}:")
        for i, alpha in enumerate(alpha_values):
            print(f"  α={alpha:6.2f}: {results[method][i]*100:6.2f}%")
    print("-" * 60)


def main():
    """메인 함수"""
    # α 값 설정 (낮을수록 Non-IID가 심함)
    alpha_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # 비교할 방법
    methods = ['fedavg', 'fixed_dp', 'quap_fl']
    
    # 실험 실행 및 그래프 생성
    generate_non_iid_impact_figure(
        alpha_values=alpha_values,
        methods=methods,
        num_rounds=50,  # 짧은 실험으로 경향성 파악
        seed=42,
        save_path='./results/20251122_183833/non_iid_impact.png'
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
