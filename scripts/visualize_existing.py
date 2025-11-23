# visualize_existing.py

"""
기존 실험 결과(JSON)를 로드하여 논문용 Figure를 생성하는 스크립트

Figure 3 (Privacy Budget)의 경우, 기존 로그에 클라이언트 참여 이력이 없으므로
동일한 시드와 설정을 사용하여 참여 이력을 시뮬레이션(재구성)하여 시각화합니다.
"""

import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List

from config.hyperparameters import HYPERPARAMETERS
from framework.server import QuAPFLServer
from utils.visualization import plot_paper_figures, plot_privacy_consumption_by_group

def load_results(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def simulate_client_participation(
    num_clients: int,
    num_rounds: int,
    seed: int,
    config: Dict
) -> List[List[int]]:
    """
    클라이언트 참여 이력 시뮬레이션
    
    주의: 실제 실험과 완벽하게 동일하지 않을 수 있음 (np.random 상태 차이 등).
    하지만 Beta 분포의 특성은 유지되므로 경향성은 파악 가능.
    """
    # 시드 설정
    np.random.seed(seed)
    
    # 서버 인스턴스를 생성하여 select_clients 로직 재사용
    # 모델이나 데이터셋은 필요 없음 (Mocking)
    class MockServer(QuAPFLServer):
        def __init__(self, config):
            self.config = config
            self.num_clients = config['num_clients']
            self.participation_distribution = config.get('participation_distribution', 'uniform')
            self.participation_mix_ratio = float(np.clip(config.get('participation_mix', 0.2), 0.0, 1.0))
            
            # 초기 가중치 설정 (server.py 로직 복사)
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
                if np.allclose(base_weights.sum(), 0.0):
                    base_weights = np.ones(self.num_clients)
            else:
                base_weights = np.ones(self.num_clients)
            
            self.participation_base_weights = base_weights / base_weights.sum()
            
    # Mock 서버 초기화
    server = MockServer(config)
    
    participation_history = []
    for t in range(num_rounds):
        # select_clients 호출
        selected = server.select_clients(t)
        participation_history.append(selected)
        
        # 실제 실험에서는 여기서 local_training과 noise addition이 일어나며
        # np.random 상태가 변함. 여기서는 이를 완벽히 모사할 수 없으므로
        # 단순히 다음 라운드 선택을 진행.
        # 따라서 실제 실험과는 참여 클라이언트가 다를 수 있음.
        
    return participation_history

def main():
    parser = argparse.ArgumentParser(description='Visualize Existing Results')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing the result JSON file')
    args = parser.parse_args()
    
    # 디렉토리 내의 QuAP-FL JSON 파일 찾기
    json_file = None
    for f in os.listdir(args.result_dir):
        if f.endswith('.json') and 'quap_fl' in f:
            json_file = os.path.join(args.result_dir, f)
            break
            
    if not json_file:
        print(f"Error: No QuAP-FL JSON file found in {args.result_dir}")
        return

    print(f"Loading results from {json_file}...")
    data = load_results(json_file)
    history = data['history']
    metadata = data['metadata']
    config = metadata['hyperparameters']
    
    # Figure 저장 경로
    save_dir = args.result_dir
    
    # 1. Figure 2: Accuracy Curve (데이터 있음)
    print("Generating Figure 2 (Accuracy Curve)...")
    # plot_paper_figures 내부에서 Accuracy Curve 생성
    # 하지만 plot_paper_figures는 client_participation이 없으면 Figure 3를 스킵함.
    # 따라서 여기서 수동으로 호출하거나 history를 보강해야 함.
    
    # 2. Figure 3: Privacy Budget (데이터 누락 -> 시뮬레이션)
    print("Simulating client participation for Figure 3...")
    num_rounds = len(history['privacy_budgets'])
    simulated_participation = simulate_client_participation(
        num_clients=config['num_clients'],
        num_rounds=num_rounds,
        seed=metadata['seed'],
        config=config
    )
    
    # history에 시뮬레이션 데이터 주입
    history['client_participation'] = simulated_participation
    
    # 시각화 생성
    print("Generating figures...")
    plot_paper_figures(
        history=history,
        num_clients=config['num_clients'],
        dataset_name=metadata['dataset'],
        save_dir=save_dir
    )
    
    print(f"Done! Figures saved in {save_dir}")

if __name__ == '__main__':
    main()
