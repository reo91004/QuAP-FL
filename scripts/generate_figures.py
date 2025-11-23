# generate_figures.py

"""
QuAP-FL 논문 Figure 생성 스크립트

이 스크립트는 MNIST 데이터셋에 대해 QuAP-FL 학습을 수행하고,
논문에 포함될 Figure 2 (Accuracy Curve)와 Figure 3 (Privacy Budget by Group)을 생성합니다.
"""

import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import random
import logging
from datetime import datetime

from framework.server import QuAPFLServer
from models.mnist_model import MNISTModel
from data.data_utils import prepare_non_iid_data
from config.hyperparameters import HYPERPARAMETERS
from utils.visualization import plot_paper_figures, generate_summary_table

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    # 설정
    SEED = 42
    DATASET = 'mnist'
    
    # 빠른 데모를 위해 라운드 수 조정 (필요시 200으로 변경)
    # 논문 재현을 위해서는 200이 권장되지만, 테스트용으로는 50도 충분함
    HYPERPARAMETERS['num_rounds'] = 200 
    
    logger.info(f"Starting QuAP-FL Figure Generation (Seed={SEED}, Dataset={DATASET})")
    set_seed(SEED)
    
    # 1. 데이터 준비
    logger.info("Preparing Non-IID Data...")
    client_indices, train_dataset, test_dataset = prepare_non_iid_data(
        dataset_name=DATASET,
        num_clients=HYPERPARAMETERS['num_clients'],
        alpha=0.5, # Non-IID degree
        seed=SEED
    )
    
    # 2. 모델 준비
    model = MNISTModel()
    
    # 3. 서버 초기화
    server = QuAPFLServer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_indices=client_indices,
        dataset_name=DATASET,
        config=HYPERPARAMETERS,
        logger=logger
    )
    
    # 4. 학습 수행
    logger.info("Starting Training...")
    history = server.train()
    
    # 5. 결과 저장 및 시각화
    # 5. 결과 저장 및 시각화
    # scripts/에서 실행되므로 상위 폴더의 results/figures로 저장
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating Figures in {output_dir}...")
    plot_paper_figures(
        history=history,
        num_clients=HYPERPARAMETERS['num_clients'],
        dataset_name=DATASET,
        save_dir=output_dir
    )
    
    # 요약 테이블 생성
    final_acc = history['test_accuracy'][-1]
    final_loss = history['test_loss'][-1]
    
    summary = generate_summary_table(
        final_accuracy=final_acc,
        final_loss=final_loss,
        dataset_name=DATASET,
        seed=SEED,
        history=history,
        config=HYPERPARAMETERS
    )
    
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
        
    logger.info(f"Done! Summary saved to {summary_path}")
    print(summary)

if __name__ == '__main__':
    main()
