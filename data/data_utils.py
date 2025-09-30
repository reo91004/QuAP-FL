# data/data_utils.py

"""
Non-IID 데이터 분할 유틸리티

Dirichlet 분포 (α=0.5) 기반 Non-IID 분할
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional


def prepare_non_iid_data(
    dataset_name: str,
    num_clients: int,
    alpha: float = 0.5,
    data_dir: str = './data',
    seed: Optional[int] = None
) -> Tuple[List[List[int]], datasets.VisionDataset, datasets.VisionDataset]:
    """
    Dirichlet 분포 기반 Non-IID 데이터 분할

    Args:
        dataset_name: 'mnist' or 'cifar10'
        num_clients: 전체 클라이언트 수
        alpha: Dirichlet 분포 파라미터 (0.5 고정값)
        data_dir: 데이터 저장 경로
        seed: 랜덤 시드

    Returns:
        client_indices: 각 클라이언트의 데이터 인덱스
        train_dataset: 학습 데이터셋
        test_dataset: 테스트 데이터셋
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 데이터셋 로드
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )

    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform_test
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist' or 'cifar10'")

    # 레이블 추출
    if hasattr(train_dataset, 'targets'):
        labels = np.array(train_dataset.targets)
    else:
        labels = np.array([y for _, y in train_dataset])

    num_classes = len(np.unique(labels))

    # Dirichlet 분할
    client_indices = dirichlet_split(
        labels, num_clients, num_classes, alpha
    )

    return client_indices, train_dataset, test_dataset


def dirichlet_split(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float
) -> List[List[int]]:
    """
    Dirichlet 분포 기반 데이터 분할

    Args:
        labels: 전체 데이터의 레이블 (shape: num_samples)
        num_clients: 클라이언트 수
        num_classes: 클래스 수
        alpha: Dirichlet 분포 파라미터

    Returns:
        각 클라이언트의 데이터 인덱스 리스트
    """
    # 각 클라이언트의 클래스 비율 생성
    # label_distribution[i, k] = 클라이언트 i의 클래스 k 비율
    label_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)

    # 클라이언트별 데이터 인덱스 초기화
    client_indices = [[] for _ in range(num_clients)]

    # 각 클래스별로 데이터를 클라이언트에게 분배
    for k in range(num_classes):
        # 클래스 k의 모든 인덱스
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # 각 클라이언트에게 할당할 비율
        proportions = label_distribution[:, k]
        proportions = proportions / proportions.sum()  # 정규화

        # 누적 비율을 사용하여 분할 지점 계산
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        # 인덱스 분할
        split_idx = np.split(idx_k, proportions)

        # 각 클라이언트에게 할당
        for i, idx in enumerate(split_idx):
            if i < num_clients:
                client_indices[i].extend(idx.tolist())

    # 각 클라이언트의 데이터 섞기
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def create_client_dataloaders(
    dataset: datasets.VisionDataset,
    client_indices: List[List[int]],
    batch_size: int,
    client_id: int
) -> DataLoader:
    """
    특정 클라이언트의 DataLoader 생성

    Args:
        dataset: 전체 데이터셋
        client_indices: 각 클라이언트의 데이터 인덱스
        batch_size: 배치 크기
        client_id: 클라이언트 ID

    Returns:
        DataLoader
    """
    if client_id < 0 or client_id >= len(client_indices):
        raise ValueError(f"Invalid client_id: {client_id}")

    indices = client_indices[client_id]

    if len(indices) == 0:
        raise ValueError(f"Client {client_id} has no data")

    # Subset 생성
    client_dataset = Subset(dataset, indices)

    # DataLoader 생성
    dataloader = DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    return dataloader


def analyze_data_distribution(
    client_indices: List[List[int]],
    labels: np.ndarray,
    num_classes: int
) -> dict:
    """
    데이터 분포 분석

    Args:
        client_indices: 각 클라이언트의 데이터 인덱스
        labels: 전체 데이터의 레이블
        num_classes: 클래스 수

    Returns:
        분석 결과 딕셔너리
    """
    num_clients = len(client_indices)

    # 각 클라이언트의 데이터 수
    client_sizes = [len(indices) for indices in client_indices]

    # 각 클라이언트의 클래스 분포
    client_class_counts = np.zeros((num_clients, num_classes))
    for i, indices in enumerate(client_indices):
        client_labels = labels[indices]
        for k in range(num_classes):
            client_class_counts[i, k] = np.sum(client_labels == k)

    # 통계 계산
    return {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'total_samples': len(labels),
        'mean_samples_per_client': float(np.mean(client_sizes)),
        'std_samples_per_client': float(np.std(client_sizes)),
        'min_samples_per_client': int(np.min(client_sizes)),
        'max_samples_per_client': int(np.max(client_sizes)),
        'class_distribution': client_class_counts
    }