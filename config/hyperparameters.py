# config/hyperparameters.py

"""
QuAP-FL 논문 재현을 위한 모든 하이퍼파라미터 고정값
이 값들을 변경하면 논문 결과를 재현할 수 없다.
"""

HYPERPARAMETERS = {
    # 연합학습 설정
    'num_clients': 100,
    'num_rounds': 200,
    'clients_per_round': 30,  # 30% 참여율

    # 로컬 학습 설정
    'local_epochs': 5,
    'local_batch_size': 32,
    'learning_rate': 0.01,
    'lr_decay': 0.99,  # 매 라운드 0.99 곱하기

    # 프라이버시 설정
    'epsilon_total': 3.0,
    'delta': 1e-5,

    # 적응형 파라미터 (절대 변경 금지)
    'alpha': 0.5,  # 적응 강도
    'beta': 2.0,   # 감소율
    'clip_quantile': 0.9,  # 90th percentile
    'clip_momentum': 0.95,  # EMA momentum

    # 참여 시뮬레이션 (이질적 참여)
    'participation_distribution': 'beta',  # Beta(2, 5)
    'participation_alpha': 2,
    'participation_beta': 5,

    # 클리핑 안정성
    'min_clip': 0.1,
    'max_clip': 10.0,
}

PRIVACY_CONFIG = {
    'epsilon_base': 0.015,  # 3.0 / 200 rounds
    'alpha': 0.5,           # 고정값 - 변경 금지
    'beta': 2.0,            # 고정값 - 변경 금지
    'delta': 1e-5           # 고정값
}

# 논문 재현을 위한 예상 중간 결과
EXPECTED_MILESTONES = {
    'mnist': {
        'round_10': (0.65, 0.70),   # 65-70% 정확도
        'round_50': (0.88, 0.92),   # 88-92% 정확도
        'round_100': (0.94, 0.96),  # 94-96% 정확도
        'round_150': (0.96, 0.98),  # 96-98% 정확도
        'round_200': (0.965, 0.976) # 96.5-97.6% 정확도 (목표: 97.1%)
    },
    'cifar10': {
        'round_10': (0.35, 0.40),   # 35-40% 정확도
        'round_50': (0.65, 0.70),   # 65-70% 정확도
        'round_100': (0.74, 0.78),  # 74-78% 정확도
        'round_150': (0.78, 0.82),  # 78-82% 정확도
        'round_200': (0.803, 0.821) # 80.3-82.1% 정확도 (목표: 81.2%)
    }
}

# 목표 정확도
TARGET_ACCURACY = {
    'mnist': 0.971,   # 97.1%
    'cifar10': 0.812  # 81.2%
}