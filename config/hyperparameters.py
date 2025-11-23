# config/hyperparameters.py

"""
QuAP-FL 하이퍼파라미터

Layer-wise Differential Privacy 기반 연합학습:
- 마지막 classification layer만 노이즈 추가하여 고차원 노이즈 문제 해결
- 서버 측 집계 후 노이즈 추가 (표준 DP-FedAvg 패러다임)
- 현실적 프라이버시-유틸리티 균형 달성
"""

HYPERPARAMETERS = {
    # 연합학습 설정
    'num_clients': 100,
    'num_rounds': 200,
    'clients_per_round': 30,  # 30% 참여율
    'warmup_rounds': 5,       # 초기 참여율 안정화를 위한 웜업 라운드

    # 로컬 학습 설정
    'local_epochs': 3,  # 충분한 학습이지만 과도한 드리프트 방지
    'local_batch_size': 32,
    'learning_rate': 0.005,  # 원래 설정 (epsilon-lr 균형)
    'lr_decay': 0.995,  # 느린 감소

    # 프라이버시 설정
    'epsilon_total': 6.0,  # 현실적 프라이버시 수준 (ε=3.0은 너무 강함)
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
    'participation_mix': 0.8,  # 균등 분포와 혼합 비율 (0~1)

    # 클리핑 안정성
    'min_clip': 0.1,
    'max_clip': 1.0,  # DP-SGD 표준 범위 (0.1-1.0)

    # Layer-wise DP 설정
    'noise_strategy': 'layer_wise',  # 'layer_wise' | 'full'
    'critical_layers': ['fc1', 'fc2', 'fc3'],      # 전체 Classification Head 보호
}

PRIVACY_CONFIG = {
    'epsilon_base': 0.03,   # 6.0 / 200 rounds
    'alpha': 0.5,           # 고정값 - 변경 금지
    'beta': 2.0,            # 고정값 - 변경 금지
    'delta': 1e-5           # 고정값
}

# 예상 중간 체크포인트
EXPECTED_MILESTONES = {
    'mnist': {
        'round_10': (0.70, 0.75),   # 70-75%
        'round_50': (0.88, 0.92),   # 88-92%
        'round_100': (0.91, 0.94),  # 91-94%
        'round_150': (0.92, 0.95),  # 92-95%
        'round_200': (0.925, 0.940) # 92.5-94.0% (목표: 93.2%)
    },
    'cifar10': {
        'round_10': (0.40, 0.45),   # 40-45%
        'round_50': (0.68, 0.72),   # 68-72%
        'round_100': (0.74, 0.78),  # 74-78%
        'round_150': (0.76, 0.79),  # 76-79%
        'round_200': (0.760, 0.780) # 76.0-78.0% (목표: 76.8%)
    }
}

# 목표 정확도 (현실적 프라이버시-유틸리티 균형)
TARGET_ACCURACY = {
    'mnist': 0.932,   # 93.2% @ ε=6.0
    'cifar10': 0.768  # 76.8% @ ε=6.0
}

# 시각화 설정
VISUALIZATION_CONFIG = {
    'enabled': True,              # 시각화 생성 여부
    'show_plot': False,           # 화면 표시 (False면 저장만)
    'dpi': 150,                   # 해상도
    'format': 'png',              # 저장 형식 (png, pdf, svg)
    'style': 'seaborn-v0_8',      # matplotlib 스타일
    'figsize': (12, 10),          # Figure 크기
    'include_table': True,        # 결과 테이블 출력 여부
}

# 출력 설정
OUTPUT_CONFIG = {
    'save_results': True,         # JSON 결과 저장
    'save_plots': True,           # 시각화 저장
    'output_dir': './results',    # 출력 디렉토리
    'log_dir': './results',       # 로그 디렉토리
}
