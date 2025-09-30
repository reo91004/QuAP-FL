# models/cifar_model.py

"""
CIFAR-10용 CNN 모델

구조:
- Conv2d(3, 64, 3, padding=1)
- MaxPool2d(2, 2)
- Conv2d(64, 128, 3, padding=1)
- MaxPool2d(2, 2)
- Conv2d(128, 256, 3, padding=1)
- MaxPool2d(2, 2)
- FC(256*4*4, 256)
- Dropout(0.5)
- FC(256, 128)
- FC(128, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10Model(nn.Module):
    """
    CIFAR-10 분류를 위한 CNN 모델
    논문에서 사용한 정확한 아키텍처
    """

    def __init__(self):
        super(CIFAR10Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, 3, 32, 32)

        Returns:
            Output logits (batch_size, 10)
        """
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8

        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4

        # Flatten
        x = x.view(-1, 256 * 4 * 4)

        # Fully connected layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Fully connected layer 2
        x = F.relu(self.fc2(x))

        # Fully connected layer 3 (output)
        x = self.fc3(x)

        return x

    def get_num_parameters(self):
        """
        모델 파라미터 수 계산

        Returns:
            총 파라미터 수
        """
        return sum(p.numel() for p in self.parameters())

    def get_gradients_as_vector(self):
        """
        모든 그래디언트를 1D 벡터로 변환

        Returns:
            1D numpy array
        """
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1).cpu().numpy())

        if len(grads) == 0:
            return None

        import numpy as np
        return np.concatenate(grads)

    def set_gradients_from_vector(self, grad_vector):
        """
        1D 벡터에서 그래디언트 복원

        Args:
            grad_vector: 1D numpy array
        """
        import numpy as np
        import torch

        idx = 0
        for param in self.parameters():
            if param.grad is not None:
                numel = param.numel()
                shape = param.shape

                param.grad = torch.from_numpy(
                    grad_vector[idx:idx+numel].reshape(shape)
                ).to(param.device).float()

                idx += numel