# models/mnist_model.py

"""
MNIST용 CNN 모델

구조:
- Conv2d(1, 32, 3, 1)
- Conv2d(32, 64, 3, 1)
- Dropout2d(0.25)
- FC(9216, 128)
- Dropout(0.5)
- FC(128, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    """
    MNIST 분류를 위한 CNN 모델
    논문에서 사용한 정확한 아키텍처
    """

    def __init__(self):
        super(MNISTModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # Dropout layers
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, 1, 28, 28)

        Returns:
            Log probabilities (batch_size, 10)
        """
        # Conv layer 1
        x = self.conv1(x)
        x = F.relu(x)

        # Conv layer 2
        x = self.conv2(x)
        x = F.relu(x)

        # Max pooling
        x = F.max_pool2d(x, 2)

        # Dropout
        x = self.dropout1(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)

        # Dropout
        x = self.dropout2(x)

        # Fully connected layer 2 (output)
        x = self.fc2(x)

        # Log softmax
        output = F.log_softmax(x, dim=1)

        return output

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