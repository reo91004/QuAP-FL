# models/__init__.py
from .mnist_model import MNISTModel
from .cifar_model import CIFAR10Model

__all__ = ['MNISTModel', 'CIFAR10Model']