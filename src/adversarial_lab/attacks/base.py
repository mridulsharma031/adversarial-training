from __future__ import annotations
from abc import ABC, abstractmethod
import torch, torch.nn as nn

class BaseAttack(ABC):
    def __init__(self, model: nn.Module, epsilon: float):
        self.model = model; self.epsilon = epsilon
    @abstractmethod
    def perturb(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, images, labels):
        was_training = self.model.training
        self.model.eval()
        adv = self.perturb(images, labels)
        self.model.train(was_training)
        return adv.detach()
    @staticmethod
    def _clip(adv, original, epsilon):
        delta = torch.clamp(adv - original, -epsilon, epsilon)
        return (original + delta).clamp(0.0, 1.0)
