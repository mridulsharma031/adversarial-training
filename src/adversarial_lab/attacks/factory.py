from __future__ import annotations
import torch.nn as nn
from adversarial_lab.attacks.fgsm import FGSM
from adversarial_lab.attacks.pgd  import PGD

def build_attack(name: str, model: nn.Module, epsilon: float, alpha: float,
                 steps: int, random_start: bool = True):
    name = name.lower()
    if name == "fgsm": return FGSM(model=model, epsilon=epsilon)
    if name == "pgd":  return PGD(model=model,  epsilon=epsilon, alpha=alpha,
                                   steps=steps, random_start=random_start)
    raise ValueError(f"Unknown attack: {name!r}")
