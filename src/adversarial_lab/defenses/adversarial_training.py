from __future__ import annotations
import torch, torch.nn as nn
from adversarial_lab.attacks.factory import build_attack

class AdversarialTrainingDefense:
    """Madry-style adversarial training (PGD or FGSM)."""
    def __init__(self, model, attack_name, epsilon, alpha, steps, random_start=True):
        self.model=model; self.attack_name=attack_name; self.epsilon=epsilon
        self.alpha=alpha; self.steps=steps; self.random_start=random_start; self._attack=None
    def _get_attack(self):
        if self._attack is None:
            self._attack = build_attack(self.attack_name,self.model,self.epsilon,self.alpha,
                                        self.steps,self.random_start)
        return self._attack
    def make_batch(self, images, labels):
        return self._get_attack()(images, labels), labels

class CleanTrainingDefense:
    """No perturbation — standard ERM training."""
    def make_batch(self, images, labels): return images, labels

def build_defense(defense, model, attack_name, epsilon, alpha, steps, random_start=True):
    if defense == "clean":      return CleanTrainingDefense()
    if defense == "adversarial": return AdversarialTrainingDefense(model,attack_name,epsilon,alpha,steps,random_start)
    raise ValueError(f"Unknown defense: {defense!r}")
