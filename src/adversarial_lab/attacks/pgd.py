from __future__ import annotations
import torch, torch.nn.functional as F
from adversarial_lab.attacks.base import BaseAttack

class PGD(BaseAttack):
    """Projected Gradient Descent (Madry et al., 2018)."""
    def __init__(self, model, epsilon, alpha, steps, random_start=True):
        super().__init__(model, epsilon)
        self.alpha=alpha; self.steps=steps; self.random_start=random_start
    @torch.enable_grad()
    def perturb(self, images, labels):
        x = images.detach()
        if self.random_start:
            adv = (x + torch.empty_like(x).uniform_(-self.epsilon,self.epsilon)).clamp(0,1)
        else:
            adv = x.clone()
        for _ in range(self.steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(adv), labels)
            grad = torch.autograd.grad(loss, adv)[0]
            adv  = self._clip(adv.detach() + self.alpha*grad.sign(), x, self.epsilon)
        return adv
