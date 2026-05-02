from __future__ import annotations
import torch, torch.nn.functional as F
from adversarial_lab.attacks.base import BaseAttack

class FGSM(BaseAttack):
    """Fast Gradient Sign Method (Goodfellow et al., 2014)."""
    @torch.enable_grad()
    def perturb(self, images, labels):
        x = images.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(self.model(x), labels)
        grad = torch.autograd.grad(loss, x)[0]
        return self._clip(x + self.epsilon * grad.sign(), images, self.epsilon)
