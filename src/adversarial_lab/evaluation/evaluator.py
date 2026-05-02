from __future__ import annotations
from pathlib import Path
import pandas as pd, torch
from tqdm import tqdm
from adversarial_lab.attacks.fgsm  import FGSM
from adversarial_lab.attacks.pgd   import PGD
from adversarial_lab.utils.logging import get_logger
from adversarial_lab.utils.metrics import EpochMetrics

logger = get_logger(__name__)

class RobustnessEvaluator:
    def __init__(self, model, criterion, device, eval_cfg, attack_cfg):
        self.model=model; self.criterion=criterion; self.device=device
        self.eval_cfg=eval_cfg; self.attack_cfg=attack_cfg

    @torch.no_grad()
    def _eval_clean(self, loader):
        self.model.eval(); m=EpochMetrics()
        for imgs,lbls in tqdm(loader,desc="  clean",leave=False):
            imgs,lbls=imgs.to(self.device),lbls.to(self.device)
            logits=self.model(imgs); loss=self.criterion(logits,lbls)
            m.update(loss.item(),(logits.argmax(1)==lbls).float().mean().item(),n=lbls.size(0))
        return {"attack":"clean","epsilon":0.0,"loss":m.avg_loss,"accuracy":m.avg_acc}

    def _eval_attack(self, loader, attack_name, epsilon):
        alpha = max(epsilon/max(self.eval_cfg.pgd_steps//2,1), 1/255)
        atk = FGSM(self.model,epsilon) if attack_name=="fgsm" else               PGD(self.model,epsilon,alpha,self.eval_cfg.pgd_steps,self.attack_cfg.random_start)
        self.model.eval(); m=EpochMetrics()
        for imgs,lbls in tqdm(loader,desc=f"  {attack_name} eps={epsilon:.4f}",leave=False):
            imgs,lbls=imgs.to(self.device),lbls.to(self.device)
            adv=atk(imgs,lbls)
            with torch.no_grad():
                logits=self.model(adv); loss=self.criterion(logits,lbls)
                m.update(loss.item(),(logits.argmax(1)==lbls).float().mean().item(),n=lbls.size(0))
        return {"attack":attack_name,"epsilon":float(epsilon),"loss":m.avg_loss,"accuracy":m.avg_acc}

    def evaluate(self, loader, output_csv=None):
        rows = [self._eval_clean(loader)]
        for eps in self.eval_cfg.epsilons:
            if eps==0.0: continue
            for atk in ("fgsm","pgd"):
                logger.info(f"Eval {atk.upper()} eps={eps:.4f}")
                rows.append(self._eval_attack(loader, atk, eps))
        df = pd.DataFrame(rows)
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True,exist_ok=True)
            df.to_csv(output_csv,index=False)
        return df
