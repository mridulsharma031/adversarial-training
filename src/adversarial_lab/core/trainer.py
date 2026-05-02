from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from adversarial_lab.callbacks.base import Callback
from adversarial_lab.defenses.adversarial_training import build_defense
from adversarial_lab.core.config import TrainingConfig, AttackConfig
from adversarial_lab.utils.logging import get_logger
from adversarial_lab.utils.metrics import EpochMetrics

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,  # can be None
        criterion: nn.Module,
        device: torch.device,
        train_cfg: TrainingConfig,
        attack_cfg: AttackConfig,
        callbacks: Optional[List[Callback]] = None,
        output_dir: str = "outputs/default",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_cfg = train_cfg
        self.attack_cfg = attack_cfg
        self.callbacks = callbacks or []
        self.output_dir = output_dir
        self.history: list[dict] = []
        self.defense = build_defense(
            train_cfg.defense,
            model,
            train_cfg.train_attack,
            attack_cfg.epsilon,
            attack_cfg.alpha,
            attack_cfg.steps,
            attack_cfg.random_start,
        )

    # -------------------------------------------------------------------------
    # Internal helper to fire callbacks
    # -------------------------------------------------------------------------
    def _run_callbacks(self, method: str, epoch: int, logs: dict) -> None:
        for cb in self.callbacks:
            getattr(cb, method)(epoch, logs)

    # -------------------------------------------------------------------------
    # One epoch of training
    # -------------------------------------------------------------------------
    def train_epoch(self, loader) -> dict:
        self.model.train()
        metrics = EpochMetrics()
        for images, labels in tqdm(loader, desc="  train", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            # -------- defense (clean or adversarial) ----------
            adv_imgs, adv_lbls = self.defense.make_batch(images, labels)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(adv_imgs)
            loss = self.criterion(logits, adv_lbls)
            loss.backward()
            self.optimizer.step()

            acc = (logits.argmax(1) == adv_lbls).float().mean().item()
            metrics.update(loss.item(), acc, n=labels.size(0))

        return {"train_loss": metrics.avg_loss, "train_acc": metrics.avg_acc}

    # -------------------------------------------------------------------------
    # One epoch of evaluation (no gradient)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def eval_epoch(self, loader, split: str = "val") -> dict:
        self.model.eval()
        metrics = EpochMetrics()
        for images, labels in tqdm(loader, desc=f"  {split}", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            acc = (logits.argmax(1) == labels).float().mean().item()
            metrics.update(loss.item(), acc, n=labels.size(0))
        return {f"{split}_loss": metrics.avg_loss, f"{split}_acc": metrics.avg_acc}

    # -------------------------------------------------------------------------
    # Main fit loop – supports resumption via ``start_epoch``
    # -------------------------------------------------------------------------
    def fit(
        self,
        train_loader,
        val_loader,
        start_epoch: int = 1,
    ) -> list[dict]:
        """
        Parameters
        ----------
        train_loader, val_loader : DataLoader
        start_epoch : int, default 1
            Epoch number to begin counting from (useful when resuming from a checkpoint).
        """
        # ---- fire on_train_start callbacks (epoch‑agnostic) ----
        for cb in self.callbacks:
            cb.on_train_start({})

        for epoch in range(start_epoch, self.train_cfg.epochs + 1):
            # ---- epoch‑start callbacks ----
            for cb in self.callbacks:
                cb.on_epoch_start(epoch, {})

            logs = {}
            logs.update(self.train_epoch(train_loader))
            logs.update(self.eval_epoch(val_loader, "val"))

            if self.scheduler:
                self.scheduler.step()

            logs["epoch"] = epoch
            logs["lr"] = self.optimizer.param_groups[0]["lr"]

            # Package everything the checkpoint callback needs
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            if self.scheduler is not None:
                state["scheduler_state_dict"] = self.scheduler.state_dict()
            logs["_state"] = state

            self.history.append({k: v for k, v in logs.items() if k != "_state"})

            logger.info(
                f"Ep {epoch:>3}/{self.train_cfg.epochs} | "
                f"train loss={logs['train_loss']:.4f} acc={logs['train_acc']:.4f} | "
                f"val   loss={logs['val_loss']:.4f} acc={logs['val_acc']:.4f}"
            )

            # ---- epoch‑end callbacks ----
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)

            # Early stopping?
            if any(getattr(cb, "stop", False) for cb in self.callbacks):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # ---- finalise ----
        for cb in self.callbacks:
            cb.on_train_end({})
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history).to_csv(
            Path(self.output_dir) / "training_history.csv", index=False
        )
        return self.history
