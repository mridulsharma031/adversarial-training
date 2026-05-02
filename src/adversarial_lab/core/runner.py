from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from adversarial_lab.core.config import ExperimentConfig
from adversarial_lab.core.trainer import Trainer
from adversarial_lab.data import get_cifar10_loaders
from adversarial_lab.evaluation.evaluator import RobustnessEvaluator
from adversarial_lab.models import build_model
from adversarial_lab.callbacks import EarlyStopping, ModelCheckpoint
from adversarial_lab.visualization import (
    save_adversarial_grid,
    plot_robustness_curves,
    plot_training_history,
)
from adversarial_lab.utils.logging import get_logger
from adversarial_lab.utils.seed import set_seed
from adversarial_lab.utils.io import ensure_dir, load_checkpoint

logger = get_logger(__name__)


def _build_scheduler(optimizer, cfg, epochs: int):
    if cfg.training.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if cfg.training.scheduler == "multistep":
        return MultiStepLR(
            optimizer, milestones=[epochs // 2, 3 * epochs // 4], gamma=0.1
        )
    raise ValueError(f"Unknown scheduler: {cfg.training.scheduler!r}")


def run_train(cfg: ExperimentConfig, checkpoint: str | None = None) -> list[dict]:
    """
    Main training entry point.
    If ``checkpoint`` points to a valid .pt file, training resumes from the
    epoch stored inside that file (optimizer & scheduler states are also restored).
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ensure_dir(cfg.output_dir)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    logger.info(f"Device: {device} | Defense: {cfg.training.defense} | Output: {out}")

    # ----- data -----
    train_loader, val_loader, _ = get_cifar10_loaders(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        use_fake_data=cfg.data.use_fake_data,
        seed=cfg.seed,
    )

    # ----- model, loss, optim, scheduler -----
    model = build_model(
        cfg.model.arch, cfg.model.num_classes, cfg.model.pretrained
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg, cfg.training.epochs)

    # ----- checkpoint resumption logic -----
    start_epoch = 1
    if checkpoint:
        ckpt = load_checkpoint(checkpoint, map_location=device)
        # Load model
        model.load_state_dict(ckpt["model_state_dict"])
        # Load optimizer
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Load scheduler if present
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        # Determine epoch to resume from (next epoch after the saved one)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info(
            f"Resuming from checkpoint {checkpoint} – starting at epoch {start_epoch}"
        )

    # ----- callbacks -----
    callbacks = [
        ModelCheckpoint(output_dir=str(out), monitor="val_acc"),
        EarlyStopping(monitor="val_acc", patience=15),
    ]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        train_cfg=cfg.training,
        attack_cfg=cfg.attack,
        callbacks=callbacks,
        output_dir=str(out),
    )
    history = trainer.fit(train_loader, val_loader, start_epoch=start_epoch)

    plot_training_history(history, str(out / "training_history.png"))
    logger.info("Training complete.")
    return history


def run_evaluate(cfg: ExperimentConfig, checkpoint: str) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ensure_dir(cfg.output_dir)

    _, _, test_loader = get_cifar10_loaders(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        use_fake_data=cfg.data.use_fake_data,
        seed=cfg.seed,
    )
    model = build_model(
        cfg.model.arch, cfg.model.num_classes, cfg.model.pretrained
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    state = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Loaded checkpoint: {checkpoint}")

    evaluator = RobustnessEvaluator(model, criterion, device, cfg.evaluation, cfg.attack)
    df = evaluator.evaluate(test_loader, output_csv=str(out / "robustness_results.csv"))
    print(df.to_string(index=False))
    plot_robustness_curves(df, str(out / "robustness_curves.png"))
    logger.info("Evaluation complete.")


def run_visualize(cfg: ExperimentConfig, checkpoint: str) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ensure_dir(cfg.output_dir)

    _, _, test_loader = get_cifar10_loaders(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        use_fake_data=cfg.data.use_fake_data,
        seed=cfg.seed,
    )
    model = build_model(
        cfg.model.arch, cfg.model.num_classes, cfg.model.pretrained
    ).to(device)
    state = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    for atk in ("fgsm", "pgd"):
        save_adversarial_grid(
            model=model,
            loader=test_loader,
            device=device,
            attack_name=atk,
            epsilon=cfg.attack.epsilon,
            alpha=cfg.attack.alpha,
            steps=cfg.attack.steps,
            num_images=cfg.evaluation.num_vis_images,
            output_path=str(out / f"{atk}_examples.png"),
        )
    logger.info("Visualization complete.")
