from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import yaml

@dataclass
class DataConfig:
    root: str = "./data"; image_size: int = 32; batch_size: int = 128
    num_workers: int = 2; use_fake_data: bool = False

@dataclass
class ModelConfig:
    arch: str = "resnet18"; num_classes: int = 10; pretrained: bool = False

@dataclass
class TrainingConfig:
    epochs: int = 30; lr: float = 0.1; momentum: float = 0.9
    weight_decay: float = 5e-4; scheduler: str = "cosine"
    defense: str = "clean"; train_attack: str = "pgd"

@dataclass
class AttackConfig:
    epsilon: float = 8/255; alpha: float = 2/255; steps: int = 7; random_start: bool = True

@dataclass
class EvaluationConfig:
    pgd_steps: int = 20
    epsilons: List[float] = field(default_factory=lambda: [0.0,2/255,4/255,8/255])
    num_vis_images: int = 8

@dataclass
class ExperimentConfig:
    name: str = "exp"; seed: int = 42; output_dir: str = "outputs/default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self): return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        exp = raw.get("experiment", {})
        cfg = cls(name=exp.get("name","exp"), seed=exp.get("seed",42), output_dir=exp.get("output_dir","outputs/default"))
        if "data"       in raw: cfg.data       = DataConfig(**raw["data"])
        if "model"      in raw: cfg.model      = ModelConfig(**raw["model"])
        if "training"   in raw: cfg.training   = TrainingConfig(**raw["training"])
        if "attack"     in raw: cfg.attack     = AttackConfig(**raw["attack"])
        if "evaluation" in raw: cfg.evaluation = EvaluationConfig(**raw["evaluation"])
        return cfg

    @classmethod
    def from_cli(cls, args) -> "ExperimentConfig":
        base = cls.from_yaml(args.config) if getattr(args,"config",None) else cls()
        if getattr(args,"output_dir",None):   base.output_dir = args.output_dir
        if getattr(args,"epochs",None):       base.training.epochs = args.epochs
        if getattr(args,"batch_size",None):   base.data.batch_size = args.batch_size
        if getattr(args,"lr",None):           base.training.lr = args.lr
        if getattr(args,"defense",None):      base.training.defense = args.defense
        if getattr(args,"epsilon",None):      base.attack.epsilon = args.epsilon
        if getattr(args,"alpha",None):        base.attack.alpha = args.alpha
        if getattr(args,"attack_steps",None): base.attack.steps = args.attack_steps
        if getattr(args,"pgd_steps",None):    base.evaluation.pgd_steps = args.pgd_steps
        if getattr(args,"seed",None):         base.seed = args.seed
        if getattr(args,"use_fake_data",False): base.data.use_fake_data = True
        return base
