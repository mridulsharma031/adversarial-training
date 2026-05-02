import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest, torch, torch.nn as nn
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader
from adversarial_lab.models  import build_model
from adversarial_lab.core    import Trainer, TrainingConfig, AttackConfig
from adversarial_lab.utils   import set_seed

EPS=8/255; ALPHA=2/255; DEVICE=torch.device("cpu")

@pytest.fixture(scope="module")
def loaders():
    t=transforms.ToTensor()
    return (DataLoader(FakeData(size=32,image_size=(3,32,32),num_classes=10,transform=t),batch_size=32),
            DataLoader(FakeData(size=16,image_size=(3,32,32),num_classes=10,transform=t),batch_size=16))

@pytest.mark.parametrize("defense",["clean","adversarial"])
def test_trainer_runs(tmp_path, loaders, defense):
    set_seed(0)
    m=build_model("resnet18",10).to(DEVICE)
    o=torch.optim.SGD(m.parameters(),lr=1e-3)
    s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=1)
    tr=Trainer(m,o,s,nn.CrossEntropyLoss(),DEVICE,
               TrainingConfig(epochs=1,defense=defense,train_attack="pgd"),
               AttackConfig(epsilon=EPS,alpha=ALPHA,steps=2),output_dir=str(tmp_path))
    h=tr.fit(*loaders)
    assert len(h)==1 and "train_loss" in h[0] and "val_acc" in h[0]
