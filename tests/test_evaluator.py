import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest, torch, torch.nn as nn
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader
from adversarial_lab.models     import build_model
from adversarial_lab.evaluation import RobustnessEvaluator
from adversarial_lab.core       import EvaluationConfig, AttackConfig

EPS=8/255; ALPHA=2/255; DEVICE=torch.device("cpu")

@pytest.fixture(scope="module")
def model(): return build_model("resnet18",10).to(DEVICE)

@pytest.fixture(scope="module")
def loader():
    return DataLoader(FakeData(size=16,image_size=(3,32,32),num_classes=10,
                               transform=transforms.ToTensor()),batch_size=16)

def test_evaluator_rows(tmp_path,model,loader):
    ev=RobustnessEvaluator(model,nn.CrossEntropyLoss(),DEVICE,
                           EvaluationConfig(pgd_steps=2,epsilons=[0.0,EPS]),
                           AttackConfig(epsilon=EPS,alpha=ALPHA,steps=2))
    df=ev.evaluate(loader,output_csv=str(tmp_path/"rob.csv"))
    assert len(df)==3 and (tmp_path/"rob.csv").exists()

def test_evaluator_accuracy_range(tmp_path,model,loader):
    ev=RobustnessEvaluator(model,nn.CrossEntropyLoss(),DEVICE,
                           EvaluationConfig(pgd_steps=2,epsilons=[0.0,EPS]),
                           AttackConfig(epsilon=EPS,alpha=ALPHA,steps=2))
    df=ev.evaluate(loader,output_csv=str(tmp_path/"rob2.csv"))
    assert (df["accuracy"]>=0).all() and (df["accuracy"]<=1).all()
