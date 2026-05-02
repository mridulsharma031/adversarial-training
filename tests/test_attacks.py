import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest, torch
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader
from adversarial_lab.models  import build_model
from adversarial_lab.attacks import FGSM, PGD

EPS=8/255; ALPHA=2/255

@pytest.fixture(scope="module")
def model(): m=build_model("resnet18",10); m.eval(); return m

@pytest.fixture(scope="module")
def batch():
    ds=FakeData(size=8,image_size=(3,32,32),num_classes=10,transform=transforms.ToTensor())
    return next(iter(DataLoader(ds, batch_size=8)))

def test_fgsm_shape(model,batch):
    imgs,lbls=batch; assert FGSM(model,EPS)(imgs,lbls).shape==imgs.shape

def test_fgsm_budget(model,batch):
    imgs,lbls=batch; adv=FGSM(model,EPS)(imgs,lbls)
    assert (adv-imgs).abs().max().item()<=EPS+1e-5

def test_fgsm_range(model,batch):
    imgs,lbls=batch; adv=FGSM(model,EPS)(imgs,lbls)
    assert adv.min().item()>=-1e-5 and adv.max().item()<=1+1e-5

def test_pgd_shape(model,batch):
    imgs,lbls=batch; assert PGD(model,EPS,ALPHA,steps=3)(imgs,lbls).shape==imgs.shape

def test_pgd_budget(model,batch):
    imgs,lbls=batch; adv=PGD(model,EPS,ALPHA,steps=3)(imgs,lbls)
    assert (adv-imgs).abs().max().item()<=EPS+1e-5

def test_pgd_range(model,batch):
    imgs,lbls=batch; adv=PGD(model,EPS,ALPHA,steps=3)(imgs,lbls)
    assert adv.min().item()>=-1e-5 and adv.max().item()<=1+1e-5

def test_mode_restored_after_fgsm(model,batch):
    imgs,lbls=batch; model.eval(); FGSM(model,EPS)(imgs,lbls); assert not model.training

def test_mode_restored_after_pgd(model,batch):
    imgs,lbls=batch; model.train(); PGD(model,EPS,ALPHA,3)(imgs,lbls); assert model.training
