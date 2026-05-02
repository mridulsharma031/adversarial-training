import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from adversarial_lab.models.normalize import NormalizeLayer
from adversarial_lab.data import CIFAR10_MEAN, CIFAR10_STD

class NormalizedResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        self.normalize = NormalizeLayer(CIFAR10_MEAN, CIFAR10_STD)
        weights  = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        if not pretrained:
            backbone.conv1  = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            backbone.maxpool = nn.Identity()
        self.backbone = backbone
    def forward(self,x): return self.backbone(self.normalize(x))

def build_model(arch="resnet18", num_classes=10, pretrained=False):
    if arch == "resnet18": return NormalizedResNet(num_classes,pretrained)
    raise ValueError(f"Unknown arch: {arch!r}")
