"""Optional AutoAttack wrapper (pip install autoattack)."""
import torch, torch.nn as nn
def run_autoattack(model,images,labels,epsilon,device,norm="Linf",version="standard"):
    try: from autoattack import AutoAttack
    except ImportError:
        raise ImportError("Run: pip install git+https://github.com/fra31/auto-attack")
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version, device=device)
    return adversary.run_standard_evaluation(images, labels, bs=images.shape[0])
