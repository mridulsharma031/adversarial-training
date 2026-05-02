from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import numpy as np, pandas as pd, torch
from adversarial_lab.attacks.fgsm import FGSM
from adversarial_lab.attacks.pgd  import PGD
from adversarial_lab.utils        import CIFAR10_CLASSES

def _label(t): return CIFAR10_CLASSES[int(t.item())]

def save_adversarial_grid(model,loader,device,attack_name,epsilon,alpha,steps,num_images,output_path):
    model.eval()
    imgs,lbls = next(iter(loader))
    imgs=imgs[:num_images].to(device); lbls=lbls[:num_images].to(device)
    atk = FGSM(model,epsilon) if attack_name=="fgsm" else PGD(model,epsilon,alpha,steps)
    adv = atk(imgs,lbls)
    with torch.no_grad():
        cp=model(imgs).argmax(1); ap=model(adv).argmax(1)
    pert=adv-imgs; scale=pert.abs().amax(dim=(1,2,3),keepdim=True).clamp_min(1e-6)
    pv=(pert/scale+1.0)/2.0
    n=num_images; fig=plt.figure(figsize=(11,3.2*n))
    gs=gridspec.GridSpec(n,3,figure=fig,hspace=0.05,wspace=0.05)
    for i in range(n):
        for j,(title,img) in enumerate([
            (f"Clean\nGT:{_label(lbls[i])} Pred:{_label(cp[i])}", imgs[i]),
            (f"{attack_name.upper()} ε={epsilon:.4f}\nPred:{_label(ap[i])}", adv[i]),
            ("Scaled perturbation", pv[i])
        ]):
            ax=fig.add_subplot(gs[i,j]); ax.imshow(img.cpu().permute(1,2,0).numpy().clip(0,1))
            ax.set_title(title,fontsize=7); ax.axis("off")
    plt.suptitle(f"Adversarial Examples — {attack_name.upper()}",fontsize=10,y=1.01)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=150,bbox_inches="tight"); plt.close(fig)

def plot_robustness_curves(df, output_path):
    fig,ax=plt.subplots(figsize=(7,4))
    clean_acc=df.loc[df["attack"]=="clean","accuracy"].values[0]
    for atk,color,ls in [("fgsm","#E84855","--"),("pgd","#3D5A80","-")]:
        sub=df[df["attack"]==atk].sort_values("epsilon")
        ax.plot(sub["epsilon"]*255,sub["accuracy"]*100,label=atk.upper(),
                color=color,linestyle=ls,marker="o",linewidth=2)
    ax.axhline(clean_acc*100,color="#444",linestyle=":",linewidth=1.5,label="Clean")
    ax.set_xlabel("Epsilon (×255)"); ax.set_ylabel("Accuracy (%)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Robustness vs. Perturbation Budget"); fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=150,bbox_inches="tight"); plt.close(fig)

def plot_training_history(history, output_path):
    df = pd.DataFrame(history) if not isinstance(history,pd.DataFrame) else history
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
    a1.plot(df["epoch"],df["train_loss"],label="Train",color="#3D5A80")
    a1.plot(df["epoch"],df["val_loss"],  label="Val",  color="#E84855",linestyle="--")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(df["epoch"],df["train_acc"]*100,label="Train",color="#3D5A80")
    a2.plot(df["epoch"],df["val_acc"]*100,  label="Val",  color="#E84855",linestyle="--")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy (%)"); a2.set_title("Accuracy"); a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle("Training History"); fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=150,bbox_inches="tight"); plt.close(fig)
