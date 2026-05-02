import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import argparse
from adversarial_lab.core import ExperimentConfig, run_train
def main():
    p = argparse.ArgumentParser(description="Train clean or adversarial CIFAR-10 model")
    p.add_argument("--config",         default=None)
    p.add_argument("--output-dir",     dest="output_dir",    default=None)
    p.add_argument("--epochs",         type=int,             default=None)
    p.add_argument("--batch-size",     dest="batch_size",    type=int, default=None)
    p.add_argument("--lr",             type=float,           default=None)
    p.add_argument("--defense",        default=None, choices=["clean","adversarial"])
    p.add_argument("--epsilon",        type=float,           default=None)
    p.add_argument("--alpha",          type=float,           default=None)
    p.add_argument("--attack-steps",   dest="attack_steps",  type=int, default=None)
    p.add_argument("--pgd-steps",      dest="pgd_steps",     type=int, default=None)
    p.add_argument("--seed",           type=int,             default=None)
    p.add_argument("--use-fake-data",  dest="use_fake_data", action="store_true")
    p.add_argument("--checkpoint",     default=None)
    args = p.parse_args()
    run_train(ExperimentConfig.from_cli(args), checkpoint=args.checkpoint)
if __name__ == "__main__": main()
