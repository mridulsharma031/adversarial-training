# Adversarial Lab — Industry-Grade Adversarial ML on CIFAR-10

A production-quality adversarial machine learning framework with FGSM, PGD,
adversarial training, robustness evaluation, and visualisation.

## Quick Start (Windows PowerShell / macOS/Linux bash)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run Tests (FakeData — no download required)

```bash
pytest tests/ -v
```

## Train a Clean Baseline (downloads CIFAR-10 once ~170 MB)

```bash
python scripts/train.py --config configs/default.yaml --output-dir outputs/baseline
```

## Adversarial Training with PGD Defense

```bash
python scripts/train.py --config configs/default.yaml \
    --defense adversarial --output-dir outputs/adv_train
```

## Quick Debug Run (FakeData, 2 epochs)

```bash
python scripts/train.py --config configs/fast_debug.yaml
```

## Evaluate Robustness

```bash
python scripts/evaluate.py \
    --checkpoint outputs/adv_train/best.pt \
    --config configs/default.yaml \
    --output-dir outputs/eval_adv
```

## Visualize Adversarial Examples

```bash
python scripts/visualize.py \
    --checkpoint outputs/adv_train/best.pt \
    --output-dir outputs/vis_adv
```

## Project Structure

```
adversarial-industry-grade/
├── configs/            # YAML configs (default.yaml, fast_debug.yaml)
├── scripts/            # train.py, evaluate.py, visualize.py
├── tests/              # pytest test suite (13 tests)
└── src/adversarial_lab/
    ├── attacks/        # BaseAttack, FGSM, PGD, AutoAttack wrapper
    ├── callbacks/      # ModelCheckpoint, EarlyStopping
    ├── core/           # ExperimentConfig, Trainer, Runner
    ├── data/           # CIFAR-10 DataLoaders
    ├── defenses/       # Clean + Adversarial training defenses
    ├── evaluation/     # RobustnessEvaluator (CSV + plots)
    ├── models/         # NormalizedResNet18
    ├── utils/          # Logging, seed, checkpointing, metrics
    └── visualization/  # Adversarial grids, robustness curves
```

## Outputs (per run)

| File | Description |
|------|-------------|
| `best.pt` | Best checkpoint by val accuracy |
| `last.pt` | Last epoch checkpoint |
| `config.json` | Full experiment config snapshot |
| `training_history.csv` | Per-epoch loss/accuracy |
| `training_history.png` | Loss + accuracy curves |
| `robustness_results.csv` | Clean, FGSM, PGD accuracy vs epsilon |
| `robustness_curves.png` | Robustness plot |
| `fgsm_examples.png` | FGSM adversarial image grid |
| `pgd_examples.png` | PGD adversarial image grid |

## References

- Goodfellow et al. (2014) — FGSM: Explaining and Harnessing Adversarial Examples
- Madry et al. (2018) — PGD: Towards Deep Learning Models Resistant to Adversarial Attacks
