from pathlib import Path
import torch
def ensure_dir(path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p
def save_checkpoint(state, path) -> None:
    p = Path(path); ensure_dir(p.parent); torch.save(state, p)
def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(str(path), map_location=map_location, weights_only=False)
