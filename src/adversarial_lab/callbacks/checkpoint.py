from pathlib import Path
from adversarial_lab.callbacks.base import Callback
from adversarial_lab.utils.io import save_checkpoint
class ModelCheckpoint(Callback):
    """Save best.pt and last.pt during training."""
    def __init__(self, output_dir, monitor="val_acc", mode="max"):
        self.output_dir=Path(output_dir); self.monitor=monitor; self.mode=mode
        self._best=float("-inf") if mode=="max" else float("inf")
    def on_epoch_end(self, epoch, logs):
        state = logs.get("_state")
        if state is None: return
        save_checkpoint(state, str(self.output_dir/"last.pt"))
        cur = logs.get(self.monitor)
        improved = (cur is not None) and ((cur>self._best) if self.mode=="max" else (cur<self._best))
        if improved: self._best=cur; save_checkpoint(state, str(self.output_dir/"best.pt"))
