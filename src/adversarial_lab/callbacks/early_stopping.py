from adversarial_lab.callbacks.base import Callback
class EarlyStopping(Callback):
    """Stop when monitored metric stops improving for `patience` epochs."""
    def __init__(self, monitor="val_acc", patience=10, mode="max"):
        self.monitor=monitor; self.patience=patience; self.mode=mode
        self._counter=0; self._best=float("-inf") if mode=="max" else float("inf"); self.stop=False
    def on_train_start(self, logs):
        self._counter=0; self._best=float("-inf") if self.mode=="max" else float("inf"); self.stop=False
    def on_epoch_end(self, epoch, logs):
        cur = logs.get(self.monitor)
        if cur is None: return
        improved = (cur > self._best) if self.mode=="max" else (cur < self._best)
        if improved: self._best=cur; self._counter=0
        else:
            self._counter += 1
            if self._counter >= self.patience: self.stop=True
