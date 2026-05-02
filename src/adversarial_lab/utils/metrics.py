from dataclasses import dataclass, field
@dataclass
class AverageMeter:
    name: str; _sum: float = field(default=0.0,repr=False); _count: int = field(default=0,repr=False)
    def update(self, val, n=1): self._sum += val*n; self._count += n
    @property
    def avg(self): return self._sum / max(self._count, 1)
    def reset(self): self._sum=0.0; self._count=0
@dataclass
class EpochMetrics:
    loss: AverageMeter = field(default_factory=lambda: AverageMeter("loss"))
    acc:  AverageMeter = field(default_factory=lambda: AverageMeter("acc"))
    def update(self, loss, acc, n=1): self.loss.update(loss,n); self.acc.update(acc,n)
    def reset(self): self.loss.reset(); self.acc.reset()
    @property
    def avg_loss(self): return self.loss.avg
    @property
    def avg_acc(self): return self.acc.avg
