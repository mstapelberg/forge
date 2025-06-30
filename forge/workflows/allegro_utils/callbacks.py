# mlip_utils/callbacks.py
import lightning as L
import torch
from typing import Any
from .custom_metrics import TailHuberLoss, FocalMSELoss

class CurriculumCallback(L.Callback):
    def __init__(self, thresholds=(5, 15), epochs=(0, 5, 10)):
        self.tiers = list(zip(epochs, thresholds + (float("inf"),)))

    def on_train_epoch_start(self, trainer, pl_module):
        e = trainer.current_epoch
        maxF = next(t for epoch, t in reversed(self.tiers) if e >= epoch)
        pl_module.loss_extra = {"max_force": maxF}  # read in loss fn

class DeltaLoggerCallback(L.Callback):
    """A callback to log the adaptive delta from TailHuberLoss."""
    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: dict, batch: Any, batch_idx: int
    ) -> None:
        """Log the 'last_delta' attribute from the TailHuberLoss metric."""
        if not hasattr(pl_module, "loss") or not hasattr(pl_module.loss, "metrics"):
            return

        # pl_module.loss is a MetricsManager (a ModuleDict), so we iterate its values.
        for metric_obj in pl_module.loss.values():
            if isinstance(metric_obj, TailHuberLoss) and hasattr(metric_obj, "last_delta"):
                pl_module.log(
                    "train_loss/auto_delta",
                    metric_obj.last_delta,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                )
                break # Assume only one such loss

class GradNormCallback(L.Callback):
    """Dynamic loss‑weight balancing à la Chen et al. 2018."""
    def __init__(self, alpha=1.5): self.alpha = alpha
    def on_before_optimizer_step(self, trainer, pl_module, _):
        losses = pl_module.logged_losses                # {name: tensor}
        if not hasattr(self, "L0"):
            self.L0 = {k: v.detach() for k, v in losses.items()}
        ratios = {k: (losses[k]/self.L0[k]).pow(self.alpha)
                  for k in losses}
        gnorms = {k: p.grad.norm()
                  for k, p in pl_module.named_parameters() if p.grad is not None}
        # update coeffs stored in pl_module.loss_coeffs
        for task in pl_module.loss_coeffs:
            target = gnorms[task] * (ratios[task] /
                                      torch.stack(list(ratios.values())).mean())
            pl_module.loss_coeffs[task] *= (gnorms[task] / target).detach() 