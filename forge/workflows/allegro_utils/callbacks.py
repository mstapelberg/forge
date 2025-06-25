# mlip_utils/callbacks.py
import lightning as L
import torch

class CurriculumCallback(L.Callback):
    def __init__(self, thresholds=(5, 15), epochs=(0, 5, 10)):
        self.tiers = list(zip(epochs, thresholds + (float("inf"),)))

    def on_train_epoch_start(self, trainer, pl_module):
        e = trainer.current_epoch
        maxF = next(t for epoch, t in reversed(self.tiers) if e >= epoch)
        pl_module.loss_extra = {"max_force": maxF}  # read in loss fn

class GradNormCallback(L.Callback):
    """Dynamic loss‑weight balancing à la Chen et al. 2018."""
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