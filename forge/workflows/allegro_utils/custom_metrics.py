# forge/workflows/allegro_utils/custom_metrics.py
from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric
from nequip.data import AtomicDataDict
from nequip.data.stats import _MeanX
from nequip.train.metrics import StratifiedHuberForceLoss

class TailMSE(_MeanX):
    """Computes a running mean of the Mean Squared Error on the tail of an error distribution.

    This metric is useful for evaluating model performance on the "hardest"
    examples in a dataset. In each `update` step (i.e., for each batch),
    it identifies a top quantile of errors and computes the MSE for this subset.
    The final metric is the mean of these per-batch tail MSEs over the whole epoch.

    This per-batch approach ensures the model focuses on high-error atoms
    within every structure, preventing them from being averaged out.

    Attributes:
        quantile (float): The quantile used for tail selection (0.0 to 1.0).
    """
    def __init__(self, quantile: float = 0.9, **kwargs):
        """Initializes the TailMSE metric.

        Args:
            quantile (float): The quantile to use for tail selection. For example,
                a value of 0.9 will compute the MSE on the 10% of examples with
                the highest absolute error within each batch. Must be between 0 and 1.
                Defaults to 0.9.
            **kwargs: Additional keyword arguments for the parent _MeanX class.
        """
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, but got {quantile}")
        self.quantile = quantile

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets for a single batch.

        Args:
            pred (torch.Tensor): The predicted tensor from the model.
            target (torch.Tensor): The ground truth tensor.
        """
        if pred.numel() == 0:
            return  # Do not update if the batch is empty

        err = pred - target
        abs_err = torch.abs(err)
        
        threshold = torch.quantile(abs_err.to(torch.float32), self.quantile).to(abs_err.device)
        tail_mask = abs_err >= threshold
        
        if not torch.any(tail_mask):
            return # Do not update if the tail is empty

        tail_err = err[tail_mask]
        squared_errors = tail_err.pow(2)
        super().update(squared_errors)

class AutoStratifiedHuberLoss(StratifiedHuberForceLoss):
    """A StratifiedHuberForceLoss that constructs its delta_dict from lists.

    This class is a convenience wrapper around NequIP's `StratifiedHuberForceLoss`.
    Instead of requiring a pre-constructed `delta_dict`, it takes two lists,
    `boundaries` and `deltas`, and builds the dictionary for you.

    This is particularly useful when the boundaries and deltas are computed
    dynamically and injected into a configuration file.

    Args:
        boundaries (List[float]): A list of lower bounds for force magnitudes.
        deltas (List[float]): A list of delta values for the Huber loss in each stratum.
        **kwargs: Additional keyword arguments for the parent class.
    """
    def __init__(self, boundaries: List[float], deltas: List[float], **kwargs):
        if len(boundaries) + 1 != len(deltas):
            raise ValueError(
                f"Number of deltas must be one greater than the number of boundaries. "
                f"Got {len(deltas)} deltas and {len(boundaries)} boundaries."
            )
        
        # The first delta corresponds to the region below the first boundary.
        # The first boundary in the dict should be 0.
        delta_dict = {0.0: deltas[0]}
        for i, boundary in enumerate(boundaries):
            delta_dict[boundary] = deltas[i+1]
        
        super().__init__(delta_dict=delta_dict, **kwargs)

class TailHuberLoss(_MeanX):
    """Computes a running mean of the Huber loss on the tail of the force error distribution.

    This metric selects a "tail" of the data based on the norm of the error
    vectors within each batch/structure. It then computes the Huber loss on this
    subset and maintains a running average of this loss across all batches.

    This per-batch approach ensures the model focuses on high-error atoms
    within every structure, preventing them from being averaged out.

    The `delta` parameter can be set to 'auto' for adaptive calculation.

    Attributes:
        delta (Union[float, str]): The threshold at which to change between L1 and L2 loss.
            If 'auto', it is dynamically calculated per batch.
        quantile (float): The quantile used for tail selection (0.0 to 1.0).
    """
    def __init__(self, delta: Union[float, str] = 1.0, quantile: float = 0.9, **kwargs):
        """Initializes the TailHuberLoss metric.

        Args:
            delta (Union[float, str]): The threshold for Huber loss. Can be a float or 'auto'.
                Defaults to 1.0.
            quantile (float): The quantile to use for tail selection.
                Defaults to 0.9.
            **kwargs: Additional keyword arguments for the parent _MeanX class.
        """
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, but got {quantile}")

        self.auto_delta = (delta == 'auto')
        self.delta = 1.0 if self.auto_delta else delta
        self.quantile = quantile

        if self.auto_delta:
            self.add_state("last_delta", default=torch.tensor(self.delta), dist_reduce_fx="mean")
        # Expose per-batch differentiable values for training loss assembly
        self.last_batch_value: Optional[torch.Tensor] = None

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets for a single batch.

        If `delta` is 'auto', it is recalculated for each batch as
        `clamp(1.5 * median(error_norm), 0.3, 5.0)`.

        Args:
            pred (torch.Tensor): The predicted tensor from the model.
            target (torch.Tensor): The ground truth tensor.
        """
        if pred.numel() == 0:
            # Differentiable zero tied to pred so grads can flow (as zero)
            self.last_batch_value = (pred - pred).sum(dim=-1)
            super().update(self.last_batch_value)
            return
            
        err = pred - target
        err_norm = torch.linalg.norm(err, dim=-1)

        if err_norm.numel() == 0:
            self.last_batch_value = (pred - pred).sum(dim=-1)
            super().update(self.last_batch_value)
            return

        threshold = torch.quantile(err_norm.to(torch.float32), self.quantile).to(err_norm.device)
        tail_mask = err_norm >= threshold

        if not torch.any(tail_mask):
            # No tail selected; expose differentiable zero tied to pred
            self.last_batch_value = (pred - pred).sum(dim=-1)
            super().update(self.last_batch_value)
            return

        tail_preds = pred[tail_mask]
        tail_targets = target[tail_mask]
        
        current_delta = self.delta
        if self.auto_delta:
            tail_err_norm = err_norm[tail_mask]
            median_err_norm = torch.median(tail_err_norm)
            current_delta_tensor = torch.clamp(1.5 * median_err_norm, 0.3, 5.0)
            self.last_delta = current_delta_tensor.detach()
            current_delta = self.last_delta.item()

        huber_losses = F.huber_loss(tail_preds, tail_targets, delta=current_delta, reduction='none')
        # Cache differentiable batch values for loss assembly
        self.last_batch_value = huber_losses
        super().update(huber_losses)

class FocalMSELoss(_MeanX):
    """A Focal Loss variant for Mean Squared Error regression.

    This loss function applies a modulating factor to the standard MSE loss. The
    modulating factor, derived from the sigmoid of the absolute error, aims to
    down-weight the loss assigned to well-regressed examples, allowing the model
    to focus more on hard, mispredicted examples.

    This is particularly useful in scenarios with noisy data or significant
    outliers.

    The loss is calculated as:
    `loss = (w * (pred - target)^2).mean()`
    where `w = (sigmoid(beta * |pred - target|) + eps)**gamma`

    Attributes:
        beta (float): Scaling factor for the sigmoid function.
        gamma (float): Focusing parameter for the down-weighting.
        eps (float): Small constant for numerical stability.
    """
    def __init__(self, beta: float = 1.0, gamma: float = 2.0, eps: float = 1e-6, **kwargs):
        """Initializes the FocalMSELoss metric.

        Args:
            beta (float): A scaling factor that controls the steepness of the
                sigmoid function. Higher values make the weighting more sensitive
                to small errors. Defaults to 1.0.
            gamma (float): A focusing parameter that controls the strength of the
                down-weighting. Higher values apply stronger down-weighting to
                easy examples. Defaults to 2.0.
            eps (float): A small constant added to the sigmoid function to prevent
                numerical instability. Defaults to 1e-6.
            **kwargs: Additional keyword arguments for the parent _MeanX class.
        """
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        This computes the focal-weighted squared error and updates the running
        mean.

        Args:
            pred (torch.Tensor): The predicted tensor from the model.
            target (torch.Tensor): The ground truth tensor.
        """
        err = pred - target
        # The modulating factor, w
        w = (torch.sigmoid(self.beta * err.abs()) + self.eps) ** self.gamma
        focal_loss = w * err.pow(2)
        super().update(focal_loss)

class ForceAngleLoss(_MeanX):
    """Computes the mean force angle loss (1 – cosθ).

    This loss penalizes deviations in the direction of the force vector,
    ignoring its magnitude. It is computed as the mean of `1 - cos(theta)`,
    where `theta` is the angle between the predicted and target force vectors.

    Attributes:
        eps (float): A small value to prevent division by zero when normalizing.
    """
    def __init__(self, eps: float = 1e-8, **kwargs):
        """Initializes the ForceAngleLoss metric.

        Args:
            eps (float): A small constant to avoid division by zero during
                normalization. Defaults to 1e-8.
            **kwargs: Additional keyword arguments for the parent _MeanX class.
        """
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        self.eps = eps

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred (torch.Tensor): The predicted force tensor.
            target (torch.Tensor): The ground truth force tensor.
        """
        pred_norm = pred.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        target_norm = target.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        cos_theta = (pred * target).sum(dim=-1, keepdim=True) / (pred_norm * target_norm)
        angle_error = torch.tensor(1.0, dtype=pred.dtype, device=pred.device) - cos_theta
        super().update(angle_error)

class StressShearMAE(_MeanX):
    """Computes the Mean Absolute Error on off-diagonal stress components.

    This metric calculates the MAE only on the shear components of the stress
    tensor (σ_xy, σ_xz, σ_yz), which correspond to indices (0,1), (0,2), and (1,2).
    This is useful for specifically targeting the shear modulus C₄₄.

    Attributes:
        offdiag (list): List of tuples with the off-diagonal indices.
    """
    offdiag: List[tuple[int, int]] = [(0, 1), (0, 2), (1, 2)]
    
    def __init__(self, **kwargs):
        """Initializes the StressShearMAE metric.

        Args:
            **kwargs: Additional keyword arguments for the parent _MeanX class.
        """
        super().__init__(modifier=torch.nn.Identity(), **kwargs)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred (torch.Tensor): The predicted stress tensor.
            target (torch.Tensor): The ground truth stress tensor.
        """
        abs_errs = []
        for i, j in self.offdiag:
            abs_errs.append(torch.abs(pred[:, i, j] - target[:, i, j]))
        
        # Stack and compute mean across shear components for each sample
        mean_abs_err = torch.mean(torch.stack(abs_errs, dim=0), dim=0)
        super().update(mean_abs_err)

class StressAngleLoss(_MeanX):
    """Computes the mean stress angle loss (1 – cosϕ).

    This loss penalizes deviations in the direction of the 6-dimensional
    Voigt-representation of the stress tensor. It is computed as the mean of
    `1 - cos(phi)`, where `phi` is the angle between the predicted and target
    stress vectors in Voigt space.

    Attributes:
        eps (float): A small value to prevent division by zero when normalizing.
    """
    def __init__(self, eps: float = 1e-8, **kwargs):
        """Initializes the StressAngleLoss metric.

        Args:
            eps (float): A small constant to avoid division by zero during
                normalization. Defaults to 1e-8.
            **kwargs: Additional keyword arguments for the parent _MeanX class.
        """
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        self.eps = eps

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred (torch.Tensor): The predicted stress tensor.
            target (torch.Tensor): The ground truth stress tensor.
        """
        voigt = self._to_voigt(pred)
        voigt_t = self._to_voigt(target)
        
        cos_phi = (voigt * voigt_t).sum(-1) / (voigt.norm(dim=-1) * voigt_t.norm(dim=-1) + self.eps)
        angle_error = torch.tensor(1.0, dtype=pred.dtype, device=pred.device) - cos_phi
        super().update(angle_error)

    @staticmethod
    def _to_voigt(stress: torch.Tensor) -> torch.Tensor:
        """Converts a 3x3 stress tensor to its 6D Voigt representation.

        Args:
            stress (torch.Tensor): A stress tensor of shape [N, 3, 3].

        Returns:
            torch.Tensor: The Voigt representation of shape [N, 6].
        """
        return torch.stack([
            stress[:, 0, 0], stress[:, 1, 1], stress[:, 2, 2],
            stress[:, 1, 2], stress[:, 0, 2], stress[:, 0, 1]
        ], dim=-1)

class VirialMSE(_MeanX):
    """Mean Squared Error on the full (symmetric) virial stress tensor.

    Computes the MSE over the six independent components of the 3×3 virial
    stress tensor in Voigt notation:

        (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)

    This captures both volumetric and shear behaviour.

    Args
    ----
    eps : float, optional
        Small constant to avoid division‑by‑zero in rare degenerate cases.
    **kwargs :
        Forwarded to the parent ``_MeanX`` class.
    """
    voigt_idx: List[Tuple[int, int]] = [
        (0, 0), (1, 1), (2, 2),  # normal stresses
        (1, 2), (0, 2), (0, 1),  # shear stresses (yz, xz, xy)
    ]

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        self.eps = eps

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate per‑sample squared error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted stress, shape ``(N, 3, 3)``.
        target : torch.Tensor
            Ground‑truth stress, same shape as ``pred``.
        """
        if pred.shape[-2:] != (3, 3) or target.shape[-2:] != (3, 3):
            raise ValueError("pred and target must have shape (..., 3, 3)")

        # Squared error for each independent component
        sq_errs = torch.stack(
            [(pred[:, i, j] - target[:, i, j])**2 for i, j in self.voigt_idx],
            dim=0,   # → (6, N)
        )

        # Mean over the six components for every sample
        mean_sq_err = sq_errs.mean(dim=0)   # → (N,)
        super().update(mean_sq_err)