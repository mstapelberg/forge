# forge/workflows/allegro_utils/custom_metrics.py
from typing import Dict, Any

import torch
import torch.nn as nn


class TailMSE(nn.Module):
    """Computes the Mean Squared Error (MSE) on the tail of the error distribution.

    This metric is useful for evaluating model performance on the "hardest"
    examples in a dataset. It first calculates the absolute error between
    predictions and targets, then identifies a top quantile of these errors,
    and finally computes the MSE only for this subset of examples.

    Args:
        quantile (float): The quantile to use for tail selection. For example,
            a value of 0.9 will compute the MSE on the 10% of examples with
            the highest absolute error. Must be between 0 and 1.
            Defaults to 0.9.

    Examples:
        In a NequIP/Allegro config.yaml, this metric can be used as follows:

        ```yaml
        val_metrics:
          _target_: nequip.train.MetricsManager
          metrics:
            - name: forces_tail_mse
              field: forces
              metric:
                _target_: forge.workflows.allegro_utils.custom_metrics.TailMSE
                quantile: 0.95
        ```
    """

    def __init__(self, quantile: float = 0.9):
        super().__init__()
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, but got {quantile}")
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the tail MSE.

        Args:
            pred (torch.Tensor): The predicted tensor from the model.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            torch.Tensor: The calculated scalar loss value for the tail. If no
            elements are in the tail (e.g., for an empty input), returns a
            tensor with value 0.0.
        """
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        err = pred - target
        abs_err = torch.abs(err)

        # Determine the error threshold for the tail
        threshold = torch.quantile(abs_err, self.quantile)

        # Find the elements in the tail
        tail_mask = abs_err >= threshold
        
        # It's possible for the tail to be empty if all errors are identical
        # and below the quantile threshold (e.g., all zeros).
        if not torch.any(tail_mask):
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Compute MSE only on the tail
        tail_err = err[tail_mask]
        return torch.mean(tail_err.pow(2)) 