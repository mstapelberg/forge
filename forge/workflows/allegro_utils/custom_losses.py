# mlip_utils/losses.py
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalMSELoss(nn.Module):
    """A Focal Loss variant for Mean Squared Error regression.

    This loss function applies a modulating factor to the standard MSE loss. The
    modulating factor, derived from the sigmoid of the absolute error, aims to
    down-weight the loss assigned to well-regressed examples, allowing the model
    to focus more on hard, mispredicted examples.

    This is particularly useful in scenarios with noisy data or significant
    outliers.

    The loss is calculated as:
    `loss = (w * (pred - target)^2).mean()`
    where `w = sigmoid(beta * |pred - target|)^gamma`

    Args:
        beta (float): A scaling factor that controls the steepness of the
            sigmoid function. Higher values make the weighting more sensitive
            to small errors. Defaults to 1.0.
        gamma (float): A focusing parameter that controls the strength of the
            down-weighting. Higher values apply stronger down-weighting to
            easy examples. Defaults to 2.0.

    Examples:
        In a NequIP/Allegro config.yaml, this loss can be used as follows:

        ```yaml
        loss:
          _target_: nequip.train.MetricsManager
          metrics:
            - name: forces_focal_mse
              field: forces
              coeff: 1.0
              metric:
                _target_: forge.workflows.allegro_utils.custom_losses.FocalMSELoss
                beta: 1.5
                gamma: 2.0
        ```
    """

    def __init__(self, beta: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the focal MSE loss.

        Args:
            pred (torch.Tensor): The predicted tensor from the model.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            torch.Tensor: The calculated scalar loss value.
        """
        err = pred - target
        # The modulating factor, w
        w = torch.sigmoid(self.beta * err.abs()) ** self.gamma
        # The final focal loss
        return (w * err.pow(2)).mean()


def focal_mse(pred, target, beta=1.0, gamma=2.0):
    err = pred - target
    w = torch.sigmoid(beta * err.abs()) ** gamma
    return (w * err.pow(2)).mean() 