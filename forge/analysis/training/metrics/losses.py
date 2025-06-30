"""Custom loss function metrics for advanced error analysis."""
from typing import Dict
import numpy as np
import logging

from .base import BaseMetric
from .registry import register_metric

logger = logging.getLogger(__name__)


class TailMSE(BaseMetric):
    """Tail Mean Squared Error focusing on high-error outliers."""
    
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        quantile: float = 0.9,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate MSE for the tail of the error distribution.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted values.
        ref : np.ndarray
            Reference values.
        quantile : float, optional
            Quantile threshold for tail selection (default: 0.9).
            
        Returns
        -------
        Dict[str, float]
            Tail MSE metric.
        """
        self.validate_inputs(pred, ref)
        
        err = pred - ref
        abs_err = np.abs(err)
        
        if abs_err.size == 0:
            return self._add_metric_suffix({"tail_mse": 0.0})
        
        threshold = np.quantile(abs_err, quantile)
        tail_mask = abs_err >= threshold
        
        if not np.any(tail_mask):
            return self._add_metric_suffix({"tail_mse": 0.0})
        
        tail_err = err[tail_mask]
        return self._add_metric_suffix({
            "tail_mse": np.mean(tail_err**2),
            "tail_fraction": np.sum(tail_mask) / len(err)
        })


class TailHuberLoss(BaseMetric):
    """Huber loss applied to the tail of the error distribution."""
    
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        delta: float = 1.0,
        quantile: float = 0.9,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate Huber loss for high-error outliers.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted values.
        ref : np.ndarray
            Reference values.
        delta : float, optional
            Huber loss threshold (default: 1.0).
        quantile : float, optional
            Quantile threshold for tail selection (default: 0.9).
            
        Returns
        -------
        Dict[str, float]
            Tail Huber loss metric.
        """
        self.validate_inputs(pred, ref)
        
        err = pred - ref
        
        # Handle vector errors (e.g., forces)
        if err.ndim > 1:
            err_norm = np.linalg.norm(err, axis=-1)
        else:
            err_norm = np.abs(err)
            
        if err_norm.size == 0:
            return self._add_metric_suffix({"tail_huber": 0.0})
        
        threshold = np.quantile(err_norm, quantile)
        tail_mask = err_norm >= threshold
        
        if not np.any(tail_mask):
            return self._add_metric_suffix({"tail_huber": 0.0})
        
        # Apply Huber loss to tail errors
        if err.ndim > 1:
            tail_preds = pred[tail_mask]
            tail_refs = ref[tail_mask]
        else:
            tail_preds = pred[tail_mask]
            tail_refs = ref[tail_mask]
        
        tail_err = tail_preds - tail_refs
        abs_tail_err = np.abs(tail_err)
        
        # Huber loss calculation
        quadratic = np.minimum(abs_tail_err, delta)
        linear = abs_tail_err - quadratic
        huber_losses = 0.5 * quadratic**2 + delta * linear
        
        return self._add_metric_suffix({
            "tail_huber": np.mean(huber_losses),
            "tail_size": np.sum(tail_mask)
        })


class FocalMSELoss(BaseMetric):
    """Focal MSE loss that emphasizes hard examples."""
    
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        beta: float = 1.0,
        gamma: float = 2.0,
        eps: float = 1e-6,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate focal MSE loss.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted values.
        ref : np.ndarray
            Reference values.
        beta : float, optional
            Sigmoid scaling parameter (default: 1.0).
        gamma : float, optional
            Focusing parameter (default: 2.0).
        eps : float, optional
            Small constant for numerical stability (default: 1e-6).
            
        Returns
        -------
        Dict[str, float]
            Focal MSE loss metric.
        """
        self.validate_inputs(pred, ref)
        
        err = pred - ref
        
        # Sigmoid weighting based on error magnitude
        sigmoid = 1 / (1 + np.exp(-beta * np.abs(err)))
        
        # Focal weight
        weight = (sigmoid + eps) ** gamma
        
        # Weighted MSE
        focal_loss = weight * err**2
        
        return self._add_metric_suffix({
            "focal_mse": np.mean(focal_loss),
            "mean_weight": np.mean(weight),
            "max_weight": np.max(weight)
        })


class ForceAngleLoss(BaseMetric):
    """Angular error between force vectors."""
    
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        eps: float = 1e-8,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate angular error between predicted and reference forces.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted force vectors of shape (N, 3).
        ref : np.ndarray
            Reference force vectors of shape (N, 3).
        eps : float, optional
            Small constant for numerical stability (default: 1e-8).
            
        Returns
        -------
        Dict[str, float]
            Force angle loss metrics.
        """
        self.validate_inputs(pred, ref)
        
        if pred.ndim != 2 or pred.shape[1] != 3:
            raise ValueError("Forces must have shape (N, 3)")
        
        # Normalize vectors
        pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
        ref_norm = np.linalg.norm(ref, axis=-1, keepdims=True)
        
        # Cosine similarity
        cos_theta = np.sum(pred * ref, axis=-1, keepdims=True) / (
            np.maximum(pred_norm, eps) * np.maximum(ref_norm, eps)
        )
        
        # Clip to valid range for arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Angular error in radians
        angle_error = np.arccos(cos_theta).squeeze()
        
        # Loss is 1 - cos(theta) to be consistent with MSE-like losses
        angle_loss = 1.0 - cos_theta.squeeze()
        
        return self._add_metric_suffix({
            "force_angle_loss": np.mean(angle_loss),
            "mean_angle_rad": np.mean(angle_error),
            "max_angle_rad": np.max(angle_error),
            "mean_angle_deg": np.mean(angle_error) * 180 / np.pi,
            "max_angle_deg": np.max(angle_error) * 180 / np.pi
        })


# Additional utility metrics
class ErrorPercentiles(BaseMetric):
    """Calculate various percentiles of the error distribution."""
    
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        percentiles: list = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate error percentiles.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted values.
        ref : np.ndarray
            Reference values.
        percentiles : list, optional
            List of percentiles to calculate (default: [50, 75, 90, 95, 99]).
            
        Returns
        -------
        Dict[str, float]
            Error percentile metrics.
        """
        self.validate_inputs(pred, ref)
        
        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99]
        
        err = pred - ref
        if err.ndim > 1:
            err_mag = np.linalg.norm(err, axis=-1)
        else:
            err_mag = np.abs(err)
        
        results = {}
        for p in percentiles:
            results[f"error_p{p}"] = np.percentile(err_mag, p)
        
        return self._add_metric_suffix(results)


# Register custom loss metrics
register_metric("tail_mse", TailMSE(),
                description="MSE for high-error tail of distribution")
register_metric("tail_huber", TailHuberLoss(),
                description="Huber loss for high-error outliers")
register_metric("focal_mse", FocalMSELoss(),
                description="Focal loss emphasizing hard examples")
register_metric("force_angle", ForceAngleLoss(),
                description="Angular error between force vectors")
register_metric("error_percentiles", ErrorPercentiles(),
                description="Various percentiles of error distribution") 