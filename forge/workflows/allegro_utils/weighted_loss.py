"""Weighted loss implementation for rare sample emphasis without oversampling."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WeightedMSELoss(nn.Module):
    """MSE loss with per-sample weights based on rarity and force magnitude.
    
    This provides an alternative to oversampling by applying higher weights
    to rare configurations during loss computation.
    """
    
    def __init__(self, 
                 rare_indices: Optional[list] = None,
                 rare_weight: float = 5.0,
                 force_weight_alpha: float = 0.25,
                 base_weight: float = 1.0):
        """
        Args:
            rare_indices: List of indices for rare configurations
            rare_weight: Weight multiplier for rare configurations
            force_weight_alpha: Scaling factor for force-based weighting
            base_weight: Base weight for all samples
        """
        super().__init__()
        self.rare_indices = set(rare_indices) if rare_indices else set()
        self.rare_weight = rare_weight
        self.force_weight_alpha = force_weight_alpha
        self.base_weight = base_weight
        
        logger.info(f"Initialized WeightedMSELoss with {len(self.rare_indices)} rare indices, "
                   f"rare_weight={rare_weight}, force_weight_alpha={force_weight_alpha}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                batch_idx: Optional[torch.Tensor] = None,
                force_norms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            pred: Predictions
            target: Targets
            batch_idx: Original indices of samples in the batch
            force_norms: Force norms for each sample (for force-based weighting)
            
        Returns:
            Weighted loss value
        """
        # Compute per-element squared error
        mse = (pred - target) ** 2
        
        # Initialize weights
        weights = torch.ones_like(mse) * self.base_weight
        
        # Apply rare sample weighting if indices provided
        if batch_idx is not None and len(self.rare_indices) > 0:
            for i, idx in enumerate(batch_idx):
                if idx.item() in self.rare_indices:
                    weights[i] *= self.rare_weight
        
        # Apply force-based weighting if force norms provided
        if force_norms is not None and self.force_weight_alpha > 0:
            # Expand force_norms to match mse shape if needed
            if force_norms.dim() < mse.dim():
                force_norms = force_norms.unsqueeze(-1).expand_as(mse)
            weights *= (1.0 + self.force_weight_alpha * force_norms)
        
        # Apply weights and compute mean
        weighted_mse = mse * weights
        return weighted_mse.mean()


class RareWeightedMetricsManager:
    """Manager to handle weighted loss computation with NequIP's MetricsManager.
    
    This wraps the standard metrics but applies sample-based weights.
    """
    
    def __init__(self, base_metrics_manager, rare_config: Dict[str, Any]):
        """
        Args:
            base_metrics_manager: The original MetricsManager from config
            rare_config: Configuration for rare sample weighting
        """
        self.base_manager = base_metrics_manager
        self.rare_indices = set(rare_config.get('rare_idx', []))
        self.rare_weight = rare_config.get('replica', 5.0)  # Use replica as weight
        self.force_alpha = rare_config.get('alpha', 0.25)
        
        logger.info(f"Initialized weighted loss manager with {len(self.rare_indices)} "
                   f"rare samples, weight={self.rare_weight}")
    
    def __call__(self, pred, ref, key, mean=True):
        """Compute weighted loss."""
        # Get base loss
        base_loss = self.base_manager(pred, ref, key, mean=False)
        
        # Apply weights based on batch indices
        if hasattr(ref, 'batch_idx'):
            weights = torch.ones_like(base_loss)
            for i, idx in enumerate(ref.batch_idx):
                if idx.item() in self.rare_indices:
                    weights[i] *= self.rare_weight
            
            # Apply force-based weighting if available
            if hasattr(ref, 'force_norm') and self.force_alpha > 0:
                weights *= (1.0 + self.force_alpha * ref.force_norm)
            
            weighted_loss = base_loss * weights
            return weighted_loss.mean() if mean else weighted_loss
        else:
            # No batch indices available, return base loss
            return base_loss.mean() if mean else base_loss 