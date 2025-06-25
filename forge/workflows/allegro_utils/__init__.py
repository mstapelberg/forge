"""
Allegro utility modules for customizing training.

This module provides enhanced training utilities for Allegro/NequIP models, 
focusing on handling rare sample emphasis and custom loss functions in 
distributed training environments.
"""

# Primary exports - V3 is now the standard implementation
from .data_v3 import CustomSamplingASEDataModuleV3
from .samplers import RareWeightedSampler
from .callbacks import CurriculumCallback, GradNormCallback
from .custom_losses import FocalMSELoss
from .custom_metrics import TailMSE
from .weighted_loss import WeightedMSELoss, RareWeightedMetricsManager

# Alias V3 as the standard CustomSamplingASEDataModule
CustomSamplingASEDataModule = CustomSamplingASEDataModuleV3

__all__ = [
    # Data Module (V3 is the standard)
    'CustomSamplingASEDataModule',  # Alias for V3
    'CustomSamplingASEDataModuleV3',
    
    # Samplers
    'RareWeightedSampler',
    
    # Loss Functions
    'FocalMSELoss',
    'WeightedMSELoss',
    'RareWeightedMetricsManager',
    
    # Metrics
    'TailMSE',
    
    # Callbacks
    'CurriculumCallback',
    'GradNormCallback',
    'FocalMSELoss',
    'TailMSE',
] 