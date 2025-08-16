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
from .custom_metrics import TailMSE, FocalMSELoss, TailHuberLoss, ForceAngleLoss, StressShearMAE, StressAngleLoss
from .config_aware_metrics import ConfigAwareStressHuber, PressureMAE, VonMisesMAE
from .weighted_loss import WeightedMSELoss, RareWeightedMetricsManager
from .pair_potential import NLH
from .custom_stats import ExtendedDataStatisticsManager

# Alias V3 as the standard CustomSamplingASEDataModule
CustomSamplingASEDataModule = CustomSamplingASEDataModuleV3

__all__ = [
    # Data Module (V3 is the standard)
    'CustomSamplingASEDataModule',  # Alias for V3
    'CustomSamplingASEDataModuleV3',
    
    # Samplers
    'RareWeightedSampler',
    
    # Metrics
    'TailMSE',
    'TailHuberLoss',
    'ForceAngleLoss',
    'StressShearMAE',
    'StressAngleLoss',
    'ConfigAwareStressHuber',
    'PressureMAE',
    'VonMisesMAE',
    
    # Callbacks
    'CurriculumCallback',
    'GradNormCallback',

    # Pair Potential
    'NLH',

    # Custom Stats
    'ExtendedDataStatisticsManager',
] 