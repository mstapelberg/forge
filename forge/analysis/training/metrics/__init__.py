"""Metrics module for forge analysis.

This module provides a comprehensive set of metrics for analyzing model errors,
including statistical metrics, spatial analysis, and custom loss functions.
"""
from .base import Metric, BaseMetric
from .registry import (
    MetricRegistry,
    register_metric,
    get_metric,
    calculate_metric,
    list_metrics,
    get_registry
)
from .statistical import (
    ForceErrorStats,
    EnergyErrorStats,
    StressErrorStats,
    gini_coefficient,
    gini_metric,
    kurtosis_metric
)
from .losses import (
    TailMSE,
    TailHuberLoss,
    FocalMSELoss,
    ForceAngleLoss,
    ErrorPercentiles
)
from .spatial import (
    MoransI,
    ErrorClustering,
    analyze_spatial_patterns
)

__all__ = [
    # Base classes
    'Metric',
    'BaseMetric',
    # Registry functions
    'MetricRegistry',
    'register_metric',
    'get_metric',
    'calculate_metric',
    'list_metrics',
    'get_registry',
    # Statistical metrics
    'ForceErrorStats',
    'EnergyErrorStats', 
    'StressErrorStats',
    'gini_coefficient',
    'gini_metric',
    'kurtosis_metric',
    # Loss metrics
    'TailMSE',
    'TailHuberLoss',
    'FocalMSELoss',
    'ForceAngleLoss',
    'ErrorPercentiles',
    # Spatial metrics
    'MoransI',
    'ErrorClustering',
    'analyze_spatial_patterns'
] 