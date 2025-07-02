"""Forge analysis training module for identifying problematic structures.

This module provides comprehensive tools for analyzing model errors on atomic
structures, identifying difficult cases, and facilitating active learning workflows.

Main Components
--------------
ErrorAnalyser : Main class for orchestrating analysis
AnalysisResults : Structured results storage with convenient access methods
Metrics : Comprehensive set of error metrics including spatial analysis
Difficulty : Scoring and categorization of challenging structures
Utils : Geometry checks, I/O, and visualization tools

Example Usage
-------------
>>> from forge.core.database import DatabaseManager
>>> from forge.analysis.training import ErrorAnalyser
>>> 
>>> # Initialize with database and calculators
>>> db = DatabaseManager()
>>> analyser = ErrorAnalyser(db, calculators=[calc1, calc2])
>>> 
>>> # Run analysis
>>> results = analyser.run(structure_ids=[1, 2, 3, 4, 5])
>>> 
>>> # Get difficult structures
>>> difficult_ids = results.get_difficult_structures(top_n=10)
>>> 
>>> # Save results
>>> analyser.save("analysis_results/")
"""

# Core components
from .core import (
    ErrorAnalyser,
    AnalysisResults,
    Evaluator
)

# Metrics
from .metrics import (
    # Registry
    register_metric,
    get_metric,
    calculate_metric,
    list_metrics,
    # Statistical
    ForceErrorStats,
    EnergyErrorStats,
    StressErrorStats,
    gini_coefficient,
    # Losses
    TailMSE,
    TailHuberLoss,
    FocalMSELoss,
    ForceAngleLoss,
    # Spatial
    MoransI,
    ErrorClustering,
    analyze_spatial_patterns
)

# Difficulty analysis
from .difficulty import (
    calculate_difficulty_score,
    rank_structures_by_difficulty,
    categorize_difficulty_causes,
    calculate_ensemble_variance,
    calculate_ensemble_agreement
)

# Utilities
from .utils import (
    # Geometry
    check_geometry,
    batch_check_geometry,
    # I/O
    save_analysis_results,
    load_analysis_results,
    export_structures_to_extxyz,
    generate_markdown_report,
    # Plotting
    plot_force_error_histogram,
    plot_qq,
    plot_structure_ngl
)

__version__ = "0.1.0"

__all__ = [
    # Core
    'ErrorAnalyser',
    'AnalysisResults',
    'Evaluator',
    # Metrics
    'register_metric',
    'get_metric',
    'calculate_metric',
    'list_metrics',
    'ForceErrorStats',
    'EnergyErrorStats',
    'StressErrorStats',
    'gini_coefficient',
    'TailMSE',
    'TailHuberLoss',
    'FocalMSELoss',
    'ForceAngleLoss',
    'MoransI',
    'ErrorClustering',
    'analyze_spatial_patterns',
    # Difficulty
    'calculate_difficulty_score',
    'rank_structures_by_difficulty',
    'categorize_difficulty_causes',
    'calculate_ensemble_variance',
    'calculate_ensemble_agreement',
    # Utils
    'check_geometry',
    'batch_check_geometry',
    'save_analysis_results',
    'load_analysis_results',
    'export_structures_to_extxyz',
    'generate_markdown_report',
    'plot_force_error_histogram',
    'plot_qq',
    'plot_structure_ngl'
] 