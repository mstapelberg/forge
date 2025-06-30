"""Difficulty scoring for structures based on multiple error characteristics."""
from typing import Dict, Optional
import numpy as np
from scipy.stats import kurtosis
import logging

logger = logging.getLogger(__name__)


def calculate_difficulty_score(
    force_error_magnitudes: np.ndarray,
    global_morans_i: float,
    dbscan_labels: np.ndarray,
    ensemble_force_variance: float,
    n_atoms: int,
    weights_dict: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Calculate a composite difficulty score for a structure.
    
    The difficulty score combines multiple components that indicate
    challenging structures for model training:
    1. Heavy-tail component: High kurtosis and extreme quantiles
    2. Spatial component: Spatial clustering of errors
    3. Ensemble variance: Disagreement between models
    
    Parameters
    ----------
    force_error_magnitudes : np.ndarray
        1D array of force error magnitudes for each atom.
    global_morans_i : float
        Global Moran's I statistic for spatial autocorrelation.
    dbscan_labels : np.ndarray
        Cluster labels from DBSCAN (-2: below threshold, -1: noise, >=0: cluster).
    ensemble_force_variance : float
        Average variance of forces across ensemble models.
    n_atoms : int
        Total number of atoms in the structure.
    weights_dict : Optional[Dict[str, float]]
        Dictionary with keys 'w_heavy_tail', 'w_spatial', 'w_ensemble'
        to weigh the components of the score.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'difficulty_metric': Total difficulty score
        - 'score_heavy_tail_metric': Heavy-tail component
        - 'score_spatial_metric': Spatial clustering component
        - 'score_ensemble_metric': Ensemble variance component
    """
    if weights_dict is None:
        weights_dict = {
            'w_heavy_tail': 1.0,
            'w_spatial': 1.0,
            'w_ensemble': 1.0
        }
    
    # --- 1. Heavy-tail component ---
    if force_error_magnitudes.size > 0:
        # Check for near-zero variance before calculating kurtosis
        if np.std(force_error_magnitudes) < 1e-9:
            kurtosis_val = 3.0
        else:
            kurtosis_val = np.nan_to_num(
                kurtosis(force_error_magnitudes, fisher=False), 
                nan=3.0
            )
        
        q99 = np.quantile(force_error_magnitudes, 0.99)
        q50 = np.quantile(force_error_magnitudes, 0.5)
        
        # Use safe logs to prevent errors with zero or negative values
        safe_log_kurtosis = np.log(kurtosis_val) if kurtosis_val > 0 else 0
        safe_log_q_ratio = np.log(q99 / q50) if q99 > 0 and q50 > 0 else 0
        heavy_tail_score = safe_log_kurtosis + safe_log_q_ratio
    else:
        heavy_tail_score = 0.0
    
    # --- 2. Spatial component ---
    # Find the size of the largest cluster of high-error atoms
    cluster_labels = dbscan_labels[dbscan_labels >= 0]
    if cluster_labels.size > 0:
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        largest_cluster_size = counts.max()
    else:
        largest_cluster_size = 0
    
    # Spatial score combines Moran's I (autocorrelation) with cluster prevalence
    spatial_score = np.abs(global_morans_i) * (largest_cluster_size / n_atoms)
    
    # --- 3. Ensemble variance component ---
    # Already normalized by the caller
    ensemble_score = ensemble_force_variance
    
    # --- 4. Composite score ---
    difficulty = (
        weights_dict['w_heavy_tail'] * heavy_tail_score +
        weights_dict['w_spatial'] * spatial_score +
        weights_dict['w_ensemble'] * ensemble_score
    )
    
    return {
        'difficulty_metric': difficulty,
        'score_heavy_tail_metric': heavy_tail_score,
        'score_spatial_metric': spatial_score,
        'score_ensemble_metric': ensemble_score
    }


def rank_structures_by_difficulty(
    structure_metrics: Dict[int, Dict[str, float]],
    score_key: str = 'difficulty_metric'
) -> list:
    """Rank structures by difficulty score.
    
    Parameters
    ----------
    structure_metrics : Dict[int, Dict[str, float]]
        Dictionary mapping structure IDs to their metrics.
    score_key : str
        Key to use for ranking (default: 'difficulty_metric').
        
    Returns
    -------
    list
        List of (structure_id, score) tuples sorted by descending score.
    """
    scored_structures = [
        (sid, metrics.get(score_key, 0.0))
        for sid, metrics in structure_metrics.items()
    ]
    
    return sorted(scored_structures, key=lambda x: x[1], reverse=True)


def categorize_difficulty_causes(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, bool]:
    """Categorize the likely causes of difficulty for a structure.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Structure metrics including kurtosis, Gini, Moran's I, etc.
    thresholds : Optional[Dict[str, float]]
        Custom thresholds for categorization.
        
    Returns
    -------
    Dict[str, bool]
        Categories of difficulty causes:
        - 'has_outliers': Heavy-tailed error distribution
        - 'has_spatial_clustering': Errors are spatially correlated
        - 'has_high_variance': High ensemble disagreement
        - 'likely_physics_issue': Combination suggesting physics problem
        - 'likely_geometry_issue': Combination suggesting geometry problem
    """
    if thresholds is None:
        thresholds = {
            'kurtosis_threshold': 5.0,
            'gini_threshold': 0.3,
            'morans_i_threshold': 0.1,
            'ensemble_var_threshold': 0.5
        }
    
    # Extract relevant metrics
    kurtosis_val = metrics.get('force_kurtosis_metric', 3.0)
    gini_val = metrics.get('force_gini_metric', 0.0)
    morans_i = metrics.get('morans_i_global_metric', 0.0)
    ensemble_var = metrics.get('score_ensemble_metric', 0.0)
    n_clusters = metrics.get('n_error_clusters_metric', 0)
    
    # Categorize
    categories = {
        'has_outliers': (
            kurtosis_val > thresholds['kurtosis_threshold'] and
            gini_val > thresholds['gini_threshold']
        ),
        'has_spatial_clustering': (
            abs(morans_i) > thresholds['morans_i_threshold'] or
            n_clusters > 0
        ),
        'has_high_variance': (
            ensemble_var > thresholds['ensemble_var_threshold']
        )
    }
    
    # Infer likely causes
    # Physics issues: localized errors with spatial patterns
    categories['likely_physics_issue'] = (
        categories['has_outliers'] and 
        categories['has_spatial_clustering']
    )
    
    # Geometry issues: high variance without clear spatial pattern
    categories['likely_geometry_issue'] = (
        categories['has_high_variance'] and
        not categories['has_spatial_clustering']
    )
    
    return categories 