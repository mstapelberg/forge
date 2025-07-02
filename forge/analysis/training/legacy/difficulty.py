from typing import Dict, List, Any
import numpy as np
from scipy.stats import kurtosis

def calculate_ensemble_variance(
    ensemble_predictions: List[Dict[str, Any]], n_atoms: int
) -> Dict[str, float]:
    """
    Calculates the variance in predictions across an ensemble of models.

    Args:
        ensemble_predictions: A list of prediction dictionaries, one for each
                              model in the ensemble. Each dict should contain
                              'energy' and 'forces'.
        n_atoms: The number of atoms in the structure.

    Returns:
        A dictionary containing the variance of energy and forces.
    """
    if not ensemble_predictions or len(ensemble_predictions) < 2:
        return {'energy_variance_per_atom': 0.0, 'force_variance': 0.0}

    # --- Energy variance ---
    # We calculate variance of total energy, then normalize by number of atoms
    energies = [p['energy'] for p in ensemble_predictions if p and 'energy' in p]
    energy_variance_per_atom = np.var(energies) / n_atoms if energies else 0.0

    # --- Force variance ---
    # This represents the average disagreement between models on the force vector.
    forces_list = [p['forces'] for p in ensemble_predictions if p and 'forces' in p]
    if not forces_list or len(forces_list) < 2:
        return {
            'energy_variance_per_atom': energy_variance_per_atom,
            'force_variance': 0.0
        }

    forces_array = np.array(forces_list)  # Shape: (n_models, n_atoms, 3)
    
    # Calculate the mean force vector for each atom across the ensemble
    mean_forces = np.mean(forces_array, axis=0)  # Shape: (n_atoms, 3)
    
    # Calculate the squared magnitude of the deviation from the mean for each model, for each atom
    # ||F_i - <F>||^2
    force_deviations_sq = np.linalg.norm(forces_array - mean_forces, axis=2)**2
    
    # Average this deviation across the models to get the variance for each atom
    force_variance_per_atom = np.mean(force_deviations_sq, axis=0) # Shape: (n_atoms,)
    
    # Average the per-atom variances to get a single scalar metric for the structure
    avg_force_variance = np.mean(force_variance_per_atom)

    # --- Stress variance ---
    # We calculate variance of total stress
    stresses = [p['stress'] for p in ensemble_predictions if p and 'stress' in p]
    stress_variance = np.var(stresses)

    return {
        'energy_variance_per_atom': energy_variance_per_atom,
        'force_variance': avg_force_variance,
        'stress_variance': stress_variance
    }

def calculate_difficulty_score(
    force_error_magnitudes: np.ndarray,
    global_morans_i: float,
    dbscan_labels: np.ndarray,
    ensemble_force_variance: float,
    n_atoms: int,
    weights_dict: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Calculates a composite difficulty score and its components for a structure.

    Args:
        force_error_magnitudes: 1D array of force error magnitudes for each atom.
        global_morans_i: The global Moran's I statistic for force errors.
        dbscan_labels: Cluster labels from DBSCAN. >= 0 for clusters,
                       -1 for noise, -2 for below-threshold atoms.
        ensemble_force_variance: The average variance of forces across the ensemble.
        n_atoms: Total number of atoms in the structure.
        weights_dict: A dictionary with keys 'w_heavy_tail', 'w_spatial', 'w_ensemble'
                      to weigh the components of the score.

    Returns:
        A dictionary containing the total difficulty score and its components.
    """
    if weights_dict is None:
        weights_dict = {'w_heavy_tail': 1.0, 'w_spatial': 1.0, 'w_ensemble': 1.0}

    # --- 1. Heavy-tail component ---
    if force_error_magnitudes.size > 0:
        # Check for near-zero variance before calculating kurtosis
        if np.std(force_error_magnitudes) < 1e-9:
             kurtosis_val = 3.0
        else:
             kurtosis_val = np.nan_to_num(kurtosis(force_error_magnitudes, fisher=False), nan=3.0)

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
    
    spatial_score = np.abs(global_morans_i) * (largest_cluster_size / n_atoms)

    # --- 3. Ensemble variance component (already provided) ---

    # --- 4. Composite score ---
    difficulty = (weights_dict['w_heavy_tail'] * heavy_tail_score +
                  weights_dict['w_spatial'] * spatial_score +
                  weights_dict['w_ensemble'] * ensemble_force_variance)

    return {
        'difficulty': difficulty,
        'score_heavy_tail': heavy_tail_score,
        'score_spatial': spatial_score,
        'score_ensemble_variance': ensemble_force_variance
    } 