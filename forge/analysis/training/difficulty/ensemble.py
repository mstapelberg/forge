"""Ensemble variance calculations for uncertainty quantification."""
from typing import Dict, List, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_ensemble_variance(
    ensemble_predictions: List[Dict[str, Any]], 
    n_atoms: int
) -> Dict[str, float]:
    """Calculate the variance in predictions across an ensemble of models.
    
    This function quantifies the disagreement between models in an ensemble,
    which can indicate structures where the models are uncertain or where
    the potential energy surface is difficult to learn.
    
    Parameters
    ----------
    ensemble_predictions : List[Dict[str, Any]]
        A list of prediction dictionaries, one for each model in the ensemble.
        Each dict should contain 'energy', 'forces', and optionally 'stress'.
    n_atoms : int
        The number of atoms in the structure for normalization.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'energy_variance_per_atom_metric': Variance of total energy per atom
        - 'force_variance_metric': Average variance of force predictions
        - 'stress_variance_metric': Variance of stress tensor components
    """
    if not ensemble_predictions or len(ensemble_predictions) < 2:
        return {
            'energy_variance_per_atom_metric': 0.0,
            'force_variance_metric': 0.0,
            'stress_variance_metric': 0.0
        }
    
    # Filter out None predictions
    valid_predictions = [p for p in ensemble_predictions if p is not None]
    if len(valid_predictions) < 2:
        return {
            'energy_variance_per_atom_metric': 0.0,
            'force_variance_metric': 0.0,
            'stress_variance_metric': 0.0
        }
    
    # --- Energy variance ---
    energies = [
        p['energy'] for p in valid_predictions 
        if p and 'energy' in p and p['energy'] is not None
    ]
    energy_variance_per_atom = np.var(energies) / n_atoms if energies else 0.0
    
    # --- Force variance ---
    forces_list = [
        p['forces'] for p in valid_predictions 
        if p and 'forces' in p and p['forces'] is not None
    ]
    
    force_variance = 0.0
    if forces_list and len(forces_list) >= 2:
        try:
            forces_array = np.array(forces_list)  # Shape: (n_models, n_atoms, 3)
            
            # Calculate mean force vector for each atom across ensemble
            mean_forces = np.mean(forces_array, axis=0)  # Shape: (n_atoms, 3)
            
            # Calculate squared magnitude of deviation from mean for each model
            force_deviations_sq = np.linalg.norm(
                forces_array - mean_forces, axis=2
            )**2  # Shape: (n_models, n_atoms)
            
            # Average deviation across models to get variance for each atom
            force_variance_per_atom = np.mean(force_deviations_sq, axis=0)
            
            # Average across atoms for single scalar metric
            force_variance = np.mean(force_variance_per_atom)
        except Exception as e:
            logger.warning(f"Error calculating force variance: {e}")
            force_variance = 0.0
    
    # --- Stress variance ---
    stresses = [
        p['stress'] for p in valid_predictions
        if p and 'stress' in p and p['stress'] is not None
    ]
    
    stress_variance = 0.0
    if stresses and len(stresses) >= 2:
        try:
            # Flatten stress tensors and calculate variance
            stress_array = np.array([s.flatten() for s in stresses])
            stress_variance = np.mean(np.var(stress_array, axis=0))
        except Exception as e:
            logger.warning(f"Error calculating stress variance: {e}")
            stress_variance = 0.0
    
    return {
        'energy_variance_per_atom_metric': float(energy_variance_per_atom),
        'force_variance_metric': float(force_variance),
        'stress_variance_metric': float(stress_variance)
    }


def calculate_ensemble_agreement(
    ensemble_predictions: List[Dict[str, Any]],
    reference: Dict[str, Any],
    threshold: float = 0.1
) -> Dict[str, float]:
    """Calculate agreement metrics between ensemble predictions.
    
    Parameters
    ----------
    ensemble_predictions : List[Dict[str, Any]]
        List of prediction dictionaries from ensemble models.
    reference : Dict[str, Any]
        Reference values (from DFT/database).
    threshold : float
        Error threshold for determining agreement.
        
    Returns
    -------
    Dict[str, float]
        Agreement metrics including fraction of models within threshold.
    """
    valid_predictions = [p for p in ensemble_predictions if p is not None]
    if not valid_predictions:
        return {'ensemble_agreement_metric': 0.0}
    
    # Calculate force errors for each model
    ref_forces = reference.get('forces')
    if ref_forces is None:
        return {'ensemble_agreement_metric': 0.0}
    
    within_threshold = 0
    for pred in valid_predictions:
        if 'forces' not in pred or pred['forces'] is None:
            continue
            
        force_errors = np.linalg.norm(pred['forces'] - ref_forces, axis=1)
        if np.mean(force_errors) < threshold:
            within_threshold += 1
    
    agreement_fraction = within_threshold / len(valid_predictions) if valid_predictions else 0.0
    
    return {
        'ensemble_agreement_metric': agreement_fraction,
        'n_models_within_threshold_metric': within_threshold
    }


def identify_uncertain_atoms(
    ensemble_force_predictions: np.ndarray,
    uncertainty_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Identify atoms with high force prediction uncertainty.
    
    Parameters
    ----------
    ensemble_force_predictions : np.ndarray
        Array of shape (n_models, n_atoms, 3) with force predictions.
    uncertainty_threshold : Optional[float]
        Threshold for high uncertainty. If None, uses 90th percentile.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing uncertain atom indices and statistics.
    """
    if ensemble_force_predictions.ndim != 3:
        raise ValueError("Expected 3D array (n_models, n_atoms, 3)")
    
    n_models, n_atoms, _ = ensemble_force_predictions.shape
    
    if n_models < 2:
        return {
            'uncertain_atom_indices': [],
            'atom_uncertainties': np.zeros(n_atoms),
            'max_uncertainty': 0.0
        }
    
    # Calculate per-atom force uncertainty (std dev of force magnitude)
    force_magnitudes = np.linalg.norm(ensemble_force_predictions, axis=2)
    atom_uncertainties = np.std(force_magnitudes, axis=0)
    
    # Determine threshold
    if uncertainty_threshold is None:
        uncertainty_threshold = np.quantile(atom_uncertainties, 0.9)
    
    # Find uncertain atoms
    uncertain_mask = atom_uncertainties > uncertainty_threshold
    uncertain_indices = np.where(uncertain_mask)[0]
    
    return {
        'uncertain_atom_indices': uncertain_indices.tolist(),
        'atom_uncertainties': atom_uncertainties,
        'max_uncertainty': float(np.max(atom_uncertainties)),
        'mean_uncertainty': float(np.mean(atom_uncertainties)),
        'n_uncertain_atoms': len(uncertain_indices),
        'uncertainty_threshold': uncertainty_threshold
    } 