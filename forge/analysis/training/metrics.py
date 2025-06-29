from typing import Dict
import numpy as np
from scipy.stats import kurtosis, jarque_bera

def gini_coefficient(x: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a 1D numpy array.
    A Gini coefficient of 0 expresses perfect equality, where all values are the
    same. A Gini coefficient of 1 expresses maximal inequality among values.
    """
    if x is None or x.size == 0:
        return 0.0
    # The calculation requires non-negative values.
    # Shift values to be non-negative if necessary.
    x_non_neg = x - np.amin(x) if np.amin(x) < 0 else x
    
    # And values cannot be all zero.
    if np.all(x_non_neg == 0):
        return 0.0

    # Sort the array
    x_sorted = np.sort(x_non_neg)
    n = x_sorted.shape[0]
    index = np.arange(1, n + 1)
    
    # Gini coefficient formula
    return (np.sum((2 * index - n - 1) * x_sorted)) / (n * np.sum(x_sorted))

def force_error_stats(f_ref: np.ndarray, f_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes a dictionary of error statistics for atomic forces.

    The statistics are calculated based on the magnitude of the error vector
    for each atom.

    Args:
        f_ref: Reference forces array of shape (N_atoms, 3).
        f_pred: Predicted forces array of shape (N_atoms, 3).

    Returns:
        A dictionary of scalar error metrics. Returns empty dict if inputs are invalid.
    """
    if f_ref is None or f_pred is None or f_ref.shape != f_pred.shape:
        return {}

    diff_vectors = f_pred - f_ref
    diff = np.linalg.norm(diff_vectors, axis=1)

    # Handle empty or all-zero cases to avoid division by zero or NaN results
    if diff.size == 0 or np.all(diff == 0):
        return {
            "RMSE": 0.0, "MAE": 0.0, "MAX": 0.0, "Q95": 0.0, "Q99": 0.0,
            "Gini": 0.0, "Kurtosis": 3.0, "JB_p": 1.0
        }

    # Check for near-zero variance to prevent catastrophic cancellation warnings
    if np.std(diff) < 1e-9:
        return {
            "RMSE": np.sqrt(np.mean(diff**2)), "MAE": np.mean(diff), "MAX": diff.max(),
            "Q95": np.quantile(diff, 0.95), "Q99": np.quantile(diff, 0.99),
            "Gini": gini_coefficient(diff), "Kurtosis": 3.0, "JB_p": 1.0
        }

    stats = {
        "RMSE": np.sqrt(np.mean(diff**2)),
        "MAE": np.mean(diff),
        "MAX": diff.max(),
        "Q95": np.quantile(diff, 0.95),
        "Q99": np.quantile(diff, 0.99),
        "Gini": gini_coefficient(diff),
        "Kurtosis": kurtosis(diff, fisher=False), # Pearson's kurtosis (normal=3)
        "JB_p": jarque_bera(diff)[1]             # p-value for normality test
    }
    return stats

def energy_error_stats(e_ref: float, e_pred: float, n_atoms: int) -> Dict[str, float]:
    """
    Computes error statistics for total energy, normalized by number of atoms.

    Args:
        e_ref: Reference total energy.
        e_pred: Predicted total energy.
        n_atoms: Number of atoms in the structure.

    Returns:
        A dictionary of per-atom energy error metrics.
    """
    if e_ref is None or e_pred is None or n_atoms == 0:
        return {}
    
    err_per_atom = (e_pred - e_ref) / n_atoms
    
    return {
        'e_ref_per_atom': e_ref / n_atoms,
        'e_pred_per_atom': e_pred / n_atoms,
        'AE_per_atom': np.abs(err_per_atom),
        'SE_per_atom': err_per_atom**2
    }

def stress_error_stats(s_ref: np.ndarray, s_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes the Frobenius norm of the error in the 3x3 stress tensor.

    Args:
        s_ref: Reference stress tensor (3x3).
        s_pred: Predicted stress tensor (3x3).

    Returns:
        A dictionary containing the Frobenius norm of the stress error.
    """
    if s_ref is None or s_pred is None or s_ref.shape != (3, 3) or s_pred.shape != (3, 3):
        return {}
    
    diff_tensor = s_pred - s_ref
    error = np.linalg.norm(diff_tensor, 'fro')
    
    return {'stress_fro_err': error} 