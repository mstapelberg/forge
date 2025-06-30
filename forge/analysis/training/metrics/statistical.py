"""Statistical metrics for force error analysis."""
from typing import Dict
import numpy as np
from scipy.stats import kurtosis, jarque_bera
import logging

from .base import BaseMetric
from .registry import register_metric

logger = logging.getLogger(__name__)


def gini_coefficient(x: np.ndarray) -> float:
    """Calculate the Gini coefficient of a 1D numpy array.
    
    A Gini coefficient of 0 expresses perfect equality, where all values are the
    same. A Gini coefficient of 1 expresses maximal inequality among values.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of values.
        
    Returns
    -------
    float
        Gini coefficient between 0 and 1.
    """
    if x is None or x.size == 0:
        return 0.0
    
    # The calculation requires non-negative values
    x_non_neg = x - np.amin(x) if np.amin(x) < 0 else x
    
    # And values cannot be all zero
    if np.all(x_non_neg == 0):
        return 0.0

    # Sort the array
    x_sorted = np.sort(x_non_neg)
    n = x_sorted.shape[0]
    index = np.arange(1, n + 1)
    
    # Gini coefficient formula
    return (np.sum((2 * index - n - 1) * x_sorted)) / (n * np.sum(x_sorted))


class ForceErrorStats(BaseMetric):
    """Comprehensive force error statistics.
    
    Computes multiple statistical metrics for force errors including:
    - RMSE, MAE, MAX
    - Quantiles (Q95, Q99)
    - Distribution shape (Gini, Kurtosis)
    - Normality test (Jarque-Bera)
    """
    
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        per_component: bool = False
    ) -> Dict[str, float]:
        """Calculate force error statistics.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted forces of shape (N_atoms, 3).
        ref : np.ndarray
            Reference forces of shape (N_atoms, 3).
        per_component : bool, optional
            If True, calculate stats per force component (x,y,z).
            
        Returns
        -------
        Dict[str, float]
            Dictionary of error statistics.
        """
        self.validate_inputs(pred, ref)
        
        if pred.ndim != 2 or pred.shape[1] != 3:
            raise ValueError("Forces must have shape (N_atoms, 3)")
        
        # Calculate error magnitudes
        diff_vectors = pred - ref
        diff_magnitudes = np.linalg.norm(diff_vectors, axis=1)
        
        # Handle empty or all-zero cases
        if diff_magnitudes.size == 0 or np.all(diff_magnitudes == 0):
            return self._add_metric_suffix({
                "force_rmse": 0.0,
                "force_mae": 0.0, 
                "force_max": 0.0,
                "force_q95": 0.0,
                "force_q99": 0.0,
                "force_gini": 0.0,
                "force_kurtosis": 3.0,
                "force_jb_pvalue": 1.0
            })
        
        # Check for near-zero variance
        results = {}
        if np.std(diff_magnitudes) < 1e-9:
            results.update({
                "force_rmse": np.sqrt(np.mean(diff_magnitudes**2)),
                "force_mae": np.mean(diff_magnitudes),
                "force_max": diff_magnitudes.max(),
                "force_q95": np.quantile(diff_magnitudes, 0.95),
                "force_q99": np.quantile(diff_magnitudes, 0.99),
                "force_gini": gini_coefficient(diff_magnitudes),
                "force_kurtosis": 3.0,
                "force_jb_pvalue": 1.0
            })
        else:
            results.update({
                "force_rmse": np.sqrt(np.mean(diff_magnitudes**2)),
                "force_mae": np.mean(diff_magnitudes),
                "force_max": diff_magnitudes.max(),
                "force_q95": np.quantile(diff_magnitudes, 0.95),
                "force_q99": np.quantile(diff_magnitudes, 0.99),
                "force_gini": gini_coefficient(diff_magnitudes),
                "force_kurtosis": kurtosis(diff_magnitudes, fisher=False),
                "force_jb_pvalue": jarque_bera(diff_magnitudes)[1]
            })
        
        # Add per-component stats if requested
        if per_component:
            for i, comp in enumerate(['x', 'y', 'z']):
                comp_diff = diff_vectors[:, i]
                results[f"force_{comp}_rmse"] = np.sqrt(np.mean(comp_diff**2))
                results[f"force_{comp}_mae"] = np.mean(np.abs(comp_diff))
        
        return self._add_metric_suffix(results)


class EnergyErrorStats(BaseMetric):
    """Energy error statistics normalized per atom."""
    
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        n_atoms: int,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate energy error statistics.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted total energy (scalar or array).
        ref : np.ndarray
            Reference total energy (scalar or array).
        n_atoms : int
            Number of atoms for normalization.
            
        Returns
        -------
        Dict[str, float]
            Per-atom energy error metrics.
        """
        # Handle scalar energies
        if np.isscalar(pred):
            pred = np.array([pred])
        if np.isscalar(ref):
            ref = np.array([ref])
            
        self.validate_inputs(pred, ref)
        
        if n_atoms <= 0:
            raise ValueError("n_atoms must be positive")
        
        # Calculate per-atom errors
        err_per_atom = (pred - ref) / n_atoms
        
        return self._add_metric_suffix({
            'energy_mae_per_atom': np.mean(np.abs(err_per_atom)),
            'energy_rmse_per_atom': np.sqrt(np.mean(err_per_atom**2)),
            'energy_max_per_atom': np.max(np.abs(err_per_atom))
        })


class StressErrorStats(BaseMetric):
    """Stress tensor error statistics."""
    
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate stress error statistics.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted stress tensor (3x3).
        ref : np.ndarray
            Reference stress tensor (3x3).
            
        Returns
        -------
        Dict[str, float]
            Stress error metrics.
        """
        self.validate_inputs(pred, ref)
        
        if pred.shape != (3, 3):
            raise ValueError("Stress must be 3x3 tensor")
        
        # Frobenius norm of error
        diff_tensor = pred - ref
        fro_error = np.linalg.norm(diff_tensor, 'fro')
        
        # Component-wise errors
        comp_errors = np.abs(diff_tensor.flatten())
        
        return self._add_metric_suffix({
            'stress_frobenius_error': fro_error,
            'stress_mae': np.mean(comp_errors),
            'stress_max': np.max(comp_errors),
            'stress_trace_error': np.abs(np.trace(pred) - np.trace(ref))
        })


# Register default statistical metrics
register_metric("force_stats", ForceErrorStats(), 
                description="Comprehensive force error statistics")
register_metric("energy_stats", EnergyErrorStats(),
                description="Per-atom energy error statistics") 
register_metric("stress_stats", StressErrorStats(),
                description="Stress tensor error statistics")


# Register individual metrics as functions
@register_metric("gini", description="Gini coefficient for error distribution")
def gini_metric(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """Calculate Gini coefficient of error magnitudes."""
    errors = np.linalg.norm(pred - ref, axis=-1)
    return {"gini": gini_coefficient(errors)}


@register_metric("kurtosis", description="Kurtosis of error distribution")
def kurtosis_metric(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """Calculate kurtosis of error magnitudes."""
    errors = np.linalg.norm(pred - ref, axis=-1)
    if np.std(errors) < 1e-9:
        return {"kurtosis": 3.0}
    return {"kurtosis": kurtosis(errors, fisher=False)} 