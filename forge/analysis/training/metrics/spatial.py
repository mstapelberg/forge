"""Spatial analysis metrics for error patterns."""
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ase import Atoms
from sklearn.cluster import DBSCAN
import logging

from .base import BaseMetric
from .registry import register_metric

logger = logging.getLogger(__name__)

# Conditional import for pysal
try:
    from pysal.lib import weights
    from pysal.explore import esda
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    logger.warning("PySAL not installed. Spatial autocorrelation metrics will be unavailable.")


class MoransI(BaseMetric):
    """Moran's I spatial autocorrelation metric."""
    
    def __init__(self, atoms: Atoms, k: int = 12, permutations: int = 99):
        """Initialize with atomic structure.
        
        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object containing the geometry.
        k : int, optional
            Number of nearest neighbors for spatial weights (default: 12).
        permutations : int, optional
            Number of permutations for significance testing (default: 99).
        """
        super().__init__()
        self.atoms = atoms
        self.k = k
        self.permutations = permutations
        
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        values: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate Moran's I for spatial autocorrelation.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted values (not used directly, for compatibility).
        ref : np.ndarray
            Reference values (not used directly, for compatibility).
        values : Optional[np.ndarray]
            Pre-computed error values to analyze. If None, uses |pred - ref|.
            
        Returns
        -------
        Dict[str, float]
            Moran's I statistics.
        """
        if not PYSAL_AVAILABLE:
            logger.warning("Skipping Moran's I calculation - PySAL not installed")
            return self._add_metric_suffix({
                "morans_i_global": 0.0,
                "morans_i_pvalue": 1.0
            })
        
        # Use provided values or compute error magnitudes
        if values is None:
            self.validate_inputs(pred, ref)
            if pred.ndim > 1:
                values = np.linalg.norm(pred - ref, axis=-1)
            else:
                values = np.abs(pred - ref)
        
        if len(self.atoms) <= self.k:
            logger.warning(
                f"Too few atoms ({len(self.atoms)}) for k={self.k} neighbors. "
                "Returning default values."
            )
            return self._add_metric_suffix({
                "morans_i_global": 0.0,
                "morans_i_pvalue": 1.0
            })
        
        try:
            # Build k-NN graph with periodic boundary conditions
            neighbor_dict = self._build_neighbor_dict()
            
            # Create spatial weights
            w = weights.W(neighbor_dict)
            w.transform = 'r'  # Row-standardize
            
            # Calculate global Moran's I
            moran_global = esda.Moran(values, w, permutations=self.permutations)
            
            # Calculate local Moran's I
            moran_local = esda.Moran_Local(values, w, permutations=self.permutations)
            
            # Store local values for per-atom analysis
            self.local_i = moran_local.Is
            self.local_p = moran_local.p_sim
            self.local_q = moran_local.q  # Quadrant codes
            
            return self._add_metric_suffix({
                "morans_i_global": float(moran_global.I),
                "morans_i_z_score": float(moran_global.z_sim),
                "morans_i_pvalue": float(moran_global.p_sim)
            })
            
        except Exception as e:
            logger.error(f"Error calculating Moran's I: {e}")
            return self._add_metric_suffix({
                "morans_i_global": 0.0,
                "morans_i_pvalue": 1.0
            })
    
    def _build_neighbor_dict(self) -> Dict[int, list]:
        """Build k-NN neighbor dictionary respecting PBCs."""
        neighbor_dict = {}
        
        for i in range(len(self.atoms)):
            # Get distances with PBC consideration
            distances = self.atoms.get_distances(i, range(len(self.atoms)), mic=True)
            # Sort and get k nearest (excluding self)
            sorted_indices = np.argsort(distances)
            neighbor_dict[i] = list(sorted_indices[1:self.k + 1])
            
        return neighbor_dict
    
    def get_local_values(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get local Moran's I values if calculated.
        
        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            Local I values, p-values, and quadrant codes if available.
        """
        if hasattr(self, 'local_i'):
            return self.local_i, self.local_p, self.local_q
        return None


class ErrorClustering(BaseMetric):
    """DBSCAN clustering of high-error atoms."""
    
    def __init__(
        self,
        atoms: Atoms,
        eps: float = 2.5,
        min_samples: int = 3,
        threshold_quantile: float = 0.95
    ):
        """Initialize clustering parameters.
        
        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object containing the geometry.
        eps : float, optional
            Maximum distance for DBSCAN neighborhood (default: 2.5 Å).
        min_samples : int, optional
            Minimum samples for core points (default: 3).
        threshold_quantile : float, optional
            Quantile for high-error threshold (default: 0.95).
        """
        super().__init__()
        self.atoms = atoms
        self.eps = eps
        self.min_samples = min_samples
        self.threshold_quantile = threshold_quantile
        
    def calculate(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        threshold: Optional[float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Find spatial clusters of high-error atoms.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted values.
        ref : np.ndarray
            Reference values.
        threshold : Optional[float]
            Error threshold. If None, uses threshold_quantile.
            
        Returns
        -------
        Dict[str, float]
            Clustering statistics.
        """
        self.validate_inputs(pred, ref)
        
        # Calculate error magnitudes
        if pred.ndim > 1:
            errors = np.linalg.norm(pred - ref, axis=-1)
        else:
            errors = np.abs(pred - ref)
        
        # Determine threshold
        if threshold is None:
            threshold = np.quantile(errors, self.threshold_quantile)
        
        # Find high-error atoms
        high_error_mask = errors > threshold
        high_error_indices = np.where(high_error_mask)[0]
        
        # Initialize labels (all atoms start as -2 = below threshold)
        self.labels = np.full(len(self.atoms), -2, dtype=int)
        
        if len(high_error_indices) < self.min_samples:
            # Not enough high-error atoms to form clusters
            return self._add_metric_suffix({
                "n_clusters": 0,
                "n_high_error_atoms": len(high_error_indices),
                "n_clustered_atoms": 0,
                "largest_cluster_size": 0,
                "clustering_ratio": 0.0
            })
        
        # Get positions of high-error atoms
        high_error_positions = self.atoms.positions[high_error_indices]
        
        # Run DBSCAN
        # Note: This doesn't account for PBCs - could be improved
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(high_error_positions)
        
        # Map back to full atom labels
        for i, idx in enumerate(high_error_indices):
            self.labels[idx] = cluster_labels[i]
        
        # Calculate statistics
        unique_clusters = cluster_labels[cluster_labels >= 0]
        n_clusters = len(np.unique(unique_clusters)) if len(unique_clusters) > 0 else 0
        n_clustered = np.sum(cluster_labels >= 0)
        
        cluster_sizes = []
        if n_clusters > 0:
            for cluster_id in np.unique(unique_clusters):
                cluster_sizes.append(np.sum(cluster_labels == cluster_id))
        
        return self._add_metric_suffix({
            "n_error_clusters": n_clusters,
            "n_high_error_atoms": len(high_error_indices),
            "n_clustered_atoms": n_clustered,
            "n_noise_atoms": np.sum(cluster_labels == -1),
            "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "mean_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "clustering_ratio": n_clustered / len(high_error_indices) if len(high_error_indices) > 0 else 0.0,
            "error_threshold": threshold
        })
    
    def get_labels(self) -> np.ndarray:
        """Get cluster labels for all atoms.
        
        Returns
        -------
        np.ndarray
            Array of cluster labels (-2: below threshold, -1: noise, >=0: cluster ID).
        """
        return self.labels.copy()


# Convenience function for spatial analysis
def analyze_spatial_patterns(
    atoms: Atoms,
    errors: np.ndarray,
    k: int = 12,
    eps: float = 2.5,
    min_samples: int = 3,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Perform complete spatial analysis of error patterns.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    errors : np.ndarray
        Error values for each atom.
    k : int, optional
        Number of neighbors for Moran's I.
    eps : float, optional
        DBSCAN epsilon parameter.
    min_samples : int, optional
        DBSCAN min_samples parameter.
    threshold : Optional[float]
        Error threshold for clustering.
        
    Returns
    -------
    Dict[str, Any]
        Combined spatial analysis results.
    """
    results = {}
    
    # Moran's I analysis
    if PYSAL_AVAILABLE:
        morans = MoransI(atoms, k=k)
        # Pass dummy arrays since we're using values parameter
        morans_results = morans.calculate(errors, errors, values=errors)
        results.update(morans_results)
        
        # Get local values
        local_values = morans.get_local_values()
        if local_values is not None:
            results['local_morans_i'] = local_values[0]
            results['local_morans_p'] = local_values[1]
    
    # Clustering analysis
    clustering = ErrorClustering(atoms, eps=eps, min_samples=min_samples)
    # Pass errors as both pred and ref for compatibility
    clustering_results = clustering.calculate(errors, np.zeros_like(errors), threshold=threshold)
    results.update(clustering_results)
    results['cluster_labels'] = clustering.get_labels()
    
    return results 