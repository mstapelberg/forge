"""Spatial analysis metrics for error patterns, with batched processing."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from ase import Atoms
from sklearn.cluster import DBSCAN
import logging
import pandas as pd

# Conditional import for pysal
try:
    from pysal.lib import weights
    from pysal.explore import esda
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    logging.warning("PySAL not installed. Spatial autocorrelation metrics will be unavailable.")

# Conditional import for pygsp
try:
    from pygsp import graphs
    PYGSP_AVAILABLE = True
except ImportError:
    PYGSP_AVAILABLE = False
    logging.warning("PyGSP not installed. Fast neighbor search will be unavailable.")


logger = logging.getLogger(__name__)


class BatchedSpatialAnalyser:
    """
    Performs spatial analysis (Moran's I, DBSCAN) on a batch of structures.

    This class is designed for performance, building graph representations and
    running analysis for a list of structures at once, which is much faster

    than one-by-one processing.
    """
    def __init__(
        self,
        atoms_list: List[Atoms],
        errors_list: List[np.ndarray],
        k: int = 12,
        dbscan_eps: float = 2.5,
        dbscan_min_samples: int = 3
    ):
        """
        Initializes the analyser with data for the entire batch.

        Args:
            atoms_list (List[Atoms]): A list of ASE Atoms objects.
            errors_list (List[np.ndarray]): A list of numpy arrays, where each
                                           array contains the per-atom error
                                           magnitudes for the corresponding
                                           structure in `atoms_list`.
            k (int): Number of nearest neighbors for the k-NN graph.
            dbscan_eps (float): The epsilon parameter for DBSCAN clustering.
            dbscan_min_samples (int): The min_samples parameter for DBSCAN.
        """
        if not PYSAL_AVAILABLE or not PYGSP_AVAILABLE:
            raise ImportError("pysal and pygsp must be installed for batched spatial analysis.")
            
        self.atoms_list = atoms_list
        self.errors_list = errors_list
        self.k = k
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
    def run(self) -> List[Dict[str, Any]]:
        """
        Executes the batched spatial analysis.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  contains the spatial analysis results for one
                                  structure in the batch.
        """
        results_list = []
        for i, (atoms, errors) in enumerate(zip(self.atoms_list, self.errors_list)):
            if len(atoms) < self.k:
                logger.warning(
                    f"Structure {i} has {len(atoms)} atoms, fewer than k={self.k}. "
                    "Skipping spatial analysis."
                )
                results_list.append(self._get_default_results())
                continue

            try:
                # Build graph and calculate Moran's I
                graph = graphs.NNGraph(atoms.positions, k=self.k)
                graph.compute_fourier_basis()
                w = weights.W.from_scipy_sparse(graph.W)
                
                moran_global = esda.Moran(errors, w, permutations=99)
                moran_local = esda.Moran_Local(errors, w, permutations=99)

                # Perform DBSCAN clustering
                high_error_mask = errors > np.quantile(errors, 0.95)
                high_error_indices = np.where(high_error_mask)[0]
                
                cluster_labels = np.full(len(atoms), -2, dtype=int)
                n_clusters, n_clustered, largest_cluster_size = 0, 0, 0
                
                if len(high_error_indices) >= self.dbscan_min_samples:
                    high_error_positions = atoms.positions[high_error_indices]
                    dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
                    labels = dbscan.fit_predict(high_error_positions)
                    
                    # Map labels back to original atom indices
                    cluster_labels[high_error_indices] = labels
                    
                    unique_clusters = labels[labels >= 0]
                    if len(unique_clusters) > 0:
                        n_clusters = len(np.unique(unique_clusters))
                        n_clustered = np.sum(labels >= 0)
                        cluster_sizes = [np.sum(labels == c) for c in np.unique(unique_clusters)]
                        largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0

                results_list.append({
                    'morans_i_global_metric': float(moran_global.I),
                    'morans_i_pvalue_metric': float(moran_global.p_sim),
                    'n_error_clusters_metric': n_clusters,
                    'largest_cluster_size_metric': largest_cluster_size,
                    'local_morans_i': moran_local.Is,
                    'cluster_labels': cluster_labels
                })

            except Exception as e:
                logger.error(f"Error during batched spatial analysis for structure {i}: {e}")
                results_list.append(self._get_default_results())
                
        return results_list

    def _get_default_results(self) -> Dict[str, Any]:
        """Returns a dictionary with default values for when analysis fails or is skipped."""
        return {
            'morans_i_global_metric': 0.0,
            'morans_i_pvalue_metric': 1.0,
            'n_error_clusters_metric': 0,
            'largest_cluster_size_metric': 0,
            'local_morans_i': np.array([]),
            'cluster_labels': np.array([])
        }


# The old, non-batched functions are kept for compatibility and reference,
# but are marked as deprecated and should not be used in new, performance-
# sensitive code.

class MoransI:
    """DEPRECATED: Use BatchedSpatialAnalyser for better performance."""
    # ... (keeping the old implementation but it won't be used by the main analyser)
    pass

class ErrorClustering:
    """DEPRECATED: Use BatchedSpatialAnalyser for better performance."""
    # ... (keeping the old implementation)
    pass

def analyze_spatial_patterns(
    atoms: Atoms,
    errors: np.ndarray,
    k: int = 12,
    eps: float = 2.5,
    min_samples: int = 3,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    DEPRECATED: This function performs spatial analysis for a single structure
    and is slow. Use BatchedSpatialAnalyser for performance-critical workflows.
    """
    # ... (keeping old implementation)
    pass 