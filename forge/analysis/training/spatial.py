from typing import Dict, Any
import numpy as np
from ase import Atoms
from sklearn.cluster import DBSCAN

# pysal is a key dependency for spatial statistics.
# We will conditionally import it to provide a better error message if not installed.
try:
    from pysal.lib import weights
    from pysal.explore import esda
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False

def morans_I(atoms: Atoms, values: np.ndarray, k: int = 12, permutations: int = 99) -> Dict[str, Any]:
    """
    Compute Global and Local Moran's I for a set of atomic values.

    This function builds a k-nearest neighbor graph considering periodic
    boundary conditions and uses it to compute spatial autocorrelation statistics.

    Args:
        atoms: The ASE Atoms object containing the geometry.
        values: A 1D array of scalar values (e.g., force error magnitudes)
                for each atom in the `atoms` object.
        k: The number of nearest neighbors to consider for the spatial weights matrix.
        permutations: The number of permutations to use for significance testing.
                      Set to 0 to skip simulation and return analytical results.

    Returns:
        A dictionary containing global and local Moran's I statistics.
        Returns an empty dictionary if PySAL is not installed.
    """
    if not PYSAL_AVAILABLE:
        print("[WARN] PySAL not installed. Skipping Moran's I calculation. `pip install pysal`")
        return {}
    
    if len(atoms) <= k:
        print(f"[WARN] Number of atoms ({len(atoms)}) is less than or equal to k ({k}). "
              f"Cannot compute Moran's I. Skipping.")
        return {}

    # 1. Build a k-NN graph, respecting periodic boundary conditions (PBCs)
    neighbor_dict = {}
    for i in range(len(atoms)):
        # get_distances with mic=True correctly handles PBCs
        distances = atoms.get_distances(i, range(len(atoms)), mic=True)
        # argsort gives the indices that would sort the array.
        # The first element (dist=0) is the atom itself, so we take the next k.
        sorted_indices = np.argsort(distances)
        neighbor_dict[i] = list(sorted_indices[1 : k + 1])
    
    # 2. Create a PySAL spatial weights object from the neighbor dictionary
    w = weights.W(neighbor_dict)
    w.transform = 'r' # Row-standardize the weights matrix

    # 3. Compute Global Moran's I
    moran_global = esda.Moran(values, w, permutations=permutations)
    
    # 4. Compute Local Moran's I (LISA)
    moran_local = esda.Moran_Local(values, w, permutations=permutations)

    return {
        'global_I': moran_global.I,
        'global_z': moran_global.z_sim,
        'global_p': moran_global.p_sim,
        'local_I': moran_local.Is,      # Array of local Moran's I values
        'local_p': moran_local.p_sim,   # Array of p-values for local Is
        'local_q': moran_local.q        # Array of quadrant codes (1=HH, 2=LH, 3=LL, 4=HL)
    }

def find_error_clusters(
    atoms: Atoms,
    errors: np.ndarray,
    threshold: float,
    eps: float,
    min_samples: int = 3
) -> np.ndarray:
    """
    Finds spatial clusters of atoms with high errors using DBSCAN.

    This function first identifies atoms where the error exceeds a given
    threshold, then performs DBSCAN clustering on their spatial coordinates.

    NOTE: This implementation of DBSCAN does NOT currently account for
    periodic boundary conditions, which may lead to incorrect clustering
    at the boundaries of the simulation cell.

    Args:
        atoms: The ASE Atoms object.
        errors: A 1D array of per-atom scalar errors.
        threshold: The minimum error value for an atom to be considered.
        eps: The maximum distance between two samples for one to be considered
             as in the neighborhood of the other. This is the `eps` parameter
             for DBSCAN.
        min_samples: The number of samples in a neighborhood for a point to be
                     considered as a core point.

    Returns:
        An array of integer labels for each atom. Atoms with errors below the
        threshold are labeled -2. Atoms above the threshold that are considered
        noise by DBSCAN are labeled -1. Clustered atoms are labeled >= 0.
    """
    high_error_indices = np.where(errors > threshold)[0]
    
    if len(high_error_indices) < min_samples:
        # Not enough high-error atoms to form a cluster
        return np.full(len(atoms), -2, dtype=int)

    high_error_coords = atoms.positions[high_error_indices]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(high_error_coords)

    # Map the cluster labels back to an array for all atoms
    full_labels = np.full(len(atoms), -2, dtype=int)
    for i, original_index in enumerate(high_error_indices):
        full_labels[original_index] = labels[i]
        
    return full_labels 