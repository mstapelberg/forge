"""Geometry sanity checks for atomic structures."""
from typing import Dict, List, Any, Optional
import numpy as np
from ase import Atoms
from ase.geometry import get_duplicate_atoms
from ase.neighborlist import NeighborList
import logging

logger = logging.getLogger(__name__)

# Default cutoff for geometric checks (Angstroms)
DEFAULT_CUTOFF = 1.2


def check_geometry(
    atoms: Atoms,
    cutoff: float = DEFAULT_CUTOFF,
    check_duplicates: bool = True,
    check_distances: bool = True
) -> Dict[str, Any]:
    """Run geometry sanity checks on an atomic structure.
    
    This check identifies pairs of atoms closer than the specified cutoff distance.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object to check.
    cutoff : float, optional
        Distance cutoff in Angstroms for checks (default: 1.2).
    check_duplicates : bool, optional
        This parameter is maintained for backward compatibility. The primary check
        is now always performed based on interatomic distances.
    check_distances : bool, optional
        If False, the geometry check is skipped entirely.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'is_valid': bool indicating if structure passes all checks
        - 'has_duplicates': (legacy) now equivalent to 'has_close_atoms'
        - 'has_close_atoms': bool indicating atoms too close
        - 'close_pairs': list of (i, j) pairs that are too close
        - 'min_distance': minimum interatomic distance found
    """
    results = {
        'is_valid': True,
        'has_duplicates': False,
        'has_close_atoms': False,
        'close_pairs': [],
        'min_distance': None
    }
    
    if not check_distances:
        return results

    # The primary and most reliable check is a manual neighbor list build
    # followed by explicit distance calculation. This replaces the separate
    # get_duplicate_atoms check for clarity and robustness.
    nlist = NeighborList(
        [cutoff / 2] * len(atoms),
        self_interaction=False,
        bothways=True
    )
    nlist.update(atoms)
    
    close_pairs = []
    all_distances = []
    
    for i in range(len(atoms)):
        indices, offsets = nlist.get_neighbors(i)
        if len(indices) > 0:
            for j, offset in zip(indices, offsets):
                if j > i:  # Avoid double counting
                    dist = atoms.get_distance(i, j, mic=True)
                    all_distances.append(dist)
                    if dist < cutoff:
                        close_pairs.append((i, j))
    
    if all_distances:
        results['min_distance'] = np.min(all_distances)

    if close_pairs:
        results['has_close_atoms'] = True
        results['has_duplicates'] = True # Set for legacy compatibility
        results['close_pairs'] = close_pairs
        results['is_valid'] = False
        logger.warning(
            f"Found {len(close_pairs)} pairs of atoms closer than {cutoff} Å"
        )
    
    return results


def batch_check_geometry(
    atoms_list: List[Atoms],
    cutoff: float = DEFAULT_CUTOFF,
    **kwargs
) -> Dict[int, Dict[str, Any]]:
    """Check geometry for multiple structures.
    
    Parameters
    ----------
    atoms_list : List[Atoms]
        List of ASE Atoms objects to check.
    cutoff : float, optional
        Distance cutoff for checks.
    **kwargs
        Additional arguments passed to check_geometry.
        
    Returns
    -------
    Dict[int, Dict[str, Any]]
        Dictionary mapping structure index to check results.
    """
    results = {}
    
    for i, atoms in enumerate(atoms_list):
        try:
            results[i] = check_geometry(atoms, cutoff=cutoff, **kwargs)
        except Exception as e:
            logger.error(f"Error checking geometry for structure {i}: {e}")
            results[i] = {
                'is_valid': False,
                'error': str(e)
            }
    
    return results


def get_interatomic_distances(
    atoms: Atoms,
    indices: Optional[List[int]] = None,
    mic: bool = True
) -> Dict[str, Any]:
    """Calculate statistics on interatomic distances.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    indices : Optional[List[int]]
        Specific atom indices to analyze. If None, uses all atoms.
    mic : bool
        Whether to use minimum image convention for periodic systems.
        
    Returns
    -------
    Dict[str, Any]
        Distance statistics including min, max, mean, and distribution.
    """
    if indices is None:
        indices = list(range(len(atoms)))
    
    n_atoms = len(indices)
    if n_atoms < 2:
        return {
            'min_distance': None,
            'max_distance': None,
            'mean_distance': None,
            'distances': []
        }
    
    # Calculate all pairwise distances
    distances = []
    for i, idx_i in enumerate(indices):
        for j in range(i + 1, n_atoms):
            idx_j = indices[j]
            dist = atoms.get_distance(idx_i, idx_j, mic=mic)
            distances.append(dist)
    
    distances = np.array(distances)
    
    return {
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'n_pairs': len(distances),
        'distance_percentiles': {
            'p1': float(np.percentile(distances, 1)),
            'p5': float(np.percentile(distances, 5)),
            'p10': float(np.percentile(distances, 10)),
            'p25': float(np.percentile(distances, 25)),
            'p50': float(np.percentile(distances, 50))
        }
    }


def identify_surface_atoms(
    atoms: Atoms,
    cutoff_factor: float = 1.2
) -> List[int]:
    """Identify surface atoms based on coordination number.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    cutoff_factor : float
        Factor to multiply natural cutoff radius.
        
    Returns
    -------
    List[int]
        Indices of atoms identified as surface atoms.
    """
    from ase.neighborlist import natural_cutoffs
    
    # Get natural cutoffs for each atom
    cutoffs = natural_cutoffs(atoms, mult=cutoff_factor)
    
    # Build neighbor list
    nlist = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nlist.update(atoms)
    
    # Calculate coordination numbers
    coord_numbers = []
    for i in range(len(atoms)):
        neighbors = nlist.get_neighbors(i)[0]
        coord_numbers.append(len(neighbors))
    
    coord_numbers = np.array(coord_numbers)
    
    # Surface atoms typically have lower coordination
    # Use median as threshold
    threshold = np.median(coord_numbers)
    surface_indices = np.where(coord_numbers < threshold)[0]
    
    return surface_indices.tolist()


def analyze_bond_lengths(
    atoms: Atoms,
    species_pairs: Optional[List[tuple]] = None,
    cutoff_factor: float = 1.2
) -> Dict[str, Dict[str, float]]:
    """Analyze bond lengths between different species.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    species_pairs : Optional[List[tuple]]
        List of (species1, species2) pairs to analyze.
        If None, analyzes all present pairs.
    cutoff_factor : float
        Factor for natural cutoff radii.
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Bond length statistics for each species pair.
    """
    from ase.neighborlist import natural_cutoffs
    from collections import defaultdict
    
    # Get unique species
    symbols = atoms.get_chemical_symbols()
    unique_species = list(set(symbols))
    
    # Determine species pairs to analyze
    if species_pairs is None:
        species_pairs = []
        for i, sp1 in enumerate(unique_species):
            for sp2 in unique_species[i:]:
                species_pairs.append((sp1, sp2))
    
    # Build neighbor list
    cutoffs = natural_cutoffs(atoms, mult=cutoff_factor)
    nlist = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nlist.update(atoms)
    
    # Collect bond lengths
    bond_lengths = defaultdict(list)
    
    for i in range(len(atoms)):
        neighbors, offsets = nlist.get_neighbors(i)
        sp1 = symbols[i]
        
        for j, offset in zip(neighbors, offsets):
            sp2 = symbols[j]
            
            # Check if this pair should be analyzed
            pair_key = None
            if (sp1, sp2) in species_pairs:
                pair_key = f"{sp1}-{sp2}"
            elif (sp2, sp1) in species_pairs:
                pair_key = f"{sp2}-{sp1}"
            
            if pair_key:
                dist = atoms.get_distance(i, j, mic=True)
                bond_lengths[pair_key].append(dist)
    
    # Calculate statistics
    results = {}
    for pair_key, lengths in bond_lengths.items():
        if lengths:
            lengths = np.array(lengths)
            results[pair_key] = {
                'count': len(lengths),
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'min': float(np.min(lengths)),
                'max': float(np.max(lengths))
            }
        else:
            results[pair_key] = {
                'count': 0,
                'mean': None,
                'std': None,
                'min': None,
                'max': None
            }
    
    return results 