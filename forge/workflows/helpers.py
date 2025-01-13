import numpy as np
from ase.build import bulk


def determine_kpoint_grid(atoms, auto_kpoints=False, base_kpts=(4,4,4), gamma=True):
    """
    Determine k-point grid by scaling relative to a reference cell size.
    For example, if you use 4x4x4 for a 128-atom cell, it will scale
    appropriately for larger/smaller cells.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        The atomic structure for which we want to compute a k-point grid.
    auto_kpoints : bool
        If True, scales k-points based on cell size.
        If False, returns base_kpts (default 4,4,4).
    base_kpts : tuple(int, int, int)
        Reference k-point mesh for your reference system
        (e.g., 4,4,4 for a 128-atom cell)
    gamma : bool
        Whether to use gamma-centered k-points.
    
    Returns:
    --------
    (kpts, gamma_flag) : ((int,int,int), bool)
        K-point mesh and gamma-centering flag
    """
    if not auto_kpoints:
        return base_kpts, gamma
    
    # Get cell vectors
    cell = atoms.get_cell()
    
    # Calculate lengths of cell vectors
    lengths = np.sqrt(np.sum(cell**2, axis=1))
    
    # Reference cell size (e.g., for 128-atom BCC cell)
    ref_atoms = bulk('V', crystalstructure='bcc', a=3.01, cubic=True)
    ref_supercell = ref_atoms * (4,4,4)
    ref_lengths = np.sqrt(np.sum(ref_supercell.get_cell()**2, axis=1))
    print(ref_lengths)
    #ref_lengths = np.array([15.0, 15.0, 15.0])  # Example for 128-atom BCC
    
    # Scale k-points inversely with cell size
    # Larger cell → fewer k-points
    # Smaller cell → more k-points
    scaling = ref_lengths / lengths
    
    # Calculate new k-points, ensuring minimum of 1
    kpts = np.maximum(1, np.round(scaling * np.array(base_kpts))).astype(int)
    
    return tuple(kpts), gamma
