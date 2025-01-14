import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, neighbor_list
from typing import List, Tuple, Dict


class WarrenCowleyCalculator:
    """Calculate Warren-Cowley short-range order parameters for an ASE Atoms object.

    The parameters quantify the deviation from random distribution of atomic species
    in different neighbor shells.
    """
    
    def __init__(self, atoms: Atoms, shells: List[float], cutoff: float = None):
        """
        Initialize the Warren-Cowley calculator.
        
        Args:
            atoms: ASE Atoms object to analyze
            shells: List of shell radii to consider (in Angstroms)
            cutoff: Maximum cutoff distance for neighbor search. If None, 
                   will be set to max(shells) + 0.5
        """
        self.atoms = atoms
        self.shells = np.array(shells)
        self.cutoff = cutoff if cutoff else max(shells) + 0.5
        
        # Get unique atomic types and their concentrations
        self.symbols = atoms.get_chemical_symbols()
        unique_symbols, counts = np.unique(self.symbols, return_counts=True)
        self.concentrations = {sym: count/len(atoms) for sym, count in zip(unique_symbols, counts)}
        
    def get_neighbors_by_shell(self) -> Dict[int, List[Tuple[int, int, float]]]:
        """Get neighbors grouped by shell."""
        # Get all neighbors within cutoff
        i, j, d = neighbor_list('ijd', self.atoms, self.cutoff)
        
        # Group by shells
        neighbors_by_shell = {}
        for shell_idx in range(len(self.shells)-1):
            min_r = self.shells[shell_idx]
            max_r = self.shells[shell_idx+1]
            shell_mask = (d >= min_r) & (d < max_r)
            neighbors_by_shell[shell_idx] = list(zip(i[shell_mask], j[shell_mask], d[shell_mask]))
            
        return neighbors_by_shell
    
    def calculate_parameters(self) -> Dict[str, np.ndarray]:
        """
        Calculate Warren-Cowley parameters for each shell.
        
        Returns:
            Dictionary mapping shell index to array of parameters.
            Parameters are stored as a matrix where entry [i,j] is the
            parameter for type i surrounded by type j.
        """
        neighbors_by_shell = self.get_neighbors_by_shell()
        parameters = {}
        
        for shell_idx, neighbors in neighbors_by_shell.items():
            if not neighbors:
                continue
                
            # Count occurrences of each type pair
            pair_counts = {}
            for i, j, _ in neighbors:
                type_i = self.symbols[i]
                type_j = self.symbols[j]
                pair_counts[(type_i, type_j)] = pair_counts.get((type_i, type_j), 0) + 1
            
            # Calculate parameters
            params = np.zeros((len(self.concentrations), len(self.concentrations)))
            for idx_i, type_i in enumerate(self.concentrations):
                for idx_j, type_j in enumerate(self.concentrations):
                    count = pair_counts.get((type_i, type_j), 0)
                    total_neighbors_i = sum(pair_counts.get((type_i, t), 0) 
                                         for t in self.concentrations)
                    if total_neighbors_i > 0:
                        pij = count / total_neighbors_i
                        params[idx_i, idx_j] = 1 - pij / self.concentrations[type_j]
            
            parameters[shell_idx] = params
            
            # Verify symmetry
            if not np.allclose(params, params.T):
                print(f"Warning: Parameters for shell {shell_idx} are not symmetric")
        
        return parameters
    
    def get_parameter_summary(self, parameters: Dict[str, np.ndarray]) -> str:
        """Generate a human-readable summary of the parameters."""
        summary = []
        types = list(self.concentrations.keys())
        
        for shell_idx, params in parameters.items():
            summary.append(f"\nShell {shell_idx}:")
            for i, type_i in enumerate(types):
                for j, type_j in enumerate(types):
                    summary.append(f"{type_i}-{type_j}: {params[i,j]:.3f}")
        
        return "\n".join(summary)


# Example usage:
if __name__ == "__main__":
    from ase.build import bulk
    
    # Create a test structure (random binary alloy)
    atoms = bulk('Cu', 'fcc', a=3.6) * (4, 4, 4)
    symbols = atoms.get_chemical_symbols()
    for i in range(len(atoms)//3):  # Replace 1/3 of atoms with Au
        symbols[i] = 'Au'
    atoms.set_chemical_symbols(symbols)
    
    # Calculate Warren-Cowley parameters
    shells = [0.0, 2.6, 3.7, 4.5]  # Typical FCC neighbor shells
    wc = WarrenCowleyCalculator(atoms, shells)
    params = wc.calculate_parameters()
    print(wc.get_parameter_summary(params))