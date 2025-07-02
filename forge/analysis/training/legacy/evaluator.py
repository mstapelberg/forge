from typing import List, Dict, Any, Optional
from ase import Atoms
import numpy as np

class Evaluator:
    def __init__(self, calculators: Any):
        """
        Initialises the evaluator with one or more ASE calculators.
        This version runs sequentially and is suitable for GPU-based calculators.

        Args:
            calculators: A single calculator or a list/tuple of calculators.
        """
        if not isinstance(calculators, (list, tuple)):
            self.calcs = [calculators]
        else:
            self.calcs = calculators

    def evaluate(self, atoms_list: List[Atoms]) -> List[Optional[Dict[str, Dict]]]:
        """
        Evaluates a list of Atoms objects with the given calculators sequentially.

        For each structure in `atoms_list`, it runs every calculator provided
        during initialization.

        Args:
            atoms_list: A list of ASE Atoms objects to evaluate.

        Returns:
            A list of dictionaries, one for each input atoms object. Each dictionary
            maps a calculator's name to its prediction dictionary.
            Example: [{'calc_0': {'energy':...}}, {'calc_0': {'energy':...}}]
        """
        if not atoms_list:
            return []

        structured_results = []
        for atoms in atoms_list:
            atom_result_dict = {}
            for i, calc in enumerate(self.calcs):
                # Use a simple name for now. Could be improved to use calc.name if available.
                calc_name = getattr(calc, 'name', f'calc_{i}')
                try:
                    # Use a copy to ensure that calculators do not interfere with each other
                    atoms_copy = atoms.copy()
                    atoms_copy.calc = calc
                    
                    energy = atoms_copy.get_potential_energy()
                    forces = atoms_copy.get_forces()
                    stress = atoms_copy.get_stress(voigt=False)
                    
                    # Some calculators might store per-atom energy in results
                    per_atom_energy = atoms_copy.calc.results.get('energies')

                    calc_result = {
                        'energy': energy,
                        'per_atom_energy': per_atom_energy, # Can be None
                        'forces': forces,
                        'stress': stress
                    }
                    atom_result_dict[calc_name] = calc_result

                except Exception as e:
                    print(f"Warning: Calculation failed for structure with calculator '{calc_name}': {e}")
                    atom_result_dict[calc_name] = None
            
            structured_results.append(atom_result_dict)
            
        return structured_results

    # No close, __del__, __enter__, or __exit__ methods are needed for the sequential version. 