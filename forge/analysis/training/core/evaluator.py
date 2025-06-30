"""Evaluator for running calculations with ASE calculators."""
from typing import List, Dict, Any, Optional, Union
from ase import Atoms
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates structures using ASE calculators with validation.
    
    This class handles sequential evaluation of structures using one or more
    ASE calculators. It validates calculators on initialization and provides
    robust error handling during evaluation.
    
    Parameters
    ----------
    calculators : Any
        A single ASE calculator or a list/tuple of calculators.
        Each calculator must have get_potential_energy, get_forces,
        and get_stress methods.
    """
    
    def __init__(self, calculators: Any):
        """Initialize evaluator with calculator validation."""
        if not isinstance(calculators, (list, tuple)):
            self.calcs = [calculators]
        else:
            self.calcs = calculators
            
        # Support single calculator in a list
        if len(self.calcs) == 1:
            logger.info("Single calculator detected in ensemble")
            
        # Generate unique names for each calculator to prevent overwrites
        self.calc_names = []
        for i, calc in enumerate(self.calcs):
            base_name = getattr(calc, 'name', 'calc')
            self.calc_names.append(f"{base_name}_{i}")
            
        # Validate calculators
        self._validate_calculators()
        
    def _validate_calculators(self) -> None:
        """Validate that all calculators have required methods.
        
        Raises
        ------
        ValueError
            If any calculator lacks required methods.
        """
        required_methods = ['get_potential_energy', 'get_forces', 'get_stress']
        
        for i, calc in enumerate(self.calcs):
            calc_name = self.calc_names[i]
            
            for method in required_methods:
                if not hasattr(calc, method):
                    raise ValueError(
                        f"Calculator '{calc_name}' missing required "
                        f"method: {method}"
                    )
            
            logger.info(f"Validated calculator: {calc_name}")
    
    def evaluate(
        self, 
        atoms_list: List[Atoms],
        properties: Optional[List[str]] = None
    ) -> List[Optional[Dict[str, Dict[str, Any]]]]:
        """Evaluate a list of Atoms objects with the calculators.
        
        Parameters
        ----------
        atoms_list : List[Atoms]
            List of ASE Atoms objects to evaluate.
        properties : Optional[List[str]]
            Properties to calculate. Default: ['energy', 'forces', 'stress'].
            
        Returns
        -------
        List[Optional[Dict[str, Dict[str, Any]]]]
            List of dictionaries, one per input atoms object. Each dictionary
            maps calculator name to its prediction dictionary containing
            requested properties. None if evaluation failed for that structure.
        """
        if not atoms_list:
            return []
            
        if properties is None:
            properties = ['energy', 'forces', 'stress']
        
        structured_results = []
        
        for i, atoms in enumerate(atoms_list):
            atom_result_dict = {}
            
            for j, calc in enumerate(self.calcs):
                calc_name = self.calc_names[j]
                
                try:
                    # Use a copy to prevent calculator interference
                    atoms_copy = atoms.copy()
                    atoms_copy.calc = calc
                    
                    calc_result = {}
                    
                    if 'energy' in properties:
                        calc_result['energy'] = atoms_copy.get_potential_energy()
                    
                    if 'forces' in properties:
                        calc_result['forces'] = atoms_copy.get_forces()
                    
                    if 'stress' in properties:
                        calc_result['stress'] = atoms_copy.get_stress(voigt=False)
                    
                    # Check for per-atom energies if available
                    if 'per_atom_energy' in properties:
                        per_atom_energy = atoms_copy.calc.results.get('energies')
                        if per_atom_energy is not None:
                            calc_result['per_atom_energy'] = per_atom_energy
                    
                    atom_result_dict[calc_name] = calc_result
                    
                except Exception as e:
                    logger.warning(
                        f"Calculation failed for structure {i} with "
                        f"calculator '{calc_name}': {e}"
                    )
                    atom_result_dict[calc_name] = None
            
            structured_results.append(atom_result_dict)
        
        return structured_results
    
    def evaluate_single(
        self,
        atoms: Atoms,
        properties: Optional[List[str]] = None
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Evaluate a single Atoms object.
        
        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object to evaluate.
        properties : Optional[List[str]]
            Properties to calculate.
            
        Returns
        -------
        Optional[Dict[str, Dict[str, Any]]]
            Dictionary mapping calculator names to results.
        """
        results = self.evaluate([atoms], properties)
        return results[0] if results else None
    
    @property
    def n_calculators(self) -> int:
        """Number of calculators in the ensemble."""
        return len(self.calcs)
    
    @property
    def calculator_names(self) -> List[str]:
        """Names of all calculators."""
        return self.calc_names 