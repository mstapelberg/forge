"""MACE backend for ensemble calculations."""

import numpy as np
import torch
from typing import List, Any, Union
from ase import Atoms

from .interface import BaseEnsembleCalculator

# Conditional import for MACE
try:
    from mace.calculators import MACECalculator
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    MACECalculator = None


class MACEBackend(BaseEnsembleCalculator):
    """MACE backend implementing BaseEnsembleCalculator interface."""
    
    def __init__(self, model_paths: Union[str, List[str]], device: str = 'cpu', 
                 default_dtype: str = 'float32', **kwargs):
        """Initialize MACE backend.
        
        Args:
            model_paths: Path(s) to MACE model file(s)
            device: Device to use ('cpu' or 'cuda')
            default_dtype: Default data type for calculations
            **kwargs: Additional arguments passed to MACECalculator
        """
        if not MACE_AVAILABLE:
            raise ImportError(
                "MACE is not available in this environment. "
                "Please install MACE or use a different backend. "
                "Install with: pip install mace-torch"
            )
            
        self._device = device
        self._default_dtype = default_dtype
        
        # Store model paths for reference
        if isinstance(model_paths, str):
            self.model_paths = [model_paths]
        else:
            self.model_paths = list(model_paths)
        
        # Initialize individual MACECalculators for each model
        self._calculators = []
        self._models = []
        
        try:
            for model_path in self.model_paths:
                calc_kwargs = {
                    'model_paths': model_path,  # Single model path for individual calculator
                    'device': device,
                    'default_dtype': default_dtype,
                    **kwargs
                }
                
                calc = MACECalculator(**calc_kwargs)
                self._calculators.append(calc)
                
                # Store reference to the underlying model for accessing properties
                # For MACE, the model is stored in calc.models[0] since we pass a single path
                if hasattr(calc, 'models') and calc.models:
                    self._models.append(calc.models[0])
                else:
                    # Fallback if models attribute structure is different
                    self._models.append(calc)
                    
        except Exception as e:
            print(f"[ERROR] Failed to initialize MACECalculator(s): {e}")
            raise
            
    def forces_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate forces using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate forces for
            
        Returns:
            Forces array of shape (n_models, n_atoms, 3)
        """
        forces_list = []
        
        for calc in self._calculators:
            # Clear previous results to prevent caching issues
            atoms.results = {}
            atoms.calc = calc
            try:
                # Force energy calculation to ensure forces are computed
                atoms.get_potential_energy()
                forces = atoms.get_forces()
                forces_list.append(forces)
            except Exception as e:
                print(f"Warning: Force calculation failed for model: {e}")
                # Return zeros if calculation fails
                return np.zeros((len(self._calculators), len(atoms), 3))
        
        return np.array(forces_list)
    
    def energies_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate energies using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate energies for
            
        Returns:
            Energies array of shape (n_models,)
        """
        energies = []
        
        for calc in self._calculators:
            atoms.results = {}
            atoms.calc = calc
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception as e:
                print(f"Warning: Energy calculation failed for model: {e}")
                # Return zeros if calculation fails
                return np.zeros(len(self._calculators))
        
        return np.array(energies)
    
    @property
    def device(self) -> str:
        """Get the device used by the calculator."""
        return self._device
    
    @property
    def models(self) -> List[Any]:
        """Get the raw models in the ensemble."""
        return self._models
    
    @property
    def z_table(self) -> Any:
        """Get the atomic number mapping table."""
        if self._calculators:
            # Get z_table from the first calculator
            return self._calculators[0].z_table
        else:
            raise RuntimeError("No calculators loaded")
    
    @property
    def r_max(self) -> float:
        """Get the cutoff radius for the models."""
        if self._models:
            try:
                # Get r_max from the first model
                r_max_val = self._models[0].r_max
                if hasattr(r_max_val, 'item'):
                    return r_max_val.item()
                else:
                    return float(r_max_val)
            except Exception as e:
                print(f"[WARN] Failed to get r_max from MACE model: {e}")
                return 5.0  # Default fallback
        else:
            raise RuntimeError("No models loaded")
    
    def get_mean_forces(self, atoms: Atoms) -> np.ndarray:
        """Get mean forces across ensemble (convenience method)."""
        all_forces = self.forces_all(atoms)
        return np.mean(all_forces, axis=0)
    
    def get_mean_energy(self, atoms: Atoms) -> float:
        """Get mean energy across ensemble (convenience method)."""
        all_energies = self.energies_all(atoms)
        return float(np.mean(all_energies)) 