"""Abstract interface for ensemble calculators."""

from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
from ase import Atoms


class BaseEnsembleCalculator(ABC):
    """Abstract base class for ensemble calculators.
    
    This interface defines the contract that all ensemble calculator backends
    must implement to work with the adversarial attack workflow.
    """
    
    @abstractmethod
    def forces_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate forces using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate forces for
            
        Returns:
            Forces array of shape (n_models, n_atoms, 3)
        """
        pass
    
    @abstractmethod
    def energies_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate energies using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate energies for
            
        Returns:
            Energies array of shape (n_models,)
        """
        pass
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Get the device (cpu/cuda) used by the calculator."""
        pass
    
    @property
    @abstractmethod
    def models(self) -> List[Any]:
        """Get the raw models in the ensemble."""
        pass
    
    @property
    @abstractmethod
    def z_table(self) -> Any:
        """Get the atomic number mapping table."""
        pass
    
    @property
    @abstractmethod
    def r_max(self) -> float:
        """Get the cutoff radius for the models."""
        pass
    
    def calculate_forces(self, atoms: Atoms) -> np.ndarray:
        """Legacy method name for backward compatibility."""
        return self.forces_all(atoms)
    
    def calculate_normalized_force_variance(self, forces: np.ndarray) -> np.ndarray:
        """Calculate normalized force variance across ensemble predictions.
        
        Args:
            forces: Forces array from forces_all() of shape (n_models, n_atoms, 3)
            
        Returns:
            Array of shape (n_atoms,) with normalized variances
        """
        # Calculate force magnitudes, avoiding division by zero
        force_magnitudes = np.linalg.norm(forces, axis=2, keepdims=True)
        force_magnitudes = np.where(force_magnitudes < 1e-10, 1.0, force_magnitudes)
        
        # Normalize forces
        normalized_forces = forces / force_magnitudes
        
        # Calculate variance across models for each atom
        atom_variances = np.var(normalized_forces, axis=0)
        total_atom_variances = np.sum(atom_variances, axis=1)
        return total_atom_variances 