"""Abstract interface for ensemble calculators."""

from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
from ase import Atoms


class BaseEnsembleCalculator(ABC):
    """Abstract base class for ensemble calculators.
    
    This interface defines the contract that all ensemble calculator backends
    must implement to work with the adversarial attack workflow.
    
    Also implements ASE calculator interface for compatibility with ASE workflows.
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
    
    @abstractmethod
    def stresses_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate stresses using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate stresses for
            
        Returns:
            Stresses array of shape (n_models, 6) in Voigt notation
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
    
    # ASE Calculator Interface Methods
    def get_potential_energy(self, atoms: Atoms = None, force_consistent: bool = False) -> float:
        """ASE calculator interface: Get potential energy.
        
        Args:
            atoms: ASE Atoms object (optional, can be None if calculator is attached)
            force_consistent: Whether to use force-consistent energy (not implemented)
            
        Returns:
            Potential energy in eV
        """
        if atoms is None:
            # Try to get atoms from the calculator if it's attached
            if hasattr(self, '_atoms'):
                atoms = self._atoms
            else:
                raise ValueError("No atoms provided and no atoms attached to calculator")
        
        # Get energies from all models and return the mean
        energies = self.energies_all(atoms)
        return float(np.mean(energies))
    
    def get_forces(self, atoms: Atoms = None) -> np.ndarray:
        """ASE calculator interface: Get forces.
        
        Args:
            atoms: ASE Atoms object (optional, can be None if calculator is attached)
            
        Returns:
            Forces array of shape (n_atoms, 3)
        """
        if atoms is None:
            # Try to get atoms from the calculator if it's attached
            if hasattr(self, '_atoms'):
                atoms = self._atoms
            else:
                raise ValueError("No atoms provided and no atoms attached to calculator")
        
        # Get forces from all models and return the mean
        forces = self.forces_all(atoms)
        return np.mean(forces, axis=0)
    
    def get_stress(self, atoms: Atoms = None) -> np.ndarray:
        """ASE calculator interface: Get stress tensor.
        
        Args:
            atoms: ASE Atoms object (optional, can be None if calculator is attached)
            
        Returns:
            Stress tensor of shape (6,) in Voigt notation
        """
        if atoms is None:
            # Try to get atoms from the calculator if it's attached
            if hasattr(self, '_atoms'):
                atoms = self._atoms
            else:
                raise ValueError("No atoms provided and no atoms attached to calculator")
        
        # Get stresses from all models and return the mean
        stresses = self.stresses_all(atoms)
        return np.mean(stresses, axis=0)
    
    def calculation_required(self, atoms: Atoms, properties: List[str]) -> List[str]:
        """ASE calculator interface: Check which properties need calculation.
        
        Args:
            atoms: ASE Atoms object
            properties: List of properties to check
            
        Returns:
            List of properties that need calculation
        """
        # For ensemble calculators, we always need to recalculate
        # since we don't cache results
        return properties
    
    def set_atoms(self, atoms: Atoms):
        """ASE calculator interface: Set atoms for the calculator.
        
        Args:
            atoms: ASE Atoms object
        """
        self._atoms = atoms
    
    def get_atoms(self) -> Atoms:
        """ASE calculator interface: Get atoms from the calculator.
        
        Returns:
            ASE Atoms object if set, None otherwise
        """
        return getattr(self, '_atoms', None) 