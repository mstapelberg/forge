# forge/potentials/mace.py
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator as BaseMACECalculator
from ..core.calculator import MLCalculator
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MACECalculator(MLCalculator):
    """MACE calculator implementation supporting single model and ensemble predictions."""
    
    def __init__(
        self,
        model_paths: Union[str, List[str]],
        device: str = 'cuda',
        default_dtype: str = 'float32',
        **kwargs
    ):
        """
        Initialize MACE calculator.
        
        Args:
            model_paths: Path or list of paths to MACE model files
            device: Device to run calculations on ('cpu' or 'cuda')
            default_dtype: Default data type for calculations ('float32' or 'float64')
            **kwargs: Additional keyword arguments passed to MACE calculator
        """
        super().__init__(model_paths, device)
        self.default_dtype = default_dtype
        self.kwargs = kwargs
        self.is_ensemble = len(self.model_paths) > 1
        
    def _initialize_models(self) -> None:
        """Initialize MACE calculator(s)."""
        try:
            if self.is_ensemble:
                logger.info(f"Initializing ensemble of {len(self.model_paths)} MACE models")
                self.calculator = BaseMACECalculator(
                    model_paths=self.model_paths,
                    device=self.device,
                    default_dtype=self.default_dtype,
                    **self.kwargs
                )
            else:
                logger.info("Initializing single MACE model")
                self.calculator = BaseMACECalculator(
                    model_paths=self.model_paths[0],
                    device=self.device,
                    default_dtype=self.default_dtype,
                    **self.kwargs
                )
        except Exception as e:
            logger.error(f"Failed to initialize MACE calculator: {str(e)}")
            raise

    def calculate_forces(self, atoms: Atoms) -> np.ndarray:
        """
        Calculate forces using MACE model(s).
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            np.ndarray: Forces array of shape (n_models, n_atoms, 3) for ensemble
                       or (1, n_atoms, 3) for single model
        """
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator
        
        try:
            # Trigger force calculation
            atoms_copy.get_potential_energy()
            
            if self.is_ensemble:
                # For ensemble, return all predictions
                forces = atoms_copy.calc.results['forces_comm']
            else:
                # For single model, wrap forces in extra dimension
                forces = atoms_copy.calc.results['forces'][np.newaxis, ...]
            
            return forces
        
        except Exception as e:
            logger.error(f"Force calculation failed: {str(e)}")
            raise

    def calculate_energy(self, atoms: Atoms) -> np.ndarray:
        """
        Calculate energy using MACE model(s).
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            np.ndarray: Energy array of shape (n_models,) for ensemble
                       or (1,) for single model
        """
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator
        
        try:
            if self.is_ensemble:
                energies = []
                # Calculate energy with each model in ensemble
                for i in range(len(self.model_paths)):
                    energy = atoms_copy.calc.get_potential_energy(model_idx=i)
                    energies.append(energy)
                return np.array(energies)
            else:
                # Single model case
                energy = atoms_copy.get_potential_energy()
                return np.array([energy])
        
        except Exception as e:
            logger.error(f"Energy calculation failed: {str(e)}")
            raise

    def calculate_structure_uncertainty(
        self,
        atoms: Atoms,
        calculate_forces: bool = True,
        calculate_energy: bool = True
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate uncertainty metrics for a structure using ensemble predictions.
        
        Args:
            atoms: ASE Atoms object
            calculate_forces: Whether to calculate force uncertainties
            calculate_energy: Whether to calculate energy uncertainties
            
        Returns:
            Dict containing uncertainty metrics:
            - 'energy_std': Standard deviation of energy predictions
            - 'force_variance': Per-atom force variances
            - 'mean_force_variance': Mean force variance across atoms
            - 'max_force_variance': Maximum force variance across atoms
        """
        if not self.is_ensemble:
            raise ValueError("Uncertainty calculation requires ensemble of models")
            
        results = {}
        
        if calculate_energy:
            energies = self.calculate_energy(atoms)
            results['energy_std'] = float(np.std(energies))
        
        if calculate_forces:
            forces = self.calculate_forces(atoms)
            force_variances = self.calculate_force_variance(forces)
            results.update({
                'force_variance': force_variances,
                'mean_force_variance': float(np.mean(force_variances)),
                'max_force_variance': float(np.max(force_variances))
            })
        
        return results

    def get_model_versions(self) -> List[str]:
        """Get version information for loaded models."""
        versions = []
        for path in self.model_paths:
            try:
                # Load model metadata
                state_dict = torch.load(path, map_location='cpu')
                version = state_dict.get('version', 'unknown')
                versions.append(f"MACE v{version}")
            except Exception as e:
                logger.warning(f"Could not get version for model {path}: {str(e)}")
                versions.append("unknown")
        return versions

    def validate_atoms(self, atoms: Atoms) -> bool:
        """
        Validate whether an Atoms object can be calculated with current models.
        
        Args:
            atoms: ASE Atoms object to validate
            
        Returns:
            bool: Whether the structure is valid for calculation
            
        Raises:
            ValueError: If structure is invalid with description of why
        """
        try:
            # Copy atoms to avoid modifying original
            atoms_copy = atoms.copy()
            atoms_copy.calc = self.calculator
            
            # Try to calculate energy (this will validate the structure)
            atoms_copy.get_potential_energy()
            return True
            
        except Exception as e:
            raise ValueError(f"Structure validation failed: {str(e)}")