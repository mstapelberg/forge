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
        """Initialize MACE calculator."""
        super().__init__(model_paths, device)
        self.default_dtype = default_dtype
        self.kwargs = kwargs
        self.is_ensemble = len(self.model_paths) > 1
        
        # Initialize list of calculators
        if self.is_ensemble:
            self.calculators = [
                BaseMACECalculator(
                    model_paths=path,
                    device=self.device,
                    default_dtype=self.default_dtype,
                    **self.kwargs
                )
                for path in self.model_paths
            ]
        else:
            self.calculator = BaseMACECalculator(
                model_paths=self.model_paths[0],
                device=self.device,
                default_dtype=self.default_dtype,
                **self.kwargs
            )

    def calculate(self, atoms: Atoms) -> Dict[str, np.ndarray]:
        return {
            'energy': self.calculate_energy(atoms)[0],
            'forces': self.calculate_forces(atoms)[0]
        }

    def calculate_ensemble(self, atoms: Atoms) -> Dict[str, List[np.ndarray]]:
        return {
            'energies': self.calculate_energy(atoms).tolist(),
            'forces': self.calculate_forces(atoms).tolist()
        }
    
    def calculate_forces(self, atoms: Atoms) -> np.ndarray:
        """Calculate forces using MACE model(s)."""
        atoms_copy = atoms.copy()
        
        try:
            if self.is_ensemble:
                forces = []
                for calc in self.calculators:
                    atoms_copy.calc = calc
                    force = atoms_copy.get_forces()
                    forces.append(force)
                return np.array(forces)
            else:
                atoms_copy.calc = self.calculator
                force = atoms_copy.get_forces()
                return np.array([force])
        except Exception as e:
            logger.error(f"Force calculation failed: {str(e)}")
            raise

    def calculate_force_variance(self, forces: np.ndarray) -> np.ndarray:
        """Calculate variance of forces across ensemble predictions."""
        if not self.is_ensemble:
            raise ValueError("Force variance requires ensemble of models")
        # Calculate variance across models for each atom/direction
        # Then sum across directions
        return np.sum(np.var(forces, axis=0), axis=1)

    def calculate_energy(self, atoms: Atoms) -> np.ndarray:
        """Calculate energy using MACE model(s)."""
        atoms_copy = atoms.copy()
        
        try:
            if self.is_ensemble:
                energies = []
                for calc in self.calculators:
                    atoms_copy.calc = calc
                    energy = atoms_copy.get_potential_energy()
                    energies.append(energy)
                return np.array(energies)
            else:
                atoms_copy.calc = self.calculator
                energy = atoms_copy.get_potential_energy()
                return np.array([energy])
        except Exception as e:
            logger.error(f"Energy calculation failed: {str(e)}")
            raise

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

    def calculate_structure_uncertainty(
        self,
        atoms: Atoms,
        calculate_forces: bool = True,
        calculate_energy: bool = True
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Calculate uncertainty metrics using ensemble predictions."""
        if not self.is_ensemble:
            raise ValueError("Uncertainty calculation requires ensemble of models")
        
        results = {}
        
        if calculate_energy:
            energies = self.calculate_energy(atoms)
            results['energy_std'] = np.std(energies)
        
        if calculate_forces:
            forces = self.calculate_forces(atoms)
            force_vars = self.calculate_force_variance(forces)
            results['force_variance'] = force_vars
            results['mean_force_variance'] = np.mean(force_vars)
            results['max_force_variance'] = np.max(force_vars)
        
        return results