"""MACE backend for ensemble calculations."""

import numpy as np
import torch
from typing import List, Any, Union
from ase import Atoms
from pathlib import Path

from .interface import BaseEnsembleCalculator

# Conditional import for MACE
try:
    from mace.calculators.mace import MACECalculator
    from mace.data import AtomicData, config_from_atoms
    from mace.tools.torch_geometric import Batch
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    MACECalculator = None
    AtomicData = None
    config_from_atoms = None
    Batch = None


class MACEBackend(BaseEnsembleCalculator):
    """MACE backend implementing BaseEnsembleCalculator interface."""
    
    def __init__(self, model_paths: Union[str, List[str]], device: str = 'cpu', **kwargs):
        """Initialize MACE backend.
        
        Args:
            model_paths: Path(s) to MACE model file(s) (.model files)
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional arguments passed to MACECalculator
        """
        if not MACE_AVAILABLE:
            raise ImportError(
                "MACE is not available in this environment. "
                "Please install MACE or use a different backend. "
                "Install with: pip install mace"
            )
            
        self._device = device
        self._kwargs = kwargs  # Store kwargs for passing to calculators
        self._atoms = None  # Store attached atoms for ASE interface
        
        # Store model paths for reference
        if isinstance(model_paths, str):
            self.model_paths = [model_paths]
        else:
            self.model_paths = list(model_paths)
        
        # Initialize individual MACECalculators for each model
        self._calculators = []
        self._models = []
        
        # Store metadata from the first model
        self._r_max = None
        self._z_table = None
        
        try:
            for i, model_path in enumerate(self.model_paths):
                # Ensure the model file exists
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                print(f"[INFO] Loading MACE model: {Path(model_path).name}")
                
                # Create MACE calculator
                calc = MACECalculator(
                    model_paths=[model_path],  # MACE expects a list
                    device=device,
                    **kwargs
                )
                
                # Extract metadata from the first model
                if i == 0:
                    # Get r_max from the model
                    if hasattr(calc, 'r_max'):
                        self._r_max = calc.r_max
                    else:
                        # Default r_max for MACE
                        self._r_max = 5.0
                    
                    # Get z_table (atomic numbers)
                    if hasattr(calc, 'z_table'):
                        self._z_table = calc.z_table
                    else:
                        # Try to get from model
                        try:
                            self._z_table = calc.model.z_table
                        except:
                            self._z_table = None
                
                self._calculators.append(calc)
                self._models.append(calc.model)
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MACE backend: {e}")
        
        if not self._calculators:
            raise RuntimeError("No calculators were successfully initialized")
        
        print(f"Successfully initialized MACEBackend with {len(self._calculators)} model(s)")
    
    def forces_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate forces using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate forces for
            
        Returns:
            Forces array of shape (n_models, n_atoms, 3)
        """
        forces_list = []
        
        for i, calc in enumerate(self._calculators):
            # Get parameters from the calculator
            r_max = calc.r_max.item() if hasattr(calc.r_max, 'item') else calc.r_max
            z_table = calc.z_table
            
            # Create MACE data format
            config = config_from_atoms(atoms)
            data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            data = Batch.from_data_list([data]).to(self._device)
            
            # Get the model
            model = self._models[i]
            
            # Forward pass to get forces
            with torch.no_grad():
                output = model(data)
                forces = output.forces.cpu().numpy()
            
            forces_list.append(forces)
        
        return np.array(forces_list)
    
    def energies_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate energies using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate energies for
            
        Returns:
            Energies array of shape (n_models,)
        """
        energies_list = []
        
        for i, calc in enumerate(self._calculators):
            # Get parameters from the calculator
            r_max = calc.r_max.item() if hasattr(calc.r_max, 'item') else calc.r_max
            z_table = calc.z_table
            
            # Create MACE data format
            config = config_from_atoms(atoms)
            data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            data = Batch.from_data_list([data]).to(self._device)
            
            # Get the model
            model = self._models[i]
            
            # Forward pass to get energy
            with torch.no_grad():
                output = model(data)
                energy = output.energy.cpu().numpy()
            
            energies_list.append(energy)
        
        return np.array(energies_list)
    
    @property
    def device(self) -> str:
        """Get the device (cpu/cuda) used by the calculator."""
        return self._device
    
    @property
    def models(self) -> List[Any]:
        """Get the raw models in the ensemble."""
        return self._models
    
    @property
    def z_table(self) -> Any:
        """Get the atomic number mapping table."""
        return self._z_table
    
    @property
    def r_max(self) -> float:
        """Get the cutoff radius for the models."""
        return self._r_max
    
    def get_mean_forces(self, atoms: Atoms) -> np.ndarray:
        """Get mean forces across all models."""
        forces = self.forces_all(atoms)
        return np.mean(forces, axis=0)
    
    def get_mean_energy(self, atoms: Atoms) -> float:
        """Get mean energy across all models."""
        energies = self.energies_all(atoms)
        return float(np.mean(energies))
    
    def stresses_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate stresses using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate stresses for
            
        Returns:
            Stresses array of shape (n_models, 6) in Voigt notation
        """
        stresses_list = []
        
        for i, calc in enumerate(self._calculators):
            # Get parameters from the calculator
            r_max = calc.r_max.item() if hasattr(calc.r_max, 'item') else calc.r_max
            z_table = calc.z_table
            
            # Create MACE data format
            config = config_from_atoms(atoms)
            data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            data = Batch.from_data_list([data]).to(self._device)
            
            # Get the model
            model = self._models[i]
            
            # Forward pass to get stress
            with torch.no_grad():
                output = model(data)
                stress = output.stress.cpu().numpy()
            
            stresses_list.append(stress)
        
        return np.array(stresses_list)
    
    def get_mean_stress(self, atoms: Atoms) -> np.ndarray:
        """Get mean stress across all models."""
        stresses = self.stresses_all(atoms)
        return np.mean(stresses, axis=0)
    
    # ASE calculator interface methods - directly calculate properties
    def get_potential_energy(self, atoms: Atoms = None, force_consistent: bool = False) -> float:
        """ASE calculator interface: Get potential energy (mean of ensemble)."""
        if atoms is None:
            atoms = self._atoms
        if atoms is None:
            raise ValueError("No atoms provided and no atoms attached to calculator")
        
        return self.get_mean_energy(atoms)
    
    def get_forces(self, atoms: Atoms = None) -> np.ndarray:
        """ASE calculator interface: Get forces (mean of ensemble)."""
        if atoms is None:
            atoms = self._atoms
        if atoms is None:
            raise ValueError("No atoms provided and no atoms attached to calculator")
        
        return self.get_mean_forces(atoms)
    
    def get_stress(self, atoms: Atoms = None) -> np.ndarray:
        """ASE calculator interface: Get stress tensor (mean of ensemble)."""
        if atoms is None:
            atoms = self._atoms
        if atoms is None:
            raise ValueError("No atoms provided and no atoms attached to calculator")
        
        return self.get_mean_stress(atoms)
    
    def set_atoms(self, atoms: Atoms):
        """ASE calculator interface: Set atoms for the calculator."""
        self._atoms = atoms
    
    def get_atoms(self) -> Atoms:
        """ASE calculator interface: Get atoms from the calculator."""
        return self._atoms 