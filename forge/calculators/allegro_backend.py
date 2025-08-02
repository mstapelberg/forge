"""Allegro backend for ensemble calculations."""

import numpy as np
import torch
from typing import List, Any, Union
from ase import Atoms
from pathlib import Path

from .interface import BaseEnsembleCalculator

# Conditional import for NequIP/Allegro
try:
    from nequip.ase import NequIPCalculator
    from nequip.model.inference_models import load_compiled_model
    from nequip.scripts._compile_utils import PAIR_NEQUIP_INPUTS, ASE_OUTPUTS
    from nequip.nn import graph_model
    from nequip.data import AtomicDataDict, from_ase
    NEQUIP_AVAILABLE = True
except ImportError:
    NEQUIP_AVAILABLE = False
    NequIPCalculator = None
    AtomicDataDict = None
    from_ase = None


class AllegroBackend(BaseEnsembleCalculator):
    """Allegro backend implementing BaseEnsembleCalculator interface.
    
    Uses NequIPCalculator for Allegro models since Allegro is built on NequIP.
    Supports both compiled (.pt2) and packaged (.zip) model formats.
    """
    
    def __init__(self, model_paths: Union[str, List[str]], device: str = 'cpu', **kwargs):
        """Initialize Allegro backend.
        
        Args:
            model_paths: Path(s) to Allegro model file(s) (.pt2 or .zip files)
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional arguments passed to NequIPCalculator (e.g., default_dtype)
        """
        if not NEQUIP_AVAILABLE:
            raise ImportError(
                "NequIP/Allegro is not available in this environment. "
                "Please install NequIP/Allegro or use a different backend. "
                "Install with: pip install nequip"
            )
            
        self._device = device
        self._kwargs = kwargs  # Store kwargs for passing to calculators
        self._atoms = None  # Store attached atoms for ASE interface
        
        # Store model paths for reference
        if isinstance(model_paths, str):
            self.model_paths = [model_paths]
        else:
            self.model_paths = list(model_paths)
        
        # Initialize individual NequIPCalculators for each model
        self._calculators = []
        self._models = []
        
        # Store metadata from the first model for r_max and z_table
        self._r_max = None
        self._type_names = None
        self._chemical_symbols = None
        
        try:
            for i, model_path in enumerate(self.model_paths):
                # Ensure the model file exists
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Determine model format and load appropriately
                model_path_obj = Path(model_path)
                file_extension = model_path_obj.suffix.lower()
                
                if file_extension == '.pt2':
                    # Compiled model - use from_compiled_model (no gradients)
                    print(f"[INFO] Loading compiled model (.pt2): {model_path_obj.name}")
                    calc = NequIPCalculator.from_compiled_model(
                        model_path, 
                        device=device,
                        chemical_symbols=kwargs.get('species_to_type_name', None),
                        **kwargs  # Pass kwargs like default_dtype
                    )
                    
                    # Extract metadata for compiled models
                    if i == 0:
                        # For compiled models, we need to load the model separately to get metadata
                        model, metadata = load_compiled_model(
                            model_path, 
                            device, 
                            PAIR_NEQUIP_INPUTS, 
                            ASE_OUTPUTS
                        )
                        self._r_max = float(metadata[graph_model.R_MAX_KEY])
                        self._type_names = metadata[graph_model.TYPE_NAMES_KEY]
                        if isinstance(self._type_names, str):
                            self._chemical_symbols = self._type_names.split(" ")
                        else:
                            self._chemical_symbols = self._type_names
                    
                elif file_extension == '.zip' or model_path.endswith('.nequip.zip'):
                    # Packaged model - use _from_packaged_model (preserves gradients!)
                    print(f"[INFO] Loading packaged model: {model_path_obj.name}")
                    calc = NequIPCalculator._from_packaged_model(
                        model_path,
                        device=device,
                        chemical_symbols=kwargs.get('species_to_type_name', None),
                        **kwargs  # Pass kwargs like default_dtype
                    )
                    
                    # Extract metadata for packaged models
                    if i == 0:
                        # For packaged models, metadata is available through the model
                        if hasattr(calc, 'model') and hasattr(calc.model, 'metadata'):
                            self._r_max = float(calc.model.metadata[graph_model.R_MAX_KEY])
                            self._type_names = calc.model.metadata[graph_model.TYPE_NAMES_KEY]
                            if isinstance(self._type_names, str):
                                self._chemical_symbols = self._type_names.split(" ")
                            else:
                                self._chemical_symbols = self._type_names
                        else:
                            # Cannot proceed without metadata - r_max is critical for calculations
                            raise RuntimeError(
                                f"Failed to extract metadata from packaged model: {model_path}. "
                                f"The model does not have accessible metadata. "
                                f"This is required for proper r_max and chemical symbols extraction. "
                                f"Please ensure the model file is valid and contains the necessary metadata."
                            )
                    
                else:
                    raise ValueError(f"Unsupported model file format: {file_extension}. "
                                   f"Supported formats: .pt2 (compiled), .zip/.nequip.zip (packaged)")
                
                self._calculators.append(calc)
                # Store the model for reference (but we'll use calculator interface for calculations)
                if hasattr(calc, 'model'):
                    self._models.append(calc.model)
                else:
                    # For compiled models, the model might not be directly accessible
                    self._models.append(None)
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Allegro backend: {e}")
        
        if not self._calculators:
            raise RuntimeError("No calculators were successfully initialized")
        
        print(f"Successfully initialized AllegroBackend with {len(self._calculators)} model(s)")
        
        # Initialize results dictionary for ASE compatibility
        self.results = {}
    
    def forces_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate forces using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate forces for
            
        Returns:
            Forces array of shape (n_models, n_atoms, 3)
        """
        forces_list = []
        
        # Store original calculator and temporarily remove it to avoid from_ase issues
        original_calc = atoms.calc
        atoms.calc = None
        
        try:
            for i, calc in enumerate(self._calculators):
                # Check if this is a compiled model (no gradients) or packaged model (with gradients)
                if self._models[i] is None:
                    # Compiled model - use calculator interface
                    atoms_copy = atoms.copy()
                    atoms_copy.calc = calc
                    forces = atoms_copy.get_forces()
                else:
                    # Packaged model - use raw model directly (supports gradients)
                    data = from_ase(atoms)
                    
                    # Apply transforms from the calculator
                    for transform in calc.transforms:
                        data = transform(data)
                    
                    # Move to device
                    data = AtomicDataDict.to_(data, self._device)
                    
                    # Get the model from the calculator
                    model = self._models[i]
                    
                    # Ensure model is in eval mode but parameters require gradients
                    model.eval()
                    for param in model.parameters():
                        param.requires_grad_(True)
                    
                    # Forward pass to get forces (allow gradients for MCMC optimization)
                    output = model(data)
                    forces = output[AtomicDataDict.FORCE_KEY].cpu().detach().numpy()
                
                forces_list.append(forces)
        finally:
            # Restore original calculator
            atoms.calc = original_calc
        
        return np.array(forces_list)
    
    def energies_all(self, atoms: Atoms) -> np.ndarray:
        """Calculate energies using all models in the ensemble.
        
        Args:
            atoms: ASE Atoms object to calculate energies for
            
        Returns:
            Energies array of shape (n_models,)
        """
        energies_list = []
        
        # Store original calculator and temporarily remove it to avoid from_ase issues
        original_calc = atoms.calc
        atoms.calc = None
        
        try:
            for i, calc in enumerate(self._calculators):
                # Check if this is a compiled model (no gradients) or packaged model (with gradients)
                if self._models[i] is None:
                    # Compiled model - use calculator interface
                    atoms_copy = atoms.copy()
                    atoms_copy.calc = calc
                    energy = atoms_copy.get_potential_energy()
                else:
                    # Packaged model - use raw model directly (supports gradients)
                    data = from_ase(atoms)
                    
                    # Apply transforms from the calculator
                    for transform in calc.transforms:
                        data = transform(data)
                    
                    # Move to device
                    data = AtomicDataDict.to_(data, self._device)
                    
                    # Get the model from the calculator
                    model = self._models[i]
                    
                    # Ensure model is in eval mode but parameters require gradients
                    model.eval()
                    for param in model.parameters():
                        param.requires_grad_(True)
                    
                    # Forward pass to get energy (allow gradients for MCMC optimization)
                    output = model(data)
                    energy = output[AtomicDataDict.TOTAL_ENERGY_KEY].cpu().detach().numpy()
                
                energies_list.append(energy)
        finally:
            # Restore original calculator
            atoms.calc = original_calc
        
        return np.array(energies_list)
    
    @property
    def device(self) -> str:
        """Get the device (cpu/cuda) used by the calculator."""
        return self._device
    
    @property
    def models(self) -> List[Any]:
        """Get the raw models in the ensemble."""
        # Filter out None values (compiled models where raw model is not accessible)
        return [model for model in self._models if model is not None]
    
    @property
    def z_table(self) -> Any:
        """Get the atomic number mapping table."""
        if self._calculators:
            return self._calculators[0].z_table
        return None
    
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
        
        # Store original calculator and temporarily remove it to avoid from_ase issues
        original_calc = atoms.calc
        atoms.calc = None
        
        try:
            for i, calc in enumerate(self._calculators):
                # Check if this is a compiled model (no gradients) or packaged model (with gradients)
                if self._models[i] is None:
                    # Compiled model - use calculator interface
                    atoms_copy = atoms.copy()
                    atoms_copy.calc = calc
                    stress = atoms_copy.get_stress()
                else:
                    # Packaged model - use raw model directly (supports gradients)
                    data = from_ase(atoms)
                    
                    # Apply transforms from the calculator
                    for transform in calc.transforms:
                        data = transform(data)
                    
                    # Move to device
                    data = AtomicDataDict.to_(data, self._device)
                    
                    # Get the model from the calculator
                    model = self._models[i]
                    
                    # Ensure model is in eval mode but parameters require gradients
                    model.eval()
                    for param in model.parameters():
                        param.requires_grad_(True)
                    
                    # Forward pass to get stress (allow gradients for MCMC optimization)
                    output = model(data)
                    
                    # Check if stress is available in the output
                    if AtomicDataDict.STRESS_KEY in output:
                        stress = output[AtomicDataDict.STRESS_KEY].cpu().detach().numpy()
                        
                        # Ensure stress is in Voigt format (6 components)
                        if stress.shape == (3, 3):
                            # Convert 3x3 tensor to Voigt notation
                            from ase.stress import full_3x3_to_voigt_6_stress
                            stress = full_3x3_to_voigt_6_stress(stress)
                        elif stress.shape == (9,):
                            # Convert flat 9-component to Voigt notation
                            stress_3x3 = stress.reshape(3, 3)
                            from ase.stress import full_3x3_to_voigt_6_stress
                            stress = full_3x3_to_voigt_6_stress(stress_3x3)
                        elif stress.shape != (6,):
                            # If it's not 6 components, we need to handle this case
                            print(f"Warning: Unexpected stress shape: {stress.shape}")
                            # Create a zero stress tensor as fallback
                            stress = np.zeros(6)
                    else:
                        # Model doesn't provide stress, use zero stress
                        print("Warning: Model does not provide stress tensor. Using zero stress.")
                        stress = np.zeros(6)
                
                stresses_list.append(stress)
        finally:
            # Restore original calculator
            atoms.calc = original_calc
        
        return np.array(stresses_list)
    
    def get_mean_stress(self, atoms: Atoms) -> np.ndarray:
        """Get mean stress across all models."""
        stresses = self.stresses_all(atoms)
        mean_stress = np.mean(stresses, axis=0)
        
        # Ensure the final stress tensor has the correct shape for ASE
        if mean_stress.shape != (6,):
            print(f"Warning: Mean stress shape is {mean_stress.shape}, expected (6,). Using zero stress.")
            mean_stress = np.zeros(6)
        
        return mean_stress
    
    # ASE calculator interface methods - directly calculate properties
    def get_potential_energy(self, atoms: Atoms = None, force_consistent: bool = False) -> float:
        """ASE calculator interface: Get potential energy (mean of ensemble)."""
        if atoms is None:
            atoms = self._atoms
        if atoms is None:
            raise ValueError("No atoms provided and no atoms attached to calculator")
        
        energy = self.get_mean_energy(atoms)
        self.results['energy'] = energy
        return energy
    
    def get_forces(self, atoms: Atoms = None) -> np.ndarray:
        """ASE calculator interface: Get forces (mean of ensemble)."""
        if atoms is None:
            atoms = self._atoms
        if atoms is None:
            raise ValueError("No atoms provided and no atoms attached to calculator")
        
        forces = self.get_mean_forces(atoms)
        self.results['forces'] = forces
        return forces
    
    def get_stress(self, atoms: Atoms = None) -> np.ndarray:
        """ASE calculator interface: Get stress tensor (mean of ensemble)."""
        if atoms is None:
            atoms = self._atoms
        if atoms is None:
            raise ValueError("No atoms provided and no atoms attached to calculator")
        
        stress = self.get_mean_stress(atoms)
        self.results['stress'] = stress
        return stress
    
    def set_atoms(self, atoms: Atoms):
        """ASE calculator interface: Set atoms for the calculator."""
        self._atoms = atoms
    
    def get_atoms(self) -> Atoms:
        """ASE calculator interface: Get atoms from the calculator."""
        return self._atoms 