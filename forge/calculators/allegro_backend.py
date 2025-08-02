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
    NEQUIP_AVAILABLE = True
except ImportError:
    NEQUIP_AVAILABLE = False
    NequIPCalculator = None


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
                    
                elif file_extension == '.zip':
                    # Packaged model - use _from_packaged_model (preserves gradients!)
                    print(f"[INFO] Loading packaged model (.zip): {model_path_obj.name}")
                    
                    calc = NequIPCalculator._from_packaged_model(
                        model_path,
                        device=device,
                        chemical_symbols=kwargs.get('species_to_type_name', None)
                        # Note: _from_packaged_model may not support all kwargs, so we keep it simple
                    )
                    
                    # IMPORTANT: Convert model to float32 to avoid dtype mismatches
                    if hasattr(calc.model, 'float'):
                        calc.model = calc.model.float()  # Convert to float32
                    
                    # Extract metadata for packaged models
                    if i == 0:
                        # For packaged models, metadata is available through the model
                        self._r_max = float(calc.model.metadata[graph_model.R_MAX_KEY])
                        self._type_names = calc.model.metadata[graph_model.TYPE_NAMES_KEY]
                        if isinstance(self._type_names, str):
                            self._chemical_symbols = self._type_names.split(" ")
                        else:
                            self._chemical_symbols = self._type_names
                
                else:
                    raise ValueError(f"Unsupported model file format: {file_extension}. Supported formats: .pt2 (compiled), .zip (packaged)")
                
                self._calculators.append(calc)
                
                # Store reference to the underlying model for accessing properties
                # The actual model is stored in calc.model
                self._models.append(calc.model)
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize NequIPCalculator(s): {e}")
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
        """Get the atomic number mapping table.
        
        For NequIP/Allegro models, this returns the chemical symbols from metadata.
        """
        if self._chemical_symbols is not None:
            return self._chemical_symbols
        elif self._calculators:
            # Fallback to getting from calculator
            try:
                calc = self._calculators[0]
                if hasattr(calc, 'chemical_symbols'):
                    return calc.chemical_symbols
                # Try to get from transforms
                for transform in calc.transforms:
                    if hasattr(transform, 'chemical_symbols'):
                        return transform.chemical_symbols
                return None
            except Exception as e:
                print(f"[WARN] Failed to get z_table from NequIP calculator: {e}")
                return None
        else:
            raise RuntimeError("No calculators loaded")
    
    @property
    def r_max(self) -> float:
        """Get the cutoff radius for the models."""
        if self._r_max is not None:
            return self._r_max
        else:
            print("[WARN] r_max not available from metadata, using default 5.0")
            return 5.0
    
    def get_mean_forces(self, atoms: Atoms) -> np.ndarray:
        """Get mean forces across ensemble (convenience method)."""
        all_forces = self.forces_all(atoms)
        return np.mean(all_forces, axis=0)
    
    def get_mean_energy(self, atoms: Atoms) -> float:
        """Get mean energy across ensemble (convenience method)."""
        all_energies = self.energies_all(atoms)
        return float(np.mean(all_energies)) 