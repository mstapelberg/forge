"""
Unified calculator interface for MACE and Allegro potentials.

This module provides a common interface for both MACE and Allegro calculators,
automatically detecting which one is available and providing consistent API.
"""

import warnings
from typing import Union, List, Dict, Optional, Any
import torch
import warnings
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

# Deprecation warning
warnings.warn(
    "forge.workflows.calculator_interface is deprecated and will be removed in a future version. "
    "Please use forge.calculators.factory instead. "
    "The new factory provides better backend support and improved functionality. "
    "Migration guide: "
    "- Replace 'from forge.workflows.calculator_interface import create_calculator' "
    "- With 'from forge.calculators.factory import create_ensemble_calculator' "
    "- Replace 'create_calculator()' with 'create_ensemble_calculator()'",
    DeprecationWarning,
    stacklevel=2
)

# Try to import MACE
try:
    from mace.calculators.mace import MACECalculator
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    MACECalculator = None

# Try to import Allegro/NequIP
try:
    from nequip.ase import NequIPCalculator
    ALLEGRO_AVAILABLE = True
except ImportError:
    ALLEGRO_AVAILABLE = False
    NequIPCalculator = None


class UnifiedCalculator:
    """
    Unified calculator interface that works with both MACE and Allegro.
    
    DEPRECATED: This class is deprecated. Use forge.calculators.factory.create_ensemble_calculator instead.
    
    This class provides a consistent API regardless of which calculator
    is being used underneath.
    """
    
    def __init__(
        self,
        model_path: Union[str, List[str]],
        calculator_type: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_dtype: str = "float32",
        use_cueq: bool = False,
        species_to_type_name: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        """
        Initialize the unified calculator.
        
        DEPRECATED: Use forge.calculators.factory.create_ensemble_calculator instead.
        
        Args:
            model_path: Path(s) to model file(s)
            calculator_type: Type of calculator ('mace', 'allegro', or None for auto-detect)
            device: Device to run calculations on
            default_dtype: Default data type for calculations
            use_cueq: Whether to use CUEQ (MACE only)
            species_to_type_name: Species mapping for Allegro
            **kwargs: Additional arguments passed to the underlying calculator
        """
        warnings.warn(
            "UnifiedCalculator is deprecated. Use forge.calculators.factory.create_ensemble_calculator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.model_path = model_path if isinstance(model_path, list) else [model_path]
        self.device = device
        self.default_dtype = default_dtype
        self.use_cueq = use_cueq
        self.species_to_type_name = species_to_type_name or {}
        self.kwargs = kwargs
        
        # Auto-detect calculator type if not specified
        if calculator_type is None:
            calculator_type = self._auto_detect_calculator_type()
        
        self.calculator_type = calculator_type
        self.calculator = self._create_calculator()
        
    def _auto_detect_calculator_type(self) -> str:
        """Auto-detect which calculator type to use based on available modules."""
        if MACE_AVAILABLE and ALLEGRO_AVAILABLE:
            # Both available, check model file extension
            if any(path.endswith('.model') for path in self.model_path):
                return 'mace'
            elif any(path.endswith('.nequip.zip') or path.endswith('.pt2') for path in self.model_path):
                return 'allegro'
            else:
                # Default to MACE if both available
                return 'mace'
        elif MACE_AVAILABLE:
            return 'mace'
        elif ALLEGRO_AVAILABLE:
            return 'allegro'
        else:
            raise ImportError("Neither MACE nor Allegro/NequIP is available. "
                            "Please install one of them.")
    
    def _create_calculator(self) -> Calculator:
        """Create the underlying calculator instance."""
        if self.calculator_type == 'mace':
            if not MACE_AVAILABLE:
                raise ImportError("MACE is not available. Please install mace.")
            
            return MACECalculator(
                model_paths=self.model_path,
                device=self.device,
                default_dtype=self.default_dtype,
                use_cueq=self.use_cueq,
                **self.kwargs
            )
            
        elif self.calculator_type == 'allegro':
            if not ALLEGRO_AVAILABLE:
                raise ImportError("Allegro/NequIP is not available. Please install nequip.")
            
            # For Allegro, we need to handle single model path
            if len(self.model_path) > 1:
                warnings.warn("Allegro calculator only supports single model. Using first model.")
            
            # Handle different Allegro model file formats
            model_path = self.model_path[0]
            
            if model_path.endswith('.zip') or model_path.endswith('.nequip.zip'):
                # Packaged model format
                return NequIPCalculator._from_packaged_model(
                    package_path=model_path,
                    chemical_symbols=self.species_to_type_name,  # Use chemical_symbols for NequIP
                    device=self.device,
                    **self.kwargs
                )
            elif model_path.endswith('.pt2'):
                # Compiled model format
                return NequIPCalculator.from_compiled_model(
                    compile_path=model_path,
                    chemical_symbols=self.species_to_type_name,  # Use chemical_symbols for NequIP
                    device=self.device,
                    **self.kwargs
                )
            else:
                raise ValueError(f"Unknown model file extension: {model_path}. "
                               f"Supported formats: .zip, .nequip.zip, .pt2")
            
        else:
            raise ValueError(f"Unknown calculator type: {self.calculator_type}")
    
    def calculate(self, atoms: Atoms) -> Dict[str, Any]:
        """
        Calculate energy and forces for a structure.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Dictionary containing calculation results
        """
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        return {
            'energy': energy,
            'forces': forces,
            'atoms': atoms
        }
    
    def get_potential_energy(self, atoms: Atoms) -> float:
        """Get potential energy for a structure."""
        atoms.calc = self.calculator
        return atoms.get_potential_energy()
    
    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """Get forces for a structure."""
        atoms.calc = self.calculator
        return atoms.get_forces()
    
    def get_stress(self, atoms: Atoms) -> np.ndarray:
        """Get stress tensor for a structure."""
        atoms.calc = self.calculator
        return atoms.get_stress()
    
    def __call__(self, atoms: Atoms) -> Dict[str, Any]:
        """Make the calculator callable."""
        return self.calculate(atoms)


def create_calculator(
    model_path: Union[str, List[str]],
    calculator_type: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> UnifiedCalculator:
    """
    Factory function to create a unified calculator.
    
    DEPRECATED: Use forge.calculators.factory.create_ensemble_calculator instead.
    
    Args:
        model_path: Path(s) to model file(s)
        calculator_type: Type of calculator ('mace', 'allegro', or None for auto-detect)
        device: Device to run calculations on
        **kwargs: Additional arguments for the calculator
        
    Returns:
        UnifiedCalculator instance
    """
    warnings.warn(
        "create_calculator is deprecated. Use forge.calculators.factory.create_ensemble_calculator instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return UnifiedCalculator(
        model_path=model_path,
        calculator_type=calculator_type,
        device=device,
        **kwargs
    )


def get_available_calculators() -> List[str]:
    """Get list of available calculator types."""
    warnings.warn(
        "get_available_calculators is deprecated. Use forge.calculators.factory.get_supported_backends instead.",
        DeprecationWarning,
        stacklevel=2
    )
    available = []
    if MACE_AVAILABLE:
        available.append('mace')
    if ALLEGRO_AVAILABLE:
        available.append('allegro')
    return available


def check_calculator_availability() -> Dict[str, bool]:
    """Check which calculators are available."""
    warnings.warn(
        "check_calculator_availability is deprecated. Use forge.calculators.factory.get_supported_backends instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return {
        'mace': MACE_AVAILABLE,
        'allegro': ALLEGRO_AVAILABLE
    } 