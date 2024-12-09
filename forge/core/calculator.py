from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
from ase import Atoms

class MLCalculator(ABC):
    """Abstract base class for machine learning calculators.
    
    Provides interface for single model and ensemble calculations of energies,
    forces, and uncertainties.
    """
    
    def __init__(self, model_paths: Union[str, List[str]], device: str = 'cuda'):
        """Initialize calculator with model path(s).
        
        Args:
            model_paths: Path(s) to model file(s). If list, enables ensemble.
            device: Device to run calculations on ('cpu' or 'cuda')
        """
        self.device = device
        if isinstance(model_paths, str):
            self.model_paths = [model_paths]
            self.is_ensemble = False
        else:
            self.model_paths = model_paths
            self.is_ensemble = True
            
    @abstractmethod
    def calculate(self, atoms: Atoms) -> Dict[str, np.ndarray]:
        """Calculate energy and forces for a structure.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Dictionary containing:
                - 'energy': Potential energy (float)
                - 'forces': Forces array (N x 3)
                - 'stress': Stress tensor (optional)
        """
        pass
    
    @abstractmethod
    def calculate_ensemble(self, atoms: Atoms) -> Dict[str, List[np.ndarray]]:
        """Calculate ensemble predictions for a structure.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Dictionary containing:
                - 'energies': List of ensemble energies
                - 'forces': List of ensemble forces arrays
                - 'stress': List of ensemble stress tensors (optional)
        """
        pass
    
    def get_uncertainty(self, predictions: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """Calculate uncertainties from ensemble predictions.
        
        Args:
            predictions: Output from calculate_ensemble()
            
        Returns:
            Dictionary containing standard deviations for:
                - 'energy': Energy uncertainty
                - 'forces': Per-atom force uncertainties
                - 'stress': Stress tensor uncertainties (if available)
        """
        uncertainties = {}
        
        if 'energies' in predictions:
            uncertainties['energy'] = np.std(predictions['energies'])
            
        if 'forces' in predictions:
            forces = np.array(predictions['forces'])
            uncertainties['forces'] = np.std(forces, axis=0)
            
        if 'stress' in predictions:
            stress = np.array(predictions['stress'])
            uncertainties['stress'] = np.std(stress, axis=0)
            
        return uncertainties
    
    @abstractmethod
    def get_model_versions(self) -> List[str]:
        """Get version information for loaded models.
        
        Returns:
            List of version strings for each model
        """
        pass