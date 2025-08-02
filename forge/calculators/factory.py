"""Factory for creating appropriate calculator backends."""

from typing import Union, List
from pathlib import Path

from .interface import BaseEnsembleCalculator

# Conditional imports for backends
try:
    from .mace_backend import MACEBackend, MACE_AVAILABLE
except ImportError:
    MACEBackend = None
    MACE_AVAILABLE = False

try:
    from .allegro_backend import AllegroBackend, NEQUIP_AVAILABLE
except ImportError:
    AllegroBackend = None
    NEQUIP_AVAILABLE = False


def create_ensemble_calculator(
    model_paths: Union[str, List[str]], 
    backend: str = 'auto',
    device: str = 'cpu', 
    **kwargs
) -> BaseEnsembleCalculator:
    """Create an appropriate ensemble calculator backend.
    
    Args:
        model_paths: Path(s) to model file(s)
        backend: Backend type ('mace', 'allegro', or 'auto' for auto-detection)
        device: Device to use ('cpu' or 'cuda')
        **kwargs: Additional arguments passed to the backend constructor
        
    Returns:
        BaseEnsembleCalculator: Appropriate backend instance
        
    Raises:
        ValueError: If backend type is unknown or cannot be auto-detected
        FileNotFoundError: If model files don't exist
    """
    # Ensure model_paths is a list
    if isinstance(model_paths, str):
        paths_list = [model_paths]
    else:
        paths_list = list(model_paths)
    
    # Check that all model files exist
    for path in paths_list:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
    
    # Auto-detect backend if needed
    if backend == 'auto' or backend is None:
        backend = _detect_backend(paths_list)
    
    # Create appropriate backend
    if backend.lower() == 'mace':
        if not MACE_AVAILABLE:
            raise ImportError(
                "MACE backend requested but MACE is not available in this environment. "
                "Please install MACE with: pip install mace-torch"
            )
        return MACEBackend(model_paths, device=device, **kwargs)
    elif backend.lower() in ['allegro', 'nequip']:
        if not NEQUIP_AVAILABLE:
            raise ImportError(
                "Allegro/NequIP backend requested but NequIP is not available in this environment. "
                "Please install NequIP with: pip install nequip"
            )
        return AllegroBackend(model_paths, device=device, **kwargs)
    else:
        available_backends = get_supported_backends()
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available backends in this environment: {available_backends}"
        )


def _detect_backend(model_paths: List[str]) -> str:
    """Auto-detect backend type from model file extensions.
    
    Args:
        model_paths: List of model file paths
        
    Returns:
        str: Detected backend type ('mace' or 'allegro')
        
    Raises:
        ValueError: If backend cannot be determined
    """
    extensions = set()
    for path in model_paths:
        ext = Path(path).suffix.lower()
        extensions.add(ext)
    
    # Check for definitive Allegro/NequIP extensions first
    allegro_definitive = {'.pt2', '.zip'}  # .pt2 is compiled, .zip is packaged NequIP/Allegro
    if any(ext in allegro_definitive for ext in extensions):
        if NEQUIP_AVAILABLE:
            return 'allegro'
        else:
            raise ImportError(
                "Detected Allegro/NequIP model files (.pt2/.zip) but NequIP is not available. "
                "Please install NequIP with: pip install nequip"
            )
    
    # Check for definitive MACE extensions
    mace_definitive = {'.model'}  # .model is typically MACE
    if any(ext in mace_definitive for ext in extensions):
        if MACE_AVAILABLE:
            return 'mace'
        else:
            raise ImportError(
                "Detected MACE model files (.model) but MACE is not available. "
                "Please install MACE with: pip install mace-torch"
            )
    
    # Handle ambiguous .pth files - could be either MACE or NequIP
    pth_extensions = {'.pth'}
    if any(ext in pth_extensions for ext in extensions):
        # Look at file names for hints
        for path in model_paths:
            path_lower = Path(path).name.lower()
            if 'allegro' in path_lower or 'nequip' in path_lower:
                if NEQUIP_AVAILABLE:
                    return 'allegro'
                else:
                    raise ImportError(
                        "Detected Allegro/NequIP model files but NequIP is not available. "
                        "Please install NequIP with: pip install nequip"
                    )
            elif 'mace' in path_lower:
                if MACE_AVAILABLE:
                    return 'mace'
                else:
                    raise ImportError(
                        "Detected MACE model files but MACE is not available. "
                        "Please install MACE with: pip install mace-torch"
                    )
        
        # If no filename hints, prefer MACE for .pth files
        if MACE_AVAILABLE:
            print(f"[INFO] Detected .pth files without clear naming hints. Defaulting to MACE backend.")
            return 'mace'
        elif NEQUIP_AVAILABLE:
            print(f"[INFO] Detected .pth files without clear naming hints. MACE not available, trying Allegro backend.")
            return 'allegro'
    
    # If no backends are available, provide helpful error
    if not MACE_AVAILABLE and not NEQUIP_AVAILABLE:
        raise ImportError(
            "No calculator backends are available in this environment. "
            "Please install either MACE (pip install mace-torch) or NequIP (pip install nequip)."
        )
    
    # If we can't determine from extensions, raise an error rather than guessing
    available_backends = get_supported_backends()
    raise ValueError(
        f"Could not auto-detect backend from file paths: {model_paths}. "
        f"File extensions found: {extensions}. "
        f"Available backends: {available_backends}. "
        f"Please specify the backend explicitly using the 'backend' parameter."
    )


def get_supported_backends() -> List[str]:
    """Get list of available backend names in this environment.
    
    Returns:
        List[str]: List of available backend names
    """
    available = []
    if MACE_AVAILABLE:
        available.append('mace')
    if NEQUIP_AVAILABLE:
        available.extend(['allegro', 'nequip'])
    return available 