from pathlib import Path
from typing import List, Optional
import os

class TestResources:
    """Utility class for managing test resources."""
    
    @classmethod
    def get_resource_dir(cls) -> Path:
        """Get the base resource directory."""
        return Path(__file__).parent

    @classmethod
    def get_test_potentials(cls, potential_type: str = 'mace', potential_suffix: str = '*.model') -> List[str]:
        """
        Get paths to test potential files.
        
        Args:
            potential_type: Type of potential ('mace', 'allegro', etc.)
            
        Returns:
            List of paths to test potential files
        """
        potentials_dir = cls.get_resource_dir() / 'potentials' / potential_type
        if not potentials_dir.exists():
            raise ValueError(f"No test potentials found for type: {potential_type}")
            
        return sorted(str(p) for p in potentials_dir.glob(potential_suffix))

    @classmethod
    def get_test_structure(cls, name: str) -> str:
        """
        Get path to test structure file.
        
        Args:
            name: Name of structure file (e.g., 'test_bulk.xyz')
            
        Returns:
            Path to structure file
        """
        structure_path = cls.get_resource_dir() / 'structures' / name
        if not structure_path.exists():
            raise ValueError(f"Test structure not found: {name}")
            
        return str(structure_path)