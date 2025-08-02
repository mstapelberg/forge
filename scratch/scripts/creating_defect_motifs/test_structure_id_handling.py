#!/usr/bin/env python3
"""
Test script to verify structure_id handling in adversarial attacks.
"""

import sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import bulk

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.core.adversarial_attack import GradientAdversarialOptimizer
from forge.calculators.factory import get_supported_backends


def test_structure_id_handling():
    """Test structure_id handling with different scenarios."""
    
    print("=== Testing Structure ID Handling ===")
    
    # Check calculator availability
    available_backends = get_supported_backends()
    available = {backend: backend in available_backends for backend in ['allegro']}
    print(f"Available calculators: {available}")
    
    if 'allegro' not in available_backends:
        print("Allegro not available, skipping test")
        return False
    
    # Create test structures
    atoms_with_id = bulk('V', 'bcc', a=3.01).repeat((2, 2, 2))
    atoms_with_id.info['structure_id'] = 99999998
    atoms_with_id.info['config_type'] = 'test_structure'
    
    atoms_without_id = bulk('V', 'bcc', a=3.01).repeat((2, 2, 2))
    # No structure_id in info
    
    # Example Allegro model path
    model_paths = [
        "/home/myless/Packages/forge/scratch/data/potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.zip",
        "/home/myless/Packages/forge/scratch/data/potentials/allegro/exploit_rmax5.50_lmax2_layers2_mlp384_seed42.nequip.zip",
        "/home/myless/Packages/forge/scratch/data/potentials/allegro/exploit_rmax5.75_lmax2_layers2_mlp384_seed42.nequip.zip"    
        ]
    
    # Check if model file exists
    if not Path(model_paths[0]).exists():
        print(f"Allegro model file not found at {model_paths[0]}")
        print("Please provide a valid Allegro model path")
        return False
    
    # Define chemical symbols mapping
    chemical_symbols = {
        'Ti': 'Ti',
        'V': 'V', 
        'Cr': 'Cr',
        'Zr': 'Zr',
        'W': 'W'
    }
    
    try:
        # Initialize optimizer
        optimizer = GradientAdversarialOptimizer(
            model_paths=model_paths,
            device="cpu",
            learning_rate=0.01,
            temperature=1000,
            include_probability=False,
            debug=True,
            backend='allegro',  # Use backend instead of calculator_type
            species_to_type_name=chemical_symbols
        )
        
        print("✅ Successfully initialized optimizer")
        
        # Test 1: Structure with ID (should work normally)
        print("\n--- Test 1: Structure with structure_id ---")
        try:
            trajectory = optimizer.optimize(
                atoms=atoms_with_id,
                generation=1,
                n_iterations=3,
                min_distance=1.5,
                output_dir='test_output',
                patience=2,
                shake=False,
                require_structure_id=False
            )
            print(f"✅ Structure with ID worked: {len(trajectory)} trajectory steps")
        except Exception as e:
            print(f"❌ Structure with ID failed: {e}")
            return False
        
        # Test 2: Structure without ID, require_structure_id=False (should use reserved ID)
        print("\n--- Test 2: Structure without ID, require_structure_id=False ---")
        try:
            trajectory = optimizer.optimize(
                atoms=atoms_without_id,
                generation=1,
                n_iterations=3,
                min_distance=1.5,
                output_dir='test_output',
                patience=2,
                shake=False,
                require_structure_id=False
            )
            print(f"✅ Structure without ID worked with reserved ID: {len(trajectory)} trajectory steps")
        except Exception as e:
            print(f"❌ Structure without ID failed: {e}")
            return False
        
        # Test 3: Structure without ID, require_structure_id=True (should fail)
        print("\n--- Test 3: Structure without ID, require_structure_id=True ---")
        try:
            trajectory = optimizer.optimize(
                atoms=atoms_without_id,
                generation=1,
                n_iterations=3,
                min_distance=1.5,
                output_dir='test_output',
                patience=2,
                shake=False,
                require_structure_id=True
            )
            print(f"❌ Structure without ID should have failed but didn't")
            return False
        except ValueError as e:
            if "structure_id" in str(e):
                print(f"✅ Structure without ID correctly failed with require_structure_id=True: {e}")
            else:
                print(f"❌ Unexpected error: {e}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error type: {e}")
            return False
        
        print("\n✅ All structure_id handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in structure_id handling test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_structure_id_handling()
    if success:
        print("\n✅ Structure ID handling test passed!")
    else:
        print("\n❌ Structure ID handling test failed!")
        sys.exit(1) 