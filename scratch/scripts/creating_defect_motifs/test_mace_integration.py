"""1:110:scratch/scripts/creating_defect_motifs/test_mace_integration.py
#!/usr/bin/env python3
"""
Test script to verify MACE integration with adversarial attacks.
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


def test_mace_integration():
    """Test MACE integration with adversarial attacks."""
    
    print("=== Testing MACE Integration ===")
    
    # Check calculator availability
    available_backends = get_supported_backends()
    available = {backend: backend in available_backends for backend in ['mace']}
    print(f"Available calculators: {available}")
    
    if 'mace' not in available_backends:
        print("MACE not available, skipping test")
        return False
    
    # Create a simple test structure
    atoms = bulk('V', 'bcc', a=3.01).repeat((2, 2, 2))
    print(f"Test structure: {len(atoms)} atoms")
    
    # Example MACE model path (you'll need to provide a real one)
    model_paths = [
        "../../data/potentials/mace/example_mace.model"
    ]
    
    # Check if model file exists
    if not Path(model_paths[0]).exists():
        print(f"MACE model file not found at {model_paths[0]}")
        print("Please provide a valid MACE model path")
        return False
    
    try:
        # Initialize optimizer with MACE
        optimizer = GradientAdversarialOptimizer(
            model_paths=model_paths,
            device="cpu",  # Use CPU for testing
            learning_rate=0.01,
            temperature=1000,
            include_probability=False,
            debug=True,
            backend='mace'
        )
        
        print(f"Successfully initialized optimizer with {optimizer.calculator.calculator_type} calculator")
        
        # Test force variance calculation
        variance, atom_variances, mean_forces = optimizer._calculate_force_variance(atoms)
        print(f"Force variance: {variance:.6f}")
        print(f"Mean forces shape: {mean_forces.shape}")
        
        # Test energy calculation
        energy = optimizer._calculate_energy(atoms)
        print(f"Energy: {energy:.6f} eV")
        
        # Test short optimization
        print("\nRunning short optimization test...")
        trajectory = optimizer.optimize(
            atoms=atoms,
            generation=1,
            n_iterations=5,  # Very short for testing
            min_distance=1.5,
            output_dir='test_output',
            patience=3,
            shake=False,
            require_structure_id=False  # Allow use of reserved ID for new structures
        )
        
        print(f"Optimization completed. Trajectory length: {len(trajectory)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing MACE integration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mace_integration()
    if success:
        print("\n✅ MACE integration test passed!")
    else:
        print("\n❌ MACE integration test failed!")
        sys.exit(1)
""" 