#!/usr/bin/env python
"""Test script to verify Allegro backend fixes."""

import sys
from pathlib import Path
import numpy as np

# Add the forge package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_allegro_backend():
    """Test the fixed Allegro backend."""
    print("=== Testing Fixed Allegro Backend ===")
    
    try:
        from forge.calculators import create_ensemble_calculator
        from ase import Atoms
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Find Allegro model files
    model_dir = Path(__file__).parent.parent / "scratch" / "data" / "potentials" / "compiled_gen-8-exploit"
    allegro_models = list(model_dir.glob("*.pt2"))
    
    if not allegro_models:
        print("No Allegro models found")
        return
    
    print(f"Found {len(allegro_models)} Allegro models")
    model_paths = [str(p) for p in allegro_models[:3]]  # Use first 3
    print(f"Using models: {[Path(p).name for p in model_paths]}")
    
    try:
        # Test auto-detection
        print("\n--- Testing Auto-Detection ---")
        calc = create_ensemble_calculator(
            model_paths=model_paths,
            backend='auto',  # Should detect Allegro
            device='cuda'
        )
        print(f"✅ Auto-detection successful: {type(calc).__name__}")
        
        # Test metadata extraction
        print("\n--- Testing Metadata Extraction ---")
        print(f"   r_max: {calc.r_max}")
        print(f"   z_table: {calc.z_table}")
        print(f"   Device: {calc.device}")
        print(f"   Number of models: {len(calc.models)}")
        
        # Create a test structure
        print("\n--- Testing Calculations ---")
        # Simple diamond structure for testing
        from ase.build import bulk
        atoms = bulk('V', 'bcc', a=3.01, cubic=True) * (4,4,4)
        print(f"Test structure: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
        
        # Test force calculation
        print("   Testing force calculation...")
        forces = calc.forces_all(atoms)
        print(f"   ✅ Force calculation successful")
        print(f"      Shape: {forces.shape}")
        print(f"      Mean magnitude: {np.mean(np.linalg.norm(forces, axis=2)):.6f}")
        
        # Test variance calculation
        print("   Testing variance calculation...")
        variance = calc.calculate_normalized_force_variance(forces)
        print(f"   ✅ Variance calculation successful")
        print(f"      Mean variance: {np.mean(variance):.6f}")
        
        # Test energy calculation
        print("   Testing energy calculation...")
        energies = calc.energies_all(atoms)
        print(f"   ✅ Energy calculation successful")
        print(f"      Shape: {energies.shape}")
        print(f"      Mean energy: {np.mean(energies):.6f} eV")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mace_detection():
    """Test that MACE files are properly detected (and fail gracefully if MACE not available)."""
    print("\n=== Testing MACE Detection ===")
    
    try:
        from forge.calculators import create_ensemble_calculator
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Find MACE model files
    model_dir = Path(__file__).parent.parent / "scratch" / "potentials" / "mace_gen_7_ensemble"
    mace_models = list(model_dir.glob("*.model"))
    
    if not mace_models:
        print("No MACE models found for testing")
        return
    
    print(f"Found {len(mace_models)} MACE models")
    model_paths = [str(p) for p in mace_models[:3]]  # Use first 3
    print(f"Using models: {[Path(p).name for p in model_paths]}")
    
    try:
        calc = create_ensemble_calculator(
            model_paths=model_paths,
            backend='auto',  # Should detect MACE or fail gracefully
            device='cuda'
        )
        print(f"✅ MACE detection successful: {type(calc).__name__}")
        
    except ImportError as e:
        print(f"✅ MACE not available, detected correctly: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    success = test_allegro_backend()
    test_mace_detection()
    
    if success:
        print("\n🎉 All Allegro fixes verified successfully!")
    else:
        print("\n❌ Some tests failed.") 