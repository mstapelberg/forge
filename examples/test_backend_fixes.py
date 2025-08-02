#!/usr/bin/env python
"""Test script to verify backend fixes work correctly."""

import numpy as np
from ase import build
from pathlib import Path
import sys
import traceback

# Add the forge package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from forge.calculators import create_ensemble_calculator
except ImportError as e:
    print(f"Error importing forge: {e}")
    sys.exit(1)

def create_test_structure():
    """Create a simple test structure."""
    atoms = build.bulk('V', 'bcc', a=3.01, cubic=True)
    atoms = atoms * (4, 4, 4)  # Small structure for testing
    return atoms

def test_backend(backend_name, model_paths, test_structure):
    """Test a specific backend."""
    print(f"\n=== Testing {backend_name.upper()} Backend ===")
    
    try:
        # Create calculator
        calc = create_ensemble_calculator(
            model_paths=model_paths,
            backend=backend_name,
            device='cuda',
            default_dtype='float32'
        )
        
        print(f"✅ Successfully created {type(calc).__name__}")
        print(f"   Number of models: {len(calc.models)}")
        print(f"   Device: {calc.device}")
        
        # Test property access
        try:
            r_max = calc.r_max
            print(f"   r_max: {r_max}")
        except Exception as e:
            print(f"   ⚠️ r_max access failed: {e}")
        
        try:
            z_table = calc.z_table
            print(f"   z_table: {type(z_table).__name__ if z_table is not None else 'None'}")
        except Exception as e:
            print(f"   ⚠️ z_table access failed: {e}")
        
        # Test force calculation
        try:
            print(f"   Testing force calculation...")
            forces = calc.forces_all(test_structure)
            print(f"   ✅ Force calculation successful")
            print(f"      Shape: {forces.shape}")
            print(f"      Mean magnitude: {np.mean(np.linalg.norm(forces.reshape(-1, 3), axis=1)):.6f}")
            
            # Test variance calculation
            variances = calc.calculate_normalized_force_variance(forces)
            print(f"   ✅ Variance calculation successful")
            print(f"      Mean variance: {np.mean(variances):.6f}")
            
        except Exception as e:
            print(f"   ❌ Force calculation failed: {e}")
            traceback.print_exc()
        
        # Test energy calculation
        try:
            print(f"   Testing energy calculation...")
            energies = calc.energies_all(test_structure)
            print(f"   ✅ Energy calculation successful")
            print(f"      Shape: {energies.shape}")
            print(f"      Mean energy: {np.mean(energies):.6f} eV")
            
        except Exception as e:
            print(f"   ❌ Energy calculation failed: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"   ❌ Backend initialization failed: {e}")
        traceback.print_exc()

def main():
    """Main test function."""
    print("=== Backend Fix Verification ===")
    
    # Create test structure
    test_structure = create_test_structure()
    print(f"Created test structure: {test_structure.get_chemical_formula()} ({len(test_structure)} atoms)")
    
    # Find model files
    mace_model_dir = Path(__file__).parent.parent / "scratch" / "potentials" / "mace_gen_7_ensemble"
    allegro_model_dir = Path(__file__).parent.parent / "scratch" / "data" / "potentials" / "compiled_gen-8-exploit"
    
    # Look for MACE models
    mace_models = list(mace_model_dir.glob("*.model"))
    if mace_models:
        print(f"\nFound {len(mace_models)} MACE model file(s)")
        test_backend('mace', mace_models[:3], test_structure)  # Test with up to 3 models
    else:
        print("\n⚠️ No MACE models found in potentials")
    
    # Look for Allegro models
    allegro_models = list(allegro_model_dir.glob("*.pt2"))
    if allegro_models:
        print(f"\nFound {len(allegro_models)} Allegro model file(s)")
        test_backend('allegro', allegro_models[:3], test_structure)  # Test with up to 3 models
    else:
        print("\n⚠️ No Allegro models found in data/potentials")
    
    # Test auto-detection
    all_models = mace_models + allegro_models
    if all_models:
        print(f"\n=== Testing Auto-Detection ===")
        try:
            calc = create_ensemble_calculator(
                model_paths=all_models[:3],  # Mix of models
                backend='auto',
                device='cuda'
            )
            print(f"✅ Auto-detection successful: {type(calc).__name__}")
        except Exception as e:
            print(f"❌ Auto-detection failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 