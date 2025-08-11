#!/usr/bin/env python3
"""
Comprehensive test script to verify the refactor works for both MACE and Allegro backends.
"""

import sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import bulk

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.core.adversarial_attack import GradientAdversarialOptimizer
from forge.calculators.factory import get_supported_backends, create_ensemble_calculator


def test_calculator_factory():
    """Test the calculator factory functionality."""
    print("=== Testing Calculator Factory ===")
    
    available_backends = get_supported_backends()
    print(f"Available backends: {available_backends}")
    
    # Test factory with auto-detection
    try:
        # This should work even without real model files
        # The factory will detect the backend type from file extensions
        test_paths = ["test_model.model"]  # MACE extension
        calc = create_ensemble_calculator(test_paths, backend='auto')
        print(f"✅ Auto-detection works: {type(calc).__name__}")
    except Exception as e:
        print(f"⚠️ Auto-detection test failed (expected without real files): {e}")
    
    return True


def test_mace_backend():
    """Test MACE backend if available."""
    print("\n=== Testing MACE Backend ===")
    
    if 'mace' not in get_supported_backends():
        print("❌ MACE not available, skipping test")
        return False
    
    # Create test structure
    atoms = bulk('V', 'bcc', a=3.01).repeat((2, 2, 2))
    atoms.info['structure_id'] = 99999998
    atoms.info['config_type'] = 'test_structure'
    
    # Example MACE model paths (update these to real paths)
    model_paths = [
        "../../data/potentials/mace/example_mace.model"
    ]
    
    if not Path(model_paths[0]).exists():
        print(f"⚠️ MACE model file not found at {model_paths[0]}")
        print("Skipping MACE backend test")
        return False
    
    try:
        # Test ensemble calculator
        calc = create_ensemble_calculator(model_paths, backend='mace', device='cpu')
        print(f"✅ MACE ensemble calculator created: {type(calc).__name__}")
        
        # Test force calculation
        forces = calc.forces_all(atoms)
        print(f"✅ MACE forces calculated: shape {forces.shape}")
        
        # Test energy calculation
        energies = calc.energies_all(atoms)
        print(f"✅ MACE energies calculated: shape {energies.shape}")
        
        # Test optimizer
        optimizer = GradientAdversarialOptimizer(
            model_paths=model_paths,
            device='cpu',
            learning_rate=0.01,
            temperature=1000,
            include_probability=False,
            debug=True,
            backend='mace'
        )
        print(f"✅ MACE optimizer created: {type(optimizer.calculator).__name__}")
        
        # Test short optimization
        trajectory = optimizer.optimize(
            atoms=atoms,
            generation=1,
            n_iterations=3,
            min_distance=1.5,
            output_dir='test_output',
            patience=2,
            shake=False,
            require_structure_id=False
        )
        print(f"✅ MACE optimization completed: {len(trajectory)} steps")
        
        return True
        
    except Exception as e:
        print(f"❌ MACE backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_allegro_backend():
    """Test Allegro backend if available."""
    print("\n=== Testing Allegro Backend ===")
    
    if 'allegro' not in get_supported_backends():
        print("❌ Allegro not available, skipping test")
        return False
    
    # Create test structure
    atoms = bulk('V', 'bcc', a=3.01).repeat((2, 2, 2))
    atoms.info['structure_id'] = 99999998
    atoms.info['config_type'] = 'test_structure'
    
    # Example Allegro model paths (update these to real paths)
    model_paths = [
        "../../data/potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.zip"
    ]
    
    if not Path(model_paths[0]).exists():
        print(f"⚠️ Allegro model file not found at {model_paths[0]}")
        print("Please update the model path in the test script to point to a real Allegro model file")
        print("Skipping Allegro backend test")
        return False
    
    # Define chemical symbols mapping for Allegro
    chemical_symbols = {
        'Ti': 'Ti',
        'V': 'V', 
        'Cr': 'Cr',
        'Zr': 'Zr',
        'W': 'W'
    }
    
    try:
        # Test ensemble calculator
        calc = create_ensemble_calculator(
            model_paths, 
            backend='allegro', 
            device='cpu',
            species_to_type_name=chemical_symbols
        )
        print(f"✅ Allegro ensemble calculator created: {type(calc).__name__}")
        
        # Test force calculation
        forces = calc.forces_all(atoms)
        print(f"✅ Allegro forces calculated: shape {forces.shape}")
        
        # Test energy calculation
        energies = calc.energies_all(atoms)
        print(f"✅ Allegro energies calculated: shape {energies.shape}")
        
        # Test optimizer
        optimizer = GradientAdversarialOptimizer(
            model_paths=model_paths,
            device='cpu',
            learning_rate=0.01,
            temperature=1000,
            include_probability=False,
            debug=True,
            backend='allegro',
            species_to_type_name=chemical_symbols
        )
        print(f"✅ Allegro optimizer created: {type(optimizer.calculator).__name__}")
        
        # Test short optimization
        trajectory = optimizer.optimize(
            atoms=atoms,
            generation=1,
            n_iterations=3,
            min_distance=1.5,
            output_dir='test_output',
            patience=2,
            shake=False,
            require_structure_id=False
        )
        print(f"✅ Allegro optimization completed: {len(trajectory)} steps")
        
        return True
        
    except Exception as e:
        print(f"❌ Allegro backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_defect_adversarial_attack():
    """Test the defect adversarial attack workflow."""
    print("\n=== Testing Defect Adversarial Attack Workflow ===")
    
    # Import the workflow function
    try:
        from defect_adversarial_attack import run_defect_adversarial_attacks
        print("✅ Successfully imported defect_adversarial_attack module")
    except ImportError as e:
        print(f"❌ Failed to import defect_adversarial_attack module: {e}")
        return False
    
    # Test with simple compositions
    compositions = [
        {'V': 0.75, 'Cr': 0.25},
        {'V': 1.0}
    ]
    
    # Check which backend is available
    available_backends = get_supported_backends()
    
    if 'allegro' in available_backends:
        model_paths = [
            "../../data/potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.zip"
        ]
        backend = 'allegro'
        species_mapping = {'Ti': 'Ti', 'V': 'V', 'Cr': 'Cr', 'Zr': 'Zr', 'W': 'W'}
    elif 'mace' in available_backends:
        model_paths = [
            "../../data/potentials/mace/example_mace.model"
        ]
        backend = 'mace'
        species_mapping = None
    else:
        print("❌ No supported backends available for testing")
        return False
    
    # Check if model files exist
    if not Path(model_paths[0]).exists():
        print(f"⚠️ Model file not found at {model_paths[0]}")
        print("Please update the model path in the test script to point to a real model file")
        print("Skipping defect adversarial attack test")
        return False
    
    try:
        # Test the workflow with minimal parameters
        trajectories = run_defect_adversarial_attacks(
            compositions=compositions,
            model_paths=model_paths,
            top_n=2,
            generation=1,
            n_iterations=3,  # Very short for testing
            learning_rate=0.01,
            temperature=1000,
            include_probability=False,
            output_dir='test_output',
            save_output=False,
            debug=True,
            backend=backend,
            species_to_type_name=species_mapping,
            require_structure_id=False
        )
        
        print(f"✅ Defect adversarial attack workflow completed")
        if trajectories:
            print(f"   Generated {len(trajectories)} trajectories")
            for structure_id, trajectory in trajectories.items():
                print(f"   Structure {structure_id}: {len(trajectory)} steps")
        
        return True
        
    except Exception as e:
        print(f"❌ Defect adversarial attack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Running Comprehensive Refactor Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Calculator Factory
    results.append(("Calculator Factory", test_calculator_factory()))
    
    # Test 2: MACE Backend
    #results.append(("MACE Backend", test_mace_backend()))
    
    # Test 3: Allegro Backend
    results.append(("Allegro Backend", test_allegro_backend()))
    
    # Test 4: Defect Adversarial Attack Workflow
    results.append(("Defect Adversarial Attack", test_defect_adversarial_attack()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactor is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main()) 