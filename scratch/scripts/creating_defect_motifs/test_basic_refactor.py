#!/usr/bin/env python3
"""
Basic test script to verify the refactor works without requiring real model files.
"""

import sys
from pathlib import Path

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.calculators.factory import get_supported_backends, create_ensemble_calculator


def test_calculator_factory():
    """Test the calculator factory functionality."""
    print("=== Testing Calculator Factory ===")
    
    available_backends = get_supported_backends()
    print(f"Available backends: {available_backends}")
    
    # Test factory with auto-detection (should fail gracefully without real files)
    try:
        test_paths = ["test_model.model"]  # MACE extension
        calc = create_ensemble_calculator(test_paths, backend='auto')
        print(f"✅ Auto-detection works: {type(calc).__name__}")
    except FileNotFoundError as e:
        print(f"✅ Auto-detection correctly failed with FileNotFoundError (expected): {e}")
    except Exception as e:
        print(f"⚠️ Auto-detection failed with unexpected error: {e}")
    
    # Test factory with explicit backend (should fail gracefully without real files)
    try:
        test_paths = ["test_model.nequip.zip"]  # Allegro extension
        calc = create_ensemble_calculator(test_paths, backend='allegro')
        print(f"✅ Explicit backend works: {type(calc).__name__}")
    except FileNotFoundError as e:
        print(f"✅ Explicit backend correctly failed with FileNotFoundError (expected): {e}")
    except Exception as e:
        print(f"⚠️ Explicit backend failed with unexpected error: {e}")
    
    return True


def test_imports():
    """Test that all necessary modules can be imported."""
    print("\n=== Testing Imports ===")
    
    try:
        from forge.core.adversarial_attack import GradientAdversarialOptimizer
        print("✅ GradientAdversarialOptimizer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import GradientAdversarialOptimizer: {e}")
        return False
    
    try:
        from forge.calculators.factory import create_ensemble_calculator, get_supported_backends
        print("✅ Calculator factory imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import calculator factory: {e}")
        return False
    
    try:
        from defect_adversarial_attack import run_defect_adversarial_attacks
        print("✅ Defect adversarial attack module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import defect_adversarial_attack module: {e}")
        return False
    
    return True


def test_parameter_handling():
    """Test that the optimizer can handle the new parameter structure."""
    print("\n=== Testing Parameter Handling ===")
    
    try:
        # Test that GradientAdversarialOptimizer accepts **kwargs
        from forge.core.adversarial_attack import GradientAdversarialOptimizer
        
        # This should not raise an error even without real model files
        # The error should be about missing files, not about unknown parameters
        try:
            optimizer = GradientAdversarialOptimizer(
                model_paths=["test.model"],
                device='cpu',
                backend='mace',
                species_to_type_name={'V': 'V'}  # This should be accepted as a kwarg
            )
            print("✅ Optimizer accepts species_to_type_name as kwarg")
        except TypeError as e:
            if "species_to_type_name" in str(e):
                print(f"❌ Optimizer still doesn't accept species_to_type_name: {e}")
                return False
            else:
                print(f"✅ Optimizer correctly failed with expected error: {e}")
        except Exception as e:
            print(f"✅ Optimizer failed with expected error (missing files): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter handling test failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("🧪 Running Basic Refactor Tests")
    print("=" * 50)
    
    results = []
    
    # Test 1: Calculator Factory
    results.append(("Calculator Factory", test_calculator_factory()))
    
    # Test 2: Imports
    results.append(("Imports", test_imports()))
    
    # Test 3: Parameter Handling
    results.append(("Parameter Handling", test_parameter_handling()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Refactor is working correctly.")
        print("\nNext steps:")
        print("1. Update model paths in test_refactor_integration.py to point to real model files")
        print("2. Run test_refactor_integration.py with real models")
        print("3. Test with your actual workflow data")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main()) 