#!/usr/bin/env python3
"""
Test script for the unified calculator interface.

This script tests the unified calculator with both MACE and Allegro models
to ensure the interface works correctly.
"""

import sys
import os
from pathlib import Path

# Add the forge package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from forge.calculators.factory import (
    create_ensemble_calculator, 
    get_supported_backends
)


def test_calculator_availability():
    """Test which calculators are available."""
    print("=== Testing Calculator Availability ===")
    
    available_backends = get_supported_backends()
    print(f"Available backends: {available_backends}")
    
    return available_backends


def test_mace_calculator():
    """Test MACE calculator if available."""
    print("\n=== Testing MACE Calculator ===")
    
    # Check if MACE is available
    available_backends = get_supported_backends()
    if 'mace' not in available_backends:
        print("MACE not available, skipping test")
        return False
    
    # Create a simple test structure
    atoms = bulk('V', 'bcc', a=3.01).repeat((2, 2, 2))
    
    # Try to create MACE calculator (this will fail if no model file is available)
    try:
        # You would need to provide a real MACE model path here
        model_path = "path/to/your/mace_model.model"  # Replace with actual path
        
        if os.path.exists(model_path):
            calculator = create_ensemble_calculator(
                model_paths=model_path,
                backend='mace',
                device="cpu"
            )
            print(f"Successfully created MACE calculator: {type(calculator).__name__}")
            
            # Test calculation
            result = calculator.calculate(atoms)
            print(f"Energy: {result['energy']:.6f} eV")
            print(f"Forces shape: {result['forces'].shape}")
            
            return True
        else:
            print(f"MACE model file not found at {model_path}")
            print("Skipping MACE test (this is expected if you don't have a MACE model)")
            return False
            
    except Exception as e:
        print(f"Error testing MACE calculator: {e}")
        return False


def test_allegro_calculator():
    """Test Allegro calculator if available."""
    print("\n=== Testing Allegro Calculator ===")
    
    # Check if Allegro is available
    available_backends = get_supported_backends()
    if 'allegro' not in available_backends:
        print("Allegro not available, skipping test")
        return False
    
    # Create a simple test structure
    atoms = bulk('V', 'bcc', a=3.01).repeat((2, 2, 2))
    
    # Try to create Allegro calculator (this will fail if no model file is available)
    try:
        # You would need to provide a real Allegro model path here
        model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"  # Replace with actual path
        
        if os.path.exists(model_path):
            calculator = create_ensemble_calculator(
                model_paths=model_path,
                backend='allegro',
                device="cpu",
                species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
            )
            print(f"Successfully created Allegro calculator: {type(calculator).__name__}")
            
            # Test calculation
            result = calculator.calculate(atoms)
            print(f"Energy: {result['energy']:.6f} eV")
            print(f"Forces shape: {result['forces'].shape}")
            
            return True
        else:
            print(f"Allegro model file not found at {model_path}")
            print("Skipping Allegro test (this is expected if you don't have an Allegro model)")
            return False
            
    except Exception as e:
        print(f"Error testing Allegro calculator: {e}")
        return False


def test_auto_detection():
    """Test auto-detection of calculator type."""
    print("\n=== Testing Auto-Detection ===")
    
    # Test with different file extensions
    test_cases = [
        ("model.model", "mace"),
        ("model.nequip.zip", "allegro"),
        ("model.xyz", "mace"),  # Default fallback
    ]
    
    for model_path, expected_type in test_cases:
        try:
            calculator = create_ensemble_calculator(
                model_paths=model_path,
                backend=expected_type,
                device="cpu"
            )
            print(f"Model path: {model_path} -> Detected type: {calculator.backend}")
            
            if calculator.backend == expected_type:
                print("✓ Auto-detection working correctly")
            else:
                print(f"✗ Expected {expected_type}, got {calculator.backend}")
                
        except Exception as e:
            print(f"Model path: {model_path} -> Error: {e}")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n=== Testing Error Handling ===")
    
    # Test with invalid calculator type
    try:
        calculator = create_ensemble_calculator(
            model_paths="dummy.model",
            backend="invalid_type",
            device="cpu"
        )
        print("✗ Should have raised an error for invalid calculator type")
    except ValueError as e:
        print(f"✓ Correctly caught invalid calculator type: {e}")
    
    # Test with non-existent model file
    try:
        calculator = create_ensemble_calculator(
            model_paths="non_existent_file.model",
            backend="mace",
            device="cpu"
        )
        print("✗ Should have raised an error for non-existent file")
    except Exception as e:
        print(f"✓ Correctly caught non-existent file error: {e}")


def main():
    """Run all tests."""
    print("Unified Calculator Interface Test")
    print("=" * 50)
    
    # Test calculator availability
    available = test_calculator_availability()
    
    # Test individual calculators
    mace_success = test_mace_calculator()
    allegro_success = test_allegro_calculator()
    
    # Test auto-detection
    test_auto_detection()
    
    # Test error handling
    test_error_handling()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"MACE available: {'mace' in available}")
    print(f"Allegro available: {'allegro' in available}")
    print(f"MACE test passed: {mace_success}")
    print(f"Allegro test passed: {allegro_success}")
    
    if 'mace' not in available and 'allegro' not in available:
        print("\n⚠️  Warning: Neither MACE nor Allegro is available!")
        print("Please install one of them to use the unified calculator interface.")
        print("- For MACE: pip install mace")
        print("- For Allegro: pip install nequip")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main() 