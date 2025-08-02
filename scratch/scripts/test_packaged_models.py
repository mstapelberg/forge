#!/usr/bin/env python
"""Test script to verify packaged models work with autograd."""

import torch
import numpy as np
from pathlib import Path
from glob import glob
from ase import Atoms

# Test imports
try:
    from forge.calculators import create_ensemble_calculator
    from nequip.data import AtomicDataDict, from_ase
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_packaged_models():
    """Test that packaged models preserve gradients."""
    
    # Find packaged models
    package_paths = glob('../data/potentials/packaged_gen-8-exploit/*.zip')
    if not package_paths:
        print("❌ No packaged models found")
        return False
    
    print(f"Found {len(package_paths)} packaged models")
    
    # Use only first 2 models for testing
    test_models = package_paths[:2]
    print(f"Testing with {len(test_models)} models:")
    for i, path in enumerate(test_models):
        print(f"  {i+1}. {Path(path).name}")
    
    try:
        # Create calculator
        calc = create_ensemble_calculator(
            model_paths=test_models,
            backend='allegro',
            device='cuda',
            default_dtype='float32'
        )
        print(f"✅ Successfully loaded {len(calc.models)} models")
        
        # Create a simple test structure
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
        ])
        atoms = Atoms('Cr4', positions=positions, cell=[5, 5, 5], pbc=True)
        
        # Test basic force calculation
        forces = calc.forces_all(atoms)
        print(f"✅ Basic force calculation successful: {forces.shape}")
        
        # Test autograd with packaged models
        print("\n🧪 Testing autograd gradient flow...")
        
        # Get raw models for autograd test
        models = calc._models
        
        # Prepare data
        data = from_ase(atoms)
        ref_calc = calc._calculators[0]
        for transform in ref_calc.transforms:
            data = transform(data)
        data = AtomicDataDict.to_(data, 'cuda')
        
        # Convert all tensors to float32
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                data[key] = value.to(dtype=torch.float32, device='cuda')
        
        # Create positions tensor that requires gradients
        positions_tensor = data[AtomicDataDict.POSITIONS_KEY].clone().detach()
        positions_tensor.requires_grad_(True)
        data[AtomicDataDict.POSITIONS_KEY] = positions_tensor
        
        print(f"📍 Input positions require_grad: {positions_tensor.requires_grad}")
        
        # Test gradient flow through first model
        model = models[0]
        model.train()  # Enable training mode for gradient computation
        
        with torch.enable_grad():
            output = model(data)
            
            # Check if forces preserve gradients
            if AtomicDataDict.FORCE_KEY in output:
                forces = output[AtomicDataDict.FORCE_KEY]
                print(f"📍 Output forces require_grad: {forces.requires_grad}")
                print(f"📍 Output forces grad_fn: {forces.grad_fn}")
                
                if forces.requires_grad and forces.grad_fn is not None:
                    print("✅ SUCCESS: Gradients are preserved through packaged model!")
                    
                    # Test backward pass
                    loss = torch.mean(forces**2)
                    loss.backward()
                    
                    if positions_tensor.grad is not None:
                        print(f"✅ SUCCESS: Backward pass works! Gradient shape: {positions_tensor.grad.shape}")
                        return True
                    else:
                        print("❌ FAILED: No gradients computed for input positions")
                        return False
                else:
                    print("❌ FAILED: Forces do not preserve gradients")
                    return False
            else:
                print("❌ FAILED: No forces in model output")
                return False
                
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Packaged Models with Autograd")
    print("=" * 50)
    
    success = test_packaged_models()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Packaged models work with autograd.")
        print("You can now run your adversarial attack script.")
    else:
        print("💥 TESTS FAILED! Check the errors above.") 