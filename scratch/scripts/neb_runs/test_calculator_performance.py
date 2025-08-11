#!/usr/bin/env python3
"""
Focused test script to identify performance bottlenecks in factory calculators.

This script tests specific performance issues:
1. Multiple model loading overhead
2. Repeated data conversion overhead
3. No caching overhead
4. Sequential vs parallel processing
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ase import Atoms
from ase.build import bulk

# Try to import forge calculators
try:
    from forge.calculators.factory import create_ensemble_calculator, get_supported_backends
    FORGE_AVAILABLE = True
except ImportError:
    FORGE_AVAILABLE = False
    print("Warning: Forge calculators not available")

# Try to import MACE and NequIP calculators directly
try:
    from mace.calculators.mace import MACECalculator
    MACE_DIRECT_AVAILABLE = True
except ImportError:
    MACE_DIRECT_AVAILABLE = False

try:
    from nequip.ase import NequIPCalculator
    NEQUIP_DIRECT_AVAILABLE = True
except ImportError:
    NEQUIP_DIRECT_AVAILABLE = False


def test_multiple_model_overhead(model_paths: List[str], backend: str = 'auto'):
    """Test the overhead of loading multiple models vs single model.
    
    Args:
        model_paths: List of model file paths
        backend: Backend type
    """
    print("=== Testing Multiple Model Loading Overhead ===")
    
    if not FORGE_AVAILABLE:
        print("Skipping - forge not available")
        return
    
    # Test structure
    atoms = bulk('V', 'bcc', a=3.01).repeat((4, 4, 4))
    print(f"Test structure: {len(atoms)} atoms")
    
    # Test with single model
    if len(model_paths) > 0:
        print(f"\n1. Single Model Performance:")
        try:
            start_time = time.time()
            single_calc = create_ensemble_calculator(
                model_paths=[model_paths[0]],
                backend=backend,
                device='cuda'
            )
            load_time_single = time.time() - start_time
            print(f"   Load time: {load_time_single:.3f}s")
            
            # Test calculation time
            start_time = time.time()
            for _ in range(10):
                energy = single_calc.get_potential_energy(atoms)
                forces = single_calc.get_forces(atoms)
            calc_time_single = (time.time() - start_time) / 10
            print(f"   Average calculation time: {calc_time_single:.4f}s")
            
        except Exception as e:
            print(f"   Error with single model: {e}")
    
    # Test with multiple models
    if len(model_paths) > 1:
        print(f"\n2. Multiple Models Performance ({len(model_paths)} models):")
        try:
            start_time = time.time()
            multi_calc = create_ensemble_calculator(
                model_paths=model_paths,
                backend=backend,
                device='cuda'
            )
            load_time_multi = time.time() - start_time
            print(f"   Load time: {load_time_multi:.3f}s")
            print(f"   Load overhead: {load_time_multi/load_time_single:.2f}x")
            
            # Test calculation time
            start_time = time.time()
            for _ in range(10):
                energy = multi_calc.get_potential_energy(atoms)
                forces = multi_calc.get_forces(atoms)
            calc_time_multi = (time.time() - start_time) / 10
            print(f"   Average calculation time: {calc_time_multi:.4f}s")
            print(f"   Calculation overhead: {calc_time_multi/calc_time_single:.2f}x")
            
        except Exception as e:
            print(f"   Error with multiple models: {e}")


def test_data_conversion_overhead(model_paths: List[str], backend: str = 'auto'):
    """Test the overhead of repeated data conversion.
    
    Args:
        model_paths: List of model file paths
        backend: Backend type
    """
    print("\n=== Testing Data Conversion Overhead ===")
    
    if not FORGE_AVAILABLE:
        print("Skipping - forge not available")
        return
    
    # Test structure
    atoms = bulk('V', 'bcc', a=3.01).repeat((4, 4, 4))
    print(f"Test structure: {len(atoms)} atoms")
    
    if len(model_paths) == 0:
        print("No model paths provided")
        return
    
    try:
        # Create calculator
        calc = create_ensemble_calculator(
            model_paths=[model_paths[0]],  # Use single model for fair comparison
            backend=backend,
            device='cuda'
        )
        
        # Test repeated calculations (should trigger data conversion each time)
        print("\n1. Repeated Calculations (no caching):")
        start_time = time.time()
        for i in range(20):
            energy = calc.get_potential_energy(atoms)
            forces = calc.get_forces(atoms)
        total_time = time.time() - start_time
        avg_time = total_time / 20
        print(f"   Average time per calculation: {avg_time:.4f}s")
        
        # Test with atoms attached (should reduce conversion overhead)
        print("\n2. Calculations with Attached Atoms:")
        if hasattr(calc, 'set_atoms'):
            # Ensemble calculator - can attach atoms
            calc.set_atoms(atoms)
            start_time = time.time()
            for i in range(20):
                energy = calc.get_potential_energy()  # No atoms parameter
                forces = calc.get_forces()  # No atoms parameter
            total_time = time.time() - start_time
            avg_time_attached = total_time / 20
            print(f"   Average time per calculation: {avg_time_attached:.4f}s")
            print(f"   Improvement: {avg_time/avg_time_attached:.2f}x")
        else:
            # Native calculator - no set_atoms method, skip this test
            print("   Skipping - native calculator doesn't support set_atoms")
            avg_time_attached = avg_time  # No improvement
        
    except Exception as e:
        print(f"Error testing data conversion: {e}")


def test_native_vs_factory_comparison(model_paths: List[str], backend: str = 'auto'):
    """Compare factory calculator vs native calculator performance.
    
    Args:
        model_paths: List of model file paths
        backend: Backend type
    """
    print("\n=== Testing Factory vs Native Calculator Performance ===")
    
    if not FORGE_AVAILABLE:
        print("Skipping - forge not available")
        return
    
    # Test structure
    atoms = bulk('V', 'bcc', a=3.01).repeat((4, 4, 4))
    print(f"Test structure: {len(atoms)} atoms")
    
    if len(model_paths) == 0:
        print("No model paths provided")
        return
    
    results = {}
    
    # Test factory calculator
    try:
        print("\n1. Factory Calculator:")
        start_time = time.time()
        factory_calc = create_ensemble_calculator(
            model_paths=[model_paths[0]],  # Use single model for fair comparison
            backend=backend,
            device='cuda'
        )
        factory_load_time = time.time() - start_time
        print(f"   Load time: {factory_load_time:.3f}s")
        
        # Test calculation time
        start_time = time.time()
        for _ in range(20):
            energy = factory_calc.get_potential_energy(atoms)
            forces = factory_calc.get_forces(atoms)
        factory_calc_time = (time.time() - start_time) / 20
        print(f"   Average calculation time: {factory_calc_time:.4f}s")
        
        results['factory'] = {
            'load_time': factory_load_time,
            'calc_time': factory_calc_time
        }
        
    except Exception as e:
        print(f"   Error with factory calculator: {e}")
    
    # Test native calculator
    try:
        print("\n2. Native Calculator:")
        if backend.lower() in ['mace', 'auto'] and MACE_DIRECT_AVAILABLE:
            start_time = time.time()
            native_calc = MACECalculator(
                model_paths=[model_paths[0]],
                device='cuda'
            )
            native_load_time = time.time() - start_time
            print(f"   Load time: {native_load_time:.3f}s")
            
            # Test calculation time
            start_time = time.time()
            for _ in range(20):
                energy = native_calc.get_potential_energy(atoms)
                forces = native_calc.get_forces(atoms)
            native_calc_time = (time.time() - start_time) / 20
            print(f"   Average calculation time: {native_calc_time:.4f}s")
            
            results['native'] = {
                'load_time': native_load_time,
                'calc_time': native_calc_time
            }
            
        elif backend.lower() in ['allegro', 'nequip', 'auto'] and NEQUIP_DIRECT_AVAILABLE:
            start_time = time.time()
            native_calc = NequIPCalculator._from_packaged_model(
                package_path=model_paths[0],
                device='cuda',
                chemical_symbols=['Ti', 'V', 'Cr', 'Zr', 'W']
            )
            native_load_time = time.time() - start_time
            print(f"   Load time: {native_load_time:.3f}s")
            
            # Test calculation time
            start_time = time.time()
            for _ in range(20):
                energy = native_calc.get_potential_energy(atoms)
                forces = native_calc.get_forces(atoms)
            native_calc_time = (time.time() - start_time) / 20
            print(f"   Average calculation time: {native_calc_time:.4f}s")
            
            results['native'] = {
                'load_time': native_load_time,
                'calc_time': native_calc_time
            }
            
        else:
            print("   No suitable native calculator available")
            
    except Exception as e:
        print(f"   Error with native calculator: {e}")
    
    # Print comparison
    if 'factory' in results and 'native' in results:
        print("\n3. Performance Comparison:")
        load_overhead = results['factory']['load_time'] / results['native']['load_time']
        calc_overhead = results['factory']['calc_time'] / results['native']['calc_time']
        
        print(f"   Load time overhead: {load_overhead:.2f}x")
        print(f"   Calculation time overhead: {calc_overhead:.2f}x")
        
        if calc_overhead > 2.0:
            print(f"   ⚠️  Factory calculator is {calc_overhead:.1f}x slower than native!")
        elif calc_overhead > 1.5:
            print(f"   ⚠️  Factory calculator is {calc_overhead:.1f}x slower than native")
        else:
            print(f"   ✅ Factory calculator performance is acceptable")


def test_optimization_performance(model_paths: List[str], backend: str = 'auto'):
    """Test optimization performance with factory vs native calculators.
    
    Args:
        model_paths: List of model file paths
        backend: Backend type
    """
    print("\n=== Testing Optimization Performance ===")
    
    if not FORGE_AVAILABLE:
        print("Skipping - forge not available")
        return
    
    # Test structure (smaller for faster optimization)
    atoms = bulk('V', 'bcc', a=3.01).repeat((4, 4, 4))  # 2 atoms
    print(f"Test structure: {len(atoms)} atoms")
    
    if len(model_paths) == 0:
        print("No model paths provided")
        return
    
    from ase.optimize import BFGS
    
    # Test factory calculator optimization
    try:
        print("\n1. Factory Calculator Optimization:")
        factory_calc = create_ensemble_calculator(
            model_paths=[model_paths[0]],
            backend=backend,
            device='cuda'
        )
        
        atoms_factory = atoms.copy()
        atoms_factory.calc = factory_calc
        
        start_time = time.time()
        optimizer = BFGS(atoms_factory, trajectory=None, logfile=None)
        optimizer.run(fmax=0.05, steps=5)  # Limited steps for testing
        factory_time = time.time() - start_time
        
        print(f"   Optimization time: {factory_time:.3f}s")
        print(f"   Steps completed: {optimizer.get_number_of_steps()}")
        
    except Exception as e:
        print(f"   Error with factory calculator optimization: {e}")
        factory_time = None
    
    # Test native calculator optimization
    try:
        print("\n2. Native Calculator Optimization:")
        if backend.lower() in ['mace', 'auto'] and MACE_DIRECT_AVAILABLE:
            native_calc = MACECalculator(
                model_paths=[model_paths[0]],
                device='cuda'
            )
        elif backend.lower() in ['allegro', 'nequip', 'auto'] and NEQUIP_DIRECT_AVAILABLE:
            native_calc = NequIPCalculator._from_packaged_model(
                package_path=model_paths[0],
                device='cuda',
                chemical_symbols=['Ti', 'V', 'Cr', 'Zr', 'W']
            )
        else:
            print("   No suitable native calculator available")
            return
        
        atoms_native = atoms.copy()
        atoms_native.calc = native_calc
        
        start_time = time.time()
        optimizer_native = BFGS(atoms_native, trajectory=None, logfile=None)
        optimizer_native.run(fmax=0.05, steps=5)  # Limited steps for testing
        native_time = time.time() - start_time
        
        print(f"   Optimization time: {native_time:.3f}s")
        print(f"   Steps completed: {optimizer_native.get_number_of_steps()}")
        
    except Exception as e:
        print(f"   Error with native calculator optimization: {e}")
        native_time = None
    
    # Print comparison
    if factory_time is not None and native_time is not None:
        print("\n3. Optimization Performance Comparison:")
        overhead = factory_time / native_time
        print(f"   Factory overhead: {overhead:.2f}x")
        
        if overhead > 2.0:
            print(f"   ⚠️  Factory calculator optimization is {overhead:.1f}x slower!")
        elif overhead > 1.5:
            print(f"   ⚠️  Factory calculator optimization is {overhead:.1f}x slower")
        else:
            print(f"   ✅ Factory calculator optimization performance is acceptable")


def main():
    """Main function to run performance tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test calculator performance bottlenecks')
    parser.add_argument('--model-paths', nargs='+', required=True,
                       help='Paths to model files')
    parser.add_argument('--backend', default='auto',
                       choices=['auto', 'mace', 'allegro', 'nequip'],
                       help='Backend type')
    
    args = parser.parse_args()
    
    # Check if model files exist
    for path in args.model_paths:
        if not Path(path).exists():
            print(f"Error: Model file not found: {path}")
            return 1
    
    print("Calculator Performance Bottleneck Analysis")
    print("="*50)
    print(f"Model paths: {args.model_paths}")
    print(f"Backend: {args.backend}")
    print(f"Available backends: {get_supported_backends() if FORGE_AVAILABLE else 'None'}")
    
    # Run tests
    test_multiple_model_overhead(args.model_paths, args.backend)
    test_data_conversion_overhead(args.model_paths, args.backend)
    test_native_vs_factory_comparison(args.model_paths, args.backend)
    test_optimization_performance(args.model_paths, args.backend)
    
    print("\n" + "="*50)
    print("PERFORMANCE RECOMMENDATIONS")
    print("="*50)
    print("1. Use native ASE calculators for NEB and structural optimization")
    print("2. Only use factory calculators for adversarial attacks (ensemble uncertainty)")
    print("3. Consider implementing caching in factory calculators")
    print("4. Consider parallel processing for ensemble calculations")
    print("5. Profile data conversion overhead and optimize if needed")
    
    return 0


if __name__ == "__main__":
    exit(main()) 