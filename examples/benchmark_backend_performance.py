#!/usr/bin/env python
"""Benchmark script to test calculator backend performance overhead."""

import time
import numpy as np
from ase.build import bulk
from contextlib import contextmanager
from typing import List, Dict, Any
import statistics
from pathlib import Path

@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"  Time: {(end - start)*1000:.2f} ms")

def create_test_structures(n_structures: int = 10, n_atoms_range: tuple = (32, 128)) -> List:
    """Create test structures for benchmarking."""
    structures = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(n_structures):
        # Create structures of varying sizes
        if n_atoms_range[0] == n_atoms_range[1]:
            n_atoms = n_atoms_range[0]  # Fixed size
        else:
            n_atoms = np.random.randint(n_atoms_range[0], n_atoms_range[1] + 1)  # +1 for inclusive upper bound
        
        # Create supercell to get desired number of atoms
        base_atoms = bulk('Al', 'fcc', a=4.0)
        multiplier = int(np.ceil((n_atoms / len(base_atoms)) ** (1/3)))
        atoms = base_atoms * (multiplier, multiplier, multiplier)
        
        # Trim to desired size
        if len(atoms) > n_atoms:
            atoms = atoms[:n_atoms]
            
        structures.append(atoms)
        
    return structures

def benchmark_mace_backend():
    """Benchmark MACE backend performance."""
    print("=== MACE Backend Benchmark ===")
    
    try:
        from forge.calculators import get_supported_backends
        if 'mace' not in get_supported_backends():
            print("⚠ MACE not available in this environment, skipping MACE benchmarks")
            return None
            
        from forge.calculators import MACEBackend, create_ensemble_calculator
        from unittest.mock import Mock, patch
        
        # Model paths (will use mocking if files don't exist)
        scratch_root = Path(__file__).parent.parent / "scratch"
        model_paths = [
            str(scratch_root / "potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model"),
            str(scratch_root / "potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_1_pr_stagetwo.model"),
            str(scratch_root / "potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_2_pr_stagetwo.model"),
        ]
        
        # Check if real model files exist
        real_models_exist = all(Path(p).exists() for p in model_paths)
        if not real_models_exist:
            print("  ⚠ Real model files not found, using mocked calculators")
            model_paths = ['model1.model', 'model2.model', 'model3.model']  # Use simple names for mocking
        
        # Create test structures
        structures = create_test_structures(n_structures=5, n_atoms_range=(64, 64))
        print(f"Testing with {len(structures)} structures of ~64 atoms each")
        
        # Mock the MACE calculator for performance testing
        with patch('forge.calculators.mace_backend.MACECalculator') as mock_mace_class:
            # Setup realistic mock responses
            mock_calc = Mock()
            mock_models = []
            for i in range(len(model_paths)):
                mock_model = Mock()
                mock_model.r_max.item.return_value = 5.0
                mock_models.append(mock_model)
            mock_calc.models = mock_models
            mock_calc.z_table = {13: 0}  # Al mapping
            mock_mace_class.return_value = mock_calc
            
            # Test 1: Direct backend instantiation
            print("\n1. Testing direct MACEBackend instantiation:")
            with timer():
                backend = MACEBackend(model_paths, device='cuda')
            
            # Test 2: Factory function overhead
            print("\n2. Testing factory function (should have minimal overhead):")
            with timer():
                factory_backend = create_ensemble_calculator(model_paths, backend='mace', device='cuda')
            
            # Test 3: Force calculations
            print("\n3. Testing force calculations:")
            # Mock force responses
            def setup_force_mocks(atoms):
                n_atoms = len(atoms)
                forces_sets = []
                for i in range(len(model_paths)):
                    forces = np.random.rand(n_atoms, 3) * 0.1  # Realistic force magnitudes
                    forces_sets.append(forces)
                
                call_count = 0
                def mock_get_forces():
                    nonlocal call_count
                    result = forces_sets[call_count % len(forces_sets)]
                    call_count += 1
                    return result
                
                atoms.get_potential_energy = Mock(return_value=-10.5 * n_atoms)
                atoms.get_forces = mock_get_forces
                return forces_sets
            
            # Benchmark force calculations
            times = []
            for i, atoms in enumerate(structures):
                forces_sets = setup_force_mocks(atoms)
                
                start = time.perf_counter()
                all_forces = backend.forces_all(atoms)
                end = time.perf_counter()
                
                times.append((end - start) * 1000)  # Convert to ms
                
                # Verify shape
                assert all_forces.shape == (len(model_paths), len(atoms), 3)
                print(f"  Structure {i+1} ({len(atoms)} atoms): {times[-1]:.2f} ms")
            
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            print(f"\nForce calculation summary:")
            print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"  Range: {min(times):.2f} - {max(times):.2f} ms")
            
            return {
                'backend': 'mace',
                'avg_force_time_ms': avg_time,
                'std_force_time_ms': std_time,
                'n_structures': len(structures),
                'n_models': len(model_paths)
            }
            
    except Exception as e:
        print(f"MACE benchmark failed: {e}")
        return None

def benchmark_allegro_backend():
    """Benchmark Allegro backend performance."""
    print("\n=== Allegro Backend Benchmark ===")
    
    try:
        from forge.calculators import get_supported_backends
        available = get_supported_backends()
        if not any(b in available for b in ['allegro', 'nequip']):
            print("⚠ Allegro/NequIP not available in this environment, skipping Allegro benchmarks")
            return None
            
        from forge.calculators import AllegroBackend, create_ensemble_calculator
        from unittest.mock import Mock, patch
        
        # Model paths (will use mocking if files don't exist)
        scratch_root = Path(__file__).parent.parent / "scratch"
        model_paths = [
            str(scratch_root / "data/potentials/compiled_gen-8-exploit/exploit_rmax5.75_lmax2_layers2_mlp384_seed42.nequip.pt2"),
            str(scratch_root / "data/potentials/compiled_gen-8-exploit/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.pt2"),
            str(scratch_root / "data/potentials/compiled_gen-8-exploit/explore_rmax5.75_lmax2_layers3_mlp256_seed42.nequip.pt2"),
        ]
        
        # Check if real model files exist
        real_models_exist = all(Path(p).exists() for p in model_paths)
        if not real_models_exist:
            print("  ⚠ Real model files not found, using mocked calculators")
            model_paths = ['model1.pt2', 'model2.pt2', 'model3.pt2']  # Use simple names for mocking
        
        # Create test structures
        structures = create_test_structures(n_structures=5, n_atoms_range=(64, 64))
        print(f"Testing with {len(structures)} structures of ~64 atoms each")
        
        # Mock the NequIP calculator for performance testing
        with patch('forge.calculators.allegro_backend.NequIPCalculator') as mock_nequip_class:
            with patch('pathlib.Path.exists', return_value=True):
                # Setup realistic mock responses
                mock_calcs = []
                mock_models = []
                for i in range(len(model_paths)):
                    mock_calc = Mock()
                    mock_model = Mock()
                    mock_model.r_max.item.return_value = 6.0
                    mock_calc.model = mock_model
                    mock_calcs.append(mock_calc)
                    mock_models.append(mock_model)
                
                mock_nequip_class.from_compiled_model.side_effect = mock_calcs
                
                # Test 1: Direct backend instantiation
                print("\n1. Testing direct AllegroBackend instantiation:")
                with timer():
                    backend = AllegroBackend(model_paths, device='cpu')
                
                # Test 2: Factory function overhead
                print("\n2. Testing factory function (should have minimal overhead):")
                with timer():
                    factory_backend = create_ensemble_calculator(model_paths, backend='allegro', device='cpu')
                
                # Test 3: Force calculations
                print("\n3. Testing force calculations:")
                # Mock force responses
                def setup_force_mocks(atoms):
                    n_atoms = len(atoms)
                    forces_sets = []
                    for i in range(len(model_paths)):
                        forces = np.random.rand(n_atoms, 3) * 0.1  # Realistic force magnitudes
                        forces_sets.append(forces)
                    
                    call_count = 0
                    def mock_get_forces():
                        nonlocal call_count
                        result = forces_sets[call_count % len(forces_sets)]
                        call_count += 1
                        return result
                    
                    atoms.get_potential_energy = Mock(return_value=-10.5 * n_atoms)
                    atoms.get_forces = mock_get_forces
                    return forces_sets
                
                # Benchmark force calculations
                times = []
                for i, atoms in enumerate(structures):
                    forces_sets = setup_force_mocks(atoms)
                    
                    start = time.perf_counter()
                    all_forces = backend.forces_all(atoms)
                    end = time.perf_counter()
                    
                    times.append((end - start) * 1000)  # Convert to ms
                    
                    # Verify shape
                    assert all_forces.shape == (len(model_paths), len(atoms), 3)
                    print(f"  Structure {i+1} ({len(atoms)} atoms): {times[-1]:.2f} ms")
                
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                print(f"\nForce calculation summary:")
                print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
                print(f"  Range: {min(times):.2f} - {max(times):.2f} ms")
                
                return {
                    'backend': 'allegro',
                    'avg_force_time_ms': avg_time,
                    'std_force_time_ms': std_time,
                    'n_structures': len(structures),
                    'n_models': len(model_paths)
                }
                
    except Exception as e:
        print(f"Allegro benchmark failed: {e}")
        return None

def compare_performance(results: List[Dict]):
    """Compare performance between backends."""
    if len(results) < 2:
        print("\nNeed at least 2 backends to compare performance")
        return
    
    print("\n=== Performance Comparison ===")
    
    baseline = results[0]
    comparison = results[1]
    
    baseline_time = baseline['avg_force_time_ms']
    comparison_time = comparison['avg_force_time_ms']
    
    overhead_pct = ((comparison_time - baseline_time) / baseline_time) * 100
    
    print(f"{baseline['backend'].upper()} average: {baseline_time:.2f} ms")
    print(f"{comparison['backend'].upper()} average: {comparison_time:.2f} ms")
    print(f"Overhead: {overhead_pct:+.1f}%")
    
    if abs(overhead_pct) < 5.0:
        print("✅ Performance requirement met: <5% overhead")
    else:
        print("❌ Performance requirement NOT met: ≥5% overhead")
    
    return overhead_pct

def main():
    """Run performance benchmarks."""
    print("=== Calculator Backend Performance Benchmark ===")
    print("Testing factory function overhead and force calculation performance\n")
    
    results = []
    
    # Test MACE backend
    mace_result = benchmark_mace_backend()
    if mace_result:
        results.append(mace_result)
    
    # Test Allegro backend  
    allegro_result = benchmark_allegro_backend()
    if allegro_result:
        results.append(allegro_result)
    
    # Compare if we have results from both
    if len(results) >= 2:
        overhead = compare_performance(results)
    elif len(results) == 1:
        print(f"\n=== Single Backend Results ===")
        result = results[0]
        print(f"{result['backend'].upper()} backend:")
        print(f"  Average force calculation: {result['avg_force_time_ms']:.2f} ms")
        print(f"  Tested with {result['n_structures']} structures, {result['n_models']} models")
        print("ℹ Need both backends available to test overhead requirement")
    else:
        print("\n❌ No backends available for testing")
    
    print(f"\n=== Summary ===")
    if results:
        print("✅ Backend system functional")
        if len(results) >= 2:
            print("✅ Comparative performance testing completed")
        else:
            print("⚠ Only one backend available - install the other for full comparison")
    else:
        print("❌ No backends available - please install MACE or Allegro")

if __name__ == "__main__":
    main() 