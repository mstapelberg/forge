#!/usr/bin/env python
"""Performance benchmark using real model files."""

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

def create_test_structures(n_structures: int = 5, size_atoms: int = 128) -> List:
    """Create test structures for benchmarking."""
    structures = []
    
    for i in range(n_structures):
        # Create a structure with approximately the desired number of atoms
        base_atoms = bulk('V', 'bcc', a=3.01, cubic=True)
        multiplier = int(np.ceil((size_atoms / len(base_atoms)) ** (1/3)))
        atoms = base_atoms * (multiplier, multiplier, multiplier)
        
        # Trim to desired size
        if len(atoms) > size_atoms:
            atoms = atoms[:size_atoms]
            
        structures.append(atoms)
        
    return structures

def benchmark_mace_backend():
    """Benchmark MACE backend with real models."""
    print("=== MACE Backend Benchmark ===")
    
    try:
        from forge.calculators import get_supported_backends, create_ensemble_calculator
        if 'mace' not in get_supported_backends():
            print("⚠ MACE not available in this environment")
            return None
            
        # Real MACE model paths
        scratch_root = Path(__file__).parent.parent / "scratch"
        model_paths = [
            str(scratch_root / "potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model"),
            str(scratch_root / "potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_1_pr_stagetwo.model"),
            str(scratch_root / "potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_2_pr_stagetwo.model"),
        ]
        
        # Check if model files exist
        existing_models = [p for p in model_paths if Path(p).exists()]
        if not existing_models:
            print(f"⚠ No MACE model files found at {scratch_root / 'potentials/mace_gen_7_ensemble/'}")
            return None
        
        print(f"Found {len(existing_models)} MACE model files")
        
        # Create test structures
        structures = create_test_structures(n_structures=5, size_atoms=64)
        print(f"Testing with {len(structures)} structures of ~64 atoms each")
        
        # Test 1: Direct backend instantiation
        print("\n1. Testing MACE calculator instantiation:")
        with timer():
            calc = create_ensemble_calculator(
                model_paths=existing_models,
                backend='mace',
                device='cuda'
            )
        
        print(f"   Created calculator with {len(calc.models)} models")
        print(f"   Device: {calc.device}")
        print(f"   r_max: {calc.r_max}")
        
        # Test 2: Factory function overhead
        print("\n2. Testing factory function overhead:")
        with timer():
            calc2 = create_ensemble_calculator(
                model_paths=existing_models,
                backend='auto',  # Should auto-detect as MACE
                device='cuda'
            )
        
        # Test 3: Force calculations
        print("\n3. Testing force calculations:")
        times = []
        
        for i, atoms in enumerate(structures):
            start = time.perf_counter()
            all_forces = calc.forces_all(atoms)
            end = time.perf_counter()
            
            calc_time = (end - start) * 1000  # Convert to ms
            times.append(calc_time)
            
            # Verify output
            assert all_forces.shape == (len(existing_models), len(atoms), 3)
            print(f"  Structure {i+1} ({len(atoms)} atoms): {calc_time:.2f} ms")
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"\nMACE Force calculation summary:")
        print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Range: {min(times):.2f} - {max(times):.2f} ms")
        
        # Test 4: Additional methods
        print("\n4. Testing additional methods:")
        test_atoms = structures[0]
        
        with timer():
            mean_forces = calc.get_mean_forces(test_atoms)
        print(f"   Mean forces shape: {mean_forces.shape}")
        
        with timer():
            all_forces = calc.forces_all(test_atoms)
            variance = calc.calculate_normalized_force_variance(all_forces)
        print(f"   Variance calculation, mean variance: {np.mean(variance):.6f}")
        
        return {
            'backend': 'mace',
            'n_models': len(existing_models),
            'avg_force_time_ms': avg_time,
            'std_force_time_ms': std_time,
            'instantiation_successful': True
        }
        
    except Exception as e:
        print(f"❌ MACE benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_allegro_backend():
    """Benchmark Allegro backend with real models."""
    print("\n=== Allegro Backend Benchmark ===")
    
    try:
        from forge.calculators import get_supported_backends, create_ensemble_calculator
        available = get_supported_backends()
        if not any(b in available for b in ['allegro', 'nequip']):
            print("⚠ Allegro/NequIP not available in this environment")
            return None
            
        # Real Allegro model paths
        scratch_root = Path(__file__).parent.parent / "scratch"
        model_paths = [
            str(scratch_root / "data/potentials/compiled_gen-8-exploit/exploit_rmax5.75_lmax2_layers2_mlp384_seed42.nequip.pt2"),
            str(scratch_root / "data/potentials/compiled_gen-8-exploit/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.pt2"),
            str(scratch_root / "data/potentials/compiled_gen-8-exploit/explore_rmax5.75_lmax2_layers3_mlp256_seed42.nequip.pt2"),
        ]
        
        # Check if model files exist
        existing_models = [p for p in model_paths if Path(p).exists()]
        if not existing_models:
            print(f"⚠ No Allegro model files found at {scratch_root / 'data/potentials/compiled_gen-8-exploit/'}")
            # Try to find any .pt2 files
            pt2_dir = scratch_root / "data/potentials/compiled_gen-8-exploit/"
            if pt2_dir.exists():
                pt2_files = list(pt2_dir.glob("*.pt2"))
                if pt2_files:
                    print(f"Found {len(pt2_files)} .pt2 files:")
                    for f in pt2_files[:3]:  # Use first 3
                        print(f"  {f}")
                    existing_models = [str(f) for f in pt2_files[:3]]
                else:
                    print(f"No .pt2 files found in {pt2_dir}")
                    return None
            else:
                print(f"Directory {pt2_dir} does not exist")
                return None
        
        print(f"Found {len(existing_models)} Allegro model files")
        
        # Create test structures
        structures = create_test_structures(n_structures=5, size_atoms=64)
        print(f"Testing with {len(structures)} structures of ~64 atoms each")
        
        # Test 1: Direct backend instantiation
        print("\n1. Testing Allegro calculator instantiation:")
        try:
            with timer():
                calc = create_ensemble_calculator(
                    model_paths=existing_models,
                    backend='allegro',
                    device='cuda'
                )
            
            print(f"   Created calculator with {len(calc.models)} models")
            print(f"   Device: {calc.device}")
            print(f"   r_max: {calc.r_max}")
        except Exception as e:
            print(f"❌ Failed to create Allegro calculator: {e}")
            return None
        
        # Test 2: Factory function overhead
        print("\n2. Testing factory function overhead:")
        with timer():
            calc2 = create_ensemble_calculator(
                model_paths=existing_models,
                backend='auto',  # Should auto-detect as Allegro
                device='cuda'
            )
        
        # Test 3: Force calculations
        print("\n3. Testing force calculations:")
        times = []
        
        for i, atoms in enumerate(structures):
            try:
                start = time.perf_counter()
                all_forces = calc.forces_all(atoms)
                end = time.perf_counter()
                
                calc_time = (end - start) * 1000  # Convert to ms
                times.append(calc_time)
                
                # Verify output
                assert all_forces.shape == (len(existing_models), len(atoms), 3)
                print(f"  Structure {i+1} ({len(atoms)} atoms): {calc_time:.2f} ms")
            except Exception as e:
                print(f"  ❌ Force calculation {i+1} failed: {e}")
                return None
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"\nAllegro Force calculation summary:")
        print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Range: {min(times):.2f} - {max(times):.2f} ms")
        
        # Test 4: Additional methods
        print("\n4. Testing additional methods:")
        test_atoms = structures[0]
        
        with timer():
            mean_forces = calc.get_mean_forces(test_atoms)
        print(f"   Mean forces shape: {mean_forces.shape}")
        
        with timer():
            all_forces = calc.forces_all(test_atoms)
            variance = calc.calculate_normalized_force_variance(all_forces)
        print(f"   Variance calculation, mean variance: {np.mean(variance):.6f}")
        
        return {
            'backend': 'allegro',
            'n_models': len(existing_models),
            'avg_force_time_ms': avg_time,
            'std_force_time_ms': std_time,
            'instantiation_successful': True
        }
        
    except Exception as e:
        print(f"❌ Allegro benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_backends(results: List[Dict]):
    """Compare performance between backends."""
    if len(results) < 2:
        print("\nNeed both backends to compare performance")
        return
    
    print("\n=== Performance Comparison ===")
    
    mace_result = next((r for r in results if r['backend'] == 'mace'), None)
    allegro_result = next((r for r in results if r['backend'] == 'allegro'), None)
    
    if not mace_result or not allegro_result:
        print("Need both MACE and Allegro results for comparison")
        return
    
    mace_time = mace_result['avg_force_time_ms']
    allegro_time = allegro_result['avg_force_time_ms']
    
    overhead = ((allegro_time - mace_time) / mace_time) * 100
    
    print(f"MACE average: {mace_time:.2f} ms ({mace_result['n_models']} models)")
    print(f"Allegro average: {allegro_time:.2f} ms ({allegro_result['n_models']} models)")
    print(f"Difference: {allegro_time - mace_time:+.2f} ms ({overhead:+.1f}%)")
    
    if abs(overhead) < 5.0:
        print("✅ Performance requirement met: <5% difference")
    else:
        print("❌ Performance difference >5%")
        if overhead > 0:
            print("   Allegro is slower than MACE")
        else:
            print("   Allegro is faster than MACE")

def main():
    """Run real model benchmarks."""
    print("=== Real Model Performance Benchmark ===")
    print("Testing with actual MACE and Allegro model files\n")
    
    results = []
    
    # Test MACE
    mace_result = benchmark_mace_backend()
    if mace_result:
        results.append(mace_result)
    
    # Test Allegro
    allegro_result = benchmark_allegro_backend()
    if allegro_result:
        results.append(allegro_result)
    
    # Compare if we have both
    if len(results) >= 2:
        compare_backends(results)
    elif len(results) == 1:
        result = results[0]
        print(f"\n=== Single Backend Results ===")
        print(f"{result['backend'].upper()} backend:")
        print(f"  Average force calculation: {result['avg_force_time_ms']:.2f} ms")
        print(f"  Models tested: {result['n_models']}")
        print("ℹ Install the other backend to compare performance")
    else:
        print("\n❌ No backends successfully tested")
    
    print(f"\n=== Summary ===")
    if results:
        print("✅ Backend system functional with real models")
        backends_tested = [r['backend'] for r in results]
        print(f"✅ Successfully tested: {', '.join(backends_tested)}")
    else:
        print("❌ No backends could be tested")
        print("  Check model file paths and backend installations")

if __name__ == "__main__":
    main() 