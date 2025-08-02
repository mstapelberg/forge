#!/usr/bin/env python
"""Practical example of using the new calculator backend system."""

import os
from pathlib import Path
from forge.core.database import DatabaseManager
from forge.workflows.adversarial_attack import run_adversarial_attacks
from forge.calculators import create_ensemble_calculator, get_supported_backends

def example_with_mace_models():
    """Example usage with MACE models."""
    print("=== Example: Using MACE Models ===")
    
    # Check if MACE is available
    available_backends = get_supported_backends()
    if 'mace' not in available_backends:
        print("⚠ MACE not available in this environment")
        print("  Install with: pip install mace-torch")
        return
    
    # Example model paths (adjust to your actual model files)
    model_paths = [
        '../potentials/mace_gen_6_ensemble/gen_7_model_0-2025-02-12_stagetwo_compiled.model',
        '../potentials/mace_gen_6_ensemble/gen_7_model_1-2025-02-12_stagetwo_compiled.model',
        '../potentials/mace_gen_6_ensemble/gen_7_model_2-2025-02-12_stagetwo_compiled.model'
    ]
    
    # Check if model files exist
    existing_models = [p for p in model_paths if Path(p).exists()]
    if not existing_models:
        print(f"⚠ No MACE model files found at specified paths")
        print(f"  Example paths: {model_paths}")
        print(f"  Please adjust paths to your actual MACE model files")
        return
    
    print(f"✓ Found {len(existing_models)} MACE model files")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Example structure IDs (adjust to your actual database IDs)
    structure_ids = [1, 2, 3, 4, 5]  # Replace with your structure IDs
    
    try:
        # Run adversarial attacks with MACE backend
        trajectories = run_adversarial_attacks(
            db_manager=db_manager,
            structure_ids=structure_ids,
            model_paths=existing_models,
            top_n=3,  # Select top 3 structures
            generation=9,
            n_iterations=50,  # Shorter for example
            learning_rate=0.01,
            temperature=1000,
            include_probability=False,
            min_distance=1.2,
            use_energy_per_atom=True,
            device='cuda',  # Use GPU if available
            debug=False,
            output_dir='../data/adversarial_attacks/example_mace',
            save_output=True,  # Save to files
            patience=25,
            shake=False,
            ranking_metric='force_rmse',
            reference_calculator='vasp',
            cache_rmse=True,
            plot_rmse_histogram=True,
            rmse_cutoff=0.1,
            backend='mace'  # Use MACE backend
        )
        
        print("✅ MACE adversarial attack completed successfully")
        
        if trajectories is None:
            print("📁 Results saved to '../data/adversarial_attacks/example_mace'")
        else:
            print(f"📊 Generated {len(trajectories)} trajectories")
            
    except Exception as e:
        print(f"❌ MACE example failed: {e}")
        print("  Check your database connection and structure IDs")

def example_with_allegro_models():
    """Example usage with Allegro models."""
    print("\n=== Example: Using Allegro Models ===")
    
    # Check if Allegro is available
    available_backends = get_supported_backends()
    if not any(b in available_backends for b in ['allegro', 'nequip']):
        print("⚠ Allegro/NequIP not available in this environment")
        print("  Install with: pip install nequip")
        return
    
    # Example model paths (adjust to your actual model files)
    model_paths = [
        '../data/potentials/compiled_gen-8-exploit/model1.nequip.pt2',
        '../data/potentials/compiled_gen-8-exploit/model2.nequip.pt2',
        '../data/potentials/compiled_gen-8-exploit/model3.nequip.pt2'
    ]
    
    # Check if model files exist
    existing_models = [p for p in model_paths if Path(p).exists()]
    if not existing_models:
        print(f"⚠ No Allegro model files found at specified paths")
        print(f"  Example paths: {model_paths}")
        print(f"  Please adjust paths to your actual Allegro .pt2 files")
        return
    
    print(f"✓ Found {len(existing_models)} Allegro model files")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Example structure IDs (adjust to your actual database IDs)
    structure_ids = [1, 2, 3, 4, 5]  # Replace with your structure IDs
    
    try:
        # Note: Current limitation - autograd optimization only works with MACE
        print("⚠ Current limitation: Autograd optimization requires MACE backend")
        print("  You can use Allegro for force calculations and ranking:")
        
        # Test force calculations with Allegro backend
        if existing_models and len(structure_ids) > 0:
            print("  Testing force calculations with Allegro backend...")
            
            # Create Allegro calculator
            calc = create_ensemble_calculator(
                model_paths=existing_models[:2],  # Use first 2 models
                backend='allegro',
                device='cuda'
            )
            
            print(f"  ✓ Created {type(calc).__name__} with {len(calc.models)} models")
            print(f"  ✓ Device: {calc.device}")
            print(f"  ✓ r_max: {calc.r_max}")
            
            # You could use this for ranking structures by force variance:
            # atoms = get_atoms_from_database(structure_ids[0])
            # all_forces = calc.forces_all(atoms)
            # variance = calc.calculate_normalized_force_variance(all_forces)
            
            print("  💡 For full adversarial attack workflow:")
            print("     Install MACE alongside Allegro, or use MACE models for optimization")
            
    except Exception as e:
        print(f"❌ Allegro example failed: {e}")

def example_force_calculation_comparison():
    """Example comparing force calculations between backends."""
    print("\n=== Example: Comparing Force Calculations ===")
    
    available_backends = get_supported_backends()
    print(f"Available backends: {available_backends}")
    
    if len(available_backends) < 2:
        print("⚠ Need at least 2 backends for comparison")
        print("  Install both MACE and Allegro to compare")
        return
    
    # This would require actual model files and atoms
    print("💡 To compare force calculations:")
    print("""
    from forge.calculators import create_ensemble_calculator
    from ase.build import bulk
    
    # Create test structure
    atoms = bulk('Al', 'fcc', a=4.0, cubic=True)
    
    # Test MACE backend
    mace_calc = create_ensemble_calculator(
        model_paths=['model1.model', 'model2.model'],
        backend='mace'
    )
    mace_forces = mace_calc.forces_all(atoms)
    mace_variance = mace_calc.calculate_normalized_force_variance(mace_forces)
    
    # Test Allegro backend  
    allegro_calc = create_ensemble_calculator(
        model_paths=['model1.pt2', 'model2.pt2'],
        backend='allegro'
    )
    allegro_forces = allegro_calc.forces_all(atoms)
    allegro_variance = allegro_calc.calculate_normalized_force_variance(allegro_forces)
    
    # Compare results
    print(f"MACE variance: {np.mean(mace_variance):.6f}")
    print(f"Allegro variance: {np.mean(allegro_variance):.6f}")
    """)

def example_auto_detection():
    """Example of automatic backend detection."""
    print("\n=== Example: Automatic Backend Detection ===")
    
    print("💡 The factory function can auto-detect backend from file extensions:")
    print("""
    from forge.calculators import create_ensemble_calculator
    
    # Auto-detect MACE from .model files
    mace_calc = create_ensemble_calculator(
        model_paths=['model1.model', 'model2.model'],
        backend='auto'  # Will detect 'mace'
    )
    
    # Auto-detect Allegro from .pt2 files
    allegro_calc = create_ensemble_calculator(
        model_paths=['model1.pt2', 'model2.pt2'], 
        backend='auto'  # Will detect 'allegro'
    )
    
    # Explicit backend specification (recommended for production)
    calc = create_ensemble_calculator(
        model_paths=your_models,
        backend='mace',  # or 'allegro'
        device='cuda'
    )
    """)

def example_migration_from_old_code():
    """Example of migrating from old MACE-specific code."""
    print("\n=== Example: Migrating from Old Code ===")
    
    print("🔄 Old MACE-specific code:")
    print("""
    # OLD CODE:
    from mace.calculators import MACECalculator
    
    calc = MACECalculator(model_paths=model_paths, device='cuda')
    # Manual force calculation and variance computation...
    """)
    
    print("\n✨ New backend-agnostic code:")
    print("""
    # NEW CODE:
    from forge.calculators import create_ensemble_calculator
    
    calc = create_ensemble_calculator(
        model_paths=model_paths, 
        backend='auto',  # or 'mace', 'allegro'
        device='cuda'
    )
    
    # Unified interface works with any backend
    all_forces = calc.forces_all(atoms)
    variance = calc.calculate_normalized_force_variance(all_forces)
    mean_forces = calc.get_mean_forces(atoms)
    """)
    
    print("\n🔄 Old workflow code:")
    print("""
    # OLD CODE:
    run_adversarial_attacks(
        # ... parameters
    )
    """)
    
    print("\n✨ New workflow code:")
    print("""
    # NEW CODE:
    run_adversarial_attacks(
        # ... same parameters
        backend='auto'  # Add this line (or omit for default)
    )
    """)

def main():
    """Run practical usage examples."""
    print("=== Practical Usage Examples for Calculator Backend System ===")
    print("This script shows how to use the new backend system in practice\n")
    
    # Show available backends
    available_backends = get_supported_backends()
    print(f"Available backends in this environment: {available_backends}")
    
    if not available_backends:
        print("\n❌ No backends available!")
        print("Please install at least one backend:")
        print("  • MACE: pip install mace-torch")
        print("  • Allegro: pip install nequip")
        return
    
    # Run examples
    example_with_mace_models()
    example_with_allegro_models()
    example_force_calculation_comparison()
    example_auto_detection()
    example_migration_from_old_code()
    
    print("\n=== Summary ===")
    print("✅ Backend system provides unified interface for MACE and Allegro")
    print("✅ Automatic backend detection from file extensions")
    print("✅ Backward compatible with existing code")
    print("⚠ Current limitation: Autograd optimization requires MACE backend")
    print("💡 Recommendation: Use appropriate backend for your model files")
    
    print(f"\n📖 For more information:")
    print(f"  • Documentation: forge/docs/calculator_backends.md")
    print(f"  • Performance tests: forge/examples/benchmark_backend_performance.py")
    print(f"  • Consistency tests: forge/examples/test_workflow_consistency.py")

if __name__ == "__main__":
    main() 