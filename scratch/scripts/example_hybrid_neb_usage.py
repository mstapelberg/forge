#!/usr/bin/env python3
"""
Example usage of the Hybrid NEB Workflow.

This script demonstrates how to use the HybridNEBWorkflow class
with different configurations for composition generation, optimization,
and NEB calculations.

Confidence: 9/10
"""

import numpy as np
from pathlib import Path
from run_hybrid_neb_workflow import HybridNEBWorkflow

def example_basic_workflow():
    """Basic example of the full workflow."""
    print("=== Basic Hybrid NEB Workflow Example ===")
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "../data/pel_het_search/example_basic_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Example existing compositions
    existing_compositions = [
        {'V': 0.85, 'Cr': 0.05, 'Ti': 0.05, 'W': 0.03, 'Zr': 0.02},
        {'V': 0.80, 'Cr': 0.08, 'Ti': 0.06, 'W': 0.04, 'Zr': 0.02},
    ]
    
    # Initialize workflow with unified calculator
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        output_dir=output_dir,
        calculator_type=None,  # Auto-detect based on model file
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # Run full workflow with minimal settings for quick testing
    results = workflow.run_full_workflow(
        existing_compositions=existing_compositions,
        n_new_compositions=1,
        elements=['V', 'Cr', 'Ti', 'W', 'Zr'],
        crystal_type='bcc',
        dimensions=[4, 4, 4],  # Small supercell for quick testing
        lattice_constant=3.01,
        temperature=873.15,
        n_steps=1000,  # Fewer steps for quick testing
        n_nearest=4,
        n_next_nearest=4,
        save_plots=True
    )
    
    print(f"Basic workflow completed!")
    print(f"Results saved to: {output_dir}")


def example_composition_generation():
    """Example focusing on composition generation."""
    print("=== Composition Generation Example ===")
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "../data/pel_het_search/example_composition_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    
    # More diverse existing compositions
    existing_compositions = [
        {'V': 0.90, 'Cr': 0.05, 'Ti': 0.03, 'W': 0.01, 'Zr': 0.01},
        {'V': 0.70, 'Cr': 0.15, 'Ti': 0.10, 'W': 0.03, 'Zr': 0.02},
        {'V': 0.60, 'Cr': 0.20, 'Ti': 0.15, 'W': 0.03, 'Zr': 0.02},
        {'V': 0.50, 'Cr': 0.25, 'Ti': 0.20, 'W': 0.03, 'Zr': 0.02},
    ]
    
    # Initialize workflow with unified calculator
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=123,
        output_dir=output_dir,
        calculator_type=None,  # Auto-detect based on model file
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # Generate compositions with specific constraints
    constraints = {
        'V': (0.4, 0.9),    # V should be 40-90%
        'Cr': (0.05, 0.3),  # Cr should be 5-30%
        'Ti': (0.02, 0.25), # Ti should be 2-25%
        'W': (0.01, 0.1),   # W should be 1-10%
        'Zr': (0.001, 0.05) # Zr should be 0.1-5%
    }
    
    new_compositions = workflow.generate_compositions(
        existing_compositions=existing_compositions,
        n_new_compositions=5,
        elements=['V', 'Cr', 'Ti', 'W', 'Zr'],
        constraints=constraints,
        balance_element='V'
    )
    
    print(f"Generated {len(new_compositions)} new compositions:")
    for i, comp in enumerate(new_compositions):
        print(f"  Composition {i+1}: {comp}")


def example_optimization_only():
    """Example focusing on structure optimization."""
    print("=== Structure Optimization Example ===")
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "../data/pel_het_search/example_optimization_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize workflow with unified calculator
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=456,
        output_dir=output_dir,
        calculator_type=None,  # Auto-detect based on model file
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # Create a specific composition
    composition = {'V': 0.80, 'Cr': 0.10, 'Ti': 0.06, 'W': 0.03, 'Zr': 0.01}
    
    # Create initial structure
    initial_structures = workflow.create_initial_structures(
        compositions=[composition],
        crystal_type='bcc',
        dimensions=[6, 6, 6],
        lattice_constant=3.01,
        balance_element='V'
    )
    
    # Optimize with different temperatures
    temperatures = [300, 600, 900]  # K
    
    for temp in temperatures:
        print(f"Optimizing at {temp}K...")
        optimized_structures = workflow.optimize_with_mcmc(
            structures=initial_structures,
            temperature=temp + 273.15,  # Convert to Kelvin
            n_steps=2000,
            convergence_window=1000,
            energy_threshold=0.0002
        )
        
        # Save optimized structure
        for i, atoms in enumerate(optimized_structures):
            write(f"{output_dir}/optimized_{temp}K_structure_{i}.xyz", atoms)


def example_neb_only():
    """Example focusing on NEB calculations."""
    print("=== NEB Calculations Example ===")
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "../data/pel_het_search/example_neb_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    
    # Initialize workflow with unified calculator
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=789,
        output_dir=output_dir,
        calculator_type=None,  # Auto-detect based on model file
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # Load an existing structure (you would replace this with your structure)
    # For this example, we'll create a simple structure
    from ase.build import bulk
    atoms = bulk('V', 'bcc', a=3.01).repeat((4, 4, 4))
    
    # Run NEB calculations
    neb_results = workflow.run_neb_calculations(
        structures=[atoms],
        n_nearest=2,
        n_next_nearest=2,
        num_images=5,
        neb_method="dyneb",
        climb=True,
        relax_fmax=0.01,
        relax_steps=50,  # Fewer steps for quick testing
        neb_fmax=0.01,
        neb_steps=100,   # Fewer steps for quick testing
        save_xyz=True,
        verbose=1
    )
    
    # Analyze results
    analysis_results = workflow.analyze_results(
        save_plots=True,
        plot_barriers=True,
        plot_compositions=False  # No compositions to plot
    )
    
    print(f"NEB calculations completed!")
    print(f"Results saved to: {output_dir}")


def example_custom_workflow():
    """Example of a custom workflow with specific requirements."""
    print("=== Custom Workflow Example ===")
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "../data/pel_het_search/example_custom_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    
    # Initialize workflow with unified calculator
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=999,
        output_dir=output_dir,
        calculator_type=None,  # Auto-detect based on model file
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # Step 1: Generate compositions with specific focus on high-Cr alloys
    existing_compositions = [
        {'V': 0.85, 'Cr': 0.10, 'Ti': 0.03, 'W': 0.01, 'Zr': 0.01},
        {'V': 0.75, 'Cr': 0.20, 'Ti': 0.03, 'W': 0.01, 'Zr': 0.01},
    ]
    
    # Focus on high-Cr compositions
    constraints = {
        'V': (0.6, 0.9),
        'Cr': (0.08, 0.35),  # Higher Cr content
        'Ti': (0.01, 0.1),
        'W': (0.005, 0.05),
        'Zr': (0.001, 0.02)
    }
    
    new_compositions = workflow.generate_compositions(
        existing_compositions=existing_compositions,
        n_new_compositions=2,
        constraints=constraints
    )
    
    # Step 2: Create structures
    initial_structures = workflow.create_initial_structures(
        compositions=new_compositions,
        crystal_type='bcc',
        dimensions=[5, 5, 5],
        lattice_constant=3.01
    )
    
    # Step 3: Optimize at high temperature for better mixing
    optimized_structures = workflow.optimize_with_mcmc(
        structures=initial_structures,
        temperature=1073.15,  # 800°C
        n_steps=5000,
        convergence_window=1000,
        energy_threshold=0.0002
    )
    
    # Step 4: Run NEB with more sampling
    neb_results = workflow.run_neb_calculations(
        structures=optimized_structures,
        n_nearest=3,
        n_next_nearest=3,
        num_images=7,
        neb_method="dyneb",
        climb=True
    )
    
    # Step 5: Analyze with custom settings
    analysis_results = workflow.analyze_results(
        save_plots=True,
        plot_barriers=True,
        plot_compositions=True
    )
    
    print(f"Custom workflow completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    import torch
    from ase.io import write
    
    print("Hybrid NEB Workflow Examples")
    print("=" * 50)
    
    # Run examples (uncomment the ones you want to try)
    
    example_basic_workflow()
    # example_composition_generation()
    # example_optimization_only()
    # example_neb_only()
    # example_custom_workflow()
    
    print("\nTo run examples, uncomment the desired function calls above.")
    print("Make sure you have the required model file and dependencies installed.") 