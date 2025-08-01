#!/usr/bin/env python3
"""
Example usage of the HybridNEBWorkflow module.

This script demonstrates how to use the HybridNEBWorkflow class
from the forge.workflows.hybrid_neb module.

Confidence: 9/10
"""

import torch
from forge.workflows.hybrid_neb import HybridNEBWorkflow

def example_basic_usage():
    """Basic example of using the HybridNEBWorkflow module."""
    print("=== Basic HybridNEBWorkflow Usage ===")
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "../data/pel_het_search/module_example_results"
    
    # Example existing compositions
    existing_compositions = [
        {'V': 0.85, 'Cr': 0.05, 'Ti': 0.05, 'W': 0.03, 'Zr': 0.02},
        {'V': 0.80, 'Cr': 0.08, 'Ti': 0.06, 'W': 0.04, 'Zr': 0.02},
    ]
    
    # Initialize workflow
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        output_dir=output_dir,
        calculator_type=None,  # Auto-detect
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # Run full workflow
    results = workflow.run_full_workflow(
        existing_compositions=existing_compositions,
        n_new_compositions=1,
        elements=['V', 'Cr', 'Ti', 'W', 'Zr'],
        crystal_type='bcc',
        dimensions=[4, 4, 4],  # Small for testing
        lattice_constant=3.01,
        temperature=873.15,
        n_steps=100,  # Few cycles for testing
        n_nearest=2,
        n_next_nearest=2,
        save_plots=True
    )
    
    print(f"Workflow completed! Results saved to: {output_dir}")
    return results

def example_step_by_step():
    """Example showing step-by-step usage of the workflow."""
    print("=== Step-by-Step Workflow Usage ===")
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "../data/pel_het_search/step_by_step_results"
    
    # Initialize workflow
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=123,
        output_dir=output_dir,
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # Step 1: Generate compositions with different constraint types
    
    # Example 1: Range constraints (recommended for exploration)
    range_constraints = {
        'V': (0.8, 0.9),     # V between 80-90%
        'Cr': (0.05, 0.15),  # Cr between 5-15%
        'Ti': (0.01, 0.1),   # Ti between 1-10%
        'W': (0.01, 0.05),   # W between 1-5%
        'Zr': (0.001, 0.02)  # Zr between 0.1-2%
    }
    
    # Example 2: Mixed constraints (ranges + specific values)
    mixed_constraints = {
        'V': (0.85, 0.95),   # V between 85-95%
        'Cr': (0.025, 0.025), # Exactly 2.5% Cr
        'Ti': (0.01, 0.05),  # Ti between 1-5%
        'W': (0.005, 0.005), # Exactly 0.5% W
        'Zr': (0.001, 0.01)  # Zr between 0.1-1%
    }
    
    # Example 3: Specific value constraints (for targeted compositions)
    specific_constraints = {
        'V': (0.85, 0.85),   # Exactly 85% V
        'Cr': (0.10, 0.10),  # Exactly 10% Cr
        'Ti': (0.03, 0.03),  # Exactly 3% Ti
        'W': (0.01, 0.01),   # Exactly 1% W
        'Zr': (0.01, 0.01)   # Exactly 1% Zr
    }
    
    existing_compositions = [{'V': 0.85, 'Cr': 0.10, 'Ti': 0.03, 'W': 0.01, 'Zr': 0.01}]
    
    # Use range constraints for this example
    compositions = workflow.generate_compositions(
        existing_compositions=existing_compositions,
        n_new_compositions=2,
        constraints=range_constraints  # Can also use mixed_constraints or specific_constraints
    )
    
    # Step 2: Create structures
    structures = workflow.create_initial_structures(
        compositions=compositions,
        crystal_type='bcc',
        dimensions=[3, 3, 3],  # Small for testing
        lattice_constant=3.01
    )
    
    # Step 3: Optimize with hybrid MCMC-MD
    optimized_structures = workflow.optimize_with_hybrid_mcmc(
        structures=structures,
        temperature=873.15,
        n_steps=50,  # Few cycles for testing
        md_steps_per_cycle=20,
        mc_steps_per_cycle=10,
        final_cell_relax=True
    )
    
    # Step 4: Run NEB calculations
    neb_results = workflow.run_neb_calculations(
        structures=optimized_structures,
        n_nearest=1,
        n_next_nearest=1,
        num_images=3,  # Few images for testing
        neb_steps=50   # Few steps for testing
    )
    
    # Step 5: Analyze results
    analysis = workflow.analyze_results(save_plots=True)
    
    print(f"Step-by-step workflow completed! Results saved to: {output_dir}")
    return analysis

def example_custom_optimization():
    """Example showing custom optimization parameters."""
    print("=== Custom Optimization Example ===")
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "../data/pel_het_search/custom_optimization_results"
    
    # Initialize workflow
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=456,
        output_dir=output_dir,
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # Create a simple structure
    from ase.build import bulk
    atoms = bulk('V', 'bcc', a=3.01).repeat((3, 3, 3))
    
    # Custom optimization with different parameters
    optimized_structures = workflow.optimize_with_hybrid_mcmc(
        structures=[atoms],
        temperature=1073.15,  # Higher temperature
        md_temperature=973.15,  # Different MD temperature
        n_steps=100,
        md_steps_per_cycle=50,
        mc_steps_per_cycle=25,
        convergence_window=500,
        energy_threshold=0.001,  # Stricter convergence
        final_cell_relax=True,
        md_timestep=0.5,  # Smaller timestep
        md_thermostat='langevin',
        friction=0.01  # Lower friction
    )
    
    print(f"Custom optimization completed! Results saved to: {output_dir}")
    return optimized_structures

if __name__ == "__main__":
    print("HybridNEBWorkflow Module Examples")
    print("=" * 50)
    
    # Run examples (uncomment the ones you want to try)
    
    example_basic_usage()
    example_step_by_step()
    example_custom_optimization()
    
    print("\nTo run examples, uncomment the desired function calls above.")
    print("Make sure you have the required model file and dependencies installed.")
    print("\nYou can also import and use the workflow directly:")
    print("from forge.workflows.hybrid_neb import HybridNEBWorkflow") 