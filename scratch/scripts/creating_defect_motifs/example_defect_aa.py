#!/usr/bin/env python3
"""
Example script demonstrating defect-aware adversarial attacks.

This script shows how to:
1. Generate defect structures with interstitials
2. Run adversarial attacks while fixing interstitial positions
3. Select N structures from each trajectory
"""

import json
import sys
from pathlib import Path

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scratch.scripts.defect_adversarial_attack import run_defect_adversarial_attacks


def create_example_compositions():
    """Create example compositions for testing."""
    compositions = [
        {'V': 0.75, 'Cr': 0.25},  # Binary alloy
        {'V': 0.50, 'Cr': 0.30, 'Ti': 0.20},  # Ternary alloy
        {'V': 1.0},  # Pure vanadium
    ]
    return compositions


def main():
    """Run example defect adversarial attacks."""
    
    # Example compositions
    compositions = create_example_compositions()
    
    # Model paths (update these to your actual model paths)
    # For MACE models:
    # model_paths = [
    #     '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model',
    #     '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_1_pr_stagetwo.model',
    #     '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_2_pr_stagetwo.model',
    # ]
    
    # For Allegro models:
    model_paths = [
        '../../data/potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.zip',
        '../../data/potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp256_seed42.nequip.zip',
        '../../data/potentials/allegro/exploit_rmax5.75_lmax2_layers2_mlp256_seed42.nequip.zip',
    ]
    
    print("=== Defect Adversarial Attack Example ===")
    print(f"Compositions: {compositions}")
    print(f"Model paths: {model_paths}")
    print("Using Allegro models with autograd optimization")
    
    # Run the workflow
    trajectories = run_defect_adversarial_attacks(
        # Input parameters
        compositions=compositions,
        model_paths=model_paths,
        top_n=5,  # Select top 5 structures
        generation=9,
        
        # Optimization parameters
        n_iterations=100,  # Shorter for example
        learning_rate=0.01,
        temperature=1000,
        include_probability=False,
        
        # Defect-specific parameters
        interstitial_tolerance=0.5,
        include_motifs=['sia', 'di-sia', 'vacancy'],  # Focus on interstitial defects
        random_seed=42,
        select_n_from_trajectory=10,  # Select 10 structures from each trajectory
        
        # Calculator-specific parameters
        calculator_type='allegro',  # Specify Allegro calculator
        species_to_type_name={'Ti': 'Ti', 'V': 'V', 'Cr': 'Cr', 'Zr': 'Zr', 'W': 'W'},  # Chemical symbols mapping for Allegro
        
        # Output parameters
        output_dir='example_defect_aa_output',
        save_output=True,
        debug=True
    )
    
    print("\n=== Results ===")
    if trajectories is not None:
        print(f"Returned {len(trajectories)} trajectories")
        for structure_id, trajectory in trajectories.items():
            print(f"  Structure {structure_id}: {len(trajectory)} structures in trajectory")
    else:
        print("Trajectories saved to files in 'example_defect_aa_output' directory")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main() 