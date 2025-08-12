#!/usr/bin/env python3
"""
Example script demonstrating the full defect motif workflow.

This script shows how to use the complete workflow with both MACE and Allegro backends.
"""

import json
import sys
from pathlib import Path

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from full_defect_motif_workflow import run_full_defect_motif_workflow


def create_example_compositions():
    """Create example compositions for testing."""
    compositions = [
        {'V': 0.75, 'Cr': 0.25},  # Binary alloy
        {'V': 0.50, 'Cr': 0.30, 'Ti': 0.20},  # Ternary alloy
        {'V': 1.0},  # Pure vanadium
    ]
    return compositions


def example_mace_workflow():
    """Example using MACE models."""
    print("=== MACE Workflow Example ===")
    
    # Example compositions
    compositions = create_example_compositions()
    
    # MACE model paths (update these to your actual model paths)
    model_paths = [
        '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model',
        '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_1_pr_stagetwo.model',
        '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_2_pr_stagetwo.model',
    ]
    
    print(f"Compositions: {compositions}")
    print(f"Model paths: {model_paths}")
    print("Using MACE models with auto-detection")
    
    # Run the workflow
    stats = run_full_defect_motif_workflow(
        # Input parameters
        compositions=compositions,
        model_paths=model_paths,
        output_dir='example_mace_workflow_output',
        
        # Adversarial attack parameters
        top_n=5,  # Select top 5 structures
        generation=9,
        n_select_from_trajectory=10,  # Select 10 structures from each trajectory
        n_iterations=50,  # Shorter for example
        learning_rate=0.01,
        temperature=1000,
        
        # Defect-specific parameters
        include_motifs=['sia', 'di-sia', 'vacancy'],  # Focus on interstitial defects
        random_seed=42,
        interstitial_tolerance=0.5,
        
        # Calculator parameters
        backend='mace',  # Explicitly specify MACE backend
        device='cpu',  # Use CPU for example
        
        # VASP parameters
        vasp_profile_name='static',
        hpc_profile_name='PSFC-GPU',
        
        # Output parameters
        debug=False,
        verbose=True
    )
    
    print(f"\n=== MACE Workflow Results ===")
    print(f"  Trajectories generated: {stats['trajectories_generated']}")
    print(f"  Structures selected: {stats['structures_selected']}")
    print(f"  VASP jobs created: {stats['vasp_jobs_created']}")
    print(f"  Output directory: example_mace_workflow_output")


def example_allegro_workflow():
    """Example using Allegro models."""
    print("\n=== Allegro Workflow Example ===")
    
    # Example compositions
    compositions = create_example_compositions()
    
    # Allegro model paths (update these to your actual model paths)
    model_paths = [
        '../../data/potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.zip',
        '../../data/potentials/allegro/exploit_rmax5.50_lmax2_layers2_mlp256_seed42.nequip.zip',
        '../../data/potentials/allegro/exploit_rmax5.50_lmax2_layers2_mlp384_seed42.nequip.zip',
    ]
    
    # Species mapping for Allegro
    species_to_type_name = {
        'Ti': 'Ti',
        'V': 'V', 
        'Cr': 'Cr',
        'Zr': 'Zr',
        'W': 'W'
    }
    
    print(f"Compositions: {compositions}")
    print(f"Model paths: {model_paths}")
    print(f"Species mapping: {species_to_type_name}")
    print("Using Allegro models with autograd optimization")
    
    # Run the workflow
    stats = run_full_defect_motif_workflow(
        # Input parameters
        compositions=compositions,
        model_paths=model_paths,
        output_dir='example_allegro_workflow_output',
        
        # Adversarial attack parameters
        top_n=5,  # Select top 5 structures
        generation=9,
        n_select_from_trajectory=10,  # Select 10 structures from each trajectory
        n_iterations=50,  # Shorter for example
        learning_rate=0.01,
        temperature=1000,
        
        # Defect-specific parameters
        include_motifs=['sia', 'di-sia', 'vacancy'],  # Focus on interstitial defects
        random_seed=42,
        interstitial_tolerance=0.5,
        
        # Calculator parameters
        backend='allegro',  # Explicitly specify Allegro backend
        species_to_type_name=species_to_type_name,  # Chemical symbols mapping for Allegro
        device='cpu',  # Use CPU for example
        
        # VASP parameters
        vasp_profile_name='static',
        hpc_profile_name='PSFC-GPU',
        
        # Output parameters
        debug=False,
        verbose=True
    )
    
    print(f"\n=== Allegro Workflow Results ===")
    print(f"  Trajectories generated: {stats['trajectories_generated']}")
    print(f"  Structures selected: {stats['structures_selected']}")
    print(f"  VASP jobs created: {stats['vasp_jobs_created']}")
    print(f"  Output directory: example_allegro_workflow_output")


def example_auto_detection():
    """Example using auto-detection of backend."""
    print("\n=== Auto-Detection Workflow Example ===")
    
    # Example compositions
    compositions = create_example_compositions()
    
    # Model paths (system will auto-detect backend based on file extension)
    model_paths = [
        '../../data/potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.zip',
        '../../data/potentials/allegro/exploit_rmax5.50_lmax2_layers2_mlp256_seed42.nequip.zip',
    ]
    
    # Species mapping for Allegro (needed even with auto-detection)
    species_to_type_name = {
        'Ti': 'Ti',
        'V': 'V', 
        'Cr': 'Cr',
        'Zr': 'Zr',
        'W': 'W'
    }
    
    print(f"Compositions: {compositions}")
    print(f"Model paths: {model_paths}")
    print("Using auto-detection of backend")
    
    # Run the workflow
    stats = run_full_defect_motif_workflow(
        # Input parameters
        compositions=compositions,
        model_paths=model_paths,
        output_dir='example_auto_workflow_output',
        
        # Adversarial attack parameters
        top_n=3,  # Select top 3 structures
        generation=9,
        n_select_from_trajectory=5,  # Select 5 structures from each trajectory
        n_iterations=30,  # Very short for example
        learning_rate=0.01,
        temperature=1000,
        
        # Defect-specific parameters
        include_motifs=['sia', 'vacancy'],  # Focus on simpler defects
        random_seed=42,
        interstitial_tolerance=0.5,
        
        # Calculator parameters
        backend='auto',  # Auto-detect backend
        species_to_type_name=species_to_type_name,  # Still needed for Allegro
        device='cpu',  # Use CPU for example
        
        # VASP parameters
        vasp_profile_name='static',
        hpc_profile_name='PSFC-GPU',
        
        # Output parameters
        debug=False,
        verbose=True
    )
    
    print(f"\n=== Auto-Detection Workflow Results ===")
    print(f"  Trajectories generated: {stats['trajectories_generated']}")
    print(f"  Structures selected: {stats['structures_selected']}")
    print(f"  VASP jobs created: {stats['vasp_jobs_created']}")
    print(f"  Output directory: example_auto_workflow_output")


def main():
    """Run example workflows."""
    print("Full Defect Motif Workflow Examples")
    print("=" * 50)
    
    # Check if model files exist before running examples
    print("Note: Update model paths in this script to point to your actual model files.")
    print("The examples below will fail if the model files don't exist.\n")
    
    # Run examples (uncomment the ones you want to try)
    
    # example_mace_workflow()
    # example_allegro_workflow()
    example_auto_detection()
    
    print("\nTo run examples, uncomment the desired function calls above.")
    print("Make sure you have the required model files and dependencies installed.")
    print("\nYou can also run the workflow directly from command line:")
    print("python full_defect_motif_workflow.py --compositions compositions.json --model-paths model1.model model2.model")


if __name__ == "__main__":
    main() 