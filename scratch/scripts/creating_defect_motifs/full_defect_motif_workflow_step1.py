#!/usr/bin/env python3
"""
Step 1: Full Defect Motif Workflow - Generate and Attack.

This script performs the first step of the complete workflow:
1. Generate defect structures for target compositions
2. Run adversarial attacks on ALL generated structures using Allegro
3. Calculate variance for each structure in the trajectory using the ensemble
4. Save complete trajectory XYZ files for later processing

Usage:
    python full_defect_motif_workflow_step1.py --compositions compositions.json --model-paths model1.nequip.zip model2.nequip.zip
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.core.defect_motifs import generate_defect_structures
from forge.calculators.factory import create_ensemble_calculator

# Import the defect adversarial attack function
from defect_adversarial_attack import run_defect_adversarial_attacks


def load_compositions_from_file(filepath: str) -> List[Dict[str, float]]:
    """
    Load target compositions from a JSON file.
    
    Args:
        filepath: Path to JSON file containing compositions
        
    Returns:
        List of composition dictionaries
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        ValueError: If the compositions are not in the expected format
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Composition file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle different possible formats
    if isinstance(data, list):
        compositions = data
    elif isinstance(data, dict) and 'compositions' in data:
        compositions = data['compositions']
    else:
        raise ValueError("JSON file must contain a list of compositions or a dict with 'compositions' key")
    
    # Validate compositions
    for i, comp in enumerate(compositions):
        if not isinstance(comp, dict):
            raise ValueError(f"Composition {i} must be a dictionary")
        if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in comp.items()):
            raise ValueError(f"Composition {i} must have string keys and numeric values")
    
    return compositions


def run_step1_workflow(
    compositions: List[Dict[str, float]],
    model_paths: List[str],
    output_dir: str,
    generation: int = 10,
    n_iterations: int = 200,
    learning_rate: float = 0.01,
    temperature: float = 1000,
    include_motifs: Optional[List[str]] = None,
    exclude_motifs: Optional[List[str]] = None,
    custom_motif_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    interstitial_tolerance: float = 0.5,
    species_to_type_name: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    debug: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Step 1 of the defect motif workflow: generate and attack.
    
    Args:
        compositions: List of composition dictionaries
        model_paths: List of paths to Allegro model files
        output_dir: Base output directory for all results
        # Step 1 now runs adversarial attacks on ALL generated structures
        generation: Generation tag for new structures
        # Step 1 now saves entire trajectories, selection happens in Step 2
        n_iterations: Number of optimization steps for adversarial attacks
        learning_rate: Optimizer learning rate
        temperature: Temperature for Boltzmann weighting
        include_motifs: List of motif types to include
        exclude_motifs: List of motif types to exclude
        custom_motif_path: Path to custom motif templates
        random_seed: Seed for random number generator
        interstitial_tolerance: Distance tolerance for identifying interstitials
        species_to_type_name: Species mapping for Allegro
        device: Compute device
        debug: Enable debug output
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing workflow statistics and results
    """
    if verbose:
        print("=== Step 1: Defect Motif Generation and Adversarial Attacks ===")
        print(f"Compositions: {len(compositions)}")
        print(f"Allegro models: {len(model_paths)}")
        print(f"Output directory: {output_dir}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    aa_output_dir = os.path.join(output_dir, "adversarial_attacks")
    trajectories_dir = os.path.join(output_dir, "trajectories")
    
    # Set default exclude motifs if none provided
    if exclude_motifs is None:
        exclude_motifs = ['sia', 'di-sia', 'di-SIA', 'di_sia', 'short_range']  # Exclude problematic interstitial motifs
        if verbose:
            print(f"Using default exclude motifs: {exclude_motifs}")
    
    # Step 1: Run adversarial attacks
    if verbose:
        print("\n--- Step 1.1: Running Adversarial Attacks with Allegro ---")
    
    trajectories = run_defect_adversarial_attacks(
        compositions=compositions,
        model_paths=model_paths,
        top_n=None,  # Run attacks on ALL generated structures
        generation=generation,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        temperature=temperature,
        include_motifs=include_motifs,
        exclude_motifs=exclude_motifs,
        custom_motif_path=custom_motif_path,
        random_seed=random_seed,
        interstitial_tolerance=interstitial_tolerance,
        backend='allegro',  # Force Allegro backend for Step 1
        species_to_type_name=species_to_type_name,
        device=device,
        debug=debug,
        output_dir=aa_output_dir,
        save_output=False,  # We want the trajectories returned
        select_n_from_trajectory=None  # Save entire trajectories
    )
    
    if not trajectories:
        raise RuntimeError("No trajectories generated from adversarial attacks")
    
    if verbose:
        print(f"Generated {len(trajectories)} trajectories")
    
    # Step 2: Save trajectories as XYZ files
    if verbose:
        print("\n--- Step 1.2: Saving Trajectory Files ---")
    
    os.makedirs(trajectories_dir, exist_ok=True)
    saved_files_count = 0
    
    for trajectory_id, trajectory in trajectories.items():
        if verbose:
            print(f"  Saving trajectory {trajectory_id}: {len(trajectory)} structures")
        
        # Save each trajectory as a separate XYZ file
        filename = os.path.join(trajectories_dir, f"trajectory_{trajectory_id}.xyz")
        
        try:
            # Remove calculators from atoms to avoid serialization issues
            for atoms in trajectory:
                atoms.calc = None
            
            # Save as XYZ file
            from ase.io import write
            write(filename, trajectory, format='extxyz')
            saved_files_count += 1
            
            if verbose:
                print(f"    Saved: {filename}")
                
        except Exception as e:
            print(f"    ERROR: Failed to save trajectory {trajectory_id}: {e}")
    
    # Step 3: Save metadata for Step 2
    if verbose:
        print("\n--- Step 1.3: Saving Metadata for Step 2 ---")
    
    metadata = {
        'step1_completed': True,
        'input_compositions': len(compositions),
        'input_models': len(model_paths),
        'trajectories_generated': len(trajectories),
        'trajectories_saved': saved_files_count,
        'generation': generation,
        'top_n': None,  # Step 1 runs attacks on all structures
        'n_select_from_trajectory': None,  # Step 1 saves entire trajectories
        'include_motifs': include_motifs,
        'exclude_motifs': exclude_motifs,
        'random_seed': random_seed,
        'output_directories': {
            'adversarial_attacks': aa_output_dir,
            'trajectories': trajectories_dir
        }
    }
    
    metadata_path = os.path.join(output_dir, "step1_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"Saved metadata to: {metadata_path}")
    
    # Compile final statistics
    final_stats = {
        'step1_completed': True,
        'input_compositions': len(compositions),
        'input_models': len(model_paths),
        'trajectories_generated': len(trajectories),
        'trajectories_saved': saved_files_count,
        'output_directories': {
            'adversarial_attacks': aa_output_dir,
            'trajectories': trajectories_dir
        }
    }
    
    # Save final statistics
    stats_path = os.path.join(output_dir, "step1_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    if verbose:
        print(f"\n=== Step 1 Complete ===")
        print(f"  Trajectories generated: {len(trajectories)}")
        print(f"  Trajectories saved: {saved_files_count}")
        print(f"  Output directory: {output_dir}")
        print(f"  Statistics saved to: {stats_path}")
        print(f"\nNext: Run Step 2 with MACE environment:")
        print(f"  python full_defect_motif_workflow_step2.py --input-dir {output_dir} --mace-model-paths your_mace_model1.model your_mace_model2.model")
    
    return final_stats


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Step 1: Generate defect motifs and run adversarial attacks with Allegro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Allegro models
  python full_defect_motif_workflow_step1.py --compositions compositions.json \\
    --model-paths model1.nequip.zip model2.nequip.zip --output-dir step1_output
  
  # With specific motifs and species mapping
  python full_defect_motif_workflow_step1.py --compositions compositions.json \\
    --model-paths model1.nequip.zip model2.nequip.zip \\
    --species-mapping '{"Ti": "Ti", "V": "V", "Cr": "Cr"}' \\
    --include-motifs vacancy surface_100 --output-dir step1_output
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--compositions', '-c',
        type=str,
        required=True,
        help='Path to JSON file containing target compositions'
    )
    
    parser.add_argument(
        '--model-paths',
        nargs='+',
        required=True,
        help='Paths to Allegro model files (.nequip.zip files)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='step1_output',
        help='Output directory for Step 1 results (default: step1_output)'
    )
    
    # Removed top-n argument - Step 1 now runs attacks on all generated structures
    
    parser.add_argument(
        '--generation',
        type=int,
        default=10,
        help='Generation tag for new structures (default: 10)'
    )
    
    # Removed n-select-from-trajectory argument - Step 1 now saves entire trajectories
    
    parser.add_argument(
        '--n-iterations',
        type=int,
        default=200,
        help='Number of optimization steps for adversarial attacks (default: 200)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Optimizer learning rate (default: 0.01)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=1000,
        help='Temperature for Boltzmann weighting (default: 1000)'
    )
    
    # Calculator parameters
    parser.add_argument(
        '--species-mapping',
        type=str,
        help='JSON string or file path for species to type mapping (required for Allegro)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Compute device (auto-detect if not specified)'
    )
    
    # Defect generation parameters
    parser.add_argument(
        '--include-motifs',
        nargs='+',
        help='Only include specific motif types'
    )
    
    parser.add_argument(
        '--exclude-motifs',
        nargs='+',
        help='Exclude specific motif types'
    )
    
    parser.add_argument(
        '--custom-motif-path',
        type=str,
        help='Path to custom motif templates'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible results (default: 42)'
    )
    
    parser.add_argument(
        '--interstitial-tolerance',
        type=float,
        default=0.5,
        help='Distance tolerance for identifying interstitials (default: 0.5 Å)'
    )
    
    # Other options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Load compositions
        compositions = load_compositions_from_file(args.compositions)
        
        # Parse species mapping if provided
        species_to_type_name = None
        if args.species_mapping:
            try:
                # Try to parse as JSON string first
                species_to_type_name = json.loads(args.species_mapping)
            except json.JSONDecodeError:
                # If that fails, try to load from file
                try:
                    with open(args.species_mapping, 'r') as f:
                        species_to_type_name = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not parse species mapping: {e}")
                    print("Using default species mapping: {'Ti': 'Ti', 'V': 'V', 'Cr': 'Cr', 'Zr': 'Zr', 'W': 'W'}")
                    species_to_type_name = {'Ti': 'Ti', 'V': 'V', 'Cr': 'Cr', 'Zr': 'Zr', 'W': 'W'}
        
        # Run Step 1 workflow
        stats = run_step1_workflow(
            compositions=compositions,
            model_paths=args.model_paths,
            output_dir=args.output_dir,
            # top_n removed - Step 1 runs attacks on all structures
            generation=args.generation,
            n_iterations=args.n_iterations,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            include_motifs=args.include_motifs,
            exclude_motifs=args.exclude_motifs,
            custom_motif_path=args.custom_motif_path,
            random_seed=args.random_seed,
            interstitial_tolerance=args.interstitial_tolerance,
            species_to_type_name=species_to_type_name,
            device=args.device,
            debug=args.debug,
            verbose=not args.quiet
        )
        
        print(f"\n✅ Step 1 completed successfully!")
        print(f"   Trajectories generated: {stats['trajectories_generated']}")
        print(f"   Trajectories saved: {stats['trajectories_saved']}")
        print(f"   Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 