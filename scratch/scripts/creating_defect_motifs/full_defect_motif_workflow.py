#!/usr/bin/env python3
"""
Full Defect Motif Workflow: Generate, Attack, Select, and Create VASP Jobs.

This script combines the complete workflow:
1. Generate defect structures for target compositions
2. Run adversarial attacks on the structures
3. Use aa_selection to choose diverse structures from trajectories
4. Create VASP jobs for the selected structures

Usage:
    python full_defect_motif_workflow.py --compositions compositions.json --model-paths model1.model model2.model
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
from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase
from forge.calculators.factory import create_ensemble_calculator
from forge.analysis.aa_selection import AAAnalyzer

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


def organize_output_structure(
    base_output_dir: str,
    composition: Dict[str, float],
    config_type: str,
    variant_index: int = 0
) -> str:
    """
    Create an organized output directory structure for VASP jobs.
    
    Args:
        base_output_dir: Base output directory
        composition: Composition dictionary
        config_type: Configuration type (e.g., 'sia_aa', 'vacancy_aa')
        variant_index: Index for multiple stoichiometry variants
        
    Returns:
        Path to the job directory
    """
    # Create composition string (e.g., "V75Cr25" for {'V': 0.75, 'Cr': 0.25})
    comp_str = ""
    for element, fraction in sorted(composition.items()):
        comp_str += f"{element}{int(fraction * 100):02d}"
    
    # Create job directory name
    if variant_index == 0:
        job_dir_name = f"{comp_str}_{config_type}"
    else:
        job_dir_name = f"{comp_str}_{config_type}_var{variant_index}"
    
    job_path = os.path.join(base_output_dir, job_dir_name)
    return job_path


def create_vasp_jobs_for_selected_structures(
    selected_structures: List[Dict[str, Any]],
    output_dir: str,
    vasp_profile_name: str = "static",
    hpc_profile_name: str = "PSFC-GPU",
    auto_kpoints: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Create VASP jobs for selected structures from adversarial attack trajectories.
    
    Args:
        selected_structures: List of dictionaries containing selected structures and metadata
        output_dir: Base output directory for all jobs
        vasp_profile_name: Name of VASP settings profile to use
        hpc_profile_name: Name of HPC profile to use
        auto_kpoints: Whether to automatically determine k-points
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing generation statistics and results
    """
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_structures': len(selected_structures),
        'successful_jobs': 0,
        'failed_jobs': 0,
        'job_directories': [],
        'errors': []
    }
    
    # Process each selected structure
    for i, structure_info in enumerate(selected_structures):
        try:
            # Extract structure and metadata
            atoms = structure_info['atoms']
            target_composition = structure_info['target_composition_input']
            config_type = structure_info['config_type']
            variant_index = structure_info.get('variant_index', 0)
            
            # Create organized output directory
            job_dir = organize_output_structure(
                output_dir,
                target_composition,
                config_type,
                variant_index
            )
            
            # Create VASP job
            prepare_vasp_job_from_ase(
                atoms=atoms,
                vasp_profile_name=vasp_profile_name,
                hpc_profile_name=hpc_profile_name,
                output_dir=job_dir,
                auto_kpoints=auto_kpoints,
                job_name=os.path.basename(job_dir)
            )
            
            # Save structure as XYZ file
            xyz_path = os.path.join(job_dir, "structure.xyz")
            atoms.write(xyz_path)
            
            # Save minimal metadata
            metadata_path = os.path.join(job_dir, "structure_info.json")
            metadata = {
                'target_composition_input': target_composition,
                'config_type': config_type,
                'variant_index': variant_index
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            stats['successful_jobs'] += 1
            stats['job_directories'].append(job_dir)
            
            if verbose:
                print(f"  [{i+1}/{len(selected_structures)}] Created job: {os.path.basename(job_dir)}")
                
        except Exception as e:
            stats['failed_jobs'] += 1
            error_msg = f"Failed to create job for structure {i}: {str(e)}"
            stats['errors'].append(error_msg)
            if verbose:
                print(f"  [{i+1}/{len(selected_structures)}] ERROR: {error_msg}")
    
    # Save overall statistics
    stats_path = os.path.join(output_dir, "generation_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    if verbose:
        print(f"\nVASP job creation complete!")
        print(f"  Successful jobs: {stats['successful_jobs']}")
        print(f"  Failed jobs: {stats['failed_jobs']}")
        print(f"  Output directory: {output_dir}")
        print(f"  Statistics saved to: {stats_path}")
    
    return stats


def run_full_defect_motif_workflow(
    compositions: List[Dict[str, float]],
    model_paths: List[str],
    output_dir: str,
    top_n: int = 10,
    generation: int = 10,
    n_select_from_trajectory: int = 10,
    n_iterations: int = 200,
    learning_rate: float = 0.01,
    temperature: float = 1000,
    include_motifs: Optional[List[str]] = None,
    exclude_motifs: Optional[List[str]] = None,
    custom_motif_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    interstitial_tolerance: float = 0.5,
    backend: Optional[str] = None,
    species_to_type_name: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    vasp_profile_name: str = "static",
    hpc_profile_name: str = "PSFC-GPU",
    auto_kpoints: bool = False,
    debug: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete defect motif workflow: generate, attack, select, and create VASP jobs.
    
    Args:
        compositions: List of composition dictionaries
        model_paths: List of paths to model files
        output_dir: Base output directory for all results
        top_n: Number of top structures to select for adversarial attack
        generation: Generation tag for new structures
        n_select_from_trajectory: Number of structures to select from each trajectory
        n_iterations: Number of optimization steps for adversarial attacks
        learning_rate: Optimizer learning rate
        temperature: Temperature for Boltzmann weighting
        include_motifs: List of motif types to include
        exclude_motifs: List of motif types to exclude
        custom_motif_path: Path to custom motif templates
        random_seed: Seed for random number generator
        interstitial_tolerance: Distance tolerance for identifying interstitials
        backend: Calculator backend ('mace', 'allegro', or 'auto')
        species_to_type_name: Species mapping for Allegro
        device: Compute device
        vasp_profile_name: Name of VASP settings profile to use
        hpc_profile_name: Name of HPC profile to use
        auto_kpoints: Whether to automatically determine k-points
        debug: Enable debug output
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing workflow statistics and results
    """
    if verbose:
        print("=== Full Defect Motif Workflow ===")
        print(f"Compositions: {len(compositions)}")
        print(f"Models: {len(model_paths)}")
        print(f"Output directory: {output_dir}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    aa_output_dir = os.path.join(output_dir, "adversarial_attacks")
    vasp_output_dir = os.path.join(output_dir, "vasp_jobs")
    
    # Step 1: Run adversarial attacks
    if verbose:
        print("\n--- Step 1: Running Adversarial Attacks ---")
    
    # Set default exclude motifs if none provided
    if exclude_motifs is None:
        exclude_motifs = ['sia', 'di-sia', 'di-SIA', 'di_sia', 'short_range']  # Exclude problematic interstitial motifs
        if verbose:
            print(f"Using default exclude motifs: {exclude_motifs}")
    
    trajectories = run_defect_adversarial_attacks(
        compositions=compositions,
        model_paths=model_paths,
        top_n=top_n,
        generation=generation,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        temperature=temperature,
        include_motifs=include_motifs,
        exclude_motifs=exclude_motifs,
        custom_motif_path=custom_motif_path,
        random_seed=random_seed,
        interstitial_tolerance=interstitial_tolerance,
        backend=backend,
        species_to_type_name=species_to_type_name,
        device=device,
        debug=debug,
        output_dir=aa_output_dir,
        save_output=False,  # We want the trajectories returned
        select_n_from_trajectory=n_select_from_trajectory
    )
    
    if not trajectories:
        raise RuntimeError("No trajectories generated from adversarial attacks")
    
    if verbose:
        print(f"Generated {len(trajectories)} trajectories")
    
    # Step 2: Collect all structures from trajectories for aa_selection
    if verbose:
        print("\n--- Step 2: Preparing Structures for Selection ---")
    
    all_structures = []
    structure_metadata = []
    
    for trajectory_id, trajectory in trajectories.items():
        for i, atoms in enumerate(trajectory):
            # Extract metadata from the original structure generation
            # We need to find the original structure info
            original_config_type = atoms.info.get('config_type', 'unknown')
            config_type = f"{original_config_type}_aa"  # Add _aa suffix
            
            # Get target composition from the original structure
            target_composition = atoms.info.get('target_composition_input', {})
            variant_index = atoms.info.get('variant_index', 0)
            
            # Add variance info for selection
            if 'variance' not in atoms.info:
                # Calculate variance if not present
                try:
                    if device is None:
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    calc = create_ensemble_calculator(
                        model_paths=model_paths, 
                        device=device, 
                        backend=backend,
                        species_to_type_name=species_to_type_name
                    )
                    forces = calc.get_forces(atoms)
                    variance = np.var(forces)
                    atoms.info['variance'] = variance
                except Exception as e:
                    if debug:
                        print(f"Warning: Could not calculate variance for structure {trajectory_id}_{i}: {e}")
                    atoms.info['variance'] = 0.0
            
            all_structures.append(atoms)
            structure_metadata.append({
                'trajectory_id': trajectory_id,
                'trajectory_index': i,
                'target_composition_input': target_composition,
                'config_type': config_type,
                'variant_index': variant_index
            })
    
    if verbose:
        print(f"Collected {len(all_structures)} structures from trajectories")
    
    # Step 3: Use aa_selection to select diverse structures
    if verbose:
        print("\n--- Step 3: Selecting Diverse Structures ---")
    
    # Initialize calculator for aa_selection
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    calc = create_ensemble_calculator(
        model_paths=model_paths, 
        device=device, 
        backend=backend,
        species_to_type_name=species_to_type_name
    )
    
    # Use aa_selection to select diverse structures
    # For now, select a reasonable number based on available structures
    n_select = min(len(all_structures), 50)  # Select up to 50 diverse structures
    
    analyzer = AAAnalyzer(all_structures, calc)
    selected_indices, plot_fig = analyzer.select_diverse_structures(
        n_select=n_select,
        plot=True
    )
    
    # Save the selection plot
    if plot_fig:
        plot_path = os.path.join(output_dir, "structure_selection_plot.png")
        plot_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_fig.close()
        if verbose:
            print(f"Selection plot saved to: {plot_path}")
    
    # Create list of selected structures with metadata
    selected_structures = []
    for idx in selected_indices:
        selected_structures.append({
            'atoms': all_structures[idx],
            **structure_metadata[idx]
        })
    
    if verbose:
        print(f"Selected {len(selected_structures)} diverse structures")
    
    # Step 4: Create VASP jobs for selected structures
    if verbose:
        print("\n--- Step 4: Creating VASP Jobs ---")
    
    vasp_stats = create_vasp_jobs_for_selected_structures(
        selected_structures=selected_structures,
        output_dir=vasp_output_dir,
        vasp_profile_name=vasp_profile_name,
        hpc_profile_name=hpc_profile_name,
        auto_kpoints=auto_kpoints,
        verbose=verbose
    )
    
    # Compile final statistics
    final_stats = {
        'workflow_completed': True,
        'input_compositions': len(compositions),
        'input_models': len(model_paths),
        'trajectories_generated': len(trajectories),
        'total_structures_from_trajectories': len(all_structures),
        'structures_selected': len(selected_structures),
        'vasp_jobs_created': vasp_stats['successful_jobs'],
        'vasp_jobs_failed': vasp_stats['failed_jobs'],
        'output_directories': {
            'adversarial_attacks': aa_output_dir,
            'vasp_jobs': vasp_output_dir
        }
    }
    
    # Save final statistics
    stats_path = os.path.join(output_dir, "workflow_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    if verbose:
        print(f"\n=== Workflow Complete ===")
        print(f"  Trajectories generated: {len(trajectories)}")
        print(f"  Structures selected: {len(selected_structures)}")
        print(f"  VASP jobs created: {vasp_stats['successful_jobs']}")
        print(f"  Output directory: {output_dir}")
        print(f"  Statistics saved to: {stats_path}")
    
    return final_stats


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Full defect motif workflow: generate, attack, select, and create VASP jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python full_defect_motif_workflow.py --compositions compositions.json \\
    --model-paths model1.model model2.model --output-dir workflow_output
  
  # With specific motifs and Allegro backend
  python full_defect_motif_workflow.py --compositions compositions.json \\
    --model-paths model1.nequip.zip model2.nequip.zip \\
    --backend allegro --species-mapping '{"Ti": "Ti", "V": "V", "Cr": "Cr"}' \\
    --include-motifs sia di-sia --output-dir workflow_output
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
        help='Paths to model files (MACE or Allegro)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='full_defect_workflow_output',
        help='Output directory for all results (default: full_defect_workflow_output)'
    )
    
    # Adversarial attack parameters
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top structures to select for adversarial attack (default: 10)'
    )
    
    parser.add_argument(
        '--generation',
        type=int,
        default=1,
        help='Generation tag for new structures (default: 1)'
    )
    
    parser.add_argument(
        '--n-select-from-trajectory',
        type=int,
        default=20,
        help='Number of structures to select from each trajectory (default: 20)'
    )
    
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
        '--backend',
        type=str,
        choices=['mace', 'allegro', 'auto'],
        help='Calculator backend type (auto-detect if not specified)'
    )
    
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
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--interstitial-tolerance',
        type=float,
        default=0.5,
        help='Distance tolerance for identifying interstitials (default: 0.5 Å)'
    )
    
    # VASP job parameters
    parser.add_argument(
        '--vasp-profile',
        type=str,
        default='static',
        help='VASP settings profile name (default: static)'
    )
    
    parser.add_argument(
        '--hpc-profile',
        type=str,
        default='PSFC-GPU',
        help='HPC profile name (default: PSFC-GPU)'
    )
    
    parser.add_argument(
        '--auto-kpoints',
        action='store_true',
        help='Automatically determine k-points'
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
        
        # Run the complete workflow
        stats = run_full_defect_motif_workflow(
            compositions=compositions,
            model_paths=args.model_paths,
            output_dir=args.output_dir,
            top_n=args.top_n,
            generation=args.generation,
            n_select_from_trajectory=args.n_select_from_trajectory,
            n_iterations=args.n_iterations,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            include_motifs=args.include_motifs,
            exclude_motifs=args.exclude_motifs,
            custom_motif_path=args.custom_motif_path,
            random_seed=args.random_seed,
            interstitial_tolerance=args.interstitial_tolerance,
            backend=args.backend,
            species_to_type_name=species_to_type_name,
            device=args.device,
            vasp_profile_name=args.vasp_profile,
            hpc_profile_name=args.hpc_profile,
            auto_kpoints=args.auto_kpoints,
            debug=args.debug,
            verbose=not args.quiet
        )
        
        print(f"\n✅ Workflow completed successfully!")
        print(f"   VASP jobs created: {stats['vasp_jobs_created']}")
        print(f"   Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 