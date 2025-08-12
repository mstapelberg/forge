#!/usr/bin/env python3
"""
Step 2: Full Defect Motif Workflow - Select and Create VASP Jobs.

This script performs the second step of the complete workflow:
1. Load trajectory XYZ files from Step 1 (with variance already calculated)
2. Use MACE embeddings and aa_selection to choose diverse structures PER TRAJECTORY
3. Create VASP jobs for the selected structures

Usage:
    python full_defect_motif_workflow_step2.py --input-dir step1_output --mace-model-path model.model --n-select 50
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from tqdm import tqdm
import ase.io
import matplotlib.pyplot as plt

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase
from forge.analysis.aa_selection import AAAnalyzer
from mace.calculators.mace import MACECalculator

os.environ['VASP_PP_PATH'] = '/home/myless/VASP/POTCAR_64_PBE/potpaw_PBE'

def load_step1_metadata(input_dir: str) -> Dict[str, Any]:
    """Load metadata from Step 1."""
    metadata_path = os.path.join(input_dir, "step1_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Step 1 metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if not metadata.get('step1_completed', False):
        raise ValueError("Step 1 metadata indicates Step 1 was not completed successfully")
    
    return metadata


def create_job_name_from_trajectory(trajectory_file: Path, selected_index: int) -> str:
    """Create a job name from trajectory file and selected structure index."""
    # Extract trajectory info from filename (e.g., "trajectory_gen_10_0.xyz")
    trajectory_name = trajectory_file.stem  # removes .xyz
    return f"{trajectory_name}_sel_{selected_index:03d}"


def create_job_dir_from_trajectory(output_dir: str, trajectory_file: Path, selected_index: int) -> str:
    """Create a job directory path from trajectory file and selected structure index."""
    job_name = create_job_name_from_trajectory(trajectory_file, selected_index)
    return os.path.join(output_dir, job_name)


def process_single_trajectory(
    trajectory_file: Path,
    calculator: MACECalculator,
    output_dir: str,
    n_select: int,
    vasp_profile_name: str,
    hpc_profile_name: str,
    auto_kpoints: bool,
    verbose: bool
) -> Dict[str, Any]:
    """Process a single trajectory file: select structures and create VASP jobs."""
    
    if verbose:
        print(f"\n--- Processing trajectory: {trajectory_file.name} ---")
    
    # Load trajectory (skip first frame as it's unoptimized)
    try:
        trajectory = ase.io.read(str(trajectory_file), index="1:")  # Skip first frame
        if verbose:
            print(f"  Loaded {len(trajectory)} structures (skipped first frame)")
    except Exception as e:
        print(f"  ERROR: Failed to load {trajectory_file}: {e}")
        return {
            'trajectory_file': str(trajectory_file),
            'success': False,
            'error': str(e),
            'structures_selected': 0,
            'vasp_jobs_created': 0
        }
    
    if not trajectory:
        print(f"  WARNING: No structures in trajectory {trajectory_file}")
        return {
            'trajectory_file': str(trajectory_file),
            'success': False,
            'error': 'No structures in trajectory',
            'structures_selected': 0,
            'vasp_jobs_created': 0
        }
    
    # Check for variance values
    structures_with_variance = []
    for i, atoms in enumerate(trajectory):
        if 'variance' in atoms.info:
            structures_with_variance.append(atoms)
        else:
            if verbose:
                print(f"    Warning: Structure {i} missing variance, skipping")
    
    if not structures_with_variance:
        print(f"  ERROR: No structures with variance values in {trajectory_file}")
        return {
            'trajectory_file': str(trajectory_file),
            'success': False,
            'error': 'No structures with variance values',
            'structures_selected': 0,
            'vasp_jobs_created': 0
        }
    
    if verbose:
        print(f"  Found {len(structures_with_variance)} structures with variance values")
    
    # Run AA selection
    try:
        if verbose:
            print(f"  Initializing AAAnalyzer...")
        analyzer = AAAnalyzer(atoms_list=structures_with_variance, calculator=calculator)
        
        if verbose:
            print(f"  Selecting up to {n_select} diverse structures...")
        
        selected_indices, fig = analyzer.select_diverse_structures(
            n_select=min(n_select, len(structures_with_variance)),
            plot=True
        )
        
        # Save plot
        if fig:
            plot_path = os.path.join(output_dir, f"{trajectory_file.stem}_selection_plot.png")
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Use plt.close(fig) not fig.close()
            if verbose:
                print(f"    Saved selection plot to: {plot_path}")
        
        selected_atoms = [structures_with_variance[i] for i in selected_indices]
        
        if verbose:
            print(f"  Selected {len(selected_atoms)} structures")
        
        # Create VASP jobs immediately after selection
        vasp_jobs_created = 0
        vasp_errors = []
        
        if verbose:
            print(f"  Creating VASP jobs for {len(selected_atoms)} structures...")
        
        for i, atoms in enumerate(selected_atoms):
            try:
                job_dir = create_job_dir_from_trajectory(output_dir, trajectory_file, i)
                job_name = create_job_name_from_trajectory(trajectory_file, i)
                
                prepare_vasp_job_from_ase(
                    atoms=atoms,
                    vasp_profile_name=vasp_profile_name,
                    hpc_profile_name=hpc_profile_name,
                    output_dir=job_dir,
                    auto_kpoints=True,  # Always use auto k-points
                    job_name=job_name
                )
                
                # Save structure as XYZ file
                xyz_path = os.path.join(job_dir, "structure.xyz")
                atoms.write(xyz_path)
                
                vasp_jobs_created += 1
                
                if verbose:
                    print(f"    [{i+1}/{len(selected_atoms)}] Created job: {job_name}")
                    
            except Exception as e:
                error_msg = f"Failed to create job {i} for {trajectory_file}: {str(e)}"
                vasp_errors.append(error_msg)
                if verbose:
                    print(f"    ERROR: {error_msg}")
        
    except Exception as e:
        print(f"  ERROR: AA selection failed for {trajectory_file}: {e}")
        return {
            'trajectory_file': str(trajectory_file),
            'success': False,
            'error': f'AA selection failed: {str(e)}',
            'structures_selected': 0,
            'vasp_jobs_created': 0
        }
    
    return {
        'trajectory_file': str(trajectory_file),
        'success': True,
        'structures_loaded': len(trajectory),
        'structures_with_variance': len(structures_with_variance),
        'structures_selected': len(selected_atoms),
        'vasp_jobs_created': vasp_jobs_created,
        'vasp_errors': vasp_errors
    }


def run_step2_workflow(
    input_dir: str,
    mace_model_path: str,
    output_dir: str,
    n_select: int = 50,
    vasp_profile_name: str = "static",
    hpc_profile_name: str = "PSFC-GPU",
    auto_kpoints: bool = False,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Step 2 of the defect motif workflow: select and create VASP jobs.
    
    Args:
        input_dir: Directory containing Step 1 output
        mace_model_path: Path to MACE model file
        output_dir: Base output directory for Step 2 results
        n_select: Number of diverse structures to select PER TRAJECTORY
        vasp_profile_name: Name of VASP settings profile to use
        hpc_profile_name: Name of HPC profile to use
        auto_kpoints: Whether to automatically determine k-points
        device: Compute device
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing workflow statistics and results
    """
    if verbose:
        print("=== Step 2: Structure Selection and VASP Job Creation ===")
        print(f"Input directory: {input_dir}")
        print(f"MACE model: {mace_model_path}")
        print(f"Output directory: {output_dir}")
        print(f"Structures per trajectory: {n_select}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Step 1 metadata
    if verbose:
        print("\n--- Loading Step 1 Metadata ---")
    
    step1_metadata = load_step1_metadata(input_dir)
    trajectories_dir = step1_metadata['output_directories']['trajectories']
    
    if verbose:
        print(f"Step 1 completed successfully")
        print(f"Trajectories directory: {trajectories_dir}")
    
    # Find trajectory files
    trajectory_files = list(Path(trajectories_dir).glob("trajectory_*.xyz"))
    
    if not trajectory_files:
        raise FileNotFoundError(f"No trajectory files found in {trajectories_dir}")
    
    if verbose:
        print(f"Found {len(trajectory_files)} trajectory files")
    
    # Initialize MACE calculator
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"\n--- Initializing MACE Calculator ---")
        print(f"Model: {mace_model_path}")
        print(f"Device: {device}")
    
    calculator = MACECalculator(
        model_paths=[mace_model_path],
        device=device,
        default_dtype="float32"
    )
    
    # Process each trajectory
    if verbose:
        print(f"\n--- Processing Trajectories ---")
    
    trajectory_results = []
    total_selected = 0
    total_jobs_created = 0
    
    for trajectory_file in tqdm(trajectory_files, desc="Processing trajectories"):
        result = process_single_trajectory(
            trajectory_file=trajectory_file,
            calculator=calculator,
            output_dir=output_dir,
            n_select=n_select,
            vasp_profile_name=vasp_profile_name,
            hpc_profile_name=hpc_profile_name,
            auto_kpoints=auto_kpoints,
            verbose=verbose
        )
        
        trajectory_results.append(result)
        
        if result['success']:
            total_selected += result['structures_selected']
            total_jobs_created += result['vasp_jobs_created']
    
    # Compile final statistics
    successful_trajectories = sum(1 for r in trajectory_results if r['success'])
    failed_trajectories = len(trajectory_results) - successful_trajectories
    
    final_stats = {
        'step2_completed': True,
        'input_trajectories': len(trajectory_files),
        'successful_trajectories': successful_trajectories,
        'failed_trajectories': failed_trajectories,
        'total_structures_selected': total_selected,
        'total_vasp_jobs_created': total_jobs_created,
        'trajectory_results': trajectory_results,
        'output_directory': output_dir
    }
    
    # Save final statistics
    stats_path = os.path.join(output_dir, "step2_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    if verbose:
        print(f"\n=== Step 2 Complete ===")
        print(f"Successful trajectories: {successful_trajectories}/{len(trajectory_files)}")
        print(f"Total structures selected: {total_selected}")
        print(f"Total VASP jobs created: {total_jobs_created}")
        print(f"Output directory: {output_dir}")
        print(f"Statistics saved to: {stats_path}")
    
    return final_stats


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Step 2: Select diverse structures and create VASP jobs with MACE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python full_defect_motif_workflow_step2.py --input-dir step1_output \\
    --mace-model-path model.model --output-dir step2_output
  
  # With custom selection parameters
  python full_defect_motif_workflow_step2.py --input-dir step1_output \\
    --mace-model-path model.model --output-dir step2_output \\
    --n-select 30 --vasp-profile relaxation
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory containing Step 1 output'
    )
    
    parser.add_argument(
        '--mace-model-path',
        type=str,
        required=True,
        help='Path to MACE model file (.model file)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='step2_output',
        help='Output directory for Step 2 results (default: step2_output)'
    )
    
    # Selection parameters
    parser.add_argument(
        '--n-select',
        type=int,
        default=50,
        help='Number of diverse structures to select PER TRAJECTORY (default: 50)'
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
    
    # Calculator parameters
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Compute device (auto-detect if not specified)'
    )
    
    # Other options
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Run Step 2 workflow
        stats = run_step2_workflow(
            input_dir=args.input_dir,
            mace_model_path=args.mace_model_path,
            output_dir=args.output_dir,
            n_select=args.n_select,
            vasp_profile_name=args.vasp_profile,
            hpc_profile_name=args.hpc_profile,
            auto_kpoints=args.auto_kpoints,
            device=args.device,
            verbose=not args.quiet
        )
        
        print(f"\n✅ Step 2 completed successfully!")
        print(f"   Successful trajectories: {stats['successful_trajectories']}/{stats['input_trajectories']}")
        print(f"   Total structures selected: {stats['total_structures_selected']}")
        print(f"   Total VASP jobs created: {stats['total_vasp_jobs_created']}")
        print(f"   Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()