#!/usr/bin/env python3
"""
Script to generate defect motifs for multiple compositions and create VASP jobs.

This script:
1. Takes a list of target compositions
2. Generates defect structures using various motif templates
3. Creates VASP job directories for each generated structure
4. Organizes output in a structured manner

Usage:
    python generate_defect_motifs.py --compositions compositions.json --output-dir output_folder
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add forge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.core.defect_motifs import generate_defect_structures
from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase


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


def save_compositions_to_file(compositions: List[Dict[str, float]], filepath: str) -> None:
    """
    Save target compositions to a JSON file.
    
    Args:
        compositions: List of composition dictionaries
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(compositions, f, indent=2)


def create_example_compositions_file(filepath: str) -> None:
    """
    Create an example compositions file with common alloy compositions.
    
    Args:
        filepath: Path to save the example file
    """
    example_compositions = [
        {'V': 0.76, 'Cr': 0.17, 'Ti': 0.07},
        {'V': 0.76, 'Cr': 0.12, 'Ti': 0.12},
        {'V': 0.76, 'Cr': 0.07, 'Ti': 0.17},
        {'V': 0.92, 'Cr': 0.04, 'Ti': 0.04},
        {'V': 0.98, 'Ti': 0.02},
    ]
    
    save_compositions_to_file(example_compositions, filepath)
    print(f"Created example compositions file: {filepath}")


def organize_output_structure(
    base_output_dir: str,
    composition: Dict[str, float],
    motif_type: str,
    variant_index: int = 0
) -> str:
    """
    Create an organized output directory structure for VASP jobs.
    
    Args:
        base_output_dir: Base output directory
        composition: Composition dictionary
        motif_type: Type of defect motif
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
        job_dir_name = f"{comp_str}_{motif_type}"
    else:
        job_dir_name = f"{comp_str}_{motif_type}_var{variant_index}"
    
    job_path = os.path.join(base_output_dir, job_dir_name)
    return job_path


def generate_and_prepare_vasp_jobs(
    compositions: List[Dict[str, float]],
    output_dir: str,
    vasp_profile_name: str = "static",
    hpc_profile_name: str = "PSFC-GPU",
    exclude_motifs: Optional[List[str]] = None,
    include_motifs: Optional[List[str]] = None,
    custom_motif_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    auto_kpoints: bool = False,
    save_structures: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate defect structures and prepare VASP jobs for multiple compositions.
    
    Args:
        compositions: List of composition dictionaries
        output_dir: Base output directory for all jobs
        vasp_profile_name: Name of VASP settings profile to use
        hpc_profile_name: Name of HPC profile to use
        exclude_motifs: List of motif types to exclude
        include_motifs: List of motif types to include (if None, uses all default)
        custom_motif_path: Path to custom motif templates
        random_seed: Seed for random number generator
        auto_kpoints: Whether to automatically determine k-points
        save_structures: Whether to save generated structures as XYZ files
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing generation statistics and results
    """
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all defect structures
    if verbose:
        print(f"Generating defect structures for {len(compositions)} compositions...")
    
    all_structures = generate_defect_structures(
        target_compositions=compositions,
        exclude_motifs=exclude_motifs,
        include_motifs=include_motifs,
        custom_motif_path=custom_motif_path,
        random_seed=random_seed
    )
    
    if verbose:
        print(f"Generated {len(all_structures)} total structures")
    
    # Statistics tracking
    stats = {
        'total_compositions': len(compositions),
        'total_structures': len(all_structures),
        'successful_jobs': 0,
        'failed_jobs': 0,
        'job_directories': [],
        'errors': []
    }
    
    # Process each generated structure
    for i, structure_info in enumerate(all_structures):
        try:
            # Create organized output directory
            job_dir = organize_output_structure(
                output_dir,
                structure_info['target_composition_input'],
                structure_info['motif_type'],
                structure_info['variant_index']
            )
            
            # Create VASP job
            prepare_vasp_job_from_ase(
                atoms=structure_info['structure'],
                vasp_profile_name=vasp_profile_name,
                hpc_profile_name=hpc_profile_name,
                output_dir=job_dir,
                auto_kpoints=auto_kpoints,
                job_name=os.path.basename(job_dir)
            )
            
            # Save structure as XYZ file if requested
            if save_structures:
                xyz_path = os.path.join(job_dir, "structure.xyz")
                structure_info['structure'].write(xyz_path)
            
            # Save structure metadata
            metadata_path = os.path.join(job_dir, "structure_info.json")
            # Convert numpy arrays to lists for JSON serialization
            metadata = {
                'target_composition_input': structure_info['target_composition_input'],
                'parsed_composition_fractional': structure_info['parsed_composition_fractional'],
                'motif_type': structure_info['motif_type'],
                'N_total_in_structure': structure_info['N_total_in_structure'],
                'atom_counts_in_structure': structure_info['atom_counts_in_structure'],
                'actual_composition_fractional': structure_info['actual_composition_fractional'],
                'variant_index': structure_info['variant_index']
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            stats['successful_jobs'] += 1
            stats['job_directories'].append(job_dir)
            
            if verbose:
                print(f"  [{i+1}/{len(all_structures)}] Created job: {os.path.basename(job_dir)}")
                
        except Exception as e:
            stats['failed_jobs'] += 1
            error_msg = f"Failed to create job for {structure_info['motif_type']} motif: {str(e)}"
            stats['errors'].append(error_msg)
            if verbose:
                print(f"  [{i+1}/{len(all_structures)}] ERROR: {error_msg}")
    
    # Save overall statistics
    stats_path = os.path.join(output_dir, "generation_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    if verbose:
        print(f"\nGeneration complete!")
        print(f"  Successful jobs: {stats['successful_jobs']}")
        print(f"  Failed jobs: {stats['failed_jobs']}")
        print(f"  Output directory: {output_dir}")
        print(f"  Statistics saved to: {stats_path}")
    
    return stats


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate defect motifs and create VASP jobs for multiple compositions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with example compositions
  python generate_defect_motifs.py --create-example compositions.json
  
  # Generate jobs for compositions
  python generate_defect_motifs.py --compositions compositions.json --output-dir vasp_jobs
  
  # Use specific motifs only
  python generate_defect_motifs.py --compositions compositions.json --output-dir vasp_jobs \\
    --include-motifs vacancy sia surface_100
  
  # Exclude certain motifs
  python generate_defect_motifs.py --compositions compositions.json --output-dir vasp_jobs \\
    --exclude-motifs liquid surface_100
        """
    )
    
    parser.add_argument(
        '--compositions', '-c',
        type=str,
        help='Path to JSON file containing target compositions'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='vasp_jobs',
        help='Output directory for VASP jobs (default: vasp_jobs)'
    )
    
    parser.add_argument(
        '--create-example',
        type=str,
        help='Create an example compositions file at the specified path'
    )
    
    parser.add_argument(
        '--vasp-profile',
        type=str,
        default='relaxation',
        help='VASP settings profile name (default: relaxation)'
    )
    
    parser.add_argument(
        '--hpc-profile',
        type=str,
        default='Perlmutter-CPU',
        help='HPC profile name (default: Perlmutter-CPU)'
    )
    
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
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--auto-kpoints',
        action='store_true',
        help='Automatically determine k-points'
    )
    
    parser.add_argument(
        '--no-save-structures',
        action='store_true',
        help='Do not save generated structures as XYZ files'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle example file creation
    if args.create_example:
        create_example_compositions_file(args.create_example)
        return
    
    # Validate required arguments
    if not args.compositions:
        parser.error("--compositions is required (or use --create-example to create an example file)")
    
    try:
        # Load compositions
        compositions = load_compositions_from_file(args.compositions)
        
        # Generate and prepare VASP jobs
        stats = generate_and_prepare_vasp_jobs(
            compositions=compositions,
            output_dir=args.output_dir,
            vasp_profile_name=args.vasp_profile,
            hpc_profile_name=args.hpc_profile,
            exclude_motifs=args.exclude_motifs,
            include_motifs=args.include_motifs,
            custom_motif_path=args.custom_motif_path,
            random_seed=args.random_seed,
            auto_kpoints=args.auto_kpoints,
            save_structures=not args.no_save_structures,
            verbose=not args.quiet
        )
        
        # Exit with error code if any jobs failed
        if stats['failed_jobs'] > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
