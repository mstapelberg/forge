#!/usr/bin/env python3
"""
Defect-aware adversarial attack workflow.

This script extends the standard adversarial attack workflow to handle defect structures
by identifying and fixing the positions of interstitial atoms during optimization.
"""

import numpy as np
import torch
import argparse
import os
import time
import pickle
import hashlib
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union, Tuple
from pathlib import Path
from ase import Atoms
from ase.io import write, read
from ase.constraints import FixAtoms
import matplotlib.pyplot as plt

from forge.core.database import DatabaseManager
from forge.core.adversarial_attack import GradientAdversarialOptimizer
from forge.core.defect_motifs import generate_defect_structures
from mace.calculators import MACECalculator


def identify_interstitial_atoms(atoms: Atoms, motif_type: str, 
                               interstitial_tolerance: float = 0.5) -> List[int]:
    """
    Identify interstitial atoms in a defect structure.
    
    Args:
        atoms: ASE Atoms object containing the structure
        motif_type: Type of defect motif (e.g., 'sia', 'di-sia', 'vacancy')
        interstitial_tolerance: Distance tolerance for identifying interstitials (Å)
        
    Returns:
        List of atom indices that are interstitials
    """
    interstitial_indices = []
    
    if motif_type in ['sia', 'di-sia']:
        # For SIA structures, identify atoms that are not at regular lattice sites
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        
        # Get the base lattice parameter (assuming cubic or similar)
        lattice_param = np.mean(np.diag(cell))
        
        # For BCC-like structures, regular sites should be at integer multiples of a/2
        # Interstitials will be at fractional positions
        for i, pos in enumerate(positions):
            # Check if position is close to a regular lattice site
            fractional_pos = pos / (lattice_param / 2)
            nearest_integer = np.round(fractional_pos)
            distance_to_lattice = np.linalg.norm(fractional_pos - nearest_integer)
            
            # If distance is large, it's likely an interstitial
            if distance_to_lattice > interstitial_tolerance:
                interstitial_indices.append(i)
    
    elif motif_type == 'vacancy':
        # For vacancy structures, no interstitials to fix
        pass
    
    elif motif_type in ['surface_100', 'surface_110', 'surface_111', 'surface_112']:
        # For surface structures, fix atoms in the bottom layers
        # This is a simplified approach - you might want to customize this
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        
        # Assume bottom 20% of atoms should be fixed
        z_positions = positions[:, 2]
        z_min, z_max = np.min(z_positions), np.max(z_positions)
        z_threshold = z_min + 0.2 * (z_max - z_min)
        
        for i, z_pos in enumerate(z_positions):
            if z_pos <= z_threshold:
                interstitial_indices.append(i)
    
    else:
        # For other motif types, no specific interstitial identification
        # You can extend this function for other defect types
        pass
    
    return interstitial_indices


def create_constrained_atoms(atoms: Atoms, fixed_indices: List[int]) -> Atoms:
    """
    Create a copy of atoms with constraints applied to fixed indices.
    
    Args:
        atoms: Original ASE Atoms object
        fixed_indices: List of atom indices to fix
        
    Returns:
        New Atoms object with constraints applied
    """
    constrained_atoms = atoms.copy()
    
    if fixed_indices:
        # Create constraint that fixes the specified atoms
        constraint = FixAtoms(indices=fixed_indices)
        constrained_atoms.set_constraint(constraint)
    
    return constrained_atoms


def run_defect_adversarial_attacks(
    model_paths: List[str],
    top_n: int,
    generation: int,
    db_manager: Optional[DatabaseManager] = None,
    structure_ids: Optional[List[int]] = None,
    compositions: Optional[List[Dict[str, float]]] = None,
    n_iterations: int = 200,
    learning_rate: float = 0.01,
    temperature: float = 1000,
    include_probability: bool = False,
    min_distance: float = 1.0,
    use_energy_per_atom: bool = True,
    device: Optional[str] = None,
    debug: bool = False,
    output_dir: str = '.',
    save_output: bool = False,
    patience: int = 25,
    shake: bool = False,
    shake_std: float = 0.05,
    ranking_metric: str = 'force_rmse',
    reference_calculator: str = 'vasp',
    cache_rmse: bool = True,
    plot_rmse_histogram: bool = True,
    rmse_cutoff: Optional[float] = None,
    interstitial_tolerance: float = 0.5,
    include_motifs: Optional[List[str]] = None,
    exclude_motifs: Optional[List[str]] = None,
    custom_motif_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    select_n_from_trajectory: Optional[int] = None,
    calculator_type: Optional[str] = None,
    species_to_type_name: Optional[Dict[str, int]] = None
) -> Union[Dict[int, List[Atoms]], None]:
    """
    Run adversarial attacks on defect structures with fixed interstitial positions.
    
    Args:
        db_manager: Database manager (if using database structures)
        structure_ids: List of structure IDs from database (if using database)
        compositions: List of compositions for defect generation (if generating new structures)
        model_paths: List of paths to MACE model files
        top_n: Number of top structures to select for attack
        generation: Generation tag for new structures
        n_iterations: Number of optimization steps
        learning_rate: Optimizer learning rate
        temperature: Temperature for Boltzmann weighting
        include_probability: Whether to weight loss by Boltzmann probability
        min_distance: Minimum allowed interatomic distance
        use_energy_per_atom: Use energy per atom for probability calculations
        device: Compute device
        debug: Enable debug printing
        output_dir: Directory to save output files
        save_output: If True, save trajectories and return None
        patience: Patience parameter for optimizer
        shake: If True, apply random shake when patience is reached
        shake_std: Standard deviation for random shake
        ranking_metric: Metric to rank structures ('variance' or 'force_rmse')
        reference_calculator: Calculator name for reference forces
        cache_rmse: Whether to cache RMSE calculations
        plot_rmse_histogram: Whether to generate RMSE distribution histogram
        rmse_cutoff: Optional RMSE cutoff threshold
        interstitial_tolerance: Distance tolerance for identifying interstitials
        include_motifs: List of motif types to include
        exclude_motifs: List of motif types to exclude
        custom_motif_path: Path to custom motif templates
        random_seed: Seed for random number generator
        select_n_from_trajectory: Number of structures to select from each trajectory
        
    Returns:
        Dictionary of trajectories or None if saving to files
    """
    
    # Validate input parameters
    if db_manager is not None and structure_ids is not None:
        # Use database structures
        print("Using structures from database...")
        initial_atoms_list = db_manager.get_batch_atoms_with_calculation(
            structure_ids, calculator=reference_calculator if ranking_metric == 'force_rmse' else None
        )
        structures_to_process = []
        
        for atoms in initial_atoms_list:
            if not atoms:
                continue
            parent_id = atoms.info.get('structure_id')
            if parent_id is None:
                continue
                
            # Extract motif type from metadata
            config_type = "unknown"
            if 'calculation_info' in atoms.info and isinstance(atoms.info['calculation_info'], dict):
                config_type = atoms.info['calculation_info'].get('config_type', config_type)
            config_type = atoms.info.get('config_type', config_type)
            
            structures_to_process.append({
                'id': parent_id,
                'atoms': atoms,
                'motif_type': config_type
            })
            
    elif compositions is not None:
        # Generate new defect structures
        print("Generating new defect structures...")
        all_structures = generate_defect_structures(
            target_compositions=compositions,
            exclude_motifs=exclude_motifs,
            include_motifs=include_motifs,
            custom_motif_path=custom_motif_path,
            random_seed=random_seed
        )
        
        structures_to_process = []
        for i, structure_info in enumerate(all_structures):
            structures_to_process.append({
                'id': f"gen_{generation}_{i}",
                'atoms': structure_info['structure'],
                'motif_type': structure_info['motif_type']
            })
    else:
        raise ValueError("Must provide either db_manager + structure_ids or compositions")
    
    print(f"Processing {len(structures_to_process)} structures...")
    
    # Initialize calculator for ranking
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ranking_calc = MACECalculator(
        model_paths=model_paths,
        device=device,
        default_dtype='float32'
    )
    
    # Calculate initial metrics and rank structures
    initial_metrics = []
    
    if ranking_metric == 'force_rmse':
        print("Calculating initial force RMSE...")
        for i, data in enumerate(tqdm(structures_to_process, desc="Initial Force RMSE Calc")):
            atoms = data['atoms'].copy()
            try:
                # Get reference forces if available
                if atoms.has("forces"):
                    ref_forces = atoms.arrays['forces']
                    atoms.calc = ranking_calc
                    model_forces = atoms.get_forces(apply_constraint=False)
                    rmse = np.sqrt(np.mean((model_forces - ref_forces)**2))
                else:
                    # If no reference forces, use variance instead
                    rmse = 0.0
                    
                initial_metrics.append({'id': data['id'], 'metric': float(rmse), 'index': i})
            except Exception as e:
                print(f"Failed force RMSE calculation for structure {data['id']}: {e}")
                initial_metrics.append({'id': data['id'], 'metric': -1, 'index': i})
    
    elif ranking_metric == 'variance':
        print("Calculating initial variances...")
        for i, data in enumerate(tqdm(structures_to_process, desc="Initial Variance Calc")):
            atoms = data['atoms']
            try:
                # Calculate force variance across ensemble
                all_forces = []
                with torch.no_grad():
                    for model in ranking_calc.models:
                        atoms.calc = model
                        all_forces.append(atoms.get_forces(apply_constraint=False))
                
                forces_array = np.array(all_forces)
                force_magnitudes = np.linalg.norm(forces_array, axis=2, keepdims=True)
                force_magnitudes = np.where(force_magnitudes < 1e-10, 1.0, force_magnitudes)
                normalized_forces = forces_array / force_magnitudes
                atom_variances = np.var(normalized_forces, axis=0)
                mean_variance = float(np.mean(np.sum(atom_variances, axis=1)))
                
                initial_metrics.append({'id': data['id'], 'metric': mean_variance, 'index': i})
            except Exception as e:
                print(f"Failed variance calculation for structure {data['id']}: {e}")
                initial_metrics.append({'id': data['id'], 'metric': -1, 'index': i})
    
    # Filter and rank structures
    valid_metrics = [v for v in initial_metrics if v['metric'] >= 0]
    if not valid_metrics:
        print("No valid initial metrics calculated. Exiting.")
        return {}
    
    valid_metrics.sort(key=lambda x: x['metric'], reverse=True)
    num_to_select = min(top_n, len(valid_metrics))
    selected_indices = [v['index'] for v in valid_metrics[:num_to_select]]
    selected_structures_data = [structures_to_process[i] for i in selected_indices]
    
    print(f"Selected top {num_to_select} structures for optimization:")
    for i, data in enumerate(selected_structures_data):
        metric_val = valid_metrics[i]['metric']
        print(f"  {i+1}. ID: {data['id']}, {ranking_metric}: {metric_val:.6f}, Motif: {data['motif_type']}")
    
    # Initialize optimizer
    optimizer = GradientAdversarialOptimizer(
        model_paths=model_paths,
        device=device,
        learning_rate=learning_rate,
        temperature=temperature,
        include_probability=include_probability,
        debug=debug,
        calculator_type=calculator_type,  # Pass from function parameters
        species_to_type_name=species_to_type_name  # Pass from function parameters
    )
    
    # Run optimization with constraints
    all_trajectories = {}
    plot_save_dir = Path(output_dir) / 'plots'
    
    for data in tqdm(selected_structures_data, desc="Defect Adversarial Attacks"):
        atoms_initial = data['atoms']
        parent_id = data['id']
        motif_type = data['motif_type']
        
        if debug:
            print(f"\n--- Optimizing Structure ID: {parent_id} (Motif: {motif_type}) ---")
        
        try:
            # Identify interstitial atoms to fix
            interstitial_indices = identify_interstitial_atoms(
                atoms_initial, motif_type, interstitial_tolerance
            )
            
            if debug and interstitial_indices:
                print(f"  Fixed {len(interstitial_indices)} interstitial atoms: {interstitial_indices}")
            
            # Create constrained structure
            constrained_atoms = create_constrained_atoms(atoms_initial, interstitial_indices)
            
            # Run optimization
            generated_trajectory = optimizer.optimize(
                atoms=constrained_atoms,
                generation=generation,
                n_iterations=n_iterations,
                min_distance=min_distance,
                output_dir=str(plot_save_dir),
                patience=patience,
                shake=shake,
                shake_std=shake_std,
                require_structure_id=args.require_structure_id
            )
            
            # Select N structures from trajectory if requested
            if select_n_from_trajectory is not None and len(generated_trajectory) > select_n_from_trajectory:
                # Select evenly spaced structures from the trajectory
                indices = np.linspace(0, len(generated_trajectory) - 1, select_n_from_trajectory, dtype=int)
                selected_trajectory = [generated_trajectory[i] for i in indices]
                all_trajectories[parent_id] = selected_trajectory
                
                if debug:
                    print(f"  Selected {len(selected_trajectory)} structures from trajectory of {len(generated_trajectory)}")
            else:
                all_trajectories[parent_id] = generated_trajectory
            
            if debug:
                print(f"  Finished optimization for {parent_id}")
                
        except Exception as e:
            print(f"Optimization failed for structure {parent_id}: {e}")
    
    # Save or return results
    if save_output:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving trajectories to: {save_path.resolve()}")
        
        saved_files_count = 0
        for parent_id, trajectory in tqdm(all_trajectories.items(), desc="Saving Trajectories"):
            # Find original structure
            original_structure = None
            for data in selected_structures_data:
                if data['id'] == parent_id:
                    original_structure = data['atoms']
                    break
            
            if original_structure is not None:
                full_trajectory = [original_structure] + trajectory
                for atom in full_trajectory:
                    atom.calc = None
                
                filename = save_path / f"structure_{parent_id}.xyz"
                try:
                    write(filename, full_trajectory, format='extxyz')
                    saved_files_count += 1
                except Exception as e:
                    print(f"Failed to save trajectory for {parent_id}: {e}")
        
        print(f"Successfully saved {saved_files_count} trajectory files.")
        return None
    else:
        return all_trajectories


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Run adversarial attacks on defect structures with fixed interstitials",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--structure-ids',
        nargs='+',
        type=int,
        help='List of structure IDs from database'
    )
    input_group.add_argument(
        '--compositions',
        type=str,
        help='Path to JSON file containing target compositions'
    )
    
    # Model and optimization parameters
    parser.add_argument(
        '--model-paths',
        nargs='+',
        required=True,
        help='Paths to model files (MACE or Allegro)'
    )
    parser.add_argument(
        '--calculator-type',
        type=str,
        choices=['mace', 'allegro'],
        help='Type of calculator to use (auto-detect if not specified)'
    )
    parser.add_argument(
        '--species-mapping',
        type=str,
        help='JSON string or file path for species to type mapping (required for Allegro)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top structures to select for attack'
    )
    parser.add_argument(
        '--generation',
        type=int,
        required=True,
        help='Generation tag for new structures'
    )
    parser.add_argument(
        '--n-iterations',
        type=int,
        default=200,
        help='Number of optimization steps'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Optimizer learning rate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1000,
        help='Temperature for Boltzmann weighting'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='defect_adversarial_attacks',
        help='Output directory'
    )
    parser.add_argument(
        '--save-output',
        action='store_true',
        help='Save trajectories to files'
    )
    parser.add_argument(
        '--select-n-from-trajectory',
        type=int,
        help='Number of structures to select from each trajectory'
    )
    
    # Defect-specific options
    parser.add_argument(
        '--interstitial-tolerance',
        type=float,
        default=0.5,
        help='Distance tolerance for identifying interstitials (Å)'
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
    
    # Other options
    parser.add_argument(
        '--device',
        type=str,
        help='Compute device (cuda/cpu)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    parser.add_argument(
        '--require-structure-id',
        action='store_true',
        help='Require structure_id in atoms info (default: use reserved ID 99999999 for new structures)'
    )
    
    args = parser.parse_args()
    
    # Load compositions if provided
    compositions = None
    if args.compositions:
        import json
        with open(args.compositions, 'r') as f:
            compositions = json.load(f)
    
    # Parse species mapping if provided
    species_to_type_name = None
    if args.species_mapping:
        import json
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
                print("Using default species mapping: {'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}")
                species_to_type_name = {'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    
    # Run the workflow
    try:
        trajectories = run_defect_adversarial_attacks(
            structure_ids=args.structure_ids,
            compositions=compositions,
            model_paths=args.model_paths,
            top_n=args.top_n,
            generation=args.generation,
            n_iterations=args.n_iterations,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            output_dir=args.output_dir,
            save_output=args.save_output,
            select_n_from_trajectory=args.select_n_from_trajectory,
            interstitial_tolerance=args.interstitial_tolerance,
            include_motifs=args.include_motifs,
            exclude_motifs=args.exclude_motifs,
            custom_motif_path=args.custom_motif_path,
            random_seed=args.random_seed,
            device=args.device,
            debug=args.debug,
            calculator_type=args.calculator_type,
            species_to_type_name=species_to_type_name
        )
        
        if trajectories is not None:
            print(f"Returned {len(trajectories)} trajectories")
        else:
            print("Trajectories saved to files")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 