#!/usr/bin/env python
# TODO on 2025-02-28
# TODO need to add metadata to the vasp jobs individually that was in the ase.info of the atoms object we wrote to
# TODO Also need to make sure the output from the aa optimization (including the trajectory) is saved to it's own folder in the aa_output 
# subdirectory of the root aa directory 
# TODO need to re-do the test so that it grabs 25 structures from the database at random, runs force variance ranking on them, and then does aa on the top 3
"""Script to run adversarial attack optimization on a batch of structures.

This script provides two main functions:
1. run_gradient_aa_optimization: Gradient-based adversarial attack optimization
2. run_aa_optimization: Monte Carlo based adversarial attack optimization

Both functions take a structure (or set of structures) and optimize them to 
maximize model variance, which indicates areas of model uncertainty.
"""

import argparse
import json
from pathlib import Path
import ase.io
import numpy as np
from monty.serialization import dumpfn

from forge.core.adversarial_attack import (
    GradientAdversarialOptimizer,
    AdversarialCalculator,
    DisplacementGenerator,
    AdversarialOptimizer
)
from forge.core.database import DatabaseManager

def run_gradient_aa_optimization(
    xyz_file: str,
    output_dir: str,
    model_paths: list[str],
    learning_rate: float = 0.01,
    n_iterations: int = 60,
    min_distance: float = 1.5,
    include_probability: bool = False,
    temperature: float = 0.86,
    device: str = "cuda",
    use_autograd: bool = False,
    save_to_database: bool = True,
    database_id: int = None,
    config_type: str = "aa-gradient",
    debug: bool = False,
):
    """Run gradient-based adversarial attack optimization on structures in XYZ file.
    
    Args:
        xyz_file: Path to input XYZ file containing structures
        output_dir: Directory to save optimization results
        model_paths: List of paths to model files
        learning_rate: Learning rate for gradient ascent
        n_iterations: Number of optimization iterations
        min_distance: Minimum allowed distance between atoms (Å)
        include_probability: Whether to include the probability term in the loss
        temperature: Temperature for probability weighting (eV)
        device: Device to run on (cpu/cuda)
        use_autograd: Whether to use the Hessian from MACECalculator for more efficient gradient calculation
        save_to_database: Whether to save optimized structures to database
        database_id: Structure ID in database (if optimizing a specific structure)
        config_type: Configuration type label for database
        debug: Whether to print debug messages
    """
    from forge.core.adversarial_attack import GradientAdversarialOptimizer
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize database connection
    db_manager = DatabaseManager() if save_to_database else None
    
    # Load structures
    atoms_list = ase.io.read(xyz_file, ':')
    print(f"[INFO] Loaded {len(atoms_list)} structures from {xyz_file}")
    
    # If we're using the probability weighting, we need to get energies for all structures
    energy_list = []
    if include_probability:
        # Get energies from atoms.info if available
        for atoms in atoms_list:
            if debug:
                print(f"[DEBUG] atoms.info: {atoms.info}")
            if 'energy' in atoms.info:
                energy_list.append(atoms.info['energy'])
            elif database_id is not None:
                # Try to get energy from database
                try:
                    calculations = db_manager.get_calculations(database_id)
                    if calculations and 'energy' in calculations[0]:
                        energy_list.append(calculations[0]['energy'])
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Failed to get energy from database: {e}")
        
        # If we couldn't get any energies, disable include_probability
        if not energy_list:
            print("[WARNING] No energies found for probability weighting. Disabling probability term.")
            include_probability = False
    
    # Initialize the gradient-based optimizer with the correct parameters
    optimizer = GradientAdversarialOptimizer(
        model_paths=model_paths,
        device=device,
        learning_rate=learning_rate,
        temperature=temperature,
        include_probability=include_probability,
        energy_list=energy_list if energy_list else None,
        debug=debug
    )
    
    # Run optimization for each structure
    results = []
    added_structure_ids = []
    
    for atoms in atoms_list:
        struct_name = atoms.info.get('structure_name', 'unknown')
        initial_variance = atoms.info.get('initial_variance', None)
        
        # Get structure ID from atoms object or parameter
        structure_id = atoms.info.get('structure_id', database_id)
        
        print(f"\n[INFO] Optimizing structure: {struct_name}")
        if initial_variance:
            print(f"[INFO] Initial variance: {initial_variance:.6f}")
        if structure_id:
            print(f"[INFO] Structure ID: {structure_id}")
        
        # Run optimization
        try:
            best_atoms, best_variance, loss_history = optimizer.optimize(
                atoms=atoms,
                n_iterations=n_iterations,
                min_distance=min_distance,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"[ERROR] Optimization failed: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            continue
        
        # Save optimized structure to database
        new_structure_id = None
        if save_to_database and db_manager is not None:
            try:
                # Ensure metadata is preserved and enriched
                best_atoms.info['initial_variance'] = initial_variance or loss_history[0]
                best_atoms.info['final_variance'] = best_variance
                best_atoms.info['optimization_method'] = 'gradient-based'
                best_atoms.info['parent_structure_name'] = struct_name
                
                # Prepare metadata for database with explicit type conversion
                metadata = {
                    'config_type': config_type,
                    'initial_variance': float(initial_variance) if initial_variance else float(loss_history[0]),
                    'final_variance': float(best_variance),
                    'learning_rate': float(learning_rate),
                    'n_iterations': int(n_iterations),
                    'min_distance': float(min_distance),
                    'include_probability': bool(include_probability),
                    'temperature': float(temperature)
                }
                
                # Add parent structure ID if available
                if structure_id is not None:
                    # Debug structure_id type
                    print(f"[DEBUG] structure_id type: {type(structure_id)}")
                    print(f"[DEBUG] structure_id value: {structure_id}")
                    
                    try:
                        parent_id = int(structure_id)
                        metadata['parent_id'] = parent_id
                    except (ValueError, TypeError) as e:
                        print(f"[ERROR] Failed to convert structure_id to int: {e}")
                        print(f"[DEBUG] Full structure_id details: {repr(structure_id)}")
                    
                    # Also fetch parent metadata to preserve lineage information
                    parent_metadata = db_manager.get_structure_metadata(int(structure_id))
                    if parent_metadata:
                        if 'config_type' in parent_metadata:
                            metadata['parent_config_type'] = str(parent_metadata['config_type'])
                        if 'composition' in parent_metadata:
                            metadata['target_composition'] = parent_metadata['composition']
                
                # Debug metadata before database insertion
                print("[DEBUG] Metadata to be inserted:")
                for key, value in metadata.items():
                    print(f"  {key}: {value} ({type(value)})")
                
                # Add structure to database
                new_structure_id = db_manager.add_structure(
                    best_atoms,
                    metadata=metadata
                )
                
                print(f"[INFO] Added optimized structure to database with ID: {new_structure_id}")
                added_structure_ids.append(new_structure_id)
                
            except Exception as e:
                print(f"[ERROR] Failed to add structure to database: {e}")
        
        # Save structure to file
        optimized_file = output_path / f"{struct_name}_optimized.xyz"
        ase.io.write(optimized_file, best_atoms, write_results=False)
        
        # Append to results
        result_data = {
            'structure_name': struct_name,
            'initial_variance': initial_variance or loss_history[0],
            'final_variance': best_variance,
            'loss_history': loss_history
        }
        
        # Add database information if available
        if structure_id:
            result_data['parent_structure_id'] = structure_id
        if new_structure_id:
            result_data['structure_id'] = new_structure_id
            
        results.append(result_data)
    
    # Save summary
        
    dumpfn({
        'input_file': xyz_file,
        'parameters': {
            'learning_rate': learning_rate,
            'n_iterations': n_iterations,
            'min_distance': min_distance,
            'include_probability': include_probability,
            'temperature': temperature,
            'device': device,
            'use_autograd': use_autograd
        },
        'added_structure_ids': added_structure_ids,
        'results': results
    }, output_path / 'optimization_summary.json', indent=2)
        
    return results, added_structure_ids

def run_aa_optimization(
    xyz_file: str,
    output_dir: str,
    model_paths: list[str],
    temperature: float = 1200.0,
    max_steps: int = 50,
    patience: int = 25,
    min_distance: float = 2.0,
    mode: str = "all",
    device: str = "cuda",
    save_to_database: bool = False,
    database_id: int = None,
    config_type: str = "aa-monte-carlo",
):
    """Run Monte Carlo adversarial attack optimization on structures in XYZ file.
    
    Args:
        xyz_file: Path to input XYZ file containing structures
        output_dir: Directory to save optimization results
        model_paths: List of paths to model files
        temperature: Temperature for adversarial optimization (K)
        max_steps: Maximum optimization steps per structure
        patience: Stop if no improvement after this many steps
        min_distance: Minimum allowed distance between atoms (Å)
        mode: Optimization mode ('all' or 'single' atom)
        device: Device to run on (cpu/cuda)
        save_to_database: Whether to save optimized structures to database
        database_id: Structure ID in database (if optimizing a specific structure)
        config_type: Configuration type label for database
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize calculator, optimizer, and database connection
    calculator = AdversarialCalculator(
        model_paths=model_paths,
        device=device
    )
    displacement_gen = DisplacementGenerator(min_distance=min_distance)
    optimizer = AdversarialOptimizer(
        adversarial_calc=calculator,
        displacement_gen=displacement_gen
    )
    
    db_manager = DatabaseManager() if save_to_database else None
    
    # Load structures
    atoms_list = ase.io.read(xyz_file, ':')
    print(f"[INFO] Loaded {len(atoms_list)} structures from {xyz_file}")
    
    # Run optimization for each structure
    results = []
    added_structure_ids = []
    
    for atoms in atoms_list:
        struct_name = atoms.info.get('structure_name', 'unknown')
        initial_variance = atoms.info.get('initial_variance', None)
        
        # Get structure ID from atoms object or parameter
        structure_id = atoms.info.get('structure_id', database_id)
        
        print(f"\n[INFO] Optimizing structure: {struct_name}")
        if initial_variance:
            print(f"[INFO] Initial variance: {initial_variance:.6f}")
        if structure_id:
            print(f"[INFO] Structure ID: {structure_id}")
        
        # Run optimization
        best_atoms, best_variance, accepted_moves = optimizer.optimize(
            atoms=atoms,
            temperature=temperature,
            max_iterations=max_steps,
            patience=patience,
            mode=mode,
            output_dir=output_dir
        )
        
        # Save optimized structure to database
        new_structure_id = None
        if save_to_database and db_manager is not None:
            try:
                # Ensure metadata is preserved and enriched
                best_atoms.info['initial_variance'] = initial_variance
                best_atoms.info['final_variance'] = best_variance
                best_atoms.info['optimization_method'] = 'monte-carlo'
                best_atoms.info['parent_structure_name'] = struct_name
                best_atoms.info['accepted_moves'] = accepted_moves
                
                # Prepare metadata for database
                metadata = {
                    'config_type': config_type,
                    'initial_variance': float(initial_variance) if initial_variance is not None else None,
                    'final_variance': float(best_variance),
                    'temperature': float(temperature),
                    'max_steps': int(max_steps),
                    'patience': int(patience),
                    'min_distance': float(min_distance),
                    'mode': mode,
                    'accepted_moves': accepted_moves
                }
                
                # Add parent structure ID if available
                if structure_id is not None:
                    metadata['parent_structure_id'] = structure_id
                    
                    # Also fetch parent metadata to preserve lineage information
                    parent_metadata = db_manager.get_structure_metadata(structure_id)
                    if parent_metadata:
                        if 'config_type' in parent_metadata:
                            metadata['parent_config_type'] = parent_metadata['config_type']
                        if 'composition' in parent_metadata:
                            metadata['target_composition'] = parent_metadata['composition']
                
                # Add structure to database
                new_structure_id = db_manager.add_structure(
                    best_atoms,
                    metadata=metadata
                )
                
                print(f"[INFO] Added optimized structure to database with ID: {new_structure_id}")
                added_structure_ids.append(new_structure_id)
                
            except Exception as e:
                print(f"[ERROR] Failed to add structure to database: {e}")
        
        # Save structure to file
        optimized_file = output_path / f"{struct_name}_optimized.xyz"
        ase.io.write(optimized_file, best_atoms, write_results=False)
        
        # Append to results
        result_data = {
            'structure_name': struct_name,
            'initial_variance': initial_variance,
            'final_variance': best_variance,
            'accepted_moves': accepted_moves
        }
        
        # Add database information if available
        if structure_id:
            result_data['parent_structure_id'] = structure_id
        if new_structure_id:
            result_data['structure_id'] = new_structure_id
            
        results.append(result_data)
    
    # Save summary
    with open(output_path / 'optimization_summary.json', 'w') as f:
        json.dump({
            'input_file': xyz_file,
            'parameters': {
                'temperature': temperature,
                'max_steps': max_steps,
                'patience': patience,
                'min_distance': min_distance,
                'mode': mode,
                'device': device
            },
            'added_structure_ids': added_structure_ids,
            'results': results
        }, f, indent=2)
    
    return results, added_structure_ids


def create_vasp_jobs_from_aa_results(
    input_dir: str,
    output_dir: str = None,
    vasp_profile: str = "vasp-static",
    hpc_profile: str = "PSFC-GPU",
    structures_per_traj: int = 5,
):
    """
    Create VASP jobs for structures that were created via adversarial attack optimization.
    
    Args:
        input_dir: Directory containing AA optimization results (either gradient or MC based)
        output_dir: Directory to create VASP jobs in (defaults to input_dir/vasp_jobs)
        vasp_profile: VASP settings profile to use
        hpc_profile: HPC profile to use
        structures_per_traj: Number of structures to select per trajectory (not used if finding by ID)
    """
    from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase
    
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise ValueError(f"Input directory not found: {input_path}")
    
    # Default output directory inside input directory
    if output_dir is None:
        output_path = input_path / "vasp_jobs"
    else:
        output_path = Path(output_dir).resolve()
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Find optimization summary files
    summary_files = list(input_path.glob("**/optimization_summary.json"))
    
    if not summary_files:
        # Try to find structures directly from database
        # This is useful if structures were added to DB but don't have VASP calcs yet
        query = {
            'config_type': ['aa-gradient', 'aa-monte-carlo'],
            'vasp': None
        }
        structure_ids = db_manager.find_structures_without_calculation(model_type='vasp*')
        
        if not structure_ids:
            print("[INFO] No AA structures found without VASP calculations.")
            return
        
        print(f"[INFO] Found {len(structure_ids)} AA structures without VASP calculations.")
        
        # Process each structure
        for structure_id in structure_ids:
            atoms = db_manager.get_structure(structure_id)
            metadata = db_manager.get_structure_metadata(structure_id)
            
            # Create job directory name
            config_type = metadata.get('config_type', 'aa')
            parent_id = metadata.get('parent_structure_id')
            variance = metadata.get('final_variance')
            
            job_name = f"aa_id{structure_id}"
            if parent_id:
                job_name += f"_from{parent_id}"
            if variance:
                job_name += f"_var{variance:.4f}"
                
            job_dir = output_path / f"job_{structure_id}_{job_name}"
            
            # Create VASP job
            prepare_vasp_job_from_ase(
                atoms=atoms,
                vasp_profile_name=vasp_profile,
                hpc_profile_name=hpc_profile,
                output_dir=str(job_dir),
                auto_kpoints=True,
                job_name=job_name
            )
            
            # Save additional metadata
            with open(job_dir / "job_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"[INFO] Created VASP job for structure {structure_id} in {job_dir}")
    
    else:
        # Process from optimization summary files
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                # Check if database IDs are available
                added_structure_ids = summary.get('added_structure_ids', [])
                
                if added_structure_ids:
                    # Use database IDs to generate VASP jobs
                    for structure_id in added_structure_ids:
                        try:
                            atoms = db_manager.get_structure(structure_id)
                            metadata = db_manager.get_structure_metadata(structure_id)
                            
                            # Create job directory name
                            config_type = metadata.get('config_type', 'aa')
                            parent_id = metadata.get('parent_structure_id')
                            variance = metadata.get('final_variance')
                            
                            job_name = f"aa_id{structure_id}"
                            if parent_id:
                                job_name += f"_from{parent_id}"
                            if variance:
                                job_name += f"_var{variance:.4f}"
                                
                            job_dir = output_path / f"job_{structure_id}_{job_name}"
                            
                            # Create VASP job
                            prepare_vasp_job_from_ase(
                                atoms=atoms,
                                vasp_profile_name=vasp_profile,
                                hpc_profile_name=hpc_profile,
                                output_dir=str(job_dir),
                                auto_kpoints=True,
                                job_name=job_name
                            )
                            
                            # Save additional metadata
                            with open(job_dir / "job_metadata.json", 'w') as f:
                                json.dump(metadata, f, indent=2)
                                
                            print(f"[INFO] Created VASP job for structure {structure_id} in {job_dir}")
                        except Exception as e:
                            print(f"[ERROR] Failed to create VASP job for structure {structure_id}: {e}")
                
                else:
                    # No database IDs, use XYZ file directly
                    results_dir = summary_file.parent
                    
                    # Try to find optimized structures
                    optimized_files = list(results_dir.glob("*_optimized.xyz"))
                    
                    for opt_file in optimized_files:
                        try:
                            atoms = ase.io.read(opt_file)
                            struct_name = atoms.info.get('structure_name', opt_file.stem)
                            
                            # Extract metadata from filename and atoms.info
                            variance = atoms.info.get('final_variance')
                            
                            job_name = f"aa_{struct_name}"
                            if variance:
                                job_name += f"_var{variance:.4f}"
                                
                            job_dir = output_path / f"job_{struct_name}_{job_name}"
                            
                            # Create VASP job
                            prepare_vasp_job_from_ase(
                                atoms=atoms,
                                vasp_profile_name=vasp_profile,
                                hpc_profile_name=hpc_profile,
                                output_dir=str(job_dir),
                                auto_kpoints=True,
                                job_name=job_name
                            )
                                
                            print(f"[INFO] Created VASP job for structure {struct_name} in {job_dir}")
                        except Exception as e:
                            print(f"[ERROR] Failed to create VASP job for file {opt_file}: {e}")
            
            except Exception as e:
                print(f"[ERROR] Failed to process summary file {summary_file}: {e}")
    
    print(f"[INFO] VASP job preparation completed. Jobs created in {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial attack optimization on structures."
    )
    parser.add_argument(
        "xyz_file",
        help="Input XYZ file containing structures"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save optimization results"
    )
    parser.add_argument(
        "--model_paths",
        nargs='+',
        required=True,
        help="Paths to model files"
    )
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="Use gradient-based optimization instead of Metropolis"
    )
    
    # Parameters for Metropolis optimization
    parser.add_argument(
        "--temperature",
        type=float,
        default=1200.0,
        help="Temperature for adversarial optimization (K) or energy scaling (eV)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum optimization steps per structure (Metropolis mode)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Stop if no improvement after this many steps (Metropolis mode)"
    )
    
    # Parameters for gradient-based optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient ascent (gradient mode)"
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=60,
        help="Number of optimization iterations (gradient mode)"
    )
    parser.add_argument(
        "--include_probability",
        action="store_true",
        help="Include probability term in adversarial loss (gradient mode)"
    )
    parser.add_argument(
        "--use_autograd",
        action="store_true",
        help="Use Hessian-based gradient calculation (faster but requires MACECalculator.get_hessian)"
    )
    
    # Common parameters
    parser.add_argument(
        "--min_distance",
        type=float,
        default=2.0,
        help="Minimum allowed distance between atoms (Å)"
    )
    parser.add_argument(
        "--mode",
        choices=['all', 'single'],
        default='all',
        help="Optimization mode ('all' or 'single' atom) (Metropolis mode)"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda'],
        default='cuda',
        help="Device to run on"
    )
    parser.add_argument(
        "--save_to_database",
        action="store_true",
        help="Save optimized structures to database"
    )
    parser.add_argument(
        "--database_id",
        type=int,
        help="Structure ID in database (if optimizing a specific structure)"
    )
    parser.add_argument(
        "--config_type",
        type=str,
        help="Configuration type label for database (defaults based on method)"
    )
    
    args = parser.parse_args()
    
    if args.gradient:
        # Use gradient-based optimization
        from forge.core.adversarial_attack import (
            GradientAdversarialCalculator,
            GradientAscentOptimizer
        )
        
        config_type = args.config_type or "aa-gradient"
        
        run_gradient_aa_optimization(
            xyz_file=args.xyz_file,
            output_dir=args.output_dir,
            model_paths=args.model_paths,
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations,
            min_distance=args.min_distance,
            include_probability=args.include_probability,
            temperature=args.temperature,
            device=args.device,
            use_autograd=args.use_autograd,
            save_to_database=args.save_to_database,
            database_id=args.database_id,
            config_type=config_type
        )
    else:
        # Use original Metropolis optimization
        config_type = args.config_type or "aa-monte-carlo"
        
        run_aa_optimization(
            xyz_file=args.xyz_file,
            output_dir=args.output_dir,
            model_paths=args.model_paths,
            temperature=args.temperature,
            max_steps=args.max_steps,
            patience=args.patience,
            min_distance=args.min_distance,
            mode=args.mode,
            device=args.device,
            save_to_database=args.save_to_database,
            database_id=args.database_id,
            config_type=config_type
        )

def vasp_jobs_from_aa_main():
    """Command-line interface for creating VASP jobs from AA optimization results."""
    parser = argparse.ArgumentParser(
        description="Create VASP jobs for structures optimized with adversarial attack."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing AA optimization results or output directory name"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to create VASP jobs in (defaults to input_dir/vasp_jobs)"
    )
    parser.add_argument(
        "--vasp_profile",
        default="static",
        help="VASP settings profile to use"
    )
    parser.add_argument(
        "--hpc_profile",
        default="default",
        help="HPC profile to use"
    )
    parser.add_argument(
        "--structures_per_traj",
        type=int,
        default=5,
        help="Number of structures to select per trajectory (when not using database)"
    )
    
    args = parser.parse_args()
    create_vasp_jobs_from_aa_results(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vasp_profile=args.vasp_profile,
        hpc_profile=args.hpc_profile,
        structures_per_traj=args.structures_per_traj
    )


def db_test_workflow_main():
    """Command-line interface for testing the database workflow with AA structures."""
    parser = argparse.ArgumentParser(
        description="Test database workflow with AA structures"
    )
    parser.add_argument(
        "--elements", 
        nargs='+',
        help="Elements to filter by"
    )
    parser.add_argument(
        "--structure_type",
        help="Structure type to filter by"
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=2,
        help="Number of structures to process"
    )
    parser.add_argument(
        "--output_dir",
        default="aa_test_workflow",
        help="Directory to save workflow results"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing MACE model files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Step 1: Initialize database connection
    db_manager = DatabaseManager()
    
    # Step 2: Find structures without VASP calculations
    if args.elements or args.structure_type:
        query_kwargs = {}
        if args.elements:
            query_kwargs['elements'] = args.elements
        if args.structure_type:
            query_kwargs['structure_type'] = args.structure_type
        
        structure_ids = db_manager.find_structures(**query_kwargs)
        if not structure_ids:
            print(f"[ERROR] No structures found matching criteria.")
            return
    else:
        # Find any structures without VASP calculations
        structure_ids = db_manager.find_structures_without_calculation(model_type='vasp*')
        if not structure_ids:
            print(f"[ERROR] No structures found without VASP calculations.")
            return
    
    # Limit to requested number
    structure_ids = structure_ids[:args.num_structures]
    print(f"[INFO] Selected {len(structure_ids)} structures for testing.")
    
    # Step 3: Create output directory
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 4: Export structures to XYZ
    xyz_file = output_path / "test_structures.xyz"
    atoms_list = []
    
    for struct_id in structure_ids:
        try:
            atoms = db_manager.get_structure(struct_id)
            atoms.info['structure_id'] = struct_id
            atoms.info['structure_name'] = f"struct_id_{struct_id}" # TODO NEED TO RESOLVE IF WE USE structure_name, structure_id, or parent_id or something else?
            atoms_list.append(atoms)
        except Exception as e:
            print(f"[ERROR] Failed to retrieve structure {struct_id}: {e}")
    
    if not atoms_list:
        print(f"[ERROR] No valid structures could be retrieved.")
        return
    
    ase.io.write(xyz_file, atoms_list)
    print(f"[INFO] Wrote {len(atoms_list)} structures to {xyz_file}")
    
    # Step 5: Run AA optimization
    models_dir = Path(args.model_dir).resolve()
    model_paths = [str(p) for p in models_dir.glob("*.model")]
    if not model_paths:
        print(f"[ERROR] No model files found in {models_dir}")
        return
    
    aa_output_dir = output_path / "aa_output"
    aa_output_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] Running gradient-based AA optimization...")
    run_gradient_aa_optimization(
        xyz_file=str(xyz_file),
        output_dir=str(aa_output_dir),
        model_paths=model_paths,
        learning_rate=0.01,
        n_iterations=10,  # Small number for testing
        min_distance=1.5,
        temperature=1000,
        include_probability=False,
        device="cpu" if not torch.cuda.is_available() else "cuda",
        save_to_database=True,
        debug=False
    )
    import os
    os.environ["VASP_PP_PATH"] = "/home/myless/Packages/VASP/POTCAR_64_PBE/"
    # Step 6: Create VASP jobs
    vasp_output_dir = output_path / "vasp_jobs"
    create_vasp_jobs_from_aa_results(
        input_dir=str(aa_output_dir),
        output_dir=str(vasp_output_dir),
        vasp_profile="static",
        hpc_profile="PSFC-GPU"
    )
    
    print(f"[INFO] Test workflow completed. Results in {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-vasp-jobs":
        # Remove the subcommand from arguments
        sys.argv.pop(1)
        vasp_jobs_from_aa_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "test-workflow":
        # Remove the subcommand from arguments
        sys.argv.pop(1)
        # Import torch here to check for CUDA
        import torch
        db_test_workflow_main()
    else:
        main()