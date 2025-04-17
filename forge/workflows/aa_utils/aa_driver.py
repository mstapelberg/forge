#!/usr/bin/env python
"""
Core functions for the adversarial attack workflow.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import ase.io
import shutil
import glob
import numpy as np
from ase import Atoms
from forge.core.database import DatabaseManager
from forge.core.adversarial_attack import AdversarialCalculator, DisplacementGenerator, AdversarialOptimizer
from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase
from forge.workflows.adversarial_attack.slurm_templates import (
    get_variance_calculation_script,
    get_gradient_aa_script,
    get_monte_carlo_aa_script
)


def select_structures_from_db(
    elements: List[str],
    structure_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    structure_ids: Optional[List[int]] = None,
    debug: bool = False
) -> Tuple[List[Atoms], List[float]]:
    """
    Selects structure Atoms objects and their energies from the database.
    Returns (list of Atoms objects, list of energies).
    """
    db_manager = DatabaseManager()
    selected_atoms_list = []
    energies = []

    if structure_ids:
        structure_ids = structure_ids
        print(f"[INFO] Selected structure IDs from input: {structure_ids}")
    else:
        query_kwargs = {}
        if elements:
            query_kwargs['elements'] = elements
            print(f"[INFO] Filtering by elements: {elements}")
        if structure_type:
            query_kwargs['structure_type'] = structure_type
            print(f"[INFO] Filtering by structure type: {structure_type}")
        if composition_constraints:
            try:
                constraints = json.loads(composition_constraints)
                query_kwargs['composition_constraints'] = constraints
                print(f"[INFO] Filtering by composition constraints: {constraints}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for composition_constraints: {e}")

        if query_kwargs:
            structure_ids = db_manager.find_structures(**query_kwargs, debug=debug)
            print(f"[INFO] Found {len(structure_ids)} structure IDs matching criteria.")
        else:
            raise ValueError("No structure selection criteria provided.")

    if not structure_ids:
        raise ValueError("No structures selected. Check selection criteria.")

    # Use the new method to get atoms with calculations directly
    selected_atoms_list = db_manager.get_atoms_with_calculations(structure_ids, model_type='vasp-static')
    
    # Filter out any structures without energy values and collect energies
    filtered_atoms_list = []
    for atoms in selected_atoms_list:
        if 'energy' in atoms.info and atoms.info['energy'] is not None:
            filtered_atoms_list.append(atoms)
            energies.append(float(atoms.info['energy']))
        elif debug:
            struct_id = atoms.info.get('structure_id', 'unknown')
            print(f"[DEBUG] Skipping structure {struct_id}: Energy is None.")
    
    selected_atoms_list = filtered_atoms_list
    
    if debug:
        print(f"[DEBUG] Selected {len(selected_atoms_list)} structures with energies.")

    return selected_atoms_list, energies


def prepare_variance_calculation_jobs(
    selected_atoms_list: List[Atoms],
    energies: List[float],
    output_dir: Path,
    model_dir: Path,
    n_batches: int,
    device: str = "cpu",
) -> Path:
    """
    Prepares variance calculation jobs.  Saves energies to a JSON file.
    """
    variance_calculation_dir = output_dir / "variance_calculations"
    variance_calculation_dir.mkdir(parents=True, exist_ok=True)
    variance_results_dir = output_dir / "variance_results"
    variance_results_dir.mkdir(parents=True, exist_ok=True)

    # --- Save energies to JSON ---
    energies_file = variance_calculation_dir / "energies.json"
    with open(energies_file, 'w') as f:
        json.dump(energies, f, indent=2)
    print(f"[INFO] Saved energies to {energies_file}")

    # --- Create 'models' directory and copy models ---
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    copied_model_paths = []

    model_dir_path = Path(model_dir).resolve()
    if not model_dir_path.is_dir():
        raise ValueError(f"Model directory not found: {model_dir_path}")

    for item in os.listdir(model_dir_path):
        if item.endswith(".model"):
            model_path = model_dir_path / item
            dest_path = models_dir / item
            shutil.copy2(model_path, dest_path)
            copied_model_paths.append(f"models/{item}")
    model_paths_str = ' '.join(copied_model_paths)

    # --- Split structures into batches ---
    n_structures = len(selected_atoms_list)
    structures_per_batch = n_structures // n_batches
    remaining_structures = n_structures % n_batches

    start_index = 0
    for batch_id in range(n_batches):
        batch_dir = variance_calculation_dir / f"batch_{batch_id}"
        batch_dir.mkdir(exist_ok=True)
        xyz_file_path = batch_dir / f"batch_{batch_id}.xyz"

        end_index = start_index + structures_per_batch
        if batch_id < remaining_structures:
            end_index += 1

        batch_atoms_list = selected_atoms_list[start_index:end_index]
        start_index = end_index

        # Assign structure names
        for i, atoms in enumerate(batch_atoms_list):
            structure_id = atoms.info.get('structure_id')
            if structure_id:
                atoms.info['structure_name'] = str(structure_id)
            else:
                atoms.info['structure_name'] = f'batch_{batch_id}_index_{i}'

        ase.io.write(xyz_file_path, batch_atoms_list)
        print(f"[INFO] Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {xyz_file_path}")

    # --- Generate SLURM array job script ---
    slurm_script_path = variance_calculation_dir / "variance_calculation_array.slurm"
    
    # Get the SLURM script from the template
    slurm_content = get_variance_calculation_script(
        output_dir=str(variance_calculation_dir.relative_to(variance_calculation_dir.parent)),
        ensemble_path="models",
        structure_file="${BATCH_DIR}/batch_${BATCH_ID}.xyz",
        n_models=len(copied_model_paths),
        array_range=f"0-{n_batches - 1}",
        compute_forces=True,
        time="24:00:00",
        mem="32G",
        cpus_per_task=8,
        gpus_per_task=1,
    )
    
    with open(slurm_script_path, "w") as f:
        f.write(slurm_content)
    print(f"[INFO] Created SLURM array script: {slurm_script_path}")

    return variance_results_dir


def combine_variance_results(variance_dir: Path, debug: bool = False) -> List[Tuple[str, float, Path]]:
    """
    Combine variance results from all batches and sort by variance.
    
    Args:
        variance_dir: Directory containing variance calculation results
        
    Returns:
        List of tuples (structure_name, variance, xyz_file_path)
    """
    all_results = []
    
    # Find all variance JSON files
    for json_file in variance_dir.glob("*_variances.json"):
        print(f"Reading variance results from {json_file}")
        with open(json_file, 'r') as f:
            batch_results = json.load(f)
            
        # Get batch number from filename
        batch_id = int(json_file.stem.split('_')[1])
        batch_dir = variance_dir.parent / "variance_calculations" / f"batch_{batch_id}"
        
        # TODO right now the xyz_path is not unique, multiple structures are in the same xyz file,
        # we need to have each struct_name and variance pointing to a unique atoms object
        # right now it's pointing to the batch_id.xyz file that contains the atoms object and many others as well 
        for struct_name, variance in batch_results.items():
            # Find corresponding XYZ file in batch directory
            xyz_file = batch_dir / f"batch_{batch_id}.xyz"
            if not xyz_file.exists():
                print(f"Warning: Could not find XYZ file {xyz_file}")
                continue
                
            # Read the XYZ file to find the structure
            atoms_list = ase.io.read(xyz_file, ':')
            for atoms in atoms_list:
                if atoms.info.get('structure_name') == struct_name:
                    all_results.append((struct_name, variance, xyz_file)) # add the structure name, variance, and xyz file path 
                    print(f"Found structure {struct_name} with variance {variance:.6f}")
                    break
    
    # Sort by variance (highest first)
    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    if debug:
        print(f"\nSorted variance results:")
        for name, var, _ in sorted_results:
            print(f"  {name}: {var:.6f}")
    return sorted_results

def prepare_gradient_aa_optimization(
    input_directory: str,
    model_dir: str,
    n_structures: int,
    n_batches: int,
    learning_rate: float = 0.01,
    n_iterations: int = 60,
    min_distance: float = 1.5,
    include_probability: bool = False,
    temperature: float = 0.86,
    device: str = "cuda",
    use_autograd: bool = False,
    debug: bool = False,
) -> None:
    """
    Prepares and launches gradient-based adversarial attack optimization jobs.
    
    This function has been enhanced to better leverage the database for structure tracking.
    It extracts structure IDs from variance results and retrieves structures directly from the database.
    """
    print("[INFO] Starting gradient-based adversarial attack optimization preparation...")
    
    # Setup directories
    input_path = Path(input_directory).resolve()
    if not input_path.is_dir():
        raise ValueError(f"Input directory not found: {input_path}")
        
    aa_output_dir = input_path.parent / "gradient_aa_optimization"
    aa_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create slurm logs directory
    slurm_logs_dir = aa_output_dir / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Copy models to aa_output_dir
    models_dir = aa_output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    copied_model_paths = []
    
    model_dir_path = Path(model_dir).resolve()
    if not model_dir_path.is_dir():
        raise ValueError(f"Model directory not found: {model_dir_path}")
        
    for item in os.listdir(model_dir_path):
        if item.endswith(".model"):
            model_path = model_dir_path / item
            dest_path = models_dir / item
            shutil.copy2(model_path, dest_path)
            copied_model_paths.append(f"models/{item}")
    
    # Combine and sort variance results
    print("[INFO] Combining variance results...")
    sorted_results = combine_variance_results(input_path, debug=debug)
    print(f"[INFO] Found {len(sorted_results)} structures with variance results")
    
    # Select top N structures
    selected_structures = sorted_results[:n_structures]
    print(f"[INFO] Selected top {len(selected_structures)} structures")
    
    # Extract structure IDs from the selected structures
    structure_ids = []
    for struct_name, variance, _ in selected_structures:
        # Extract structure ID from name if possible (format: struct_id_XXX)
        if struct_name.startswith("struct_id_"):
            try:
                structure_id = int(struct_name.split("_")[-1])
                structure_ids.append((structure_id, variance))
            except ValueError:
                if debug:
                    print(f"[DEBUG] Could not extract structure ID from name: {struct_name}")
                continue
    
    if not structure_ids:
        raise ValueError("No valid structure IDs found in variance results.")
    
    print(f"[INFO] Extracted {len(structure_ids)} structure IDs from variance results")
    
    # Create batches
    structures_per_batch = len(structure_ids) // n_batches
    remaining_structures = len(structure_ids) % n_batches
    
    start_index = 0
    for batch_id in range(n_batches):
        batch_dir = aa_output_dir / f"batch_{batch_id}"
        batch_dir.mkdir(exist_ok=True)
        
        end_index = start_index + structures_per_batch
        if batch_id < remaining_structures:
            end_index += 1
        
        batch_struct_ids = structure_ids[start_index:end_index]
        start_index = end_index
        
        # Get structures with calculations directly from database
        ids_only = [sid for sid, _ in batch_struct_ids]
        batch_atoms_list = db_manager.get_atoms_with_calculations(ids_only, model_type='vasp-static')
        
        # Add variance information to atoms objects
        for i, atoms in enumerate(batch_atoms_list):
            structure_id, variance = batch_struct_ids[i]
            atoms.info['structure_id'] = structure_id
            atoms.info['structure_name'] = str(structure_id)
            atoms.info['initial_variance'] = variance
        
        # Create batch XYZ file
        batch_xyz = batch_dir / f"batch_{batch_id}.xyz"
        ase.io.write(batch_xyz, batch_atoms_list)
        print(f"[INFO] Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {batch_xyz}")
        
        # Save batch metadata with structure IDs
        batch_meta = {
            'structures': []
        }
        
        for i, atoms in enumerate(batch_atoms_list):
            structure_id = atoms.info['structure_id']
            structure_meta = {
                'name': str(structure_id),
                'variance': atoms.info['initial_variance'],
                'structure_id': structure_id
            }
            batch_meta['structures'].append(structure_meta)
            
        with open(batch_dir / 'batch_metadata.json', 'w') as f:
            json.dump(batch_meta, f, indent=2)
    
    # Generate SLURM array job script
    slurm_script_path = aa_output_dir / "gradient_aa_optimization_array.slurm"
    
    # Get the SLURM script from the template
    slurm_content = get_gradient_aa_script(
        output_dir=".",
        ensemble_path="models",
        structure_file="${BATCH_DIR}/batch_${BATCH_ID}.xyz",
        n_steps=n_iterations,
        step_size=learning_rate,
        array_range=f"0-{n_batches - 1}",
        time="24:00:00",
        mem="32G",
        cpus_per_task=8,
        gpus_per_task=1,
        use_probability_weighting=include_probability,
        temperature=temperature,
        force_only=True,
        save_trajectory=True,
        save_forces=True
    )
    
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_content)
    print(f"[INFO] Created SLURM array script: {slurm_script_path}")
    print("[INFO] Run the generated SLURM script to start gradient-based AA optimization.")
    
def prepare_aa_optimization(
    input_directory: str,
    model_dir: str,
    n_structures: int,
    n_batches: int,
    temperature: float = 1200.0,
    max_steps: int = 50,
    patience: int = 25,
    min_distance: float = 2.0,
    mode: str = "all",
    device: str = "cuda",
    debug: bool = False,
) -> None:
    """
    Prepares and launches adversarial attack optimization jobs.
    """
    print("[INFO] Starting adversarial attack optimization preparation...")
    
    # Setup directories
    input_path = Path(input_directory).resolve()
    if not input_path.is_dir():
        raise ValueError(f"Input directory not found: {input_path}")
        
    aa_output_dir = input_path.parent / "aa_optimization"
    aa_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy models to aa_output_dir
    models_dir = aa_output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    copied_model_paths = []
    
    model_dir_path = Path(model_dir).resolve()
    if not model_dir_path.is_dir():
        raise ValueError(f"Model directory not found: {model_dir_path}")
        
    for item in os.listdir(model_dir_path):
        if item.endswith(".model"):
            model_path = model_dir_path / item
            dest_path = models_dir / item
            shutil.copy2(model_path, dest_path)
            copied_model_paths.append(f"models/{item}")
    model_paths_str = ' '.join(copied_model_paths)
    
    # Combine and sort variance results
    print("[INFO] Combining variance results...")
    sorted_results = combine_variance_results(input_path)
    print(f"[INFO] Found {len(sorted_results)} structures with variance results")
    
    # Select top N structures
    selected_structures = sorted_results[:n_structures]
    print(f"[INFO] Selected top {len(selected_structures)} structures")
    
    # Create batches
    structures_per_batch = n_structures // n_batches
    remaining_structures = n_structures % n_batches
    
    start_index = 0
    for batch_id in range(n_batches):
        batch_dir = aa_output_dir / f"batch_{batch_id}"
        batch_dir.mkdir(exist_ok=True)
        
        end_index = start_index + structures_per_batch
        if batch_id < remaining_structures:
            end_index += 1
            
        batch_structures = selected_structures[start_index:end_index]
        start_index = end_index
        
        # Create batch XYZ file
        batch_xyz = batch_dir / f"batch_{batch_id}.xyz"
        batch_atoms_list = []
        
        for struct_name, variance, xyz_path in batch_structures:
            atoms = ase.io.read(xyz_path)
            atoms.info['structure_name'] = struct_name
            atoms.info['initial_variance'] = variance
            batch_atoms_list.append(atoms)
            
        ase.io.write(batch_xyz, batch_atoms_list)
        print(f"[INFO] Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {batch_xyz}")
        
        # Save batch metadata
        batch_meta = {
            'structures': [
                {
                    'name': name,
                    'variance': var,
                    'xyz_path': str(xyz_path.relative_to(input_path.parent))
                }
                for name, var, xyz_path in batch_structures
            ]
        }
        with open(batch_dir / 'batch_metadata.json', 'w') as f:
            json.dump(batch_meta, f, indent=2)
    
    # Generate SLURM array job script
    slurm_script_path = aa_output_dir / "aa_optimization_array.slurm"
    
    # Get the SLURM script from the template
    slurm_content = get_monte_carlo_aa_script(
        output_dir=".",
        ensemble_path="models",
        structure_file="${BATCH_DIR}/batch_${BATCH_ID}.xyz",
        n_steps=max_steps,
        max_displacement=0.1,
        array_range=f"0-{n_batches - 1}",
        time="24:00:00",
        mem="32G",
        cpus_per_task=8,
        gpus_per_task=1,
        temperature=temperature,
        force_only=True,
        save_trajectory=True
    )
    
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_content)
    print(f"[INFO] Created SLURM array script: {slurm_script_path}")
    print("[INFO] Run the generated SLURM script to start AA optimization.")


def select_structures_from_trajectory(
    trajectory_file: Path,
    n_structures: int,
    metadata_file: Optional[Path] = None,
) -> List[Tuple[Atoms, int, float]]:
    """
    Select structures from an AA trajectory, working backwards from the final structure.
    
    Args:
        trajectory_file: Path to XYZ trajectory file
        n_structures: Number of structures to select
        metadata_file: Optional path to optimization summary JSON
        
    Returns:
        List of tuples (atoms, step_number, variance)
    """
    # Read trajectory
    trajectory = ase.io.read(trajectory_file, ':')
    if not trajectory:
        raise ValueError(f"No structures found in {trajectory_file}")
    
    # Load metadata if available
    step_variances = {}
    if metadata_file and metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            # Extract step variances from accepted moves in results
            for result in metadata.get('results', []):
                if trajectory_file.stem.startswith(result['structure_name']):
                    # Each accepted move represents a step in the trajectory
                    # The variance at each step is the "current_variance" at that point
                    step_variances = {
                        step: var
                        for step, var in enumerate(result.get('step_variances', []))
                    }
                    break
    
    # Calculate indices working backwards from end
    n_steps = len(trajectory)
    if n_steps < n_structures + 1:  # +1 because we skip the first structure
        raise ValueError(
            f"Trajectory only has {n_steps} structures, "
            f"cannot select {n_structures} structures"
        )
    
    # Calculate evenly spaced indices, working backwards from the end
    indices = [
        n_steps - 1 - i * (n_steps - 2) // (n_structures - 1)
        for i in range(n_structures)
    ]
    
    # Select structures
    selected = []
    for idx in indices:
        atoms = trajectory[idx]
        variance = step_variances.get(idx, atoms.info.get('variance', None))
        # Add temperature to atoms.info for later use
        if metadata_file and metadata_file.exists():
            atoms.info['temperature'] = metadata.get('parameters', {}).get('temperature')
        selected.append((atoms, idx, variance))
    
    return selected


def prepare_vasp_jobs(
    input_directory: str,
    output_directory: str,
    vasp_profile: str = "static",
    hpc_profile: str = "default",
    structures_per_traj: int = 1,
    debug: bool = False,
) -> None:
    """
    Creates VASP jobs for structures optimized by adversarial attack.
    
    Args:
        input_directory: Directory containing AA optimization results
        output_directory: Directory to create VASP jobs in
        vasp_profile: VASP settings profile to use
        hpc_profile: HPC profile to use
        structures_per_traj: Number of structures to select per trajectory
        debug: Enable debug output
    """
    print("[INFO] Starting VASP job preparation from AA results...")
    
    input_path = Path(input_directory).resolve()
    output_path = Path(output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Find all AA trajectories
    batch_dirs = [d for d in input_path.glob("batch_*/") if d.is_dir()]
    for batch_dir in batch_dirs:
        batch_id = int(batch_dir.name.split('_')[1])
        aa_results_dir = batch_dir / "aa_results"
        if not aa_results_dir.exists():
            print(f"[WARNING] No AA results found in {batch_dir}")
            continue
            
        # Process each trajectory in the batch
        for traj_file in aa_results_dir.glob("*_adversarial.xyz"):
            # Get parent structure name from trajectory filename
            parent_name = traj_file.stem.replace('_adversarial', '')
            meta_file = aa_results_dir / "optimization_summary.json"
            
            try:
                # Select structures from trajectory
                selected_structures = select_structures_from_trajectory(
                    traj_file,
                    structures_per_traj,
                    meta_file
                )
                
                # Process each selected structure
                for atoms, step, variance in selected_structures:
                    # Get parent structure metadata
                    parent_id = atoms.info.get('structure_id')
                    if not parent_id:
                        print(f"[WARNING] No parent ID for {traj_file}")
                        continue
                        
                    parent_meta = db_manager.get_structure_metadata(parent_id)
                    
                    # Prepare structure metadata
                    config_type = parent_meta.get('config_type', 'aa')
                    if config_type != 'aa':
                        config_type = f"{config_type}_aa"
                        
                    structure_meta = {
                        "parent_structure_id": parent_id,
                        "parent_config_type": parent_meta.get('config_type'),
                        "config_type": config_type,
                        "aa_temperature": atoms.info.get('temperature'),
                        "aa_step": step,
                        "initial_variance": atoms.info.get('initial_variance'),
                        "step_variance": variance,
                        "batch_id": batch_id
                    }
                    
                    # Add structure to database
                    new_id = db_manager.add_structure(
                        atoms,
                        metadata=structure_meta
                    )
                    
                    # Create VASP job directory name
                    job_name = f"{parent_name}_b{batch_id}_s{step}_T{structure_meta['aa_temperature']}"
                    job_dir = output_path / f"job_{new_id}_{job_name}"
                    
                    # Create VASP job
                    prepare_vasp_job_from_ase(
                        atoms=atoms,
                        vasp_profile_name=vasp_profile,
                        hpc_profile_name=hpc_profile,
                        output_dir=str(job_dir),
                        auto_kpoints=True,
                        DEBUG=debug,
                        job_name=job_name
                    )
                    
                    # Save additional metadata in job directory
                    with open(job_dir / "job_metadata.json", 'w') as f:
                        json.dump(structure_meta, f, indent=2)
                        
                    print(f"[INFO] Created VASP job for structure {new_id} in {job_dir}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {traj_file}: {e}")
                if debug:
                    raise
                continue
    
    print("[INFO] VASP job preparation completed.")


def generate_workflow_readme(output_dir: Path, rel_path: str, model_dir: str, elements: str, n_batches: int, n_structs: int, hpc_profile: str, extra_args: str) -> None:
    """
    Generate a README file explaining the AA workflow steps.
    
    Args:
        output_dir: Path to the workflow directory
        rel_path: Relative path to the workflow directory
        model_dir: Path to the model directory
        elements: Elements used in the workflow (or empty string if using structure_ids)
        n_batches: Number of batches in the workflow
        n_structs: Number of structures in the workflow
        hpc_profile: HPC profile used in the workflow
        extra_args: Extra arguments used in the workflow
    """
    # Handle case where elements is None
    elements_str = ' '.join(elements) if elements else ''
    
    readme_content = f"""# Gradient-Based Adversarial Attack Workflow

This directory contains a gradient-based adversarial attack (AA) workflow. Follow these steps to run the workflow:

## Step 1: Calculate Force Variances

First, run the variance calculation jobs to identify structures with high model disagreement:

```bash
# Navigate to this directory
cd {rel_path}

# Submit the variance calculation SLURM array job
sbatch variance_calculations/variance_calculation_array.slurm
```

This will create variance results in the `variance_results` directory.

## Step 2: Run Gradient-Based Adversarial Attack Optimization

After the variance calculations complete, run the gradient-based AA optimization on the highest-variance structures:

```bash
# From the workflow root directory
forge run-gradient-aa-jobs \\
    --input_directory variance_results \\
    --model_dir {model_dir} \\
    --n_structures {n_structs} \\
    --n_batches {n_batches} \\
    --learning_rate 0.01 \\
    --n_iterations 100 \\
    --min_distance 1.5 \\
    --device cuda

# Submit the gradient-based AA optimization SLURM array job
sbatch gradient_aa_optimization/gradient_aa_optimization_array.slurm
```

This will create AA optimization results in `gradient_aa_optimization/batch_*/aa_results/`.

## Step 3: Create VASP Jobs

Finally, create VASP jobs for the optimized structures:

```bash
forge create-aa-vasp-jobs \\
    --input_directory gradient_aa_optimization \\
    --output_directory vasp_jobs \\
    --vasp_profile static \\
    --hpc_profile {hpc_profile} \\
    --structures_per_traj 5
```

This will create VASP job directories in `vasp_jobs/` that can be submitted to the cluster.

## Directory Structure

- `variance_calculations/`: Initial variance calculation jobs
- `variance_results/`: Results from variance calculations
- `gradient_aa_optimization/`: Gradient-based adversarial attack optimization jobs and results
- `models/`: Copied MACE model files
- `vasp_jobs/`: Generated VASP job directories

## Workflow Parameters

The current workflow was created with:
```bash
forge create-aa-jobs \\
    --output_dir {rel_path} \\
    --model_dir {model_dir} \\
    {f'--elements {elements_str}' if elements_str else ''} \\
    --n_batches {n_batches} \\
    {extra_args}
```

## Legacy Monte Carlo Approach

If you need to use the legacy Monte Carlo approach instead of gradient-based optimization, run:

```bash
# From the workflow root directory
forge run-aa-jobs \\
    --input_directory variance_results \\
    --model_dir {model_dir} \\
    --n_structures {n_structs} \\
    --n_batches {n_batches} \\
    --temperature 1200 \\
    --max_steps 50 \\
    --device cuda

# Submit the Monte Carlo AA optimization SLURM array job
sbatch aa_optimization/aa_optimization_array.slurm
```

For more information about the workflow, see the forge documentation.
"""
    
    # Write README
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme_content)


def prepare_aa_workflow(
    output_dir: str,
    model_dir: str,
    elements: List[str],
    n_batches: int,
    structure_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    structure_ids: Optional[List[int]] = None,
    debug: bool = False,
) -> None:
    """
    Prepares the initial adversarial attack workflow directory and variance calculation jobs.
    """
    print("[INFO] Starting adversarial attack workflow preparation...")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Select structures from database
    print("[INFO] Step 1: Selecting structures...")
    try:
        selected_atoms_list, energies = select_structures_from_db(
            elements=elements,
            structure_type=structure_type,
            composition_constraints=composition_constraints,
            structure_ids=structure_ids,
            debug=debug
        )
    except ValueError as e:
        print(f"[ERROR] Structure selection failed: {e}")
        return
    print(f"[INFO] Selected {len(selected_atoms_list)} structures.")

    # Prepare variance calculation jobs
    print("[INFO] Step 2: Preparing variance calculation jobs...")
    variance_results_dir = prepare_variance_calculation_jobs(
        selected_atoms_list=selected_atoms_list,
        energies=energies,
        output_dir=output_path,
        model_dir=Path(model_dir),
        n_batches=n_batches,
        device="cpu"  # Default to CPU for variance calcs
    )
    
    # Generate README with workflow instructions
    extra_args = []
    if structure_type:
        extra_args.append(f"--structure_type {structure_type}")
    if composition_constraints:
        extra_args.append(f"--composition_constraints '{composition_constraints}'")
    if structure_ids:
        extra_args.append(f"--structure_ids {' '.join(map(str, structure_ids))}")
    if debug:
        extra_args.append("--debug")
        
    generate_workflow_readme(
        output_path,
        rel_path=os.path.relpath(output_path),
        model_dir=os.path.relpath(model_dir),
        elements=elements or [],  # Pass empty list if elements is None
        n_batches=n_batches,
        n_structs=len(selected_atoms_list),
        hpc_profile="PSFC-GPU",  # Could make this configurable
        extra_args=' \\\n    '.join(extra_args) if extra_args else ''
    )
    
    print(f"[INFO] Variance calculation jobs prepared in: {output_path}/variance_calculations")
    print(f"[INFO] Variance results will be saved to: {variance_results_dir}")
    print("[INFO] Created README.md with workflow instructions")
    print("[INFO] Run the generated SLURM script to start variance calculations.")
