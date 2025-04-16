#!/usr/bin/env python
"""
Functions for setting up the stages of the adversarial attack workflow.

This module contains functions called by the forge command-line interface
to prepare directories, input files, and submission scripts for each step:
1. Initial variance calculation.
2. Adversarial attack optimization (Gradient-based or Monte Carlo).
3. VASP job creation for resulting structures.
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
# Removed unused imports: DisplacementGenerator, AdversarialOptimizer, AdversarialCalculator
# Import from the correct file now for VASP prep helper
from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase
from forge.workflows.adversarial_attack.slurm_templates import (
    get_variance_calculation_script,
    get_gradient_aa_script,
    get_monte_carlo_aa_script
)


# --- select_structures_from_db function ---
# (Add/improve docstring)
def select_structures_from_db(
    elements: Optional[List[str]] = None, # Make explicit that it can be None
    structure_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    structure_ids: Optional[List[int]] = None,
    debug: bool = False
) -> Tuple[List[Atoms], List[float]]:
    """
    Selects structure Atoms objects and their energies from the database.

    Retrieves structures based on provided element/type/composition filters
    or a specific list of structure IDs. It preferentially fetches structures
    that have associated VASP static calculation results to obtain energies.

    Args:
        elements: List of elements to filter structures by (ignored if structure_ids provided).
        structure_type: Structure type filter (e.g., 'bulk').
        composition_constraints: JSON string for composition constraints.
        structure_ids: Specific list of structure IDs to fetch.
        debug: Enable debug output.

    Returns:
        Tuple containing:
        - List of selected ase.Atoms objects with info dictionary populated.
        - List of corresponding energies (float) from VASP calculations.

    Raises:
        ValueError: If no selection criteria are provided or no structures are found.
    """
    db_manager = DatabaseManager()
    selected_atoms_list = []
    energies = [] # Store energies corresponding to selected_atoms_list

    if structure_ids:
        # Use provided IDs directly
        print(f"[INFO] Selecting specified structure IDs: {structure_ids}")
    elif elements:
        # Query based on elements and other optional filters
        query_kwargs = {'elements': elements}
        print(f"[INFO] Querying database for structures containing: {elements}")
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

        structure_ids = db_manager.find_structures(**query_kwargs, debug=debug)
        print(f"[INFO] Found {len(structure_ids)} structure IDs matching criteria.")
    else:
        # No criteria provided
        raise ValueError("Structure selection requires either --elements or --structure_ids.")

    if not structure_ids:
        print("[WARNING] No structures found matching the selection criteria.")
        return [], [] # Return empty lists if no structures found

    # Use the method to get atoms with VASP static calculations preferentially
    # This method populates atoms.info['energy'] if found
    selected_atoms_list = db_manager.get_atoms_with_calculations(
        structure_ids,
        model_type='vasp-static' # Prioritize getting energy from static calc
    )

    # Filter out structures where energy couldn't be retrieved and collect energies
    filtered_atoms_list = []
    final_energies = [] # Use a new list for energies corresponding to filtered_atoms_list
    retrieved_ids = set() # Keep track of IDs we got Atoms for

    for atoms in selected_atoms_list:
         struct_id = atoms.info.get('structure_id')
         if struct_id is None:
             if debug: print(f"[DEBUG] Skipping structure with missing ID in info dict.")
             continue # Should not happen if retrieved correctly

         retrieved_ids.add(struct_id)

         # Check if energy was populated by get_atoms_with_calculations
         if 'energy' in atoms.info and atoms.info['energy'] is not None:
             filtered_atoms_list.append(atoms)
             final_energies.append(float(atoms.info['energy']))
         elif debug:
             print(f"[DEBUG] Skipping structure {struct_id}: Energy not found in vasp-static calculation results.")

    # Check if any requested IDs were not retrieved at all
    if debug:
        missed_ids = set(structure_ids) - retrieved_ids
        if missed_ids:
            print(f"[DEBUG] Could not retrieve Atoms objects for IDs: {missed_ids}")

    if not filtered_atoms_list:
         print("[WARNING] No structures with associated energies were found for the selection.")
         # Optionally, we could return structures even without energy, but the current
         # design seems to rely on having energies for the variance calculation step.
         # Let's return empty for now to match previous apparent behavior.
         return [], []

    print(f"[INFO] Successfully retrieved {len(filtered_atoms_list)} structures with energy data.")
    if debug:
        print(f"[DEBUG] Final selected structure IDs: {[a.info['structure_id'] for a in filtered_atoms_list]}")

    return filtered_atoms_list, final_energies


# --- prepare_variance_calculation_jobs function ---
# (Add/improve docstring)
def prepare_variance_calculation_jobs(
    selected_atoms_list: List[Atoms],
    # energies: List[float], # Energies not directly used here, handled by db query now
    output_dir: Path,
    model_dir: Path,
    n_batches: int,
    device: str = "cpu", # Variance calc often feasible on CPU
) -> Path:
    """
    Prepares batch directories and SLURM script for initial variance calculation.

    Splits the selected structures into batches, writes them to XYZ files,
    copies the model ensemble, and generates a SLURM array job script to
    calculate force variances.

    Args:
        selected_atoms_list: List of input ase.Atoms objects.
        output_dir: The main workflow output directory.
        model_dir: Path to the directory containing MACE models.
        n_batches: Number of batches (SLURM jobs) to create.
        device: Device argument for the variance calculation script (currently unused
                in template, but kept for potential future use).

    Returns:
        Path to the directory where variance results will be stored.
    """
    variance_calculation_dir = output_dir / "variance_calculations"
    variance_calculation_dir.mkdir(parents=True, exist_ok=True)
    variance_results_dir = output_dir / "variance_results" # Expected output location
    variance_results_dir.mkdir(parents=True, exist_ok=True)
    slurm_log_dir = variance_calculation_dir / "slurm_logs" # Logs inside job dir
    slurm_log_dir.mkdir(exist_ok=True)

    # --- No need to save energies separately if obtained via get_atoms_with_calculations ---
    # energies_file = variance_calculation_dir / "energies.json"
    # with open(energies_file, 'w') as f:
    #     json.dump(energies, f, indent=2)
    # print(f"[INFO] Saved energies to {energies_file}") # Removed

    # --- Create 'models' directory and copy models ---
    # This copies models into the main workflow dir, accessible by all steps
    models_dest_dir = output_dir / "models"
    models_dest_dir.mkdir(exist_ok=True)
    copied_model_rel_paths = [] # Relative paths from output_dir

    model_src_path = Path(model_dir).resolve()
    if not model_src_path.is_dir():
        raise ValueError(f"Model source directory not found: {model_src_path}")

    num_models_copied = 0
    for item in os.listdir(model_src_path):
        if item.endswith(".model"):
            src_file = model_src_path / item
            dest_file = models_dest_dir / item
            # Avoid re-copying if exists? Use copy2 to preserve metadata.
            if not dest_file.exists() or os.stat(src_file).st_mtime != os.stat(dest_file).st_mtime:
                 shutil.copy2(src_file, dest_file)
            copied_model_rel_paths.append(f"models/{item}") # Path relative to output_dir
            num_models_copied += 1
    print(f"[INFO] Ensured {num_models_copied} models are present in {models_dest_dir}")
    if num_models_copied == 0:
        raise ValueError(f"No *.model files found in source directory: {model_src_path}")

    # --- Split structures into batches ---
    n_structures = len(selected_atoms_list)
    if n_structures == 0:
        print("[WARNING] No structures provided for variance calculation batching.")
        # Create an empty results dir marker? Or just let it be empty.
        return variance_results_dir

    # Ensure n_batches is not greater than n_structures
    if n_batches > n_structures:
        print(f"[WARNING] Number of batches ({n_batches}) > number of structures ({n_structures}). Setting n_batches = {n_structures}.")
        n_batches = n_structures
    if n_batches <= 0:
         print(f"[WARNING] Invalid number of batches ({n_batches}). Setting n_batches = 1.")
         n_batches = 1

    structures_per_batch = n_structures // n_batches
    remaining_structures = n_structures % n_batches

    start_index = 0
    actual_n_batches = 0
    for batch_id in range(n_batches):
        batch_dir = variance_calculation_dir / f"batch_{batch_id}"
        batch_dir.mkdir(exist_ok=True)

        end_index = start_index + structures_per_batch
        if batch_id < remaining_structures:
            end_index += 1

        batch_atoms_list = selected_atoms_list[start_index:end_index]

        # Skip creating batch if no structures assigned (can happen if n_batches > n_structures initially)
        if not batch_atoms_list:
             print(f"[DEBUG] Skipping batch {batch_id} as it has no structures.")
             continue

        actual_n_batches += 1 # Count only non-empty batches
        xyz_file_path = batch_dir / f"batch_{batch_id}.xyz"

        # Assign structure names based on ID for consistency
        for i, atoms in enumerate(batch_atoms_list):
            structure_id = atoms.info.get('structure_id')
            if structure_id:
                atoms.info['structure_name'] = f"struct_{structure_id}" # Consistent naming
            else:
                # Fallback if ID somehow missing (shouldn't happen with db retrieval)
                atoms.info['structure_name'] = f'batch_{batch_id}_index_{i}'

        # Write batch structures to XYZ file
        ase.io.write(xyz_file_path, batch_atoms_list, format="extxyz") # Use extxyz for info dict
        print(f"[INFO] Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {xyz_file_path}")

        start_index = end_index

    if actual_n_batches == 0:
         print("[ERROR] No non-empty batches were created for variance calculation.")
         # Return expected results dir, but it will remain empty
         return variance_results_dir

    # --- Generate SLURM array job script ---
    slurm_script_path = variance_calculation_dir / "variance_calculation_array.slurm"

    # Define paths relative to the *execution directory* of the SLURM script,
    # which we assume is the main workflow output directory `output_dir`.
    # The script will cd into `output_dir` before running commands.
    # TODO: The variance calculation script itself isn't implemented yet.
    # Placeholder: assume a script exists or needs to be created at:
    # forge.workflows.adversarial_attack.calculate_variance_script (or similar)
    variance_command = "echo 'Variance calculation script not implemented yet'" # Placeholder

    # TODO: Need to implement the actual variance calculation script/functionality.
    # For now, the SLURM script generation is illustrative.
    # Let's comment out SLURM script generation until the target script exists.

    # slurm_content = get_variance_calculation_script(
    #     output_dir_rel="variance_results", # Relative path for results storage
    #     log_dir_rel="variance_calculations/slurm_logs", # Relative path for logs
    #     model_dir_rel="models", # Relative path to models dir
    #     batch_script_rel_path="variance_calculations/batch_${SLURM_ARRAY_TASK_ID}/batch_${SLURM_ARRAY_TASK_ID}.xyz",
    #     array_range=f"0-{actual_n_batches - 1}",
    #     n_models=num_models_copied,
    #     compute_forces=True, # Assuming variance is based on forces
    #     # Add other SLURM params as needed: time, mem, cpus, gpus, account, partition
    # )

    # with open(slurm_script_path, "w") as f:
    #     f.write(slurm_content)
    # print(f"[INFO] Created SLURM array script for variance calculation (Placeholder): {slurm_script_path}")
    print(f"[WARNING] SLURM script generation for variance calculation is currently disabled pending implementation of the calculation script.")


    # Return the path where results *should* appear
    return variance_results_dir


# --- combine_variance_results function ---
# (Add/improve docstring)
def combine_variance_results(variance_dir: Path, debug: bool = False) -> List[Tuple[str, float, int]]:
    """
    Combine variance results from batch JSON files and sort by variance.

    Reads '*_variances.json' files within the specified directory, extracts
    structure names (expected format 'struct_ID') and their calculated variances,
    and returns a list sorted from highest to lowest variance.

    Args:
        variance_dir: Directory containing the variance result JSON files
                      (e.g., 'variance_results').
        debug: Enable debug output.

    Returns:
        List of tuples: (structure_name, variance, structure_id), sorted by
        variance in descending order. Returns empty list if no results found.
    """
    all_results = []
    processed_struct_names = set()

    # Find all variance JSON files produced by the (hypothetical) variance calc script
    variance_files = list(variance_dir.glob("*_variances.json"))
    if not variance_files:
        print(f"[WARNING] No variance result files (*_variances.json) found in {variance_dir}")
        return []

    print(f"[INFO] Found {len(variance_files)} variance result files in {variance_dir}")

    for json_file in variance_files:
        if debug: print(f"[DEBUG] Reading variance results from {json_file}")
        try:
            with open(json_file, 'r') as f:
                batch_results = json.load(f) # Expected format: { "struct_ID": variance_value, ... }
        except json.JSONDecodeError:
            print(f"[WARNING] Could not decode JSON from file: {json_file}. Skipping.")
            continue
        except Exception as e:
            print(f"[WARNING] Error reading file {json_file}: {e}. Skipping.")
            continue

        # Process results in the batch file
        for struct_name, variance in batch_results.items():
            if struct_name in processed_struct_names:
                 print(f"[WARNING] Duplicate structure name '{struct_name}' found across batch results. Using first encountered value.")
                 continue

            # Attempt to extract structure ID from name
            try:
                 if not struct_name.startswith("struct_"):
                     raise ValueError("Name does not start with 'struct_'")
                 structure_id = int(struct_name.split('_')[1])
            except (IndexError, ValueError):
                 print(f"[WARNING] Could not extract structure ID from name: '{struct_name}'. Skipping.")
                 continue

            # Validate variance value
            try:
                variance_float = float(variance)
            except (ValueError, TypeError):
                print(f"[WARNING] Invalid variance value '{variance}' for structure '{struct_name}'. Skipping.")
                continue

            all_results.append((struct_name, variance_float, structure_id))
            processed_struct_names.add(struct_name)
            if debug: print(f"[DEBUG] Added structure {struct_name} (ID: {structure_id}) with variance {variance_float:.6f}")

    if not all_results:
        print("[WARNING] No valid variance results could be extracted from the files.")
        return []

    # Sort by variance (highest first)
    # Sort key: lambda x: x[1] (the variance value)
    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)

    print(f"[INFO] Combined and sorted {len(sorted_results)} structure variance results.")
    if debug:
        print(f"\n[DEBUG] Top 5 variance results:")
        for name, var, sid in sorted_results[:5]:
            print(f"  ID: {sid:<6} Name: {name:<15} Variance: {var:.6f}")

    # Return list of (name, variance, id) tuples
    return sorted_results


# --- prepare_gradient_aa_optimization function ---
# (Add/improve docstring)
def prepare_gradient_aa_optimization(
    input_directory: str, # Should be variance_results dir
    model_dir: str, # Original model source dir (used for path in README)
    n_structures: int,
    n_batches: int = 1,
    learning_rate: float = 0.01, # Pass through for summary/README
    n_iterations: int = 60,   # Pass through for summary/README
    min_distance: float = 1.5,    # Pass through for summary/README
    include_probability: bool = False,# Pass through for summary/README
    temperature: float = 0.86, # Pass through for summary/README
    device: str = "cuda", # Device for SLURM request
    # Removed use_autograd as it wasn't used
    debug: bool = False,
) -> None:
    """
    Prepares batch directories and SLURM script for Gradient-Based AA optimization.

    Reads combined variance results, selects the top N structures, retrieves
    them from the database, creates new batch directories/XYZ files for the
    optimization step, and generates the SLURM array job script to run the
    gradient-based optimization engine.

    Args:
        input_directory: Directory containing combined variance results JSON files.
                         Typically 'variance_results'.
        model_dir: Path to the original model directory (used for README).
        n_structures: Number of highest-variance structures to select.
        n_batches: Number of batches (SLURM jobs) to split optimization into.
        learning_rate: Learning rate parameter for the optimization engine.
        n_iterations: Number of iterations parameter for the optimization engine.
        min_distance: Minimum distance parameter for the optimization engine.
        include_probability: Probability weighting flag for the optimization engine.
        temperature: Temperature (eV) parameter for probability weighting.
        device: Device to request for SLURM jobs ('cpu' or 'cuda').
        debug: Enable detailed debug output.

    Raises:
        ValueError: If input directory or models are not found, or no structures selected.
    """
    print("[INFO] Preparing Gradient-Based Adversarial Attack optimization jobs...")

    # --- Setup directories ---
    variance_results_path = Path(input_directory).resolve()
    if not variance_results_path.is_dir():
        raise ValueError(f"Variance results directory not found: {variance_results_path}")

    # Parent directory is the main workflow dir
    workflow_dir = variance_results_path.parent
    aa_output_dir = workflow_dir / "gradient_aa_optimization"
    aa_output_dir.mkdir(parents=True, exist_ok=True)
    slurm_log_dir = aa_output_dir / "slurm_logs" # Logs inside job dir
    slurm_log_dir.mkdir(exist_ok=True)

    # Models should already be copied in the 'models' subdir of workflow_dir
    models_path = workflow_dir / "models"
    if not models_path.is_dir() or not any(models_path.glob("*.model")):
         # Attempt to copy models if missing (e.g., if run out of order)
         print(f"[WARNING] Models directory '{models_path}' not found or empty. Attempting to copy from '{model_dir}'...")
         prepare_variance_calculation_jobs([], workflow_dir, Path(model_dir), 1) # Hacky way to trigger model copy
         if not models_path.is_dir() or not any(models_path.glob("*.model")):
              raise ValueError(f"Models directory '{models_path}' is still missing or empty after attempting copy.")


    # --- Combine and sort variance results ---
    print("[INFO] Combining and sorting variance results...")
    # Expects list of (name, variance, id)
    sorted_results = combine_variance_results(variance_results_path, debug=debug)
    if not sorted_results:
        raise ValueError(f"No valid variance results found in {variance_results_path}.")
    print(f"[INFO] Found {len(sorted_results)} structures with variance results.")

    # --- Select top N structures ---
    if n_structures <= 0:
        raise ValueError("Number of structures (n_structures) must be positive.")
    selected_structures_info = sorted_results[:n_structures]
    print(f"[INFO] Selected top {len(selected_structures_info)} structures for optimization.")
    if debug:
        print("[DEBUG] Selected structure IDs and variances:")
        for name, var, sid in selected_structures_info:
            print(f"  ID: {sid:<6} Variance: {var:.6f}")

    # Extract structure IDs to retrieve from database
    structure_ids_to_retrieve = [sid for name, var, sid in selected_structures_info]
    variance_map = {sid: var for name, var, sid in selected_structures_info} # Map ID to initial variance

    # --- Retrieve selected structures from Database ---
    print("[INFO] Retrieving selected structures from database...")
    db_manager = DatabaseManager()
    # Get structures, preferably with energy info if needed later by optimizer
    # Although for gradient AA, energy is only used if include_probability=True
    retrieved_atoms_list = db_manager.get_atoms_with_calculations(
         structure_ids_to_retrieve,
         calculator='vasp'
    )

    # Filter and add initial variance info
    final_atoms_for_batching = []
    retrieved_ids = set()
    for atoms in retrieved_atoms_list:
        struct_id = atoms.info.get('structure_id')
        if struct_id in variance_map:
            atoms.info['initial_variance'] = variance_map[struct_id]
            # Ensure consistent naming based on ID
            atoms.info['structure_name'] = f"struct_{struct_id}"
            final_atoms_for_batching.append(atoms)
            retrieved_ids.add(struct_id)
        elif debug:
            print(f"[DEBUG] Structure ID {struct_id} retrieved but not in top variance list?")

    # Check if all requested structures were retrieved
    missing_ids = set(structure_ids_to_retrieve) - retrieved_ids
    if missing_ids:
        print(f"[WARNING] Could not retrieve the following selected structure IDs from database: {missing_ids}")

    if not final_atoms_for_batching:
        raise ValueError("Failed to retrieve any of the selected top-variance structures from the database.")

    n_structures_for_aa = len(final_atoms_for_batching)
    print(f"[INFO] Retrieved {n_structures_for_aa} structures for AA optimization.")


    # --- Create batches for AA optimization ---
    n_aa_batches = n_batches # Use the provided n_batches for AA step
    if n_aa_batches > n_structures_for_aa:
        print(f"[WARNING] AA batches ({n_aa_batches}) > AA structures ({n_structures_for_aa}). Setting n_aa_batches = {n_structures_for_aa}.")
        n_aa_batches = n_structures_for_aa
    if n_aa_batches <= 0:
        print(f"[WARNING] Invalid number of AA batches ({n_aa_batches}). Setting n_aa_batches = 1.")
        n_aa_batches = 1

    structures_per_aa_batch = n_structures_for_aa // n_aa_batches
    remaining_aa_structures = n_structures_for_aa % n_aa_batches

    start_index = 0
    actual_n_aa_batches = 0
    for batch_id in range(n_aa_batches):
        batch_dir = aa_output_dir / f"batch_{batch_id}"
        batch_dir.mkdir(exist_ok=True)

        end_index = start_index + structures_per_aa_batch
        if batch_id < remaining_aa_structures:
            end_index += 1

        batch_atoms_list = final_atoms_for_batching[start_index:end_index]

        if not batch_atoms_list:
             if debug: print(f"[DEBUG] Skipping AA batch {batch_id} as it has no structures.")
             continue

        actual_n_aa_batches += 1
        batch_xyz = batch_dir / f"batch_{batch_id}.xyz"

        # Write batch XYZ file (structures already have names and initial variance)
        ase.io.write(batch_xyz, batch_atoms_list, format="extxyz")
        print(f"[INFO] AA Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {batch_xyz}")

        # Save batch metadata (optional but useful for tracking)
        batch_meta = {
            'batch_id': batch_id,
            'structures': [
                {
                    'name': atoms.info['structure_name'],
                    'id': atoms.info['structure_id'],
                    'initial_variance': atoms.info['initial_variance'],
                    'energy': atoms.info.get('energy') # Include energy if available
                } for atoms in batch_atoms_list
            ]
        }
        with open(batch_dir / 'batch_metadata.json', 'w') as f:
            json.dump(batch_meta, f, indent=2)

        start_index = end_index

    if actual_n_aa_batches == 0:
        raise ValueError("No non-empty AA optimization batches were created.")


    # --- Generate SLURM array job script for AA optimization ---
    slurm_script_path = aa_output_dir / "gradient_aa_optimization_array.slurm"

    # Define paths relative to the workflow root directory (where sbatch is run)
    model_dir_rel = "models"
    batch_base_rel = "gradient_aa_optimization" # Relative path to batch dirs

    # Set GPU requirement based on device
    gpus_per_task = 1 if device == "cuda" else 0

    # Get the SLURM script content from the template
    slurm_content = get_gradient_aa_script(
        batch_base_rel=batch_base_rel, # Base dir for batch-specific files/output
        log_dir_rel=f"{batch_base_rel}/slurm_logs",
        model_dir_rel=model_dir_rel,
        # Structure file path is now relative to batch_base_rel
        structure_file_rel="batch_${SLURM_ARRAY_TASK_ID}/batch_${SLURM_ARRAY_TASK_ID}.xyz",
        # Output dir for the engine script is the batch directory itself
        engine_output_dir_rel="batch_${SLURM_ARRAY_TASK_ID}",
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        min_distance=min_distance,
        include_probability=include_probability,
        temperature=temperature, # eV
        device=device,
        array_range=f"0-{actual_n_aa_batches - 1}",
        # Add SLURM resource parameters (adjust defaults as needed)
        time="24:00:00",
        mem="32G",
        cpus_per_task=8,
        gpus_per_task=gpus_per_task,
        save_trajectory=True, # Assuming default behavior is to save
        # database_id is not needed here, passed per structure inside engine if applicable
    )

    with open(slurm_script_path, 'w') as f:
        f.write(slurm_content)
    print(f"[INFO] Created Gradient AA SLURM array script: {slurm_script_path}")
    print(f"[INFO] Submit the job from the workflow directory ('{workflow_dir.name}') using: sbatch {aa_output_dir.name}/{slurm_script_path.name}")


# --- Renamed prepare_aa_optimization -> prepare_monte_carlo_aa_optimization ---
def prepare_monte_carlo_aa_optimization(
    input_directory: str, # Should be variance_results dir
    model_dir: str, # Original model source dir (used for path in README)
    n_structures: int,
    n_batches: int = 1,
    temperature: float = 1200.0, # K for Metropolis
    max_steps: int = 50,
    patience: int = 25,
    min_distance: float = 2.0,
    max_displacement: float = 0.1, # Added
    mode: str = "all",
    device: str = "cuda",
    debug: bool = False,
) -> None:
    """
    Prepares batch directories and SLURM script for Monte Carlo AA optimization.

    Reads combined variance results, selects top N structures, retrieves them,
    creates batch directories/XYZ files, and generates the SLURM array job
    script to run the Monte Carlo optimization engine.

    Args:
        input_directory: Directory containing combined variance results JSON files.
        model_dir: Path to the original model directory.
        n_structures: Number of highest-variance structures to select.
        n_batches: Number of batches (SLURM jobs) to split optimization into.
        temperature: Temperature (K) for Metropolis acceptance criterion.
        max_steps: Maximum number of Monte Carlo steps.
        patience: Patience for stopping optimization.
        min_distance: Minimum allowed interatomic distance (Å).
        max_displacement: Maximum distance (Å) an atom can be moved per MC step.
        mode: Atom displacement mode ('all' or 'single').
        device: Device to request for SLURM jobs ('cpu' or 'cuda').
        debug: Enable detailed debug output.

    Raises:
        ValueError: If inputs are invalid or structures cannot be retrieved.
    """
    print("[INFO] Preparing Monte Carlo Adversarial Attack optimization jobs...")

    # --- Setup directories ---
    variance_results_path = Path(input_directory).resolve()
    if not variance_results_path.is_dir():
        raise ValueError(f"Variance results directory not found: {variance_results_path}")

    workflow_dir = variance_results_path.parent
    aa_output_dir = workflow_dir / "monte_carlo_aa_optimization" # Changed dir name
    aa_output_dir.mkdir(parents=True, exist_ok=True)
    slurm_log_dir = aa_output_dir / "slurm_logs"
    slurm_log_dir.mkdir(exist_ok=True)

    # Ensure models exist
    models_path = workflow_dir / "models"
    if not models_path.is_dir() or not any(models_path.glob("*.model")):
         print(f"[WARNING] Models directory '{models_path}' not found or empty. Attempting to copy from '{model_dir}'...")
         prepare_variance_calculation_jobs([], workflow_dir, Path(model_dir), 1)
         if not models_path.is_dir() or not any(models_path.glob("*.model")):
              raise ValueError(f"Models directory '{models_path}' is still missing or empty after attempting copy.")

    # --- Combine and sort variance results ---
    print("[INFO] Combining and sorting variance results...")
    sorted_results = combine_variance_results(variance_results_path, debug=debug)
    if not sorted_results:
        raise ValueError(f"No valid variance results found in {variance_results_path}.")
    print(f"[INFO] Found {len(sorted_results)} structures with variance results.")

    # --- Select top N structures ---
    if n_structures <= 0:
        raise ValueError("Number of structures (n_structures) must be positive.")
    selected_structures_info = sorted_results[:n_structures]
    print(f"[INFO] Selected top {len(selected_structures_info)} structures for optimization.")

    structure_ids_to_retrieve = [sid for name, var, sid in selected_structures_info]
    variance_map = {sid: var for name, var, sid in selected_structures_info}

    # --- Retrieve selected structures from Database ---
    print("[INFO] Retrieving selected structures from database...")
    db_manager = DatabaseManager()
    retrieved_atoms_list = db_manager.get_atoms_with_calculations(
         structure_ids_to_retrieve, model_type='vasp-static' # Get energy if possible
    )

    final_atoms_for_batching = []
    retrieved_ids = set()
    for atoms in retrieved_atoms_list:
        struct_id = atoms.info.get('structure_id')
        if struct_id in variance_map:
            atoms.info['initial_variance'] = variance_map[struct_id]
            atoms.info['structure_name'] = f"struct_{struct_id}"
            final_atoms_for_batching.append(atoms)
            retrieved_ids.add(struct_id)

    missing_ids = set(structure_ids_to_retrieve) - retrieved_ids
    if missing_ids:
        print(f"[WARNING] Could not retrieve the following selected structure IDs from database: {missing_ids}")

    if not final_atoms_for_batching:
        raise ValueError("Failed to retrieve any of the selected top-variance structures from the database.")

    n_structures_for_aa = len(final_atoms_for_batching)
    print(f"[INFO] Retrieved {n_structures_for_aa} structures for AA optimization.")

    # --- Create batches for AA optimization ---
    n_aa_batches = n_batches
    if n_aa_batches > n_structures_for_aa:
        print(f"[WARNING] AA batches ({n_aa_batches}) > AA structures ({n_structures_for_aa}). Setting n_aa_batches = {n_structures_for_aa}.")
        n_aa_batches = n_structures_for_aa
    if n_aa_batches <= 0: n_aa_batches = 1

    structures_per_aa_batch = n_structures_for_aa // n_aa_batches
    remaining_aa_structures = n_structures_for_aa % n_aa_batches

    start_index = 0
    actual_n_aa_batches = 0
    for batch_id in range(n_aa_batches):
        batch_dir = aa_output_dir / f"batch_{batch_id}"
        batch_dir.mkdir(exist_ok=True)

        end_index = start_index + structures_per_aa_batch
        if batch_id < remaining_aa_structures: end_index += 1

        batch_atoms_list = final_atoms_for_batching[start_index:end_index]

        if not batch_atoms_list: continue

        actual_n_aa_batches += 1
        batch_xyz = batch_dir / f"batch_{batch_id}.xyz"
        ase.io.write(batch_xyz, batch_atoms_list, format="extxyz")
        print(f"[INFO] MC AA Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {batch_xyz}")

        batch_meta = {
            'batch_id': batch_id,
            'structures': [{'name': a.info['structure_name'], 'id': a.info['structure_id'], 'initial_variance': a.info['initial_variance']} for a in batch_atoms_list]
        }
        with open(batch_dir / 'batch_metadata.json', 'w') as f:
            json.dump(batch_meta, f, indent=2)

        start_index = end_index

    if actual_n_aa_batches == 0:
        raise ValueError("No non-empty Monte Carlo AA optimization batches were created.")

    # --- Generate SLURM array job script ---
    slurm_script_path = aa_output_dir / "monte_carlo_aa_optimization_array.slurm"

    model_dir_rel = "models"
    batch_base_rel = "monte_carlo_aa_optimization" # Relative path

    gpus_per_task = 1 if device == "cuda" else 0

    slurm_content = get_monte_carlo_aa_script(
        batch_base_rel=batch_base_rel,
        log_dir_rel=f"{batch_base_rel}/slurm_logs",
        model_dir_rel=model_dir_rel,
        structure_file_rel="batch_${SLURM_ARRAY_TASK_ID}/batch_${SLURM_ARRAY_TASK_ID}.xyz",
        engine_output_dir_rel="batch_${SLURM_ARRAY_TASK_ID}",
        max_steps=max_steps,
        patience=patience,
        temperature=temperature, # K
        min_distance=min_distance,
        max_displacement=max_displacement, # Pass arg
        mode=mode,
        device=device,
        array_range=f"0-{actual_n_aa_batches - 1}",
        time="24:00:00",
        mem="32G",
        cpus_per_task=8,
        gpus_per_task=gpus_per_task,
        save_trajectory=True, # Assuming default behavior
    )

    with open(slurm_script_path, 'w') as f:
        f.write(slurm_content)
    print(f"[INFO] Created Monte Carlo AA SLURM array script: {slurm_script_path}")
    print(f"[INFO] Submit the job from the workflow directory ('{workflow_dir.name}') using: sbatch {aa_output_dir.name}/{slurm_script_path.name}")


# --- select_structures_from_trajectory function ---
# (Add/improve docstring)
def select_structures_from_trajectory(
    trajectory_file: Path,
    n_structures: int,
    optimization_summary: Optional[dict] = None, # Pass summary dict directly
    struct_name_filter: Optional[str] = None, # Filter for specific structure in summary
) -> List[Tuple[Atoms, int, float]]:
    """
    Selects structures from an AA optimization trajectory file (XYZ).

    Reads an XYZ trajectory, selects N structures evenly spaced working
    backwards from the final frame, and attempts to associate variance values
    from the provided optimization summary data.

    Args:
        trajectory_file: Path to the XYZ trajectory file (e.g., *_adversarial.xyz).
        n_structures: Number of structures to select from the trajectory.
        optimization_summary: Loaded JSON data from 'optimization_summary.json'.
        struct_name_filter: The 'structure_name' to look for within the summary's
                            'results' list to find the correct variance history.

    Returns:
        List of tuples: (atoms, step_number, variance). Variance may be None
        if not found in the summary. Returns empty list on failure.

    Raises:
        ValueError: If trajectory is empty or fewer structures than requested exist.
    """
    if not trajectory_file.exists():
        print(f"[WARNING] Trajectory file not found: {trajectory_file}")
        return []

    try:
        trajectory = ase.io.read(trajectory_file, ':')
        if not trajectory:
            print(f"[WARNING] No structures found in trajectory file: {trajectory_file}")
            return []
    except Exception as e:
        print(f"[WARNING] Failed to read trajectory file {trajectory_file}: {e}")
        return []

    n_total_steps = len(trajectory) # Total number of frames

    # --- Extract step variances and parameters from summary ---
    step_variances = {} # Map: step_index -> variance
    aa_temperature = None # Temperature used during AA optimization
    initial_variance_from_summary = None

    if optimization_summary and struct_name_filter:
        # Find the specific structure's result in the summary
        struct_result = None
        for res in optimization_summary.get('results', []):
            if res.get('structure_name') == struct_name_filter:
                struct_result = res
                break

        if struct_result:
            # Try getting step variances (MC saves this, gradient saves loss_history)
            if 'step_variances' in struct_result: # MC format
                step_variances = {i: v for i, v in enumerate(struct_result['step_variances'])}
            elif 'loss_history' in struct_result: # Gradient format (loss=variance)
                 step_variances = {i: v for i, v in enumerate(struct_result['loss_history'])}

            initial_variance_from_summary = struct_result.get('initial_variance')

            # Get AA temperature from parameters block
            method = optimization_summary.get('parameters', {}).get('method')
            aa_temperature = optimization_summary.get('parameters', {}).get('temperature')
            # Note: Temp has different units (K vs eV) depending on method, store raw value.


    # --- Select structure indices ---
    # We want N structures, including the last one.
    if n_structures <= 0:
        print("[WARNING] Number of structures per trajectory must be positive.")
        return []
    if n_structures > n_total_steps:
         print(f"[WARNING] Requested {n_structures} structures, but trajectory only has {n_total_steps}. Selecting all.")
         n_structures = n_total_steps

    # Calculate indices: include last frame (n_total_steps - 1), then space out
    # the remaining N-1 structures over the first n_total_steps - 1 frames.
    # Example: 10 steps (indices 0-9), n_structures = 3
    # Indices: 9, (9 - 1*(8 // 2)) = 5, (9 - 2*(8 // 2)) = 1. Indices: [9, 5, 1]
    # Example: 10 steps, n_structures = 1. Indices: [9]
    # Example: 10 steps, n_structures = 10. Indices: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    if n_structures == 1:
         indices = [n_total_steps - 1]
    else:
        # Ensure division by zero doesn't occur if n_structures=1 was handled above
        step_interval = (n_total_steps - 1) // (n_structures - 1) if n_structures > 1 else 0
        # Handle potential floating point issues with linspace, manual calculation is safer
        indices = []
        for i in range(n_structures):
             # Calculate index relative to the end
             idx = (n_total_steps - 1) - i * step_interval
             # Ensure index is non-negative (can happen if interval is large)
             indices.append(max(0, idx))
        # Ensure indices are unique and sorted (descending for "backwards")
        indices = sorted(list(set(indices)), reverse=True)


    # --- Get selected Atoms objects and add metadata ---
    selected_tuples = []
    for step_index in indices:
        try:
            atoms = trajectory[step_index].copy() # Get a copy
            step_variance = step_variances.get(step_index) # Get variance for this step if available

            # --- Populate atoms.info for VASP job creation ---
            # Keep original info if present
            original_info = trajectory[step_index].info

            # Add AA-specific metadata
            atoms.info['aa_step'] = step_index
            atoms.info['aa_step_variance'] = step_variance
            atoms.info['aa_temperature'] = aa_temperature # Temp used during AA run
            # Keep initial variance from summary if available
            atoms.info['initial_variance'] = initial_variance_from_summary

            # Crucially, try to preserve parent ID if present in original info
            if 'parent_structure_id' in original_info:
                atoms.info['parent_structure_id'] = original_info['parent_structure_id']
            elif 'structure_id' in original_info: # Fallback if only original ID is there
                 atoms.info['parent_structure_id'] = original_info['structure_id']

            # Preserve original config type if available
            if 'config_type' in original_info:
                 atoms.info['parent_config_type'] = original_info['config_type']


            selected_tuples.append((atoms, step_index, step_variance))
        except IndexError:
            print(f"[WARNING] Could not retrieve step {step_index} from trajectory {trajectory_file}")
        except Exception as e:
            print(f"[WARNING] Error processing step {step_index} from {trajectory_file}: {e}")

    if debug:
        print(f"[DEBUG] Selected {len(selected_tuples)} structures from {trajectory_file} at steps: {indices}")

    return selected_tuples


# --- prepare_vasp_jobs function ---
# (Add/improve docstring)
def prepare_vasp_jobs(
    input_directory: str, # AA results dir (e.g., gradient_aa_optimization)
    output_directory: str, # Dir to create VASP jobs in
    vasp_profile: str = "static",
    hpc_profile: str = "default",
    structures_per_traj: int = 1,
    debug: bool = False,
) -> None:
    """
    Creates VASP jobs for structures resulting from AA optimization.

    Scans the AA results directory for trajectories, selects structures based on
    'structures_per_traj', adds them to the database with appropriate metadata,
    and prepares VASP calculation directories using specified profiles.

    Args:
        input_directory: Directory containing AA optimization results (e.g.,
                         'gradient_aa_optimization' or 'monte_carlo_aa_optimization').
                         Expected to contain batch_*/aa_results subdirectories.
        output_directory: Directory where the VASP job subdirectories will be created.
        vasp_profile: Name of the VASP settings profile (defined in forge config).
        hpc_profile: Name of the HPC profile for job scripts (defined in forge config).
        structures_per_traj: Number of structures to select from each trajectory
                             (e.g., 1 for final, >1 for intermediates).
        debug: Enable detailed debug output.
    """
    print("[INFO] Starting VASP job preparation from AA results...")

    input_path = Path(input_directory).resolve()
    output_path = Path(output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize database manager
    db_manager = DatabaseManager()

    # Find all AA result directories within batches
    aa_results_dirs = list(input_path.glob("batch_*/aa_results"))
    if not aa_results_dirs:
        print(f"[WARNING] No 'aa_results' subdirectories found within {input_path}/batch_*")
        # Maybe check for results directly in input_directory? (If not batched)
        if any(input_path.glob("*_adversarial.xyz")):
             print(f"[INFO] Found trajectory files directly in {input_path}. Processing...")
             aa_results_dirs = [input_path] # Process input dir itself
        else:
             return # Exit if no results found anywhere

    print(f"[INFO] Found {len(aa_results_dirs)} AA result locations to process.")

    # Keep track of created jobs
    created_job_count = 0

    # Process each results directory
    for results_dir in aa_results_dirs:
        # Find the optimization summary for this batch/result set
        summary_file = results_dir.parent / "optimization_summary.json" # Summary is usually one level up
        if not summary_file.exists():
             # Check inside results dir as fallback
             summary_file = results_dir / "optimization_summary.json"

        optimization_summary = None
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    optimization_summary = json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not load optimization summary {summary_file}: {e}")
        else:
            print(f"[WARNING] Optimization summary not found for results in {results_dir}")


        # Find trajectories in the results directory
        trajectory_files = list(results_dir.glob("*_adversarial.xyz"))
        if not trajectory_files and optimization_summary:
             # Maybe only optimized files were saved, not trajectory
             trajectory_files = list(results_dir.glob("*_optimized.xyz"))


        if not trajectory_files:
             print(f"[INFO] No trajectory or optimized structure files found in {results_dir}. Skipping.")
             continue

        print(f"[INFO] Processing {len(trajectory_files)} trajectories/structures in {results_dir}...")

        # Process each trajectory/optimized structure file
        for traj_file in trajectory_files:
            # Extract base name to identify structure in summary
            # Assumes traj filename is like 'struct_ID_adversarial.xyz' or 'struct_ID_optimized.xyz'
            if traj_file.name.endswith("_adversarial.xyz"):
                 struct_name_filter = traj_file.stem.replace('_adversarial', '')
            elif traj_file.name.endswith("_optimized.xyz"):
                 struct_name_filter = traj_file.stem.replace('_optimized', '')
            else: # Fallback if naming convention differs
                 struct_name_filter = traj_file.stem
            if debug: print(f"[DEBUG] Processing file: {traj_file.name}, Base name: {struct_name_filter}")

            try:
                # Select structures using the helper function
                selected_structures = select_structures_from_trajectory(
                    traj_file,
                    structures_per_traj,
                    optimization_summary, # Pass loaded summary
                    struct_name_filter   # Pass name to filter summary data
                )

                if not selected_structures:
                    print(f"[INFO] No structures selected from {traj_file.name}. Skipping.")
                    continue

                # Process each selected structure (Atoms, step_index, variance)
                for atoms, step, variance in selected_structures:
                    # --- Prepare metadata for database and VASP job ---
                    parent_id = atoms.info.get('parent_structure_id')
                    parent_config_type = atoms.info.get('parent_config_type', 'unknown')

                    # Define new config type based on parent
                    if parent_config_type and parent_config_type != 'unknown':
                        config_type = f"{parent_config_type}_aa_s{step}"
                    else:
                        config_type = f"aa_s{step}" # Fallback

                    # Construct metadata dictionary
                    structure_meta = {
                        "parent_structure_id": parent_id,
                        "parent_config_type": parent_config_type,
                        "config_type": config_type,
                        "aa_step": step,
                        "aa_step_variance": variance,
                        "aa_temperature": atoms.info.get('aa_temperature'), # Temp used during AA
                        "initial_variance": atoms.info.get('initial_variance'), # From original structure
                        # Add batch ID if possible (how to get this reliably?)
                        # 'batch_id': batch_id # Needs passing down or extracting from path
                    }
                    # Clean up None values for cleaner JSON/DB entry
                    structure_meta = {k: v for k, v in structure_meta.items() if v is not None}


                    # --- Add structure to database ---
                    try:
                        # Add the selected structure (potentially intermediate frame)
                        new_id = db_manager.add_structure(
                            atoms,
                            metadata=structure_meta
                        )
                        print(f"[INFO] Added AA structure step {step} from {struct_name_filter} to DB with ID: {new_id}")
                    except Exception as e:
                         print(f"[ERROR] Failed to add structure step {step} from {struct_name_filter} to database: {e}")
                         if debug: raise
                         continue # Skip VASP job creation if DB add failed

                    # --- Create VASP job directory name ---
                    # Include new ID, base name, step, maybe variance
                    job_name_parts = [
                        f"id{new_id}",
                        struct_name_filter,
                        f"s{step}"
                    ]
                    if variance is not None: job_name_parts.append(f"var{variance:.3f}")
                    job_name = "_".join(job_name_parts)
                    job_dir = output_path / f"job_{job_name}" # Use descriptive name

                    # --- Create VASP job using db_to_vasp helper ---
                    try:
                        prepare_vasp_job_from_ase(
                            atoms=atoms, # Use the atoms object with populated info
                            vasp_profile_name=vasp_profile,
                            hpc_profile_name=hpc_profile,
                            output_dir=str(job_dir),
                            auto_kpoints=True, # Default to auto k-points
                            DEBUG=debug,
                            job_name=job_name # Pass job name for SLURM script
                        )

                        # Save the detailed metadata alongside VASP inputs
                        with open(job_dir / "job_metadata.json", 'w') as f:
                            # Ensure metadata is JSON serializable (float precision might need care)
                            json.dump(structure_meta, f, indent=2)

                        print(f"[INFO] Created VASP job for structure {new_id} in: {job_dir.name}")
                        created_job_count += 1

                    except Exception as e:
                        print(f"[ERROR] Failed to create VASP job for structure {new_id} (step {step} from {struct_name_filter}): {e}")
                        if debug: raise
                        # Optionally, remove the structure from DB if VASP prep failed? Risky.
                        continue

            except Exception as e:
                print(f"[ERROR] Failed to process trajectory file {traj_file}: {e}")
                if debug: raise
                continue # Move to next trajectory file

    print(f"\n[INFO] VASP job preparation completed. Created {created_job_count} jobs in {output_path}")


# --- Add generate_workflow_readme function back ---
def generate_workflow_readme(
    output_dir: Path,
    model_dir_rel: str, # Relative path to model dir from workflow root
    elements: Optional[List[str]],
    structure_ids: Optional[List[int]], # Keep track if IDs were used
    n_batches_variance: int, # Number of batches for variance calc
    n_structures_selected: int, # Number of structures selected for AA
    hpc_profile_vasp: str, # Example HPC profile for VASP step
    example_aa_n_batches: int = 1, # Example N batches for AA step in README
    example_structures_per_traj: int = 1, # Example for VASP step
    # Include args that were passed to create-aa-jobs for the record
    structure_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    debug: bool = False,

) -> None:
    """
    Generates a README.md file within the workflow directory explaining the steps.

    Args:
        output_dir: Path object for the main workflow directory.
        model_dir_rel: Relative path to the model directory from the workflow root.
        elements: Elements used for initial selection (if any).
        structure_ids: Specific structure IDs used for initial selection (if any).
        n_batches_variance: Number of batches created for the variance calculation step.
        n_structures_selected: Number of structures selected for AA optimization (used in example).
        hpc_profile_vasp: Example HPC profile name for the VASP job creation command.
        example_aa_n_batches: Example number of batches for AA optimization step in README.
        example_structures_per_traj: Example number of structures per trajectory for VASP step.
        structure_type: Original filter argument.
        composition_constraints: Original filter argument.
        debug: Original debug flag.
    """
    workflow_dir_name = output_dir.name
    variance_calc_dir = "variance_calculations"
    variance_results_dir = "variance_results"
    gradient_aa_dir = "gradient_aa_optimization"
    mc_aa_dir = "monte_carlo_aa_optimization"
    vasp_jobs_dir = "vasp_jobs"
    models_dir = "models"

    # --- Build the initial command used ---
    create_cmd_parts = [
        "forge create-aa-jobs",
        f"    --output_dir {workflow_dir_name}",
        f"    --model_dir {model_dir_rel}", # Use relative path provided
    ]
    if structure_ids:
         create_cmd_parts.append(f"    --structure_ids {' '.join(map(str, structure_ids))}")
    elif elements:
         create_cmd_parts.append(f"    --elements {' '.join(elements)}")
    create_cmd_parts.append(f"    --n_batches {n_batches_variance}") # Num batches for variance
    if structure_type:
        create_cmd_parts.append(f"    --structure_type {structure_type}")
    if composition_constraints:
        # Handle potential quotes in the JSON string for display
        constraints_display = composition_constraints.replace("'", "\\'")
        create_cmd_parts.append(f"    --composition_constraints '{constraints_display}'")
    if debug:
         create_cmd_parts.append("    --debug")

    initial_command = " \\\n".join(create_cmd_parts)

    # --- README Content ---
    readme_content = f"""# Adversarial Attack Workflow: {workflow_dir_name}

This directory contains an Adversarial Attack (AA) workflow generated by `forge`. The goal is to identify structures where the model ensemble disagrees the most (high variance) and potentially refine the models using data from these structures.

## Workflow Steps

Follow these steps to execute the workflow:

**1. (Implement &) Run Initial Variance Calculation**

   *   **(TODO)** The SLURM script `{variance_calc_dir}/variance_calculation_array.slurm` is currently a **placeholder**. You need to implement the actual variance calculation logic it calls, which should process the XYZ files in `{variance_calc_dir}/batch_*/` and write variance results (e.g., `{{ "struct_ID": variance, ... }}`) to JSON files in `{variance_results_dir}/`.
   *   Once implemented, submit the job from this directory (`{workflow_dir_name}/`):
       ```bash
       sbatch {variance_calc_dir}/variance_calculation_array.slurm
       ```
   *   Wait for the jobs to complete.

**2. Prepare Adversarial Attack Optimization Jobs**

   *   After variance calculations are complete and results are in `{variance_results_dir}/`, choose **one** of the following methods:

   *   **Option A: Gradient-Based AA**
       ```bash
       # Run from this directory ({workflow_dir_name}/)
       forge run-gradient-aa-jobs \\
           --input_directory {variance_results_dir} \\
           --model_dir {model_dir_rel} \\
           --n_structures {n_structures_selected} `# Number of top variance structures` \\
           --n_batches {example_aa_n_batches} `# Number of optimization jobs` \\
           --learning_rate 0.01 \\
           --n_iterations 60 \\
           --device cuda `# or cpu`
       ```
       This creates job setup in `{gradient_aa_dir}/` and the SLURM script `{gradient_aa_dir}/gradient_aa_optimization_array.slurm`.

   *   **Option B: Monte Carlo AA**
       ```bash
       # Run from this directory ({workflow_dir_name}/)
       forge run-aa-jobs \\
           --input_directory {variance_results_dir} \\
           --model_dir {model_dir_rel} \\
           --n_structures {n_structures_selected} `# Number of top variance structures` \\
           --n_batches {example_aa_n_batches} `# Number of optimization jobs` \\
           --temperature 1200 `# Metropolis temp (K)` \\
           --max_steps 50 \\
           --max_displacement 0.1 \\
           --device cuda `# or cpu`
       ```
       This creates job setup in `{mc_aa_dir}/` and the SLURM script `{mc_aa_dir}/monte_carlo_aa_optimization_array.slurm`.

**3. Run Adversarial Attack Optimization**

   *   Submit the SLURM script corresponding to your chosen method (A or B):
       ```bash
       # If using Gradient-Based:
       sbatch {gradient_aa_dir}/gradient_aa_optimization_array.slurm

       # If using Monte Carlo:
       sbatch {mc_aa_dir}/monte_carlo_aa_optimization_array.slurm
       ```
   *   Wait for jobs to complete. Results (optimized structures, trajectories, summaries) will be inside the respective batch directories (e.g., `{gradient_aa_dir}/batch_*/aa_results/`).

**4. Create VASP Jobs**

   *   Generate VASP calculation directories for selected structures from the AA optimization results:
       ```bash
       # Run from this directory ({workflow_dir_name}/)
       # Adjust --input_directory based on method used in step 2/3
       forge create-aa-vasp-jobs \\
           --input_directory {gradient_aa_dir} `# or {mc_aa_dir}` \\
           --output_directory {vasp_jobs_dir} \\
           --vasp_profile static `# Your desired VASP profile` \\
           --hpc_profile {hpc_profile_vasp} `# Your HPC profile` \\
           --structures_per_traj {example_structures_per_traj} `# Select final (1) or more structures`
       ```
   *   This creates VASP job directories in `{vasp_jobs_dir}/`.

**5. Submit VASP Jobs**

   *   Navigate into the individual job directories within `{vasp_jobs_dir}/` and submit them according to your cluster's procedures (usually involves running `sbatch job_script.slurm` inside each job folder).

## Directory Structure

*   `{models_dir}/`: Copied MACE model ensemble files.
*   `{variance_calc_dir}/`: Batch files and SLURM script for initial variance calculation.
*   `{variance_results_dir}/`: Expected location for variance calculation results (JSON files).
*   `{gradient_aa_dir}/` (if used): Batch files, SLURM script, and results for Gradient-Based AA.
*   `{mc_aa_dir}/` (if used): Batch files, SLURM script, and results for Monte Carlo AA.
*   `{vasp_jobs_dir}/`: Generated VASP job directories.
*   `README.md`: This file.

## Workflow Creation Parameters

This workflow was generated using the following command:
```bash
{initial_command}
```

"""

    # Write the README file
    readme_path = output_dir / "README.md"
    try:
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"[INFO] Generated workflow README.md in {output_dir}")
    except IOError as e:
        print(f"[ERROR] Failed to write README.md: {e}")


# --- prepare_aa_workflow function ---
# (Add/improve docstring)
def prepare_aa_workflow(
    output_dir: str,
    model_dir: str,
    elements: Optional[List[str]] = None,
    n_batches: int = 1,
    structure_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    structure_ids: Optional[List[int]] = None,
    debug: bool = False,
) -> None:
    """
    Prepares the initial adversarial attack workflow directory structure and
    variance calculation jobs.

    This is the entry point for the 'create-aa-jobs' command. It selects
    structures from the database, sets up the main output directory, copies
    models, prepares the first stage (variance calculation batches and
    a placeholder SLURM script), and generates a README file.

    Args:
        output_dir: Path to the directory where the workflow will be created.
        model_dir: Path to the directory containing the MACE model ensemble.
        elements: List of elements to select structures by (if structure_ids is None).
        n_batches: Number of batches for the variance calculation step.
        structure_type: Optional structure type filter for database query.
        composition_constraints: Optional JSON string for composition constraints.
        structure_ids: Optional list of specific structure IDs to use.
        debug: Enable detailed debug output.
    """
    print("[INFO] Preparing Adversarial Attack workflow directory...")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Workflow directory: {output_path}")

    # --- Step 1: Select structures from database ---
    print("[INFO] Selecting initial structures from database...")
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
        # Potentially exit or raise here depending on desired behavior
        return # Exit gracefully if selection fails
    if not selected_atoms_list:
        print("[ERROR] No structures were selected. Exiting workflow preparation.")
        return
    n_structures_initial = len(selected_atoms_list) # Get number selected
    print(f"[INFO] Selected {n_structures_initial} structures for the workflow.")


    # --- Step 2: Prepare variance calculation jobs ---
    print("[INFO] Preparing variance calculation jobs...")
    try:
        variance_results_dir = prepare_variance_calculation_jobs(
            selected_atoms_list=selected_atoms_list,
            output_dir=output_path,
            model_dir=Path(model_dir),
            n_batches=n_batches,
        )
    except ValueError as e:
         print(f"[ERROR] Failed to prepare variance calculation jobs: {e}")
         return
    except Exception as e:
         print(f"[ERROR] An unexpected error occurred during variance job prep: {e}")
         if debug: raise
         return

    # --- Step 3: Generate Workflow README.md ---
    # Use relative path for model_dir in README for portability
    try:
         model_dir_rel = os.path.relpath(model_dir, start=output_path.parent)
    except ValueError: # Handle case where paths are on different drives (Windows)
         model_dir_rel = str(Path(model_dir).resolve()) # Use absolute path as fallback


    generate_workflow_readme(
         output_dir=output_path,
         model_dir_rel=model_dir_rel,
         elements=elements,
         structure_ids=structure_ids,
         n_batches_variance=n_batches, # Actual batches used for variance
         n_structures_selected=min(20, n_structures_initial), # Example for README, maybe 20 or total
         hpc_profile_vasp="default", # Example profile name
         example_aa_n_batches=max(1, n_batches // 2), # Example for README
         example_structures_per_traj=1, # Example for README
         structure_type=structure_type,
         composition_constraints=composition_constraints,
         debug=debug
    )


    print(f"\n[INFO] Workflow setup complete in: {output_path}")
    print(f"[INFO] Initial structures batched in: {output_path / 'variance_calculations'}")
    print(f"[INFO] Models copied to: {output_path / 'models'}")
    print(f"[INFO] Variance results will be stored in: {variance_results_dir.relative_to(output_path)}")
    print(f"[INFO] Instructions saved in: {output_path / 'README.md'}") # Added this line
    print("[INFO] Next steps:")
    print("  1. Implement and run the variance calculation (e.g., using the generated SLURM script - currently placeholder).")
    print(f"  2. Run 'forge run-gradient-aa-jobs --input_directory {variance_results_dir.relative_to(output_path)} ...' or 'forge run-aa-jobs ...'")
    print(f"  3. Submit the generated AA optimization SLURM script.")
    print(f"  4. Run 'forge create-aa-vasp-jobs ...' to generate VASP calculations.")
