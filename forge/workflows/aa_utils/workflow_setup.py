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
from typing import List, Dict, Optional, Tuple, Any, Sequence
import ase.io
import shutil
import glob
import numpy as np
from ase import Atoms
from ase.io import read, write
from forge.core.database import DatabaseManager, fix_numpy
from forge.workflows.profiles import ProfileManager
# Removed unused imports: DisplacementGenerator, AdversarialOptimizer, AdversarialCalculator
# Import from the correct file now for VASP prep helper
from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase
from forge.workflows.adversarial_attack.slurm_templates import (
    get_variance_calculation_script,
    get_gradient_aa_script,
    get_monte_carlo_aa_script
)
import logging

# --- select_structures_from_db function ---
# (Add/improve docstring)
def select_structures_from_db(
    output_dir: Path, # ADDED: Directory to save reference data
    elements: Optional[List[str]] = None, # Make explicit that it can be None
    config_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    structure_ids: Optional[List[int]] = None,
    db_manager: Optional[DatabaseManager] = None, # Allow passing existing manager
    debug: bool = False
) -> List[Atoms]: # CHANGED: Return only list of cleaned Atoms
    """
    Selects structures from the database, saves their reference VASP calculations,
    and returns a list of cleaned Atoms objects (geometry + metadata only).

    Args:
        output_dir: The root directory for the AA workflow, used to determine
                    where to save the 'initial_references.json'.
        elements: List of elements to filter structures by. Ignored if structure_ids provided.
        config_type: Metadata config_type filter (e.g., 'bulk', 'surface').
        composition_constraints: JSON string for composition constraints (e.g., '{"Ti": [0, 1]}').
        structure_ids: List of specific structure IDs to select, bypassing other filters.
        db_manager: Optional pre-initialized DatabaseManager instance.
        debug: Enable debug output.

    Returns:
        List of cleaned ASE Atoms objects (metadata preserved, calculation results removed).

    Raises:
        ValueError: If no selection criteria are given or no structures are found/valid.
    """
    if not db_manager:
        # Initialize DB Manager if not provided
        try:
            db_manager = DatabaseManager(debug=debug)
            close_db = True # Flag to close connection later if created here
        except Exception as e:
            print(f"[ERROR] Failed to initialize DatabaseManager: {e}")
            raise
    else:
        close_db = False # Don't close connection if it was passed in

    selected_structure_ids: List[int] = []

    try: # Wrap DB operations in try...finally to ensure connection closure
        if structure_ids:
            selected_structure_ids = structure_ids
            print(f"[INFO] Using provided structure IDs: {selected_structure_ids}")
        else:
            if not elements:
                raise ValueError("Either 'elements' or 'structure_ids' must be provided.")

            query_kwargs = {'elements': elements}
            print(f"[INFO] Filtering by elements: {elements}")
            if config_type:
                # Assuming find_structures_by_metadata is the intended function for config_type
                # query_kwargs['metadata_filters'] = {'config_type': config_type} # Old way?
                 # Using the structure_type arg in find_structures instead
                query_kwargs['structure_type'] = config_type
                print(f"[INFO] Filtering by config_type: {config_type}")

            if composition_constraints:
                try:
                    constraints = json.loads(composition_constraints)
                    query_kwargs['composition_constraints'] = constraints
                    print(f"[INFO] Filtering by composition constraints: {constraints}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON for composition_constraints: {e}")

            # Use find_structures or find_structures_by_metadata based on needs
            # Assuming find_structures covers the current filters
            # TODO: come back to this to make sure the find_structures is correctly searching by element.
            selected_structure_ids = db_manager.find_structures(**query_kwargs, debug=debug)
            print(f"[INFO] Found {len(selected_structure_ids)} structure IDs matching criteria.")

        if not selected_structure_ids:
            raise ValueError("No structures selected or found matching criteria.")

        # --- Fetch structures AND their VASP calculations in one batch ---
        print("[INFO] Fetching structures and their latest VASP calculations from database...")
        # Default calculator is 'vasp' in get_batch_atoms_with_calculation
        rich_atoms_list = db_manager.get_batch_atoms_with_calculation(selected_structure_ids)
        print(f"[INFO] Retrieved {len(rich_atoms_list)} structures with calculation data.")

        initial_references = {}
        cleaned_atoms_list = []
        missing_data_count = 0

        # --- Process fetched structures: extract references, clean atoms ---
        print("[INFO] Processing structures: extracting reference data and cleaning Atoms objects...")
        for atoms in rich_atoms_list:
            atoms_info = atoms.info
            atoms_arrays = atoms.arrays

            struct_id = atoms_info.get('structure_id')
            # Extract reference data (energy/forces should be top-level from get_batch...)
            energy = atoms_info.get('energy')
            forces = atoms_arrays.get('forces')
            # calculation_info = atoms_info.get('calculation_info', {}) # Contains metadata about the calc

            if struct_id is None:
                 if debug: print("[DEBUG] Skipping structure with no structure_id in info.")
                 missing_data_count += 1
                 continue

            # Check if essential reference data is present
            if energy is None or forces is None:
                if debug: print(f"[DEBUG] Skipping structure ID {struct_id}: Missing energy or forces.")
                missing_data_count += 1
                continue

            # Store reference data (use NumPy arrays for forces if needed downstream)
            initial_references[struct_id] = {
                'energy': float(energy), # Ensure float
                # Convert forces back to list for JSON serialization if needed,
                # or keep as np.array if processing happens before saving JSON.
                # Plan indicates saving to JSON, so list is better.
                'forces': forces.tolist() if isinstance(forces, np.ndarray) else forces
                # Can add other calculation metadata here if needed, e.g.,
                # 'calculator': calculation_info.get('calculator'),
                # 'source_path': calculation_info.get('calculation_source_path'),
            }

            # --- Clean the Atoms object ---
            # Remove calculator
            atoms.calc = None
            # Remove calculation results from info and arrays
            atoms.info.pop('energy', None)
            atoms.info.pop('stress', None) # Also remove stress if present
            atoms.info.pop('calculation_info', None) # Remove the dict holding calc metadata
            atoms.arrays.pop('forces', None)

            # Append the cleaned atoms object
            cleaned_atoms_list.append(atoms)

        if missing_data_count > 0:
             print(f"[WARNING] Skipped {missing_data_count} structures due to missing ID, energy, or forces.")

        if not cleaned_atoms_list:
            raise ValueError("No valid structures with required calculation data found.")

        # --- Save initial references to JSON ---
        variance_calc_dir = output_dir / "variance_calculations"
        variance_calc_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        ref_file = variance_calc_dir / "initial_references.json"

        try:
            with open(ref_file, 'w') as f:
                # Use fix_numpy just in case, although energy/forces were handled
                json.dump(fix_numpy(initial_references), f, indent=2)
            print(f"[INFO] Saved initial reference data (energy/forces) for {len(initial_references)} structures to: {ref_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save initial references JSON to {ref_file}: {e}")
            # Decide if this is a critical error or just a warning
            raise # Re-raise for now

        print(f"[INFO] Returning {len(cleaned_atoms_list)} cleaned Atoms objects.")
        return cleaned_atoms_list

    finally:
        # Ensure DB connection is closed if opened by this function
        if close_db and db_manager and db_manager.conn:
            db_manager.close_connection()
            if debug: print("[DEBUG] Database connection closed by select_structures_from_db.")


# --- prepare_variance_calculation_jobs function ---
def prepare_variance_calculation_jobs(
    selected_atoms_list: List[Atoms], # Input list of *cleaned* atoms
    output_dir: Path,
    model_dir: Path, # Source directory for models
    n_batches: int,
    hpc_profile_name: str = "default",
    debug: bool = False, # Add debug flag
) -> Path: # Return path to results dir, not script
    """
    Prepares SLURM array job for variance calculation on cleaned structures.

    Takes a list of cleaned Atoms objects, copies necessary models, splits
    structures into batches, writes batch XYZ files, loads the specified
    HPC profile, and generates a SLURM submission script.

    Args:
        selected_atoms_list: List of cleaned ASE Atoms objects (geometry + metadata).
        output_dir: Root directory for the AA workflow.
        model_dir: Path to the directory containing MACE *.model files.
        n_batches: Number of batches (SLURM jobs) to split calculations into.
        hpc_profile_name: Name of the HPC profile (e.g., 'PSFC-GPU') to use for SLURM script.
        debug: Enable debug output.

    Returns:
        Path to the directory where variance results JSON files will be stored.

    Raises:
        ValueError: If model directory is invalid or no models are found.
        FileNotFoundError: If HPC profile is not found.
    """
    print("[INFO] Preparing variance calculation jobs...")

    # Define directories relative to output_dir
    variance_calculation_dir = output_dir / "variance_calculations"
    variance_results_dir = output_dir / "variance_results" # Where engine saves results
    slurm_logs_dir = variance_calculation_dir / "slurm_logs" # Log dir inside calc dir
    workflow_models_dir = output_dir / "models" # Models copied to workflow root

    # Create directories
    variance_calculation_dir.mkdir(parents=True, exist_ok=True)
    variance_results_dir.mkdir(parents=True, exist_ok=True)
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    workflow_models_dir.mkdir(parents=True, exist_ok=True) # Create models dir at workflow root

    # --- Copy Models ---
    print(f"[INFO] Copying models from {model_dir} to {workflow_models_dir}")
    copied_model_paths = []
    model_dir_path = Path(model_dir).resolve()
    if not model_dir_path.is_dir():
        raise ValueError(f"Model directory not found: {model_dir_path}")

    n_models_copied = 0
    for item in model_dir_path.glob("*.model"):
        if item.is_file():
            dest_path = workflow_models_dir / item.name
            try:
                shutil.copy2(item, dest_path)
                # Store relative path for script generation
                copied_model_paths.append(str(Path("models") / item.name))
                n_models_copied += 1
            except Exception as e:
                 print(f"[WARN] Failed to copy model {item.name}: {e}")

    if n_models_copied == 0:
        raise ValueError(f"No *.model files found or copied from {model_dir_path}")
    print(f"[INFO] Copied {n_models_copied} models.")

    # --- Split structures into batches ---
    n_structures = len(selected_atoms_list)
    if n_structures == 0:
        print("[WARN] No structures provided to prepare_variance_calculation_jobs. Skipping batch creation.")
        # Return variance_results_dir as jobs won't run but dir exists
        return variance_results_dir

    # Ensure n_batches is not greater than n_structures
    if n_batches > n_structures:
        print(f"[WARN] Number of batches ({n_batches}) > number of structures ({n_structures}). Setting n_batches = {n_structures}.")
        n_batches = n_structures
    elif n_batches <= 0:
        print(f"[WARN] Invalid number of batches ({n_batches}). Setting n_batches = 1.")
        n_batches = 1


    structures_per_batch, remainder = divmod(n_structures, n_batches)
    print(f"[INFO] Splitting {n_structures} structures into {n_batches} batches (~{structures_per_batch} per batch).")

    start_index = 0
    for batch_id in range(n_batches):
        batch_dir = variance_calculation_dir / f"batch_{batch_id}"
        batch_dir.mkdir(exist_ok=True)
        # XYZ file saved inside the specific batch directory
        xyz_file_path = batch_dir / f"batch_{batch_id}.xyz"

        # Calculate number of structures for this batch
        count = structures_per_batch + (1 if batch_id < remainder else 0)
        end_index = start_index + count

        batch_atoms_list = selected_atoms_list[start_index:end_index]
        start_index = end_index

        # Assign structure_name based on structure_id in info (important for results matching)
        for i, atoms in enumerate(batch_atoms_list):
            struct_id = atoms.info.get('structure_id')
            if struct_id is not None:
                 # Use a consistent naming scheme, e.g., struct_id_XXX
                 atoms.info['structure_name'] = f"struct_id_{struct_id}"
            else:
                 # Fallback if ID somehow missing (shouldn't happen with cleaned list)
                 atoms.info['structure_name'] = f'batch_{batch_id}_index_{i}_noID'
                 if debug: print(f"[DEBUG] Warning: Structure at index {i} in batch {batch_id} missing structure_id.")

        # Write batch XYZ file
        try:
            ase.io.write(xyz_file_path, batch_atoms_list, format="extxyz") # Use extxyz to preserve info
            print(f"[INFO] Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {xyz_file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write XYZ file {xyz_file_path}: {e}")
            # Decide whether to continue or raise

    # --- Load HPC Profile ---
    print(f"[INFO] Loading HPC profile: {hpc_profile_name}")
    try:
        # Correct path assuming profiles are in standard location relative to setup.py
        # Adjust based on actual project structure if needed
        # Assuming workflow_setup.py is in forge/workflows/adversarial_attack/
        hpc_profile_dir = Path(__file__).parent.parent / "hpc_profiles"
        profile_manager = ProfileManager(profile_directory=hpc_profile_dir)
        profile_manager.load_profile(hpc_profile_name)
        hpc_profile = profile_manager.get_profile(hpc_profile_name)
        hpc_profile['name'] = hpc_profile_name # Add name to dict for script header
    except FileNotFoundError:
        print(f"[ERROR] HPC profile '{hpc_profile_name}.json' not found in {hpc_profile_dir}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load HPC profile '{hpc_profile_name}': {e}")
        raise

    # --- Generate SLURM Array Job Script ---
    slurm_script_path = variance_calculation_dir / "variance_calculation_array.slurm"

    # Define paths relative to the workflow root directory (output_dir)
    # These paths will be used *inside* the SLURM script, relative to where sbatch is run
    output_dir_rel = variance_results_dir.relative_to(output_dir) # e.g., "variance_results"
    log_dir_rel = slurm_logs_dir.relative_to(output_dir) # e.g., "variance_calculations/slurm_logs"
    model_dir_rel = workflow_models_dir.relative_to(output_dir) # e.g., "models"
    # Path template to the *input* XYZ file for each batch job
    # Relative path from workflow root to the XYZ file within its batch dir
    batch_script_rel_path = str(variance_calculation_dir.relative_to(output_dir) / "batch_${SLURM_ARRAY_TASK_ID}" / "batch_${SLURM_ARRAY_TASK_ID}.xyz")
    # Example: "variance_calculations/batch_${SLURM_ARRAY_TASK_ID}/batch_${SLURM_ARRAY_TASK_ID}.xyz"

    # Extract necessary SLURM parameters from the loaded profile
    slurm_directives = hpc_profile.get("slurm_directives", {})
    job_time = slurm_directives.get("time", "06:00:00") # Default time
    cpus_per_task = int(slurm_directives.get("cpus-per-task", 1)) # Ensure int
    gpus_per_task = 0 # Default to 0

    # Handle different ways GPUs might be specified (gres or gpus key)
    if "gpus" in slurm_directives:
        try:
            gpus_per_task = int(slurm_directives["gpus"])
        except (ValueError, TypeError):
            print(f"[WARN] Could not parse 'gpus' directive: {slurm_directives['gpus']}. Defaulting to 0.")
    elif "gres" in slurm_directives and isinstance(slurm_directives["gres"], str) and "gpu" in slurm_directives["gres"]:
         try:
             # Attempt to parse standard gres format like "gpu:2" or "gpu:v100:1"
             parts = slurm_directives["gres"].split(":")
             gpus_per_task = int(parts[-1]) # Assume last part is the count
         except (ValueError, TypeError, IndexError):
             print(f"[WARN] Could not parse GPU count from 'gres' directive: {slurm_directives['gres']}. Defaulting to 0.")

    account = slurm_directives.get("account")
    partition = slurm_directives.get("partition")

    # Get the SLURM script content from the template function
    slurm_content = get_variance_calculation_script(
        output_dir_rel=str(output_dir_rel),
        log_dir_rel=str(log_dir_rel),
        model_dir_rel=str(model_dir_rel),
        batch_script_rel_path=batch_script_rel_path,
        array_range=f"0-{n_batches - 1}",
        n_models=n_models_copied,
        time=job_time,
        cpus_per_task=cpus_per_task,
        gpus_per_task=gpus_per_task, # Pass GPU count
        hpc_profile=hpc_profile, # Pass the whole profile dict
        account=account,
        partition=partition,
    )

    try:
        with open(slurm_script_path, "w") as f:
            f.write(slurm_content)
        print(f"[INFO] Created SLURM array script: {slurm_script_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write SLURM script {slurm_script_path}: {e}")
        raise

    # Return the path where results *will be* stored
    return variance_results_dir


# --- combine_variance_results function ---
# (Add/improve docstring)
def combine_variance_results(variance_dir: Path, debug: bool = False) -> List[Tuple[int, float]]: # Return (ID, variance)
    """
    Combine variance results from batch JSON files and sort by variance.

    Reads '*_variances.json' files within the specified directory, expects
    keys to be string representations of integer structure IDs, and returns
    a list sorted from highest to lowest variance.

    Args:
        variance_dir: Directory containing the variance result JSON files
                      (e.g., 'variance_results').
        debug: Enable debug output.

    Returns:
        List of tuples: (structure_id, variance), sorted by
        variance in descending order. Returns empty list if no results found.
    """
    all_results_dict = {} # Use dict to avoid duplicates if ID appears in multiple files

    variance_files = list(variance_dir.glob("*_variances.json"))
    if not variance_files:
        print(f"[WARNING] No variance result files (*_variances.json) found in {variance_dir}")
        return []

    print(f"[INFO] Found {len(variance_files)} variance result files in {variance_dir}. Combining...")

    files_read = 0
    total_entries = 0
    for json_file in variance_files:
        if debug: print(f"[DEBUG] Reading variance results from {json_file}")
        try:
            with open(json_file, 'r') as f:
                batch_results = json.load(f) # Expected format: { "struct_ID_str": variance_value, ... }
            files_read += 1
        except json.JSONDecodeError:
            print(f"[WARNING] Could not decode JSON from file: {json_file}. Skipping.")
            continue
        except Exception as e:
            print(f"[WARNING] Error reading file {json_file}: {e}. Skipping.")
            continue

        # Process results, converting keys to int IDs
        for struct_id_str, variance in batch_results.items():
            try:
                structure_id = int(struct_id_str)
                variance_float = float(variance) # Ensure variance is float
                
                # Store or update the variance for this ID
                # If duplicate ID found, keep the one from the later processed file (no specific reason, just need a rule)
                all_results_dict[structure_id] = variance_float
                total_entries += 1

            except (ValueError, TypeError):
                print(f"[WARNING] Invalid structure ID ('{struct_id_str}') or variance ('{variance}') found in {json_file}. Skipping entry.")
                continue
                
    if files_read == 0:
        print(f"[WARNING] Could not read any variance data from files in {variance_dir}.")
        return []
        
    if not all_results_dict:
        print(f"[WARNING] No valid variance entries found across {files_read} files.")
        return []

    # Convert dict to list of tuples and sort by variance (descending)
    sorted_results = sorted(all_results_dict.items(), key=lambda item: item[1], reverse=True)

    print(f"[INFO] Combined {len(sorted_results)} unique variance results from {files_read} files.")
    if debug:
        print(f"\n[DEBUG] Top 5 Variance Results:")
        for sid, var in sorted_results[:5]:
            print(f"  Structure ID {sid}: {var:.6f}")

    # Return list of (structure_id, variance) tuples
    return sorted_results 


# --- prepare_gradient_aa_optimization function ---
# (Add/improve docstring)
def prepare_gradient_aa_optimization(
    input_directory: str, # Should be variance_results dir
    model_dir: str, # Original model source dir (used for path in README)
    n_structures: int,
    n_batches: int = 1,
    learning_rate: float = 0.01,
    n_iterations: int = 60,
    min_distance: float = 1.5,
    include_probability: bool = False,
    temperature: float = 0.86, # eV
    # device: str = "cuda", # Device selected in SLURM script now
    hpc_profile_name: str = "PSFC-GPU-AA",
    debug: bool = False,
) -> None: # Returns None, prepares files and script
    """
    Prepares SLURM array job for gradient-based AA optimization.

    Selects top N structures based on combined variance results, retrieves
    them from the database, splits them into batches, copies models,
    and generates the SLURM submission script.

    Args:
        input_directory: Path to the directory containing combined variance results
                         (typically 'variance_results').
        model_dir: Path to the *original* directory containing MACE models (for copying).
        n_structures: Number of highest-variance structures to select for optimization.
        n_batches: Number of batches (SLURM jobs) to split optimization into.
        learning_rate: Learning rate for gradient ascent steps.
        n_iterations: Number of optimization iterations.
        min_distance: Minimum allowed interatomic distance (Å).
        include_probability: Whether to include probability weighting term.
        temperature: Temperature (eV) for probability weighting.
        hpc_profile_name: Name of the HPC profile for SLURM script generation.
        debug: Enable detailed debug output.
    """
    print("[INFO] Starting Gradient AA optimization job preparation...")

    # --- Setup Directories ---
    variance_results_path = Path(input_directory).resolve()
    if not variance_results_path.is_dir():
        raise ValueError(f"Variance results directory not found: {variance_results_path}")

    # Create the main directory for this AA method
    aa_output_dir = variance_results_path.parent / "gradient_aa_optimization"
    aa_output_dir.mkdir(parents=True, exist_ok=True)
    slurm_logs_dir = aa_output_dir / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)
    workflow_models_dir = aa_output_dir / "models" # Models copied inside AA dir
    workflow_models_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy Models --- # Copy models specific to this step
    print(f"[INFO] Copying models from {model_dir} to {workflow_models_dir}")
    n_models_copied = 0
    model_source_path = Path(model_dir).resolve()
    if not model_source_path.is_dir():
         raise ValueError(f"Model source directory not found: {model_source_path}")
    for item in model_source_path.glob("*.model"):
        if item.is_file():
            try:
                shutil.copy2(item, workflow_models_dir / item.name)
                n_models_copied += 1
            except Exception as e:
                 print(f"[WARN] Failed to copy model {item.name}: {e}")
    if n_models_copied == 0:
         raise ValueError(f"No *.model files found or copied from {model_source_path}")
    print(f"[INFO] Copied {n_models_copied} models.")

    # --- Combine and Sort Variance Results --- # Use updated function
    print(f"[INFO] Combining variance results from {variance_results_path}...")
    # Expected return: List[Tuple[structure_id, variance]]
    sorted_variance_results = combine_variance_results(variance_results_path, debug=debug)

    if not sorted_variance_results:
        print("[ERROR] No variance results found or processed. Cannot prepare AA jobs.")
        return
    print(f"[INFO] Found {len(sorted_variance_results)} structures with variance results.")

    # --- Select Top N Structures --- # Select based on ID, variance tuples
    if n_structures > len(sorted_variance_results):
        print(f"[WARNING] Requested {n_structures} structures, but only {len(sorted_variance_results)} available. Using all available.")
        n_structures = len(sorted_variance_results)
    elif n_structures <= 0:
        print("[ERROR] Number of structures for AA must be positive.")
        return

    # Get list of (ID, variance) tuples for the top N
    selected_structures_info = sorted_variance_results[:n_structures]
    selected_ids = [sid for sid, _ in selected_structures_info]
    print(f"[INFO] Selected top {len(selected_ids)} structures for Gradient AA based on variance.")
    if debug:
        print(f"[DEBUG] Selected IDs and Variances:")
        for sid, var in selected_structures_info[:5]: # Print top 5
             print(f"  ID: {sid}, Variance: {var:.6f}")

    # --- Prepare Batches --- #
    # Ensure n_batches is valid
    if n_batches > len(selected_ids):
        print(f"[WARN] Number of batches ({n_batches}) > number of selected structures ({len(selected_ids)}). Setting n_batches = {len(selected_ids)}.")
        n_batches = len(selected_ids)
    elif n_batches <= 0:
        print(f"[WARN] Invalid number of batches ({n_batches}). Setting n_batches = 1.")
        n_batches = 1

    structures_per_batch, remainder = divmod(len(selected_ids), n_batches)
    print(f"[INFO] Splitting {len(selected_ids)} structures into {n_batches} AA batches (~{structures_per_batch} per batch).")

    # Need DB manager to fetch the actual structures for batching
    db_manager = None
    try:
        db_manager = DatabaseManager(debug=debug)

        current_index = 0
        for batch_id in range(n_batches):
            batch_dir = aa_output_dir / f"batch_{batch_id}"
            batch_dir.mkdir(exist_ok=True)
            # Engine output goes into a subfolder named after the batch
            engine_output_dir = batch_dir / f"aa_batch_{batch_id}_output" 
            # We don't create engine_output_dir here; the engine itself should do that.
            xyz_file_path = batch_dir / f"batch_{batch_id}_input.xyz" # Input XYZ for this batch

            # Determine structure IDs and variances for this batch
            count = structures_per_batch + (1 if batch_id < remainder else 0)
            batch_structures_info = selected_structures_info[current_index : current_index + count]
            batch_ids = [sid for sid, _ in batch_structures_info]
            current_index += count

            if not batch_ids:
                 print(f"[WARN] Batch {batch_id} has no structures assigned. Skipping.")
                 continue

            # Fetch the actual Atoms objects for this batch (use cleaned atoms)
            # We need the geometry and original metadata, so fetch without calculation data
            batch_atoms_map = db_manager.get_structures_batch(batch_ids)
            batch_atoms_list = [batch_atoms_map.get(sid) for sid in batch_ids if batch_atoms_map.get(sid) is not None]

            if len(batch_atoms_list) != len(batch_ids):
                 print(f"[WARN] Batch {batch_id}: Could not retrieve all expected structures from DB ({len(batch_atoms_list)}/{len(batch_ids)} found).")

            if not batch_atoms_list:
                 print(f"[WARN] Batch {batch_id}: No valid structures retrieved from DB. Skipping.")
                 continue

            # Add initial variance info back to Atoms objects for the engine
            variance_map = dict(batch_structures_info)
            for atoms in batch_atoms_list:
                struct_id = atoms.info.get('structure_id')
                if struct_id in variance_map:
                    atoms.info['initial_variance'] = variance_map[struct_id]
                # Ensure structure_name is set using ID for consistency
                atoms.info['structure_name'] = f"struct_id_{struct_id}"

            # Write batch XYZ file
            try:
                ase.io.write(xyz_file_path, batch_atoms_list, format="extxyz")
                print(f"[INFO] Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {xyz_file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to write XYZ file {xyz_file_path}: {e}")

    finally:
         if db_manager: db_manager.close_connection()

    # --- Load HPC Profile --- # (Same logic as variance job prep)
    print(f"[INFO] Loading HPC profile: {hpc_profile_name}")
    try:
        hpc_profile_dir = Path(__file__).parent.parent / "hpc_profiles"
        profile_manager = ProfileManager(profile_directory=hpc_profile_dir)
        profile_manager.load_profile(hpc_profile_name)
        hpc_profile = profile_manager.get_profile(hpc_profile_name)
        hpc_profile['name'] = hpc_profile_name
    except FileNotFoundError: print(f"[ERROR] HPC profile '{hpc_profile_name}.json' not found."); return
    except Exception as e: print(f"[ERROR] Failed to load HPC profile '{hpc_profile_name}': {e}"); return

    # --- Generate SLURM Script --- #
    slurm_script_path = aa_output_dir / "gradient_aa_optimization_array.slurm"

    # Define paths relative to the AA output directory for use inside the script
    # The script will be run from aa_output_dir
    log_dir_rel = slurm_logs_dir.relative_to(aa_output_dir) # e.g., "slurm_logs"
    model_dir_rel = workflow_models_dir.relative_to(aa_output_dir) # e.g., "models"
    # Base path for batch directories relative to aa_output_dir
    batch_base_rel = "." # Script runs from aa_output_dir
    # Relative path from batch_base_rel to the input XYZ file
    structure_file_rel_template = f"batch_${{BATCH_ID}}/batch_${{BATCH_ID}}_input.xyz"
    # Relative path from batch_base_rel to where the engine should save output
    engine_output_dir_rel_template = f"batch_${{BATCH_ID}}/aa_batch_${{BATCH_ID}}_output"

    # Extract SLURM params
    slurm_directives = hpc_profile.get("slurm_directives", {})
    job_time = slurm_directives.get("time", "24:00:00")
    cpus_per_task = int(slurm_directives.get("cpus-per-task", 1))
    gpus_per_task = 0 # Default
    if "gpus" in slurm_directives:
         try: gpus_per_task = int(slurm_directives["gpus"]);
         except (ValueError, TypeError): pass
    elif "gres" in slurm_directives and isinstance(slurm_directives["gres"], str) and "gpu" in slurm_directives["gres"]:
         try: parts = slurm_directives["gres"].split(":"); gpus_per_task = int(parts[-1]);
         except (ValueError, TypeError, IndexError): pass
    account = slurm_directives.get("account")
    partition = slurm_directives.get("partition")

    # Determine device based on GPU request for SLURM script args
    device = "cuda" if gpus_per_task > 0 else "cpu"

    slurm_content = get_gradient_aa_script(
        batch_base_rel=batch_base_rel,
        log_dir_rel=str(log_dir_rel),
        model_dir_rel=str(model_dir_rel),
        structure_file_rel=structure_file_rel_template,
        engine_output_dir_rel=engine_output_dir_rel_template,
        array_range=f"0-{n_batches - 1}",
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        min_distance=min_distance,
        include_probability=include_probability,
        temperature=temperature, # eV
        device=device, # Pass detected device
        save_trajectory=True, # Hardcode saving trajectory for now
        time=job_time,
        cpus_per_task=cpus_per_task,
        gpus_per_task=gpus_per_task,
        hpc_profile=hpc_profile,
        account=account,
        partition=partition,
    )

    try:
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_content)
        print(f"[INFO] Created Gradient AA SLURM script: {slurm_script_path}")
        print(f"[INFO] Run this script from the '{aa_output_dir}' directory.")
    except Exception as e:
        print(f"[ERROR] Failed to write SLURM script {slurm_script_path}: {e}")


# --- prepare_monte_carlo_aa_optimization function ---
# (Add/improve docstring)
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
    # device: str = "cuda", # Device selected in SLURM script now
    hpc_profile_name: str = "default",
    debug: bool = False,
) -> None:
    """
    Prepares SLURM array job for Monte Carlo AA optimization.

    Selects top N structures based on combined variance results, retrieves
    them from the database, splits them into batches, copies models,
    and generates the SLURM submission script.

    Args:
        input_directory: Path to the directory containing combined variance results
                         (typically 'variance_results').
        model_dir: Path to the *original* directory containing MACE models (for copying).
        n_structures: Number of highest-variance structures to select for optimization.
        n_batches: Number of batches (SLURM jobs) to split optimization into.
        temperature: Temperature (K) for Metropolis acceptance.
        max_steps: Maximum number of MC steps.
        patience: Stop if max variance doesn't improve for this many steps.
        min_distance: Minimum allowed interatomic distance (Å).
        max_displacement: Maximum distance an atom can be moved per step (Å).
        mode: Atom displacement mode ('all' or 'single').
        hpc_profile_name: Name of the HPC profile for SLURM script generation.
        debug: Enable detailed debug output.
    """
    print("[INFO] Starting Monte Carlo AA optimization job preparation...")

    # --- Setup Directories --- (Similar to Gradient AA)
    variance_results_path = Path(input_directory).resolve()
    if not variance_results_path.is_dir():
        raise ValueError(f"Variance results directory not found: {variance_results_path}")

    aa_output_dir = variance_results_path.parent / "monte_carlo_aa_optimization"
    aa_output_dir.mkdir(parents=True, exist_ok=True)
    slurm_logs_dir = aa_output_dir / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)
    workflow_models_dir = aa_output_dir / "models"
    workflow_models_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy Models --- (Identical to Gradient AA)
    print(f"[INFO] Copying models from {model_dir} to {workflow_models_dir}")
    n_models_copied = 0
    model_source_path = Path(model_dir).resolve()
    if not model_source_path.is_dir():
         raise ValueError(f"Model source directory not found: {model_source_path}")
    for item in model_source_path.glob("*.model"):
        if item.is_file():
            try:
                shutil.copy2(item, workflow_models_dir / item.name)
                n_models_copied += 1
            except Exception as e:
                 print(f"[WARN] Failed to copy model {item.name}: {e}")
    if n_models_copied == 0:
         raise ValueError(f"No *.model files found or copied from {model_source_path}")
    print(f"[INFO] Copied {n_models_copied} models.")


    # --- Combine and Sort Variance Results --- (Identical to Gradient AA)
    print(f"[INFO] Combining variance results from {variance_results_path}...")
    sorted_variance_results = combine_variance_results(variance_results_path, debug=debug)
    if not sorted_variance_results:
        print("[ERROR] No variance results found or processed. Cannot prepare AA jobs.")
        return
    print(f"[INFO] Found {len(sorted_variance_results)} structures with variance results.")

    # --- Select Top N Structures --- (Identical to Gradient AA)
    if n_structures > len(sorted_variance_results):
        print(f"[WARNING] Requested {n_structures} structures, but only {len(sorted_variance_results)} available. Using all available.")
        n_structures = len(sorted_variance_results)
    elif n_structures <= 0:
        print("[ERROR] Number of structures for AA must be positive.")
        return
    selected_structures_info = sorted_variance_results[:n_structures]
    selected_ids = [sid for sid, _ in selected_structures_info]
    print(f"[INFO] Selected top {len(selected_ids)} structures for Monte Carlo AA based on variance.")
    if debug:
        print(f"[DEBUG] Selected IDs and Variances:")
        for sid, var in selected_structures_info[:5]:
             print(f"  ID: {sid}, Variance: {var:.6f}")

    # --- Prepare Batches --- (Identical to Gradient AA)
    if n_batches > len(selected_ids):
        print(f"[WARN] Number of batches ({n_batches}) > number of selected structures ({len(selected_ids)}). Setting n_batches = {len(selected_ids)}.")
        n_batches = len(selected_ids)
    elif n_batches <= 0:
        print(f"[WARN] Invalid number of batches ({n_batches}). Setting n_batches = 1.")
        n_batches = 1
    structures_per_batch, remainder = divmod(len(selected_ids), n_batches)
    print(f"[INFO] Splitting {len(selected_ids)} structures into {n_batches} AA batches (~{structures_per_batch} per batch).")

    db_manager = None
    try:
        db_manager = DatabaseManager(debug=debug)
        current_index = 0
        for batch_id in range(n_batches):
            batch_dir = aa_output_dir / f"batch_{batch_id}"
            batch_dir.mkdir(exist_ok=True)
            engine_output_dir = batch_dir / f"aa_batch_{batch_id}_output" 
            xyz_file_path = batch_dir / f"batch_{batch_id}_input.xyz"

            count = structures_per_batch + (1 if batch_id < remainder else 0)
            batch_structures_info = selected_structures_info[current_index : current_index + count]
            batch_ids = [sid for sid, _ in batch_structures_info]
            current_index += count

            if not batch_ids: continue # Skip empty batch

            batch_atoms_map = db_manager.get_structures_batch(batch_ids)
            batch_atoms_list = [batch_atoms_map.get(sid) for sid in batch_ids if batch_atoms_map.get(sid) is not None]
            if len(batch_atoms_list) != len(batch_ids): print(f"[WARN] Batch {batch_id}: DB retrieval mismatch.")
            if not batch_atoms_list: continue # Skip if no atoms retrieved

            variance_map = dict(batch_structures_info)
            for atoms in batch_atoms_list:
                struct_id = atoms.info.get('structure_id')
                if struct_id in variance_map: atoms.info['initial_variance'] = variance_map[struct_id]
                # Ensure structure_name is set using ID for consistency
                atoms.info['structure_name'] = f"struct_id_{struct_id}"

            try:
                ase.io.write(xyz_file_path, batch_atoms_list, format="extxyz")
                print(f"[INFO] Batch {batch_id}: Wrote {len(batch_atoms_list)} structures to {xyz_file_path}")
            except Exception as e: print(f"[ERROR] Failed to write XYZ file {xyz_file_path}: {e}")
    finally:
         if db_manager: db_manager.close_connection()

    # --- Load HPC Profile --- (Identical to Gradient AA)
    print(f"[INFO] Loading HPC profile: {hpc_profile_name}")
    try:
        hpc_profile_dir = Path(__file__).parent.parent / "hpc_profiles"
        profile_manager = ProfileManager(profile_directory=hpc_profile_dir)
        profile_manager.load_profile(hpc_profile_name)
        hpc_profile = profile_manager.get_profile(hpc_profile_name)
        hpc_profile['name'] = hpc_profile_name
    except FileNotFoundError: print(f"[ERROR] HPC profile '{hpc_profile_name}.json' not found."); return
    except Exception as e: print(f"[ERROR] Failed to load HPC profile '{hpc_profile_name}': {e}"); return

    # --- Generate SLURM Script --- # (Similar path logic to Gradient AA)
    slurm_script_path = aa_output_dir / "monte_carlo_aa_optimization_array.slurm"

    log_dir_rel = slurm_logs_dir.relative_to(aa_output_dir)
    model_dir_rel = workflow_models_dir.relative_to(aa_output_dir)
    batch_base_rel = "."
    structure_file_rel_template = f"batch_${{BATCH_ID}}/batch_${{BATCH_ID}}_input.xyz"
    engine_output_dir_rel_template = f"batch_${{BATCH_ID}}/aa_batch_${{BATCH_ID}}_output"

    # Extract SLURM params (Identical to Gradient AA)
    slurm_directives = hpc_profile.get("slurm_directives", {})
    job_time = slurm_directives.get("time", "24:00:00")
    cpus_per_task = int(slurm_directives.get("cpus-per-task", 1))
    gpus_per_task = 0
    if "gpus" in slurm_directives:
         try: gpus_per_task = int(slurm_directives["gpus"]);
         except (ValueError, TypeError): pass
    elif "gres" in slurm_directives and isinstance(slurm_directives["gres"], str) and "gpu" in slurm_directives["gres"]:
         try: parts = slurm_directives["gres"].split(":"); gpus_per_task = int(parts[-1]);
         except (ValueError, TypeError, IndexError): pass
    account = slurm_directives.get("account")
    partition = slurm_directives.get("partition")
    device = "cuda" if gpus_per_task > 0 else "cpu"

    slurm_content = get_monte_carlo_aa_script(
        batch_base_rel=batch_base_rel,
        log_dir_rel=str(log_dir_rel),
        model_dir_rel=str(model_dir_rel),
        structure_file_rel=structure_file_rel_template,
        engine_output_dir_rel=engine_output_dir_rel_template,
        array_range=f"0-{n_batches - 1}",
        max_steps=max_steps,
        patience=patience,
        temperature=temperature, # K
        min_distance=min_distance,
        max_displacement=max_displacement,
        mode=mode,
        device=device,
        save_trajectory=True,
        time=job_time,
        cpus_per_task=cpus_per_task,
        gpus_per_task=gpus_per_task,
        hpc_profile=hpc_profile,
        account=account,
        partition=partition,
    )

    try:
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_content)
        print(f"[INFO] Created Monte Carlo AA SLURM script: {slurm_script_path}")
        print(f"[INFO] Run this script from the '{aa_output_dir}' directory.")
    except Exception as e:
        print(f"[ERROR] Failed to write SLURM script {slurm_script_path}: {e}")


# --- select_structures_from_trajectory function ---
# (Add/improve docstring)
def select_structures_from_trajectory(
    trajectory_file: Path,
    # n_structures: int, # Replaced by selection_mode and value
    selection_mode: str = 'total', # 'total' or 'every_n'
    selection_value: int = 1,
    optimization_summary: Optional[dict] = None, # Pass summary dict directly
    struct_name_filter: Optional[str] = None, # Filter for specific structure in summary
    debug: bool = False,
) -> List[Tuple[Atoms, int, Optional[float]]]: # Variance can be None
    """
    Selects structures from an AA optimization trajectory file (XYZ).

    Reads an XYZ trajectory, selects structures based on the chosen mode,
    always including the final structure, and attempts to associate variance values
    from the provided optimization summary data.

    Args:
        trajectory_file: Path to the XYZ trajectory file (e.g., *_adversarial.xyz).
        selection_mode: How to select structures from trajectories ('total' or 'every_n').
        selection_value: N value for the chosen selection mode.
        optimization_summary: Loaded JSON data from 'optimization_summary.json'.
        struct_name_filter: The 'structure_name' to look for within the summary's
                            'results' list to find the correct variance history.
        debug: Enable debug output.

    Returns:
        List of tuples: (atoms, step_number, variance). Variance may be None
        if not found in the summary. Returns empty list on failure.

    Raises:
        ValueError: If trajectory is empty or selection parameters are invalid.
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

    n_total_steps = len(trajectory) # Total number of frames (indices 0 to n-1)
    final_index = n_total_steps - 1

    # --- Extract step variances and parameters from summary ---
    step_variances = {} # Map: step_index -> variance
    aa_temperature = None # Temperature used during AA optimization
    initial_variance_from_summary = None

    if optimization_summary and struct_name_filter:
        struct_result = None
        for res in optimization_summary.get('results', []):
            if res.get('structure_name') == struct_name_filter:
                struct_result = res
                break
        if struct_result:
            if 'step_variances' in struct_result:
                step_variances = {i: v for i, v in enumerate(struct_result['step_variances'])}
            elif 'loss_history' in struct_result:
                 step_variances = {i: v for i, v in enumerate(struct_result['loss_history'])}
            initial_variance_from_summary = struct_result.get('initial_variance')
            method = optimization_summary.get('parameters', {}).get('method')
            aa_temperature = optimization_summary.get('parameters', {}).get('temperature')

    # --- Select structure indices based on mode ---
    indices = set() # Use a set to avoid duplicates

    # Always include the final structure
    if final_index >= 0:
        indices.add(final_index)

    if selection_mode == 'total':
        n_structures_to_select = selection_value
        if n_structures_to_select <= 0:
            print("[WARNING] 'total' selection requires a positive value. Selecting only final structure.")
            n_structures_to_select = 1
        if n_structures_to_select > n_total_steps:
             print(f"[WARNING] Requested total {n_structures_to_select} structures, but trajectory only has {n_total_steps}. Selecting all available.")
             n_structures_to_select = n_total_steps

        # Select n_structures_to_select, including the last one (already added)
        if n_structures_to_select > 1:
             # Select remaining N-1 structures from indices 0 to N-2
             num_remaining_to_select = n_structures_to_select - 1
             available_indices_range = final_index # Indices 0 to final_index - 1

             if available_indices_range > 0:
                 # Calculate spacing across the available earlier indices
                 step_interval = available_indices_range / num_remaining_to_select
                 for i in range(num_remaining_to_select):
                     # Calculate index by spacing back from the second-to-last index
                     idx = round((final_index - 1) - i * step_interval)
                     indices.add(max(0, int(idx))) # Ensure non-negative integer index
             else:
                 # Not enough earlier structures, just add index 0 if it exists and isn't final
                 if final_index > 0:
                     indices.add(0)

    elif selection_mode == 'every_n':
        step = selection_value
        if step <= 0:
            print("[WARNING] 'every_n' selection requires a positive step value. Selecting only final structure.")
            step = n_total_steps # Effectively only selects the last one

        # Add every Nth structure counting backwards from the end
        current_index = final_index
        while current_index >= 0:
            indices.add(current_index)
            current_index -= step

    else:
        raise ValueError(f"Invalid selection_mode: '{selection_mode}'. Choose 'total' or 'every_n'.")

    # Convert set to sorted list (descending indices)
    sorted_indices = sorted(list(indices), reverse=True)

    # --- Get selected Atoms objects and add metadata ---
    selected_tuples = []
    for step_index in sorted_indices:
        try:
            atoms = trajectory[step_index].copy() # Get a copy
            step_variance: Optional[float] = step_variances.get(step_index) # Variance might be None

            # Populate atoms.info for VASP job creation
            original_info = trajectory[step_index].info
            atoms.info['aa_step'] = step_index
            atoms.info['aa_step_variance'] = step_variance
            atoms.info['aa_temperature'] = aa_temperature
            atoms.info['initial_variance'] = initial_variance_from_summary
            if 'parent_structure_id' in original_info:
                atoms.info['parent_structure_id'] = original_info['parent_structure_id']
            elif 'structure_id' in original_info:
                 atoms.info['parent_structure_id'] = original_info['structure_id']
            if 'config_type' in original_info:
                 atoms.info['parent_config_type'] = original_info['config_type']

            selected_tuples.append((atoms, step_index, step_variance))
        except IndexError:
            print(f"[WARNING] Could not retrieve step {step_index} from trajectory {trajectory_file}")
        except Exception as e:
            print(f"[WARNING] Error processing step {step_index} from {trajectory_file}: {e}")

    if debug:
        print(f"[DEBUG] Selected {len(selected_tuples)} structures from {trajectory_file} at steps: {sorted_indices}")

    return selected_tuples


# --- prepare_vasp_jobs function ---
# (Add/improve docstring)
def prepare_vasp_jobs(
    input_directory: str, # AA results dir (e.g., gradient_aa_optimization)
    output_directory: str, # Dir to create VASP jobs in
    vasp_profile: str = "static",
    hpc_profile: str = "default",
    selection_mode: str = 'total', # Default to selecting N total
    selection_value: int = 1,      # Default to selecting only the final structure
    generation: Optional[int] = None, # ADD generation parameter
    debug: bool = False,
) -> None:
    """
    Creates VASP jobs for structures resulting from AA optimization.

    Scans the AA results directory for trajectories, selects structures based on
    `selection_mode` and `selection_value`, adds them to the database with
    appropriate metadata, and prepares VASP calculation directories using
    specified profiles.

    Args:
        input_directory: Directory containing AA optimization results (e.g.,
                         'gradient_aa_optimization' or 'monte_carlo_aa_optimization').
                         Expected to contain batch_*/aa_results subdirectories or results directly.
        output_directory: Directory where the VASP job subdirectories will be created.
        vasp_profile: Name of the VASP settings profile (defined in forge config).
        hpc_profile: Name of the HPC profile for job scripts (defined in forge config).
        selection_mode: How to select structures from trajectories ('total' or 'every_n').
        selection_value: N value for the chosen selection mode.
        generation: Optional integer to assign as the generation number in metadata.
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
        if any(input_path.glob("*_adversarial.xyz")) or any(input_path.glob("*_optimized.xyz")):
             print(f"[INFO] Found trajectory/optimized files directly in {input_path}. Processing...")
             aa_results_dirs = [input_path] # Process input dir itself
        else:
             print(f"[INFO] No AA result files found in {input_path}. Exiting VASP prep.")
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
            print(f"[WARNING] Optimization summary not found for results in {results_dir}. Variance info might be missing.")


        # Find trajectories or optimized files in the results directory
        # Prioritize trajectory files if they exist
        result_files = list(results_dir.glob("*_adversarial.xyz"))
        if not result_files:
             result_files = list(results_dir.glob("*_optimized.xyz"))

        if not result_files:
             print(f"[INFO] No trajectory or optimized structure files found in {results_dir}. Skipping.")
             continue

        print(f"[INFO] Processing {len(result_files)} trajectories/structures in {results_dir}...")

        # Process each trajectory/optimized structure file
        for result_file in result_files:
            # Extract base name to identify structure in summary
            if result_file.name.endswith("_adversarial.xyz"):
                 struct_name_filter = result_file.stem.replace('_adversarial', '')
            elif result_file.name.endswith("_optimized.xyz"):
                 struct_name_filter = result_file.stem.replace('_optimized', '')
            else:
                 struct_name_filter = result_file.stem
            if debug: print(f"[DEBUG] Processing file: {result_file.name}, Base name: {struct_name_filter}")

            try:
                # Select structures using the updated helper function
                selected_structures = select_structures_from_trajectory(
                    result_file,
                    selection_mode=selection_mode,
                    selection_value=selection_value,
                    optimization_summary=optimization_summary,
                    struct_name_filter=struct_name_filter,
                    debug=debug # Pass debug flag
                )

                if not selected_structures:
                    print(f"[INFO] No structures selected from {result_file.name}. Skipping.")
                    continue

                # Process each selected structure (Atoms, step_index, variance)
                for atoms, step, variance in selected_structures:
                    # --- Prepare metadata for database and VASP job ---
                    parent_id = atoms.info.get('parent_structure_id')
                    parent_config_type = atoms.info.get('parent_config_type', 'unknown')
                    optimization_method = atoms.info.get('optimization_method', 'aa') # Get method if saved

                    # Define new config type based on parent and step
                    if parent_config_type and parent_config_type != 'unknown':
                        config_type = f"{parent_config_type}_{optimization_method}" # Removed step number
                    else:
                        config_type = f"{optimization_method}" # Fallback, removed step number

                    # Construct metadata dictionary
                    structure_meta = {
                        "parent_structure_id": parent_id,
                        "parent_config_type": parent_config_type,
                        "config_type": config_type,
                        "aa_step": step,
                        "aa_step_variance": variance,
                        "aa_temperature": atoms.info.get('aa_temperature'), # Temp used during AA
                        "initial_variance": atoms.info.get('initial_variance'), # From original structure
                        "optimization_method": optimization_method,
                        # Add batch ID if possible (how to get this reliably?)
                        # 'batch_id': batch_id # Needs passing down or extracting from path
                    }
                    # Add generation if provided
                    if generation is not None:
                        structure_meta["generation"] = generation

                    # Clean up None values for cleaner JSON/DB entry
                    structure_meta = {k: v for k, v in structure_meta.items() if v is not None}


                    # --- Add structure to database ---
                    try:
                        # Add the selected structure (potentially intermediate frame)
                        new_id = db_manager.add_structure(
                            atoms,
                            metadata=structure_meta,
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
                print(f"[ERROR] Failed to process trajectory file {result_file}: {e}")
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
    # example_structures_per_traj: int = 1, # Replaced by selection mode/value
    example_selection_mode: str = 'total',
    example_selection_value: int = 1,
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
        example_selection_mode: Example trajectory selection mode for VASP step.
        example_selection_value: Example N value for trajectory selection.
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

**1. Run Initial Variance Calculation**

   *   Submit the SLURM job to calculate initial force variances:
       ```bash
       # Run from this directory ({workflow_dir_name}/)
       sbatch {variance_calc_dir}/variance_calculation_array.slurm
       ```
   *   Wait for the jobs to complete. Results (JSON files) will be saved in `{variance_results_dir}/`.

**2. Prepare Adversarial Attack Optimization Jobs**

   *   After variance calculations are complete, choose **one** of the following methods:

   *   **Option A: Gradient-Based AA**
       ```bash
       # Run from this directory ({workflow_dir_name}/)
       forge run-gradient-aa-jobs \\
           --input-directory {variance_results_dir} \\
           --model-dir {models_dir} \\
           --n-structures {n_structures_selected} `# Number of top variance structures` \\
           --n-batches {example_aa_n_batches} `# Number of optimization jobs` \\
           --learning-rate 0.01 \\
           --n-iterations 60 \\
           --device cuda `# or cpu`
       ```
       This creates job setup in `{gradient_aa_dir}/` and the SLURM script `{gradient_aa_dir}/gradient_aa_optimization_array.slurm`.

   *   **Option B: Monte Carlo AA**
       ```bash
       # Run from this directory ({workflow_dir_name}/)
       forge run-aa-jobs \\
           --input-directory {variance_results_dir} \\
           --model-dir {models_dir} \\
           --n-structures {n_structures_selected} `# Number of top variance structures` \\
           --n-batches {example_aa_n_batches} `# Number of optimization jobs` \\
           --temperature 1200 `# Metropolis temp (K)` \\
           --max-steps 50 \\
           --max-displacement 0.1 \\
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
           --input-directory {gradient_aa_dir} `# or {mc_aa_dir}` \\
           --output-directory {vasp_jobs_dir} \\
           --vasp-profile static `# Your desired VASP profile` \\
           --hpc-profile {hpc_profile_vasp} `# Your HPC profile` \\
           --selection-mode {example_selection_mode} \\
           --selection-value {example_selection_value} `# Select structures from trajectory`
       ```
   *   This creates VASP job directories in `{vasp_jobs_dir}/` and adds the selected structures to the database.

**5. Submit VASP Jobs**

   *   Navigate into the individual job directories within `{vasp_jobs_dir}/` and submit them according to your cluster's procedures (usually involves running `sbatch job_script.slurm` inside each job folder).
   *   Alternatively, if you want to run VASP on a different machine, you can later query the database for structures with `config_type` like `*_aa_s*` that don't have VASP calculations and generate jobs there.

## Directory Structure

*   `{models_dir}/`: Copied MACE model ensemble files.
*   `{variance_calc_dir}/`: Batch files and SLURM script for initial variance calculation.
*   `{variance_results_dir}/`: Results from variance calculations (JSON files).
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


# --- Main Workflow Preparation Function ---

# UPDATED function: Orchestrates the initial workflow setup
def prepare_aa_workflow(
    output_dir: str, # Changed to string for CLI compatibility, convert to Path internally
    model_dir: str, # Changed to string for CLI compatibility, convert to Path internally
    elements: Optional[List[str]] = None,
    n_batches_variance: int = 1,
    structure_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    structure_ids: Optional[List[int]] = None,
    hpc_profile_name: str = "default", # Added hpc_profile_name
    debug: bool = False,
    # --- Add parameters needed for README generation --- #
    # These might come from the CLI command or have defaults
    n_structures_aa: int = 10, # Example: number of structures to select for AA later
    hpc_profile_vasp: str = "default", # Example VASP HPC profile for README
    example_aa_n_batches: int = 1, # Example N batches for AA step in README
    example_selection_mode: str = 'total',
    example_selection_value: int = 1,
) -> None:
    """
    Prepares the initial adversarial attack workflow directory.

    This involves selecting structures, saving reference data, preparing
    variance calculation jobs (including copying models and generating SLURM script),
    and generating a README file with instructions.

    Args:
        output_dir: Directory path to create the workflow in.
        model_dir: Path to the directory containing MACE models.
        elements: List of elements to filter structures by.
        n_batches_variance: Number of batches for the initial variance calculation.
        structure_type: Optional structure type filter.
        composition_constraints: Optional JSON string for composition constraints.
        structure_ids: Optional list of specific structure IDs to use.
        hpc_profile_name: Name of the HPC profile for variance calculation SLURM script.
        debug: Enable detailed debug output.
        n_structures_aa: Number of structures planned for subsequent AA step (for README).
        hpc_profile_vasp: Name of HPC profile typically used for VASP (for README).
        example_aa_n_batches: Example number of batches for AA step (for README).
        example_selection_mode: Example trajectory selection mode for VASP step.
        example_selection_value: Example N value for trajectory selection.
    """
    print("[INFO] Starting adversarial attack workflow preparation...")
    output_path = Path(output_dir).resolve()
    model_path = Path(model_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Select Structures, Save References, Get Cleaned List ---
    print("[INFO] Step 1: Selecting structures and saving reference data...")
    try:
        # Pass output_path to select_structures_from_db for saving references
        cleaned_atoms_list = select_structures_from_db(
            output_dir=output_path,
            elements=elements,
            config_type=structure_type,
            composition_constraints=composition_constraints,
            structure_ids=structure_ids,
            # db_manager=None, # Let it initialize its own
            debug=debug
        )
    except ValueError as e:
        print(f"[ERROR] Structure selection failed: {e}")
        return # Stop workflow if selection fails
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during structure selection: {e}")
        # Optionally re-raise or log traceback if debug is True
        if debug: import traceback; traceback.print_exc()
        return

    if not cleaned_atoms_list:
        print("[ERROR] No valid structures were selected or processed. Aborting workflow preparation.")
        return

    num_selected_structures = len(cleaned_atoms_list)
    print(f"[INFO] Successfully selected and cleaned {num_selected_structures} structures.")

    # --- Step 2: Prepare Variance Calculation Jobs ---
    print("[INFO] Step 2: Preparing variance calculation jobs...")
    try:
        variance_results_dir = prepare_variance_calculation_jobs(
            selected_atoms_list=cleaned_atoms_list,
            output_dir=output_path,
            model_dir=model_path,
            n_batches=n_batches_variance,
            hpc_profile_name=hpc_profile_name,
            debug=debug
        )
    except (ValueError, FileNotFoundError, IOError) as e:
        print(f"[ERROR] Failed to prepare variance calculation jobs: {e}")
        return # Stop workflow if job prep fails
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during variance job preparation: {e}")
        if debug: import traceback; traceback.print_exc()
        return

    print(f"[INFO] Variance calculation jobs prepared. Results will appear in: {variance_results_dir}")

    # --- Step 3: Generate README ---
    print("[INFO] Step 3: Generating workflow README.md...")
    try:
        # Use relative paths for README clarity
        try:
            model_dir_rel = os.path.relpath(model_path, output_path)
        except ValueError:
            # Handle cases where paths are on different drives (Windows)
            print(f"[WARN] Cannot determine relative path for model directory '{model_path}' from '{output_path}'. Using absolute path in README.")
            model_dir_rel = str(model_path)

        # Determine if structure IDs were the primary selection method
        used_structure_ids = bool(structure_ids)

        generate_workflow_readme(
            output_dir=output_path,
            model_dir_rel=model_dir_rel,
            elements=elements if not used_structure_ids else None, # Pass elements only if IDs weren't primary
            structure_ids=structure_ids if used_structure_ids else None, # Pass IDs if they were used
            n_batches_variance=n_batches_variance,
            n_structures_selected=n_structures_aa, # Number actually selected
            hpc_profile_vasp=hpc_profile_vasp, # Example VASP profile
            example_aa_n_batches=example_aa_n_batches,
            example_selection_mode=example_selection_mode,
            example_selection_value=example_selection_value,
            # Pass filters used for selection (relevant if IDs weren't used)
            structure_type=structure_type if not used_structure_ids else None,
            composition_constraints=composition_constraints if not used_structure_ids else None,
            debug=debug # Pass debug flag to README generator if it uses it
        )
        print(f"[INFO] Generated README.md in {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate README.md: {e}")
        if debug: import traceback; traceback.print_exc()
        # Workflow is mostly setup, maybe don't abort just for README failure?
        print("[WARN] README generation failed, but workflow setup might be partially complete.")

    print("[INFO] Adversarial attack workflow preparation completed successfully.")
    print(f"[INFO] Next steps: Submit the SLURM script at {output_path / 'variance_calculations' / 'variance_calculation_array.slurm'}")

