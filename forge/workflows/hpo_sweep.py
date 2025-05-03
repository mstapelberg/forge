import os
import json
import random
import math
import itertools
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import subprocess # Keep for potential future use, but not for command guessing

from forge.core.database import DatabaseManager
# Assuming these functions primarily generate config files and data links/copies
# and we can potentially ignore the .sh they might create.
# We will now use _save_structures_to_xyz and _replace_properties directly here.
from forge.workflows.db_to_mace import _get_vasp_structures, _save_structures_to_xyz, _replace_properties # Removed GPUConfig import, prepare_mace_job import for now
from forge.workflows.db_to_allegro import prepare_allegro_job # Keep this import for now

# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- REMOVED Helper function to guess training command ---
# def _get_training_command(model_type: str, run_dir: Path) -> Optional[str]:
#     ... # Removed

def run_hpo_sweep(
    db_manager: DatabaseManager,
    model_type: str, # 'mace' or 'allegro'
    base_sweep_dir: Union[str, Path],
    sweep_params: Dict[str, List[Any]],
    fixed_params: Dict[str, Any],
    num_seeds: int = 1,
    k_folds: Optional[int] = None,
    test_ratio: Optional[float] = 0.1, # Only used if k_folds > 1
    master_seed: int = 42,
):
    """
    Runs a hyperparameter optimization sweep for MACE or Allegro models.

    Generates job directories with configuration files (config.yaml) for each
    hyperparameter combination. Prepares training/validation/test data centrally.
    Creates a mapping file and a single SLURM job array script for efficient submission.

    Args:
        db_manager: DatabaseManager instance.
        model_type: The type of model ('mace' or 'allegro').
        base_sweep_dir: The root directory where sweep results will be stored.
        sweep_params: Dictionary where keys are parameter names and values are
                      lists of hyperparameter values to sweep over.
        fixed_params: Dictionary of parameters that remain constant for all runs.
                      Must include necessary parameters for the respective
                      prepare_job function (e.g., gpu_config, num_structures or
                      structure_ids). Ratio parameters are used only if k_folds=None/1.
        num_seeds: Number of times to repeat each HPO/fold combination with
                   different training seeds.
        k_folds: If greater than 1, perform k-fold cross-validation. If None or 1,
                 perform a standard train/val/test split based on ratios in
                 fixed_params.
        test_ratio: Fraction of data to hold out as a final test set when
                    k_folds > 1. Defaults to 0.1. Also used as the test set
                    fraction when k_folds is None/1.
        master_seed: The main random seed for shuffling and initial data splits.
    """
    base_sweep_dir = Path(base_sweep_dir).resolve() # Use absolute path
    base_sweep_dir.mkdir(parents=True, exist_ok=True)
    central_data_dir = base_sweep_dir / "data"
    central_data_dir.mkdir(exist_ok=True)
    random.seed(master_seed) # For initial shuffling

    logger.info(f"Starting HPO sweep for {model_type} in {base_sweep_dir} (Job Array Mode)")
    logger.info(f"Sweep parameters: {json.dumps(sweep_params, indent=2)}")
    # Use default=str for things like Path objects if present
    logger.info(f"Fixed parameters: {json.dumps(fixed_params, indent=2, default=str)}")
    logger.info(f"Repetitions per setting: {num_seeds}")
    logger.info(f"K-Fold Cross-Validation: {'Enabled (k=' + str(k_folds) + ')' if k_folds and k_folds > 1 else 'Disabled'}")

    # --- 1. Validate Inputs ---
    if model_type not in ['mace', 'allegro']:
        raise ValueError("model_type must be 'mace' or 'allegro'")
    if k_folds is not None and k_folds <= 1:
        k_folds = None # Treat k=1 as no k-fold
    if k_folds and (test_ratio is None or not (0 < test_ratio < 1)):
         raise ValueError("test_ratio must be between 0 and 1 when k_folds > 1")
    # If not k-fold, we need train/val/test ratios OR rely on defaults within prepare_job (less ideal now)
    if not k_folds:
        # Standard split uses test_ratio directly now, need train/val too
        if 'train_ratio' not in fixed_params or 'val_ratio' not in fixed_params:
             raise ValueError("When k_folds is not used, 'train_ratio' and 'val_ratio' must be specified in fixed_params.")
        train_ratio = fixed_params['train_ratio']
        val_ratio = fixed_params['val_ratio']
        std_test_ratio = test_ratio if test_ratio is not None else fixed_params.get('test_ratio', None) # Allow test_ratio override
        if std_test_ratio is None:
             raise ValueError("When k_folds is not used, 'test_ratio' must be specified in fixed_params or as argument.")
        if not math.isclose(train_ratio + val_ratio + std_test_ratio, 1.0):
            logger.warning(f"Provided train/val/test ratios ({train_ratio}, {val_ratio}, {std_test_ratio}) do not sum to 1. Normalizing.")
            total_ratio = train_ratio + val_ratio + std_test_ratio
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            std_test_ratio /= total_ratio
            logger.info(f"Normalized ratios: train={train_ratio:.3f}, val={val_ratio:.3f}, test={std_test_ratio:.3f}")
        # Store the potentially normalized ratios back for splitting logic
        effective_ratios = {'train': train_ratio, 'val': val_ratio, 'test': std_test_ratio}

    if 'structure_ids' not in fixed_params and 'num_structures' not in fixed_params:
         raise ValueError("fixed_params must include either 'structure_ids' or 'num_structures'")
    if 'structure_ids' in fixed_params and 'num_structures' in fixed_params:
        logger.warning("Both 'structure_ids' and 'num_structures' provided in fixed_params. 'structure_ids' will be used.")
        if 'num_structures' in fixed_params: del fixed_params['num_structures'] # Prioritize structure_ids


    # --- 2. Get and Prepare Structure IDs ---
    structure_ids: List[int]
    if 'structure_ids' in fixed_params:
        structure_ids = fixed_params['structure_ids']
        logger.info(f"Using {len(structure_ids)} provided structure IDs.")
    else:
        num_structures = fixed_params['num_structures']
        logger.info(f"Fetching up to {num_structures} structures from the database...")
        try:
            # Example: Get all structure IDs with VASP calculations
            all_db_ids = _get_vasp_structures(db_manager)
        except Exception as e:
            logger.error(f"Failed to fetch initial structure IDs from database: {e}", exc_info=True)
            raise

        if len(all_db_ids) < num_structures:
             logger.warning(f"Requested {num_structures} structures, but only {len(all_db_ids)} relevant structures found in DB. Using all available.")
             structure_ids = all_db_ids
        else:
             structure_ids = random.sample(all_db_ids, num_structures)
        logger.info(f"Selected {len(structure_ids)} structures for the sweep.")

    # Shuffle the final list of IDs once
    random.shuffle(structure_ids)
    total_structures = len(structure_ids)
    if total_structures == 0:
        raise ValueError("No structures selected for the sweep. Cannot proceed.")


    # --- 3. Prepare Central Data Splits ---
    # Dictionary to store absolute paths to data files for each split context
    # Key: fold index (int, -1 for standard split)
    # Value: Dict {'train': Path, 'val': Path, 'test': Path}
    central_data_paths: Dict[int, Dict[str, Path]] = {}

    if k_folds:
        logger.info(f"Preparing central data for {k_folds}-fold split (test_ratio={test_ratio})")
        test_size = math.ceil(total_structures * test_ratio)
        if test_size == 0 and total_structures > 0:
             logger.warning("Holdout test set size calculated as 0.")
        if test_size >= total_structures:
             raise ValueError(f"test_ratio ({test_ratio}) is too large, no structures left for training/validation.")

        test_ids = structure_ids[:test_size]
        train_val_ids = structure_ids[test_size:]
        num_train_val = len(train_val_ids)
        logger.info(f"Split: {len(test_ids)} holdout test, {num_train_val} train+validation")

        if num_train_val < k_folds:
            raise ValueError(f"Not enough structures ({num_train_val}) for {k_folds} folds after removing test set.")

        # Determine fold sizes
        fold_size = num_train_val // k_folds
        extra = num_train_val % k_folds
        folds_ids: List[List[int]] = []
        start_idx = 0
        for i in range(k_folds):
            end_idx = start_idx + fold_size + (1 if i < extra else 0)
            folds_ids.append(train_val_ids[start_idx:end_idx])
            start_idx = end_idx
        logger.info(f"Created {k_folds} folds with sizes: {[len(f) for f in folds_ids]}")

        # --- Save data centrally for each fold ---
        for fold_idx in range(k_folds):
            fold_data_dir = central_data_dir / f"fold_{fold_idx}"
            fold_data_dir.mkdir(parents=True, exist_ok=True)

            # Define target paths in the central directory
            central_train_file = fold_data_dir / "train.xyz"
            central_val_file = fold_data_dir / "val.xyz"
            central_test_file = fold_data_dir / "test.xyz" # Holdout test set saved per fold dir
            central_splits_json = fold_data_dir / "structure_splits.json" # Record IDs saved here

            # Store absolute paths for later use
            central_data_paths[fold_idx] = {
                "train": central_train_file.resolve(),
                "val": central_val_file.resolve(),
                "test": central_test_file.resolve(),
            }

            # Check if ALL files already exist for this fold
            if (central_train_file.exists() and
                central_val_file.exists() and
                (central_test_file.exists() or len(test_ids) == 0) and
                central_splits_json.exists()):
                logger.info(f"Central data for fold {fold_idx} already exists in {fold_data_dir}. Skipping creation.")
                # Ensure paths are still stored if skipping
                if fold_idx not in central_data_paths:
                     central_data_paths[fold_idx] = {
                        "train": central_train_file.resolve(),
                        "val": central_val_file.resolve(),
                        "test": central_test_file.resolve(),
                    }
                continue # Skip saving if already done

            logger.info(f"Creating central data for fold {fold_idx} in {fold_data_dir}...")

            # Prepare train/val IDs for this fold
            val_ids_for_fold = folds_ids[fold_idx]
            train_ids_for_fold = list(itertools.chain.from_iterable(
                folds_ids[j] for j in range(k_folds) if j != fold_idx
            ))

            # Save structures to central directory
            logger.info(f"  Saving {len(train_ids_for_fold)} train structures to {central_train_file}")
            saved_train_ids = _save_structures_to_xyz(db_manager, train_ids_for_fold, central_train_file)
            if model_type == 'mace': _replace_properties(central_train_file)

            logger.info(f"  Saving {len(val_ids_for_fold)} validation structures to {central_val_file}")
            saved_val_ids = _save_structures_to_xyz(db_manager, val_ids_for_fold, central_val_file)
            if model_type == 'mace': _replace_properties(central_val_file)

            saved_test_ids = []
            if test_ids: # Only save if test set exists
                logger.info(f"  Saving {len(test_ids)} test structures to {central_test_file}")
                saved_test_ids = _save_structures_to_xyz(db_manager, test_ids, central_test_file)
                if model_type == 'mace': _replace_properties(central_test_file)
            else:
                logger.info("  No holdout test structures to save.")

            # Save the IDs actually written to this central directory's json
            split_id_info = {'train': saved_train_ids, 'val': saved_val_ids, 'test': saved_test_ids}
            try:
                with open(central_splits_json, 'w') as f:
                    json.dump(split_id_info, f, indent=2)
            except Exception as e:
                 logger.error(f"Failed to save central structure splits JSON for fold {fold_idx}: {e}", exc_info=True)
                 # Decide whether to raise or just warn? Warn for now.

    else:
        # Standard split: Save data once centrally
        logger.info(f"Preparing central data for standard train/val/test split.")
        split_data_dir = central_data_dir / "all"
        split_data_dir.mkdir(parents=True, exist_ok=True)

        central_train_file = split_data_dir / "train.xyz"
        central_val_file = split_data_dir / "val.xyz"
        central_test_file = split_data_dir / "test.xyz"
        central_splits_json = split_data_dir / "structure_splits.json"

        # Store absolute paths for later use (using fold index -1)
        central_data_paths[-1] = {
            "train": central_train_file.resolve(),
            "val": central_val_file.resolve(),
            "test": central_test_file.resolve(),
        }

        # Check if ALL files already exist
        if (central_train_file.exists() and
            central_val_file.exists() and
            central_test_file.exists() and
            central_splits_json.exists()):
            logger.info(f"Central data for standard split already exists in {split_data_dir}. Skipping creation.")
            # Ensure paths are still stored if skipping
            if -1 not in central_data_paths:
                central_data_paths[-1] = {
                    "train": central_train_file.resolve(),
                    "val": central_val_file.resolve(),
                    "test": central_test_file.resolve(),
                }
        else:
            logger.info(f"Creating central data for standard split in {split_data_dir}...")

            # Calculate split sizes based on effective ratios
            tr = effective_ratios['train']
            vr = effective_ratios['val']
            ter = effective_ratios['test']
            train_size = math.floor(total_structures * tr)
            val_size = math.floor(total_structures * vr)
            test_size = total_structures - train_size - val_size # Remainder is test

            logger.info(f"Splitting {total_structures} structures into: {train_size} train, {val_size} val, {test_size} test (using master_seed {master_seed})")

            # IDs are already shuffled from step 2
            train_ids = structure_ids[:train_size]
            val_ids = structure_ids[train_size : train_size + val_size]
            test_ids = structure_ids[train_size + val_size :]

            # Save structures
            logger.info(f"  Saving {len(train_ids)} train structures to {central_train_file}")
            saved_train_ids = _save_structures_to_xyz(db_manager, train_ids, central_train_file)
            if model_type == 'mace': _replace_properties(central_train_file)

            logger.info(f"  Saving {len(val_ids)} validation structures to {central_val_file}")
            saved_val_ids = _save_structures_to_xyz(db_manager, val_ids, central_val_file)
            if model_type == 'mace': _replace_properties(central_val_file)

            logger.info(f"  Saving {len(test_ids)} test structures to {central_test_file}")
            saved_test_ids = _save_structures_to_xyz(db_manager, test_ids, central_test_file)
            if model_type == 'mace': _replace_properties(central_test_file)

            # Save the IDs actually written
            split_id_info = {'train': saved_train_ids, 'val': saved_val_ids, 'test': saved_test_ids}
            try:
                with open(central_splits_json, 'w') as f:
                    json.dump(split_id_info, f, indent=2)
            except Exception as e:
                 logger.error(f"Failed to save central structure splits JSON for standard split: {e}", exc_info=True)


    # --- 4. Generate Hyperparameter Combinations ---
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    hpo_combinations = list(itertools.product(*param_values))
    logger.info(f"Generated {len(hpo_combinations)} hyperparameter combinations.")

    # --- 5. Iterate and Prepare Job Configs ---
    job_details_list: List[Dict[str, Any]] = [] # Store details for mapping file
    task_id_counter = 0
    # Removed: first_run_command = None

    # --- Dynamically import prepare_job function ---
    prepare_job_func = None
    if model_type == 'mace':
        try:
            from forge.workflows.db_to_mace import prepare_mace_job
            prepare_job_func = prepare_mace_job
            logger.info("Successfully imported prepare_mace_job.")
        except ImportError as e:
            logger.error(f"Failed to import prepare_mace_job: {e}", exc_info=True)
            raise ImportError("Could not import prepare_mace_job. Ensure it exists and dependencies are met.") from e
    elif model_type == 'allegro':
        try:
            from forge.workflows.db_to_allegro import prepare_allegro_job
            prepare_job_func = prepare_allegro_job
            logger.info("Successfully imported prepare_allegro_job.")
        except ImportError as e:
            logger.error(f"Failed to import prepare_allegro_job: {e}", exc_info=True)
            raise ImportError("Could not import prepare_allegro_job. Ensure it exists and dependencies are met.") from e


    for combo_idx, combo_values in enumerate(hpo_combinations):
        current_hpo_params = dict(zip(param_names, combo_values))

        # Create unique directory name for HPO combo
        hpo_combo_str_parts = []
        for k, v in sorted(current_hpo_params.items()):
            v_str = f"{v:.3g}" if isinstance(v, float) else str(v) # Compact float format
            hpo_combo_str_parts.append(f"{k}-{v_str}")
        hpo_combo_str = "_".join(hpo_combo_str_parts)
        max_len = 100
        if len(hpo_combo_str) > max_len:
            hpo_combo_hash = hashlib.md5(hpo_combo_str.encode()).hexdigest()[:8]
            hpo_dir_name = f"hpo_{combo_idx:03d}_{hpo_combo_hash}"
        else:
            hpo_dir_name = f"hpo_{combo_idx:03d}_{hpo_combo_str}"
        # Sanitize directory name further if needed (replace invalid chars)
        hpo_dir_name = hpo_dir_name.replace(" ", "_").replace("/", "_").replace("\\", "_")


        hpo_combo_dir = base_sweep_dir / hpo_dir_name
        # No need to create here, prepare_job should handle run_dir creation

        logger.info(f"Processing HPO Combination {combo_idx + 1}/{len(hpo_combinations)} ({hpo_dir_name})")

        num_loops = k_folds if k_folds else 1
        for loop_idx in range(num_loops): # Loop over folds or just once
            fold_idx = loop_idx if k_folds is not None else -1 # -1 indicates standard split

            # Get the central data paths for this fold/split context
            current_data_paths = central_data_paths.get(fold_idx)
            if not current_data_paths:
                logger.error(f"Missing central data paths for fold index {fold_idx}. Skipping combinations for this fold.")
                continue # Skip HPO combos for this fold if data is missing

            for seed_idx in range(num_seeds):
                run_seed = master_seed + loop_idx * num_seeds + seed_idx # Unique seed
                fold_str = f"_fold{fold_idx}" if k_folds is not None else "_all"
                seed_str = f"_seed{seed_idx}"
                run_name = f"{hpo_dir_name}{fold_str}{seed_str}"
                # Specific directory for this unique run (hpo_fold_seed)
                run_dir = hpo_combo_dir / f"run{fold_str}{seed_str}"
                # No need to create run_dir here, prepare_job should do it

                logger.info(f"  Preparing Task {task_id_counter} (Run: {run_name}) in {run_dir}")

                # Prepare parameters for the job function
                run_params_specific = fixed_params.copy() # Start with fixed params
                run_params_specific.update(current_hpo_params) # Add sweep params
                run_params_specific['seed'] = run_seed # Add the specific run seed

                # Add absolute paths to central data files
                run_params_specific['data_train_path'] = str(current_data_paths['train'])
                run_params_specific['data_val_path'] = str(current_data_paths['val'])
                run_params_specific['data_test_path'] = str(current_data_paths['test'])

                # Remove parameters no longer needed by prepare_job
                run_params_specific.pop('structure_ids', None)
                run_params_specific.pop('num_structures', None)
                run_params_specific.pop('train_ratio', None)
                run_params_specific.pop('val_ratio', None)
                run_params_specific.pop('test_ratio', None)
                run_params_specific.pop('external_data_source_dir', None) # If it was ever present
                run_params_specific.pop('slurm_job_name', None) # Not needed by prepare_job
                run_params_specific.pop('slurm_partition', None)
                run_params_specific.pop('slurm_time', None)
                run_params_specific.pop('slurm_mem', None)
                run_params_specific.pop('slurm_cpus_per_gpu', None)
                run_params_specific.pop('slurm_output', None)
                run_params_specific.pop('slurm_error', None)
                run_params_specific.pop('gpu_config', None)
                # Need to keep gpu_config for resource allocation in SLURM script

                # --- Call prepare_job function ---
                # We expect this to create run_dir and config.yaml inside it
                try:
                    # Prepare the job using the dynamically imported function
                    prepare_job_func(
                        db_manager=db_manager, # Pass db_manager if needed (e.g., for symbols in Allegro)
                        job_name=run_name,     # Unique name for this run (hpo_fold_seed)
                        job_dir=run_dir,       # Directory for this specific run's config
                        **run_params_specific  # Pass all other relevant params
                    )
                    # --- Job prepared successfully ---

                    # Add job details to the list for the mapping file
                    job_details_list.append({
                        "task_id": task_id_counter,
                        "job_dir": str(run_dir.resolve()), # Absolute path to the run dir
                        "run_name": run_name,
                        "hpo_params": current_hpo_params,
                        "fold": fold_idx,
                        "seed": run_seed,
                    })
                    task_id_counter += 1

                except Exception as e:
                     logger.error(f"Error preparing job config for Task {task_id_counter} ({run_name}): {e}", exc_info=True)
                     # Continue to next job config generation

    total_jobs_prepared = len(job_details_list)
    logger.info(f"Finished preparing configurations for {total_jobs_prepared} jobs.")

    if total_jobs_prepared == 0:
        logger.error("No job configurations were successfully prepared. Exiting.")
        return # Exit if no jobs

    # --- 6. Generate Mapping File ---
    mapping_file_path = base_sweep_dir / "job_array_mapping.json"
    try:
        with open(mapping_file_path, "w") as f:
            json.dump(job_details_list, f, indent=2)
        logger.info(f"Generated job mapping file: {mapping_file_path}")
    except Exception as e:
        logger.error(f"Failed to generate mapping file: {e}", exc_info=True)
        raise # Stop if mapping file fails


    # --- 7. Generate Master Job Array Submission Script ---
    submit_script_path = base_sweep_dir / "submit_array_job.sh"
    try:
        # --- Extract SLURM resource requests from fixed_params ---
        # Use get() with defaults based on user feedback
        gpu_config = fixed_params.get('gpu_config', {"count": 1, "type": None}) # Default 1 GPU
        num_gpus = gpu_config.get('count', 1)
        gpu_type = gpu_config.get('type', None) # e.g., 'rtx6000', 'a100'
        gpu_constraint_str = f"\n#SBATCH --constraint={gpu_type}" if gpu_type else ""
        gpu_gres_str = f"\n#SBATCH --gres=gpu:{num_gpus}" if num_gpus > 0 else ""

        # Default SLURM params (modify as needed)
        slurm_job_name = fixed_params.get('slurm_job_name', f"{model_type}_hpo_sweep")
        slurm_partition = fixed_params.get('slurm_partition', 'regular') # Use 'regular' from template
        slurm_time = fixed_params.get('slurm_time', '5-06:00:00') # Use 5 days 6 hours from input
        slurm_mem_per_gpu = fixed_params.get('slurm_mem_per_gpu', '32G') # Example, adjust as needed
        slurm_mem = fixed_params.get('slurm_mem', f"{int(slurm_mem_per_gpu[:-1])*num_gpus}G" if num_gpus > 0 else '32G') # Calculate based on GPUs or default
        slurm_cpus = fixed_params.get('slurm_cpus_per_task', 8) # Use 8 from input
        slurm_nodes = fixed_params.get('num_nodes', 1) # Use num_nodes from fixed_params if exists
        slurm_output_dir = base_sweep_dir / 'slurm_out'
        slurm_error_dir = base_sweep_dir / 'slurm_err'
        slurm_output = str(slurm_output_dir / '%A_%a.out') # Array output files in sweep dir
        slurm_error = str(slurm_error_dir / '%A_%a.err')   # Array error files in sweep dir

        # Determine the training command based on model type
        if model_type == 'mace':
            training_command = 'srun mace_run_train --config="config.yaml"'
        elif model_type == 'allegro':
            # Use srun based on template provided
            training_command = 'srun nequip-train -cn config.yaml'
        else:
            training_command = "# ERROR: Unknown model_type. Please specify training command."


        script_content = f"""\
#!/bin/bash
# SLURM Job Array Submission Script
# Generated for sweep in: {base_sweep_dir}

# --- SLURM Directives ---
#SBATCH --job-name={slurm_job_name}
#SBATCH --partition={slurm_partition}
#SBATCH --time={slurm_time}
#SBATCH --nodes={slurm_nodes}
#SBATCH --ntasks-per-node=1 # Typically 1 task, handling parallelism internally (e.g., torchrun, srun) or via multi-CPU request
#SBATCH --cpus-per-task={slurm_cpus}{gpu_gres_str}{gpu_constraint_str}
#SBATCH --mem={slurm_mem}
#SBATCH --output={slurm_output}
#SBATCH --error={slurm_error}
# The actual array directive. Jobs are indexed from 0 to N-1.
#SBATCH --array=0-{total_jobs_prepared - 1}

# --- Environment Setup (Modify as needed) ---
echo "Setting up environment..."
# Example: Load Conda
# IMPORTANT: Source the correct conda setup script for your system.
#            Common locations: ~/anaconda3/etc/profile.d/conda.sh
#                           ~/miniconda3/etc/profile.d/conda.sh
#                           /path/to/shared/conda/etc/profile.d/conda.sh
#            Using the specific path from user templates:
source /home/myless/.mambaforge/etc/profile.d/conda.sh || {{ echo "Error: Failed to source conda."; exit 1; }}

# Example: Activate Conda environment
# IMPORTANT: Replace 'your_conda_env_name' with the actual environment name.
#            Using names from templates:
CONDA_ENV_NAME="{ 'mace-cueq' if model_type == 'mace' else 'allegro-new' }"
conda activate "$CONDA_ENV_NAME" || {{ echo "Error: Failed to activate conda env '$CONDA_ENV_NAME'"; exit 1; }}
echo "Conda environment '$CONDA_ENV_NAME' activated."

# Optional: Set OMP_NUM_THREADS (adjust based on cpus-per-task and library needs)
# export OMP_NUM_THREADS={slurm_cpus} # Set based on allocated CPUs
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / {num_gpus if num_gpus > 0 else 1})) # Match template logic more closely? Or just set to cpus-per-task? Let's match cpus-per-task for now.
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

# --- Job Logic ---
echo "Starting job logic for Task ID: $SLURM_ARRAY_TASK_ID"
MAPPING_FILE="{mapping_file_path}"
TASK_ID=${{SLURM_ARRAY_TASK_ID}}

# Create output/error directories if they don't exist
echo "Ensuring SLURM output/error directories exist..."
mkdir -p "{slurm_output_dir}"
mkdir -p "{slurm_error_dir}"

# Extract job directory using jq (ensure jq is available on cluster nodes)
# Alternatively, use a small Python script here if jq is not reliable.
echo "Extracting job directory from mapping file..."
JOB_DIR=$(jq -r --argjson tid "$TASK_ID" '.[] | select(.task_id == $tid) | .job_dir' "$MAPPING_FILE")

# Basic error checking for jq command and JOB_DIR extraction
if [ $? -ne 0 ] || [ -z "$JOB_DIR" ]; then
  echo "Error: Failed to extract job directory for task ID $TASK_ID using jq from $MAPPING_FILE" >&2
  # Fallback attempt using python (requires python in the environment)
  PYTHON_CMD="import sys, json; mapping=json.load(sys.stdin); print(next((item['job_dir'] for item in mapping if item['task_id'] == int(sys.argv[1])), ''))"
  JOB_DIR=$(python -c "$PYTHON_CMD" "$TASK_ID" < "$MAPPING_FILE")
  if [ -z "$JOB_DIR" ]; then
      echo "Error: Python fallback also failed to extract job directory." >&2
      exit 1
  fi
  echo "Successfully extracted job directory using Python fallback."
fi

echo "Job Directory: $JOB_DIR"

# Navigate to the job directory
echo "Changing to job directory..."
cd "$JOB_DIR" || {{ echo "Error: Failed to cd into $JOB_DIR"; exit 1; }}
echo "Current directory: $(pwd)"

# --- Execute Training ---
echo "Executing training command..."
echo "Command: {training_command}"

# Execute the command
{training_command}

# Check the exit code of the training command
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Training command failed with exit code $EXIT_CODE" >&2
    exit $EXIT_CODE
fi

echo "Job completed successfully."
exit 0
"""

        with open(submit_script_path, "w") as f:
            f.write(script_content)

        submit_script_path.chmod(0o755)
        logger.info(f"Generated job array submission script: {submit_script_path}")
        logger.info(f"----> IMPORTANT: Review '{submit_script_path.name}' to verify environment setup (conda path/env name) and resource requests before submitting with 'sbatch {submit_script_path.name}'. <----")


    except Exception as e:
        logger.error(f"Failed to generate job array submission script: {e}", exc_info=True)

    logger.info("Sweep preparation complete.")

# Example usage structure (adapt your script to use this)
# if __name__ == "__main__":
#     # 1. Define db_manager, sweep_output_dir, sweep_parameters, fixed_parameters, etc.
#     # ... (like in your provided script)
#
#     # 2. Initialize DatabaseManager
#     db_manager = DatabaseManager(...)
#
#     # 3. Call run_hpo_sweep
#     try:
#         run_hpo_sweep(
#             db_manager=db_manager,
#             model_type='allegro', # or 'mace'
#             base_sweep_dir=sweep_output_dir,
#             sweep_params=sweep_parameters,
#             fixed_params=fixed_parameters,
#             num_seeds=num_repetitions,
#             k_folds=num_k_folds,
#             test_ratio=holdout_test_ratio,
#             master_seed=main_seed
#         )
#     except Exception as e:
#         logger.error(f"Sweep failed: {e}", exc_info=True)
#     finally:
#         db_manager.close_connection()

