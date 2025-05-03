import os
import json
import random
import math
import itertools
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import subprocess # Needed to attempt getting the train command

from forge.core.database import DatabaseManager
# Assuming these functions primarily generate config files and data links/copies
# and we can potentially ignore the .sh they might create.
from forge.workflows.db_to_mace import prepare_mace_job, _get_vasp_structures, _save_structures_to_xyz, _replace_properties, GPUConfig
from forge.workflows.db_to_allegro import prepare_allegro_job

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper function to guess training command (modify as needed) ---
def _get_training_command(model_type: str, run_dir: Path) -> Optional[str]:
    """Attempts to determine the training command based on generated files."""
    run_dir = Path(run_dir)
    if model_type == 'mace':
        # Look for the MACE config file (adjust name if necessary)
        config_files = list(run_dir.glob('*_config.yaml')) # Or params.json?
        if config_files:
            # Example command - ** ADJUST BASED ON YOUR MACE TRAINING SCRIPT **
            return f"python /path/to/your/mace/train.py --config {config_files[0].name}"
        # Fallback: Look for a standard script name if config not found
        elif (run_dir / "run_mace_train.py").exists():
             return "python run_mace_train.py" # Assumes it reads local config
        else:
             logger.warning(f"Could not determine MACE training command for {run_dir}")
             return None
    elif model_type == 'allegro':
        # Look for Allegro config file (adjust name if necessary)
        config_files = list(run_dir.glob('config.yaml')) # Common name
        if config_files:
             # Example command - ** ADJUST BASED ON YOUR ALLEGRO TRAINING SCRIPT **
             # Using torchrun is common for Allegro
             num_gpus = 1 # TODO: Get this from fixed_params['gpu_config']['count'] if possible
             return f"torchrun --standalone --nproc_per_node={num_gpus} /path/to/your/allegro/train.py --config-file {config_files[0].name}"
        else:
            logger.warning(f"Could not determine Allegro training command for {run_dir}")
            return None
    else:
        return None

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

    Generates job directories and configuration files for each hyperparameter
    combination. Creates a mapping file and a single SLURM job array script
    for efficient submission.

    Args:
        db_manager: DatabaseManager instance.
        model_type: The type of model ('mace' or 'allegro').
        base_sweep_dir: The root directory where sweep results will be stored.
        sweep_params: Dictionary where keys are parameter names and values are
                      lists of hyperparameter values to sweep over.
        fixed_params: Dictionary of parameters that remain constant for all runs.
                      Must include necessary parameters for the respective
                      prepare_job function (e.g., gpu_config, num_structures or
                      structure_ids, splitting ratios if not doing k-fold).
        num_seeds: Number of times to repeat each HPO/fold combination with
                   different training seeds.
        k_folds: If greater than 1, perform k-fold cross-validation. If None or 1,
                 perform a standard train/val/test split based on ratios in
                 fixed_params.
        test_ratio: Fraction of data to hold out as a final test set when
                    k_folds > 1. Defaults to 0.1.
        master_seed: The main random seed for shuffling and initial data splits.
    """
    base_sweep_dir = Path(base_sweep_dir)
    base_sweep_dir.mkdir(parents=True, exist_ok=True)
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
    if not k_folds and 'train_ratio' not in fixed_params and 'val_ratio' not in fixed_params and 'test_ratio' not in fixed_params:
         # Allow skipping if prepare_job handles defaults
         logger.warning("Train/val/test ratios not specified in fixed_params and k-fold is disabled. Assuming prepare_job handles defaults.")
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
        # This might need adjustment based on _get_vasp_structures implementation
        # It should return IDs that have the necessary calculation data.
        # For simplicity, assume it returns relevant structure IDs directly.
        try:
            # Example: Get all structure IDs if _get_vasp_structures isn't suitable
            with db_manager.conn.cursor() as cur:
                cur.execute("SELECT structure_id FROM structures ORDER BY structure_id")
                all_db_ids = [row[0] for row in cur.fetchall()]
            # all_db_ids = _get_vasp_structures(db_manager) # Use if appropriate
        except Exception as e:
            logger.error(f"Failed to fetch initial structure IDs from database: {e}", exc_info=True)
            raise

        if len(all_db_ids) < num_structures:
             logger.warning(f"Requested {num_structures} structures, but only {len(all_db_ids)} found in DB. Using all available.")
             structure_ids = all_db_ids
        else:
             structure_ids = random.sample(all_db_ids, num_structures)
        logger.info(f"Selected {len(structure_ids)} structures for the sweep.")

    # Shuffle the final list of IDs once
    random.shuffle(structure_ids)
    total_structures = len(structure_ids)
    if total_structures == 0:
        raise ValueError("No structures selected for the sweep. Cannot proceed.")


    # --- 3. Define Data Splits (and save if k-fold) ---
    test_ids: List[int] = []
    train_val_ids: List[int] = structure_ids # Default if not k-fold
    folds_ids: List[List[int]] = [] # List of lists of IDs, one list per fold
    shared_data_dirs: Dict[int, Path] = {} # Map fold_idx to shared data dir path

    if k_folds:
        logger.info(f"Performing {k_folds}-fold split with test_ratio={test_ratio}")
        test_size = math.ceil(total_structures * test_ratio)
        if test_size == 0 and total_structures > 0:
             logger.warning("Test set size calculated as 0. No holdout test set will be created.")
        # Allow test_size == total_structures? Should be prevented by test_ratio<1 validation
        if test_size >= total_structures:
             raise ValueError(f"test_ratio ({test_ratio}) is too large, no structures left for training/validation.")

        test_ids = structure_ids[:test_size]
        train_val_ids = structure_ids[test_size:]
        num_train_val = len(train_val_ids)
        logger.info(f"Split: {len(test_ids)} test, {num_train_val} train+validation")

        if num_train_val < k_folds:
            raise ValueError(f"Not enough structures ({num_train_val}) for {k_folds} folds after removing test set.")

        fold_size = num_train_val // k_folds
        extra = num_train_val % k_folds

        start_idx = 0
        for i in range(k_folds):
            end_idx = start_idx + fold_size + (1 if i < extra else 0)
            folds_ids.append(train_val_ids[start_idx:end_idx])
            start_idx = end_idx
        logger.info(f"Created {k_folds} folds with sizes: {[len(f) for f in folds_ids]}")

        # --- Save data centrally for each fold ---
        for fold_idx in range(k_folds):
            # Use a consistent naming scheme, incorporating model type maybe?
            shared_fold_data_dir = base_sweep_dir / f"_shared_data_{model_type}_fold_{fold_idx}"
            shared_data_dirs[fold_idx] = shared_fold_data_dir

            # Define target paths in the shared directory
            shared_train_file = shared_fold_data_dir / "train.xyz"
            shared_val_file = shared_fold_data_dir / "val.xyz"
            shared_test_file = shared_fold_data_dir / "test.xyz" # Holdout test set
            shared_splits_json = shared_fold_data_dir / "structure_splits.json"

            # Check if data already exists for this fold
            # Be more robust: check if ALL files exist
            if (shared_train_file.exists() and
                shared_val_file.exists() and
                (shared_test_file.exists() or len(test_ids) == 0) and # Only check test if needed
                shared_splits_json.exists()):
                logger.info(f"Shared data for fold {fold_idx} already exists in {shared_fold_data_dir}. Skipping creation.")
                continue # Skip saving if already done

            logger.info(f"Creating shared data for fold {fold_idx} in {shared_fold_data_dir}...")
            shared_fold_data_dir.mkdir(parents=True, exist_ok=True)

            # Prepare train/val IDs for this fold
            val_ids_for_fold = folds_ids[fold_idx]
            train_ids_for_fold = list(itertools.chain.from_iterable(
                folds_ids[j] for j in range(k_folds) if j != fold_idx
            ))

            # Save structures to shared directory
            logger.info(f"  Saving {len(train_ids_for_fold)} train structures to {shared_train_file}")
            saved_train = _save_structures_to_xyz(db_manager, train_ids_for_fold, shared_train_file)
            if model_type == 'mace': _replace_properties(shared_train_file) # Apply MACE specific property changes if needed

            logger.info(f"  Saving {len(val_ids_for_fold)} validation structures to {shared_val_file}")
            saved_val = _save_structures_to_xyz(db_manager, val_ids_for_fold, shared_val_file)
            if model_type == 'mace': _replace_properties(shared_val_file)

            saved_test = []
            if test_ids: # Only save if test set exists
                logger.info(f"  Saving {len(test_ids)} test structures to {shared_test_file}")
                saved_test = _save_structures_to_xyz(db_manager, test_ids, shared_test_file)
                if model_type == 'mace': _replace_properties(shared_test_file)
            else:
                logger.info("  No holdout test structures to save.")


            # Save the IDs used in this shared directory
            split_id_info = {'train': saved_train, 'val': saved_val, 'test': saved_test}
            try:
                with open(shared_splits_json, 'w') as f:
                    json.dump(split_id_info, f, indent=2)
            except Exception as e:
                 logger.error(f"Failed to save shared structure splits JSON for fold {fold_idx}: {e}", exc_info=True)
                 # Decide whether to raise or just warn

    else:
        # Standard split: Ratios will be used directly by prepare_mace/allegro_job
        logger.info(f"Using standard train/val/test split ratios from fixed_params (if provided).")


    # --- 4. Generate Hyperparameter Combinations ---
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    hpo_combinations = list(itertools.product(*param_values))
    logger.info(f"Generated {len(hpo_combinations)} hyperparameter combinations.")

    # --- 5. Iterate and Prepare Job Configs ---
    job_details_list: List[Dict[str, Any]] = [] # Store details for mapping file
    task_id_counter = 0
    first_run_command = None # Store the command from the first successful job prep

    for combo_idx, combo_values in enumerate(hpo_combinations):
        current_hpo_params = dict(zip(param_names, combo_values))

        # Create a unique identifier for this HPO combo for dir naming
        # Ensure values are suitable for file paths (e.g., convert floats)
        hpo_combo_str_parts = []
        for k, v in sorted(current_hpo_params.items()):
            v_str = f"{v:.2e}" if isinstance(v, float) else str(v) # Format float
            hpo_combo_str_parts.append(f"{k}-{v_str}")
        hpo_combo_str = "_".join(hpo_combo_str_parts)

        # Limit length and create hash if too long
        max_len = 100 # Max length for the string part of the dir name
        if len(hpo_combo_str) > max_len:
            hpo_combo_hash = hashlib.md5(hpo_combo_str.encode()).hexdigest()[:8]
            hpo_dir_name = f"hpo_{combo_idx:03d}_{hpo_combo_hash}"
        else:
            hpo_dir_name = f"hpo_{combo_idx:03d}_{hpo_combo_str}"

        hpo_combo_dir = base_sweep_dir / hpo_dir_name
        hpo_combo_dir.mkdir(exist_ok=True)

        # Save the params for this combo
        with open(hpo_combo_dir / "hpo_params.json", "w") as f:
            json.dump(current_hpo_params, f, indent=2)

        logger.info(f"Processing HPO Combination {combo_idx + 1}/{len(hpo_combinations)} ({hpo_dir_name})")
        # logger.debug(f"Params: {current_hpo_params}") # Use debug level

        run_params_base = fixed_params.copy()
        run_params_base.update(current_hpo_params)

        num_loops = k_folds if k_folds else 1
        for loop_idx in range(num_loops): # Loop over folds or just once
            fold_idx = loop_idx if k_folds else -1 # -1 indicates no fold

            for seed_idx in range(num_seeds):
                run_seed = master_seed + loop_idx * num_seeds + seed_idx # Unique seed
                fold_str = f"_fold{fold_idx}" if k_folds else ""
                run_name = f"{hpo_dir_name}{fold_str}_seed{seed_idx}"
                run_dir = hpo_combo_dir / f"run{fold_str}_seed{seed_idx}"
                run_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"  Preparing Task {task_id_counter} (Run: {run_name})")

                # Prepare parameters for the job function
                run_params_specific = run_params_base.copy()
                run_params_specific['seed'] = run_seed

                if k_folds:
                    # Use shared data dir for this fold
                    shared_data_dir_for_fold = shared_data_dirs.get(fold_idx)
                    if not shared_data_dir_for_fold or not shared_data_dir_for_fold.exists():
                         logger.error(f"Shared data directory for fold {fold_idx} missing. Skipping Task {task_id_counter}.")
                         continue # Skip this specific task
                    run_params_specific['external_data_source_dir'] = str(shared_data_dir_for_fold.resolve())
                    # Remove ratio/ID params if they exist, as external data is used
                    run_params_specific.pop('train_ratio', None)
                    run_params_specific.pop('val_ratio', None)
                    run_params_specific.pop('test_ratio', None)
                    run_params_specific.pop('structure_ids', None)
                    run_params_specific.pop('num_structures', None)
                else:
                    # Standard split: ensure necessary params are present
                     if 'structure_ids' not in run_params_specific and 'num_structures' not in run_params_specific:
                         # Pass the initially selected and shuffled IDs
                         run_params_specific['structure_ids'] = structure_ids
                     if not all(k in run_params_specific for k in ['train_ratio', 'val_ratio', 'test_ratio']):
                          logger.warning(f"Train/val/test ratios not specified for Task {task_id_counter}. Relying on prepare_job defaults.")


                # --- Call prepare_job function ---
                # We assume this creates config files etc. inside run_dir
                # We will ignore any generated .sh file for now.
                try:
                    if model_type == 'mace':
                        prepare_mace_job(
                            db_manager=db_manager,
                            job_name=run_name,
                            job_dir=run_dir,
                            **run_params_specific
                        )
                    elif model_type == 'allegro':
                        # Allegro might need element list from db_manager even with external data?
                        prepare_allegro_job(
                            db_manager=db_manager,
                            job_name=run_name,
                            job_dir=run_dir,
                            **run_params_specific
                        )
                    # --- Job prepared successfully ---

                    # Attempt to determine the command needed to run the job
                    if first_run_command is None:
                        first_run_command = _get_training_command(model_type, run_dir)
                        if first_run_command:
                             logger.info(f"Determined base training command: '{first_run_command}'")
                        else:
                             logger.error("Could not determine training command. Please specify manually in 'submit_array_job.sh'.")
                             # Set a placeholder to avoid repeated warnings
                             first_run_command = "# Error: Could not determine command. Edit this script."


                    # Add job details to the list for the mapping file
                    job_details_list.append({
                        "task_id": task_id_counter,
                        "job_dir": str(run_dir.resolve()),
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
        # Use get() with defaults to avoid errors if keys are missing
        gpu_config = fixed_params.get('gpu_config', {})
        num_gpus = gpu_config.get('count', 1)
        gpu_type = gpu_config.get('type', None) # e.g., 'rtx6000', 'a100'
        gpu_constraint = f":{gpu_type}" if gpu_type else ""

        # Default SLURM params (modify as needed)
        slurm_job_name = fixed_params.get('slurm_job_name', f"{model_type}_hpo_sweep")
        slurm_partition = fixed_params.get('slurm_partition', 'sched_mit_ccrp') # Example partition
        slurm_time = fixed_params.get('slurm_time', '12:00:00') # Example time limit
        slurm_mem = fixed_params.get('slurm_mem', '32G') # Example memory
        slurm_cpus = fixed_params.get('slurm_cpus_per_gpu', 4) * num_gpus # Cpus per task based on GPUs
        slurm_output = fixed_params.get('slurm_output', 'slurm_out/%A_%a.out') # Array output files
        slurm_error = fixed_params.get('slurm_error', 'slurm_err/%A_%a.err')   # Array error files

        training_command = first_run_command if first_run_command else "# TODO: Add training command here (e.g., python train.py --config config.yaml)"

        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# SLURM Job Array Submission Script\n")
            f.write(f"# Generated for sweep in: {base_sweep_dir.resolve()}\n\n")

            # --- SLURM Directives ---
            f.write(f"#SBATCH --job-name={slurm_job_name}\n")
            f.write(f"#SBATCH --partition={slurm_partition}\n")
            f.write(f"#SBATCH --time={slurm_time}\n")
            f.write(f"#SBATCH --nodes=1\n") # Assuming single-node jobs
            f.write(f"#SBATCH --ntasks-per-node=1\n")
            f.write(f"#SBATCH --cpus-per-task={slurm_cpus}\n")
            f.write(f"#SBATCH --mem={slurm_mem}\n")
            if num_gpus > 0:
                 f.write(f"#SBATCH --gres=gpu{gpu_constraint}:{num_gpus}\n")
            # Create output/error directories if they don't exist
            output_dir = Path(slurm_output).parent
            error_dir = Path(slurm_error).parent
            f.write(f"#SBATCH --output={slurm_output}\n")
            f.write(f"#SBATCH --error={slurm_error}\n")
            # The actual array directive
            f.write(f"#SBATCH --array=0-{total_jobs_prepared - 1}\n\n")

             # --- Environment Setup (Modify as needed) ---
            f.write("# --- Environment Setup ---\n")
            f.write("# module load anaconda/2021a # Example module loading\n")
            f.write("# source activate /path/to/your/conda/env # Example conda activation\n\n")


            # --- Job Logic ---
            f.write("# --- Job Logic ---\n")
            f.write(f"MAPPING_FILE=\"{mapping_file_path.resolve()}\"\n")
            f.write("TASK_ID=${SLURM_ARRAY_TASK_ID}\n\n")

            # Create output/error directories before job starts
            f.write(f'mkdir -p {output_dir}\n')
            f.write(f'mkdir -p {error_dir}\n\n')

            # Use jq to parse the JSON mapping file (ensure jq is available on cluster nodes)
            f.write("# Extract job directory using jq (ensure jq is available)\n")
            # This jq command selects the object where task_id == $TASK_ID and then gets the job_dir value
            f.write('JOB_DIR=$(jq -r --argjson tid "$TASK_ID" \'.[] | select(.task_id == $tid) | .job_dir\' "$MAPPING_FILE")\n\n')

            # Basic error checking
            f.write('if [ -z "$JOB_DIR" ]; then\n')
            f.write('  echo "Error: Could not find job directory for task ID $TASK_ID in $MAPPING_FILE" >&2\n')
            f.write('  exit 1\n')
            f.write('fi\n\n')

            f.write('echo "Running job for Task ID: $TASK_ID"\n')
            f.write('echo "Job Directory: $JOB_DIR"\n\n')

            # Navigate to the job directory
            f.write('cd "$JOB_DIR" || { echo "Error: Failed to cd into $JOB_DIR"; exit 1; }\n\n')

            # Execute the training command
            f.write("# --- Execute Training ---\n")
            f.write("# IMPORTANT: Verify this is the correct command to launch training\n")
            f.write(f"{training_command}\n\n")

            f.write("echo \"Job completed successfully.\"\n")
            f.write("exit 0\n")


        submit_script_path.chmod(0o755)
        logger.info(f"Generated job array submission script: {submit_script_path}")
        logger.info(f"----> IMPORTANT: Edit '{submit_script_path.name}' to verify environment setup and the training command: '{training_command}' <----")


    except Exception as e:
        logger.error(f"Failed to generate job array submission script: {e}", exc_info=True)

    logger.info("Sweep preparation complete.")
