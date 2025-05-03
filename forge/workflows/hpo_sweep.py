import os
import json
import random
import math
import itertools
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging # Add logging

from forge.core.database import DatabaseManager
from forge.workflows.db_to_mace import prepare_mace_job, _get_vasp_structures, _save_structures_to_xyz, _replace_properties, GPUConfig
from forge.workflows.db_to_allegro import prepare_allegro_job

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    Generates job directories and scripts for each hyperparameter combination,
    optionally performing k-fold cross-validation and seed repetitions.

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

    logger.info(f"Starting HPO sweep for {model_type} in {base_sweep_dir}")
    logger.info(f"Sweep parameters: {json.dumps(sweep_params, indent=2)}")
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
    if not k_folds and not all(k in fixed_params for k in ['train_ratio', 'val_ratio', 'test_ratio']):
        raise ValueError("fixed_params must include 'train_ratio', 'val_ratio', 'test_ratio' when not using k-fold")
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
        all_db_ids = _get_vasp_structures(db_manager)
        if len(all_db_ids) < num_structures:
             raise ValueError(f"Requested {num_structures} structures, but only {len(all_db_ids)} found with VASP calculations.")
        structure_ids = random.sample(all_db_ids, num_structures)
        logger.info(f"Selected {len(structure_ids)} structures.")

    # Shuffle the final list of IDs once
    random.shuffle(structure_ids)
    total_structures = len(structure_ids)

    # --- 3. Define Data Splits (and save if k-fold) ---
    test_ids: List[int] = []
    train_val_ids: List[int] = structure_ids # Default if not k-fold
    folds_ids: List[List[int]] = [] # List of lists of IDs, one list per fold
    shared_data_dirs: Dict[int, Path] = {} # Map fold_idx to shared data dir path

    if k_folds:
        logger.info(f"Performing {k_folds}-fold split with test_ratio={test_ratio}")
        test_size = math.ceil(total_structures * test_ratio)
        if test_size == 0 and total_structures > 0:
             logger.warning("Test set size is 0. Consider a larger dataset or smaller test_ratio.")
        if test_size >= total_structures:
             raise ValueError("test_ratio is too large, no structures left for training/validation.")

        test_ids = structure_ids[:test_size]
        train_val_ids = structure_ids[test_size:]
        num_train_val = len(train_val_ids)
        logger.info(f"Split: {len(test_ids)} test, {num_train_val} train+validation")

        if num_train_val < k_folds:
            raise ValueError(f"Not enough structures ({num_train_val}) for {k_folds} folds.")

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
            shared_fold_data_dir = base_sweep_dir / f"_shared_data_fold_{fold_idx}"
            shared_data_dirs[fold_idx] = shared_fold_data_dir

            # Define target paths in the shared directory
            # Using generic names here for simplicity
            shared_train_file = shared_fold_data_dir / "train.xyz"
            shared_val_file = shared_fold_data_dir / "val.xyz"
            shared_test_file = shared_fold_data_dir / "test.xyz"
            shared_splits_json = shared_fold_data_dir / "structure_splits.json"

            # Check if data already exists for this fold
            if (shared_train_file.exists() and
                shared_val_file.exists() and
                shared_test_file.exists() and
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
            _replace_properties(shared_train_file)

            logger.info(f"  Saving {len(val_ids_for_fold)} validation structures to {shared_val_file}")
            saved_val = _save_structures_to_xyz(db_manager, val_ids_for_fold, shared_val_file)
            _replace_properties(shared_val_file)

            logger.info(f"  Saving {len(test_ids)} test structures to {shared_test_file}")
            saved_test = _save_structures_to_xyz(db_manager, test_ids, shared_test_file)
            _replace_properties(shared_test_file)

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
        logger.info(f"Using standard train/val/test split ratios from fixed_params.")

    # --- 4. Generate Hyperparameter Combinations ---
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    hpo_combinations = list(itertools.product(*param_values))
    logger.info(f"Generated {len(hpo_combinations)} hyperparameter combinations.")

    # --- 5. Iterate and Prepare Jobs ---
    job_script_paths: List[Path] = [] # Store paths to generated .sh files

    for combo_idx, combo_values in enumerate(hpo_combinations):
        current_hpo_params = dict(zip(param_names, combo_values))

        # Create a unique identifier for this HPO combo for dir naming
        hpo_combo_str = "_".join(f"{k}-{v}" for k, v in sorted(current_hpo_params.items()))
        hpo_combo_hash = hashlib.md5(hpo_combo_str.encode()).hexdigest()[:8] # Short hash
        hpo_dir_name = f"hpo_{combo_idx:03d}_{hpo_combo_hash}"
        hpo_combo_dir = base_sweep_dir / hpo_dir_name
        hpo_combo_dir.mkdir(exist_ok=True)

        # Save the params for this combo
        with open(hpo_combo_dir / "hpo_params.json", "w") as f:
            json.dump(current_hpo_params, f, indent=2)

        logger.info(f"Processing HPO Combination {combo_idx + 1}/{len(hpo_combinations)} ({hpo_dir_name})")
        logger.info(f"Params: {current_hpo_params}")

        run_params_base = fixed_params.copy()
        run_params_base.update(current_hpo_params)

        if k_folds:
            # --- K-Fold Workflow --- 
            for fold_idx in range(k_folds):
                 # Get the path to the shared data for this fold
                shared_data_dir_for_fold = shared_data_dirs.get(fold_idx)
                if not shared_data_dir_for_fold or not shared_data_dir_for_fold.exists():
                     logger.error(f"Shared data directory for fold {fold_idx} not found or not created. Skipping runs for this fold.")
                     continue # Skip to the next fold if shared data is missing

                for seed_idx in range(num_seeds):
                    run_seed = master_seed + fold_idx * num_seeds + seed_idx # Unique seed for training run
                    run_name = f"{hpo_dir_name}_fold{fold_idx}_seed{seed_idx}"
                    run_dir = hpo_combo_dir / f"fold_{fold_idx}_seed_{seed_idx}"
                    # Data dir will be created inside prepare_job if needed (for symlinks)
                    run_dir.mkdir(parents=True, exist_ok=True)

                    logger.info(f"  Preparing Fold {fold_idx}, Seed {seed_idx} (Run: {run_name})")

                    # Prepare parameters for the job function
                    run_params_specific = run_params_base.copy()
                    run_params_specific['seed'] = run_seed
                    # Remove ratio keys as splitting is handled externally
                    run_params_specific.pop('train_ratio', None)
                    run_params_specific.pop('val_ratio', None)
                    run_params_specific.pop('test_ratio', None)
                    # Remove structure count/ID keys
                    run_params_specific.pop('structure_ids', None)
                    run_params_specific.pop('num_structures', None)

                    # Add the path to the shared data directory
                    run_params_specific['external_data_source_dir'] = shared_data_dir_for_fold

                    # Call the prepare function (now expects external_data_source_dir)
                    try:
                        if model_type == 'mace':
                            prepare_mace_job(
                                db_manager=db_manager,
                                job_name=run_name,
                                job_dir=run_dir,
                                **run_params_specific
                            )
                            script_path = run_dir / f"{run_name}_train.sh"
                            if script_path.exists():
                                job_script_paths.append(script_path.resolve())
                            else:
                                logger.error(f"MACE script not generated: {script_path}")

                        elif model_type == 'allegro':
                            prepare_allegro_job(
                                db_manager=db_manager, # Still needed for chemical symbols from shared data
                                job_name=run_name,
                                job_dir=run_dir,
                                **run_params_specific
                            )
                            script_path = run_dir / f"{run_name}.sh"
                            if script_path.exists():
                                job_script_paths.append(script_path.resolve())
                            else:
                                logger.error(f"Allegro script not generated: {script_path}")
                    except Exception as e:
                         logger.error(f"Error preparing job for {run_name}: {e}", exc_info=True)

        else:
            # --- Standard Train/Val/Test Workflow --- 
             for seed_idx in range(num_seeds):
                 run_seed = master_seed + seed_idx # Unique seed for training run
                 run_name = f"{hpo_dir_name}_seed_{seed_idx}"
                 run_dir = hpo_combo_dir / f"seed_{seed_idx}"
                 run_dir.mkdir(parents=True, exist_ok=True)

                 logger.info(f"  Preparing Seed {seed_idx} (Run: {run_name})")

                 # Prepare parameters for the job function
                 run_params_specific = run_params_base.copy()
                 run_params_specific['seed'] = run_seed
                 # Ensure structure IDs or num_structures is present if needed
                 if 'structure_ids' not in run_params_specific and 'num_structures' not in run_params_specific:
                     run_params_specific['structure_ids'] = structure_ids
                 # Ensure ratios are present
                 if not all(k in run_params_specific for k in ['train_ratio', 'val_ratio', 'test_ratio']):
                     # This shouldn't happen based on initial validation, but good practice
                     raise ValueError(f"Missing train/val/test ratios for non-kfold run: {run_name}")

                 # Call the prepare function (will perform its own splitting)
                 try:
                     if model_type == 'mace':
                         prepare_mace_job(
                             db_manager=db_manager,
                             job_name=run_name,
                             job_dir=run_dir,
                             **run_params_specific
                         )
                         script_path = run_dir / f"{run_name}_train.sh"
                         if script_path.exists():
                              job_script_paths.append(script_path.resolve())
                         else:
                              logger.error(f"MACE script not generated: {script_path}")

                     elif model_type == 'allegro':
                         prepare_allegro_job(
                             db_manager=db_manager,
                             job_name=run_name,
                             job_dir=run_dir,
                             **run_params_specific
                         )
                         script_path = run_dir / f"{run_name}.sh"
                         if script_path.exists():
                             job_script_paths.append(script_path.resolve())
                         else:
                             logger.error(f"Allegro script not generated: {script_path}")
                 except Exception as e:
                     logger.error(f"Error preparing job for {run_name}: {e}", exc_info=True)


    # --- 6. Generate Master Submission Script ---
    submit_script_path = base_sweep_dir / "submit_all_jobs.sh"
    try:
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Master script to submit all HPO sweep jobs\n")
            f.write(f"# Generated for sweep in: {base_sweep_dir.resolve()}\n")
            f.write(f"# Total jobs to submit: {len(job_script_paths)}\n\n")
            f.write(f"SWEEP_BASE_DIR=\"{base_sweep_dir.resolve()}\"\n\n") # Store base dir

            for script_path in job_script_paths:
                job_dir = script_path.parent
                script_name = script_path.name
                # Use the resolved absolute path for cd for robustness
                quoted_absolute_job_dir = f'"{job_dir.resolve()}"'
                quoted_script_name = f'"{script_name}"'

                f.write(f'echo "Submitting job in {job_dir.resolve()}"\n')
                f.write(f"cd {quoted_absolute_job_dir} || {{ echo \"Failed to cd into {job_dir.resolve()}\"; exit 1; }}\n")
                f.write(f"sbatch {quoted_script_name}\n")
                f.write("sleep 1 # Add a small delay to avoid overloading SLURM\n")
                f.write("\n")

        submit_script_path.chmod(0o755)
        logger.info(f"Generated master submission script: {submit_script_path}")

    except Exception as e:
        logger.error(f"Failed to generate master submission script: {e}", exc_info=True)

    logger.info(f"Total job scripts generated: {len(job_script_paths)}")
    logger.info("Sweep preparation complete.")
    logger.info("IMPORTANT: Review the generated scripts before submitting.")

# Example Usage (commented out):
# if __name__ == "__main__":
#     db_path = Path("/path/to/your/database.db") # CHANGE ME
#     db_manager = DatabaseManager(db_path)
#     output_dir = Path("./mace_hpo_sweep_example")

#     sweep_config = {
#         'lr': [0.01, 0.005],
#         'num_channels': [64, 128],
#         'r_max': [4.5, 5.0]
#     }

#     fixed_config = {
#         'gpu_config': {"count": 1, "type": "a100"},
#         'num_structures': 100, # Use 100 random structures
#         # 'structure_ids': [1, 5, 10, ...], # Alternatively, provide specific IDs
#         'e0s': 'average',
#         'num_interactions': 2,
#         'max_L': 0,
#         'forces_weight': 50.0,
#         'energy_weight': 1.0,
#         'stress_weight': 25.0,
#         # If not using k-fold, provide ratios:
#         # 'train_ratio': 0.8,
#         # 'val_ratio': 0.1,
#         # 'test_ratio': 0.1
#     }

#     run_hpo_sweep(
#         db_manager=db_manager,
#         model_type='mace',
#         base_sweep_dir=output_dir,
#         sweep_params=sweep_config,
#         fixed_params=fixed_config,
#         num_seeds=2,
#         k_folds=3,         # Perform 3-fold CV
#         test_ratio=0.1,    # Hold out 10% for final test
#         master_seed=12345
#     )
