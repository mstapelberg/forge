#!/usr/bin/env python

import os
import numpy as np
import random
from pathlib import Path
from forge.core.database import DatabaseManager
from forge.workflows.hpo_sweep import run_hpo_sweep

# --- Configuration ---\n

# 1. Database Path (IMPORTANT: Change this to your actual database file)
#db_file = Path(os.environ.get("FORGE_DB_PATH", "/path/to/your/forge_database.db"))
#if not db_file.exists() or str(db_file) == "/path/to/your/forge_database.db":
    #print(f"Error: Database file not found at '{db_file}'")
    #print("Please set the FORGE_DB_PATH environment variable or modify the script.")
    #exit(1)

random.seed(42)
RANDOM_ID_MIN = 4
RANDOM_ID_MAX = 23000
INPUT_FILE = 'selected_ids_gen6_k10000.txt' # File with initial IDs

# Removed global initialization: db_manager = DatabaseManager()


# 2. Output Directory for the Sweep
sweep_output_dir = Path("./allegro_hpo_kfold")

# 3. Hyperparameters to Sweep Over
#    Keys are the parameter names expected by prepare_allegro_job
#    Values are lists of the values to try for each parameter
sweep_parameters = {
    'num_layers': [1, 2],
    'l_max': [1, 2],
    'num_scalar_features': [32,64,128],
    'num_tensor_features': [16,32,64],
    'mlp_width': [128, 256, 512],
    # Add other parameters here if needed, e.g.:
    # 'lr': [0.001, 0.0005],
    # 'r_max': [4.5, 5.0],
}

# ID processing logic moved inside main block


# 4. Fixed Parameters (Constant across all runs in the sweep)
#    Include parameters required by prepare_allegro_job that are *not* swept
fixed_parameters = {
    # GPU Configuration
    'gpu_config': {"count": 1, "type": "rtx6000"}, # Adjust GPU count and type as needed

    # Data Selection (choose one)
    #'num_structures': 500,      # Select 100 random structures from the DB
    # 'structure_ids': [1, 5, 10, 20, ...], # Or provide a specific list of IDs
    # 'structure_ids': structure_ids, # This will be set inside the main block


    # Allegro Model Hyperparameters (set defaults or desired fixed values)
    'r_max': 5.0,
    #'num_scalar_features': 128,
    #'num_tensor_features': 32,
    'mlp_depth': 2,
    #'mlp_width': 128,

    # Training Parameters
    'max_epochs': 200,          # Adjust number of epochs
    'lr': 0.001,
    'schedule': None,           # Use default annealing schedule in prepare_allegro_job
    'loss_coeffs': None,        # Use default loss coefficients in prepare_allegro_job
    'project': "allegro-hpo-sweep", # WandB project name

    # SLURM Parameters (used by template script)
    'num_nodes': 1,

    # Note: We don't need train/val/test_ratio here because k_folds is > 1
}

# 5. Sweep Control Parameters
num_repetitions = 0          # Number of times to repeat each HPO/fold with different seeds
num_k_folds = 5              # Number of folds for cross-validation
holdout_test_ratio = 0.1    # Fraction of data reserved for the final test set (0.0 to 1.0)
main_seed = 0             # Master seed for initial shuffling

# --- Run the Sweep ---\n

if __name__ == "__main__":
    print("--- Starting Allegro HPO Sweep ---")
    #print(f"Database: {db_file}")
    print(f"Output Directory: {sweep_output_dir}")
    print(f"Sweeping over: {sweep_parameters}")

    # Initialize the Database Manager here
    print("Initializing DatabaseManager...")
    db_manager = None # Initialize to None
    try:
        db_manager = DatabaseManager()
    except Exception as e:
        print(f"Error initializing DatabaseManager: {e}")
        exit(1)
    print("DatabaseManager initialized.")

    # --- Process Structure IDs (Moved Here) ---
    print("Processing structure IDs...")
    try:
        initial_structure_ids = np.loadtxt(INPUT_FILE, dtype=int).tolist()
        print(f"Loaded {len(initial_structure_ids)} initial IDs from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        if db_manager: db_manager.close_connection() # Close connection if open
        exit(1)
    except Exception as e:
        print(f"Error loading initial IDs: {e}")
        if db_manager: db_manager.close_connection()
        exit(1)

    try:
        print("Fetching dimer IDs...")
        dimer_ids = db_manager.find_structures_by_metadata(metadata_filters={'config_type':'dimer'})
        print(f"Found {len(dimer_ids)} dimer IDs.")
    except Exception as e:
        print(f"Warning: Could not fetch dimer IDs ({e}). Proceeding without dimer removal.")
        dimer_ids = []

    # Convert to sets
    initial_set = set(initial_structure_ids)
    dimer_set = set(dimer_ids)

    # Calculate how many were removed
    removed_dimer_ids = initial_set.intersection(dimer_set)
    count_removed = len(removed_dimer_ids)

    # Calculate remaining IDs after removal
    remaining_ids = initial_set.difference(removed_dimer_ids)

    # Determine how many replacements are needed
    count_needed = count_removed

    # Find potential replacements
    # Combine all IDs that cannot be used for replacement
    forbidden_ids = dimer_set.union(remaining_ids)
    count_added = 0
    attempts = 0
    max_attempts = count_needed * 500 # Generous attempt limit

    # Set to store added IDs to ensure uniqueness
    added_ids_set = set()

    print(f"Attempting to replace {count_removed} removed dimer IDs...")
    while count_added < count_needed and attempts < max_attempts:
        random_id = random.randint(RANDOM_ID_MIN, RANDOM_ID_MAX)
        if random_id not in forbidden_ids and random_id not in added_ids_set:
            added_ids_set.add(random_id)
            count_added += 1
        attempts += 1

    # Report results
    print(f"Initial number of IDs: {len(initial_set)}")
    print(f"Number of dimer IDs found and removed: {count_removed}")
    print(f"Number of random IDs added as replacements: {count_added}")
    if count_added < count_needed:
        print(f"Warning: Could only add {count_added} replacements, needed {count_needed}.")

    # Combine the sets
    structure_ids_set = remaining_ids.union(added_ids_set)

    # Convert to a sorted list
    structure_ids = sorted(list(structure_ids_set))

    print(f"Final number of IDs for sweep: {len(structure_ids)}")
    # -----------------------------

    # Add the final structure IDs to fixed_parameters
    fixed_parameters['structure_ids'] = structure_ids

    print(f"Final fixed parameters (structure_ids updated):")
    # Avoid printing huge list, just print the count
    temp_fixed = fixed_parameters.copy()
    temp_fixed['structure_ids'] = f"{len(fixed_parameters['structure_ids'])} IDs"
    print(temp_fixed)
    print(f"K-Folds: {num_k_folds}")
    print(f"Repetitions per fold: {num_repetitions}")
    print(f"Test set ratio: {holdout_test_ratio}")


    # Execute the HPO sweep function using the single db_manager instance
    try:
        run_hpo_sweep(
            db_manager=db_manager, # Use the initialized db_manager
            model_type='allegro',
            base_sweep_dir=sweep_output_dir,
            sweep_params=sweep_parameters,
            fixed_params=fixed_parameters,
            num_seeds=num_repetitions,
            k_folds=num_k_folds,
            test_ratio=holdout_test_ratio,
            master_seed=main_seed
        )
        print("--- HPO Sweep Setup Complete ---")
        print(f"Job directories and submission script generated in: {sweep_output_dir}")
        print(f"Review the 'submit_all_jobs.sh' script before running.")

    except ValueError as ve:
        print(f"--- Configuration Error ---")
        print(ve)
    except Exception as e:
        print(f"--- An unexpected error occurred during sweep setup ---")
        print(f"Error: {e}")
    finally:
        # Ensure DB connection is closed cleanly
        if db_manager:
            print("Closing database connection...")
            db_manager.close_connection()
            print("Database connection closed.")