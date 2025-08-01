#!/usr/bin/env python

import os
from pathlib import Path
from forge.core.database import DatabaseManager
from forge.workflows.hpo_sweep import run_hpo_sweep

# --- Configuration ---

# 1. Database Path (IMPORTANT: Change this to your actual database file)
#db_file = Path(os.environ.get("FORGE_DB_PATH", "/path/to/your/forge_database.db"))
#if not db_file.exists() or str(db_file) == "/path/to/your/forge_database.db":
    #print(f"Error: Database file not found at '{db_file}'")
    #print("Please set the FORGE_DB_PATH environment variable or modify the script.")
    #exit(1)

# 2. Output Directory for the Sweep
sweep_output_dir = Path("./allegro_hpo_kfold_example")

# 3. Hyperparameters to Sweep Over
#    Keys are the parameter names expected by prepare_allegro_job
#    Values are lists of the values to try for each parameter
sweep_parameters = {
    'num_layers': [1, 2],
    'l_max': [1, 2],
    # Add other parameters here if needed, e.g.:
    # 'lr': [0.001, 0.0005],
    # 'r_max': [4.5, 5.0],
}

# 4. Fixed Parameters (Constant across all runs in the sweep)
#    Include parameters required by prepare_allegro_job that are *not* swept
fixed_parameters = {
    # GPU Configuration
    'gpu_config': {"count": 1, "type": "rtx6000"}, # Adjust GPU count and type as needed

    # Data Selection (choose one)
    'num_structures': 500,      # Select 100 random structures from the DB
    # 'structure_ids': [1, 5, 10, 20, ...], # Or provide a specific list of IDs

    # Allegro Model Hyperparameters (set defaults or desired fixed values)
    'r_max': 5.0,
    'num_scalar_features': 128,
    'num_tensor_features': 32,
    'mlp_depth': 2,
    'mlp_width': 128,

    # Training Parameters
    'max_epochs': 50,          # Adjust number of epochs
    'lr': 0.001,
    'schedule': None,           # Use default annealing schedule in prepare_allegro_job
    'loss_coeffs': None,        # Use default loss coefficients in prepare_allegro_job
    'project': "allegro-hpo-sweep-example", # WandB project name

    # SLURM Parameters (used by template script)
    'num_nodes': 1,

    # Note: We don't need train/val/test_ratio here because k_folds is > 1
}

# 5. Sweep Control Parameters
num_repetitions = 2          # Number of times to repeat each HPO/fold with different seeds
num_k_folds = 2              # Number of folds for cross-validation
holdout_test_ratio = 0.1    # Fraction of data reserved for the final test set (0.0 to 1.0)
main_seed = 1234             # Master seed for initial shuffling

# --- Run the Sweep ---

if __name__ == "__main__":
    print("--- Starting Allegro HPO Sweep ---")
    #print(f"Database: {db_file}")
    print(f"Output Directory: {sweep_output_dir}")
    print(f"Sweeping over: {sweep_parameters}")
    print(f"Fixed parameters: {fixed_parameters}")
    print(f"K-Folds: {num_k_folds}")
    print(f"Repetitions per fold: {num_repetitions}")
    print(f"Test set ratio: {holdout_test_ratio}")

    # Initialize the Database Manager
    try:
        dbm = DatabaseManager()
    except Exception as e:
        print(f"Error initializing DatabaseManager: {e}")
        exit(1)

    # Execute the HPO sweep function
    try:
        run_hpo_sweep(
            db_manager=dbm,
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
        exit(1)
    except Exception as e:
        print(f"--- An unexpected error occurred ---")
        print(e)
        import traceback
        print(traceback.format_exc())
        exit(1)

