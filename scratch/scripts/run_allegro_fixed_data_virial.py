# scratch/scripts/run_allegro_loss_experiment.py
import logging
import json
from pathlib import Path
from itertools import product
import random

from forge.core.database import DatabaseManager
from forge.workflows.db_to_allegro import prepare_allegro_job

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_and_summarize(job_dir, **kwargs):
    """Helper to run prepare_allegro_job and print a summary."""
    # Check if config already exists (since we're using existing data files)
    config_path = job_dir / "config.yaml"
    if config_path.exists():
        logger.info(f"Config for job '{kwargs['job_name']}' already exists. Skipping job preparation.")
        return

    logger.info("-" * 80)
    logger.info(f"Preparing job '{kwargs['job_name']}' in directory: {job_dir.resolve()}")
    
    # Use a fresh DB connection for each job preparation (still needed for chemical symbol extraction)
    with DatabaseManager() as db_manager:
        try:
            structure_splits = prepare_allegro_job(db_manager=db_manager, job_dir=job_dir, **kwargs)
            
            logger.info("\n  SUCCESS: Allegro job prepared.")
            logger.info(f"  - Job Directory: {job_dir.resolve()}")
            logger.info("  - Config file created: config.yaml")
            logger.info("  - Using existing data files: train.xyz, val.xyz, test.xyz")
            
            logger.info("\n  To run this training job, execute:")
            logger.info(f"  cd {job_dir.resolve()}")
            logger.info("  nequip-train config.yaml")
            logger.info("-" * 80 + "\n")

        except Exception as e:
            logger.error(f"Failed to prepare job '{kwargs['job_name']}': {e}", exc_info=True)
            logger.info("-" * 80 + "\n")


def main():
    """Main function to prepare a systematic Allegro training experiment for loss functions."""
    logger.info("Using existing data files (train.xyz, val.xyz, test.xyz)...")
    
    # Define paths to existing data files
    data_train_path = Path("data/train.xyz")
    data_val_path = Path("data/val.xyz") 
    data_test_path = Path("data/test.xyz")
    
    # Check if data files exist
    for data_path in [data_train_path, data_val_path, data_test_path]:
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return
    
    logger.info("All data files found. Proceeding with experiment setup...")

    # --- Define Experimental Configurations ---
    experiment_name = "allegro_virial_loss_study_no_dimers_no_sr_no_val_b"
    base_dir = Path(f"../data/allegro_experiments/{experiment_name}")

    # --------------- Grid definition -------------
    # Main sweep: r_max, num_layers, virial toggle, schedule toggle
    r_max_values    = [5, 6]
    num_layers_vals = [2, 3]
    l_max_fixed     = 2                   # as requested

    virial_options  = [False, True]       # False = no virial; True = add virial_mse
    sched_options   = [False, True]       # False = static    True = two-phase

    # Base static coefficients (used for non-scheduled runs and as starting point for scheduled runs)
    base_loss_coeffs = {
        "total_energy": {"coeff": 1.0, "metric": "mse"},
        "forces":       {"coeff": 10.0, "metric": "tail_huber",
                        "params": {"quantile": 0.75, "delta": 1.0}},
        "stress":       {"coeff": 100.0, "metric": "mse"},
    }

    # ---------------------------------------------------------------------------

    for r_max, n_layers, use_virial, use_sched in product(
            r_max_values, num_layers_vals, virial_options, sched_options):

        vir_flag   = "virial" if use_virial else "no_virial"
        sched_flag = "sched"  if use_sched  else "static"
        job_name   = f"r{r_max}_L{n_layers}_{vir_flag}_{sched_flag}"
        job_dir    = base_dir / job_name

        # --- Build loss coefficients and schedule ------------------------------------------
        loss_coeffs = base_loss_coeffs.copy()
        loss_schedule = None
        
        if use_virial:
            loss_coeffs["virial"] = {"coeff": 25.0, "metric": "virial_mse"}
        
        if use_sched:
            # Create the loss schedule - phase transition at epoch 80
            # The keys in the schedule should match the metric names that will be generated
            # from the loss_coeffs (e.g., "forces_tail_huber", "stress_mse", etc.)
            loss_schedule = {
                80: {  # At epoch 80, change coefficients
                    "per_atom_energy_mse": 1.0,     # energy stays the same
                    "forces_tail_huber": 5.0,       # forces drop from 10 to 5
                    "stress_mse": 50.0,              # stress drops from 100 to 50
                }
            }
            
            # Add virial to schedule if enabled (only appears in phase 2)
            if use_virial:
                loss_schedule[80]["virial_mse"] = 25.0

        # ----------------------------------------------------------------------
        params = {
            "job_name": job_name,
            # HPO mode - use existing data files
            "data_train_path": data_train_path,
            "data_val_path": data_val_path,
            "data_test_path": data_test_path,
            "chemical_symbols_list": ['Ti', 'V', 'Cr', 'Zr', 'W'],  # Provide chemical symbols
            "seed": 42,
            "batch_size": 1,
            "max_epochs": 120 if use_sched else 100,
            "r_max": r_max,
            "l_max": l_max_fixed,
            "num_layers": n_layers,
            # keep the other defaults (tensor/scalar feats, width) unchanged
            "wandb_project": experiment_name,
            "checkpoint_monitor_key": "val0_epoch/weighted_sum",   # monitor stress RMSE
            "loss_coeffs": loss_coeffs,
            "loss_schedule": loss_schedule,        # Now properly separated
            "sampler": None,                       # rare-sampling removed
            "sampler_params": None,
            "use_delta_logger": False,             # not needed with fixed δ
        }

        run_and_summarize(job_dir, **params)

if __name__ == "__main__":
    main()
