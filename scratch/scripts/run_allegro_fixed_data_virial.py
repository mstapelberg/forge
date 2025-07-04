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
    # Check if data already exists
    train_xyz_path = job_dir / "data" / f"{kwargs['job_name']}_train.xyz"
    if train_xyz_path.exists():
        logger.info(f"Data for job '{kwargs['job_name']}' already exists. Skipping data preparation.")
        return

    logger.info("-" * 80)
    logger.info(f"Preparing job '{kwargs['job_name']}' in directory: {job_dir.resolve()}")
    
    # Use a fresh DB connection for each job preparation
    with DatabaseManager() as db_manager:
        try:
            structure_splits = prepare_allegro_job(db_manager=db_manager, job_dir=job_dir, **kwargs)
            
            logger.info("\n  SUCCESS: Allegro job prepared.")
            logger.info(f"  - Job Directory: {job_dir.resolve()}")
            logger.info("  - Config file created: config.yaml")
            if structure_splits:
                for split_name, struct_ids in structure_splits.items():
                    logger.info(f"    - {split_name}: {len(struct_ids)} structures")
            
            logger.info("\n  To run this training job, execute:")
            logger.info(f"  cd {job_dir.resolve()}")
            logger.info("  nequip-train config.yaml")
            logger.info("-" * 80 + "\n")

        except Exception as e:
            logger.error(f"Failed to prepare job '{kwargs['job_name']}': {e}", exc_info=True)
            logger.info("-" * 80 + "\n")


def main():
    """Main function to prepare a systematic Allegro training experiment for loss functions."""
    logger.info("Initializing database connection...")
    
    bad_ids_path = Path("./analysis_output_full/bad_structure_ids.json")
    bad_ids_set = set(json.load(bad_ids_path.open())) if bad_ids_path.exists() else set()
    logger.info(f"Loaded {len(bad_ids_set)} bad structure IDs to exclude.")

    with DatabaseManager() as db_manager:
        all_ids = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
        dimer_ids = db_manager.find_structures_by_metadata({'config_type': 'dimer'})
        sr_dimer_ids = db_manager.find_structures_by_metadata({'config_type': 'short_range_dimer'})
        sr_ids = db_manager.find_structures_by_metadata({'config_type': 'short_range'})
        
        # Filter out bad IDs and dimers
        structure_ids = [sid for sid in all_ids if sid not in dimer_ids and sid not in sr_dimer_ids and sid not in sr_ids and sid not in bad_ids_set]
        
        structure_ids = structure_ids
        logger.info(f"Found {len(structure_ids)} structures for the experiment after filtering.")
        
        rare_ids_path = Path("./analysis_output_full/rare_structure_ids.json")
        rare_ids_set = set(json.load(rare_ids_path.open())) if rare_ids_path.exists() else set()
        logger.info(f"Loaded {len(rare_ids_set)} rare structure IDs.")

    if not structure_ids:
        logger.error("No structures found. Exiting.")
        return

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
            "structure_ids": structure_ids,
            "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
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
