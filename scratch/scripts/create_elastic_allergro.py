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
        #all_ids = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
        all_ids = db_manager.find_structures_by_metadata({'config_type': 'elastic', 'config_type': 'bcc' }, operator='contains')
        dimer_ids = db_manager.find_structures_by_metadata({'config_type': 'dimer'})
        sr_dimer_ids = db_manager.find_structures_by_metadata({'config_type': 'short_range_dimer'})
        sr_ids = db_manager.find_structures_by_metadata({'config_type': 'short_range'})
        
        # Filter out bad IDs and dimers
        structure_ids = [sid for sid in all_ids if sid not in dimer_ids and sid not in sr_dimer_ids and sid not in sr_ids and sid not in bad_ids_set]
        
        structure_ids = structure_ids
        logger.info(f"Found {len(structure_ids)} structures for the experiment after filtering.")
        
        #rare_ids_path = Path("./analysis_output_full/rare_structure_ids.json")
        #rare_ids_set = set(json.load(rare_ids_path.open())) if rare_ids_path.exists() else set()
        rare_ids_set = set()
        logger.info(f"Loaded {len(rare_ids_set)} rare structure IDs.")

    if not structure_ids:
        logger.error("No structures found. Exiting.")
        return

    # --- Split Rare IDs for Val-B ---
    #val_b_frac = 0.5  # Use 50% of rare structures for the 'hard' validation set
    val_b_frac = 0.0
    num_val_b = int(len(rare_ids_set) * val_b_frac)
    val_b_ids = set(random.sample(list(rare_ids_set), num_val_b))
    rare_ids_for_training_pool = rare_ids_set - val_b_ids
    logger.info(f"Assigned {len(val_b_ids)} structures to Val-B. Kept {len(rare_ids_for_training_pool)} in the main pool.")

    # Indices for sampler still need to be computed based on the final training set
    structure_id_to_index = {sid: i for i, sid in enumerate(structure_ids) if sid not in val_b_ids}
    rare_structure_indices = [structure_id_to_index[sid] for sid in rare_ids_for_training_pool if sid in structure_id_to_index]

    experiment_name = "allegro_simple_elastic"
    base_dir = Path(f"../data/allegro_experiments/{experiment_name}")

    # --- Define Experimental Configurations ---
    base_loss_coeffs = {
        "total_energy": {"coeff": 1.0, "metric": "mse"},
        "stress": {"coeff": 50.0, "metric": "mse"},
    }

    force_loss_configs = [
        {"key": "base_huber_q0.75", "params": {"quantile": 0.75, "delta": 1.0}},
        {"key": "huber_q0.65", "params": {"quantile": 0.65, "delta": 1.0}},
        {"key": "huber_q0.65_auto_delta", "params": {"quantile": 0.65, "delta": "auto"}},
        {"key": "huber_q0.85", "params": {"quantile": 0.85, "delta": 1.0}},
    ]

    sampling_configs = [
        {"sampler_key": "no_sampler", "sampler": None, "sampler_params": None},
        {"sampler_key": "rare_sampling", "sampler": "rare_weighted", "sampler_params": {"replica": 5, "rare_idx": rare_structure_indices}},
    ]
    
    for force_config, sampling_config in product(force_loss_configs, sampling_configs):
        job_name = f"{force_config['key']}_{sampling_config['sampler_key']}"
        job_dir = base_dir / job_name

        # Combine base loss with the additive component
        final_loss_coeffs = base_loss_coeffs.copy()
        final_loss_coeffs["forces"] = {"coeff": 10.0, "metric": "tail_huber", "params": force_config["params"]}
        
        use_delta_logger_flag = (force_config["params"].get("delta") == "auto")

        params = {
            "job_name": job_name,
            "structure_ids": structure_ids,
            #"val_b_ids": list(val_b_ids),
            "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
            "seed": 42,
            "batch_size": 1,
            "max_epochs": 400,
            "wandb_project": experiment_name,
            "checkpoint_monitor_key": "val0_epoch/weighted_sum", # Now monitor val_b
            "loss_coeffs": final_loss_coeffs,
            "sampler": sampling_config["sampler"],
            "sampler_params": sampling_config.get("sampler_params"),
            "use_delta_logger": use_delta_logger_flag,
        }
        
        run_and_summarize(job_dir, **params)

if __name__ == "__main__":
    main()