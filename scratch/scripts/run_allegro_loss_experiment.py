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
    with DatabaseManager() as db_manager:
        all_ids = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
        dimer_ids = db_manager.find_structures_by_metadata({'config_type': 'dimer'})
        structure_ids = [sid for sid in all_ids if sid not in dimer_ids]
        structure_ids = structure_ids[:5000]
        logger.info(f"Found {len(structure_ids)} structures for the experiment.")
        
        rare_ids_path = Path("./rare_structure_ids.json")
        rare_ids_set = set(json.load(rare_ids_path.open())) if rare_ids_path.exists() else set()
        logger.info(f"Loaded {len(rare_ids_set)} rare structure IDs.")

    if not structure_ids:
        logger.error("No structures found. Exiting.")
        return

    # --- Split Rare IDs for Val-B ---
    val_b_frac = 0.5  # Use 50% of rare structures for the 'hard' validation set
    num_val_b = int(len(rare_ids_set) * val_b_frac)
    val_b_ids = set(random.sample(list(rare_ids_set), num_val_b))
    rare_ids_for_training_pool = rare_ids_set - val_b_ids
    logger.info(f"Assigned {len(val_b_ids)} structures to Val-B. Kept {len(rare_ids_for_training_pool)} in the main pool.")

    # Indices for sampler still need to be computed based on the final training set
    structure_id_to_index = {sid: i for i, sid in enumerate(structure_ids) if sid not in val_b_ids}
    rare_structure_indices = [structure_id_to_index[sid] for sid in rare_ids_for_training_pool if sid in structure_id_to_index]

    experiment_name = "allegro_additive_loss_study"
    base_dir = Path(f"../data/allegro_experiments/{experiment_name}")

    # --- Define Experimental Configurations ---
    base_loss_coeffs = {
        "total_energy": {"coeff": 1.0, "metric": "mse"},
        "forces": {"coeff": 10.0, "metric": "auto_stratified_huber"},
        "stress": {"coeff": 50.0, "metric": "mse"},
    }

    additive_losses = [
        {"loss_key": "base_only", "loss": {}},
        {"loss_key": "force_angle", "loss": {"forces_angle": {"coeff": 10.0, "metric": "force_angle", "field": "forces"}}},
        {"loss_key": "stress_angle", "loss": {"stress_angle": {"coeff": 25.0, "metric": "stress_angle", "field": "stress"}}},
        {"loss_key": "stress_shear", "loss": {"stress_shear": {"coeff": 25.0, "metric": "stress_shear_mae", "field": "stress"}}},
        {"loss_key": "focal_force", "loss": {"forces_focal": {"coeff": 10.0, "metric": "focal_mse", "params": {"beta": "auto"}, "field": "forces"}}},
    ]

    sampling_configs = [
        {"sampler_key": "no_sampler", "sampler": None},
        {"sampler_key": "rare_sampling", "sampler": "rare_weighted", "sampler_params": {"replica": 5, "rare_idx": rare_structure_indices}},
    ]
    
    for additive_config, sampling_config in product(additive_losses, sampling_configs):
        job_name = f"{additive_config['loss_key']}_{sampling_config['sampler_key']}"
        job_dir = base_dir / job_name

        # Combine base loss with the additive component
        final_loss_coeffs = {**base_loss_coeffs, **additive_config['loss']}

        params = {
            "job_name": job_name,
            "structure_ids": structure_ids,
            "val_b_ids": list(val_b_ids),
            "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
            "seed": 42,
            "max_epochs": 150,
            "wandb_project": experiment_name,
            "checkpoint_monitor_key": "val1_epoch/forces_mae", # Now monitor val_b
            "loss_coeffs": final_loss_coeffs,
            "sampler": sampling_config["sampler"],
            "sampler_params": sampling_config.get("sampler_params"),
        }
        
        run_and_summarize(job_dir, **params)

if __name__ == "__main__":
    main()
