# scratch/scripts/run_allegro_fixed_data_tail_angle_schedule.py
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
    bad_ids_set_1 = set(json.load(bad_ids_path.open())) if bad_ids_path.exists() else set()
    logger.info(f"Loaded {len(bad_ids_set_1)} bad structure IDs from {bad_ids_path}.")
    
    bad_ids_path_2 = Path("./bad_structure_ids_gen8.json")
    bad_ids_set_2 = set(json.load(bad_ids_path_2.open())) if bad_ids_path_2.exists() else set()
    logger.info(f"Loaded {len(bad_ids_set_2)} bad structure IDs from {bad_ids_path_2}.")

    bad_ids_set = bad_ids_set_1.union(bad_ids_set_2)
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
        
        rare_ids_path = Path("./rare_structure_ids_gen8.json")
        rare_ids_set = set(json.load(rare_ids_path.open())) if rare_ids_path.exists() else set()
        logger.info(f"Loaded {len(rare_ids_set)} rare structure IDs from {rare_ids_path}.")

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

    experiment_name = "allegro_gen8_custom_loss_functions_fixed_data"
    base_dir = Path(f"../data/allegro_experiments/{experiment_name}")

    # --- Define Experimental Configurations ---
    base_loss_coeffs = {
        "total_energy": {"coeff": 1.0, "metric": "mse"},
        "stress": {"coeff": 50.0, "metric": "mse"},
    }

    loss_configs = [
        {
            "key": "mse",
            "loss": {
                "forces": {"coeff": 10.0, "metric": "mse"}
            }
        },
        {
            "key": "tail_huber_q0.75_d1",
            "loss": {
                "forces": {"coeff": 10.0, "metric": "tail_huber", "params": {"quantile": 0.75, "delta": 1.0}}
            }
        },
        {
            "key": "force_angle",
            "loss": {
                "forces_angle": {"coeff": 10.0, "metric": "forces_angle", "field": "forces"}
            }
        },
        {
            "key": "tail_huber_plus_force_angle",
            "loss": {
                "forces": {"coeff": 10.0, "metric": "tail_huber", "params": {"quantile": 0.75, "delta": 1.0}},
                "forces_angle": {"coeff": 10.0, "metric": "forces_angle", "field": "forces"}
            }
        }
    ]

    sampling_configs = [
        {"sampler_key": "no_sampler", "sampler": None, "sampler_params": None},
        {"sampler_key": "rare_sampling_r3", "sampler": "rare_weighted", "sampler_params": {"replica": 3}},
        {"sampler_key": "rare_sampling_r5", "sampler": "rare_weighted", "sampler_params": {"replica": 5}},
    ]
    
    for loss_config, sampling_config in product(loss_configs, sampling_configs):
        job_name = f"{loss_config['key']}_{sampling_config['sampler_key']}"
        job_dir = base_dir / job_name

        # Combine base loss with the additive component
        final_loss_coeffs = base_loss_coeffs.copy()
        final_loss_coeffs.update(loss_config['loss'])
        
        current_sampler_params = sampling_config.get("sampler_params")
        if current_sampler_params:
            current_sampler_params["rare_idx"] = rare_structure_indices
        
        use_delta_logger_flag = False
        if 'forces' in final_loss_coeffs and final_loss_coeffs['forces'].get('metric') == 'tail_huber':
            if final_loss_coeffs['forces'].get('params', {}).get('delta') == 'auto':
                 use_delta_logger_flag = True


        params = {
            "job_name": job_name,
            "structure_ids": structure_ids,
            #"val_b_ids": list(val_b_ids),
            "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
            "seed": 42,
            "batch_size": 1,
            "r_max" : 6.0,
            "l_max" : 2,
            "num_layers" : 3,
            "max_epochs": 100,
            "wandb_project": experiment_name,
            "checkpoint_monitor_key": "val0_epoch/weighted_sum", # Now monitor val_b
            "loss_coeffs": final_loss_coeffs,
            "sampler": sampling_config["sampler"],
            "sampler_params": current_sampler_params,
            "use_delta_logger": use_delta_logger_flag,
            "mlp_width": 256

        }
        
        run_and_summarize(job_dir, **params)

if __name__ == "__main__":
    main()