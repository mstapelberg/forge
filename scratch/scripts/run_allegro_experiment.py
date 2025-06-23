# scratch/scripts/run_allegro_experiment.py
import logging
import json
from pathlib import Path

from forge.core.database import DatabaseManager
from forge.workflows.db_to_allegro import prepare_allegro_job

# Configure logging to see the output from the workflow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_and_summarize(job_dir, **kwargs):
    """Helper to run prepare_allegro_job and print a summary."""
    logger.info("-" * 80)
    logger.info(f"Preparing job '{kwargs['job_name']}' in directory: {job_dir.resolve()}")
    
    try:
        structure_splits = prepare_allegro_job(job_dir=job_dir, **kwargs)
        
        logger.info("\n  SUCCESS: Allegro job prepared.")
        logger.info(f"  - Job Directory: {job_dir.resolve()}")
        logger.info("  - Config file created: config.yaml")
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
    """
    Main function to prepare a multi-stage Allegro training experiment.
    """
    # --- 1. Initialize a temporary Database Connection to get IDs ---
    # This connection will be closed after we get the initial data.
    logger.info("Initializing database connection to fetch initial dataset...")
    with DatabaseManager() as db_manager:
        logger.info("Finding structures with completed VASP calculations...")
        structure_ids = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
        logger.info(f"Found {len(structure_ids)} structures with VASP calculations.")
        dimer_ids = db_manager.find_structures_by_metadata({'config_type': 'dimer'})
        logger.info(f"Found {len(dimer_ids)} structures with dimer config type that are bad.")
        # remove the dimer ids from the structure ids
        structure_ids = [sid for sid in structure_ids if sid not in dimer_ids] # remove the dimer ids from the structure ids
        logger.info(f"Found {len(structure_ids)} structures for the experiment.")
        
        # Load rare IDs from file
        rare_ids_path = Path("./rare_structure_ids.json")
        if not rare_ids_path.exists():
            logger.warning(f"'{rare_ids_path}' not found. Sampler will not use rare IDs.")
            rare_ids_set = set()
        else:
            with open(rare_ids_path, 'r') as f:
                rare_ids_set = set(json.load(f))
            logger.info(f"Loaded {len(rare_ids_set)} rare structure IDs from {rare_ids_path.resolve()}")

    if not structure_ids:
        logger.error("No structures found. Exiting.")
        return
        
    # Convert the loaded structure IDs to dataset indices for the sampler
    structure_id_to_index = {sid: i for i, sid in enumerate(structure_ids)}
    rare_structure_indices = [structure_id_to_index[sid] for sid in rare_ids_set if sid in structure_id_to_index]
    logger.info(f"Identified {len(rare_structure_indices)} rare structures within the current dataset to be used for sampling.")

    # --- 2. Define Experiment Parameters ---
    experiment_name = "allegro_training_study_1"
    base_experiment_dir = Path(f"../data/allegro_experiments/{experiment_name}")
    
    # Base arguments shared across all experimental phases.
    # A fresh DatabaseManager will be created for each phase.
    base_args = {
        "structure_ids": structure_ids,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "seed": 42,
        "max_epochs": 400,
        "wandb_project": "allegro_training_study_1",
        "checkpoint_monitor_key": "val0_epoch/weighted_sum", # Monitor stress RMSE for saving checkpoints
    }

    # --- 3. Run All Experimental Phases ---

    # Phase 1: Use the built-in LossCoefficientScheduler
    phase1_args = {
        **base_args,
        "job_name": "phase1_loss_schedule",
        "loss_function": "huber",
        "loss_params": {"delta": 1.0},
        "loss_schedule": {
            0:   {"forces_huber": 10.0, "stress_huber": 100.0},
            150: {"forces_huber": 50.0, "stress_huber": 50.0},
            300: {"forces_huber": 100.0, "stress_huber": 10.0},
        }
    }
    with DatabaseManager() as db:
        run_and_summarize(base_experiment_dir / phase1_args["job_name"], db_manager=db, **phase1_args)

    # Phase 2: Data-centric (Rare-weighted Sampler) - with implementation choice
    phase2_args = {
        **base_args,
        "job_name": "phase2_rare_sampling_v3",
        "sampler": "rare_weighted",
        "sampler_params": {"replica": 5, "alpha": 0.25, "rare_idx": rare_structure_indices},
        "sampler_implementation": "v3"  # Can change to 'v2' to test V2 implementation
    }
    with DatabaseManager() as db:
        run_and_summarize(base_experiment_dir / phase2_args["job_name"], db_manager=db, **phase2_args)
    
    # Phase 2b: Same but with V2 implementation for comparison
    phase2b_args = {
        **base_args,
        "job_name": "phase2_rare_sampling_v2",
        "sampler": "rare_weighted",
        "sampler_params": {"replica": 5, "alpha": 0.25, "rare_idx": rare_structure_indices},
        "sampler_implementation": "v2"  # Using V2 implementation
    }
    # Uncomment to test V2:
    # with DatabaseManager() as db:
    #     run_and_summarize(base_experiment_dir / phase2b_args["job_name"], db_manager=db, **phase2b_args)

    # Phase 3: Batch Size Sweep (Larger Batch Size)
    phase3_args = {
        **base_args,
        "job_name": "phase3_large_batch",
        "batch_size": 16, # Default is 4
    }
    with DatabaseManager() as db:
        run_and_summarize(base_experiment_dir / phase3_args["job_name"], db_manager=db, **phase3_args)

    # Phase 4: Integrated "Best" Settings (No longer uses custom callbacks)
    phase4_args = {
        **base_args,
        "job_name": "phase4_integrated_final",
        "loss_function": "huber",
        "loss_params": {"delta": 1.0},
        "sampler": "rare_weighted",
        "sampler_params": {"replica": 3, "rare_idx": rare_structure_indices},
        "extra_trainer_params": {"accumulate_grad_batches": 4},
    }
    with DatabaseManager() as db:
        run_and_summarize(base_experiment_dir / phase4_args["job_name"], db_manager=db, **phase4_args)


if __name__ == "__main__":
    main() 