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
    # --- 1. Initialize Database Connection ---
    logger.info("Initializing database connection...")
    db_manager = DatabaseManager()

    # --- 2. Query Structures for Training ---
    logger.info("Finding structures with completed VASP calculations...")
    # NOTE: Customize this query to select the base dataset for your experiment
    structure_ids = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
    logger.info(f"Found {len(structure_ids)} structures for the experiment.")

    if not structure_ids:
        logger.error("No structures found. Exiting.")
        return
        
    # --- 3. Load Pre-identified Rare Structures ---
    rare_ids_path = Path("./rare_structure_ids.json")
    if not rare_ids_path.exists():
        logger.warning(
            f"'{rare_ids_path}' not found. "
            f"Run 'identify_rare_structures.py' first to enable rare sampling."
        )
        rare_ids_set = set()
    else:
        with open(rare_ids_path, 'r') as f:
            rare_ids_set = set(json.load(f))
        logger.info(f"Loaded {len(rare_ids_set)} rare structure IDs from {rare_ids_path.resolve()}")

    # Convert the loaded structure IDs to dataset indices for the sampler
    # The sampler operates on the indices of the `structure_ids` list, not the IDs themselves.
    structure_id_to_index = {sid: i for i, sid in enumerate(structure_ids)}
    rare_structure_indices = [structure_id_to_index[sid] for sid in rare_ids_set if sid in structure_id_to_index]
    
    if rare_ids_set and not rare_structure_indices:
        logger.warning("Loaded rare structure IDs do not overlap with the experiment's dataset.")
    else:
        logger.info(f"Identified {len(rare_structure_indices)} rare structures within the current dataset to be used for sampling.")

    # --- 4. Define Experiment Parameters ---
    experiment_name = "allegro_training_study_1"
    base_experiment_dir = Path(f"../data/allegro_experiments/{experiment_name}")
    
    # Base arguments shared across all experimental phases
    base_args = {
        "db_manager": db_manager,
        "structure_ids": structure_ids,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "seed": 42,
        "max_epochs": 1000,
    }

    # --- 5. Run All Experimental Phases ---

    # Phase 1: Robust Loss (Focal Loss) + Gradient Norm Clipping
    phase1_args = {
        **base_args,
        "job_name": "phase1_focal_loss_gradnorm",
        "loss_function": "focal",
        "loss_params": {"beta": 1.0, "gamma": 2.0},
        "callbacks": ["grad_norm"],
        "callback_params": {"grad_norm": {"alpha": 1.5}}
    }
    run_and_summarize(base_experiment_dir / phase1_args["job_name"], **phase1_args)

    # Phase 2: Data-centric (Rare-weighted Sampler)
    phase2_args = {
        **base_args,
        "job_name": "phase2_rare_sampling",
        "sampler": "rare_weighted",
        "sampler_params": {"replica": 5, "alpha": 0.25, "rare_idx": rare_structure_indices}
    }
    run_and_summarize(base_experiment_dir / phase2_args["job_name"], **phase2_args)

    # Phase 3: Batch Size Sweep (Larger Batch Size)
    phase3_args = {
        **base_args,
        "job_name": "phase3_large_batch",
        "batch_size": 16, # Default is 4
    }
    run_and_summarize(base_experiment_dir / phase3_args["job_name"], **phase3_args)

    # Phase 4: Integrated "Best" Settings
    phase4_args = {
        **base_args,
        "job_name": "phase4_integrated_final",
        "loss_function": "huber",
        "loss_params": {"delta": 1.0},
        "sampler": "rare_weighted",
        "sampler_params": {"replica": 3, "rare_idx": rare_structure_indices},
        "callbacks": ["curriculum"],
        "extra_trainer_params": {"accumulate_grad_batches": 4},
    }
    run_and_summarize(base_experiment_dir / phase4_args["job_name"], **phase4_args)


if __name__ == "__main__":
    main() 