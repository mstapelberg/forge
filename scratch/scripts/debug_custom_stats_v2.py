# scratch/scripts/debug_custom_stats_v2.py
import logging
from pathlib import Path
import yaml
import random

from forge.core.database import DatabaseManager
from forge.workflows.db_to_allegro import prepare_allegro_job
from hydra.utils import instantiate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Prepares an Allegro job and then programmatically runs the statistics calculation."""
    experiment_name = "debug_stats_run"
    job_name = "debug_job"
    base_dir = Path(f"./{experiment_name}")
    job_dir = base_dir / job_name

    # --- 1. Prepare an Allegro job with multiple validation sets ---
    logger.info(f"Preparing Allegro job '{job_name}' in {job_dir}...")
    with DatabaseManager() as db_manager:
        # Get enough structures for train, val_a, and val_b
        all_ids = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
        if len(all_ids) < 70:
            logger.error(f"Not enough structures found in DB ({len(all_ids)}). Need at least 70 for this test. Exiting.")
            return

        random.shuffle(all_ids)
        val_b_ids = all_ids[:20]
        structure_ids_for_split = all_ids[20:70] # Use next 50 for train/val_a
        logger.info(f"Using {len(structure_ids_for_split)} structures for train/val_a and {len(val_b_ids)} for val_b.")

        prepare_allegro_job(
            db_manager=db_manager,
            job_name=job_name,
            job_dir=job_dir,
            structure_ids=structure_ids_for_split,
            val_b_ids=val_b_ids,
            train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, # Splits the main pool
            loss_coeffs={"total_energy": {"coeff": 1.0, "metric": "mse"}}, # Simple loss
            max_epochs=1, # Not training, just for config
            r_max=5.0,
        )
    logger.info("Job preparation complete. Config.yaml created.")

    # --- 2. Load the generated config and run statistics ---
    config_path = job_dir / "config.yaml"
    logger.info(f"Loading config from {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Instantiating DataModule from config...")
    data_module = instantiate(config['data'])
    data_module.prepare_data()
    data_module.setup("fit") # Use 'fit' to prepare train and val splits

    # --- 3. Verify validation dataloaders ---
    logger.info("Checking validation dataloaders...")
    val_dataloaders = data_module.val_dataloader()
    if isinstance(val_dataloaders, list):
        logger.info(f"SUCCESS: Found {len(val_dataloaders)} validation dataloaders.")
        for i, dl in enumerate(val_dataloaders):
            logger.info(f"  - Val dataloader {i} has {len(dl.dataset)} samples.")
    else:
        logger.warning("Found only one validation dataloader, not a list as expected.")

    # --- 4. The main test: Calculate statistics on the training set ---
    logger.info("Getting training dataloader...")
    train_dataloader = data_module.train_dataloader()
    
    # In NequIP, the DataStatisticsManager is an attribute of the DataModule
    stats_manager = data_module.dataset_statistic_manager
    logger.info(f"Using stats manager: {type(stats_manager)}")

    logger.info("Resetting and calculating statistics...")
    stats_manager.reset()
    final_stats = stats_manager.get_statistics(train_dataloader)

    logger.info("\n--- FINAL STATISTICS ---")
    for key, value in final_stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v:.4f}")
        elif isinstance(value, list):
             logger.info(f"  {key}: {[f'{v:.4f}' for v in value]}")
        else:
            logger.info(f"  {key}: {value:.4f}")
    logger.info("------------------------")
    logger.info("\nStatistics calculation successful.")

if __name__ == "__main__":
    main() 