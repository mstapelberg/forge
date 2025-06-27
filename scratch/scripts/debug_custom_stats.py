# scratch/scripts/debug_custom_stats.py
import logging
from pathlib import Path
import torch
from forge.core.database import DatabaseManager
from forge.workflows.allegro_utils.custom_stats import ExtendedDataStatisticsManager
from nequip.data.datamodule import ASEDataModule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    A minimal script to test and debug the ExtendedDataStatisticsManager.
    """
    chemical_symbols = ["Ti", "V", "Cr", "Zr", "W"]
    # --- 1. Get a small subset of data ---
    logger.info("Initializing database to fetch a small test dataset...")
    with DatabaseManager() as db_manager:
        all_ids = db_manager.get_all_structure_ids()
        if len(all_ids) > 1000:
            test_ids = all_ids[:1000]
            logger.info(f"Using the first 1000 structures for the test.")
        else:
            test_ids = all_ids
            logger.info(f"Using all {len(all_ids)} available structures for the test.")

    if not test_ids:
        logger.error("No structures found in the database. Exiting.")
        return

    # --- 2. Create a temporary data file ---
    temp_dir = Path("./temp_stats_debug")
    temp_dir.mkdir(exist_ok=True)
    train_xyz_path = temp_dir / "train.xyz"
    val_xyz_path = temp_dir / "val.xyz"

    logger.info(f"Saving {len(test_ids)} structures to temporary files...")
    with DatabaseManager() as db_manager:
        atoms_list = db_manager.get_batch_atoms_with_calculation(test_ids)
        # Split the list for train/val
        train_atoms = atoms_list[:800]
        val_atoms = atoms_list[800:]
        from ase.io import write
        write(train_xyz_path, train_atoms, format='extxyz')
        write(val_xyz_path, val_atoms, format='extxyz')

    # --- 3. Instantiate the DataStatisticsManager ---
    logger.info("Instantiating ExtendedDataStatisticsManager...")
    stats_manager = ExtendedDataStatisticsManager(
        dataloader_kwargs={'batch_size': 32, 'num_workers': 0},
        type_names=chemical_symbols
    )

    # --- 4. Create a DataLoader ---
    # We use nequip's ASEDataModule to create a dataset and dataloader
    # This mimics how it's used in a real training run.
    data_module = ASEDataModule(
        train_file_path=str(train_xyz_path),
        val_file_path=str(val_xyz_path),
        seed=42
    )
    data_module.setup(stage="fit")
    dataloader = data_module.train_dataloader()
    
    # --- 5. Run the Statistics Computation ---
    logger.info("Calculating statistics... (this may take a moment)")
    stats_manager.reset() # Ensure state is clean
    final_stats = stats_manager.get_statistics(dataloader)

    # --- 6. Print the results ---
    logger.info("-" * 80)
    logger.info("Custom statistics calculation complete. Final Results:")
    for key, value in final_stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 80)
    
    # Clean up temporary file
    import shutil
    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main() 