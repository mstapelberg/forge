import logging
from pathlib import Path

from forge.core.database import DatabaseManager
from forge.workflows.db_to_allegro import prepare_allegro_job

# Configure logging to see the output from the workflow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to prepare and set up an Allegro training job.
    """
    # --- 1. Initialize Database Connection ---
    logger.info("Initializing database connection...")
    db_manager = DatabaseManager()

    # --- 2. Query Structures for Training ---
    # Find all structures that have completed VASP calculations.
    # You can customize the query by changing the metadata filter.
    logger.info("Finding structures with completed VASP calculations...")
    structure_ids = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
    logger.info(f"Found {len(structure_ids)} structures with VASP calculations.")
    dimer_ids = db_manager.find_structures_by_metadata({'config_type': 'dimer'})
    logger.info(f"Found {len(dimer_ids)} structures with dimer config type that are bad.")
    # remove the dimer ids from the structure ids
    structure_ids = [sid for sid in structure_ids if sid not in dimer_ids] # remove the dimer ids from the structure ids

    # rmove the short_range_dimer ids from the structure ids too 
    short_range_dimer_ids = db_manager.find_structures_by_metadata({'config_type': 'short_range_dimer'})
    logger.info(f"Found {len(short_range_dimer_ids)} structures with short range dimer config type that are bad.")
    structure_ids = [sid for sid in structure_ids if sid not in short_range_dimer_ids]

    logger.info(f"Removing {len(dimer_ids) + len(short_range_dimer_ids)} structures with dimer/short_range_dimer config type that are bad. \n Now have {len(structure_ids)} structures with VASP calculations.")

    if not structure_ids:
        logger.error("No structures found. Exiting.")
        return

    # --- 3. Define Job Parameters ---
    # These parameters define the names and locations for the training job files.
    job_name = "allegro_gen_8_2025-06-20_no_dimers"
    base_job_dir = Path("../data/allegro_jobs")

    # To run an ensemble, you can loop over this script and change the seed.
    # For example, to create 3 models:
    # for i in range(3):
    #     seed = i * 100
    #     job_name = f"allegro_example_job_model_{i}"
    #     job_dir = base_job_dir / job_name
    #     ... call prepare_allegro_job ...
    
    seed = 0
    job_dir = base_job_dir / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preparing job '{job_name}' in directory: {job_dir.resolve()}")

    # --- 4. Prepare the Allegro Job ---
    # This function will:
    #   - Split the data into training, validation, and test sets.
    #   - Save the data as .xyz files in the job_dir/data directory.
    #   - Create a 'config.yaml' file required to run Allegro/NequIP.
    structure_splits = prepare_allegro_job(
        db_manager=db_manager,
        job_name=job_name,
        job_dir=job_dir,
        structure_ids=structure_ids,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
        # You can customize Allegro hyperparameters here
        r_max=5.0,
        l_max=2,
        num_layers=2,
        max_epochs=1000,
        lr=0.001,
        loss_coeffs={"total_energy": {"coeff": 1.0, "per_atom": True}, "forces": {"coeff": 10.0}, "stress": {"coeff": 25.0}},
    )

    # --- 5. Print Job Summary ---
    logger.info("\nAllegro Training Job Summary:")
    logger.info(f"  Job Name: {job_name}")
    logger.info(f"  Job Directory: {job_dir.resolve()}")
    logger.info("  Config file created: config.yaml")
    
    logger.info("\nData Split:")
    for split_name, struct_ids in structure_splits.items():
        logger.info(f"  - {split_name}: {len(struct_ids)} structures")

    data_dir = job_dir / "data"
    train_file = data_dir / f"{job_name}_train.xyz"
    val_file = data_dir / f"{job_name}_val.xyz"
    test_file = data_dir / f"{job_name}_test.xyz"

    logger.info("\nCreated data files:")
    if train_file.exists():
        logger.info(f"  - Training data: {train_file.resolve()}")
    if val_file.exists():
        logger.info(f"  - Validation data: {val_file.resolve()}")
    if test_file.exists():
        logger.info(f"  - Test data: {test_file.resolve()}")
        
    # --- 6. Instructions for Running the Training Job ---
    logger.info("\nTo run the Allegro training job, execute the following commands:")
    logger.info(f"cd {job_dir.resolve()}")
    logger.info("nequip-train config.yaml")
    logger.info("\nThis command assumes you have the NequIP/Allegro environment set up.")
    logger.info("The training will be managed by the settings within the 'config.yaml' file.")
    logger.info("For multi-GPU or multi-node training, you might need to adjust the 'config.yaml' or use a launcher like `torchrun`.")

if __name__ == "__main__":
    main() 