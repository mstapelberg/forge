import logging
import json
from pathlib import Path
from itertools import product
import random
from typing import Any, Dict, List, Optional

from forge.core.database import DatabaseManager
from forge.workflows.db_to_allegro import prepare_allegro_job
from itertools import product

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_and_summarize(job_dir: Path, **kwargs: Any) -> None:
    """Prepare an Allegro job and log a concise summary.

    This helper delegates to the core job-preparation routine and prints a
    short status report including split sizes. If the job's data files already
    exist, the function returns early.

    Args:
        job_dir (Path): Target directory for the job. Data and config will be
            written under this directory.
        **kwargs (Any): Keyword arguments passed through to
            `prepare_allegro_job` (e.g., `job_name`, `structure_ids`, ratios,
            Allegro hyperparameters, and fixed `test_structure_ids`).

    Returns:
        None: This function performs I/O and logging side effects only.

    Raises:
        RuntimeError: If job preparation fails internally. The underlying
            exception is logged with traceback for easier debugging.

    Examples:
        >>> run_and_summarize(Path("./my_job"), job_name="exp1", structure_ids=[1,2,3],\
        ...                   train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)
    """
    # Check if data already exists
    train_xyz_path = job_dir / "data" / f"{kwargs['job_name']}_train.xyz"
    if train_xyz_path.exists():
        logger.info(f"Data for job '{kwargs['job_name']}' already exists. Skipping data preparation.")
        return

    logger.info("-" * 80)
    logger.info(f"Preparing job '{kwargs['job_name']}' in directory: {job_dir.resolve()}")

    # Use a fresh DB connection for each job preparation
    # Optional: fixed test set for generation 10
    test_structure_ids = []
    candidate_paths = [
        Path("./test_structure_ids_gen10.json"),
        Path("./scratch/scripts/dataset_analysis/test_structure_ids_gen10.json"),
        Path("./dataset_analysis/test_structure_ids_gen10.json"),
    ]
    for test_ids_file in candidate_paths:
        if test_ids_file.exists():
            try:
                test_structure_ids = [int(x) for x in json.load(test_ids_file.open())]
                logger.info(f"Loaded {len(test_structure_ids)} fixed test structure IDs from {test_ids_file}.")
                break
            except Exception as e:
                logger.warning(f"Failed to read test IDs from {test_ids_file}: {e}")

    with DatabaseManager() as db_manager:
        try:
            # Inject fixed test IDs if not provided via kwargs
            if test_structure_ids and not kwargs.get("test_structure_ids"):
                kwargs["test_structure_ids"] = test_structure_ids
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

def main() -> None:
    """Prepare a set of Allegro training jobs with varying q-filtering.

    The test set and random seed are held fixed across all jobs. For each
    specified q-folder, this script loads `bad_structure_ids.json` and excludes
    those IDs from the train/val candidate pool to study the impact of
    progressively stronger filtering on train and test performance.

    Returns:
        None
    """
    logger.info("Initializing database connection...")

    # Load multiple known sources of bad IDs and merge
    bad_id_files = [
        Path("./analysis_output_full/bad_structure_ids.json"),
        Path("./bad_structure_ids_gen8.json"),
        Path("./bad_structure_ids_gen9.json"),
        Path("./bad_structure_ids_gen10.json"),
        Path("../data/dataset_analysis_output_gen_8/bad_structure_ids.json"),
        Path("../data/dataset_analysis_output_gen_9/bad_structure_ids.json"),
        Path("../data/dataset_analysis_output_gen_10/bad_structure_ids.json"),
    ]
    bad_ids_set = set()
    for p in bad_id_files:
        try:
            if p.exists():
                ids = json.load(p.open())
                bad_ids_set.update(int(x) for x in ids)
                logger.info(f"Loaded {len(ids)} bad structure IDs from {p}.")
        except Exception as e:
            logger.warning(f"Failed to read bad IDs from {p}: {e}")
    logger.info(f"Total unique bad structure IDs to exclude: {len(bad_ids_set)}")

    with DatabaseManager() as db_manager:
        all_ids: List[int] = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
        dimer_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'dimer'})
        sr_dimer_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'short_range_dimer'})
        sr_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'short_range'})
        sr_aa_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'short_range_aa'})
        dimer_aa_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'dimer_aa'})

        # Base filtering: remove dimers/short-range and globally bad IDs
        base_candidate_ids: List[int] = [
            sid for sid in all_ids
            if sid not in dimer_ids
            and sid not in sr_dimer_ids
            and sid not in sr_ids
            and sid not in sr_aa_ids
            and sid not in dimer_aa_ids
            and sid not in bad_ids_set
        ]

        logger.info(f"Found {len(base_candidate_ids)} base-candidate structures after common filtering.")

    if not base_candidate_ids:
        logger.error("No structures found. Exiting.")
        return

    # Fixed test set: load once and pass explicitly to all runs
    fixed_test_structure_ids: List[int] = []
    test_id_candidates = [
        Path("./test_structure_ids_gen10.json"),
        Path("./scratch/scripts/dataset_analysis/test_structure_ids_gen10.json"),
        Path("./dataset_analysis/test_structure_ids_gen10.json"),
    ]
    for test_ids_file in test_id_candidates:
        if test_ids_file.exists():
            try:
                fixed_test_structure_ids = [int(x) for x in json.load(test_ids_file.open())]
                logger.info(f"Loaded {len(fixed_test_structure_ids)} fixed test IDs from {test_ids_file}.")
                break
            except Exception as e:
                logger.warning(f"Failed to read fixed test IDs from {test_ids_file}: {e}")

    if not fixed_test_structure_ids:
        logger.warning("Proceeding without an explicit fixed test set; test split will be random.")

    experiment_name = "allegro-exploit-gen-10-qfilter"
    base_dir = Path(f"../data/allegro_experiments/{experiment_name}")

    # Define base loss configurations
    tail_huber_delta1_loss = {
        "total_energy": {"coeff": 1.0, "metric": "mse"},
        "forces": {"coeff": 10.0, "metric": "tail_huber", "params": {"quantile": 0.75, "delta": 1.0}},
        "stress": {"coeff": 100.0, "metric": "mse"}
    }

    runs = [
        # Exploit baseline hyperparameters; we will vary only the q-filter
        {
            "phase": "exploit",
            "r_max": 5.50,
            "l_max": 1, # [1,2],
            "num_layers": 1, # [2,3],
            "mlp_width": 256,
            "loss_coeffs": tail_huber_delta1_loss,
        }
    ]

    for run_config in runs:
        phase = run_config["phase"]
        # Normalize hyperparameters to lists for Cartesian product
        r_max_values = run_config["r_max"] if isinstance(run_config["r_max"], list) else [run_config["r_max"]]
        l_max_values = run_config["l_max"] if isinstance(run_config["l_max"], list) else [run_config["l_max"]]
        num_layers_values = run_config["num_layers"] if isinstance(run_config["num_layers"], list) else [run_config["num_layers"]]
        mlp_width_values = run_config.get("mlp_width", 256)
        mlp_width_values = mlp_width_values if isinstance(mlp_width_values, list) else [mlp_width_values]

        final_loss_coeffs = run_config["loss_coeffs"]

        for r_max, l_max_val, num_layers_val, mlp_width_val in product(
            r_max_values, l_max_values, num_layers_values, mlp_width_values
        ):
            seed = 42

            # Vary q-folder filtering while keeping seed and test set fixed
            q_folders: List[str] = ["q-95", "q-97.5", "q-99", "q-99.5"]
            q_base_dir = Path("/home/myless/Packages/forge/scratch/data/ensemble_stats_runs/run_20250812_101647_merged_results")

            for q_label in q_folders:
                q_bad_path = q_base_dir / q_label / "bad_structure_ids.json"
                q_bad_ids: List[int] = []
                try:
                    if q_bad_path.exists():
                        q_bad_ids = [int(x) for x in json.load(q_bad_path.open())]
                        logger.info(f"Loaded {len(q_bad_ids)} q-filter bad IDs from {q_bad_path}.")
                    else:
                        logger.warning(f"q-filter file not found: {q_bad_path}")
                except Exception as e:
                    logger.warning(f"Failed to read q-filter bad IDs from {q_bad_path}: {e}")
                    q_bad_ids = []

                # Apply q-specific filtering to the base candidate pool (test set remains fixed)
                q_filtered_candidate_ids: List[int] = [sid for sid in base_candidate_ids if sid not in set(q_bad_ids)]

                job_name = (
                    f"{phase}_{q_label}_rmax{r_max:.2f}_lmax{l_max_val}_"
                    f"layers{num_layers_val}_mlp{mlp_width_val}_seed{seed}"
                )
                job_dir = base_dir / job_name

                use_delta_logger_flag = False
                if 'forces' in final_loss_coeffs and final_loss_coeffs['forces'].get('metric') == 'tail_huber':
                    if final_loss_coeffs['forces'].get('params', {}).get('delta') == 'auto':
                        use_delta_logger_flag = True

                params: Dict[str, Any] = {
                    "job_name": job_name,
                    "structure_ids": q_filtered_candidate_ids,
                    "train_ratio": 0.8,
                    "val_ratio": 0.1,
                    "test_ratio": 0.1,
                    "seed": seed,
                    "batch_size": 1,
                    "r_max": r_max,
                    "l_max": l_max_val,
                    "num_layers": num_layers_val,
                    "mlp_width": mlp_width_val,
                    "max_epochs": 50,
                    "wandb_project": experiment_name,
                    "checkpoint_monitor_key": "val0_epoch/weighted_sum",
                    "loss_coeffs": final_loss_coeffs,
                    "sampler": None,
                    "sampler_params": None,
                    "use_delta_logger": use_delta_logger_flag,
                    # Keep the test set fixed across q-variants
                    "test_structure_ids": fixed_test_structure_ids,
                }

                run_and_summarize(job_dir, **params)

if __name__ == "__main__":
    main() 