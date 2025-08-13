"""Simplified Allegro job runner using precomputed filtered IDs.

This script mirrors the behavior of `run_allegro_gen_10_production.py` but avoids
reading any untracked or large data under `scratch/data/`. Instead, it loads the
final excluded structure ID list and reason annotations from the tracked files:

- `scratch/scripts/dataset_analysis/filtered_out_ids.json`
- `scratch/scripts/dataset_analysis/filtered_out_ids_with_reasons.json`

It then prepares Allegro jobs using the same database filtering and
hyperparameters as the production script, excluding all IDs from the loaded list
globally (including from the fixed test set if present). Per-job exclusion files
are also saved alongside the job folder for traceability.
"""

import logging
import json
from pathlib import Path
from itertools import product
from typing import Any, Dict, List, Optional, Set

from forge.core.database import DatabaseManager
from forge.workflows.db_to_allegro import prepare_allegro_job


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_and_summarize(job_dir: Path, **kwargs: Any) -> None:
    """Prepare an Allegro job and log a concise summary.

    Args:
        job_dir (Path): Target directory for the job.
        **kwargs (Any): Passed to `prepare_allegro_job`.

    Returns:
        None

    Examples:
        >>> run_and_summarize(Path("./my_job"), job_name="exp1", structure_ids=[1,2,3],\
        ...                   train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)
    """
    train_xyz_path = job_dir / "data" / f"{kwargs['job_name']}_train.xyz"
    if train_xyz_path.exists():
        logger.info(f"Data for job '{kwargs['job_name']}' already exists. Skipping data preparation.")
        return

    logger.info("-" * 80)
    logger.info(f"Preparing job '{kwargs['job_name']}' in directory: {job_dir.resolve()}")

    # Optional fixed test set (Gen-10)
    test_structure_ids: List[int] = []
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
    """Prepare Allegro jobs using precomputed filtered ID lists.

    Reads excluded IDs and their reasons from tracked JSON files under
    `scratch/scripts/dataset_analysis/`. Applies those exclusions to the
    database-derived candidate pool (and fixed test set), then builds jobs with
    the same hyperparameters as the production script.

    Returns:
        None
    """
    logger.info("Initializing database connection...")

    # Load precomputed exclusions from tracked files (prefer dataset_analysis/)
    filtered_ids_path = Path("dataset_analysis/filtered_out_ids.json")
    filtered_reasons_path = Path("dataset_analysis/filtered_out_ids_with_reasons.json")
    excluded_ids: Set[int] = set()
    reasons_payload: Optional[Dict[str, Any]] = None
    try:
        if filtered_ids_path.exists():
            excluded_ids = set(int(x) for x in json.load(filtered_ids_path.open()))
            logger.info(f"Loaded {len(excluded_ids)} excluded IDs from {filtered_ids_path}.")
        else:
            logger.warning(f"Excluded list file not found: {filtered_ids_path}")
        if filtered_reasons_path.exists():
            reasons_payload = json.load(filtered_reasons_path.open())
            logger.info(f"Loaded exclusion reasons from {filtered_reasons_path}.")
            # Merge IDs from reasons file to be robust if filtered_out_ids.json is stale
            try:
                reasons_ids = {int(entry.get('structure_id')) for entry in reasons_payload.get('excluded', []) if entry.get('structure_id') is not None}
                if reasons_ids:
                    before_merge = len(excluded_ids)
                    excluded_ids |= reasons_ids
                    logger.info(f"Merged {len(reasons_ids)} IDs from reasons file (excluded size: {before_merge} -> {len(excluded_ids)}).")
            except Exception as e_ids:
                logger.warning(f"Failed to merge IDs from reasons payload: {e_ids}")
        else:
            logger.warning(f"Exclusion reasons file not found: {filtered_reasons_path}")
    except Exception as e:
        logger.error(f"Failed to load filtered ID files: {e}")
        excluded_ids = set()
        reasons_payload = None

    with DatabaseManager() as db_manager:
        all_ids: List[int] = db_manager.find_structures_by_metadata({'generation': 0}, operator='>=')
        dimer_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'dimer'})
        sr_dimer_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'short_range_dimer'})
        sr_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'short_range'})
        sr_aa_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'short_range_aa'})
        dimer_aa_ids: List[int] = db_manager.find_structures_by_metadata({'config_type': 'dimer_aa'})

        base_candidate_ids: List[int] = [
            sid for sid in all_ids
            if sid not in dimer_ids
            and sid not in sr_dimer_ids
            and sid not in sr_ids
            and sid not in sr_aa_ids
            and sid not in dimer_aa_ids
        ]

        logger.info(f"Found {len(base_candidate_ids)} base-candidate structures after common filtering.")

    if not base_candidate_ids:
        logger.error("No structures found. Exiting.")
        return

    # Fixed test set
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

    # Apply precomputed exclusions globally
    if excluded_ids:
        before = len(base_candidate_ids)
        base_candidate_ids = [sid for sid in base_candidate_ids if sid not in excluded_ids]
        logger.info(f"Filtered base candidates by precomputed IDs: {before} -> {len(base_candidate_ids)} (removed {before - len(base_candidate_ids)})")

        if fixed_test_structure_ids:
            before_test = len(fixed_test_structure_ids)
            fixed_test_structure_ids = [sid for sid in fixed_test_structure_ids if sid not in excluded_ids]
            removed = before_test - len(fixed_test_structure_ids)
            if removed > 0:
                logger.info(f"Removed {removed} precomputed excluded IDs from the fixed test set.")

    experiment_name = "allegro-exploit-gen-10-production-simple"
    base_dir = Path(f"../data/allegro_experiments/{experiment_name}")

    tail_huber_delta1_loss = {
        "total_energy": {"coeff": 1.0, "metric": "mse"},
        "forces": {"coeff": 10.0, "metric": "tail_huber", "params": {"quantile": 0.75, "delta": 1.0}},
        "stress": {"coeff": 100.0, "metric": "mse"},
    }

    runs = [
        {
            "phase": "exploit",
            "r_max": 5.50,
            "l_max": [1, 2],
            "num_layers": [1, 2],
            "mlp_width": 256,
            "loss_coeffs": tail_huber_delta1_loss,
        }
    ]

    for run_config in runs:
        phase = run_config["phase"]
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

            # Keep the same naming convention and q label as production; no q-filtering here
            q_label = "q-97.5"

            q_filtered_candidate_ids: List[int] = [sid for sid in base_candidate_ids if sid not in excluded_ids]
            # Safety check: ensure no excluded IDs remain
            leaked = sorted(list(set(q_filtered_candidate_ids) & set(excluded_ids)))
            if leaked:
                logger.warning(f"Found {len(leaked)} excluded IDs still present in candidates; first few: {leaked[:20]}")

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
                "wandb_project": "allegro-exploit-gen-10-production",
                "checkpoint_monitor_key": "val0_epoch/weighted_sum",
                "loss_coeffs": final_loss_coeffs,
                "sampler": None,
                "sampler_params": None,
                "use_delta_logger": use_delta_logger_flag,
                "test_structure_ids": fixed_test_structure_ids,
            }

            # Persist per-job excluded IDs next to the job folder
            try:
                job_dir.mkdir(parents=True, exist_ok=True)
                job_excluded_ids = sorted(list(excluded_ids))
                with (job_dir / 'filtered_out_ids.json').open('w') as f:
                    json.dump(job_excluded_ids, f, indent=2)
                if reasons_payload is not None:
                    with (job_dir / 'filtered_out_ids_with_reasons.json').open('w') as f:
                        json.dump(reasons_payload, f, indent=2)
            except Exception as e_save:
                logger.warning(f"Failed to save per-job filtered IDs for {job_name}: {e_save}")

            run_and_summarize(job_dir, **params)


if __name__ == "__main__":
    main()


