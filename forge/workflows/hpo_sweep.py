import json
import random
import math
import itertools
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

from forge.core.database import DatabaseManager
from forge.workflows.db_to_mace import (
    _get_vasp_structures,
    _save_structures_to_xyz,
    _replace_properties,
)
from forge.workflows.db_to_allegro import _extract_chemical_symbols

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_hpo_sweep(
    db_manager: DatabaseManager,
    model_type: str,
    base_sweep_dir: Union[str, Path],
    sweep_params: Dict[str, List[Any]],
    fixed_params: Dict[str, Any],
    num_seeds: int = 1,
    k_folds: Optional[int] = None,
    test_ratio: Optional[float] = 0.1,
    master_seed: int = 42,
):
    # ------------------------- validation ---------------------------
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")

    if model_type not in {"mace", "allegro"}:
        raise ValueError("model_type must be 'mace' or 'allegro'")

    if k_folds is not None and k_folds <= 1:
        k_folds = None
    if k_folds and not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be between 0 and 1 when k_folds > 1")

    base_sweep_dir = Path(base_sweep_dir).resolve()
    base_sweep_dir.mkdir(parents=True, exist_ok=True)
    central_data_dir = base_sweep_dir / "data"
    central_data_dir.mkdir(exist_ok=True)

    random.seed(master_seed)

    logger.info(f"Starting HPO sweep for **{model_type}** in {base_sweep_dir}")
    logger.info(f"Grid params:\n{json.dumps(sweep_params, indent=2)}")
    logger.info(f"Repetitions per point: {num_seeds}")
    logger.info(
        f"Cross-validation: {'%d-fold' % k_folds if k_folds else 'disabled'}"
    )

    # ---------------- structure-ID selection -----------------------
    if "structure_ids" not in fixed_params and "num_structures" not in fixed_params:
        raise ValueError(
            "fixed_params must contain 'structure_ids' or 'num_structures'"
        )
    if "structure_ids" in fixed_params and "num_structures" in fixed_params:
        logger.warning("Ignoring 'num_structures' in favour of provided IDs")
        fixed_params.pop("num_structures")

    if "structure_ids" in fixed_params:
        structure_ids: List[int] = fixed_params["structure_ids"]
        logger.info(f"Using {len(structure_ids)} pre-selected structure IDs")
    else:
        n = fixed_params["num_structures"]
        all_db_ids = _get_vasp_structures(db_manager)
        structure_ids = (
            all_db_ids if len(all_db_ids) <= n else random.sample(all_db_ids, n)
        )
        logger.info(f"Sampled {len(structure_ids)} structures from DB")

    if not structure_ids:
        raise ValueError("No structures available for the sweep")

    random.shuffle(structure_ids)
    total_structures = len(structure_ids)      # ── NEW ───────────────────

    # --------------- ratios for single-split mode -------------------
    if not k_folds:
        for key in ("train_ratio", "val_ratio"):
            if key not in fixed_params:
                raise ValueError(
                    f"{key} must be set in fixed_params when k_folds is None"
                )
        tr = fixed_params["train_ratio"]
        vr = fixed_params["val_ratio"]
        ter = (
            test_ratio
            if test_ratio is not None
            else fixed_params.get("test_ratio", None)
        )
        if ter is None:
            raise ValueError(
                "test_ratio (arg or fixed_params) required when k_folds is None"
            )
        if not math.isclose(tr + vr + ter, 1.0):
            s = tr + vr + ter
            tr, vr, ter = tr / s, vr / s, ter / s
            logger.warning(
                "train/val/test ratios did not sum to 1 – normalised"
            )
        effective_ratios = {"train": tr, "val": vr, "test": ter}

    # ---------------- symbol extraction once -----------------------
    all_symbols = _extract_chemical_symbols(db_manager, structure_ids)
    if not all_symbols:
        raise ValueError("Could not determine chemical symbols for dataset")
    logger.info(f"Symbols: {all_symbols}")

    # -----------------------------------------------------------------
    # 3.  Build central data splits   (unchanged block you pasted in)
    # -----------------------------------------------------------------
    # (your pasted code block starts here; it uses total_structures)
    # -----------------------------------------------------------------
    # --- 3. Prepare Central Data Splits ---
    central_data_paths: Dict[int, Dict[str, Path]] = {}

    if k_folds:
        logger.info(
            f"Preparing central data for {k_folds}-fold (test_ratio={test_ratio})"
        )
        test_size = math.ceil(total_structures * test_ratio)
        if test_size == 0 and total_structures > 0:
            logger.warning("Hold-out test set size computed as 0")
        if test_size >= total_structures:
            raise ValueError(
                f"test_ratio={test_ratio} leaves no data for training/val"
            )

        test_ids = structure_ids[:test_size]
        train_val_ids = structure_ids[test_size:]
        num_train_val = len(train_val_ids)

        if num_train_val < k_folds:
            raise ValueError(
                f"Only {num_train_val} structures for {k_folds} folds"
            )

        fold_size = num_train_val // k_folds
        extra = num_train_val % k_folds
        folds_ids: List[List[int]] = []
        start = 0
        for i in range(k_folds):
            end = start + fold_size + (1 if i < extra else 0)
            folds_ids.append(train_val_ids[start:end])
            start = end

        for fold_idx in range(k_folds):
            fold_dir = central_data_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_file = fold_dir / "train.xyz"
            val_file = fold_dir / "val.xyz"
            test_file = fold_dir / "test.xyz"
            splits_json = fold_dir / "structure_splits.json"

            central_data_paths[fold_idx] = {
                "train": train_file.resolve(),
                "val": val_file.resolve(),
                "test": test_file.resolve(),
            }

            if (
                train_file.exists()
                and val_file.exists()
                and (test_file.exists() or not test_ids)
                and splits_json.exists()
            ):
                logger.info(f"Fold {fold_idx} data exists – skipping write")
                continue

            val_ids = folds_ids[fold_idx]
            train_ids = list(
                itertools.chain.from_iterable(
                    folds_ids[j] for j in range(k_folds) if j != fold_idx
                )
            )

            _save_structures_to_xyz(db_manager, train_ids, train_file)
            _save_structures_to_xyz(db_manager, val_ids, val_file)
            if test_ids:
                _save_structures_to_xyz(db_manager, test_ids, test_file)

            with open(splits_json, "w") as f:
                json.dump(
                    {"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2
                )

    else:
        logger.info("Preparing central data for single split")
        split_dir = central_data_dir / "all"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_file = split_dir / "train.xyz"
        val_file = split_dir / "val.xyz"
        test_file = split_dir / "test.xyz"
        splits_json = split_dir / "structure_splits.json"

        central_data_paths[-1] = {
            "train": train_file.resolve(),
            "val": val_file.resolve(),
            "test": test_file.resolve(),
        }

        if (
            train_file.exists()
            and val_file.exists()
            and test_file.exists()
            and splits_json.exists()
        ):
            logger.info("Single-split data exists – skipping write")
        else:
            tr = effective_ratios["train"]
            vr = effective_ratios["val"]
            ter = effective_ratios["test"]

            n_train = math.floor(total_structures * tr)
            n_val = math.floor(total_structures * vr)
            n_test = total_structures - n_train - n_val

            train_ids = structure_ids[:n_train]
            val_ids = structure_ids[n_train : n_train + n_val]
            test_ids = structure_ids[n_train + n_val :]

            _save_structures_to_xyz(db_manager, train_ids, train_file)
            _save_structures_to_xyz(db_manager, val_ids, val_file)
            _save_structures_to_xyz(db_manager, test_ids, test_file)

            with open(splits_json, "w") as f:
                json.dump(
                    {"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2
                )

    # -----------------------------------------------------------------
    # 4. Hyper-parameter grid
    # -----------------------------------------------------------------
    param_names = list(sweep_params)
    hpo_combinations = list(itertools.product(*sweep_params.values()))
    logger.info(f"Grid has {len(hpo_combinations)} points.")

    # -----------------------------------------------------------------
    # 5. Iterate over grid / folds / seeds
    # -----------------------------------------------------------------
    job_details_list: List[Dict[str, Any]] = []
    task_id_counter = 0

    # pick the right prepare_* func lazily (avoids circular imports)
    if model_type == "mace":
        from forge.workflows.db_to_mace import prepare_mace_job as prepare_job_func
    else:
        from forge.workflows.db_to_allegro import prepare_allegro_job as prepare_job_func

    # sanity-check: unknown keys in fixed_params that WILL reach prepare_*()
    _KNOWN = {
        "seed",
        "data_train_path",
        "data_val_path",
        "data_test_path",
        "chemical_symbols_list",
        # add every real argument accepted by both prepare_* functions
        "max_epochs",
        "schedule",
        "project",
        "loss_coeffs",
        "lr",
        "r_max",
        "l_max",
        "num_layers",
        "num_scalar_features",
        "num_tensor_features",
        "mlp_depth",
        "mlp_width",
    }
    unknown = set(fixed_params) - _KNOWN
    if unknown:
        logger.warning(f"These keys are not used by prepare_job: {unknown}")  #  <<< FIX

    for combo_idx, combo_vals in enumerate(hpo_combinations):
        current_hpo_params = dict(zip(param_names, combo_vals))
        # nicer directory name (trimmed hash if too long)
        dir_stub = "_".join(f"{k}-{v}" for k, v in current_hpo_params.items())
        h = hashlib.md5(dir_stub.encode()).hexdigest()[:6] if len(dir_stub) > 90 else ""
        combo_dir = f"hpo_{combo_idx:03d}_{h or dir_stub}"
        combo_path = base_sweep_dir / combo_dir

        logger.info(f"[{combo_idx+1}/{len(hpo_combinations)}] {combo_dir}")

        n_folds = k_folds or 1
        for fold in range(n_folds):
            data_paths = central_data_paths.get(fold if k_folds else -1)
            if not data_paths:
                logger.error(f"Missing central data for fold {fold}")
                continue

            for seed_ix in range(num_seeds):
                run_seed = master_seed + fold * num_seeds + seed_ix
                run_suffix = (
                    (f"_fold{fold}" if k_folds else "_all") + f"_seed{seed_ix}"
                )
                run_name = combo_dir + run_suffix
                run_dir = combo_path / f"run{run_suffix}"
                logger.debug(f"Preparing {run_name}")

                run_params = {
                    **fixed_params,
                    **current_hpo_params,
                    "seed": run_seed,
                    "data_train_path": str(data_paths["train"]),
                    "data_val_path": str(data_paths["val"]),
                    "data_test_path": str(data_paths["test"]),
                    "chemical_symbols_list": all_symbols,
                }

                # strip keys that the prep func should not get
                for k in (
                    "structure_ids",
                    "num_structures",
                    "train_ratio",
                    "val_ratio",
                    "test_ratio",
                    "external_data_source_dir",
                    "slurm_job_name",
                    "slurm_partition",
                    "slurm_time",
                    "slurm_mem",
                    "slurm_cpus_per_gpu",
                    "slurm_output",
                    "slurm_error",
                    "num_nodes",
                    "gpu_config",  #  <<< FIX  (remove before calling)
                ):
                    run_params.pop(k, None)

                try:
                    prepare_job_func(
                        db_manager=db_manager,
                        job_name=run_name,
                        job_dir=run_dir,
                        **run_params,
                    )
                    job_details_list.append(
                        {
                            "task_id": task_id_counter,
                            "job_dir": str(run_dir.resolve()),
                            "run_name": run_name,
                            "hpo_params": current_hpo_params,
                            "fold": fold if k_folds else -1,
                            "seed": run_seed,
                        }
                    )
                    task_id_counter += 1
                except Exception as exc:
                    logger.error(
                        f"Failed to prepare job {run_name}: {exc}", exc_info=True
                    )

    if not job_details_list:
        raise RuntimeError("Prepared 0 jobs – nothing more to do.")

    # -----------------------------------------------------------------
    # 6. Mapping JSON
    # -----------------------------------------------------------------
    mapping_file = base_sweep_dir / "job_array_mapping.json"
    mapping_file.write_text(json.dumps(job_details_list, indent=2))
    logger.info(f"Wrote mapping to {mapping_file}")

    # -----------------------------------------------------------------
    # 7. Master SLURM array script
    # -----------------------------------------------------------------
    gpu_cfg = fixed_params.get("gpu_config", {"count": 1, "type": None})
    num_gpus = gpu_cfg.get("count", 1)
    gpu_type = gpu_cfg.get("type")
    gres = f"#SBATCH --gres=gpu:{num_gpus}\n" if num_gpus else ""
    constraint = f"#SBATCH --constraint={gpu_type}\n" if gpu_type else ""

    slurm_out_dir = (base_sweep_dir / "slurm_out").resolve()
    slurm_err_dir = (base_sweep_dir / "slurm_err").resolve()
    slurm_out_dir.mkdir(exist_ok=True)
    slurm_err_dir.mkdir(exist_ok=True)

    train_cmd = (
        "srun nequip-train -cn config.yaml"
        if model_type == "allegro"
        else "srun python -u run_train.py --config=config.yaml"
    )

    script = f"""#!/bin/bash
#SBATCH --job-name={fixed_params.get('slurm_job_name', model_type+'_hpo')}
#SBATCH --partition={fixed_params.get('slurm_partition','regular')}
#SBATCH --time={fixed_params.get('slurm_time','5-00:00:00')}
#SBATCH --nodes={fixed_params.get('num_nodes',1)}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={fixed_params.get('slurm_cpus_per_task',8)}
{gres}{constraint}#SBATCH --mem={fixed_params.get('slurm_mem','32G')}
#SBATCH --array=0-{task_id_counter-1}%32
#SBATCH --output={slurm_out_dir}/%A_%a.out
#SBATCH --error={slurm_err_dir}/%A_%a.err

set -euo pipefail
echo "Task $SLURM_ARRAY_TASK_ID starting on $(hostname) at $(date)"

# Purge module environment only if 'module' exists
if command -v module &>/dev/null; then
    module purge
fi

source /home/myless/.mambaforge/etc/profile.d/conda.sh
conda activate {"allegro-new" if model_type=="allegro" else "mace-cueq"}

JOB_DIR=$(python - "$SLURM_ARRAY_TASK_ID" <<PY
import json, sys, pathlib
tid = int(sys.argv[1])
mf = pathlib.Path("{mapping_file}").read_text()
jobs = json.loads(mf)
print(next(j["job_dir"] for j in jobs if j["task_id"] == tid))
PY
)

cd "$JOB_DIR"
echo "Running in $PWD"
{train_cmd}
echo "Finished at $(date)"
"""

    submit_script_path = base_sweep_dir / "submit_array_job.sh"
    submit_script_path.write_text(script)
    submit_script_path.chmod(0o755)
    logger.info(f"Wrote SLURM script to {submit_script_path}") 