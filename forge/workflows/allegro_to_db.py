#!/usr/bin/env python3
"""
Script: allegro_to_db.py
Description: Process Allegro MLIP experiment runs to compute dataset metrics and upload to a database.
"""
import argparse
import json
import os
import logging
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

try:
    import ase.io
except ImportError:
    ase = None

try:
    import wandb
except ImportError:
    wandb = None

# Try to import Forge DatabaseManager if available
try:
    from forge.core.database import DatabaseManager
except ImportError:
    DatabaseManager = None

def compute_dataset_hash(splits_path: Path) -> str:
    """Compute a hash for the dataset defined by structure_splits.json."""
    import hashlib
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    # Ensure deterministic ordering of data for hashing
    sorted_splits = {}
    for key in sorted(splits.keys()):
        val = splits[key]
        if isinstance(val, list):
            try:
                sorted_splits[key] = sorted(val)
            except Exception:
                sorted_splits[key] = val  # If not sortable (complex types), use as-is
        else:
            sorted_splits[key] = val
    # Stable string representation for hashing
    splits_str = json.dumps(sorted_splits, sort_keys=True)
    hash_val = hashlib.md5(splits_str.encode('utf-8')).hexdigest()
    return hash_val

def compute_avg_metrics(xyz_path: Path):
    """Compute average force magnitude and average stress magnitude from an .xyz file."""
    if ase is None:
        raise RuntimeError("ASE is required to parse .xyz files. Please install the 'ase' package.")
    sum_force = 0.0
    count_force = 0
    sum_stress = 0.0
    count_stress = 0
    # Iterate through all structures in the .xyz file
    for atoms in ase.io.iread(str(xyz_path), index=":"):
        if atoms is None:
            continue
        # Forces: compute per-atom force magnitudes
        try:
            if 'forces' in atoms.arrays:
                forces = atoms.get_array('forces')
            else:
                forces = atoms.get_array('REF_force')
        except Exception:
            forces = None
        if forces is not None:
            mags = np.linalg.norm(forces, axis=1)
            sum_force += mags.sum()
            count_force += mags.size
        # Stress: check if global stress is present in Atoms.info or virial
        stress = None
        if 'stress' in atoms.info:
            stress = atoms.info['stress']
        elif 'REF_stress' in atoms.info:
            stress = atoms.info['REF_stress']
        # elif 'virial' in atoms.info:
        #     # If virial is present instead of stress, we could derive stress norm from it
        #     stress = atoms.info['virial']
        if stress is not None:
            arr = np.array(stress)
            if arr.size in (3, 6) or arr.shape == (3, 3):
                s_norm = float(np.linalg.norm(arr))
            else:
                s_norm = None
            if s_norm is not None:
                sum_stress += s_norm
                count_stress += 1
    avg_force = sum_force / count_force if count_force > 0 else None
    avg_stress = sum_stress / count_stress if count_stress > 0 else None
    return avg_force, avg_stress


def find_one(pattern: str) -> Path | None:
    hits = glob(pattern)
    return Path(hits[0]) if hits else None

def process_run(run_dir: Path, wandb_runs: dict = None):
    run_name = run_dir.name
    splits_file = run_dir / "structure_splits.json"
    config_file = run_dir / "config.yaml"

    # flexible search for split files
    train_file = find_one(str(run_dir / "data" / "*train*.xyz"))
    val_file   = find_one(str(run_dir / "data" / "*val*.xyz"))
    test_file  = find_one(str(run_dir / "data" / "*test*.xyz"))

    required = [splits_file, config_file, train_file, val_file, test_file]
    if not all(required) or not all(f.exists() for f in required):
        logging.warning(f"Skipping run '{run_name}': required files missing "
                        f"(found train={bool(train_file)}, val={bool(val_file)}, "
                        f"test={bool(test_file)})")
        return None
    data = {"run_name": run_name}
    # Compute dataset hash
    try:
        data["dataset_hash"] = compute_dataset_hash(splits_file)
    except Exception as e:
        logging.error(f"Run '{run_name}': failed to compute dataset hash ({e}).")
        return None
    # Number of structures in each split (from structure_splits.json)
    try:
        with open(splits_file, 'r') as f:
            splits = json.load(f)
            data["n_train"] = len(splits.get("train", []))
            data["n_val"]   = len(splits.get("val", []))
            data["n_test"]  = len(splits.get("test", []))
    except Exception as e:
        logging.warning(f"Run '{run_name}': could not read structure_splits.json ({e}).")
        data["n_train"] = data["n_val"] = data["n_test"] = None
    # Compute dataset-level metrics for train, val, and test splits
    for subset, xyz_path in [("train", train_file), ("val", val_file), ("test", test_file)]:
        try:
            avg_f, avg_s = compute_avg_metrics(xyz_path)
        except Exception as e:
            logging.error(f"Run '{run_name}': error computing metrics for {subset} ({e}).")
            avg_f, avg_s = None, None
        data[f"avg_{subset}_force_mag"] = avg_f
        data[f"avg_{subset}_stress_mag"] = avg_s
    # Extract config and validation metrics from wandb if available
    if wandb_runs and run_name in wandb_runs:
        run = wandb_runs[run_name]
        # Training config hyperparameters (flat values only)
        try:
            for key, val in run.config.items():
                if isinstance(val, (int, float, str, bool)):
                    data[f"config_{key}"] = val
        except Exception as e:
            logging.warning(f"Run '{run_name}': error reading wandb config ({e}).")
        # Selected validation/test metrics from run summary
        summary = run.summary
        metric_keys = [
            # Validation (val0) metrics
            "val0_epoch/forces_mae", "val0_epoch/forces_rmse",
            "val0_epoch/stress_mae", "val0_epoch/stress_rmse",
            "val0_epoch/per_atom_energy_mae", "val0_epoch/per_atom_energy_rmse",
            "val0_epoch/total_energy_mae", "val0_epoch/total_energy_rmse",
            "val0_epoch/weighted_sum",
            # Secondary validation (val1) metrics if present
            "val1_epoch/forces_mae", "val1_epoch/forces_rmse",
            "val1_epoch/stress_mae", "val1_epoch/stress_rmse",
            "val1_epoch/per_atom_energy_mae", "val1_epoch/per_atom_energy_rmse",
            "val1_epoch/total_energy_mae", "val1_epoch/total_energy_rmse",
            "val1_epoch/weighted_sum",
            # Test set metrics
            "test0_epoch/forces_mae", "test0_epoch/forces_rmse",
            "test0_epoch/stress_mae", "test0_epoch/stress_rmse",
            "test0_epoch/per_atom_energy_mae", "test0_epoch/per_atom_energy_rmse",
            "test0_epoch/total_energy_mae", "test0_epoch/total_energy_rmse",
            "test0_epoch/weighted_sum"
        ]
        for mk in metric_keys:
            if mk in summary:
                # Friendly column name (e.g., "val_forces_rmse")
                col_name = mk.replace("0_epoch/", "_").replace("/", "_")
                data[col_name] = summary[mk]
    else:
        logging.debug(f"Run '{run_name}': no wandb data available.")
    return data

def main():
    parser = argparse.ArgumentParser(description="Process Allegro experiment folders and upload results to a database.")
    parser.add_argument("experiment_path", help="Path to the experiment folder (containing run subfolders).")
    parser.add_argument("--wandb-project", help="Weights & Biases project name for run data (optional).")
    parser.add_argument("--wandb-entity",  help="W&B entity (username/team) if required for the project.")
    parser.add_argument("--no-db", action="store_true", help="Skip database upload (only summarize results).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for debugging.")
    args = parser.parse_args()
    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    exp_path = Path(args.experiment_path)
    if not exp_path.is_dir():
        logging.error(f"Provided experiment_path '{exp_path}' is not a valid directory.")
        return
    experiment_name = exp_path.name
    # Initialize wandb API and fetch runs if project is specified
    wandb_runs = None
    if args.wandb_project:
        if wandb is None:
            logging.error("W&B project specified but wandb is not installed. Skipping wandb integration.")
        else:
            try:
                api = wandb.Api()
                project_path = args.wandb_project
                if args.wandb_entity:
                    project_path = f"{args.wandb_entity}/{args.wandb_project}"
                # Filter runs by group (experiment name) for efficiency, if group was used in W&B
                runs = api.runs(project_path, filters={"group": experiment_name})
                wandb_runs = {run.name: run for run in runs}
                logging.info(f"Retrieved {len(wandb_runs)} runs from W&B project '{args.wandb_project}'.")
            except Exception as e:
                logging.error(f"Failed to fetch runs from W&B: {e}")
    # Process each run folder within the experiment directory
    run_dirs = [d for d in exp_path.iterdir() if d.is_dir()]
    run_data_list = []
    failed_count = 0
    for run_dir in run_dirs:
        logging.info(f"Processing run directory: {run_dir.name}")
        result = process_run(run_dir, wandb_runs)
        if result is None:
            failed_count += 1  # run skipped due to errors
        else:
            run_data_list.append(result)
    # Compile results into a pandas DataFrame
    df = pd.DataFrame(run_data_list) if run_data_list else pd.DataFrame()
    total_runs = len(run_dirs)
    success_count = len(run_data_list)
    logging.info(f"Processing complete. {success_count} runs processed successfully, {failed_count} skipped.")
    if not df.empty:
        logging.debug("Sample of compiled run data:\n" + df.head().to_string(index=False))
    # Upload results to database if not skipped
    if not args.no_db:
        if DatabaseManager is None:
            logging.warning("DatabaseManager not available. Skipping database upload.")
        else:
            try:
                db = DatabaseManager()  # Assumes internal DB configuration is handled
                # TODO: Implement actual database insertion logic (e.g., using db connection or pd.to_sql)
                logging.info("Uploading run data to database table (placeholder implementation).")
            except Exception as e:
                logging.error(f"Database upload failed: {e}")
    else:
        logging.info("Database upload skipped by user request (--no-db).")

if __name__ == "__main__":
    main()
