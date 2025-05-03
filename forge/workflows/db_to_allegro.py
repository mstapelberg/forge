# forge/workflows/db_to_allegro.py
import os
import json
import random
import math
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Union, Any
import logging # Add logging
import yaml # <-- Add PyYAML import

from ase.data import atomic_numbers
from forge.core.database import DatabaseManager

# Reuse existing functions for splitting and saving
from forge.workflows.db_to_mace import (
    _get_vasp_structures,
    _prepare_structure_splits,
    _save_structures_to_xyz,
    _replace_properties,
    # Remove _create_symlink_if_needed as it's no longer used here
)

# Configure logging
logger = logging.getLogger(__name__)

# Remove GPUConfig as it's no longer used
# class GPUConfig(TypedDict):
#     count: int
#     type: str


def _extract_chemical_symbols(
    db_manager: DatabaseManager,
    structure_ids: List[int]
) -> List[str]:
    """
    Batch-fetch all atoms, collect unique chemical symbols, and sort by atomic number.
    Returns empty list if structure_ids is empty.
    """
    if not structure_ids:
        return []
    # logger.debug(f"Extracting symbols for {len(structure_ids)} IDs") # Verbose
    try:
        # Try getting atoms with calculation first
        atoms_list = db_manager.get_batch_atoms_with_calculation(
            structure_ids, calculator='vasp'
        )
    except Exception as e_calc:
        logger.warning(f"Failed getting atoms with calculation for symbol extraction: {e_calc}. Trying without calc.")
        try:
            # Fallback to getting just atoms if calc retrieval fails
            atoms_list = db_manager.get_batch_atoms(structure_ids)
        except Exception as e_atoms:
            logger.error(f"Failed getting atoms even without calculation: {e_atoms}")
            return [] # Cannot determine symbols

    syms = set()
    processed_ids = set()
    for atoms in atoms_list:
        # Check if atoms object is valid
        if hasattr(atoms, 'get_chemical_symbols') and hasattr(atoms, 'info'):
            struct_id = atoms.info.get('structure_id', 'unknown')
            processed_ids.add(struct_id)
            try:
                syms.update(atoms.get_chemical_symbols())
            except Exception as e:
                logger.warning(f"Failed to get symbols for structure {struct_id}: {e}")
        else:
            logger.warning("Invalid object received instead of ASE Atoms in symbol extraction.")

    # Check if all requested IDs were processed
    requested_ids = set(structure_ids)
    missing_ids = requested_ids - processed_ids
    if missing_ids:
        logger.warning(f"Could not retrieve atom objects for {len(missing_ids)} structure IDs during symbol extraction (e.g., {list(missing_ids)[:5]})")

    if not syms:
        logger.warning("No chemical symbols found for the provided structure IDs.")
        return []

    # Sort symbols by atomic number
    try:
        sorted_syms = sorted(list(syms), key=lambda s: atomic_numbers[s])
        # logger.debug(f"Found symbols: {sorted_syms}") # Verbose
        return sorted_syms
    except KeyError as e:
        logger.error(f"Unknown chemical symbol encountered: {e}. Cannot sort symbols.")
        return list(syms)
    except Exception as e:
        logger.error(f"Unexpected error sorting symbols: {e}")
        return list(syms)

def prepare_allegro_job(
    db_manager: DatabaseManager,
    job_name: str,
    job_dir: Union[str, Path],
    # --- Arguments for HPO/Pre-split Mode ---
    data_train_path: Optional[Union[str, Path]] = None,
    data_val_path: Optional[Union[str, Path]] = None,
    data_test_path: Optional[Union[str, Path]] = None,
    chemical_symbols_list: Optional[List[str]] = None, # <-- Will be provided by HPO sweep
    # --- Arguments for Standalone Mode (ignored if data paths provided) ---
    seed: int = 0,
    num_structures: Optional[int] = None,
    structure_ids: Optional[List[int]] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    # --- Allegro Hyperparameters (used in config.yaml) ---
    max_epochs: int = 1000,
    schedule: Optional[Dict[str, float]] = None, # Validation schedule overrides
    project: str = "allegro-forge", # WandB project
    loss_coeffs: Optional[Dict[str, float]] = None, # Loss coefficients
    lr: float = 0.001,
    r_max: float = 5.0,
    l_max: int = 1,
    num_layers: int = 2,
    num_scalar_features: int = 128,
    num_tensor_features: int = 32,
    mlp_depth: int = 2,
    mlp_width: int = 128,
    # devices: Optional[int] = None, # Devices determined by runner/SLURM
    # num_nodes: int = 1, # Nodes determined by runner/SLURM
    # --- Removed Parameters ---
    # gpu_config removed (handled by runner/SLURM)
    # num_ensemble removed (handled by HPO script)
    # base_name removed (unused)
    # external_data_source_dir removed
) -> Dict[str, List[int]]:
    """
    Prepare Allegro training config (config.yaml) from database or pre-split data.

    Operates in two modes:
    1. HPO Mode: If `data_train_path` is provided, uses the given absolute paths
       and `chemical_symbols_list` (if provided) to generate `config.yaml`.
       Ignores `seed`, `num_structures`, `structure_ids`, `*_ratio`.
    2. Standalone Mode: If `data_train_path` is None, performs structure
       selection, splitting, saving to `job_dir/data/`, and symbol extraction.
       Uses relative paths in `config.yaml`. Requires `num_structures` or
       `structure_ids`, and `*_ratio`.

    Args:
        db_manager: DatabaseManager instance.
        job_name: Unique name for this specific run (used for filenames if splitting).
        job_dir: Directory for this specific run (config.yaml is saved here).
        data_train_path: Absolute path to pre-generated training data (HPO mode).
        data_val_path: Absolute path to pre-generated validation data (HPO mode).
        data_test_path: Absolute path to pre-generated test data (HPO mode).
        chemical_symbols_list: List of chemical symbols (optional in HPO mode).
        seed: Random seed for standalone splitting/selection.
        num_structures: Number of structures to select (standalone mode).
        structure_ids: List of structure IDs to use (standalone mode).
        train_ratio: Training fraction (standalone mode).
        val_ratio: Validation fraction (standalone mode).
        test_ratio: Testing fraction (standalone mode).
        max_epochs: Training epochs.
        schedule: Validation schedule overrides.
        project: WandB project name.
        loss_coeffs: Loss coefficients.
        lr: Learning rate.
        r_max: Cutoff radius.
        l_max: Max angular momentum.
        num_layers: Number of layers.
        num_scalar_features: Scalar feature dimension.
        num_tensor_features: Tensor feature dimension.
        mlp_depth: MLP depth.
        mlp_width: MLP width.

    Returns:
        Dict mapping 'train', 'val', 'test' to lists of structure_ids used.
        Returns IDs from `structure_splits.json` if run in standalone mode.
        Returns empty dict if run in HPO mode (IDs are handled by HPO script).

    Raises:
        ValueError: If invalid arguments are provided for the chosen mode.
        FileNotFoundError: If data files/dirs are missing in HPO mode.
    """
    logger.debug(f"[{job_name}] Entered prepare_allegro_job")
    job_dir = Path(job_dir)
    job_data_dir = job_dir / "data" # Target directory for data if splitting internally
    job_dir.mkdir(parents=True, exist_ok=True) # Ensure job_dir exists for config.yaml

    saved_structure_ids: Dict[str, List[int]] = {'train': [], 'val': [], 'test': []}
    chemical_symbols: Optional[List[str]] = chemical_symbols_list # Prioritize passed list
    config_data_train_path: str = ""
    config_data_val_path: str = ""
    config_data_test_path: str = ""
    is_hpo_mode = data_train_path is not None

    if is_hpo_mode:
        # --- HPO Mode ---
        logger.info(f"[{job_name}] Running in HPO mode. Using provided data paths.")
        logger.debug(f"[{job_name}] Provided paths: train='{data_train_path}', val='{data_val_path}', test='{data_test_path}'")
        if not data_val_path or not data_test_path:
            raise ValueError("In HPO mode, data_train_path, data_val_path, and data_test_path must all be provided.")

        data_train_path = Path(data_train_path)
        data_val_path = Path(data_val_path)
        data_test_path = Path(data_test_path)

        # Log absolute paths
        abs_train_path = data_train_path.resolve()
        abs_val_path = data_val_path.resolve()
        abs_test_path = data_test_path.resolve()
        logger.debug(f"[{job_name}] Resolved absolute paths: train='{abs_train_path}', val='{abs_val_path}', test='{abs_test_path}'")

        if not data_train_path.exists(): raise FileNotFoundError(f"Provided train data not found: {data_train_path}")
        if not data_val_path.exists(): raise FileNotFoundError(f"Provided validation data not found: {data_val_path}")
        # Test file might be empty if test_ratio was 0
        if not data_test_path.exists() and data_test_path.stat().st_size > 0:
            logger.warning(f"Provided test data not found: {data_test_path}")

        # Use absolute paths in config for HPO mode
        config_data_train_path = str(abs_train_path)
        config_data_val_path = str(abs_val_path)
        config_data_test_path = str(abs_test_path)

        # --- MODIFIED: Symbol Handling ---
        if chemical_symbols is None or not chemical_symbols: # Check if symbols were NOT passed
            logger.warning(f"[{job_name}] Chemical symbols not provided by caller. Attempting extraction from training data splits file (less efficient)...")
            # Fallback to original logic (less efficient)
            abs_train_path = Path(data_train_path).resolve()
            splits_json_path = abs_train_path.parent / "structure_splits.json"
            logger.debug(f"[{job_name}] Looking for splits file at: {splits_json_path}")
            if splits_json_path.exists():
                logger.debug(f"[{job_name}] Found splits file.")
                try:
                    with open(splits_json_path, 'r') as f:
                        split_ids_info = json.load(f)
                    logger.debug(f"[{job_name}] Successfully loaded splits JSON.")
                    all_ids = list(set(split_ids_info.get('train', []) +
                                       split_ids_info.get('val', []) +
                                       split_ids_info.get('test', [])))
                    logger.debug(f"[{job_name}] Extracted {len(all_ids)} unique IDs from splits file.")
                    if not all_ids:
                        logger.warning(f"No structure IDs found in {splits_json_path} for symbol extraction.")
                        chemical_symbols = []
                    else:
                        logger.debug(f"[{job_name}] Calling _extract_chemical_symbols (Fallback)...")
                        chemical_symbols = _extract_chemical_symbols(db_manager, all_ids)
                        logger.debug(f"[{job_name}] _extract_chemical_symbols returned: {chemical_symbols}")
                except Exception as e:
                    logger.error(f"Failed to load {splits_json_path} or extract symbols: {e}. Cannot determine chemical symbols.", exc_info=True)
                    chemical_symbols = [] # Set empty on error
            else:
                logger.warning(f"Cannot find splits file at {splits_json_path} to extract symbols in HPO mode fallback. Proceeding with empty symbol list.")
                chemical_symbols = []
            # --- End of Fallback Logic ---
        else:
            logger.info(f"[{job_name}] Using chemical symbols provided by caller: {chemical_symbols}")
        # --- End of MODIFIED Symbol Handling ---

    else:
        # --- Standalone Mode ---
        logger.info(f"[{job_name}] Running in Standalone mode. Preparing data in {job_dir}.")
        job_data_dir.mkdir(parents=True, exist_ok=True) # Ensure data subdir exists

        # 1) Validate input for splitting
        if structure_ids is not None and num_structures is not None:
            raise ValueError("Cannot specify both structure_ids and num_structures")
        if structure_ids is None and num_structures is None:
            raise ValueError("Must specify either structure_ids or num_structures in standalone mode.")
        if train_ratio is None or val_ratio is None or test_ratio is None:
            raise ValueError("train_ratio, val_ratio, and test_ratio must be provided in standalone mode.")
        if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
            # Allow slight deviation, but normalize if needed (like in HPO script)
             logger.warning(f"Provided train/val/test ratios sum to {train_ratio + val_ratio + test_ratio}. Normalizing.")
             total_ratio = train_ratio + val_ratio + test_ratio
             train_ratio /= total_ratio
             val_ratio /= total_ratio
             test_ratio /= total_ratio

        # 2) Fetch or randomly sample structure IDs
        if structure_ids:
            final_ids = structure_ids
        else:
            assert num_structures is not None # Help type checker
            logger.info(f"Fetching up to {num_structures} structures...")
            all_db_ids = _get_vasp_structures(db_manager)
            if len(all_db_ids) < num_structures:
                 logger.warning(f"Requested {num_structures} structures, but only {len(all_db_ids)} found. Using all available.")
                 final_ids = all_db_ids
            else:
                 random.seed(seed)
                 final_ids = random.sample(all_db_ids, num_structures)
            logger.info(f"Selected {len(final_ids)} structures.")

        if not final_ids:
             raise ValueError("No structures selected for standalone run. Cannot proceed.")
        all_used_ids = final_ids

        # 3) Determine chemical symbols from the dataset
        if chemical_symbols is None: # Only calculate if not already provided (unlikely in standalone)
             chemical_symbols = _extract_chemical_symbols(db_manager, all_used_ids)

        # 4) Split structures and write .xyz via db_to_mace helper
        # Saves xyz into job_data_dir (using job_name as prefix) and json into job_dir
        saved_structure_ids = _prepare_structure_splits(
            db_manager=db_manager,
            structure_ids=final_ids,
            job_name=job_name, # Allegro uses job_name as data prefix
            job_dir=job_dir, # Use run dir for json file
            data_dir=job_data_dir, # Use run data dir for xyz files
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
        # Note: _prepare_structure_splits now returns the *intended* splits,
        # need to check if files actually exist for path setting.

        # Define relative paths for config.yaml
        train_file_rel = f"data/{job_name}_train.xyz"
        val_file_rel = f"data/{job_name}_val.xyz"
        test_file_rel = f"data/{job_name}_test.xyz"

        # Check if files were created before setting paths
        if (job_dir / train_file_rel).exists():
            config_data_train_path = train_file_rel
        else:
            logger.error(f"Training file {train_file_rel} was not created successfully.")
            # Decide whether to raise error or allow config generation with missing path
            raise FileNotFoundError(f"Training file {train_file_rel} failed to generate.")

        if (job_dir / val_file_rel).exists():
            config_data_val_path = val_file_rel
        else:
             logger.error(f"Validation file {val_file_rel} was not created successfully.")
             raise FileNotFoundError(f"Validation file {val_file_rel} failed to generate.")

        if (job_dir / test_file_rel).exists():
             config_data_test_path = test_file_rel
        else:
             # Test set might be empty, don't raise error but log
             logger.warning(f"Test file {test_file_rel} was not created (might be intended if test_ratio was 0).")
             config_data_test_path = test_file_rel # Still add path to config

    # --- Generate config.yaml -------------------------------------------------
    logger.info(f"[{job_name}] Generating Hydra/Lightning-style config.yaml…")

    if not chemical_symbols:
        raise ValueError(f"[{job_name}] Could not determine chemical symbols.")

    # ----------  schedule / loss defaults (missing before)  -------------------
    effective_schedule = schedule if schedule is not None else {
        "factor": 0.5,   # learning-rate reduction factor
        "patience": 25,  # epochs with no improvement
        # optional; used below, otherwise computed as 0.8*max_epochs
        # "start_epoch": int(0.8 * max_epochs),
    }

    effective_loss_coeffs = loss_coeffs if loss_coeffs is not None else {
        "total_energy": {"coeff": 1.0, "per_atom": True},
        "forces":       {"coeff": 50.0},
        "stress":       {"coeff": 25.0},
    }
    # --------------------------------------------------------------------------

    # >>> HYDRA TEMPLATE BUILD  -------------------------------------------------
    data_prefix = job_name
    gpu_count   = 1
    num_nodes   = 1

    # schedule numbers
    sched_start_epoch = effective_schedule.get("start_epoch",
                                               int(0.8 * max_epochs))
    sched_energy = effective_schedule.get("factor", 0.5)
    sched_forces = sched_energy
    sched_stress = sched_energy

    # loss scalars
    loss_E = effective_loss_coeffs["total_energy"]["coeff"]
    loss_F = effective_loss_coeffs["forces"]["coeff"]
    loss_S = effective_loss_coeffs["stress"]["coeff"]

    template_cfg = {
        "run": ["val", "test", "train", "val", "test"],

        # ---------------- basic keys ----------------
        "cutoff_radius": r_max,
        "chemical_symbols": chemical_symbols,
        "model_type_names": chemical_symbols,
        "seed": seed,
        "job_name": job_name,

        # ---------------- data block ----------------
        "data": {
            "_target_": "nequip.data.datamodule.ASEDataModule",
            "train_file_path": config_data_train_path,
            "val_file_path":   config_data_val_path,
            "test_file_path":  config_data_test_path,
            "ase_args": {"format": "extxyz"},
            "key_mapping": {"REF_energy": "total_energy",
                            "REF_force":  "forces",
                            "REF_stress": "stress"},
            "transforms": [
                {"_target_": "nequip.data.transforms.NeighborListTransform",
                 "r_max": r_max},
                {"_target_":
                     "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                 "chemical_symbols": chemical_symbols},
            ],
            "seed": seed,
            # simple DataLoader settings
            "train_dataloader": {"_target_": "torch.utils.data.DataLoader",
                                 "batch_size": 4},
            "val_dataloader":   {"_target_": "torch.utils.data.DataLoader",
                                 "batch_size": 4},
            "test_dataloader":  "${data.val_dataloader}",
            "stats_manager": {
                "_target_": "nequip.data.CommonDataStatisticsManager",
                "type_names": chemical_symbols,
            },
        },

        # ---------------- trainer -------------------
        "trainer": {
            "_target_": "lightning.Trainer",
            "accelerator": "gpu",
            "devices": gpu_count,
            "num_nodes": num_nodes,
            "max_epochs": max_epochs,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 5,
            "callbacks": [
                {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "dirpath": f"results/{job_name}",
                    "save_last": True,
                },
                {
                    "_target_": "nequip.train.callbacks.LossCoefficientScheduler",
                    "schedule": {
                        sched_start_epoch: {
                            "per_atom_energy_mse": sched_energy,
                            "forces_mse":          sched_forces,
                            "stress_mse":          sched_stress,
                        }
                    },
                },
            ],
            "logger": {
                "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                "project": project,
                "name": job_name,
                "save_dir": "results",
            },
        },

        # --------------- training module -------------
        "training_module": {
            "_target_": "nequip.train.EMALightningModule",
            "loss": {
                "_target_": "nequip.train.EnergyForceStressLoss",
                "per_atom_energy": True,
                "coeffs": {
                    "total_energy": loss_E,
                    "forces":       loss_F,
                    "stress":       loss_S,
                },
            },
            "val_metrics": {
                "_target_": "nequip.train.EnergyForceStressMetrics",
                "coeffs": {
                    "per_atom_energy_mae": loss_E,
                    "forces_mae":          loss_F,
                    "stress_mae":          loss_S,
                },
            },
            "test_metrics": "${training_module.val_metrics}",

            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": lr,
            },

            # --------------- model --------------------
            "model": {
                "_target_": "allegro.model.AllegroModel",
                "seed": seed,
                "model_dtype": "float32",
                "type_names": chemical_symbols,
                "r_max": r_max,

                "scalar_embed": {
                    "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                    "num_bessels": 8,
                    "bessel_trainable": False,
                    "polynomial_cutoff_p": 6,
                    "two_body_embedding_dim": 32,
                    "two_body_mlp_hidden_layers_depth": 2,
                    "two_body_mlp_hidden_layers_width": 64,
                    "two_body_mlp_nonlinearity": "silu",
                },

                "l_max": l_max,
                "parity_setting": "o3_full",
                "num_layers": num_layers,
                "num_scalar_features": num_scalar_features,
                "num_tensor_features": num_tensor_features,
                "tp_path_channel_coupling": False,
                "allegro_mlp_hidden_layers_depth": mlp_depth,
                "allegro_mlp_hidden_layers_width": mlp_width,

                "avg_num_neighbors": "${training_data_stats:num_neighbors_mean}",
                "per_type_energy_shifts":
                    "${training_data_stats:per_atom_energy_mean}",
                "per_type_energy_scales":
                    "${training_data_stats:forces_rms}",
                "per_type_energy_scales_trainable": False,
                "per_type_energy_shifts_trainable": False,

                "pair_potential": {
                    "_target_": "nequip.nn.pair_potential.ZBL",
                    "units": "metal",
                    "chemical_species": chemical_symbols,
                },
            },
        },

        # ---------------- misc ----------------------
        "global_options": {"allow_tf32": False},
    }
    # -----------------------------------------------------------------
    # <<< HYDRA TEMPLATE BUILD  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # -----------------------------------------------------------------

    yaml_path = job_dir / "config.yaml"
    try:
        with yaml_path.open("w") as f:
            yaml.dump(template_cfg, f, sort_keys=False)
        logger.info(f"[{job_name}] Wrote Hydra-template config: {yaml_path}")
    except Exception as e:
        logger.error(f"Failed to write config.yaml: {e}", exc_info=True)
        raise

    # return split IDs only for standalone mode
    return saved_structure_ids if not is_hpo_mode else {}

# Remove the old template-based generation logic
# (Removed make_run function, template loading, common_replacements, generation loop)
# ... (rest of file, including _extract_chemical_symbols if it wasn't moved/changed) ...
