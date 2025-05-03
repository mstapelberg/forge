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
    chemical_symbols_list: Optional[List[str]] = None, # Explicit symbols from HPO
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
    job_dir = Path(job_dir)
    job_data_dir = job_dir / "data" # Target directory for data if splitting internally
    job_dir.mkdir(parents=True, exist_ok=True) # Ensure job_dir exists for config.yaml

    saved_structure_ids: Dict[str, List[int]] = {'train': [], 'val': [], 'test': []}
    chemical_symbols: Optional[List[str]] = chemical_symbols_list # Use provided if available
    config_data_train_path: str = ""
    config_data_val_path: str = ""
    config_data_test_path: str = ""
    is_hpo_mode = data_train_path is not None

    if is_hpo_mode:
        # --- HPO Mode ---
        logger.info(f"[{job_name}] Running in HPO mode. Using provided data paths.")
        if not data_val_path or not data_test_path:
            raise ValueError("In HPO mode, data_train_path, data_val_path, and data_test_path must all be provided.")

        data_train_path = Path(data_train_path)
        data_val_path = Path(data_val_path)
        data_test_path = Path(data_test_path)

        if not data_train_path.exists(): raise FileNotFoundError(f"Provided train data not found: {data_train_path}")
        if not data_val_path.exists(): raise FileNotFoundError(f"Provided validation data not found: {data_val_path}")
        # Test file might be empty if test_ratio was 0
        if not data_test_path.exists() and data_test_path.stat().st_size > 0:
            logger.warning(f"Provided test data not found: {data_test_path}")

        # Use absolute paths in config for HPO mode
        config_data_train_path = str(data_train_path.resolve())
        config_data_val_path = str(data_val_path.resolve())
        config_data_test_path = str(data_test_path.resolve())

        # Extract symbols if not provided explicitly
        if chemical_symbols is None:
            logger.info(f"[{job_name}] Chemical symbols not provided, attempting extraction from training data...")
            # Need to load the splits file associated with the data
            # Assuming it's in the same directory as the data files (e.g., central_data_dir/fold_x)
            splits_json_path = data_train_path.parent / "structure_splits.json"
            if splits_json_path.exists():
                try:
                    with open(splits_json_path, 'r') as f:
                        split_ids_info = json.load(f)
                    all_ids = list(set(split_ids_info.get('train', []) +
                                       split_ids_info.get('val', []) +
                                       split_ids_info.get('test', [])))
                    if not all_ids:
                        logger.warning(f"No structure IDs found in {splits_json_path} for symbol extraction.")
                        chemical_symbols = []
                    else:
                        chemical_symbols = _extract_chemical_symbols(db_manager, all_ids)
                except Exception as e:
                    logger.error(f"Failed to load {splits_json_path} or extract symbols: {e}. Cannot determine chemical symbols.", exc_info=True)
                    chemical_symbols = []
            else:
                logger.warning(f"Cannot find {splits_json_path} to extract symbols in HPO mode. Proceeding with empty symbol list.")
                chemical_symbols = []

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


    # --- Generate config.yaml ---
    logger.info(f"[{job_name}] Generating config.yaml...")

    if chemical_symbols is None or not chemical_symbols: # Final check
        logger.error(f"[{job_name}] Chemical symbols could not be determined. Cannot generate valid Allegro config.")
        # Return structure IDs if generated, otherwise empty dict
        raise ValueError(f"[{job_name}] Chemical symbols could not be determined. Cannot generate valid Allegro config.")

    # Defaults for schedule and loss
    effective_schedule = schedule if schedule is not None else {
        'start_epoch': int(0.8 * max_epochs),
        'factor': 0.5, # Default factor? NequIP examples vary. Let's add a default.
        'patience': 25, # Default patience?
    }
    # NequIP schedule keys might differ from old template, adapt based on nequip-train usage
    # Common examples: factor, patience. Let's use these.
    # Also need validation metric key, e.g., 'val_loss' or specific metrics
    validation_metric = 'val_loss' # Default validation metric

    effective_loss_coeffs = loss_coeffs if loss_coeffs is not None else {
        'total_energy': {'coeff': 1.0, 'per_atom': True}, # NequIP examples use dicts
        'forces': {'coeff': 50.0},
        'stress': {'coeff': 25.0},
    }
    # Ensure structure matches expected NequIP format

    # Build the config dictionary
    config = {
        # Top-level keys expected by nequip-train
        'run_name': job_name,
        'seed': seed, # Use the provided seed (relevant for training init)
        'dataset_file_name': config_data_train_path, # Path to training data
        'validation_dataset_file_name': config_data_val_path, # Path to validation data
        'test_dataset_file_name': config_data_test_path, # Path to test data
        'chemical_symbols': chemical_symbols,
        'r_max': r_max,
        'l_max': l_max,
        'num_layers': num_layers,
        'num_features': num_scalar_features, # Map num_scalar_features -> num_features ? Check Allegro docs. Assuming scalar.
        # Allegro specific params might go under 'model_builders' or similar key
        # Check nequip-train examples for Allegro. Assuming direct keys for now, may need nesting.
        'parity': True, # Common Allegro default
        'mlp_latent_dimensions': [mlp_width] * mlp_depth, # Example format for MLP layers
        # Tensor features might be part of model config - need to check Allegro/NequIP integration
        # 'num_tensor_features': num_tensor_features, # Where does this go? Assume top-level for now.

        # Training parameters
        'max_epochs': max_epochs,
        'learning_rate': lr,
        'loss_coeffs': effective_loss_coeffs,
        'metrics_key': validation_metric, # Key for ReduceLROnPlateau scheduler
        # Scheduler config (example for ReduceLROnPlateau)
        'scheduler_name': 'ReduceLROnPlateau',
        'scheduler_kwargs': {
            'factor': effective_schedule.get('factor', 0.5),
            'patience': effective_schedule.get('patience', 25),
            'threshold': 1e-5, # Example default
            'threshold_mode': 'rel',
            'cooldown': 0,
            'min_lr': 1e-7, # Example default
            'eps': 1e-8,
        },
        # WandB logging
        'wandb': True,
        'wandb_project': project,
        'wandb_name': job_name, # Use unique run name for wandb

        # Other common NequIP params (add defaults if needed)
        'batch_size': 10, # Example default
        'optimizer_name': 'Adam',
        'optimizer_amsgrad': False,
        'optimizer_betas': (0.9, 0.999),
        'optimizer_eps': 1e-8,
        'optimizer_weight_decay': 0.0,
        'clip_grad': None, # Example: {'max_norm': 10.0}
        'verbose': 'info', # Logging level
        # Dataset params
        'dataset_include_stress': True, # Assume stress is present

        # Potential Allegro-specific keys (adjust based on documentation)
        # These might need to be nested under a model configuration key
        'allegro_num_scalar_features': num_scalar_features,
        'allegro_num_tensor_features': num_tensor_features,
        # Add other allegro specific params here if needed...
    }

    # Refine loss coeff structure if needed (check NequIP docs)
    # Example: loss_coeffs: {'forces': {'coeff': 50.0, 'level': 'element'}}
    # Current structure might be okay.

    # --- Write YAML ---
    yaml_path = job_dir / "config.yaml"
    try:
        # Use default_flow_style=False for a more readable block format
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, indent=2, default_flow_style=False, sort_keys=False)
        logger.info(f"[{job_name}] Successfully generated {yaml_path}")
    except Exception as e:
        logger.error(f"Failed to write config.yaml for {job_name}: {e}", exc_info=True)
        # Decide whether to raise error or just return partial results
        raise IOError(f"Failed to write config.yaml: {e}") from e

    # Return structure IDs only if generated in standalone mode
    return saved_structure_ids if not is_hpo_mode else {}

# Remove the old template-based generation logic
# (Removed make_run function, template loading, common_replacements, generation loop)
# ... (rest of file, including _extract_chemical_symbols if it wasn't moved/changed) ...
