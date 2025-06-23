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

# --- NEW: Import custom components for type hinting and path resolution ---
from forge.workflows.allegro_utils.callbacks import CurriculumCallback, GradNormCallback
from forge.workflows.allegro_utils.custom_losses import FocalMSELoss
from forge.workflows.allegro_utils.custom_metrics import TailMSE
from forge.workflows.allegro_utils.samplers import RareWeightedSampler
from forge.workflows.allegro_utils.data import CustomSamplingASEDataModule
# ---

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
            atoms_map = db_manager.get_structures_batch(structure_ids)
            atoms_list = list(atoms_map.values())
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
    # --- NEW: Arguments for custom training components ---
    loss_function: str = "mse", # 'mse' or 'focal' or 'huber'
    loss_params: Optional[Dict[str, Any]] = None,
    loss_schedule: Optional[Dict[int, Dict[str, float]]] = None, # For LossCoefficientScheduler
    sampler: Optional[str] = None, # 'rare_weighted'
    sampler_params: Optional[Dict[str, Any]] = None,
    extra_val_metrics: Optional[List[str]] = None, # 'tail_mse'
    extra_val_metric_params: Optional[Dict[str, Any]] = None,
    extra_trainer_params: Optional[Dict[str, Any]] = None,
    checkpoint_monitor_key: str = "val0_epoch/stress_rmse", # Metric to monitor for saving checkpoints
    # --- Allegro Hyperparameters (used in config.yaml) ---
    max_epochs: int = 400,
    batch_size: int = 4,
    wandb_project: Optional[str] = None,
    loss_coeffs: Optional[Dict[str, float]] = None, # Loss coefficients
    lr: float = 0.001,
    r_max: float = 5.0,
    l_max: int = 2,
    num_layers: int = 2,
    num_scalar_features: int = 128,
    num_tensor_features: int = 64,
    mlp_depth: int = 2,
    mlp_width: int = 512,
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
        loss_function: The loss function to use ('mse', 'focal', or 'huber').
        loss_params: Parameters for the chosen loss function.
        loss_schedule: A dictionary defining epochs and new loss coefficients for the scheduler.
        sampler: The data sampler to use (e.g., 'rare_weighted').
        sampler_params: Parameters for the chosen sampler.
        extra_val_metrics: List of extra validation metrics to add (e.g., 'tail_mse').
        extra_val_metric_params: Parameters for the validation metrics.
        extra_trainer_params: Extra parameters to pass to the lightning.Trainer.
        checkpoint_monitor_key: The metric key for ModelCheckpoint to monitor.
        max_epochs: Training epochs.
        batch_size: DataLoader batch size.
        wandb_project: Name of the WandB project.
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

    # --- NEW: Load the base configuration from YAML ---
    base_config_path = Path(__file__).parent / "allegro_configs" / "base.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base configuration file not found at {base_config_path}")

    logger.info(f"[{job_name}] Loading base configuration from: {base_config_path}")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Override base config with provided arguments ---
    logger.info(f"[{job_name}] Overriding base config with job-specific parameters...")

    if not chemical_symbols:
        raise ValueError(f"[{job_name}] Could not determine chemical symbols.")

    # Update basic keys
    config['job_name'] = job_name
    config['seed'] = seed
    config['chemical_symbols'] = chemical_symbols
    config['model_type_names'] = chemical_symbols
    config['training_module']['model']['type_names'] = chemical_symbols
    config['data']['stats_manager']['type_names'] = chemical_symbols
    config['data']['transforms'][1]['chemical_symbols'] = chemical_symbols
    config['training_module']['model']['pair_potential']['chemical_species'] = chemical_symbols

    # Update data paths
    config['data']['train_file_path'] = config_data_train_path
    config['data']['val_file_path'] = config_data_val_path
    config['data']['test_file_path'] = config_data_test_path

    # Update model hyperparameters
    config['training_module']['model']['r_max'] = r_max
    config['cutoff_radius'] = r_max
    config['training_module']['model']['l_max'] = l_max
    config['training_module']['model']['num_layers'] = num_layers
    config['training_module']['model']['num_scalar_features'] = num_scalar_features
    config['training_module']['model']['num_tensor_features'] = num_tensor_features
    config['training_module']['model']['allegro_mlp_hidden_layers_depth'] = mlp_depth
    config['training_module']['model']['allegro_mlp_hidden_layers_width'] = mlp_width
    
    # Update trainer and optimizer parameters
    config['trainer']['max_epochs'] = max_epochs
    config['training_module']['optimizer']['lr'] = lr
    if wandb_project:
        config['wandb_name'] = wandb_project
        if 'logger' in config['trainer'] and 'project' in config['trainer']['logger']:
            config['trainer']['logger']['project'] = wandb_project
    if 'logger' in config['trainer'] and 'name' in config['trainer']['logger']:
        config['trainer']['logger']['name'] = job_name # Update WandB run name

    # --- NEW: Use loss coefficients from base config unless overridden ---
    if loss_coeffs:
        effective_loss_coeffs = loss_coeffs
    else:
        # Try to parse from the base config's metrics list
        try:
            parsed_coeffs = {}
            base_metrics = config.get('training_module', {}).get('loss', {}).get('metrics', [])
            for metric in base_metrics:
                field = metric.get('field', {})
                field_name = field if isinstance(field, str) else field.get('field')
                
                if 'energy' in field_name:
                    parsed_coeffs['total_energy'] = {'coeff': metric.get('coeff')}
                elif 'forces' in field_name:
                    parsed_coeffs['forces'] = {'coeff': metric.get('coeff')}
                elif 'stress' in field_name:
                    parsed_coeffs['stress'] = {'coeff': metric.get('coeff')}
            
            if 'total_energy' in parsed_coeffs or 'forces' in parsed_coeffs:
                 effective_loss_coeffs = parsed_coeffs
                 logger.info("Successfully parsed loss coefficients from base.yaml.")
            else:
                 raise ValueError("No valid coefficients found in base config")
        except (ValueError, TypeError, AttributeError):
            logger.warning("Could not parse loss coefficients from base.yaml, using default values.")
            effective_loss_coeffs = {"total_energy": {"coeff": 1.0}, "forces": {"coeff": 10.0}, "stress": {"coeff": 100.0}}

    # --- NEW: Dynamically build loss function configuration ---
    loss_metrics = []
    
    loss_function_map = {
        "mse": "nequip.train.MeanSquaredError",
        "focal": "forge.workflows.allegro_utils.custom_losses.FocalMSELoss",
        "huber": "nequip.train.HuberLoss",
        "stratified_huber": "nequip.train.StratifiedHuberForceLoss", # Add stratified huber
    }
    
    if loss_function not in loss_function_map:
        raise ValueError(f"Unsupported loss function '{loss_function}'. Available: {list(loss_function_map.keys())}")

    loss_target = loss_function_map[loss_function]
    
    # Energy
    energy_metric = {
        "name": f"per_atom_energy_{loss_function}",
        "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"},
        "metric": {"_target_": loss_target},
        "coeff": effective_loss_coeffs.get("total_energy", {}).get("coeff", 1.0),
    }
    if loss_params:
        energy_metric["metric"].update(loss_params)
    loss_metrics.append(energy_metric)
    
    # Forces
    forces_metric = {
        "name": f"forces_{loss_function}",
        "field": "forces",
        "metric": {"_target_": loss_target},
        "coeff": effective_loss_coeffs.get("forces", {}).get("coeff", 10.0),
    }
    if loss_params:
        forces_metric["metric"].update(loss_params)
    loss_metrics.append(forces_metric)

    # Stress (optional, only if coeff is provided)
    if "stress" in effective_loss_coeffs and effective_loss_coeffs.get("stress", {}).get("coeff", 0) > 0:
        stress_metric = {
            "name": f"stress_{loss_function}",
            "field": "stress",
            "metric": {"_target_": loss_target},
            "coeff": effective_loss_coeffs["stress"]["coeff"],
        }
        if loss_params:
            stress_metric["metric"].update(loss_params)
        loss_metrics.append(stress_metric)

    # --- NEW: Dynamically build validation metrics ---
    val_metrics = []
    # Standard metrics - now including both MAE and RMSE
    val_metrics.extend([
        # MAE
        {"name": "per_atom_energy_mae", "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"}, "metric": {"_target_": "nequip.train.MeanAbsoluteError"}},
        {"name": "forces_mae", "field": "forces", "metric": {"_target_": "nequip.train.MeanAbsoluteError"}},
        {"name": "stress_mae", "field": "stress", "metric": {"_target_": "nequip.train.MeanAbsoluteError"}, "ignore_nan": True},
        # RMSE - These will be used for the weighted sum
        {"name": "per_atom_energy_rmse", "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"}, "metric": {"_target_": "nequip.train.RootMeanSquaredError"}, "coeff": 1.0},
        {"name": "forces_rmse", "field": "forces", "metric": {"_target_": "nequip.train.RootMeanSquaredError"}, "coeff": 1.0},
        {"name": "stress_rmse", "field": "stress", "metric": {"_target_": "nequip.train.RootMeanSquaredError"}, "ignore_nan": True, "coeff": 1.0},
    ])
    # Extra metrics
    if extra_val_metrics:
        metric_map = {"tail_mse": "forge.workflows.allegro_utils.custom_metrics.TailMSE"}
        for metric_name in extra_val_metrics:
            if metric_name not in metric_map:
                raise ValueError(f"Unsupported validation metric '{metric_name}'.")
            
            metric_params = (extra_val_metric_params or {}).get(metric_name, {})
            
            # Add for forces
            val_metrics.append({
                "name": f"forces_{metric_name}",
                "field": "forces",
                "metric": {"_target_": metric_map[metric_name], **metric_params}
            })
            # Add for energy
            val_metrics.append({
                "name": f"per_atom_energy_{metric_name}",
                "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"},
                "metric": {"_target_": metric_map[metric_name], **metric_params}
            })

    # --- NEW: Dynamically build dataloader config ---
    train_dataloader_config = {
        "_target_": "torch.utils.data.DataLoader",
        "batch_size": batch_size
    }
    
    # --- NEW: Build sampler config separately ---
    sampler_config = None
    if sampler == 'rare_weighted':
        sampler_config = {
            "_target_": "forge.workflows.allegro_utils.samplers.RareWeightedSampler",
            **(sampler_params or {})
        }
    elif sampler is not None:
        raise ValueError(f"Unsupported sampler '{sampler}'.")

    # --- NEW: Build callbacks using the robust, supported scheduler ---
    # Ensure callbacks list exists in the config
    if 'callbacks' not in config.get('trainer', {}):
        config.setdefault('trainer', {})['callbacks'] = []
    
    # Find and update the ModelCheckpoint, or add a default one
    checkpoint_callback = next((cb for cb in config['trainer']['callbacks'] if 'ModelCheckpoint' in cb.get('_target_', '')), None)
    if checkpoint_callback:
        checkpoint_callback['dirpath'] = f"results/{job_name}"
        checkpoint_callback['monitor'] = checkpoint_monitor_key
    else:
        config['trainer']['callbacks'].append({
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint", 
            "dirpath": f"results/{job_name}", 
            "monitor": checkpoint_monitor_key,
            "save_last": True
        })

    # Add or update the LossCoefficientScheduler
    config['trainer']['callbacks'] = [cb for cb in config['trainer']['callbacks'] if 'LossCoefficientScheduler' not in cb.get('_target_', '')]
    if loss_schedule:
        config['trainer']['callbacks'].append({
            "_target_": "nequip.train.callbacks.LossCoefficientScheduler",
            "schedule": loss_schedule
        })

    # Add default monitoring callbacks if they don't already exist
    if not any('LearningRateMonitor' in cb.get('_target_', '') for cb in config['trainer']['callbacks']):
        config['trainer']['callbacks'].append({
            "_target_": "lightning.pytorch.callbacks.LearningRateMonitor",
            "logging_interval": "epoch",
        })
    if not any('LossCoefficientMonitor' in cb.get('_target_', '') for cb in config['trainer']['callbacks']):
        config['trainer']['callbacks'].append({
            "_target_": "nequip.train.callbacks.LossCoefficientMonitor",
            "frequency": 1,
            "interval": "epoch",
        })

    # --- Apply the dynamically generated sections to the config ---
    config['training_module']['loss'] = {"_target_": "nequip.train.MetricsManager", "metrics": loss_metrics}
    config['training_module']['val_metrics'] = {"_target_": "nequip.train.MetricsManager", "metrics": val_metrics}
    
    # Update dataloader configurations
    config['data']['train_dataloader'] = train_dataloader_config
    
    # Update val_dataloader batch_size if it exists
    if 'val_dataloader' in config['data'] and isinstance(config['data']['val_dataloader'], dict):
        config['data']['val_dataloader']['batch_size'] = batch_size
    
    # Fix test_dataloader interpolation if it references val_dataloader
    if 'test_dataloader' in config['data'] and config['data']['test_dataloader'] == '${data.val_dataloader}':
        # test_dataloader should reference val_dataloader, which still exists
        logger.debug(f"[{job_name}] test_dataloader correctly references val_dataloader")
        # No change needed - the reference is correct

    # --- NEW: Use the custom datamodule if a sampler is specified ---
    if sampler_config:
        logger.info(f"[{job_name}] Using custom sampler: {sampler}")
        logger.debug(f"[{job_name}] Sampler config: {sampler_config}")
        config['data']['_target_'] = "forge.workflows.allegro_utils.data.CustomSamplingASEDataModule"
        config['data']['sampler_config'] = sampler_config
        
        # Verify custom module is importable
        try:
            from forge.workflows.allegro_utils.data import CustomSamplingASEDataModule
            logger.debug(f"[{job_name}] Successfully imported CustomSamplingASEDataModule")
        except ImportError as e:
            logger.error(f"[{job_name}] Failed to import CustomSamplingASEDataModule: {e}")
            raise

    if extra_trainer_params:
        config['trainer'].update(extra_trainer_params)

    # --- Add debugging information ---
    logger.debug(f"[{job_name}] Final config data section keys: {list(config.get('data', {}).keys())}")
    if 'data' in config:
        for key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
            if key in config['data']:
                value = config['data'][key]
                if isinstance(value, dict):
                    logger.debug(f"[{job_name}] {key} is a dict with keys: {list(value.keys())}")
                else:
                    logger.debug(f"[{job_name}] {key} = {value}")
    
    # --- Write the final config.yaml ---
    yaml_path = job_dir / "config.yaml"
    try:
        with yaml_path.open("w") as f:
            # Use a custom dumper to handle complex objects if necessary, but default should be fine
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
        logger.info(f"[{job_name}] Wrote final config, built from base: {yaml_path}")
    except Exception as e:
        logger.error(f"Failed to write config.yaml: {e}", exc_info=True)
        raise

    return saved_structure_ids if not is_hpo_mode else {}

# Remove the old template-based generation logic
# (Removed make_run function, template loading, common_replacements, generation loop)
# ... (rest of file, including _extract_chemical_symbols if it wasn't moved/changed) ...
