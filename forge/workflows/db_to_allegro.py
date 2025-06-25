# forge/workflows/db_to_allegro.py
import os
import json
import random
import math
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Union, Any
import logging
import yaml

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

# --- Import custom components for type hinting and path resolution ---
from forge.workflows.allegro_utils.callbacks import CurriculumCallback, GradNormCallback
from forge.workflows.allegro_utils.custom_losses import FocalMSELoss
from forge.workflows.allegro_utils.custom_metrics import TailMSE
from forge.workflows.allegro_utils.samplers import RareWeightedSampler
from forge.workflows.allegro_utils.data_v3 import CustomSamplingASEDataModuleV3
# ---

def _extract_chemical_symbols(
    db_manager: DatabaseManager,
    structure_ids: List[int]
) -> List[str]:
    """Extracts, sorts, and returns unique chemical symbols from a list of structures.

    This function fetches the ASE Atoms objects corresponding to the provided
    structure IDs from the database. It first attempts to retrieve them with

    associated VASP calculation data, falling back to fetching just the
    structures if that fails. It then gathers all unique chemical symbols
    from the structures and sorts them by their atomic number.

    Args:
        db_manager (DatabaseManager): An instance of the database manager to
            query for structures.
        structure_ids (List[int]): A list of structure IDs to process.

    Returns:
        List[str]: A sorted list of unique chemical symbols (e.g., ['Cr', 'Ti', 'V']).
            Returns an empty list if no structures are found or no symbols can be
            extracted.
    """
    if not structure_ids:
        return []
    try:
        atoms_list = db_manager.get_batch_atoms_with_calculation(
            structure_ids, calculator='vasp'
        )
    except Exception as e_calc:
        logger.warning(f"Failed getting atoms with calculation for symbol extraction: {e_calc}. Trying without calc.")
        try:
            atoms_map = db_manager.get_structures_batch(structure_ids)
            atoms_list = list(atoms_map.values())
        except Exception as e_atoms:
            logger.error(f"Failed getting atoms even without calculation: {e_atoms}")
            return []

    syms = set()
    for atoms in atoms_list:
        if hasattr(atoms, 'get_chemical_symbols'):
            syms.update(atoms.get_chemical_symbols())
        else:
            logger.warning("Invalid object received instead of ASE Atoms in symbol extraction.")

    if not syms:
        logger.warning("No chemical symbols found for the provided structure IDs.")
        return []

    try:
        return sorted(list(syms), key=lambda s: atomic_numbers[s])
    except KeyError as e:
        logger.error(f"Unknown chemical symbol encountered: {e}. Cannot sort symbols.")
        return list(syms)

def _prepare_data_for_allegro(
    db_manager: DatabaseManager,
    job_name: str,
    job_dir: Path,
    data_train_path: Optional[Union[str, Path]],
    data_val_path: Optional[Union[str, Path]],
    data_test_path: Optional[Union[str, Path]],
    chemical_symbols_list: Optional[List[str]],
    seed: int,
    num_structures: Optional[int],
    structure_ids: Optional[List[int]],
    train_ratio: Optional[float],
    val_ratio: Optional[float],
    test_ratio: Optional[float],
) -> Dict[str, Any]:
    """Prepares data for an Allegro job, supporting both HPO and standalone modes.

    In HPO (Hyper-Parameter Optimization) mode, triggered when `data_train_path`
    is provided, this function uses existing data files. It resolves their
    absolute paths and extracts chemical symbols if they are not provided.

    In standalone mode, it selects structures from the database based on
    `structure_ids` or `num_structures`, splits them into training, validation,
    and test sets, and saves them to new `.xyz` files within the job directory.

    Args:
        db_manager (DatabaseManager): The database manager for structure retrieval.
        job_name (str): The name of the job, used for naming output files.
        job_dir (Path): The directory for the job's output.
        data_train_path (Optional[Union[str, Path]]): Path to the training data.
            If provided, activates HPO mode.
        data_val_path (Optional[Union[str, Path]]): Path to the validation data.
        data_test_path (Optional[Union[str, Path]]): Path to the test data.
        chemical_symbols_list (Optional[List[str]]): A predefined list of chemical
            symbols. If None, they are extracted from the data.
        seed (int): The random seed for data splitting in standalone mode.
        num_structures (Optional[int]): The number of structures to sample from
            the database in standalone mode.
        structure_ids (Optional[List[int]]): Specific structure IDs to use in
            standalone mode.
        train_ratio (Optional[float]): The fraction of data for the training set.
        val_ratio (Optional[float]): The fraction of data for the validation set.
        test_ratio (Optional[float]): The fraction of data for the test set.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "train_path": Path to the training data file.
            - "val_path": Path to the validation data file.
            - "test_path": Path to the test data file.
            - "chemical_symbols": A list of unique, sorted chemical symbols.
            - "structure_splits": A dict with the train/val/test structure IDs.
                Empty in HPO mode.

    Raises:
        ValueError: If required arguments for a specific mode are missing.
        FileNotFoundError: If a provided data file in HPO mode does not exist.
    """
    is_hpo_mode = data_train_path is not None
    job_data_dir = job_dir / "data"
    
    if is_hpo_mode:
        logger.info(f"[{job_name}] Running in HPO mode. Using provided data paths.")
        if not data_val_path or not data_test_path:
            raise ValueError("In HPO mode, data_train_path, data_val_path, and data_test_path must all be provided.")

        train_path = Path(data_train_path).resolve()
        val_path = Path(data_val_path).resolve()
        test_path = Path(data_test_path).resolve()
        
        if not train_path.exists(): raise FileNotFoundError(f"Provided train data not found: {train_path}")
        if not val_path.exists(): raise FileNotFoundError(f"Provided validation data not found: {val_path}")
        if not test_path.exists() and test_path.stat().st_size > 0:
            logger.warning(f"Provided test data not found: {test_path}")

        if not chemical_symbols_list:
            logger.warning(f"[{job_name}] Chemical symbols not provided. Extracting from splits file...")
            splits_json_path = train_path.parent / "structure_splits.json"
            if splits_json_path.exists():
                with open(splits_json_path, 'r') as f:
                    split_ids_info = json.load(f)
                all_ids = list(set(split_ids_info.get('train', []) + split_ids_info.get('val', [])))
                chemical_symbols = _extract_chemical_symbols(db_manager, all_ids)
            else:
                logger.warning(f"Cannot find {splits_json_path}. Proceeding with empty symbol list.")
                chemical_symbols = []
        else:
            chemical_symbols = chemical_symbols_list

        return {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "chemical_symbols": chemical_symbols,
            "structure_splits": {},
        }
    else:
        logger.info(f"[{job_name}] Running in Standalone mode. Preparing data in {job_dir}.")
        job_data_dir.mkdir(parents=True, exist_ok=True)

        if structure_ids is None and num_structures is None:
            raise ValueError("Must specify either structure_ids or num_structures in standalone mode.")
        if train_ratio is None or val_ratio is None or test_ratio is None:
            raise ValueError("All ratios must be provided in standalone mode.")

        if structure_ids:
            final_ids = structure_ids
        else:
            all_db_ids = _get_vasp_structures(db_manager)
            num_to_sample = min(num_structures, len(all_db_ids))
            random.seed(seed)
            final_ids = random.sample(all_db_ids, num_to_sample)
        
        if not final_ids:
            raise ValueError("No structures selected for standalone run.")

        chemical_symbols = _extract_chemical_symbols(db_manager, final_ids)
        
        structure_splits = _prepare_structure_splits(
            db_manager, final_ids, job_name, job_dir, job_data_dir,
            train_ratio, val_ratio, test_ratio, seed
        )

        return {
            "train_path": f"data/{job_name}_train.xyz",
            "val_path": f"data/{job_name}_val.xyz",
            "test_path": f"data/{job_name}_test.xyz",
            "chemical_symbols": chemical_symbols,
            "structure_splits": structure_splits,
        }

def _build_loss_metrics(loss_function: str, loss_params: Optional[Dict[str, Any]], loss_coeffs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Constructs the loss metrics list for the Allegro configuration.

    This function dynamically builds the list of metrics that will be used for
    the loss function during training. It supports multiple loss types
    (e.g., 'mse', 'huber') and applies them to energy, forces, and optionally
    stress, each with a specific coefficient.

    Args:
        loss_function (str): The name of the loss function to use (e.g., 'mse').
        loss_params (Optional[Dict[str, Any]]): A dictionary of parameters for the
            loss function (e.g., `{'delta': 1.0}` for Huber loss).
        loss_coeffs (Optional[Dict[str, Any]]): A dictionary specifying the
            coefficients for 'total_energy', 'forces', and 'stress' losses.
            If None, default values are used.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            defines a component of the total loss function for the NequIP
            MetricsManager.

    Raises:
        ValueError: If an unsupported `loss_function` is provided.
    """
    loss_function_map = {
        "mse": "nequip.train.MeanSquaredError",
        "focal": "forge.workflows.allegro_utils.custom_losses.FocalMSELoss",
        "huber": "nequip.train.HuberLoss",
        "stratified_huber": "nequip.train.StratifiedHuberForceLoss",
        "weighted_mse": "forge.workflows.allegro_utils.weighted_loss.WeightedMSELoss",
    }
    if loss_function not in loss_function_map:
        raise ValueError(f"Unsupported loss function '{loss_function}'.")
    loss_target = loss_function_map[loss_function]
    
    effective_loss_coeffs = loss_coeffs or {"total_energy": {"coeff": 1.0}, "forces": {"coeff": 10.0}, "stress": {"coeff": 100.0}}
    
    metrics = []
    energy_metric = {
        "name": f"per_atom_energy_{loss_function}",
        "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"},
        "metric": {"_target_": loss_target, **(loss_params or {})},
        "coeff": effective_loss_coeffs.get("total_energy", {}).get("coeff", 1.0),
    }
    metrics.append(energy_metric)
    forces_metric = {
        "name": f"forces_{loss_function}",
        "field": "forces",
        "metric": {"_target_": loss_target, **(loss_params or {})},
        "coeff": effective_loss_coeffs.get("forces", {}).get("coeff", 10.0),
    }
    metrics.append(forces_metric)
    if "stress" in effective_loss_coeffs and effective_loss_coeffs.get("stress", {}).get("coeff", 0) > 0:
        stress_metric = {
            "name": f"stress_{loss_function}",
            "field": "stress",
            "metric": {"_target_": loss_target, **(loss_params or {})},
            "coeff": effective_loss_coeffs["stress"]["coeff"],
        }
        metrics.append(stress_metric)
    return metrics

def _build_validation_metrics(extra_val_metrics: Optional[List[str]], extra_val_metric_params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Constructs the validation metrics list for the Allegro configuration.

    This function starts with a default set of validation metrics (MAE and RMSE
    for energy, forces, and stress) and appends additional, optional metrics
    as specified.

    Args:
        extra_val_metrics (Optional[List[str]]): A list of names of extra
            validation metrics to include (e.g., 'tail_mse').
        extra_val_metric_params (Optional[Dict[str, Any]]): A dictionary of
            parameters for the extra validation metrics, keyed by the metric name.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries defining the validation
            metrics for the NequIP MetricsManager.
    """
    val_metrics = [
        {"name": "per_atom_energy_mae", "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"}, "metric": {"_target_": "nequip.train.MeanAbsoluteError"}},
        {"name": "forces_mae", "field": "forces", "metric": {"_target_": "nequip.train.MeanAbsoluteError"}},
        {"name": "stress_mae", "field": "stress", "metric": {"_target_": "nequip.train.MeanAbsoluteError"}, "ignore_nan": True},
        {"name": "per_atom_energy_rmse", "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"}, "metric": {"_target_": "nequip.train.RootMeanSquaredError"}, "coeff": 1.0},
        {"name": "forces_rmse", "field": "forces", "metric": {"_target_": "nequip.train.RootMeanSquaredError"}, "coeff": 1.0},
        {"name": "stress_rmse", "field": "stress", "metric": {"_target_": "nequip.train.RootMeanSquaredError"}, "ignore_nan": True, "coeff": 1.0},
    ]
    if extra_val_metrics:
        metric_map = {"tail_mse": "forge.workflows.allegro_utils.custom_metrics.TailMSE"}
        for metric_name in extra_val_metrics:
            if metric_name in metric_map:
                params = (extra_val_metric_params or {}).get(metric_name, {})
                val_metrics.append({"name": f"forces_{metric_name}", "field": "forces", "metric": {"_target_": metric_map[metric_name], **params}})
                val_metrics.append({"name": f"per_atom_energy_{metric_name}", "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"}, "metric": {"_target_": metric_map[metric_name], **params}})
    return val_metrics
    
def _update_trainer_callbacks(config: Dict[str, Any], job_name: str, checkpoint_monitor_key: str, loss_schedule: Optional[Dict[int, Dict[str, float]]]):
    """Updates or adds the necessary training callbacks to the configuration.

    This function ensures that the configuration's trainer section has the
    required callbacks for model checkpointing, learning rate monitoring, and
    loss coefficient monitoring. It also adds a loss coefficient scheduler if
    a schedule is provided.

    Args:
        config (Dict[str, Any]): The Allegro configuration dictionary, which will
            be modified in place.
        job_name (str): The name of the job, used to define the checkpoint directory.
        checkpoint_monitor_key (str): The validation metric to monitor for saving
            the best model checkpoint.
        loss_schedule (Optional[Dict[int, Dict[str, float]]]): An optional schedule
            for the LossCoefficientScheduler.
    """
    if 'callbacks' not in config.get('trainer', {}):
        config.setdefault('trainer', {})['callbacks'] = []

    checkpoint_callback = next((cb for cb in config['trainer']['callbacks'] if 'ModelCheckpoint' in cb.get('_target_', '')), None)
    if checkpoint_callback:
        checkpoint_callback['dirpath'] = f"results/{job_name}"
        checkpoint_callback['monitor'] = checkpoint_monitor_key
    else:
        config['trainer']['callbacks'].append({
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint", "dirpath": f"results/{job_name}", 
            "monitor": checkpoint_monitor_key, "save_last": True
        })

    config['trainer']['callbacks'] = [cb for cb in config['trainer']['callbacks'] if 'LossCoefficientScheduler' not in cb.get('_target_', '')]
    if loss_schedule:
        config['trainer']['callbacks'].append({
            "_target_": "nequip.train.callbacks.LossCoefficientScheduler", "schedule": loss_schedule
        })
    
    if not any('LearningRateMonitor' in cb.get('_target_', '') for cb in config['trainer']['callbacks']):
        config['trainer']['callbacks'].append({"_target_": "lightning.pytorch.callbacks.LearningRateMonitor", "logging_interval": "epoch"})
    if not any('LossCoefficientMonitor' in cb.get('_target_', '') for cb in config['trainer']['callbacks']):
        config['trainer']['callbacks'].append({"_target_": "nequip.train.callbacks.LossCoefficientMonitor", "frequency": 1, "interval": "epoch"})

def _build_allegro_config(
    job_name: str,
    seed: int,
    data_paths: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Builds the complete Allegro/NequIP configuration dictionary.

    This function loads a base configuration from a YAML file, then populates
    and overrides it with job-specific parameters. This includes setting data
    paths, chemical species, model hyperparameters, loss functions, metrics,
    and callbacks.

    Args:
        job_name (str): The unique name for the job.
        seed (int): The random seed for the training run.
        data_paths (Dict[str, Any]): A dictionary containing the paths to the
            train, validation, and test datasets, as well as the list of
            chemical symbols.
        **kwargs: A dictionary of keyword arguments containing all other
            hyperparameters and settings for the job (e.g., `r_max`, `l_max`,
            `batch_size`, `loss_function`).

    Returns:
        Dict[str, Any]: The fully-populated Allegro configuration dictionary.

    Raises:
        ValueError: If chemical symbols cannot be determined or if an
            unsupported sampler is requested.
        FileNotFoundError: If the `base.yaml` configuration file cannot be found.
    """
    base_config_path = Path(__file__).parent / "allegro_configs" / "base.yaml"
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    chemical_symbols = data_paths["chemical_symbols"]
    if not chemical_symbols:
        raise ValueError(f"[{job_name}] Could not determine chemical symbols.")

    config.update({
        'job_name': job_name, 'seed': seed, 'chemical_symbols': chemical_symbols,
        'model_type_names': chemical_symbols,
    })
    
    # Update nested keys that depend on chemical symbols
    config['training_module']['model']['type_names'] = chemical_symbols
    config['data']['stats_manager']['type_names'] = chemical_symbols
    config['data']['transforms'][1]['chemical_symbols'] = chemical_symbols
    config['training_module']['model']['pair_potential']['chemical_species'] = chemical_symbols
    
    # Update data paths
    config['data']['train_file_path'] = data_paths["train_path"]
    config['data']['val_file_path'] = data_paths["val_path"]
    config['data']['test_file_path'] = data_paths["test_path"]

    # Update hyperparameters from kwargs
    config['training_module']['model']['r_max'] = kwargs['r_max']
    config['cutoff_radius'] = kwargs['r_max']
    config['training_module']['model']['l_max'] = kwargs['l_max']
    config['training_module']['model']['num_layers'] = kwargs['num_layers']
    config['training_module']['model']['num_scalar_features'] = kwargs['num_scalar_features']
    config['training_module']['model']['num_tensor_features'] = kwargs['num_tensor_features']
    config['training_module']['model']['allegro_mlp_hidden_layers_depth'] = kwargs.get('mlp_depth', 2)
    config['training_module']['model']['allegro_mlp_hidden_layers_width'] = kwargs.get('mlp_width', 512)
    config['trainer']['max_epochs'] = kwargs['max_epochs']
    config['training_module']['optimizer']['lr'] = kwargs['lr']
    
    if kwargs.get('wandb_project'):
        config['wandb_name'] = kwargs['wandb_project']
        if 'logger' in config['trainer'] and 'project' in config['trainer']['logger']:
            config['trainer']['logger']['project'] = kwargs['wandb_project']
    if 'logger' in config['trainer'] and 'name' in config['trainer']['logger']:
        config['trainer']['logger']['name'] = job_name

    loss_metrics = _build_loss_metrics(
        kwargs['loss_function'], kwargs.get('loss_params'), kwargs.get('loss_coeffs')
    )
    config['training_module']['loss'] = {"_target_": "nequip.train.MetricsManager", "metrics": loss_metrics}

    val_metrics = _build_validation_metrics(
        kwargs.get('extra_val_metrics'), kwargs.get('extra_val_metric_params')
    )
    config['training_module']['val_metrics'] = {"_target_": "nequip.train.MetricsManager", "metrics": val_metrics}
    
    config['data']['train_dataloader'] = {
        "_target_": "torch.utils.data.DataLoader",
        "batch_size": kwargs['batch_size']
    }
    if 'val_dataloader' in config['data']:
        config['data']['val_dataloader']['batch_size'] = kwargs['batch_size']

    if kwargs.get('sampler'):
        sampler_map = {'rare_weighted': "forge.workflows.allegro_utils.samplers.RareWeightedSampler"}
        sampler_target = sampler_map.get(kwargs['sampler'])
        if not sampler_target:
            raise ValueError(f"Unsupported sampler '{kwargs['sampler']}'")
        config['data']['_target_'] = "forge.workflows.allegro_utils.data_v3.CustomSamplingASEDataModuleV3"
        config['data']['sampler_config'] = {
            "_target_": sampler_target,
            **(kwargs.get('sampler_params') or {})
        }
    
    _update_trainer_callbacks(
        config, job_name, kwargs['checkpoint_monitor_key'], kwargs.get('loss_schedule')
    )

    if kwargs.get('extra_trainer_params'):
        config['trainer'].update(kwargs['extra_trainer_params'])
        
    return config

def prepare_allegro_job(
    db_manager: DatabaseManager,
    job_name: str,
    job_dir: Union[str, Path],
    # HPO/Pre-split Mode Arguments
    data_train_path: Optional[Union[str, Path]] = None,
    data_val_path: Optional[Union[str, Path]] = None,
    data_test_path: Optional[Union[str, Path]] = None,
    chemical_symbols_list: Optional[List[str]] = None,
    # Standalone Mode Arguments
    seed: int = 0,
    num_structures: Optional[int] = None,
    structure_ids: Optional[List[int]] = None,
    train_ratio: Optional[float] = 0.8,
    val_ratio: Optional[float] = 0.1,
    test_ratio: Optional[float] = 0.1,
    # Custom Training Component Arguments
    loss_function: str = "mse",
    loss_params: Optional[Dict[str, Any]] = None,
    loss_schedule: Optional[Dict[int, Dict[str, float]]] = None,
    sampler: Optional[str] = None,
    sampler_params: Optional[Dict[str, Any]] = None,
    extra_val_metrics: Optional[List[str]] = None,
    extra_val_metric_params: Optional[Dict[str, Any]] = None,
    extra_trainer_params: Optional[Dict[str, Any]] = None,
    checkpoint_monitor_key: str = "val0_epoch/stress_rmse",
    # Allegro Hyperparameters
    max_epochs: int = 400,
    batch_size: int = 4,
    wandb_project: Optional[str] = None,
    loss_coeffs: Optional[Dict[str, Any]] = None,
    lr: float = 0.001,
    r_max: float = 5.0,
    l_max: int = 2,
    num_layers: int = 2,
    num_scalar_features: int = 128,
    num_tensor_features: int = 64,
    mlp_depth: int = 2,
    mlp_width: int = 512,
) -> Dict[str, List[int]]:
    """Prepares an Allegro training job by generating a complete `config.yaml`.

    This function serves as the main entry point for creating an Allegro job.
    It orchestrates the two main steps:
    1.  Data Preparation (`_prepare_data_for_allegro`): Handles the sourcing,
        splitting, and saving of training/validation/test datasets.
    2.  Configuration Building (`_build_allegro_config`): Constructs the
        final `config.yaml` from a base template and the specific parameters
        of the job.

    The function supports two primary modes of operation:
    -   **Standalone Mode**: When no data paths are provided, it fetches structures
        directly from the database.
    -   **HPO Mode**: When `data_train_path` is provided, it uses pre-existing
        data splits, which is typical for hyper-parameter optimization sweeps.

    Args:
        db_manager (DatabaseManager): An instance of the database manager.
        job_name (str): A unique name for the training job.
        job_dir (Union[str, Path]): The directory where the job's `config.yaml`
            and any generated data will be saved.
        data_train_path (Optional[Union[str, Path]]): Path to pre-split training
            data. Supplying this activates HPO mode.
        data_val_path (Optional[Union[str, Path]]): Path to pre-split validation data.
        data_test_path (Optional[Union[str, Path]]): Path to pre-split test data.
        chemical_symbols_list (Optional[List[str]]): A pre-defined list of
            chemical symbols.
        seed (int): Random seed for reproducibility.
        num_structures (Optional[int]): Number of structures to sample from the DB.
        structure_ids (Optional[List[int]]): Specific list of structure IDs to use.
        train_ratio (Optional[float]): Fraction of data for the training set.
        val_ratio (Optional[float]): Fraction of data for the validation set.
        test_ratio (Optional[float]): Fraction of data for the test set.
        loss_function (str): The loss function to use (e.g., 'mse', 'huber').
        loss_params (Optional[Dict[str, Any]]): Parameters for the loss function.
        loss_schedule (Optional[Dict[int, Dict[str, float]]]): Schedule for the
            LossCoefficientScheduler.
        sampler (Optional[str]): The data sampler to use (e.g., 'rare_weighted').
        sampler_params (Optional[Dict[str, Any]]): Parameters for the sampler.
        extra_val_metrics (Optional[List[str]]): Additional validation metrics.
        extra_val_metric_params (Optional[Dict[str, Any]]): Parameters for extra metrics.
        extra_trainer_params (Optional[Dict[str, Any]]): Extra parameters for the
            PyTorch Lightning Trainer.
        checkpoint_monitor_key (str): Metric to monitor for saving checkpoints.
        max_epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training and validation.
        wandb_project (Optional[str]): Name of the Weights & Biases project.
        loss_coeffs (Optional[Dict[str, Any]]): Coefficients for the loss terms.
        lr (float): Learning rate.
        r_max (float): Cutoff radius for atomic environments.
        l_max (int): Maximum angular momentum for spherical harmonics.
        num_layers (int): Number of interaction layers in the model.
        num_scalar_features (int): Dimension of scalar features.
        num_tensor_features (int): Dimension of tensor features.
        mlp_depth (int): Depth of the MLPs in the model.
        mlp_width (int): Width of the MLPs in the model.

    Returns:
        Dict[str, List[int]]: A dictionary mapping 'train', 'val', 'test' to lists
        of the structure IDs used in each set. Returns an empty dictionary in
        HPO mode, as the splits are external.
    """
    logger.debug(f"[{job_name}] Entered prepare_allegro_job")
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare data paths and symbols
    data_details = _prepare_data_for_allegro(
        db_manager=db_manager, job_name=job_name, job_dir=job_dir,
        data_train_path=data_train_path, data_val_path=data_val_path,
        data_test_path=data_test_path, chemical_symbols_list=chemical_symbols_list,
        seed=seed, num_structures=num_structures, structure_ids=structure_ids,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
    )
    
    # Explicitly gather all keyword arguments for the config builder
    # to avoid the `locals()` trap.
    config_kwargs = {
        'data_train_path': data_train_path, 'data_val_path': data_val_path,
        'data_test_path': data_test_path, 'chemical_symbols_list': chemical_symbols_list,
        'num_structures': num_structures, 'structure_ids': structure_ids,
        'train_ratio': train_ratio, 'val_ratio': val_ratio, 'test_ratio': test_ratio,
        'loss_function': loss_function, 'loss_params': loss_params,
        'loss_schedule': loss_schedule, 'sampler': sampler,
        'sampler_params': sampler_params, 'extra_val_metrics': extra_val_metrics,
        'extra_val_metric_params': extra_val_metric_params,
        'extra_trainer_params': extra_trainer_params,
        'checkpoint_monitor_key': checkpoint_monitor_key, 'max_epochs': max_epochs,
        'batch_size': batch_size, 'wandb_project': wandb_project,
        'loss_coeffs': loss_coeffs, 'lr': lr, 'r_max': r_max, 'l_max': l_max,
        'num_layers': num_layers, 'num_scalar_features': num_scalar_features,
        'num_tensor_features': num_tensor_features, 'mlp_depth': mlp_depth,
        'mlp_width': mlp_width,
    }

    # Step 2: Build the configuration dictionary
    config = _build_allegro_config(
        job_name=job_name, seed=seed, data_paths=data_details, **config_kwargs
    )

    # Step 3: Write the final config.yaml
    yaml_path = job_dir / "config.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    logger.info(f"[{job_name}] Wrote final config: {yaml_path}")

    return data_details.get("structure_splits", {})
