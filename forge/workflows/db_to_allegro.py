# forge/workflows/db_to_allegro.py
import os
import json
import random
import math
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Union, Any
import logging # Add logging

from ase.data import atomic_numbers
from forge.core.database import DatabaseManager

# Reuse existing functions for splitting and saving
from forge.workflows.db_to_mace import (
    _get_vasp_structures,
    _prepare_structure_splits,
    _save_structures_to_xyz,
    _replace_properties,
    _create_symlink_if_needed
)

# Configure logging
logger = logging.getLogger(__name__)

class GPUConfig(TypedDict):
    count: int
    type: str


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
    gpu_config: GPUConfig = {"count": 4, "type": "rtx6000"},
    seed: int = 0,
    num_structures: Optional[int] = None,
    structure_ids: Optional[List[int]] = None,
    train_ratio: Optional[float] = None, # Optional for pre-split
    val_ratio: Optional[float] = None,   # Optional for pre-split
    test_ratio: Optional[float] = None,  # Optional for pre-split
    max_epochs: int = 1000,
    schedule: Optional[Dict[str, float]] = None,
    project: str = "allegro-forge",
    save_dir: Optional[str] = None,
    loss_coeffs: Optional[Dict[str, float]] = None,
    lr: float = 0.001,
    r_max: float = 5.0,
    l_max: int = 1,
    num_layers: int = 2,
    num_scalar_features: int = 128,
    num_tensor_features: int = 32,
    mlp_depth: int = 2,
    mlp_width: int = 128,
    devices: Optional[int] = None,
    num_nodes: int = 1,
    num_ensemble: Optional[int] = None,
    base_name: Optional[str] = None, # Deprecated for Allegro?
    external_data_source_dir: Optional[Union[str, Path]] = None # New parameter
) -> Dict[str, List[int]]:
    """
    Prepare structure splits and generate Allegro YAML + SLURM scripts.
    Handles standard splitting, using pre-split data found in job_dir/data,
    or using externally prepared data (via symlinks).

    Args:
        db_manager: DatabaseManager instance.
        job_name: Unique name for this specific run.
        job_dir: Directory for this specific run.
        gpu_config: GPU config.
        seed: Random seed for splitting (if not pre-split) and training.
        num_structures: Number of structures to select (if not pre-split).
        structure_ids: List of structure IDs to use (if not pre-split).
        train_ratio: Training fraction (required if not pre-split).
        val_ratio: Validation fraction (required if not pre-split).
        test_ratio: Testing fraction (required if not pre-split).
        max_epochs: Training epochs.
        schedule: Annealing schedule.
        project: WandB project name.
        save_dir: Output directory within the run.
        loss_coeffs: Loss coefficients.
        lr: Learning rate.
        r_max: Cutoff radius.
        l_max: Max angular momentum.
        num_layers: Number of layers.
        num_scalar_features: Scalar feature dimension.
        num_tensor_features: Tensor feature dimension.
        mlp_depth: MLP depth.
        mlp_width: MLP width.
        devices: Number of devices (GPUs), defaults to gpu_config['count'].
        num_nodes: Number of nodes for SLURM.
        num_ensemble: Number of ensemble models to train (requires template support).
        base_name: (Currently unused by Allegro preparation, job_name determines file names)
        external_data_source_dir: Optional path to a directory containing pre-split
                                  'train.xyz', 'val.xyz', 'test.xyz', and
                                  'structure_splits.json'. If provided, symlinks
                                  will be created in job_dir/data.

    Returns:
        Dict mapping 'train', 'val', 'test' to lists of structure_ids used.
        Returns IDs from pre-split file if data exists, otherwise from new split.

    Raises:
        ValueError: If invalid arguments are provided.
        FileNotFoundError: If external_data_source_dir or its contents are missing.
    """
    job_dir = Path(job_dir)
    job_data_dir = job_dir / "data" # Target directory for data/symlinks
    job_data_dir.mkdir(parents=True, exist_ok=True)

    # Allegro templates use job_name directly for data paths (DATA_PREFIX)
    # effective_base_name = base_name if base_name is not None else job_name

    chemical_symbols: List[str] = []
    saved_structure_ids: Dict[str, List[int]] = {'train': [], 'val': [], 'test': []}
    all_used_ids: List[int] = [] # Keep track of all IDs used for symbol extraction
    data_prep_skipped = False
    run_names_to_generate = [job_name] # Default to single run
    if num_ensemble and num_ensemble > 1:
        run_names_to_generate = [f"{job_name}_ensemble_{idx}" for idx in range(num_ensemble)]

    # --- Data Handling Strategy ---
    if external_data_source_dir:
        # Strategy 1: Use externally provided data via symlinks
        external_data_path = Path(external_data_source_dir)
        logger.info(f"[{job_name}] Using external data source: {external_data_path}")
        if not external_data_path.is_dir():
            raise FileNotFoundError(f"External data source directory not found: {external_data_path}")

        external_splits_json = external_data_path / "structure_splits.json"
        if not external_splits_json.exists():
             raise FileNotFoundError(f"structure_splits.json not found in external source: {external_splits_json}")

        try:
            with open(external_splits_json, 'r') as f:
                saved_structure_ids = json.load(f)
            logger.info(f"[{job_name}] Loaded structure IDs from {external_splits_json}")
            all_used_ids = list(set(saved_structure_ids.get('train', []) +
                                  saved_structure_ids.get('val', []) +
                                  saved_structure_ids.get('test', [])))
            if not all_used_ids:
                logger.warning(f"[{job_name}] {external_splits_json} contained no structure IDs.")
            else:
                chemical_symbols = _extract_chemical_symbols(db_manager, all_used_ids)
        except Exception as e:
            raise IOError(f"Failed to read {external_splits_json} or extract symbols: {e}") from e

        # Create relative symlinks in job_data_dir for each run name expected
        # Allegro script expects files named {run_name}_{split}.xyz
        for run_name in run_names_to_generate:
            logger.info(f"  Ensuring symlinks exist for run: {run_name}")
            for split in ["train", "val", "test"]:
                src_file = external_data_path / f"{split}.xyz" # Generic name in source dir
                link_name = job_data_dir / f"{run_name}_{split}.xyz" # Name expected by script
                _create_symlink_if_needed(link_name, src_file)

        data_prep_skipped = True

    else:
        # Strategy 2: Check for pre-existing data in job_data_dir
        # Allegro uses job_name as prefix by default in template
        train_file = job_data_dir / f"{job_name}_train.xyz"
        val_file = job_data_dir / f"{job_name}_val.xyz"
        test_file = job_data_dir / f"{job_name}_test.xyz"
        splits_json_file = job_dir / "structure_splits.json" # JSON is always in job_dir

        if train_file.exists() and val_file.exists() and test_file.exists():
            logger.info(f"[{job_name}] Found existing data files (named {job_name}_...) in {job_data_dir}. Skipping data preparation.")
            data_prep_skipped = True
            if splits_json_file.exists():
                try:
                    with open(splits_json_file, 'r') as f:
                        saved_structure_ids = json.load(f)
                    logger.info(f"[{job_name}] Loaded structure IDs from {splits_json_file}")
                    # Still need symbols
                    all_used_ids = list(set(saved_structure_ids.get('train', []) +
                                          saved_structure_ids.get('val', []) +
                                          saved_structure_ids.get('test', [])))
                    if not all_used_ids:
                         logger.warning(f"[{job_name}] {splits_json_file} contained no IDs.")
                    else:
                        chemical_symbols = _extract_chemical_symbols(db_manager, all_used_ids)
                except Exception as e:
                    logger.error(f"Failed to load {splits_json_file} or extract symbols: {e}. Cannot proceed reliably.")
                    chemical_symbols = [] # Ensure it's empty list if failed
            else:
                logger.warning(f"Pre-split data files found, but {splits_json_file} is missing.")
                logger.warning("Cannot determine chemical symbols. Proceeding with empty symbol list.")
                chemical_symbols = []

    # Strategy 3: Perform splitting if needed
    if not data_prep_skipped:
        logger.info(f"[{job_name}] Data not found. Proceeding with standard data preparation.")
        # --- Original Data Prep Logic ---
        # 1) Validate input for splitting
        if structure_ids is not None and num_structures is not None:
            raise ValueError("Cannot specify both structure_ids and num_structures")
        if structure_ids is None and num_structures is None:
            raise ValueError("Must specify either structure_ids or num_structures when data is not pre-split and no external source is given.")
        if train_ratio is None or val_ratio is None or test_ratio is None:
            raise ValueError("train_ratio, val_ratio, and test_ratio must be provided when data is not pre-split and no external source is given.")
        if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
            raise ValueError("train_ratio, val_ratio, test_ratio must sum to ~1.0")

        # 2) Fetch or randomly sample structure IDs
        if structure_ids:
            final_ids = structure_ids
        else:
            assert num_structures is not None # Help type checker
            logger.info(f"Fetching up to {num_structures} structures...")
            all_db_ids = _get_vasp_structures(db_manager)
            if len(all_db_ids) < num_structures:
                raise ValueError(
                    f"Not enough structures ({len(all_db_ids)}) for requested {num_structures}"
                )
            random.seed(seed)
            final_ids = random.sample(all_db_ids, num_structures)
            logger.info(f"Selected {len(final_ids)} structures.")

        all_used_ids = final_ids # All selected IDs are used here

        # 3) Determine chemical symbols from the dataset
        if all_used_ids:
             chemical_symbols = _extract_chemical_symbols(db_manager, all_used_ids)
        else:
             logger.warning("No structure IDs selected, cannot determine chemical symbols.")
             chemical_symbols = []

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
        # --- End of Original Data Prep Logic ---

    # --- YAML and SLURM Script Generation ---

    logger.info(f"[{job_name}] Generating YAML/SLURM script(s)...")
    # Check if chemical symbols were determined (could fail in strategy 2)
    if not chemical_symbols:
        logger.error(f"[{job_name}] Chemical symbols could not be determined. Cannot generate valid Allegro config.")
        return saved_structure_ids # Return IDs, but indicate failure

    chem_yaml = '[' + ', '.join(f"'{s}'" for s in chemical_symbols) + ']'

    # 5) Defaults for schedule and loss
    effective_schedule = schedule if schedule is not None else {
        'start_epoch': int(0.8 * max_epochs),
        'per_atom_energy_mse': 10.0,
        'forces_mse': 1.0,
        'stress_mse': 0.25,
    }
    default_schedule_keys = ['start_epoch', 'per_atom_energy_mse', 'forces_mse', 'stress_mse']
    for key in default_schedule_keys:
        if key not in effective_schedule:
            raise ValueError(f"Missing key '{key}' in schedule for {job_name}")

    effective_loss_coeffs = loss_coeffs if loss_coeffs is not None else {
        'total_energy': 1.0, 'forces': 50.0, 'stress': 25.0
    }
    default_loss_keys = ['total_energy', 'forces', 'stress']
    for key in default_loss_keys:
         if key not in effective_loss_coeffs:
             raise ValueError(f"Missing key '{key}' in loss_coeffs for {job_name}")

    effective_save_dir = save_dir if save_dir is not None else f"results/{job_name}" # Base save dir on job_name
    effective_devices = devices if devices is not None else gpu_config['count']

    # 6) Prepare base replacements dict (will be customized in make_run)
    common_replacements: Dict[str, str] = {
        #'JOB_NAME': job_name, # Set in make_run
        #'DATA_PREFIX': job_name, # Set in make_run
        'R_MAX': str(r_max),
        'CHEMICAL_SYMBOLS': chem_yaml,
        #'SEED': str(seed), # Set in make_run
        'GPU_COUNT': str(effective_devices),
        'NUM_NODES': str(num_nodes),
        'MAX_EPOCHS': str(max_epochs),
        'START_EPOCH': str(effective_schedule['start_epoch']),
        'SCHEDULE_ENERGY': str(effective_schedule['per_atom_energy_mse']),
        'SCHEDULE_FORCES': str(effective_schedule['forces_mse']),
        'SCHEDULE_STRESS': str(effective_schedule['stress_mse']),
        'PROJECT': project,
        #'SAVE_DIR': effective_save_dir, # Set in make_run
        'LOSS_ENERGY': str(effective_loss_coeffs['total_energy']),
        'LOSS_FORCES': str(effective_loss_coeffs['forces']),
        'LOSS_STRESS': str(effective_loss_coeffs['stress']),
        'LR': str(lr),
        'L_MAX': str(l_max),
        'NUM_LAYERS': str(num_layers),
        'NUM_SCALAR': str(num_scalar_features),
        'NUM_TENSOR': str(num_tensor_features),
        'MLP_DEPTH': str(mlp_depth),
        'MLP_WIDTH': str(mlp_width),
        'GPU_TYPE': gpu_config['type'],
        #'TRAIN_FILE': f"data/{job_name}_train.xyz", # Set in make_run
        #'VAL_FILE':   f"data/{job_name}_val.xyz", # Set in make_run
        #'TEST_FILE':  f"data/{job_name}_test.xyz", # Set in make_run
    }

    # Template file paths
    yaml_tmpl = Path(__file__).parent / 'templates' / 'allegro_template.yaml'
    slurm_tmpl = Path(__file__).parent / 'templates' / 'allegro_slurm_template.sh'

    if not yaml_tmpl.exists(): raise FileNotFoundError(f"YAML Template missing: {yaml_tmpl}")
    if not slurm_tmpl.exists(): raise FileNotFoundError(f"SLURM Template missing: {slurm_tmpl}")

    # --- Generation Loop --- 

    def make_run(run_name: str, run_seed: int):
        logger.info(f"  Generating files for run: {run_name} (seed: {run_seed})")
        current_replacements = common_replacements.copy()
        # Set run-specific values
        current_replacements['JOB_NAME'] = run_name
        current_replacements['SEED'] = str(run_seed)
        current_replacements['SAVE_DIR'] = f"results/{run_name}"
        current_replacements['DATA_PREFIX'] = run_name
        current_replacements['TRAIN_FILE'] = f"data/{run_name}_train.xyz"
        current_replacements['VAL_FILE']   = f"data/{run_name}_val.xyz"
        current_replacements['TEST_FILE']  = f"data/{run_name}_test.xyz"

        # Write YAML from template
        try:
            txt = yaml_tmpl.read_text()
            for k, v in current_replacements.items():
                placeholder = f'${{{k}}}' # Template uses ${KEY} format
                txt = txt.replace(placeholder, v)

            yaml_path = job_dir / f"{run_name}.yaml"
            yaml_path.write_text(txt)
        except Exception as e:
            logger.error(f"Failed to generate YAML for {run_name}: {e}", exc_info=True)
            return # Skip slurm script if YAML failed

        # Write SLURM script
        try:
            sl = slurm_tmpl.read_text()
            slurm_repl = {
                'JOB_NAME': run_name,
                'GPU_COUNT': current_replacements['GPU_COUNT'],
                'GPU_TYPE': current_replacements['GPU_TYPE'],
                'NUM_NODES': current_replacements['NUM_NODES'],
                'YAML_FILE': f"{run_name}.yaml" # Use the generated YAML name
            }
            for k, v in slurm_repl.items():
                placeholder = f'${{{k}}}'
                sl = sl.replace(placeholder, v)

            script_path = job_dir / f"{run_name}.sh"
            script_path.write_text(sl)
            script_path.chmod(0o755)
        except Exception as e:
            logger.error(f"Failed to generate SLURM script for {run_name}: {e}", exc_info=True)

    # Emit jobs for all required run names
    logger.info(f"[{job_name}] Emitting generation tasks for: {run_names_to_generate}")
    if data_prep_skipped and len(run_names_to_generate) > 1:
         logger.warning("Ensemble members will use the same underlying data (via symlinks). Only training seed differs.")

    current_seed = seed # Start with the base seed provided
    for run_name in run_names_to_generate:
        make_run(run_name, current_seed)
        current_seed += 1 # Increment seed for next potential ensemble member

    return saved_structure_ids
