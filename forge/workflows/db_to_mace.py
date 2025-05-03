import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypedDict
from ase.io import write
import shutil
import math
import logging # Add logging
import yaml # <-- Add PyYAML import
from forge.core.database import DatabaseManager  # Add this import

# Configure logging
logger = logging.getLogger(__name__)

def prepare_mace_job(
    db_manager: DatabaseManager,
    job_name: str, # Unique name for this run, also used as base for file names
    job_dir: Union[str, Path],
    # --- Arguments for HPO/Pre-split Mode ---
    data_train_path: Optional[Union[str, Path]] = None,
    data_val_path: Optional[Union[str, Path]] = None,
    data_test_path: Optional[Union[str, Path]] = None,
    # --- Arguments for Standalone Mode (ignored if data paths provided) ---
    structure_ids: Optional[List[int]] = None,
    num_structures: Optional[int] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    # --- MACE Hyperparameters (used in config.yaml) ---
    e0s: str = "default", # E0s mode ("default", "average") or JSON string
    seed: int = 42,
    num_interactions: int = 2,
    num_channels: int = 128,
    max_L: int = 0,
    r_max: float = 5.0,
    lr: float = 0.001,
    forces_weight: float = 50.0,
    energy_weight: float = 1.0,
    stress_weight: float = 25.0,
    # --- Removed Parameters ---
    # gpu_config removed (handled by runner/SLURM)
    # num_ensemble removed (handled by HPO script)
    # base_name removed (using job_name)
    # external_data_source_dir removed
) -> Dict[str, List[int]]:
    """
    Prepare MACE training config (config.yaml) from database or pre-split data.

    Operates in two modes:
    1. HPO Mode: If `data_train_path` is provided, uses the given absolute paths
       to generate `config.yaml`. Ignores `seed`, `num_structures`,
       `structure_ids`, `*_ratio`.
    2. Standalone Mode: If `data_train_path` is None, performs structure
       selection, splitting, saving to `job_dir/data/`, and property replacement.
       Uses relative paths in `config.yaml`. Requires `num_structures` or
       `structure_ids`, and `*_ratio`.

    Args:
        db_manager: DatabaseManager instance for accessing structure data
        job_name: Name for the training job (used as unique identifier and file base name)
        job_dir: Directory to create job in (config.yaml is saved here).
        data_train_path: Absolute path to pre-generated training data (HPO mode).
        data_val_path: Absolute path to pre-generated validation data (HPO mode).
        data_test_path: Absolute path to pre-generated test data (HPO mode).
        structure_ids: Optional list of specific structure IDs to use (standalone mode).
        num_structures: Optional number of random structures to select (standalone mode).
        train_ratio: Fraction for training (standalone mode).
        val_ratio: Fraction for validation (standalone mode).
        test_ratio: Fraction for testing (standalone mode).
        e0s: "default", "average", or a JSON string representing the E0s dictionary.
        seed: Random seed for standalone structure selection/splitting and training.
        num_interactions: MACE num_interactions parameter.
        num_channels: MACE num_channels parameter.
        max_L: MACE max_L parameter.
        r_max: MACE r_max parameter.
        lr: MACE learning rate.
        forces_weight: MACE forces_weight parameter.
        energy_weight: MACE energy_weight parameter.
        stress_weight: MACE stress_weight parameter.

    Returns:
        Dict mapping 'train', 'val', 'test' to lists of structure_ids used.
        Returns IDs from `structure_splits.json` if run in standalone mode.
        Returns empty dict if run in HPO mode (IDs are handled by HPO script).

    Raises:
        ValueError: If invalid arguments are provided for the chosen mode.
        FileNotFoundError: If data files/dirs are missing in HPO mode.
        IOError: If config.yaml generation fails.
    """
    job_dir = Path(job_dir)
    job_data_dir = job_dir / "data" # Target directory for data if splitting internally
    job_dir.mkdir(parents=True, exist_ok=True) # Ensure job_dir exists for config.yaml

    # Use job_name as the base name for files generated in standalone mode
    file_base_name = job_name

    saved_structure_ids: Dict[str, List[int]] = {'train': [], 'val': [], 'test': []}
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
        if not data_test_path.exists() and data_test_path.stat().st_size > 0:
            logger.warning(f"Provided test data not found: {data_test_path}")

        # Use absolute paths in config for HPO mode
        config_data_train_path = str(data_train_path.resolve())
        config_data_val_path = str(data_val_path.resolve())
        config_data_test_path = str(data_test_path.resolve())

        # No data splitting or property replacement needed in HPO mode

    else:
        # --- Standalone Mode ---
        logger.info(f"[{job_name}] Running in Standalone mode. Preparing data in {job_dir}.")
        job_data_dir.mkdir(parents=True, exist_ok=True) # Ensure data subdir exists

        # Validate input arguments for splitting
        if train_ratio is None or val_ratio is None or test_ratio is None:
            raise ValueError("train_ratio, val_ratio, and test_ratio must be provided in standalone mode.")
        if not isinstance(train_ratio + val_ratio + test_ratio, float):
             raise ValueError("Ratios must be floats")
        if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
            # Normalize ratios if they don't sum to 1
            logger.warning(f"Provided train/val/test ratios sum to {train_ratio + val_ratio + test_ratio}. Normalizing.")
            total_ratio = train_ratio + val_ratio + test_ratio
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

        if e0s not in ["default", "average"]:
            try:
                # Check if it's a valid JSON string representing a dictionary
                e0s_dict = json.loads(e0s)
                if not isinstance(e0s_dict, dict):
                     raise ValueError("e0s string must be a valid JSON dictionary if not 'default' or 'average'.")
            except json.JSONDecodeError:
                 raise ValueError("e0s string must be 'default', 'average', or a valid JSON dictionary string.")

        # Validate structure selection arguments
        if structure_ids is not None and num_structures is not None:
            raise ValueError("Cannot specify both structure_ids and num_structures")
        if structure_ids is None and num_structures is None:
            raise ValueError("Must specify either structure_ids or num_structures in standalone mode")

        # Get structure IDs
        final_structure_ids: List[int]
        if structure_ids is not None:
            final_structure_ids = structure_ids
        else:
            assert num_structures is not None  # Help type checker
            logger.info(f"Fetching up to {num_structures} structures...")
            all_structures = _get_vasp_structures(db_manager)
            if len(all_structures) < num_structures:
                 logger.warning(f"Requested {num_structures} structures, but only {len(all_structures)} found. Using all available.")
                 final_structure_ids = all_structures
            else:
                 random.seed(seed) # Use provided seed for sampling
                 final_structure_ids = random.sample(all_structures, num_structures)
            logger.info(f"Selected {len(final_structure_ids)} structures.")

        if not final_structure_ids:
             raise ValueError("No structures selected for standalone run. Cannot proceed.")

        # Prepare data splits (using file_base_name for filenames)
        # Saves xyz into job_data_dir and json into job_dir
        # _prepare_structure_splits also handles _replace_properties now
        saved_structure_ids = _prepare_structure_splits(
            db_manager=db_manager,
            structure_ids=final_structure_ids,
            job_name=file_base_name, # Use base name for file naming
            job_dir=job_dir, # For json file location
            data_dir=job_data_dir, # For xyz file location
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed # Use provided seed for splitting
        )

        # Define relative paths for config.yaml
        train_file_rel = f"data/{file_base_name}_train.xyz"
        val_file_rel = f"data/{file_base_name}_val.xyz"
        test_file_rel = f"data/{file_base_name}_test.xyz"

        # Check if files were created before setting paths
        if (job_dir / train_file_rel).exists():
            config_data_train_path = train_file_rel
        else:
            logger.error(f"Training file {train_file_rel} was not created successfully.")
            raise FileNotFoundError(f"Training file {train_file_rel} failed to generate.")

        if (job_dir / val_file_rel).exists():
            config_data_val_path = val_file_rel
        else:
             logger.error(f"Validation file {val_file_rel} was not created successfully.")
             raise FileNotFoundError(f"Validation file {val_file_rel} failed to generate.")

        if (job_dir / test_file_rel).exists():
             config_data_test_path = test_file_rel
        else:
             logger.warning(f"Test file {test_file_rel} was not created (might be intended if test_ratio was 0).")
             config_data_test_path = test_file_rel # Still add path to config

    # --- Generate config.yaml --- 
    logger.info(f"[{job_name}] Generating config.yaml...")

    # Handle E0s formatting for YAML
    final_e0s: Union[str, Dict[int, float]]
    if e0s == "average":
        final_e0s = "average"
    elif e0s == "default":
        # Use the hardcoded default dict
        final_e0s = {22: -2.15203187, 23: -3.55411419, 24: -5.42767241, 40: -2.3361286, 74: -4.55186158}
    else:
        # Assume e0s is a valid JSON string, parse it into a dict
        try:
            final_e0s = json.loads(e0s)
            # Ensure keys are integers if possible (MACE might prefer this)
            final_e0s = {int(k): v for k, v in final_e0s.items()}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse provided e0s JSON string '{e0s}': {e}. Using default.")
            final_e0s = {22: -2.15203187, 23: -3.55411419, 24: -5.42767241, 40: -2.3361286, 74: -4.55186158}

    # Build the config dictionary
    config = {
        # MACE specific keys (refer to mace_run_train documentation)
        'name': job_name, # Model name / run name
        'seed': seed,
        'train_file': config_data_train_path,
        'valid_file': config_data_val_path,
        'test_file': config_data_test_path,
        'E0s': final_e0s,
        'r_max': r_max,
        'num_interactions': num_interactions,
        'num_hidden_layers': num_interactions, # Often tied? Check MACE docs. Assume equal for now.
        'hidden_irreps': f'{num_channels}x0e', # Example format, adjust based on max_L etc.
        'max_L': max_L,
        'correlation': 3, # Common default
        'error_table': 'PerAtomRMSE', # Common default
        'default_dtype': 'float64', # Common default

        # Training parameters
        'lr': lr,
        'batch_size': 10, # Example default, maybe make configurable?
        'valid_batch_size': 10, # Example default
        'max_num_epochs': 500, # Example default, maybe make configurable?
        'patience': 50, # Example default
        'eval_interval': 1, # Example default
        'loss': 'weighted', # Common default for energy/forces/stress
        'energy_weight': energy_weight,
        'forces_weight': forces_weight,
        'stress_weight': stress_weight,

        # Logging
        'log_dir': './logs', # Relative to job_dir
        'wandb': True,
        'wandb_project': 'mace-forge', # Example project name, maybe make configurable?
        'wandb_name': job_name, # Use unique run name
        'wandb_log_hypers': ['num_interactions', 'num_channels', 'max_L', 'lr', 'forces_weight'], # Example

        # Add other MACE parameters as needed based on mace_run_train
        # 'model': 'MACE', # Usually inferred?
        # 'swa': None, # Stochastic Weight Averaging config
        # 'ema': None, # Exponential Moving Average config
        # 'scheduler': 'ReduceLROnPlateau', # Example
        # 'lr_factor': 0.8, # Example
        # 'scheduler_patience': 20, # Example
    }

    # Adjust hidden_irreps based on max_L if necessary (needs MACE expertise)
    if max_L == 0:
        config['hidden_irreps'] = f'{num_channels}x0e'
    elif max_L == 1:
        config['hidden_irreps'] = f'{num_channels}x0e + {num_channels}x1o' # Example
    else:
        # Placeholder for higher L, adjust based on MACE documentation
        config['hidden_irreps'] = f'{num_channels}x0e + {num_channels}x1o + ...'
        logger.warning(f"Hidden irreps format for max_L={max_L} needs verification.")

    # --- Write YAML ---
    yaml_path = job_dir / "config.yaml"
    try:
        # Use default_flow_style=False for a more readable block format
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, indent=2, default_flow_style=False, sort_keys=False)
        logger.info(f"[{job_name}] Successfully generated {yaml_path}")
    except Exception as e:
        logger.error(f"Failed to write config.yaml for {job_name}: {e}", exc_info=True)
        raise IOError(f"Failed to write config.yaml: {e}") from e

    # Return structure IDs only if generated in standalone mode
    return saved_structure_ids if not is_hpo_mode else {}

def _get_vasp_structures(db_manager) -> List[int]:
    """Get all structures with complete VASP calculations."""
    # Find structures with VASP calculations
    structures = []

    # Query for structures with completed VASP calculations
    with db_manager.conn.cursor() as cur:
        query = """
            SELECT DISTINCT s.structure_id
            FROM structures s
            JOIN calculations c ON s.structure_id = c.structure_id
            WHERE c.calculator LIKE 'vasp%'
            AND c.energy IS NOT NULL
            AND c.forces IS NOT NULL
            AND c.stress IS NOT NULL
        """
        # AND c.metadata->>'status' = 'completed'
        #print(f"Executing query: {query}")  # Debug
        cur.execute(query)
        structures = [row[0] for row in cur.fetchall()]
        print(f"Found {len(structures)} structures")  # Debug

    return structures

def _split_structures(structures: List[int], ratios: List[int]) -> List[List[int]]:
    """Split list of structures according to ratios."""
    total = len(structures)
    splits = []
    start = 0

    for ratio in ratios[:-1]:  # Handle all but last split
        split_size = int(total * ratio)
        splits.append(structures[start:start + split_size])
        start += split_size

    # Add remaining structures to last split
    splits.append(structures[start:])

    return splits

def _save_structures_to_xyz(
    db_manager: DatabaseManager,
    structure_ids: List[int],
    output_path: Path
) -> List[int]:
    """Save structures with attached VASP calculation data to an XYZ file."""
    if not structure_ids:
        # Don't print warning here, let the caller handle verbosity if needed
        # print("[WARN] No structure IDs provided to save.")
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists even if file is empty
        output_path.touch() # Create empty file
        return []

    # print(f"Attempting to fetch and save {len(structure_ids)} structures to {output_path}") # Verbose
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch structures with latest VASP calculation data attached in batch
    try:
        atoms_with_calc = db_manager.get_batch_atoms_with_calculation(
            structure_ids, calculator='vasp'
        )
    except Exception as e:
        print(f"[ERROR] Failed to fetch batch atoms with calculation: {e}")
        import traceback
        print(traceback.format_exc())
        return [] # Return empty list if batch fetch fails

    structures_to_write = []
    saved_ids = []
    skipped_ids = [] # Track IDs skipped due to missing calculation or other issues

    # Create a map for quick lookup
    atoms_map = {atoms.info['structure_id']: atoms for atoms in atoms_with_calc}

    # Iterate through the original list to maintain order somewhat,
    # but process the fetched data
    for i, struct_id in enumerate(structure_ids):
        if struct_id in atoms_map:
            atoms = atoms_map[struct_id]
            # Check if calculation data was successfully attached by the batch method
            if 'calculation_info' in atoms.info and 'energy' in atoms.info:
                # Prepare atoms for writing (remove potentially large/unneeded info)
                keys_to_keep = {'energy', 'stress', 'structure_id', 'calculation_info'}
                atoms.info = {k: v for k, v in atoms.info.items() if k in keys_to_keep or not k.startswith('_')}

                structures_to_write.append(atoms)
                saved_ids.append(struct_id)
            else:
                 # Structure was found, but no VASP calculation was attached (or energy was missing)
                 # print(f"[INFO] Skipping structure {struct_id}: No completed VASP calculation data found/attached.") # Verbose
                 skipped_ids.append(struct_id)
        else:
            # Structure ID was not found in the initial batch retrieval
            # print(f"[WARN] Structure ID {struct_id} not found in database during batch fetch.") # Verbose
            skipped_ids.append(struct_id)

        # Print progress periodically? Maybe too verbose for HPO.
        # if (i + 1) % 100 == 0:
        #     print(f"Processed {i+1}/{len(structure_ids)} requested structures...")

    # print(f"\nSuccessfully prepared {len(structures_to_write)} structures for writing.") # Verbose
    # if skipped_ids:
        # print(f"Skipped {len(skipped_ids)} structures due to missing data or errors.") # Verbose

    # Write the collected structures to the XYZ file
    if structures_to_write:
        try:
            # Using ase.io.write which handles writing multiple frames
            write(output_path, structures_to_write, format='extxyz')
            # print(f"Wrote {len(structures_to_write)} structures to {output_path}") # Verbose
        except Exception as e:
            print(f"[ERROR] Error writing structures to {output_path}: {e}")
            import traceback
            print(traceback.format_exc())
    # else: # Write empty file if no structures were valid
         # write(output_path, [], format='extxyz') # ensure file exists
         # print(f"[WARN] No valid structures with calculation data found to write to {output_path}!") # Verbose


    return saved_ids # Return IDs of structures actually written

def _replace_properties(xyz_path: Path):
    """Replace property names in xyz file."""
    replacements = {
        ' energy=': ' REF_energy=',
        'stress=': 'REF_stress=',
        ':forces:': ':REF_force:'
    }

    # Check if file exists and is not empty before reading
    if not xyz_path.exists() or xyz_path.stat().st_size == 0:
        # print(f"[INFO] Skipping property replacement for non-existent or empty file: {xyz_path}")
        return # Nothing to do

    try:
        with open(xyz_path, 'r') as f:
            content = f.readlines()
    except Exception as e:
        print(f"[ERROR] Failed to read {xyz_path} for property replacement: {e}")
        return

    new_content = []
    modified = False
    for line in content:
        original_line = line
        # Avoid replacing properties within comments or unrelated lines
        if ' Properties=' in line:
             for old_prop, new_prop in replacements.items():
                 # Use word boundaries or specific patterns if needed, but simple replace might be okay for ASE format
                 if old_prop in line and not line.strip().startswith('free_energy='):
                     line = line.replace(old_prop, new_prop)
        new_content.append(line)
        if line != original_line:
            modified = True

    if modified:
        try:
            with open(xyz_path, 'w') as f:
                f.writelines(new_content)
        except Exception as e:
            print(f"[ERROR] Failed to write updated properties to {xyz_path}: {e}")

def _prepare_structure_splits(
    db_manager,
    structure_ids: List[int],
    job_name: str, # Base name used for filenames
    job_dir: Path, # The specific run directory for saving splits.json
    data_dir: Path, # The data directory (e.g., run_dir/data)
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Dict[str, List[int]]:
    """Prepare structure splits, save to XYZ, run property replacement, and save ID map."""
    total = len(structure_ids)
    # Ensure ratios sum close to 1 (copied logic from prepare_mace_job)
    if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
         logger.warning(f"Ratios sum to {train_ratio + val_ratio + test_ratio}. Normalizing.")
         total_ratio = train_ratio + val_ratio + test_ratio
         train_ratio /= total_ratio
         val_ratio /= total_ratio
         test_ratio /= total_ratio

    # Calculate sizes (use floor for train/val, remainder for test - matches hpo_sweep logic)
    train_size = math.floor(total * train_ratio)
    val_size = math.floor(total * val_ratio)
    test_size = total - train_size - val_size # Remainder is test

    logger.info(f"Splitting {total} structures into: {train_size} train, {val_size} val, {test_size} test (using seed {seed})")

    # Shuffle and split structures
    random.seed(seed)
    shuffled_ids = structure_ids.copy()
    random.shuffle(shuffled_ids)

    # Create the splits
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:train_size + val_size]
    test_ids = shuffled_ids[train_size + val_size:]

    split_mapping = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    # Save structure splits to xyz files and track saved IDs
    saved_structure_ids_actual = {} # Track IDs actually saved
    any_save_failed = False
    for split_name, struct_list in split_mapping.items():
        # Use job_name (base_name) for the xyz filenames
        xyz_path = data_dir / f"{job_name}_{split_name}.xyz"
        logger.info(f"  Saving {len(struct_list)} {split_name} structures to {xyz_path}...")
        saved_ids = _save_structures_to_xyz(db_manager, struct_list, xyz_path)
        saved_structure_ids_actual[split_name] = saved_ids

        # Check if saving actually worked as expected (simple length check)
        if len(saved_ids) != len(struct_list):
             logger.warning(f"Mismatch in saved {split_name} IDs: requested {len(struct_list)}, saved {len(saved_ids)}.")
             # Don't mark as failed, but log is important

        # Run property replacement script only if file seems valid
        if xyz_path.exists() and xyz_path.stat().st_size > 0:
             try:
                 logger.info(f"    Running property replacement on {xyz_path.name}...")
                 _replace_properties(xyz_path)
             except Exception as e:
                 logger.error(f"    Failed replacing properties in {xyz_path}: {e}", exc_info=True)
                 any_save_failed = True
        elif not xyz_path.exists():
             logger.warning(f"    Skipping property replacement: {xyz_path.name} does not exist.")
             if struct_list: # If we expected structures, this is an error
                 any_save_failed = True
        else: # File exists but is empty
            logger.info(f"    Skipping property replacement: {xyz_path.name} is empty.")


    # Save structure ID mapping (in the specific run's job_dir)
    # Save the *intended* split mapping, as this is what the config generation used
    splits_json_path = job_dir / "structure_splits.json"
    try:
        with open(splits_json_path, 'w') as f:
            # Save the intended split (split_mapping), not necessarily what was successfully written (saved_structure_ids_actual)
            json.dump(split_mapping, f, indent=2)
        logger.info(f"  Saved intended structure ID splits to {splits_json_path}")
    except Exception as e:
        logger.error(f"Failed to save structure splits mapping to {splits_json_path}: {e}", exc_info=True)
        any_save_failed = True

    if any_save_failed:
         logger.error(f"Issues encountered during data splitting/saving/property replacement for {job_name}. Check logs.")
         # Decide whether to raise an error here or let the caller handle potentially missing files
         # raise IOError(f"Data preparation failed for {job_name}")

    # Return the mapping of split name to the list of IDs *intended* for that split
    return split_mapping