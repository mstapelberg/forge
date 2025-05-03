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
from forge.core.database import DatabaseManager  # Add this import

# Configure logging
logger = logging.getLogger(__name__)

class GPUConfig(TypedDict):
    count: int
    type: str

def _create_symlink_if_needed(link_name: Path, target_file: Path):
    """Creates or corrects a relative symlink.

    Args:
        link_name: The path where the symlink should be created.
        target_file: The path to the actual data file.
    """
    target_file = target_file.resolve()
    link_name = link_name.resolve() # Resolve link path itself
    link_dir = link_name.parent

    if not target_file.exists():
        raise FileNotFoundError(f"Symlink target does not exist: {target_file}")

    try:
        rel_path = os.path.relpath(target_file, link_dir)
    except ValueError:
        # Fallback for different drives (Windows)
        rel_path = str(target_file)

    link_exists = link_name.exists()
    is_correct_link = False
    if link_name.is_symlink():
        try:
            current_target_str = os.readlink(link_name)
            # Resolve current target relative to link directory
            current_target_path = (link_dir / current_target_str).resolve()
            if current_target_path == target_file:
                is_correct_link = True
            else:
                logger.warning(f"Symlink {link_name.name} points incorrectly. Target: {current_target_path}, Expected: {target_file}")
        except OSError as e:
            logger.warning(f"Could not read existing symlink {link_name.name}: {e}")

    if not is_correct_link:
        if link_exists:
            logger.warning(f"Removing existing file/link at {link_name.name} to create correct symlink.")
            link_name.unlink(missing_ok=True)
        try:
            os.symlink(rel_path, link_name, target_is_directory=False)
            logger.info(f"  Created symlink: {link_name.name} -> {rel_path}")
        except OSError as e:
            logger.error(f"Failed to create symlink {link_name.name}: {e}", exc_info=True)
            raise # Re-raise after logging

def prepare_mace_job(
    db_manager: DatabaseManager,
    job_name: str,
    job_dir: Union[str, Path],
    gpu_config: GPUConfig = {"count": 1, "type": "rtx6000"},
    structure_ids: Optional[List[int]] = None,
    num_structures: Optional[int] = None,
    train_ratio: Optional[float] = None, # Made optional for pre-split case
    val_ratio: Optional[float] = None,   # Made optional for pre-split case
    test_ratio: Optional[float] = None,  # Made optional for pre-split case
    e0s: str = "default",
    seed: int = 42,
    num_ensemble: Optional[int] = None,
    num_interactions: int = 2,
    num_channels: int = 128,
    max_L: int = 0,
    r_max: float = 5.0,
    lr: float = 0.001,
    forces_weight: float = 50.0,
    energy_weight: float = 1.0,
    stress_weight: float = 25.0,
    base_name: Optional[str] = None, # DEPRECATED? MACE script uses base_name for data paths
    external_data_source_dir: Optional[Union[str, Path]] = None # New parameter
) -> Dict[str, List[int]]:
    """
    Prepare a MACE training job from database structures.
    Handles standard splitting, using pre-split data found in job_dir/data,
    or using externally prepared data (via symlinks).

    Args:
        db_manager: DatabaseManager instance for accessing structure data
        job_name: Name for the training job (used as unique identifier for this run)
        job_dir: Directory to create job in
        gpu_config: GPU configuration with count and type
        structure_ids: Optional list of specific structure IDs to use (if not pre-split)
        num_structures: Optional number of random structures to select (if not pre-split)
        train_ratio: Fraction for training (required if not pre-split and no external source)
        val_ratio: Fraction for validation (required if not pre-split and no external source)
        test_ratio: Fraction for testing (required if not pre-split and no external source)
        e0s: "default" or "average" for E0 configuration.
        seed: Random seed for structure selection/splitting (if not pre-split) and training.
        num_ensemble: Optional number of ensemble models to create (uses job_name as base)
        num_interactions: MACE num_interactions parameter
        num_channels: MACE num_channels parameter
        max_L: MACE max_L parameter
        r_max: MACE r_max parameter
        lr: MACE learning rate
        forces_weight: MACE forces_weight parameter
        energy_weight: MACE energy_weight parameter
        stress_weight: MACE stress_weight parameter
        base_name: Base name for generated MACE data files IF splitting internally.
                   If None, defaults to job_name.
                   If external_data_source_dir is used, this is ignored for file splitting
                   but is still used in the generated training script (template needs ${BASE_NAME}).
        external_data_source_dir: Optional path to a directory containing pre-split
                                  'train.xyz', 'val.xyz', 'test.xyz', and
                                  'structure_splits.json'. If provided, symlinks
                                  will be created in job_dir/data.

    Returns:
        Dict mapping 'train', 'val', 'test' to lists of structure_ids used.
        Returns IDs from pre-split file if data exists, otherwise from new split.

    Raises:
        ValueError: If invalid arguments are provided
        FileNotFoundError: If external_data_source_dir or its contents are missing.
    """
    job_dir = Path(job_dir)
    job_data_dir = job_dir / "data" # Target directory for data/symlinks for this run
    job_data_dir.mkdir(parents=True, exist_ok=True)

    # Determine base name for internal splitting / script generation
    script_base_name = base_name if base_name is not None else job_name

    saved_structure_ids: Dict[str, List[int]] = {'train': [], 'val': [], 'test': []}
    data_prep_skipped = False

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
        except Exception as e:
            raise IOError(f"Failed to read {external_splits_json}: {e}") from e

        # Create relative symlinks in job_data_dir pointing to external files
        # MACE script template uses ${BASE_NAME} for data files, so links need that name.
        for split in ["train", "val", "test"]:
            src_file = external_data_path / f"{split}.xyz" # Generic name in source dir
            link_name = job_data_dir / f"{script_base_name}_{split}.xyz" # Name expected by script
            _create_symlink_if_needed(link_name, src_file)

        data_prep_skipped = True

    else:
        # Strategy 2: Check for pre-existing data in job_data_dir (standard HPO non-kfold)
        # Check for files named according to script_base_name
        train_file = job_data_dir / f"{script_base_name}_train.xyz"
        val_file = job_data_dir / f"{script_base_name}_val.xyz"
        test_file = job_data_dir / f"{script_base_name}_test.xyz"
        splits_json_file = job_dir / "structure_splits.json" # JSON is always in job_dir

        if train_file.exists() and val_file.exists() and test_file.exists():
            logger.info(f"[{job_name}] Found existing data files (named {script_base_name}_...) in {job_data_dir}. Skipping data preparation.")
            data_prep_skipped = True
            if splits_json_file.exists():
                try:
                    with open(splits_json_file, 'r') as f:
                        saved_structure_ids = json.load(f)
                    logger.info(f"[{job_name}] Loaded structure IDs from {splits_json_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {splits_json_file}: {e}. Returning empty dict.")
                    saved_structure_ids = {'train': [], 'val': [], 'test': []}
            else:
                 logger.warning(f"Pre-split data files found, but {splits_json_file} is missing. Cannot return structure IDs.")
                 saved_structure_ids = {'train': [], 'val': [], 'test': []}

    # Strategy 3: If no external source and no pre-existing files, perform splitting
    if not data_prep_skipped:
        logger.info(f"[{job_name}] Data not found. Proceeding with standard data preparation.")
        # --- Original Data Prep Logic ---
        # Validate input arguments for splitting
        assert isinstance(job_name, str), "job_name must be a string"
        if train_ratio is None or val_ratio is None or test_ratio is None:
            raise ValueError("train_ratio, val_ratio, and test_ratio must be provided when data is not pre-split and no external source is given.")
        assert isinstance(train_ratio + val_ratio + test_ratio, float), "Ratios must be floats"
        assert 0.99 <= train_ratio + val_ratio + test_ratio <= 1.01, "Ratios must sum to ~1"
        assert all(0 <= r <= 1 for r in [train_ratio, val_ratio, test_ratio]), "Ratios must be between 0 and 1"
        assert e0s in ["default", "average"], "e0s must be 'default' or 'average'"

        # Validate structure selection arguments
        if structure_ids is not None and num_structures is not None:
            raise ValueError("Cannot specify both structure_ids and num_structures")
        if structure_ids is None and num_structures is None:
            raise ValueError("Must specify either structure_ids or num_structures when data is not pre-split")

        # Get structure IDs
        final_structure_ids: List[int]
        if structure_ids is not None:
            final_structure_ids = structure_ids
        else:
            assert num_structures is not None  # Help type checker
            all_structures = _get_vasp_structures(db_manager)
            if len(all_structures) < num_structures:
                raise ValueError(
                    f"Not enough structures in database. "
                    f"Requested {num_structures}, but only found {len(all_structures)}"
                )
            random.seed(seed) # Use provided seed for sampling
            final_structure_ids = random.sample(all_structures, num_structures)

        # Prepare data splits (using script_base_name for filenames)
        # Saves xyz into job_data_dir and json into job_dir
        saved_structure_ids = _prepare_structure_splits(
            db_manager=db_manager,
            structure_ids=final_structure_ids,
            job_name=script_base_name, # Use base name for file naming
            job_dir=job_dir, # For json file location
            data_dir=job_data_dir, # For xyz file location
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed # Use provided seed for splitting
        )
        # --- End of Original Data Prep Logic ---

    # --- Script Generation (runs regardless of data prep method) ---
    logger.info(f"[{job_name}] Generating training script(s)...")
    if num_ensemble is None:
        # Single model case
        _create_training_script(
            job_dir=job_dir,
            job_name=job_name, # Use the unique run name for the script/job
            base_name=script_base_name, # Use script_base_name for data file reference
            gpu_config=gpu_config,
            e0s=e0s,
            seed=seed, # Use the specific seed for this run
            num_interactions=num_interactions,
            num_channels=num_channels,
            max_L=max_L,
            r_max=r_max,
            lr=lr,
            forces_weight=forces_weight,
            energy_weight=energy_weight,
            stress_weight=stress_weight
        )
    else:
         # Create multiple models with different seeds
        if data_prep_skipped:
             logger.warning(f"[{job_name}] num_ensemble > 1 requested, but data was prepared externally or existed. "
                   "All ensemble members will use the same data. Only the training seed will differ.")

        for model_idx in range(num_ensemble):
            # Generate unique job name for ensemble member, but use the same base name for data reference
            model_job_name = f"{job_name}_ensemble_{model_idx}"
            model_seed = seed + model_idx # Ensure different training seed for each member
            _create_training_script(
                job_dir=job_dir, # Still in the same run directory
                job_name=model_job_name, # Unique name for this script/job
                base_name=script_base_name, # Base name for data file reference (symlinks/files)
                gpu_config=gpu_config,
                e0s=e0s,
                seed=model_seed, # Use ensemble-specific seed
                num_interactions=num_interactions,
                num_channels=num_channels,
                max_L=max_L,
                r_max=r_max,
                lr=lr,
                forces_weight=forces_weight,
                energy_weight=energy_weight,
                stress_weight=stress_weight
            )
            logger.warning(f"[{job_name}] Ensemble script generation for MACE needs review. Ensure template/MACE handles ensemble correctly.")


    return saved_structure_ids # Return the structure IDs (either loaded or newly generated)

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


def _create_training_script(
    job_dir: Path,
    job_name: str, # This is the specific run name (e.g., hpo_..._fold_..._seed)
    base_name: str, # This is the name used for data files (e.g., hpo_... or original job name)
    gpu_config: Dict,
    e0s: str = "default",
    seed: int = 42,
    num_interactions: int = 2,
    num_channels: int = 128,
    max_L: int = 0,
    r_max: float = 5.0,
    lr: float = 0.001,
    forces_weight: float = 50.0,
    energy_weight: float = 1.0,
    stress_weight: float = 25.0
):
    """Create MACE training script from template."""
    template_path = Path(__file__).parent / "templates" / "mace_train_template.sh"

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, 'r') as f:
        template = f.read()

    # Determine E0s string based on input
    if e0s == "average":
        e0s_str = "average"
    else: # Default to the hardcoded string if not "average"
        # Ensure the default string is valid JSON or the expected format
        # Original was: '{22: -2.15203187, 23 : -3.55411419, 24 : -5.42767241, 40 : -2.3361286, 74 : -4.55186158}'
        # Let's ensure keys are strings if needed by MACE
        default_e0s_dict = {22: -2.15203187, 23 : -3.55411419, 24 : -5.42767241, 40 : -2.3361286, 74 : -4.55186158}
        # MACE might expect a string representation of the dict. Check MACE docs.
        # Assuming string representation is fine for now.
        e0s_str = json.dumps(default_e0s_dict).replace(" ", "") # Compact JSON string


    # Replace template variables using exact placeholder names from template
    replacements = {
        '${JOB_NAME}': job_name, # SBATCH job name (unique run name)
        '${NTASKS_PER_NODE}': str(gpu_config['count']),
        '${GPUS_PER_NODE}': str(gpu_config['count']),
        '${GPU_TYPE}': gpu_config['type'],
        '${RUN_NAME}': job_name, # MACE --name argument (unique run name)
        '${NUM_INTERACTIONS}': str(num_interactions),
        '${NUM_CHANNELS}': str(num_channels),
        '${MAX_L}': str(max_L),
        # Use the correctly formatted E0s string, ensure quotes if needed by shell
        '${E0S_STR}': f"'{e0s_str}'", # Add quotes for shell interpretation
        '${FORCES_WEIGHT}': str(forces_weight),
        '${ENERGY_WEIGHT}': str(energy_weight),
        '${STRESS_WEIGHT}': str(stress_weight),
        '${R_MAX}': str(r_max),
        '${LR}': str(lr),
        '${BASE_NAME}': base_name, # Base name for data files (train/val/test paths)
        '${SEED}': str(seed), # Use the specific training seed for this run
        # WandB run name should also be the unique job name
        '${WANDB_NAME}': job_name
    }

    script_content = template
    for placeholder, value in replacements.items():
        # Need to be careful with replacing placeholders within values, simple replace should be ok here
        script_content = script_content.replace(placeholder, value)

    # Write training script (using the unique job_name for the script file)
    script_path = job_dir / f"{job_name}_train.sh"
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        # Make script executable
        script_path.chmod(0o755)
    except Exception as e:
        print(f"[ERROR] Failed to write or chmod script {script_path}: {e}")


def _prepare_structure_splits(
    db_manager,
    structure_ids: List[int],
    job_name: str, # Should be the base_name used for filenames
    job_dir: Path, # The specific run directory for saving splits.json
    data_dir: Path, # The data directory (e.g., run_dir/data)
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Dict[str, List[int]]:
    """Prepare structure splits for training, validation, and testing."""
    total = len(structure_ids)
    # Ensure ratios sum close to 1
    if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
         print(f"[Warning] Ratios sum to {train_ratio + val_ratio + test_ratio}. Adjusting test ratio.")
         test_ratio = 1.0 - train_ratio - val_ratio
         if test_ratio < 0: test_ratio = 0 # Avoid negative test ratio

    val_size = math.ceil(total * val_ratio)
    # Calculate test_size carefully to avoid exceeding total when combined with val_size
    test_size = math.ceil(total * test_ratio)
    if val_size + test_size > total:
        # Prioritize validation set size, reduce test set size
        test_size = total - val_size
        if test_size < 0: test_size = 0 # Ensure non-negative

    train_size = total - val_size - test_size
    if train_size < 0: train_size = 0 # Ensure non-negative

    # Adjust sizes slightly if rounding caused total mismatch, prioritize train size
    current_total = train_size + val_size + test_size
    diff = total - current_total
    train_size += diff # Add/remove difference from train set

    print(f"Splitting {total} structures into: {train_size} train, {val_size} val, {test_size} test (using seed {seed})")

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
    saved_structure_ids = {}
    any_save_failed = False
    for split_name, struct_list in split_mapping.items():
        # Use job_name (which is the base_name) for the xyz filenames
        xyz_path = data_dir / f"{job_name}_{split_name}.xyz"
        # print(f"\nProcessing {split_name} split with {len(struct_list)} structures for {xyz_path}") # Verbose
        saved_ids = _save_structures_to_xyz(db_manager, struct_list, xyz_path)
        saved_structure_ids[split_name] = saved_ids

        # Check if saving actually worked as expected (simple length check)
        if len(saved_ids) != len(struct_list):
             print(f"[Warning] Mismatch in saved {split_name} IDs: requested {len(struct_list)}, saved {len(saved_ids)}.")
             # Decide if this is critical? Maybe allow continuation but warn.

        # Run property replacement script only if file seems valid
        if xyz_path.exists() and xyz_path.stat().st_size > 0:
             try:
                 _replace_properties(xyz_path)
             except Exception as e:
                 print(f"[ERROR] Failed replacing properties in {xyz_path}: {e}")
                 any_save_failed = True # Treat property replacement failure seriously?

    # Save structure ID mapping (in the specific run's job_dir)
    splits_json_path = job_dir / "structure_splits.json"
    try:
        with open(splits_json_path, 'w') as f:
            json.dump(saved_structure_ids, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save structure splits mapping to {splits_json_path}: {e}")
        any_save_failed = True

    if any_save_failed:
         print(f"[Warning] Issues encountered during data splitting/saving for {job_name}. Check logs.")

    # Return the mapping of split name to the list of IDs *intended* for that split,
    # even if saving failed for some, as the caller might need the intended list.
    # The saved_structure_ids dict contains IDs actually saved. Maybe return that instead?
    # Let's return the intended split_mapping, as that's what downstream might expect.
    # The warnings should indicate if saving failed.
    return split_mapping


def _create_single_model(
    job_name: str,
    job_dir: Path,
    data_dir: Path,
    structure_splits: Dict[str, List[int]],
    gpu_config: Dict,
    e0s: str = "default"
) -> List[int]:
    """Create a single MACE model training script and return saved structure IDs."""
    # This function seems deprecated by the logic integrated into prepare_mace_job
    # Let's comment it out or remove it.
    raise DeprecationWarning("_create_single_model is likely deprecated, use prepare_mace_job directly.")
    # # Create training script
    # _create_training_script(
    #     job_dir=job_dir,
    #     job_name=job_name,
    #     base_name=job_name, # Assume base_name is job_name here
    #     gpu_config=gpu_config,
    #     e0s=e0s
    #     # Need to pass other MACE params here... this function is incomplete
    # )
    # return structure_splits['train'] # Returning only train seems wrong too