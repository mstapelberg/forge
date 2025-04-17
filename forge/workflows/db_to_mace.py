import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypedDict
from ase.io import write
import shutil
import math
from forge.core.database import DatabaseManager  # Add this import

class GPUConfig(TypedDict):
    count: int
    type: str

def prepare_mace_job(
    db_manager: DatabaseManager,
    job_name: str,
    job_dir: Union[str, Path],
    gpu_config: GPUConfig = {"count": 1, "type": "rtx6000"},
    structure_ids: Optional[List[int]] = None,
    num_structures: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
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
    stress_weight: float = 25.0
) -> Dict[str, List[int]]:
    """
    Prepare a MACE training job from database structures.
    
    Args:
        db_manager: DatabaseManager instance for accessing structure data
        job_name: Name for the training job (used as base name for files/WandB)
        job_dir: Directory to create job in
        gpu_config: GPU configuration with count and type
        structure_ids: Optional list of specific structure IDs to use
        num_structures: Optional number of random structures to select
        train_ratio: Fraction of data for training (0-1)
        val_ratio: Fraction of data for validation (0-1)
        test_ratio: Fraction of data for testing (0-1)
        e0s: "default" or "average" for E0 configuration.
             "default" uses hardcoded values '{22: -2.15...}'
        seed: Random seed for structure selection and splitting
        num_ensemble: Optional number of ensemble models to create
        num_interactions: MACE num_interactions parameter
        num_channels: MACE num_channels parameter
        max_L: MACE max_L parameter
        r_max: MACE r_max parameter
        lr: MACE learning rate
        forces_weight: MACE forces_weight parameter
        energy_weight: MACE energy_weight parameter
        stress_weight: MACE stress_weight parameter
        
    Returns:
        Dict mapping 'train', 'val', 'test' to lists of structure_ids used in each split.
        
    Raises:
        ValueError: If invalid arguments are provided
    """
    # Validate input arguments
    assert isinstance(job_name, str), "job_name must be a string"
    assert isinstance(train_ratio + val_ratio + test_ratio, float), "Ratios must be floats"
    assert 0.99 <= train_ratio + val_ratio + test_ratio <= 1.01, "Ratios must sum to 1"
    assert all(0 <= r <= 1 for r in [train_ratio, val_ratio, test_ratio]), "Ratios must be between 0 and 1"
    assert e0s in ["default", "average"], "e0s must be 'default' or 'average'"
    
    # Validate structure selection arguments
    if structure_ids is not None and num_structures is not None:
        raise ValueError("Cannot specify both structure_ids and num_structures")
    if structure_ids is None and num_structures is None:
        raise ValueError("Must specify either structure_ids or num_structures")
    
    # Validate ensemble configuration
    if num_ensemble is not None:
        if not isinstance(num_ensemble, int) or num_ensemble <= 0:
            raise ValueError("num_ensemble must be a positive integer")
    
    # Create job directory
    job_dir = Path(job_dir)
    data_dir = job_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        random.seed(seed)
        final_structure_ids = random.sample(all_structures, num_structures)
    
    # Prepare data splits (done once for all models)
    saved_structure_ids = _prepare_structure_splits(
        db_manager=db_manager,
        structure_ids=final_structure_ids,
        job_name=job_name,
        job_dir=job_dir,
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Create training scripts
    if num_ensemble is None:
        # Single model case
        _create_training_script(
            job_dir=job_dir,
            job_name=job_name,
            base_name=job_name,
            gpu_config=gpu_config,
            e0s=e0s,
            seed=seed,
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
        for model_idx in range(num_ensemble):
            model_job_name = f"{job_name}_model_{model_idx}"
            _create_training_script(
                job_dir=job_dir,
                job_name=model_job_name,
                base_name=job_name,
                gpu_config=gpu_config,
                e0s=e0s,
                seed=model_idx,
                num_interactions=num_interactions,
                num_channels=num_channels,
                max_L=max_L,
                r_max=r_max,
                lr=lr,
                forces_weight=forces_weight,
                energy_weight=energy_weight,
                stress_weight=stress_weight
            )
    
    return saved_structure_ids

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
            WHERE c.model_type LIKE 'vasp%'
            AND c.energy IS NOT NULL
            AND c.forces IS NOT NULL
            AND c.stress IS NOT NULL
        """
        # AND c.metadata->>'status' = 'completed'
        print(f"Executing query: {query}")  # Debug
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
        print("[WARN] No structure IDs provided to save.")
        return []

    print(f"Attempting to fetch and save {len(structure_ids)} structures to {output_path}")
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
                # Optional: Add extra validation if needed, though shape checks are in batch method
                # forces = atoms.arrays.get('forces')
                # n_atoms = len(atoms)
                # if forces is not None and forces.shape != (n_atoms, 3):
                #     print(f"[WARN] Skipping structure {struct_id} due to force shape mismatch (this shouldn't happen after batch retrieval)")
                #     skipped_ids.append(struct_id)
                #     continue

                # Prepare atoms for writing (remove potentially large/unneeded info)
                # Keep essential calculation info if needed downstream, but maybe not full calc dict
                keys_to_keep = {'energy', 'stress', 'structure_id', 'calculation_info'}
                atoms.info = {k: v for k, v in atoms.info.items() if k in keys_to_keep or not k.startswith('_')}

                structures_to_write.append(atoms)
                saved_ids.append(struct_id)
            else:
                 # Structure was found, but no VASP calculation was attached (or energy was missing)
                 print(f"[INFO] Skipping structure {struct_id}: No completed VASP calculation data found/attached.")
                 skipped_ids.append(struct_id)
        else:
            # Structure ID was not found in the initial batch retrieval
            print(f"[WARN] Structure ID {struct_id} not found in database during batch fetch.")
            skipped_ids.append(struct_id)

        # Print progress periodically
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(structure_ids)} requested structures...")

    print(f"\nSuccessfully prepared {len(structures_to_write)} structures for writing.")
    if skipped_ids:
        print(f"Skipped {len(skipped_ids)} structures due to missing data or errors.")

    # Write the collected structures to the XYZ file
    if structures_to_write:
        try:
            # Using ase.io.write which handles writing multiple frames
            write(output_path, structures_to_write, format='extxyz')
            print(f"Wrote {len(structures_to_write)} structures to {output_path}")
        except Exception as e:
            print(f"[ERROR] Error writing structures to {output_path}: {e}")
            import traceback
            print(traceback.format_exc())
            # Depending on desired behavior, maybe return partial saved_ids or raise error
    else:
        print("[WARN] No valid structures with calculation data found to write!")

    return saved_ids # Return IDs of structures actually written

def _replace_properties(xyz_path: Path):
    """Replace property names in xyz file."""
    replacements = {
        ' energy=': ' REF_energy=',
        'stress=': 'REF_stress=',
        ':forces:': ':REF_force:'
    }
    
    with open(xyz_path, 'r') as f:
        content = f.readlines()
    
    new_content = []
    for line in content:
        for old_prop, new_prop in replacements.items():
            if old_prop in line and not line.strip().startswith('free_energy='):
                line = line.replace(old_prop, new_prop)
        new_content.append(line)
    
    with open(xyz_path, 'w') as f:
        f.writelines(new_content)

def _create_training_script(
    job_dir: Path,
    job_name: str,
    base_name: str,
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
        e0s_str = '{22: -2.15203187, 23 : -3.55411419, 24 : -5.42767241, 40 : -2.3361286, 74 : -4.55186158}'
    
    # Replace template variables using exact placeholder names from template
    replacements = {
        '${JOB_NAME}': job_name, # SBATCH job name
        '${NTASKS_PER_NODE}': str(gpu_config['count']),
        '${GPUS_PER_NODE}': str(gpu_config['count']),
        '${GPU_TYPE}': gpu_config['type'],
        '${RUN_NAME}': job_name, # MACE --name argument
        '${NUM_INTERACTIONS}': str(num_interactions),
        '${NUM_CHANNELS}': str(num_channels),
        '${MAX_L}': str(max_L),
        '${E0S_STR}': e0s_str, # Use the processed E0s string
        '${FORCES_WEIGHT}': str(forces_weight),
        '${ENERGY_WEIGHT}': str(energy_weight),
        '${STRESS_WEIGHT}': str(stress_weight),
        '${R_MAX}': str(r_max),
        '${LR}': str(lr),
        '${BASE_NAME}': base_name, # Base name for data files
        '${SEED}': str(seed),
        '${WANDB_NAME}': job_name # WandB run name
    }
    
    script_content = template
    for placeholder, value in replacements.items():
        script_content = script_content.replace(placeholder, value)
    
    # Write training script
    script_path = job_dir / f"{job_name}_train.sh" # Use job_name for script file
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    script_path.chmod(0o755)

def _prepare_structure_splits(
    db_manager,
    structure_ids: List[int],
    job_name: str,
    job_dir: Path,
    data_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Dict[str, List[int]]:
    """Prepare structure splits for training, validation, and testing."""
    total = len(structure_ids)
    val_size = math.ceil(total * val_ratio)
    test_size = math.ceil(total * test_ratio)
    train_size = total - val_size - test_size

    print(f"Splitting {total} structures into: {train_size} train, {val_size} val, {test_size} test")

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
    for split_name, struct_list in split_mapping.items():
        xyz_path = data_dir / f"{job_name}_{split_name}.xyz"
        print(f"\nProcessing {split_name} split with {len(struct_list)} structures")
        saved_structure_ids[split_name] = _save_structures_to_xyz(
            db_manager, struct_list, xyz_path)
        
        # Run property replacement script
        _replace_properties(xyz_path)
    
    # Save structure ID mapping
    with open(job_dir / "structure_splits.json", 'w') as f:
        json.dump(saved_structure_ids, f, indent=2)
    
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
    # Create training script
    _create_training_script(
        job_dir=job_dir,
        job_name=job_name,
        base_name=job_name,
        gpu_config=gpu_config,
        e0s=e0s
    )
    
    return structure_splits['train']