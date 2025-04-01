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
    num_ensemble: Optional[int] = None
) -> Dict[str, List[int]]:
    """
    Prepare a MACE training job from database structures.
    
    Args:
        db_manager: DatabaseManager instance for accessing structure data
        job_name: Name for the training job
        job_dir: Directory to create job in
        gpu_config: GPU configuration with count and type
        structure_ids: Optional list of specific structure IDs to use
        num_structures: Optional number of random structures to select
        train_ratio: Fraction of data for training (0-1)
        val_ratio: Fraction of data for validation (0-1)
        test_ratio: Fraction of data for testing (0-1)
        e0s: "default" or "average" for E0 configuration
        seed: Random seed for structure selection and splitting
        num_ensemble: Optional number of ensemble models to create
        
    Returns:
        Dict mapping 'train', 'val', 'test' to lists of structure_ids
        
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
            gpu_config=gpu_config,
            e0s=e0s,
            seed=seed
        )
    else:
        # Create multiple models with different seeds
        for model_idx in range(num_ensemble):
            model_name = f"{job_name}_model_{model_idx}"
            _create_training_script(
                job_dir=job_dir,
                job_name=model_name,
                gpu_config=gpu_config,
                e0s=e0s,
                seed=model_idx  # Each model gets its index as seed
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
    db_manager,
    structure_ids: List[int],
    output_path: Path
) -> List[int]:
    """Save structures to xyz file and return list of saved structure IDs."""
    structures = []
    saved_ids = []
    failed_loads = []
    
    print(f"Attempting to save {len(structure_ids)} structures to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Track mismatches for debugging
    mismatches = []
    
    for i, struct_id in enumerate(structure_ids):
        try:
            calcs = db_manager.get_calculations(struct_id, model_type='vasp*')
            if not calcs:
                if i % 100 == 0:
                    print(f"No VASP calculations found for structure {struct_id}")
                continue
            
            calc = calcs[-1]
            
            try:
                atoms = db_manager.get_structure(struct_id)
            except Exception as e:
                print(f"Failed to load structure {struct_id}: {str(e)}")
                failed_loads.append({
                    'structure_id': struct_id,
                    'error': str(e)
                })
                continue
            
            # Validate array shapes
            forces = np.array(calc['forces']) if calc.get('forces') is not None else None
            stress = np.array(calc['stress']) if calc.get('stress') is not None else None
            n_atoms = len(atoms)
            
            if forces is not None and forces.shape[0] != n_atoms:
                mismatch = {
                    'structure_id': struct_id,
                    'n_atoms': n_atoms,
                    'forces_shape': forces.shape,
                    'model_type': calc['model_type'],
                    'calculation_id': calc['id']
                }
                mismatches.append(mismatch)
                print(f"WARNING: Structure {struct_id} has {n_atoms} atoms but forces shape is {forces.shape}")
                continue  # Skip this structure
            
            # Print debug info every 100 structures
            if i % 100 == 0:
                print(f"Processing structure {i}/{len(structure_ids)}: ID={struct_id}")
                print(f"Structure {struct_id} calculation info:")
                print(f"  model_type: {calc.get('model_type')}")
                energy = calc.get('energy')
                energy = energy[0] if isinstance(energy, list) else energy
                print(f"  energy: {energy}")
                print(f"  forces shape: {forces.shape if forces is not None else None}")
                print(f"  stress shape: {stress.shape if stress is not None else None}")
                print(f"  atoms info: {n_atoms} atoms, cell={atoms.cell.tolist()}")
            
            # Create calculator with validated arrays
            atoms.arrays['forces'] = forces  # Forces go in arrays since it's per-atom
            atoms.info['energy'] = energy    # Energy goes in info
            if stress is not None:           # Only add stress if it exists
                atoms.info['stress'] = stress
            
            # Remove any empty keys from atoms.info
            atoms.info = {k: v for k, v in atoms.info.items() if v is not None}
            
            structures.append(atoms)
            saved_ids.append(struct_id)
            
        except Exception as e:
            print(f"Error processing structure {struct_id}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"Successfully processed {len(structures)} structures")
    
    if mismatches:
        print("\nFound atom count / forces shape mismatches:")
        for m in mismatches:
            print(f"Structure {m['structure_id']} (calc {m['calculation_id']}):")
            print(f"  {m['n_atoms']} atoms but forces shape {m['forces_shape']}")
            print(f"  model_type: {m['model_type']}")
    
    if failed_loads:
        print("\nFailed to load these structures:")
        for f in failed_loads:
            print(f"Structure {f['structure_id']}: {f['error']}")
    
    if structures:
        try:
            with open(output_path, 'w') as f:
                for atoms in structures:
                    write(f, atoms, format='extxyz')
            print(f"Wrote {len(structures)} structures to {output_path}")
        except Exception as e:
            print(f"Error writing structures to {output_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    else:
        print("No structures to write!")
    
    return saved_ids

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

# TODO: modify the template and allow for more flexibility to change the parameters
def _create_training_script(
    job_dir: Path,
    job_name: str,
    gpu_config: Dict,
    e0s: str = "default",
    seed: int = 42
):
    """Create MACE training script from template."""
    template_path = Path(__file__).parent / "templates" / "mace_train_template.sh"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Get base job name without model suffix for data files
    base_name = job_name.split('_model_')[0]
    
    # Replace template variables
    replacements = {
        'gen_6_model_0_L0_isolated_energy': job_name,  # Use full name for job
        '--ntasks-per-node=4': f"--ntasks-per-node={gpu_config['count']}",
        '--gres=gpu:4': f"--gres=gpu:{gpu_config['count']}",
        '--constraint=rtx6000': f"--constraint={gpu_config['type']}",
        '--name=\'gen_6_model_0_L0_isolated-2026-01-16\'': f"--name=\'{job_name}\'",
        '--train_file="data/gen_6_2025-01-16_train.xyz"': f'--train_file="data/{base_name}_train.xyz"',  # Use base name
        '--valid_file="data/gen_6_2025-01-16_val.xyz"': f'--valid_file="data/{base_name}_val.xyz"',      # Use base name
        '--test_file="data/gen_6_2025-01-16_test.xyz"': f'--test_file="data/{base_name}_test.xyz"',      # Use base name
        '--wandb_name=\'gen_6_model_0_L0_isolated-2025-01-16\'': f"--wandb_name=\'{job_name}\'",
        '--seed=0': f'--seed={seed}'
    }
    
    script_content = template
    for old, new in replacements.items():
        script_content = script_content.replace(old, new)
    
    # Handle E0s configuration
    if e0s == "average":
        script_content = script_content.replace(
            "--E0s='{22: -2.15203187, 23 : -3.55411419, 24 : -5.42767241, 40 : -2.3361286, 74 : -4.55186158}'",
            "--E0s='average'"
        )
    
    # Write training script
    script_path = job_dir / f"{job_name}_train.sh"
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
        gpu_config=gpu_config,
        e0s=e0s
    )
    
    return structure_splits['train']