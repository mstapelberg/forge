# forge/workflows/db_to_allegro.py
import os
import json
import random
import math
from pathlib import Path
from typing import TypedDict, List, Dict, Optional

from ase.data import atomic_numbers
from forge.core.database import DatabaseManager

# Reuse existing functions for splitting and saving
from forge.workflows.db_to_mace import (
    _get_vasp_structures,
    _prepare_structure_splits,
)

class GPUConfig(TypedDict):
    count: int
    type: str


def _extract_chemical_symbols(
    db_manager: DatabaseManager,
    structure_ids: List[int]
) -> List[str]:
    """
    Batch-fetch all atoms, collect unique chemical symbols, and sort by atomic number.
    """
    atoms_list = db_manager.get_batch_atoms_with_calculation(
        structure_ids, calculator='vasp'
    )
    syms = {sym for atoms in atoms_list for sym in atoms.get_chemical_symbols()}
    return sorted(syms, key=lambda s: atomic_numbers[s])


def prepare_allegro_job(
    db_manager: DatabaseManager,
    job_name: str,
    job_dir: Path,
    gpu_config: GPUConfig = {"count": 4, "type": "rtx6000"},
    seed: int = 0,
    num_structures: Optional[int] = None,
    structure_ids: Optional[List[int]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
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
) -> Dict[str, List[int]]:
    """
    Prepare structure splits and generate Allegro YAML + SLURM scripts.

    Returns a mapping from split name to saved structure IDs.
    """
    # 1) Validate input
    if structure_ids is not None and num_structures is not None:
        raise ValueError("Cannot specify both structure_ids and num_structures")
    if structure_ids is None and num_structures is None:
        raise ValueError("Must specify either structure_ids or num_structures")

    # 2) Fetch or randomly sample structure IDs
    all_ids = _get_vasp_structures(db_manager)
    if structure_ids:
        final_ids = structure_ids
    else:
        if len(all_ids) < num_structures:
            raise ValueError(
                f"Not enough structures ({len(all_ids)}) for requested {num_structures}"
            )
        random.seed(seed)
        final_ids = random.sample(all_ids, num_structures)  # type: ignore

    # 3) Determine chemical symbols from the dataset
    chemical_symbols = _extract_chemical_symbols(db_manager, final_ids)
    chem_yaml = '[' + ', '.join(f"'{s}'" for s in chemical_symbols) + ']'

    # 4) Split structures and write .xyz via db_to_mace helper
    data_dir = job_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    saved_ids = _prepare_structure_splits(
        db_manager=db_manager,
        structure_ids=final_ids,
        job_name=job_name,
        job_dir=job_dir,
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # 5) Defaults for schedule and loss
    if schedule is None:
        schedule = {
            'start_epoch': int(0.8 * max_epochs),
            'per_atom_energy_mse': 10.0,
            'forces_mse': 1.0,
            'stress_mse': 0.25,
        }
    if loss_coeffs is None:
        loss_coeffs = {'total_energy': 1.0, 'forces': 50.0, 'stress': 25.0}
    if save_dir is None:
        save_dir = f"results/{job_name}"
    if devices is None:
        devices = gpu_config['count']

    # 6) Prepare replacements dict
    common = {
        'JOB_NAME': job_name,
        'R_MAX': str(r_max),
        'CHEMICAL_SYMBOLS': chem_yaml,
        'SEED': str(seed),
        'GPU_COUNT': str(devices),
        'NUM_NODES': str(num_nodes),
        'MAX_EPOCHS': str(max_epochs),
        'START_EPOCH': str(schedule['start_epoch']),
        'SCHEDULE_ENERGY': str(schedule['per_atom_energy_mse']),
        'SCHEDULE_FORCES': str(schedule['forces_mse']),
        'SCHEDULE_STRESS': str(schedule['stress_mse']),
        'PROJECT': project,
        'SAVE_DIR': save_dir,
        'LOSS_ENERGY': str(loss_coeffs['total_energy']),
        'LOSS_FORCES': str(loss_coeffs['forces']),
        'LOSS_STRESS': str(loss_coeffs['stress']),
        'LR': str(lr),
        'L_MAX': str(l_max),
        'NUM_LAYERS': str(num_layers),
        'NUM_SCALAR': str(num_scalar_features),
        'NUM_TENSOR': str(num_tensor_features),
        'MLP_DEPTH': str(mlp_depth),
        'MLP_WIDTH': str(mlp_width),
        'GPU_TYPE': gpu_config['type'],
    }

        # 6) Prepare replacements dict
    common = {
        'JOB_NAME': job_name,
        'DATA_PREFIX': job_name,
        'R_MAX': str(r_max),
        'CHEMICAL_SYMBOLS': chem_yaml,
        'SEED': str(seed),
        'GPU_COUNT': str(devices),
        'NUM_NODES': str(num_nodes),
        'MAX_EPOCHS': str(max_epochs),
        'START_EPOCH': str(schedule['start_epoch']),
        'SCHEDULE_ENERGY': str(schedule['per_atom_energy_mse']),
        'SCHEDULE_FORCES': str(schedule['forces_mse']),
        'SCHEDULE_STRESS': str(schedule['stress_mse']),
        'PROJECT': project,
        'SAVE_DIR': save_dir,
        'LOSS_ENERGY': str(loss_coeffs['total_energy']),
        'LOSS_FORCES': str(loss_coeffs['forces']),
        'LOSS_STRESS': str(loss_coeffs['stress']),
        'LR': str(lr),
        'L_MAX': str(l_max),
        'NUM_LAYERS': str(num_layers),
        'NUM_SCALAR': str(num_scalar_features),
        'NUM_TENSOR': str(num_tensor_features),
        'MLP_DEPTH': str(mlp_depth),
        'MLP_WIDTH': str(mlp_width),
        'GPU_TYPE': gpu_config['type'],
        # Data file paths (use base job_name for splits)
        'TRAIN_FILE': f"data/{job_name}_train.xyz",
        'VAL_FILE':   f"data/{job_name}_val.xyz",
        'TEST_FILE':  f"data/{job_name}_test.xyz",
    }

    # Template file paths
    yaml_tmpl = Path(__file__).parent / 'templates' / 'allegro_template.yaml'
    slurm_tmpl = Path(__file__).parent / 'templates' / 'allegro_slurm_template.sh'

    def make_run(rname: str, rseed: int):
        # update name & seed, but preserve data file placeholders
        common['JOB_NAME'] = rname
        common['SEED'] = str(rseed)

        # Write YAML from template
        txt = yaml_tmpl.read_text()
        for k, v in common.items():
            txt = txt.replace(f'${{{k}}}', v)
        (job_dir / f"{rname}.yaml").write_text(txt)

        # Write SLURM script
        sl = slurm_tmpl.read_text()
        repl = {
            'JOB_NAME': rname,
            'GPU_COUNT': common['GPU_COUNT'],
            'GPU_TYPE': common['GPU_TYPE'],
            'NUM_NODES': common['NUM_NODES'],
            'YAML_FILE': f"{rname}.yaml"
        }
        for k, v in repl.items():
            sl = sl.replace(f'${{{k}}}', v)
        script_path = job_dir / f"{rname}.sh"
        script_path.write_text(sl)
        script_path.chmod(0o755)

    # 7) Emit jobs (single or ensemble)
    if num_ensemble and num_ensemble > 1:
        for idx in range(num_ensemble):
            make_run(f"{job_name}_model_{idx}", idx)
    else:
        make_run(job_name, seed)

    return saved_ids
