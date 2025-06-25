import yaml
import pytest
from pathlib import Path
import shutil
import json

from forge.core.database import DatabaseManager
from forge.workflows.db_to_allegro import prepare_allegro_job as prepare_allegro_job_new
from forge.workflows.db_to_allegro_old import prepare_allegro_job as prepare_allegro_job_old

def fake_prepare_splits(db_manager, structure_ids, job_name, job_dir, data_dir, train_ratio, val_ratio, test_ratio, seed):
    """
    A fake _prepare_structure_splits that creates empty files and returns dummy splits
    to avoid needing a real database in tests.
    """
    (data_dir / f"{job_name}_train.xyz").touch()
    (data_dir / f"{job_name}_val.xyz").touch()
    (data_dir / f"{job_name}_test.xyz").touch()
    
    num_train = int(len(structure_ids) * train_ratio)
    num_val = int(len(structure_ids) * val_ratio)
    
    train_ids = structure_ids[:num_train]
    val_ids = structure_ids[num_train:num_train+num_val]
    test_ids = structure_ids[num_train+num_val:]
    
    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    
    with open(job_dir / f"{job_name}_structure_splits.json", "w") as f:
        json.dump(splits, f)
        
    return splits

def fake_extract_symbols(db_manager, structure_ids):
    """A fake _extract_chemical_symbols that returns a fixed list."""
    return ["V", "Cr", "Ti"]

@pytest.fixture
def db_with_structures():
    """Provides a database manager with a few test structures."""
    with DatabaseManager(dry_run=True) as db:
        # In dry_run mode, this doesn't actually hit a DB but allows us to get IDs
        yield db

@pytest.fixture
def job_dirs(tmp_path):
    """Creates temporary directories for the old and new job outputs."""
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    yield old_dir, new_dir
    shutil.rmtree(old_dir)
    shutil.rmtree(new_dir)

def run_and_compare_jobs(db_manager, old_dir, new_dir, common_params):
    """Helper function to run both old and new job preps and compare results."""
    # Run the old version
    splits_old = prepare_allegro_job_old(
        db_manager=db_manager,
        job_dir=old_dir,
        **common_params
    )

    # Run the new (to be refactored) version
    splits_new = prepare_allegro_job_new(
        db_manager=db_manager,
        job_dir=new_dir,
        **common_params
    )

    # Compare the returned structure splits
    assert splits_old == splits_new, "Returned structure splits do not match"

    # Compare the generated config.yaml files
    config_old_path = old_dir / "config.yaml"
    config_new_path = new_dir / "config.yaml"

    assert config_old_path.exists(), "Old config.yaml was not created"
    assert config_new_path.exists(), "New config.yaml was not created"

    with open(config_old_path, 'r') as f:
        config_old = yaml.safe_load(f)
    
    with open(config_new_path, 'r') as f:
        config_new = yaml.safe_load(f)

    assert config_old == config_new, "Generated config.yaml files are not identical"

def test_standalone_mode_equivalence(db_with_structures, job_dirs, monkeypatch):
    """
    Tests that the refactored `prepare_allegro_job` behaves identically
    to the old version in standalone mode.
    """
    old_dir, new_dir = job_dirs

    # Monkeypatch the functions that fail in dry_run mode
    monkeypatch.setattr("forge.workflows.db_to_allegro_old._prepare_structure_splits", fake_prepare_splits)
    monkeypatch.setattr("forge.workflows.db_to_allegro._prepare_structure_splits", fake_prepare_splits)
    monkeypatch.setattr("forge.workflows.db_to_allegro_old._extract_chemical_symbols", fake_extract_symbols)
    monkeypatch.setattr("forge.workflows.db_to_allegro._extract_chemical_symbols", fake_extract_symbols)
    
    # Use a small, fixed list of structure IDs for reproducibility
    structure_ids = list(range(1, 21))
    
    params = {
        "job_name": "standalone_test",
        "structure_ids": structure_ids,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "seed": 42,
        "r_max": 4.5,
        "l_max": 1,
        "num_layers": 3,
        "loss_function": "huber",
        "loss_params": {"delta": 0.5},
        # Add required keys that were missing for the config builder
        "max_epochs": 1,
        "batch_size": 2,
        "lr": 0.01,
        "checkpoint_monitor_key": "val0_epoch/forces_mae",
        "num_scalar_features": 8,
        "num_tensor_features": 8,
        "mlp_depth": 1,
        "mlp_width": 16,
    }
    
    run_and_compare_jobs(db_with_structures, old_dir, new_dir, params)

def test_hpo_mode_equivalence(db_with_structures, job_dirs, monkeypatch):
    """
    Tests that the refactored `prepare_allegro_job` behaves identically
    to the old version in HPO mode.
    """
    old_dir, new_dir = job_dirs

    # This test doesn't write files, but it does extract symbols, which needs patching.
    monkeypatch.setattr("forge.workflows.db_to_allegro_old._extract_chemical_symbols", fake_extract_symbols)
    monkeypatch.setattr("forge.workflows.db_to_allegro._extract_chemical_symbols", fake_extract_symbols)
    
    # Create dummy data files for HPO mode
    data_dir = old_dir.parent / "hpo_data"
    data_dir.mkdir()
    train_path = data_dir / "train.xyz"
    val_path = data_dir / "val.xyz"
    test_path = data_dir / "test.xyz"
    train_path.touch()
    val_path.touch()
    test_path.touch()
    
    params = {
        "job_name": "hpo_test",
        "data_train_path": str(train_path),
        "data_val_path": str(val_path),
        "data_test_path": str(test_path),
        "chemical_symbols_list": ["Ti", "V", "Cr"],
        "seed": 123,
        "lr": 0.005,
        "max_epochs": 10,
        "sampler": "rare_weighted",
        "sampler_params": {"replica": 3, "alpha": 0.5, "rare_idx": [1, 5, 10]},
        # Add required keys
        "batch_size": 2,
        "r_max": 5.0,
        "l_max": 2,
        "num_layers": 2,
        "loss_function": "mse",
        "checkpoint_monitor_key": "val0_epoch/forces_mae",
        "num_scalar_features": 8,
        "num_tensor_features": 8,
        "mlp_depth": 1,
        "mlp_width": 16,
    }
    
    run_and_compare_jobs(db_with_structures, old_dir, new_dir, params) 