# forge/tests/test_allegro_customization.py
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock

from forge.workflows.db_to_allegro import prepare_allegro_job

# Mock the DatabaseManager, since we are not testing database interactions here
@pytest.fixture
def mock_db_manager():
    """Fixture for a mocked DatabaseManager."""
    db_manager = MagicMock()
    # Mock the function that returns all structure IDs
    # This is the key fix: ensure that the structure selection step finds IDs.
    db_manager.get_all_structure_ids.return_value = list(range(100))
    # Mock the chemical symbol extraction to return a fixed list
    db_manager.get_batch_atoms_with_calculation.return_value = []
    return db_manager

@pytest.fixture
def temp_job_dir(tmp_path):
    """Fixture to create a temporary directory for job outputs."""
    return tmp_path

def run_prepare_and_load_config(db_manager, job_dir, **kwargs):
    """Helper function to run prepare_allegro_job and load the resulting YAML."""
    # Provide minimal required args for standalone mode
    base_args = {
        "db_manager": db_manager,
        "job_name": "test_job",
        "job_dir": job_dir,
        "num_structures": 10,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "chemical_symbols_list": ["H", "O"], # Provide symbols directly to avoid db calls
    }
    
    # Mock the structure splitting function to avoid file system operations
    # and return dummy splits.
    with pytest.MonkeyPatch.context() as m:
        # Mock the function that gets IDs from the DB within the workflow
        # to ensure it uses our mocked manager's return value.
        m.setattr(
            "forge.workflows.db_to_allegro._get_vasp_structures",
            lambda dbm: dbm.get_all_structure_ids()
        )
        m.setattr(
            "forge.workflows.db_to_allegro._prepare_structure_splits", 
            lambda *args, **kwargs: {"train": [1], "val": [2], "test": [3]}
        )
        # Also mock file existence checks
        m.setattr("pathlib.Path.exists", lambda self: True)
        
        # Merge user-provided kwargs with base args
        final_args = {**base_args, **kwargs}
        
        prepare_allegro_job(**final_args)

    config_path = job_dir / "config.yaml"
    assert config_path.exists()
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def test_default_mse_loss(mock_db_manager, temp_job_dir):
    """Tests that the default configuration uses MSE loss."""
    config = run_prepare_and_load_config(mock_db_manager, temp_job_dir)
    loss_metrics = config["training_module"]["loss"]["metrics"]
    
    assert config["training_module"]["loss"]["_target_"] == "nequip.train.MetricsManager"
    
    force_loss = next(m for m in loss_metrics if m["field"] == "forces")
    assert force_loss["metric"]["_target_"] == "nequip.train.MeanSquaredError"

def test_focal_loss_selection(mock_db_manager, temp_job_dir):
    """Tests selecting the focal loss function with parameters."""
    loss_params = {"beta": 1.5, "gamma": 2.5}
    config = run_prepare_and_load_config(
        mock_db_manager, 
        temp_job_dir,
        loss_function="focal",
        loss_params=loss_params,
    )
    
    loss_metrics = config["training_module"]["loss"]["metrics"]
    force_loss = next(m for m in loss_metrics if m["field"] == "forces")
    
    assert force_loss["metric"]["_target_"] == "forge.workflows.allegro_utils.custom_losses.FocalMSELoss"
    assert force_loss["metric"]["beta"] == 1.5
    assert force_loss["metric"]["gamma"] == 2.5

def test_rare_weighted_sampler(mock_db_manager, temp_job_dir):
    """Tests the configuration of the RareWeightedSampler."""
    sampler_params = {"replica": 5, "alpha": 0.5, "rare_idx": [1, 2, 3]}
    config = run_prepare_and_load_config(
        mock_db_manager,
        temp_job_dir,
        sampler="rare_weighted",
        sampler_params=sampler_params,
    )
    
    sampler_config = config["data"]["train_dataloader"]["sampler"]
    assert sampler_config["_target_"] == "forge.workflows.allegro_utils.samplers.RareWeightedSampler"
    assert sampler_config["replica"] == 5
    assert sampler_config["alpha"] == 0.5
    assert sampler_config["rare_idx"] == [1, 2, 3]

def test_custom_callbacks(mock_db_manager, temp_job_dir):
    """Tests the inclusion of custom callbacks."""
    callback_params = {"curriculum": {"epochs": [0, 10, 20]}}
    config = run_prepare_and_load_config(
        mock_db_manager,
        temp_job_dir,
        callbacks=["curriculum", "grad_norm"],
        callback_params=callback_params
    )
    
    callbacks = config["trainer"]["callbacks"]
    
    curriculum_cb = next(cb for cb in callbacks if "CurriculumCallback" in cb["_target_"])
    grad_norm_cb = next(cb for cb in callbacks if "GradNormCallback" in cb["_target_"])
    
    assert curriculum_cb is not None
    assert grad_norm_cb is not None
    assert curriculum_cb["epochs"] == [0, 10, 20]

def test_extra_validation_metric(mock_db_manager, temp_job_dir):
    """Tests adding an extra validation metric like TailMSE."""
    metric_params = {"tail_mse": {"quantile": 0.95}}
    config = run_prepare_and_load_config(
        mock_db_manager,
        temp_job_dir,
        extra_val_metrics=["tail_mse"],
        extra_val_metric_params=metric_params,
    )
    
    val_metrics = config["training_module"]["val_metrics"]["metrics"]
    
    tail_mse_metric = next(m for m in val_metrics if "tail_mse" in m["name"])
    
    assert tail_mse_metric["metric"]["_target_"] == "forge.workflows.allegro_utils.custom_metrics.TailMSE"
    assert tail_mse_metric["metric"]["quantile"] == 0.95
    
    # Ensure standard MAE metrics are still present
    mae_metric = next(m for m in val_metrics if m["name"] == "forces_mae")
    assert mae_metric is not None

def test_invalid_argument_raises_error(mock_db_manager, temp_job_dir):
    """Tests that providing an unsupported component name raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported loss function 'invalid_loss'"):
        run_prepare_and_load_config(mock_db_manager, temp_job_dir, loss_function="invalid_loss")
        
    with pytest.raises(ValueError, match="Unsupported sampler 'invalid_sampler'"):
        run_prepare_and_load_config(mock_db_manager, temp_job_dir, sampler="invalid_sampler")

    with pytest.raises(ValueError, match="Unsupported callback 'invalid_callback'"):
        run_prepare_and_load_config(mock_db_manager, temp_job_dir, callbacks=["invalid_callback"])
        
    with pytest.raises(ValueError, match="Unsupported validation metric 'invalid_metric'"):
        run_prepare_and_load_config(mock_db_manager, temp_job_dir, extra_val_metrics=["invalid_metric"]) 