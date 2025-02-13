import pytest
import os
import json
from pathlib import Path
import filecmp
import tempfile
import shutil
import numpy as np

from forge.workflows.db_to_mace import prepare_mace_job, _replace_properties, _create_training_script

@pytest.fixture(scope="module")
def test_job_dir():
    """Create job directory for all tests."""
    # Get package root directory
    root_dir = Path(__file__).parent.parent
    job_dir = root_dir / "tests" / "test_mace_jobs" / "gen_0_test-2025-02-17"
    
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)
    return job_dir

def test_ensemble_creation(db_manager, test_job_dir):
    """Test creation of ensemble training scripts."""
    job_name = "gen_0_test-2025-02-17"
    gpu_config = {"count": 2, "type": "v100s"}
    
    structure_ids = prepare_mace_job(
        db_manager=db_manager,
        job_name=job_name,
        job_dir=test_job_dir,
        gpu_config=gpu_config,
        num_structures=1000,
        seed=42,
        num_ensemble=3  # Create 3 ensemble models
    )
    
    # Verify splits (only once since all models share the same data)
    assert len(structure_ids['train']) == 800
    assert len(structure_ids['val']) == 100
    assert len(structure_ids['test']) == 100
    
    # Verify training scripts for each model
    for model_idx in range(3):
        model_name = f"{job_name}_model_{model_idx}"
        
        # Verify training script matches template
        root_dir = Path(__file__).parent.parent
        template_path = root_dir / "tests" / "resources" / "templates" / f"{model_name}.sh"
        generated_path = test_job_dir / f"{model_name}_train.sh"
        
        assert template_path.exists(), f"Template not found: {template_path}"
        assert generated_path.exists(), f"Generated script not found: {generated_path}"
        
        # Print contents for debugging
        print(f"\nComparing files for model {model_idx}:")
        with open(template_path) as f:
            template_content = f.read()
        with open(generated_path) as f:
            generated_content = f.read()
            
        if template_content != generated_content:
            print("\nTemplate content:")
            print(template_content)
            print("\nGenerated content:")
            print(generated_content)
            
        assert filecmp.cmp(template_path, generated_path, shallow=False)

def test_property_replacement(test_job_dir):
    """Test property replacement in generated XYZ files."""
    data_dir = test_job_dir / "data"
    
    # Check all xyz files in data directory
    for xyz_file in data_dir.glob("*.xyz"):
        with open(xyz_file) as f:
            content = f.read()
            # Original properties should be replaced
            assert " energy=" not in content
            assert " stress=" not in content
            assert ":forces:" not in content
            # New properties should be present
            assert "REF_energy=" in content
            assert "REF_stress=" in content
            assert ":REF_force:" in content

def test_train_val_test_split(db_manager, test_job_dir):
    """Test consistency of train/val/test splits with seed control."""
    job_name = "gen_0_test-2025-02-17_model_0"
    
    # Run with seed=42
    split_1 = prepare_mace_job(
        db_manager=db_manager,
        job_name=job_name,
        job_dir=test_job_dir,
        num_structures=1000,
        seed=42
    )
    
    # Clear directory and run again with same seed
    shutil.rmtree(test_job_dir)
    test_job_dir.mkdir()
    
    split_2 = prepare_mace_job(
        db_manager=db_manager,
        job_name=job_name,
        job_dir=test_job_dir,
        num_structures=1000,
        seed=42
    )
    
    # Verify splits are identical
    assert split_1['train'] == split_2['train']
    assert split_1['val'] == split_2['val']
    assert split_1['test'] == split_2['test']
    
    # Verify split ratios (assuming default 80/10/10)
    total_structures = 1000
    assert len(split_1['train']) == int(0.8 * total_structures)
    assert len(split_1['val']) == int(0.1 * total_structures)
    assert len(split_1['test']) >= int(0.1 * total_structures) - 1  # Account for rounding


def test_gpu_configuration(db_manager, test_job_dir):
    """Test GPU configuration in generated scripts."""
    job_name = "gen_0_test-2025-02-17_model_0"
    gpu_config = {"count": 2, "type": "v100s"}
    
    prepare_mace_job(
        db_manager=db_manager,
        job_name=job_name,
        job_dir=test_job_dir,
        gpu_config=gpu_config,
        num_structures=1000,
        seed=42
    )
    
    script_path = test_job_dir / f"{job_name}_train.sh"
    
    with open(script_path) as f:
        content = f.read()
        assert "#SBATCH --ntasks-per-node=2" in content
        assert "#SBATCH --gres=gpu:2" in content
        assert "#SBATCH --constraint=v100s" in content 