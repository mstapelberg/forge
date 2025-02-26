#!/usr/bin/env python
"""Test module for adversarial attack workflow."""

import os
import shutil
from pathlib import Path
import pytest
import ase.io
import numpy as np
import json
import random

from forge.core.database import DatabaseManager
from forge.core.adversarial_attack import AdversarialCalculator, AdversarialOptimizer
from forge.workflows.adversarial_attack import (
    create_aa_jobs,
    run_aa_jobs,
    create_aa_vasp_jobs
)


@pytest.fixture
def test_output_dir():
    """Create and clean the test output directory."""
    output_dir = Path(__file__).parent / "test_output" / "aa_workflow"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


@pytest.fixture
def test_structures():
    """Get 10 random structures from the database containing V, Cr, Ti, W, or Zr."""
    # First use real database to get structures
    db_manager = DatabaseManager()
    
    # Query structures containing our target elements
    target_elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    all_structure_ids = db_manager.find_structures(
        elements=target_elements,
        debug=True
    )
    print(f"\nFound {len(all_structure_ids)} structures containing {', '.join(target_elements)}")
    
    # Randomly select 10 structures
    n_structures = min(10, len(all_structure_ids))
    selected_ids = random.sample(all_structure_ids, n_structures)
    print(f"Selected {n_structures} random structures for testing")
    
    # Get the structures
    structures = []
    for struct_id in selected_ids:
        atoms = db_manager.get_structure(struct_id)
        # Get metadata for structure info
        metadata = db_manager.get_structure_metadata(struct_id)
        atoms.info.update(metadata)  # Include original metadata
        atoms.info['structure_name'] = f"test_struct_{struct_id}"
        atoms.info['structure_id'] = struct_id
        structures.append(atoms)
        print(f"  Added structure {struct_id}: {atoms.get_chemical_formula()}")
    
    return structures


@pytest.fixture
def test_models():
    """Get path to test MACE models."""
    test_dir = Path(__file__).parent
    model_dir = test_dir / "resources" / "potentials" / "mace"
    if not model_dir.exists():
        pytest.skip("Test MACE models not found")
    return model_dir


def test_adversarial_calculator(test_structures, test_models):
    """Test AdversarialCalculator functionality."""
    model_paths = list(test_models.glob("*.model"))
    if not model_paths:
        pytest.skip("No MACE models found in test resources")
    
    print(f"\nFound {len(model_paths)} MACE models:")
    for path in model_paths:
        print(f"  {path.name}")
    
    calculator = AdversarialCalculator(
        model_paths=model_paths,
        device='cpu'
    )
    
    # Test force calculation on first structure
    atoms = test_structures[0]
    print(f"\nTesting force calculation on {atoms.get_chemical_formula()}")
    print(f"Structure info: {atoms.info}")  # Print structure info for debugging
    forces = calculator.calculate_forces(atoms)
    print(f"Force array shape: {forces.shape}")
    assert forces.shape[0] == len(model_paths)  # Number of models
    assert forces.shape[1] == len(atoms)  # Number of atoms
    assert forces.shape[2] == 3  # Force components
    assert not np.allclose(forces, 0)  # Forces should not be all zero
    
    # Test variance calculation
    variances = calculator.calculate_normalized_force_variance(forces)
    print(f"Variance array shape: {variances.shape}")
    print(f"Mean variance: {np.mean(variances):.6f}")
    assert len(variances) == len(atoms)
    assert np.all(variances >= 0)  # Variances should be non-negative
    assert not np.allclose(variances, 0)  # Variances should not be all zero


def test_adversarial_optimization(test_structures, test_models, test_output_dir):
    """Test adversarial optimization on a single structure."""
    model_paths = list(test_models.glob("*.model"))
    if not model_paths:
        pytest.skip("No MACE models found in test resources")
    
    # Initialize dry run database for testing
    db_manager = DatabaseManager(dry_run=True)
    
    calculator = AdversarialCalculator(
        model_paths=model_paths,
        device='cpu'
    )
    optimizer = AdversarialOptimizer(calculator)
    
    # Run optimization with small number of steps
    atoms = test_structures[0]
    print(f"\nRunning optimization on {atoms.get_chemical_formula()}")
    print(f"Structure info: {atoms.info}")  # Print structure info for debugging
    
    # Calculate initial variance
    forces = calculator.calculate_forces(atoms)
    initial_variance = float(np.mean(calculator.calculate_normalized_force_variance(forces)))
    print(f"Initial variance: {initial_variance:.6f}")
    
    best_atoms, best_variance, accepted = optimizer.optimize(
        atoms=atoms,
        temperature=1200.0,
        max_iterations=5,
        patience=3,
        mode='all',
        output_dir=str(test_output_dir)
    )
    
    print(f"Best variance: {best_variance:.6f}")
    print(f"Accepted moves: {accepted}")
    
    # Check outputs
    assert best_variance >= initial_variance  # Should be at least as good as initial
    assert accepted >= 0
    assert (test_output_dir / f"{atoms.info['structure_name']}_adversarial.xyz").exists()
    assert (test_output_dir / "optimization_summary.json").exists()


def test_complete_workflow(test_structures, test_models, test_output_dir):
    """Test the complete adversarial attack workflow."""
    # Initialize dry run database for testing
    db_manager = DatabaseManager(dry_run=True)
    
    # Get structure IDs
    structure_ids = [atoms.info['structure_id'] for atoms in test_structures]
    print(f"\nTesting workflow with {len(structure_ids)} structures")
    
    # Step 1: Create initial workflow
    workflow_dir = test_output_dir
    create_aa_jobs(
        output_dir=str(workflow_dir),
        model_dir=str(test_models),
        elements=None,  # Not needed since we're using structure_ids
        n_batches=2,  # 5 structures per batch
        structure_ids=structure_ids,
        debug=True
    )
    
    # Instead of running SLURM jobs, calculate variances directly
    variance_dir = workflow_dir / "variance_results"
    variance_dir.mkdir(parents=True, exist_ok=True)
    
    calculator = AdversarialCalculator(
        model_paths=list(test_models.glob("*.model")),
        device='cpu'
    )
    
    # Calculate variances for each structure
    for batch_id in range(2):
        batch_dir = workflow_dir / "variance_calculations" / f"batch_{batch_id}"
        xyz_file = batch_dir / f"batch_{batch_id}.xyz"
        if xyz_file.exists():
            print(f"\nProcessing batch {batch_id}")
            atoms_list = ase.io.read(xyz_file, ':')
            variances = {}
            for atoms in atoms_list:
                print(f"  Calculating variance for {atoms.info['structure_name']}")
                forces = calculator.calculate_forces(atoms)
                atom_variances = calculator.calculate_normalized_force_variance(forces)
                structure_variance = float(np.mean(atom_variances))
                variances[atoms.info['structure_name']] = structure_variance
                print(f"    Variance: {structure_variance:.6f}")
            
            with open(variance_dir / f"batch_{batch_id}_variances.json", 'w') as f:
                json.dump(variances, f)
    
    # Step 2: Run AA optimization on top structure only
    run_aa_jobs(
        input_directory=str(variance_dir),
        model_dir=str(test_models),
        n_structures=1,  # Only optimize top structure
        n_batches=1,  # Single batch since only one structure
        temperature=1200.0,
        max_steps=5,
        patience=3,
        debug=True
    )
    
    # Instead of SLURM jobs, run optimization directly
    aa_dir = workflow_dir / "aa_optimization"
    batch_dir = aa_dir / "batch_0"
    xyz_file = batch_dir / "batch_0.xyz"
    if xyz_file.exists():
        print("\nRunning optimization for top structure")
        atoms_list = ase.io.read(xyz_file, ':')
        results_dir = batch_dir / "aa_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        atoms = atoms_list[0]  # Only one structure
        print(f"  Optimizing {atoms.info['structure_name']}")
        optimizer = AdversarialOptimizer(calculator)
        best_atoms, best_variance, accepted = optimizer.optimize(
            atoms=atoms,
            temperature=1200.0,
            max_iterations=5,
            patience=3,
            mode='all',
            output_dir=str(results_dir)
        )
        print(f"    Best variance: {best_variance:.6f}")
        print(f"    Accepted moves: {accepted}")
    
    # Step 3: Create VASP jobs
    create_aa_vasp_jobs(
        input_directory=str(aa_dir),
        output_directory=str(workflow_dir / "vasp_jobs"),
        hpc_profile="Perlmutter-GPU",
        debug=True
    )
    
    # Verify outputs
    assert workflow_dir.exists()
    assert (workflow_dir / "README.md").exists()
    assert variance_dir.exists()
    assert aa_dir.exists()
    assert (workflow_dir / "vasp_jobs").exists()
    
    # Check that we can read the example commands
    with open(workflow_dir / "README.md", 'r') as f:
        content = f.read()
        assert 'forge create-aa-jobs' in content
        assert 'forge run-aa-jobs' in content
        assert 'forge create-aa-vasp-jobs' in content 