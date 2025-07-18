#!/usr/bin/env python
"""Demonstration of the new calculator backend system."""

import numpy as np
from ase.build import bulk
from forge.calculators import create_ensemble_calculator, get_supported_backends


def demo_calculator_backends():
    """Demonstrate the new calculator backend system."""
    
    print("=== Calculator Backend Demonstration ===\n")
    
    # Show supported backends
    backends = get_supported_backends()
    print(f"Supported backends: {backends}\n")
    
    # Create sample atoms
    atoms = bulk('Al', 'fcc', a=4.0, cubic=True)  # 4 atoms
    print(f"Sample structure: {atoms.get_chemical_formula()} with {len(atoms)} atoms\n")
    
    # Example model paths (these would be real paths in practice)
    mace_models = ['model1.model', 'model2.model', 'model3.model']
    allegro_models = ['model1.nequip.pt2', 'model2.nequip.pt2', 'model3.nequip.pt2']
    
    print("=== MACE Backend Example ===")
    print("# Auto-detection from .model files:")
    print(f"create_ensemble_calculator({mace_models}, backend='auto')")
    print("# Explicit MACE backend:")
    print(f"create_ensemble_calculator({mace_models}, backend='mace')")
    print()
    
    print("=== Allegro Backend Example ===")
    print("# Auto-detection from .pt2 files:")
    print(f"create_ensemble_calculator({allegro_models}, backend='auto')")
    print("# Explicit Allegro backend:")
    print(f"create_ensemble_calculator({allegro_models}, backend='allegro')")
    print()
    
    print("=== Usage in Adversarial Attack Workflow ===")
    print("""
# With MACE models (auto-detected):
trajectories = run_adversarial_attacks(
    db_manager=db_manager,
    model_paths=['model1.model', 'model2.model'],
    backend='auto',  # Will auto-detect MACE
    # ... other parameters
)

# With Allegro models (explicit):
trajectories = run_adversarial_attacks(
    db_manager=db_manager,
    model_paths=['model1.pt2', 'model2.pt2'],
    backend='allegro',  # Explicitly specify Allegro
    # ... other parameters
)

# For mixed scenarios or uncertainty:
trajectories = run_adversarial_attacks(
    db_manager=db_manager,
    model_paths=model_paths,
    backend='mace',  # Force specific backend
    # ... other parameters
)
""")

    print("=== Interface Methods Available ===")
    print("""
# All backends implement BaseEnsembleCalculator interface:
calculator = create_ensemble_calculator(model_paths, backend='auto')

# Get forces from all models: shape (n_models, n_atoms, 3)
all_forces = calculator.forces_all(atoms)

# Get energies from all models: shape (n_models,)
all_energies = calculator.energies_all(atoms)

# Get mean forces/energies (convenience methods)
mean_forces = calculator.get_mean_forces(atoms)
mean_energy = calculator.get_mean_energy(atoms)

# Calculate force variance
force_variance = calculator.calculate_normalized_force_variance(all_forces)

# Access properties
device = calculator.device
models = calculator.models
z_table = calculator.z_table
r_max = calculator.r_max
""")


def demo_workflow_integration():
    """Show how the workflow integrates with the new backend system."""
    
    print("\n=== Workflow Integration ===\n")
    
    print("The adversarial attack workflow now supports a 'backend' parameter:")
    print("""
from forge.workflows.adversarial_attack import run_adversarial_attacks

# Example with MACE models:
trajectories = run_adversarial_attacks(
    db_manager=db_manager,
    structure_ids=[1, 2, 3, 4, 5],
    model_paths=['../models/mace_model_1.model', '../models/mace_model_2.model'],
    top_n=10,
    generation=9,
    n_iterations=100,
    learning_rate=0.01,
    temperature=1000,
    include_probability=False,
    min_distance=1.2,
    backend='auto',  # ← NEW PARAMETER
    # ... other existing parameters
)

# Example with Allegro models:
trajectories = run_adversarial_attacks(
    db_manager=db_manager,
    structure_ids=[1, 2, 3, 4, 5],
    model_paths=['../models/allegro_model_1.pt2', '../models/allegro_model_2.pt2'],
    top_n=10,
    generation=9,
    n_iterations=100,
    learning_rate=0.01,
    temperature=1000,
    include_probability=False,
    min_distance=1.2,
    backend='allegro',  # ← NEW PARAMETER
    # ... other existing parameters
)
""")


if __name__ == "__main__":
    demo_calculator_backends()
    demo_workflow_integration() 