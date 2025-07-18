from forge.core.database import DatabaseManager
from forge.workflows.adversarial_attack import run_adversarial_attacks  # Now supports backend parameter
import random
import numpy as np 
from ase.io import read, write
import json
from glob import glob
import pandas as pd
from pathlib import Path

db_manager = DatabaseManager()

rare_ids = json.load(open('./analysis_output_full_gen8/rare_structure_ids.json'))

filtered_ids = json.load(open('./analysis_output_full_gen8/selected_for_augmentation.json'))

calcs = db_manager.get_batch_atoms_with_calculation(filtered_ids)

# Model paths - these are Allegro models (.pt2 files)
model_paths = glob('../data/potentials/compiled_gen-8-exploit/*.nequip.pt2')
package_paths = glob('../data/potentials/packaged_gen-8-exploit/*.zip')
print(f"Found {len(model_paths)} compiled Allegro model files")
print(f"Found {len(package_paths)} packaged Allegro model files")

# SOLUTION: Use packaged models (.zip) instead of compiled models (.pt2) for gradient flow!
# Packaged models preserve the full PyTorch computational graph needed for autograd
use_packaged_models = True

# Limit the number of models to avoid memory issues and speed up testing
max_models = 5  # Use only first 3 models for testing
print(f"[INFO] Limiting to {max_models} models for testing (you can increase this)")

if use_packaged_models and package_paths:
    print(f"[INFO] Using packaged models (.zip) for gradient-based optimization")
    models_to_use = package_paths[:max_models]  # Limit to first N models
    print(f"Selected {len(models_to_use)} models:")
    for i, model_path in enumerate(models_to_use):
        print(f"  {i+1}. {Path(model_path).name}")
else:
    print(f"[INFO] Using compiled models (.pt2) - gradients may not work for optimization")
    models_to_use = model_paths[:max_models]  # Limit to first N models
    print(f"Selected {len(models_to_use)} models:")
    for i, model_path in enumerate(models_to_use):
        print(f"  {i+1}. {Path(model_path).name}")

random.seed(42)

# Now using the updated workflow with backend support
trajectories = run_adversarial_attacks(
    db_manager = db_manager,
    model_paths = models_to_use,  # Use limited set of packaged models for gradient flow!
    structure_ids = filtered_ids,  # Test with just 10 structures
    generation = 9,
    n_iterations=200,
    learning_rate=0.01,
    temperature=1000,
    include_probability=False,
    min_distance=1.2,
    use_energy_per_atom=True,
    device='cuda',
    debug=False,  # Temporarily enable to see output keys
    top_n=250,  # Reduce for testing
    save_output=True,
    output_dir='../data/adversarial_attacks/gen_9_umap_filtered',
    patience=25,
    shake=False,
    ranking_metric='force_rmse',
    reference_calculator='vasp',
    cache_rmse=True,
    plot_rmse_histogram=True,
    rmse_cutoff=0.1,
    backend='allegro'  # ← NEW: Explicitly specify Allegro backend
    # Alternative: backend='auto' would auto-detect from .zip extensions
)

# --- Handle None return value when saving --- 
if trajectories is not None:
    print(f"Number of trajectories returned: {len(trajectories)}")
else:
    print("Trajectories were saved to files.")

print(f"All done")

# Demonstration of the new calculator interface (optional)
print("\n=== Demonstrating New Calculator Interface ===")

try:
    # You can now also use the calculator interface directly if needed
    from forge.calculators import create_ensemble_calculator
    
    # Create calculator with auto-detection
    calc = create_ensemble_calculator(
        model_paths=model_paths[:2],  # Use first 2 models for demo
        backend='auto',  # Will auto-detect as 'allegro' from .pt2 files
        device='cuda'
    )
    
    print(f"Created {type(calc).__name__} with {len(calc.models)} models")
    print(f"Device: {calc.device}")
    print(f"r_max: {calc.r_max}")
    
    # Example usage with a sample structure
    if calcs:
        sample_atoms = calcs[0]
        print(f"Testing with sample structure: {sample_atoms.get_chemical_formula()}")
        
        # Get forces from all models
        all_forces = calc.forces_all(sample_atoms)
        print(f"All forces shape: {all_forces.shape}")  # Should be (n_models, n_atoms, 3)
        
        # Get mean forces
        mean_forces = calc.get_mean_forces(sample_atoms)  
        print(f"Mean forces shape: {mean_forces.shape}")  # Should be (n_atoms, 3)
        
        # Calculate force variance
        force_variance = calc.calculate_normalized_force_variance(all_forces)
        print(f"Force variance shape: {force_variance.shape}")  # Should be (n_atoms,)
        print(f"Mean force variance: {np.mean(force_variance):.6f}")
        
except Exception as e:
    print(f"Calculator demo failed: {e}")
    print("This is expected if model files don't exist - the workflow will work when models are available") 